#!/usr/bin/env python3
"""Produce frozen SemTalk Base predictions for the SHOW validation split.

This entry point uses the neutral ``semtalk_base_inference_core.py`` callable
closure.  It accepts only an explicit ``--split val`` and never exposes a test
input.

The transaction has three small, explicit phases:

* ``prepare`` validates the complete 22-candidate bundle, the exact 1,715
  clip validation cache, and the frozen downstream pipeline once.
* ``shard`` runs one deterministic modulo shard.  ``--num-shards`` is
  configurable (including 16 shards across two eight-GPU workers).
* ``finalize`` is CPU-only and single-writer.  It validates exact-once shard
  closure, independently copies every result into an unpublished generation,
  writes the selector-compatible lineage, and publishes the whole generation
  with one directory rename.

Only the registered 22 epochs from e1 through e400 are accepted.  Withdrawn
e30/epoch-30 and Speaker2 labels, as well as every test-labelled path, fail
before model load.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_long_val_contract as selector  # noqa: E402


PREFLIGHT_FORMAT = (
    "semtalk_show_base_official_adapt_val_inference_preflight_v1"
)
SHARD_FORMAT = "semtalk_show_base_official_adapt_val_inference_shard_v1"
ASSIGNMENT = "canonical_position_modulo_num_shards"
FINAL_DIRECTORY = "final"
SHARDS_DIRECTORY = "shards"
LINEAGE_FILENAME = "val-inference-lineage.json"
SHARD_MANIFEST_FILENAME = "shard_manifest.jsonl"
SHARD_RECEIPT_FILENAME = "shard_receipt.json"
MAX_NUM_SHARDS = 256
SMPLX_POSE_DIM = 165
PINNED_JOINT_CONTEXT_SOURCE = {
    "relative_path": "utils/show_base_joints.py",
    "sha256": (
        "4ae09ddc025a4585b92a50d7f85ace330c398cd676c2905e38a4e9b0f1d56a38"
    ),
    "git_blob_sha1": "9c3ccd0136d989b430ecc9d21090c059ef7fe5a6",
}
PINNED_JOINT_AUTHORITY_SOURCE = {
    "relative_path": "dataloaders/data_tools.py",
    "sha256": (
        "6fd248c2a13ce164c80bab2d76012fa57e4d5fcb507e43e6a5309c1b8fac4db8"
    ),
    "git_blob_sha1": "7f59fa57d94880b92a4455eb8f8e33c89ea8a3c7",
}
PINNED_JOINT_MASK_SHA256 = {
    "face": (
        "803564ee8b4f306b80611eb6574c35d626155a89c38dca22756219cbecdf7021"
    ),
    "upper": (
        "5ee4e5ffc6429f11547451a29a8ddbdb47d9b8f4af665a5e247d101e307d575a"
    ),
    "hands": (
        "ded83502732ea488834d9af1fb9a5df711ea5cf6b7f2b4b150f69adefe2d09e7"
    ),
    "lower": (
        "4aad0acba09e535d5a6b5b2aa333d1227cdd7ba5b8b21df0660f2fcb3a010776"
    ),
}


class ValInferenceContractError(RuntimeError):
    """Raised when validation-only inference cannot be proven."""


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


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _payload_sha(value: Mapping[str, Any]) -> str:
    return selector.canonical_json_sha256(dict(value))


def _with_payload_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_payload_sha256"] = _payload_sha(result)
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return selector.sha256_file(path)


def _artifact(path: Path, *, payload_sha: str | None = None) -> dict[str, str]:
    resolved, _payload, observed, _metadata = _safe_file_snapshot(
        path,
        "artifact",
    )
    result = {
        "path": str(resolved),
        "sha256": observed,
    }
    if payload_sha is not None:
        result["receipt_payload_sha256"] = payload_sha
    return result


def _output_artifact(path: Path) -> dict[str, Any]:
    resolved, payload, observed, _metadata = _safe_file_snapshot(
        path,
        "output artifact",
    )
    return {
        "path": str(resolved),
        "sha256": observed,
        "bytes": len(payload),
    }


def _reject_forbidden(value: object, label: str) -> None:
    text = str(value).replace("\\", "/")
    for component in text.split("/"):
        normalized = component.casefold()
        if re.search(
            r"(^|[^a-z0-9])(?:e|epoch)[-_]?30([^a-z0-9]|$)",
            normalized,
        ) or re.search(
            r"(^|[^a-z0-9])speaker[-_]?2([^a-z0-9]|$)",
            normalized,
        ):
            raise ValInferenceContractError(
                f"{label} contains withdrawn e30 or forbidden Speaker2: "
                f"{value}"
            )


def _reject_path(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValInferenceContractError(f"{label} must be absolute: {path}")
    _reject_forbidden(path, label)
    selector.reject_test_path(path, label)


def _reject_absolute_paths_in_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_absolute_paths_in_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_absolute_paths_in_tree(child, f"{label}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        _reject_path(Path(value), label)


def _regular_file(path: Path, label: str) -> Path:
    _reject_path(path, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValInferenceContractError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    _reject_path(resolved, label)
    return resolved


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
    path: Path,
    label: str,
    *,
    expected_sha: str | None = None,
) -> tuple[Path, bytes, str, os.stat_result]:
    """Read one immutable regular-file snapshot through one descriptor."""
    expected = (
        selector.require_sha256(expected_sha, f"{label} SHA")
        if expected_sha is not None
        else None
    )
    resolved = _regular_file(path, label)
    try:
        path_before = os.lstat(resolved)
    except OSError as error:
        raise ValInferenceContractError(
            f"cannot safely inspect {label}: {resolved}"
        ) from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
        path_before.st_mode
    ):
        raise ValInferenceContractError(
            f"{label} must be a regular non-symlink file: {resolved}"
        )
    parts = resolved.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise ValInferenceContractError(
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
            raise ValInferenceContractError(
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
            raise ValInferenceContractError(
                f"{label} changed while it was read: {resolved}"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise ValInferenceContractError(
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
            raise ValInferenceContractError(
                f"{label} path changed while it was read: {resolved}"
            )
        observed = _sha256_bytes(payload)
        if expected is not None and observed != expected:
            raise ValInferenceContractError(
                f"{label} SHA mismatch: {observed} != {expected}"
            )
        return resolved, payload, observed, after
    except ValInferenceContractError:
        raise
    except OSError as error:
        raise ValInferenceContractError(
            f"cannot safely read {label}: {resolved}"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _directory(path: Path, label: str, *, create: bool = False) -> Path:
    _reject_path(path, label)
    if create and not os.path.lexists(path):
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            # Concurrent shard workers may create the same safe parent.
            pass
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValInferenceContractError(
            f"{label} must be a non-symlink directory: {path}"
        )
    resolved = path.resolve(strict=True)
    _reject_path(resolved, label)
    return resolved


def _verified_bytes(path: Path, expected_sha: str, label: str) -> bytes:
    _resolved, payload, _observed, _metadata = _safe_file_snapshot(
        path,
        label,
        expected_sha=expected_sha,
    )
    return payload


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return selector._strict_json_bytes(payload, label)
    except Exception as error:
        raise ValInferenceContractError(str(error)) from error


def _verified_json(
    path: Path,
    expected_sha: str,
    label: str,
) -> dict[str, Any]:
    value = _strict_json_bytes(_verified_bytes(path, expected_sha, label), label)
    if not isinstance(value, dict):
        raise ValInferenceContractError(f"{label} must be a JSON object")
    return value


def _strict_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        return selector._strict_jsonl(payload, label)
    except Exception as error:
        raise ValInferenceContractError(str(error)) from error


def _write_new(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
        os.rename(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_inside_generation(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _copy_inside_generation(
    source: Path,
    destination: Path,
    *,
    expected_sha: str,
    expected_bytes: int,
) -> dict[str, Any]:
    source, source_payload, observed, source_stat = _safe_file_snapshot(
        source,
        "shard output",
        expected_sha=expected_sha,
    )
    copied = len(source_payload)
    if copied != expected_bytes:
        raise ValInferenceContractError(
            f"copied shard output changed: {source}"
        )
    try:
        with destination.open("xb") as output:
            output.write(source_payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    try:
        (
            destination_resolved,
            destination_payload,
            destination_sha,
            destination_stat,
        ) = _safe_file_snapshot(
            destination,
            "copied shard output",
            expected_sha=expected_sha,
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if destination_payload != source_payload or destination_sha != observed:
        destination.unlink(missing_ok=True)
        raise ValInferenceContractError(
            f"copied shard output changed: {source}"
        )
    if (
        source_stat.st_dev == destination_stat.st_dev
        and source_stat.st_ino == destination_stat.st_ino
    ):
        destination.unlink(missing_ok=True)
        raise ValInferenceContractError("final output must not hardlink a shard")
    return {
        "path": str(destination_resolved),
        "sha256": observed,
        "bytes": copied,
    }


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(stage: Path, final: Path) -> None:
    if stage.parent != final.parent:
        raise ValInferenceContractError("generation rename must share one parent")
    if os.path.lexists(final):
        raise FileExistsError(f"refusing to overwrite {final}")
    _fsync_dir(stage)
    _fsync_dir(stage.parent)
    os.rename(stage, final)
    _fsync_dir(final.parent)


def _epoch(value: str) -> int:
    try:
        epoch = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch must be an integer") from error
    if epoch not in selector.EXPECTED_CANDIDATE_EPOCHS:
        raise argparse.ArgumentTypeError(
            "epoch must be one of "
            + ",".join(
                str(value)
                for value in selector.EXPECTED_CANDIDATE_EPOCHS
            )
            + "; e30 is withdrawn"
        )
    return epoch


def _num_shards(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "num-shards must be an integer"
        ) from error
    if not 1 <= result <= MAX_NUM_SHARDS:
        raise argparse.ArgumentTypeError(
            f"num-shards must be in [1,{MAX_NUM_SHARDS}]"
        )
    return result


def _preflight_artifact(path: Path, expected_sha: str) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    resolved = _regular_file(path, "validation inference preflight")
    payload = _verified_json(resolved, expected_sha, "validation preflight")
    expected_keys = {
        "format",
        "status",
        "split",
        "test_visible",
        "candidate_epochs",
        "candidate_bundle",
        "val_inputs_receipt",
        "pipeline_receipt",
        "pipeline_source",
        "inference_entrypoint",
        "coverage",
        "receipt_payload_sha256",
    }
    if set(payload) != expected_keys:
        raise ValInferenceContractError("validation preflight schema mismatch")
    claimed = selector.require_sha256(
        payload["receipt_payload_sha256"],
        "preflight payload SHA",
    )
    body = dict(payload)
    body.pop("receipt_payload_sha256")
    if _payload_sha(body) != claimed:
        raise ValInferenceContractError("preflight payload SHA mismatch")
    if (
        payload["format"] != PREFLIGHT_FORMAT
        or payload["status"] != "complete"
        or payload["split"] != "val"
        or payload["test_visible"] is not False
        or payload["candidate_epochs"]
        != list(selector.EXPECTED_CANDIDATE_EPOCHS)
        or not isinstance(payload["pipeline_source"], dict)
        or payload["pipeline_source"].get("origin")
        != "git@github.com:Xiangyue-Zhang/SemTalk.git"
        or payload["pipeline_source"].get("clean") is not True
        or payload["pipeline_source"].get("detached") is not True
        or payload["pipeline_source"].get("local_branches_at_commit") != []
    ):
        raise ValInferenceContractError("preflight is not val-only")
    artifact = {
        "path": str(resolved),
        "sha256": expected_sha,
        "receipt_payload_sha256": claimed,
    }
    return artifact, payload


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    _reject_path(args.output, "preflight output")
    if os.path.lexists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    val_artifact, coverage = selector.validate_val_inputs(
        args.val_inputs,
        args.expected_val_inputs_sha256,
    )
    pipeline_artifact, pipeline = selector.validate_pipeline(
        args.pipeline,
        args.expected_pipeline_sha256,
    )
    expected_selected = {
        stage: pipeline["fixed_checkpoints"][stage]["sha256"]
        for stage in ("face", "hands", "upper", "lower", "global")
    }
    bundle = selector.validate_candidate_bundle(
        manifest_path=args.candidate_manifest,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
        status_path=args.candidate_status,
        expected_status_sha256=args.expected_candidate_status_sha256,
        frozen_inputs_path=args.frozen_inputs,
        expected_frozen_inputs_sha256=args.expected_frozen_inputs_sha256,
        expected_selected_prerequisite_sha256=expected_selected,
    )
    for value in (
        args.output,
        bundle,
        val_artifact,
        coverage,
        pipeline_artifact,
        pipeline,
    ):
        _reject_forbidden(value, "preflight input")
    preflight = _with_payload_sha(
        {
            "format": PREFLIGHT_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "candidate_epochs": list(selector.EXPECTED_CANDIDATE_EPOCHS),
            "candidate_bundle": {
                **bundle,
                "candidates": {
                    str(epoch): bundle["candidates"][epoch]
                    for epoch in selector.EXPECTED_CANDIDATE_EPOCHS
                },
            },
            "val_inputs_receipt": val_artifact,
            "pipeline_receipt": pipeline_artifact,
            "pipeline_source": pipeline["source"],
            "inference_entrypoint": pipeline["inference_entrypoint"],
            "coverage": selector.public_val_coverage(coverage),
        }
    )
    _write_new(args.output, _canonical_json_bytes(preflight))
    return _artifact(
        args.output,
        payload_sha=preflight["receipt_payload_sha256"],
    )


def _load_pinned_helper(
    pipeline: Mapping[str, Any],
) -> ModuleType:
    entrypoint = pipeline["inference_helper"]
    path = _regular_file(
        Path(entrypoint["path"]),
        "pinned validation inference helper",
    )
    source_snapshot = _verified_bytes(
        path,
        entrypoint["sha256"],
        "pinned validation inference helper",
    )
    expected_helper = pipeline.get("source_closure", {}).get(
        "scripts/show_base/semtalk_base_inference_core.py"
    )
    if entrypoint != expected_helper:
        raise ValInferenceContractError(
            "inference helper differs from the fresh source closure"
        )
    name = f"_semtalk_base_inference_core_{entrypoint['sha256']}"
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = importlib.util.spec_from_loader(
        name,
        loader=None,
        origin=str(path),
    )
    try:
        code = compile(source_snapshot, str(path), "exec")
        exec(code, module.__dict__)
    except Exception as error:
        raise ValInferenceContractError(
            f"cannot execute verified helper snapshot {path}"
        ) from error
    required = set(selector.INFERENCE_HELPERS) | {
        "_strict_load_freeze_eval",
        "_normalize_data_parallel_state",
        "_read_verified_checkpoint_snapshot",
        "_torch_load_checkpoint",
        "_finite_state_dict",
        "_validate_base_model_state_schema",
        "_model_args",
        "_joint_masks",
        "deterministic_npz_bytes",
        "RELEASED_ALL_SPEAKERS_MODELS",
        "OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT",
        "PRE_FRAMES",
        "STRIDE",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise ValInferenceContractError(
            f"pinned inference helper lacks {missing}"
        )
    return module


def _pinned_helper_root(
    helper: ModuleType,
    pipeline: Mapping[str, Any],
) -> Path:
    entrypoint = _regular_file(
        Path(pipeline["inference_helper"]["path"]),
        "pinned validation inference helper",
    )
    helper_file_value = getattr(helper, "__file__", None)
    if not isinstance(helper_file_value, str) or not helper_file_value:
        raise ValInferenceContractError(
            "pinned inference helper has no source path"
        )
    helper_file = _regular_file(
        Path(helper_file_value),
        "loaded pinned validation inference helper",
    )
    if helper_file != entrypoint:
        raise ValInferenceContractError(
            "loaded pinned inference helper path mismatch"
        )
    if (
        helper_file.name != "semtalk_base_inference_core.py"
        or helper_file.parent.name != "show_base"
        or helper_file.parent.parent.name != "scripts"
    ):
        raise ValInferenceContractError(
            "pinned inference helper is outside scripts/show_base"
        )
    root = helper_file.parents[2]
    if root / "scripts" / "show_base" / helper_file.name != helper_file:
        raise ValInferenceContractError(
            "cannot resolve pinned inference helper project root"
        )
    return root


def _pinned_joint_mask_arrays(
    helper: ModuleType,
    pipeline: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if (
        type(getattr(helper, "POSE_DIM", None)) is not int
        or helper.POSE_DIM != SMPLX_POSE_DIM
    ):
        raise ValInferenceContractError(
            "pinned inference helper has unexpected SMPL-X pose dimension"
        )
    root = _pinned_helper_root(helper, pipeline)
    context_entry = pipeline.get("source_closure", {}).get(
        PINNED_JOINT_CONTEXT_SOURCE["relative_path"]
    )
    authority_entry = pipeline.get("source_closure", {}).get(
        PINNED_JOINT_AUTHORITY_SOURCE["relative_path"]
    )
    if not isinstance(context_entry, dict) or not isinstance(
        authority_entry, dict
    ):
        raise ValInferenceContractError(
            "joint-mask sources are absent from the fresh source closure"
        )
    source_path = root / PINNED_JOINT_CONTEXT_SOURCE["relative_path"]
    payload = _verified_bytes(
        source_path,
        PINNED_JOINT_CONTEXT_SOURCE["sha256"],
        "pinned joint-context source",
    )
    if (
        _git_blob_sha1(payload)
        != PINNED_JOINT_CONTEXT_SOURCE["git_blob_sha1"]
        or source_path != Path(context_entry.get("path", ""))
        or context_entry.get("sha256")
        != PINNED_JOINT_CONTEXT_SOURCE["sha256"]
        or context_entry.get("git_blob_sha1")
        != PINNED_JOINT_CONTEXT_SOURCE["git_blob_sha1"]
    ):
        raise ValInferenceContractError(
            "pinned joint-context Git blob mismatch"
        )
    authority_path = root / PINNED_JOINT_AUTHORITY_SOURCE["relative_path"]
    authority_payload = _verified_bytes(
        authority_path,
        PINNED_JOINT_AUTHORITY_SOURCE["sha256"],
        "pinned joint authority source",
    )
    if (
        _git_blob_sha1(authority_payload)
        != PINNED_JOINT_AUTHORITY_SOURCE["git_blob_sha1"]
        or authority_path != Path(authority_entry.get("path", ""))
        or authority_entry.get("sha256")
        != PINNED_JOINT_AUTHORITY_SOURCE["sha256"]
        or authority_entry.get("git_blob_sha1")
        != PINNED_JOINT_AUTHORITY_SOURCE["git_blob_sha1"]
    ):
        raise ValInferenceContractError(
            "pinned joint authority Git blob mismatch"
        )
    module = ModuleType(
        "_semtalk_pinned_show_base_joints_"
        f"{PINNED_JOINT_CONTEXT_SOURCE['sha256']}"
    )
    module.__file__ = str(source_path)
    try:
        code = compile(
            payload,
            str(source_path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except Exception as error:
        raise ValInferenceContractError(
            "cannot execute pinned joint-context source"
        ) from error
    formal_joint_context = getattr(module, "formal_joint_context", None)
    if not callable(formal_joint_context):
        raise ValInferenceContractError(
            "pinned joint-context source lacks formal_joint_context"
        )
    try:
        context = formal_joint_context("beat_smplx_joints")
    except Exception as error:
        raise ValInferenceContractError(
            "pinned formal joint context failed"
        ) from error
    if (
        not isinstance(context, dict)
        or set(context)
        != {"ori_joint_list", "target_joint_sets", "masks", "joints"}
        or type(context["joints"]) is not int
        or context["joints"] != SMPLX_POSE_DIM // 3
    ):
        raise ValInferenceContractError(
            "pinned formal joint context has invalid schema"
        )
    source = context["ori_joint_list"]
    targets = context["target_joint_sets"]
    masks = context["masks"]
    expected_joint_counts = {"face": 1, "upper": 13, "hands": 30, "lower": 9}
    if (
        not isinstance(source, dict)
        or len(source) != SMPLX_POSE_DIM // 3
        or not isinstance(targets, dict)
        or set(targets) != set(expected_joint_counts)
        or not isinstance(masks, dict)
        or set(masks) != set(expected_joint_counts)
    ):
        raise ValInferenceContractError(
            "pinned formal joint context has invalid coverage"
        )
    for index, (joint_name, location) in enumerate(source.items(), start=1):
        if (
            not isinstance(joint_name, str)
            or not joint_name
            or not isinstance(location, list)
            or len(location) != 2
            or type(location[0]) is not int
            or type(location[1]) is not int
            or location != [3, index * 3]
        ):
            raise ValInferenceContractError(
                "pinned SMPL-X source joint layout is malformed"
            )

    for name, expected_count in expected_joint_counts.items():
        target = targets[name]
        if (
            not isinstance(target, dict)
            or len(target) != expected_count
            or any(
                not isinstance(joint_name, str)
                or type(width) is not int
                or width != 3
                or joint_name not in source
                for joint_name, width in target.items()
            )
        ):
            raise ValInferenceContractError(
                f"{name}: malformed pinned joint target"
            )

    result: dict[str, np.ndarray] = {}
    observed = np.zeros(SMPLX_POSE_DIM, dtype=np.int64)
    for name in ("face", "upper", "hands", "lower"):
        expected = np.zeros(SMPLX_POSE_DIM, dtype=bool)
        for joint_name in targets[name]:
            width, end = source[joint_name]
            expected[end - width : end] = True
        raw_mask = masks[name]
        if (
            not isinstance(raw_mask, np.ndarray)
            or raw_mask.shape != (SMPLX_POSE_DIM,)
            or not np.issubdtype(raw_mask.dtype, np.number)
            or not bool(np.isfinite(raw_mask).all())
            or not bool(np.logical_or(raw_mask == 0, raw_mask == 1).all())
        ):
            raise ValInferenceContractError(
                f"{name}: malformed pinned joint mask"
            )
        mask = raw_mask.astype(bool, copy=True)
        if (
            not np.array_equal(mask, expected)
            or int(mask.sum()) != expected_joint_counts[name] * 3
            or _sha256_bytes(mask.tobytes())
            != PINNED_JOINT_MASK_SHA256[name]
        ):
            raise ValInferenceContractError(
                f"{name}: pinned joint mask contract mismatch"
            )
        observed += mask.astype(np.int64)
        if name != "face":
            result[name] = mask
    if bool((observed > 1).any()):
        raise ValInferenceContractError(
            "face/upper/hands/lower joint masks overlap"
        )
    receipt = _with_payload_sha(
        {
            "format": "semtalk_pinned_joint_masks_v1",
            "source": dict(pipeline["source"]),
            "context_source": dict(context_entry),
            "authority_source": dict(authority_entry),
            "pose_dim": SMPLX_POSE_DIM,
            "dtype": "bool",
            "masks": {
                name: {
                    "elements": expected_joint_counts[name] * 3,
                    "sha256": PINNED_JOINT_MASK_SHA256[name],
                }
                for name in ("face", "upper", "hands", "lower")
            },
        }
    )
    return result, receipt


def _joint_masks_from_arrays(
    arrays: Mapping[str, np.ndarray],
    device: Any,
) -> dict[str, Any]:
    import torch

    if set(arrays) != {"upper", "hands", "lower"}:
        raise ValInferenceContractError(
            "pinned joint mask array coverage mismatch"
        )
    result: dict[str, Any] = {}
    for name in ("upper", "hands", "lower"):
        mask = arrays[name]
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.dtype(bool)
            or mask.shape != (SMPLX_POSE_DIM,)
        ):
            raise ValInferenceContractError(
                f"{name}: malformed pinned joint mask array"
            )
        result[name] = torch.from_numpy(mask).to(device=device)
    return result


def _load_preflight_children(
    preflight: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    val_artifact = preflight["val_inputs_receipt"]
    pipeline_artifact = preflight["pipeline_receipt"]
    val_inputs = _verified_json(
        Path(val_artifact["path"]),
        val_artifact["sha256"],
        "frozen val inputs",
    )
    audited_pipeline_artifact, pipeline = selector.validate_pipeline(
        Path(pipeline_artifact["path"]),
        pipeline_artifact["sha256"],
        expected_source=preflight["pipeline_source"],
    )
    if (
        val_inputs.get("split") != "val"
        or val_inputs.get("test_visible") is not False
        or pipeline.get("split") != "val"
        or pipeline.get("test_visible") is not False
        or pipeline.get("source") != preflight["pipeline_source"]
        or pipeline.get("inference_entrypoint")
        != preflight["inference_entrypoint"]
        or any(
            audited_pipeline_artifact.get(key) != pipeline_artifact.get(key)
            for key in ("path", "sha256", "receipt_payload_sha256")
        )
    ):
        raise ValInferenceContractError("preflight child is not val-only")

    canonical_artifact = val_inputs["canonical_manifest"]
    canonical_payload = _verified_bytes(
        Path(canonical_artifact["path"]),
        canonical_artifact["sha256"],
        "canonical val manifest",
    )
    canonical_rows = _strict_jsonl_bytes(
        canonical_payload,
        "canonical val manifest",
    )
    expected = preflight["coverage"]
    if len(canonical_rows) != selector.EXPECTED_VAL_CLIPS:
        raise ValInferenceContractError("canonical val rows != 1715")

    audio_by_id: dict[str, dict[str, Any]] = {}
    for artifact in val_inputs["audio_manifests"]:
        payload = _verified_bytes(
            Path(artifact["path"]),
            artifact["sha256"],
            "validation audio manifest",
        )
        for row in _strict_jsonl_bytes(payload, "validation audio manifest"):
            clip_id = row.get("clip_id")
            if (
                not isinstance(clip_id, str)
                or clip_id in audio_by_id
                or row.get("split") != "val"
            ):
                raise ValInferenceContractError(
                    "audio manifests are not exact-once val"
                )
            audio_by_id[clip_id] = row

    ordered = expected.get("_ordered_clips")
    if ordered is not None:
        raise ValInferenceContractError(
            "public preflight must not expose private coverage fields"
        )
    seen: set[str] = set()
    for position, row in enumerate(canonical_rows):
        clip_id = row.get("clip_id")
        frames = selector.require_exact_int(
            row.get("frames"),
            f"canonical row {position} frames",
        )
        if (
            row.get("split") != "val"
            or not isinstance(clip_id, str)
            or clip_id in seen
            or selector.canonical_clip_id(clip_id)
            is None  # pragma: no cover - helper raises first
            or clip_id not in audio_by_id
            or audio_by_id[clip_id].get("frames") != frames
        ):
            raise ValInferenceContractError(
                f"invalid canonical/audio row at {position}"
            )
        seen.add(clip_id)
        for key in ("canonical_npz",):
            if key not in row:
                raise ValInferenceContractError(
                    f"canonical row lacks {key}"
                )
            _reject_path(Path(str(row[key])), f"{clip_id} canonical path")
        _reject_path(
            Path(str(audio_by_id[clip_id]["audio_feature_npz"])),
            f"{clip_id} audio path",
        )
    if len(seen) != selector.EXPECTED_VAL_CLIPS:
        raise ValInferenceContractError("val coverage is not exact 1715")
    return val_inputs, pipeline, canonical_rows, audio_by_id


def _pinned_project_module(
    pipeline: Mapping[str, Any],
    *,
    module_name: str,
    relative_path: str,
) -> ModuleType:
    """Import one project module only from the fresh pinned checkout."""

    entry = pipeline.get("source_closure", {}).get(relative_path)
    if not isinstance(entry, dict):
        raise ValInferenceContractError(
            f"{relative_path} is absent from the fresh source closure"
        )
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ValInferenceContractError(
            f"{module_name} has no auditable source path"
        )
    resolved = _regular_file(Path(module_file), f"pinned {module_name}")
    snapshot = _verified_bytes(
        resolved,
        entry.get("sha256"),
        f"pinned {module_name}",
    )
    if (
        resolved != Path(str(entry.get("path", "")))
        or len(snapshot) != entry.get("bytes")
    ):
        raise ValInferenceContractError(
            f"{module_name} was imported from another checkout"
        )
    return module


def _load_models(
    helper: ModuleType,
    *,
    epoch: int,
    preflight: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    feature_builder = _pinned_project_module(
        pipeline,
        module_name="scripts.show_base.build_base_features",
        relative_path="scripts/show_base/build_base_features.py",
    )
    semtalk_module = _pinned_project_module(
        pipeline,
        module_name="models.semtalk",
        relative_path="models/semtalk.py",
    )

    candidate = preflight["candidate_bundle"]["candidates"][str(epoch)]
    bundle = preflight["candidate_bundle"]
    updates_per_epoch = selector.require_exact_int(
        bundle.get("updates_per_epoch"),
        "selected Base topology updates per epoch",
    )
    if updates_per_epoch not in {248, 1988}:
        raise ValInferenceContractError(
            "selected Base topology updates per epoch changed"
        )
    for label, artifact in (
        ("Base candidate", candidate),
        ("Base candidate manifest", bundle["manifest"]),
        ("Base candidate status", bundle["status"]),
        ("Base frozen inputs", bundle["frozen_inputs"]),
    ):
        _reject_path(Path(artifact["path"]), label)
    frozen_payload = _verified_json(
        Path(bundle["frozen_inputs"]["path"]),
        bundle["frozen_inputs"]["sha256"],
        "Base frozen inputs",
    )
    _reject_absolute_paths_in_tree(frozen_payload, "Base frozen inputs")
    candidate_resolved, candidate_snapshot, observed_candidate_sha = (
        helper._read_verified_checkpoint_snapshot(
            Path(candidate["path"]),
            candidate["sha256"],
            "preflight-bound official-adapt Base candidate",
        )
    )
    if len(candidate_snapshot) != candidate["bytes"]:
        raise ValInferenceContractError("Base candidate byte count changed")
    base_payload = helper._torch_load_checkpoint(
        candidate_snapshot,
        candidate_resolved,
    )
    if set(base_payload) != {"model_state", "audit"}:
        raise ValInferenceContractError(
            "invalid official-adapt Base candidate envelope"
        )
    helper._finite_state_dict(
        base_payload["model_state"],
        candidate_resolved,
    )
    helper._validate_base_model_state_schema(
        base_payload["model_state"],
        candidate_resolved,
    )
    audit = base_payload.get("audit")
    if (
        not isinstance(audit, dict)
        or audit.get("format")
        != helper.OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT
        or audit.get("completed_epochs") != epoch
        or audit.get("optimizer_updates")
        != epoch * updates_per_epoch
        or audit.get("frozen_receipt_sha256")
        != bundle["frozen_inputs"]["receipt_sha256"]
        or audit.get("official_base_checkpoint_sha256")
        != helper.RELEASED_ALL_SPEAKERS_MODELS["base"]["sha256"]
        or audit.get("speaker_scope") != "SHOW_All"
        or audit.get("speaker_rows") != [0, 1, 2, 3]
        or audit.get("vq_models_in_training_graph") is not False
        or audit.get("all_model_state_tensors_finite") is not True
    ):
        raise ValInferenceContractError(
            "Base candidate audit is not preflight/frozen-input bound"
        )
    base = semtalk_module.semtalk_base(helper._model_args()).to(device)
    helper._strict_load_freeze_eval(
        base,
        helper._normalize_data_parallel_state(
            base_payload["model_state"],
            Path(candidate["path"]),
        ),
        path=Path(candidate["path"]),
    )
    models: dict[str, Any] = {"base": base}
    receipts: dict[str, Any] = {
        "base": {
            "path": str(candidate_resolved),
            "sha256": observed_candidate_sha,
            "bytes": len(candidate_snapshot),
            "stage": "base",
            "candidate_epoch": epoch,
            "optimizer_updates": audit["optimizer_updates"],
            "updates_per_epoch": updates_per_epoch,
            "frozen_receipt_sha256": audit["frozen_receipt_sha256"],
            "strict_state_dict_load": True,
            "all_model_state_tensors_finite": True,
            "frozen_eval": True,
        }
    }
    fixed = pipeline["fixed_checkpoints"]
    selection = pipeline["prerequisite_selection"]
    try:
        selected_models, selected_records, bridge = (
            feature_builder.load_val_selected_models(
                SimpleNamespace(
                    prerequisite_selection_json=Path(selection["path"]),
                    expected_prerequisite_selection_sha256=selection[
                        "sha256"
                    ],
                    device=device,
                ),
                retain_global=True,
            )
        )
    except Exception as error:
        raise ValInferenceContractError(
            "cannot strict-load the five validation-selected SHOW models"
        ) from error
    expected_stages = {"face", "hands", "upper", "lower", "global"}
    if (
        set(selected_models) != expected_stages
        or set(selected_records) != expected_stages
        or bridge.get("selection")
        != {
            key: selection[key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        }
    ):
        raise ValInferenceContractError(
            "five-model validation selection bridge changed"
        )
    for stage in ("face", "hands", "upper", "lower", "global"):
        record = selected_records[stage]
        actual = {
            "stage": record.get("formal_stage"),
            "path": record.get("path"),
            "sha256": record.get("sha256"),
            "bytes": record.get("bytes"),
            "source": record.get("prerequisite_source"),
            "selection_split": record.get("selection_split"),
            "test_visible": record.get("test_visible"),
            "epoch": record.get("selected_epoch"),
            "optimizer_updates": record.get(
                "selected_optimizer_updates"
            ),
            "updates_per_epoch": record.get(
                "selected_updates_per_epoch"
            ),
            "candidate_audit_sha256": record.get(
                "candidate_audit_sha256"
            ),
            "selection_metric": record.get("selection_metric"),
            "measurement_receipt": record.get("measurement_receipt"),
        }
        if actual != fixed[stage]:
            raise ValInferenceContractError(
                f"actual-loaded {stage} model differs from fresh fixed five"
            )
        models[stage] = selected_models[stage]
        receipts[stage] = {
            **actual,
            "prerequisite_selection": dict(selection),
            "model_state_tensors": record.get("model_state_tensors"),
            "model_state_schema_sha256": record.get(
                "model_state_schema_sha256"
            ),
            "strict_state_dict_load": record.get(
                "strict_state_dict_load"
            ),
            "all_model_state_tensors_finite": record.get(
                "all_model_state_tensors_finite"
            ),
            "frozen_eval": record.get("frozen_eval"),
        }

    # Reject an already-imported shadow package even when the checkpoint
    # loader itself returned plausible receipt dictionaries.
    for module_name, relative in (
        ("models.motion_representation", "models/motion_representation.py"),
        ("models.motion_encoder", "models/motion_encoder.py"),
        ("models.rvq", "models/rvq.py"),
        ("models.encdec", "models/encdec.py"),
        ("models.residual_vq", "models/residual_vq.py"),
        ("models.quantizer", "models/quantizer.py"),
        ("models.resnet", "models/resnet.py"),
    ):
        _pinned_project_module(
            pipeline,
            module_name=module_name,
            relative_path=relative,
        )

    for model in models.values():
        model.eval()
        model.requires_grad_(False)
    if set(models) != {"base", *expected_stages} or set(receipts) != {
        "base",
        *expected_stages,
    }:
        raise ValInferenceContractError(
            "runtime model set is not Base plus five SHOW prerequisites"
        )
    torch.cuda.empty_cache()
    return models, receipts


def _set_deterministic(seed: int) -> None:
    import random
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _runtime(device: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    parsed = torch.device(device)
    if parsed.type != "cuda" or not torch.cuda.is_available():
        raise ValInferenceContractError("formal shard inference requires CUDA")
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    contract = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "seed": seed,
        "deterministic_algorithms": True,
        "window": 64,
        "pre_frames": 4,
        "stride": 60,
    }
    return contract, {
        "device": str(parsed),
        "name": properties.name,
        "major": int(properties.major),
        "minor": int(properties.minor),
        "total_memory": int(properties.total_memory),
    }


def _shard_name(shard_id: int, num_shards: int) -> str:
    return f"shard-{shard_id:05d}-of-{num_shards:05d}"


def _validate_output_root(path: Path) -> Path:
    _reject_path(path, "inference output root")
    return _directory(path, "inference output root", create=True)


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.shard_id < args.num_shards:
        raise ValInferenceContractError("shard-id is outside num-shards")
    preflight_artifact, preflight = _preflight_artifact(
        args.preflight,
        args.expected_preflight_sha256,
    )
    if str(args.epoch) == "30":
        raise ValInferenceContractError("e30 is withdrawn")
    candidate = preflight["candidate_bundle"]["candidates"][str(args.epoch)]
    _reject_forbidden(candidate, "Base candidate")
    val_inputs, pipeline, canonical_rows, audio_by_id = (
        _load_preflight_children(preflight)
    )
    helper = _load_pinned_helper(pipeline)
    joint_mask_arrays, joint_mask_receipt = _pinned_joint_mask_arrays(
        helper,
        pipeline,
    )
    output_root = _validate_output_root(args.output_root)
    shards_root = output_root / SHARDS_DIRECTORY
    _directory(shards_root, "shards root", create=True)
    name = _shard_name(args.shard_id, args.num_shards)
    final_root = shards_root / name
    _reject_path(final_root, "shard output")
    if os.path.lexists(final_root):
        raise FileExistsError(f"refusing to overwrite {final_root}")
    stage = shards_root / f".{name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    _reject_path(stage, "shard staging output")
    stage.mkdir()
    prediction_stage = stage / "predictions" / "val"
    ground_truth_stage = stage / "ground-truth" / "val"
    prediction_stage.mkdir(parents=True)
    ground_truth_stage.mkdir(parents=True)
    prediction_final = final_root / "predictions" / "val"
    ground_truth_final = final_root / "ground-truth" / "val"

    try:
        _set_deterministic(args.seed)
        runtime_contract, device_receipt = _runtime(args.device, args.seed)
        runtime_contract["joint_masks"] = joint_mask_receipt
        runtime_contract["auxiliary_loss_bypass"] = (
            helper._inference_auxiliary_loss_bypass_receipt()
        )
        torch = __import__("torch")
        masks = _joint_masks_from_arrays(
            joint_mask_arrays,
            torch.device(args.device),
        )
        models, model_receipts = _load_models(
            helper,
            epoch=args.epoch,
            preflight=preflight,
            pipeline=pipeline,
            device=args.device,
        )
        rows: list[dict[str, Any]] = []
        frame_count = 0
        for position, canonical_row in enumerate(canonical_rows):
            if position % args.num_shards != args.shard_id:
                continue
            clip_id = str(canonical_row["clip_id"])
            output_id = selector.canonical_clip_id(clip_id)
            canonical, frames = helper._load_canonical_clip(canonical_row)
            audio = helper._load_audio_features(
                audio_by_id[clip_id],
                expected_frames=frames,
            )
            expected_calls = max(
                1,
                math.ceil(
                    (frames - helper.PRE_FRAMES) / helper.STRIDE
                ),
            )
            with (
                torch.inference_mode(),
                helper._inference_only_auxiliary_loss_bypass(
                    models["base"],
                    expected_calls=expected_calls,
                ),
            ):
                prediction = helper._infer_clip(
                    pose=canonical["pose"],
                    trans=canonical["trans"],
                    beat=audio["beat"],
                    hubert=audio["hubert"],
                    speaker_id=helper.SHOW_SPEAKER_IDS[
                        canonical_row["speaker"]
                    ],
                    models=models,
                    masks=masks,
                    device=args.device,
                )
            prediction_arrays = helper._output_arrays(
                betas=canonical["beta"][0],
                poses=prediction["poses"],
                expressions=prediction["expressions"],
                trans=prediction["trans"],
            )
            ground_truth_arrays = helper._output_arrays(
                betas=canonical["beta"][0],
                poses=canonical["pose"],
                expressions=canonical["facial"],
                trans=canonical["trans"],
            )
            for role, arrays in (
                ("prediction", prediction_arrays),
                ("ground truth", ground_truth_arrays),
            ):
                for field, array in arrays.items():
                    value = np.asarray(array)
                    if value.dtype.kind in "fc" and not np.isfinite(value).all():
                        raise ValInferenceContractError(
                            f"{clip_id} {role} {field} is non-finite"
                        )
            prediction_payload = helper.deterministic_npz_bytes(
                prediction_arrays
            )
            ground_truth_payload = helper.deterministic_npz_bytes(
                ground_truth_arrays
            )
            prediction_path = prediction_stage / f"res_{output_id}.npz"
            ground_truth_path = ground_truth_stage / f"gt_{output_id}.npz"
            _write_inside_generation(prediction_path, prediction_payload)
            _write_inside_generation(ground_truth_path, ground_truth_payload)
            rows.append(
                {
                    "canonical_position": position,
                    "global_index": canonical_row["global_index"],
                    "split": "val",
                    "source_clip_id": clip_id,
                    "canonical_clip_id": output_id,
                    "frames": frames,
                    "epoch": args.epoch,
                    "candidate_checkpoint_sha256": candidate["sha256"],
                    "prediction": {
                        "path": str(
                            prediction_final / prediction_path.name
                        ),
                        "sha256": _sha256_bytes(prediction_payload),
                        "bytes": len(prediction_payload),
                    },
                    "ground_truth": {
                        "path": str(
                            ground_truth_final / ground_truth_path.name
                        ),
                        "sha256": _sha256_bytes(ground_truth_payload),
                        "bytes": len(ground_truth_payload),
                    },
                }
            )
            frame_count += frames
            if args.progress_every and len(rows) % args.progress_every == 0:
                print(
                    f"shard {args.shard_id}/{args.num_shards}: "
                    f"{len(rows)} clips",
                    flush=True,
                )
        expected_count = sum(
            1
            for position in range(selector.EXPECTED_VAL_CLIPS)
            if position % args.num_shards == args.shard_id
        )
        if len(rows) != expected_count:
            raise ValInferenceContractError("shard clip count mismatch")
        manifest_path = stage / SHARD_MANIFEST_FILENAME
        _write_inside_generation(
            manifest_path,
            _canonical_jsonl_bytes(rows),
        )
        manifest_final = final_root / SHARD_MANIFEST_FILENAME
        receipt = _with_payload_sha(
            {
                "format": SHARD_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "epoch": args.epoch,
                "candidate_checkpoint": {
                    "path": candidate["path"],
                    "sha256": candidate["sha256"],
                },
                "preflight_receipt": preflight_artifact,
                "assignment": ASSIGNMENT,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "clip_count": len(rows),
                "frame_count": frame_count,
                "prediction_files": len(rows),
                "ground_truth_files": len(rows),
                "manifest": {
                    "path": str(manifest_final),
                    "sha256": _sha256_file(manifest_path),
                },
                "model_receipts": model_receipts,
                "model_receipts_sha256": _payload_sha(model_receipts),
                "runtime_contract": runtime_contract,
                "runtime_contract_sha256": _payload_sha(runtime_contract),
                "device": device_receipt,
                "exact_once": True,
                "finite": True,
            }
        )
        _write_inside_generation(
            stage / SHARD_RECEIPT_FILENAME,
            _canonical_json_bytes(receipt),
        )
        _fsync_dir(prediction_stage)
        _fsync_dir(prediction_stage.parent)
        _fsync_dir(ground_truth_stage)
        _fsync_dir(ground_truth_stage.parent)
        _fsync_dir(stage)
        _publish_directory(stage, final_root)
        return _artifact(
            final_root / SHARD_RECEIPT_FILENAME,
            payload_sha=receipt["receipt_payload_sha256"],
        )
    except BaseException:
        if os.path.lexists(stage):
            shutil.rmtree(stage)
        raise


def _validate_shard(
    *,
    output_root: Path,
    shard_id: int,
    num_shards: int,
    epoch: int,
    candidate: Mapping[str, Any],
    preflight_artifact: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    canonical_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = output_root / SHARDS_DIRECTORY / _shard_name(shard_id, num_shards)
    root = _directory(root, f"shard {shard_id}")
    receipt_path = _regular_file(
        root / SHARD_RECEIPT_FILENAME,
        f"shard {shard_id} receipt",
    )
    (
        receipt_path,
        receipt_snapshot,
        _receipt_sha,
        _receipt_metadata,
    ) = _safe_file_snapshot(
        receipt_path,
        f"shard {shard_id} receipt",
    )
    receipt = _strict_json_bytes(
        receipt_snapshot,
        f"shard {shard_id} receipt",
    )
    if not isinstance(receipt, dict):
        raise ValInferenceContractError("shard receipt must be an object")
    expected_receipt_keys = {
        "format",
        "status",
        "split",
        "test_visible",
        "epoch",
        "candidate_checkpoint",
        "preflight_receipt",
        "assignment",
        "shard_id",
        "num_shards",
        "clip_count",
        "frame_count",
        "prediction_files",
        "ground_truth_files",
        "manifest",
        "model_receipts",
        "model_receipts_sha256",
        "runtime_contract",
        "runtime_contract_sha256",
        "device",
        "exact_once",
        "finite",
        "receipt_payload_sha256",
    }
    if set(receipt) != expected_receipt_keys:
        raise ValInferenceContractError("shard receipt schema mismatch")
    claimed = receipt.get("receipt_payload_sha256")
    body = dict(receipt)
    body.pop("receipt_payload_sha256", None)
    if (
        receipt.get("format") != SHARD_FORMAT
        or receipt.get("status") != "complete"
        or receipt.get("split") != "val"
        or receipt.get("test_visible") is not False
        or receipt.get("epoch") != epoch
        or receipt.get("candidate_checkpoint")
        != {"path": candidate["path"], "sha256": candidate["sha256"]}
        or receipt.get("preflight_receipt") != preflight_artifact
        or receipt.get("assignment") != ASSIGNMENT
        or receipt.get("shard_id") != shard_id
        or receipt.get("num_shards") != num_shards
        or receipt.get("exact_once") is not True
        or receipt.get("finite") is not True
        or not isinstance(claimed, str)
        or _payload_sha(body) != claimed
        or receipt.get("model_receipts_sha256")
        != _payload_sha(receipt.get("model_receipts", {}))
        or receipt.get("runtime_contract_sha256")
        != _payload_sha(receipt.get("runtime_contract", {}))
    ):
        raise ValInferenceContractError(f"invalid shard receipt {shard_id}")
    model_receipts = receipt.get("model_receipts")
    expected_model_stages = {
        "base",
        "face",
        "hands",
        "upper",
        "lower",
        "global",
    }
    if not isinstance(model_receipts, dict) or set(model_receipts) != (
        expected_model_stages
    ):
        raise ValInferenceContractError(
            f"shard {shard_id} model receipt coverage changed"
        )
    base_receipt = model_receipts["base"]
    updates_per_epoch = selector.require_exact_int(
        preflight.get("candidate_bundle", {}).get("updates_per_epoch"),
        "selected Base topology updates per epoch",
    )
    expected_base_keys = {
        "path",
        "sha256",
        "bytes",
        "stage",
        "candidate_epoch",
        "optimizer_updates",
        "updates_per_epoch",
        "frozen_receipt_sha256",
        "strict_state_dict_load",
        "all_model_state_tensors_finite",
        "frozen_eval",
    }
    if (
        not isinstance(base_receipt, dict)
        or set(base_receipt) != expected_base_keys
        or {
            key: base_receipt[key]
            for key in ("path", "sha256", "bytes")
        }
        != {
            key: candidate[key]
            for key in ("path", "sha256", "bytes")
        }
        or base_receipt["stage"] != "base"
        or base_receipt["candidate_epoch"] != epoch
        or base_receipt["updates_per_epoch"]
        != updates_per_epoch
        or base_receipt["optimizer_updates"]
        != epoch * updates_per_epoch
        or base_receipt["frozen_receipt_sha256"]
        != preflight["candidate_bundle"]["frozen_inputs"][
            "receipt_sha256"
        ]
        or base_receipt["strict_state_dict_load"] is not True
        or base_receipt["all_model_state_tensors_finite"] is not True
        or base_receipt["frozen_eval"] is not True
    ):
        raise ValInferenceContractError(
            f"shard {shard_id} Base actual-load receipt changed"
        )
    fixed = pipeline["fixed_checkpoints"]
    expected_prerequisite_keys = {
        "stage",
        "path",
        "sha256",
        "bytes",
        "source",
        "selection_split",
        "test_visible",
        "epoch",
        "optimizer_updates",
        "updates_per_epoch",
        "candidate_audit_sha256",
        "selection_metric",
        "measurement_receipt",
        "prerequisite_selection",
        "model_state_tensors",
        "model_state_schema_sha256",
        "strict_state_dict_load",
        "all_model_state_tensors_finite",
        "frozen_eval",
    }
    fixed_keys = set(next(iter(fixed.values())))
    for model_stage in ("face", "hands", "upper", "lower", "global"):
        actual = model_receipts[model_stage]
        if (
            not isinstance(actual, dict)
            or set(actual) != expected_prerequisite_keys
            or {key: actual[key] for key in fixed_keys}
            != fixed[model_stage]
            or actual["prerequisite_selection"]
            != pipeline["prerequisite_selection"]
            or type(actual["model_state_tensors"]) is not int
            or actual["model_state_tensors"] <= 0
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(actual["model_state_schema_sha256"]),
            )
            or actual["strict_state_dict_load"] is not True
            or actual["all_model_state_tensors_finite"] is not True
            or actual["frozen_eval"] is not True
        ):
            raise ValInferenceContractError(
                f"shard {shard_id} {model_stage} actual-load receipt "
                "differs from the fresh fixed five"
            )
    manifest = receipt.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != {"path", "sha256"}:
        raise ValInferenceContractError("invalid shard manifest receipt")
    manifest_path = _regular_file(
        Path(manifest["path"]),
        f"shard {shard_id} manifest",
    )
    if manifest_path != root / SHARD_MANIFEST_FILENAME:
        raise ValInferenceContractError("shard manifest path mismatch")
    rows = _strict_jsonl_bytes(
        _verified_bytes(
            manifest_path,
            manifest["sha256"],
            f"shard {shard_id} manifest",
        ),
        f"shard {shard_id} manifest",
    )
    expected_positions = [
        position
        for position in range(len(canonical_rows))
        if position % num_shards == shard_id
    ]
    if (
        [row.get("canonical_position") for row in rows]
        != expected_positions
        or receipt.get("clip_count") != len(rows)
        or receipt.get("prediction_files") != len(rows)
        or receipt.get("ground_truth_files") != len(rows)
        or receipt.get("frame_count")
        != sum(canonical_rows[position]["frames"] for position in expected_positions)
    ):
        raise ValInferenceContractError("shard coverage mismatch")
    prediction_dir = _directory(root / "predictions" / "val", "shard predictions")
    ground_truth_dir = _directory(
        root / "ground-truth" / "val",
        "shard ground truth",
    )
    expected_prediction: set[Path] = set()
    expected_ground_truth: set[Path] = set()
    row_keys = {
        "canonical_position",
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
    for row, position in zip(rows, expected_positions):
        if set(row) != row_keys:
            raise ValInferenceContractError("shard row schema mismatch")
        canonical = canonical_rows[position]
        clip_id = canonical["clip_id"]
        output_id = selector.canonical_clip_id(clip_id)
        if (
            row.get("global_index") != canonical["global_index"]
            or row.get("split") != "val"
            or row.get("source_clip_id") != clip_id
            or row.get("canonical_clip_id") != output_id
            or row.get("frames") != canonical["frames"]
            or row.get("epoch") != epoch
            or row.get("candidate_checkpoint_sha256")
            != candidate["sha256"]
        ):
            raise ValInferenceContractError(
                f"shard row mismatch at canonical position {position}"
            )
        for role, directory, filename, expected_set in (
            (
                "prediction",
                prediction_dir,
                f"res_{output_id}.npz",
                expected_prediction,
            ),
            (
                "ground_truth",
                ground_truth_dir,
                f"gt_{output_id}.npz",
                expected_ground_truth,
            ),
        ):
            artifact = row.get(role)
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"path", "sha256", "bytes"}
            ):
                raise ValInferenceContractError(f"invalid {role} receipt")
            path = _regular_file(Path(artifact["path"]), f"shard {role}")
            if path.parent != directory or path.name != filename:
                raise ValInferenceContractError(f"shard {role} path mismatch")
            if (
                path.stat().st_size != artifact["bytes"]
                or _sha256_file(path) != artifact["sha256"]
            ):
                raise ValInferenceContractError(f"shard {role} changed")
            expected_set.add(path)
    if (
        set(prediction_dir.iterdir()) != expected_prediction
        or set(ground_truth_dir.iterdir()) != expected_ground_truth
    ):
        raise ValInferenceContractError("shard output directory has extras")
    if set(root.iterdir()) != {
        root / SHARD_RECEIPT_FILENAME,
        root / SHARD_MANIFEST_FILENAME,
        root / "predictions",
        root / "ground-truth",
    }:
        raise ValInferenceContractError("shard root has unexpected entries")
    return receipt, rows


@contextmanager
def _finalize_lock(output_root: Path) -> Iterable[None]:
    lock_path = output_root / ".finalize.lock"
    descriptor = os.open(
        str(lock_path),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValInferenceContractError(
                "another finalizer holds the output lock"
            ) from error
        yield
    finally:
        os.close(descriptor)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    preflight_artifact, preflight = _preflight_artifact(
        args.preflight,
        args.expected_preflight_sha256,
    )
    candidate = preflight["candidate_bundle"]["candidates"][str(args.epoch)]
    _reject_forbidden(candidate, "Base candidate")
    _val_inputs, pipeline, canonical_rows, _audio = (
        _load_preflight_children(preflight)
    )
    output_root = _validate_output_root(args.output_root)
    final_root = output_root / FINAL_DIRECTORY
    _reject_path(final_root, "final validation output")
    with _finalize_lock(output_root):
        if os.path.lexists(final_root):
            raise FileExistsError(f"refusing to overwrite {final_root}")
        receipts: list[dict[str, Any]] = []
        all_rows: dict[int, dict[str, Any]] = {}
        common_model_sha: str | None = None
        common_runtime_sha: str | None = None
        for shard_id in range(args.num_shards):
            receipt, rows = _validate_shard(
                output_root=output_root,
                shard_id=shard_id,
                num_shards=args.num_shards,
                epoch=args.epoch,
                candidate=candidate,
                preflight_artifact=preflight_artifact,
                preflight=preflight,
                pipeline=pipeline,
                canonical_rows=canonical_rows,
            )
            if common_model_sha is None:
                common_model_sha = receipt["model_receipts_sha256"]
                common_runtime_sha = receipt["runtime_contract_sha256"]
            elif (
                receipt["model_receipts_sha256"] != common_model_sha
                or receipt["runtime_contract_sha256"] != common_runtime_sha
            ):
                raise ValInferenceContractError(
                    "shards disagree on model or software runtime"
                )
            receipts.append(receipt)
            for row in rows:
                position = row["canonical_position"]
                if position in all_rows:
                    raise ValInferenceContractError(
                        "duplicate canonical position across shards"
                    )
                all_rows[position] = row
        if set(all_rows) != set(range(selector.EXPECTED_VAL_CLIPS)):
            raise ValInferenceContractError(
                "shards do not exactly cover 1,715 validation clips"
            )

        stage = output_root / (
            f".{FINAL_DIRECTORY}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        _reject_path(stage, "final staging generation")
        stage.mkdir()
        prediction_stage = stage / "predictions" / "val"
        ground_truth_stage = stage / "ground-truth" / "val"
        prediction_stage.mkdir(parents=True)
        ground_truth_stage.mkdir(parents=True)
        prediction_final = final_root / "predictions" / "val"
        ground_truth_final = final_root / "ground-truth" / "val"
        final_rows: list[dict[str, Any]] = []
        try:
            for position in range(selector.EXPECTED_VAL_CLIPS):
                source = all_rows[position]
                output_id = source["canonical_clip_id"]
                prediction_name = f"res_{output_id}.npz"
                ground_truth_name = f"gt_{output_id}.npz"
                prediction = _copy_inside_generation(
                    Path(source["prediction"]["path"]),
                    prediction_stage / prediction_name,
                    expected_sha=source["prediction"]["sha256"],
                    expected_bytes=source["prediction"]["bytes"],
                )
                ground_truth = _copy_inside_generation(
                    Path(source["ground_truth"]["path"]),
                    ground_truth_stage / ground_truth_name,
                    expected_sha=source["ground_truth"]["sha256"],
                    expected_bytes=source["ground_truth"]["bytes"],
                )
                prediction["path"] = str(prediction_final / prediction_name)
                ground_truth["path"] = str(
                    ground_truth_final / ground_truth_name
                )
                final_rows.append(
                    {
                        "global_index": source["global_index"],
                        "split": "val",
                        "source_clip_id": source["source_clip_id"],
                        "canonical_clip_id": output_id,
                        "frames": source["frames"],
                        "epoch": args.epoch,
                        "candidate_checkpoint_sha256": candidate["sha256"],
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                    }
                )
            clip_payload = "".join(
                f"{row['canonical_clip_id']}\n" for row in final_rows
            ).encode("utf-8")
            clip_stage = stage / "talkshow_eval_clip_ids.txt"
            manifest_stage = stage / "final_manifest.jsonl"
            _write_inside_generation(clip_stage, clip_payload)
            _write_inside_generation(
                manifest_stage,
                _canonical_jsonl_bytes(final_rows),
            )
            coverage = preflight["coverage"]
            lineage = _with_payload_sha(
                {
                    "format": selector.VAL_INFERENCE_LINEAGE_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": args.epoch,
                    "candidate_checkpoint": {
                        "path": candidate["path"],
                        "sha256": candidate["sha256"],
                    },
                    "val_inputs_receipt": preflight[
                        "val_inputs_receipt"
                    ],
                    "pipeline_receipt": preflight["pipeline_receipt"],
                    "prediction_dir": str(prediction_final),
                    "ground_truth_dir": str(ground_truth_final),
                    "final_manifest": {
                        "path": str(final_root / manifest_stage.name),
                        "sha256": _sha256_file(manifest_stage),
                    },
                    "clip_manifest": {
                        "path": str(final_root / clip_stage.name),
                        "sha256": _sha256_file(clip_stage),
                    },
                    "clip_count": coverage["clip_count"],
                    "frame_count": coverage["frame_count"],
                    "window_count": coverage["window_count"],
                    "uncovered_tail_frames": coverage[
                        "uncovered_tail_frames"
                    ],
                    "clip_ids_sha256": coverage["clip_ids_sha256"],
                    "talkshow_window_manifest_sha256": coverage[
                        "talkshow_window_manifest_sha256"
                    ],
                    "prediction_files": selector.EXPECTED_VAL_CLIPS,
                    "ground_truth_files": selector.EXPECTED_VAL_CLIPS,
                    "exact_once": True,
                    "finite": True,
                }
            )
            lineage_stage = stage / LINEAGE_FILENAME
            _write_inside_generation(
                lineage_stage,
                _canonical_json_bytes(lineage),
            )
            _fsync_dir(prediction_stage)
            _fsync_dir(prediction_stage.parent)
            _fsync_dir(ground_truth_stage)
            _fsync_dir(ground_truth_stage.parent)
            _fsync_dir(stage)
            _publish_directory(stage, final_root)
        except BaseException:
            if os.path.lexists(stage):
                shutil.rmtree(stage)
            raise

        lineage_path = final_root / LINEAGE_FILENAME
        lineage_artifact = _artifact(
            lineage_path,
            payload_sha=lineage["receipt_payload_sha256"],
        )
        expected_coverage = {
            **preflight["coverage"],
            "_ordered_clips": [
                {
                    "global_index": row["global_index"],
                    "source_clip_id": row["clip_id"],
                    "canonical_clip_id": selector.canonical_clip_id(
                        row["clip_id"]
                    ),
                    "frames": row["frames"],
                }
                for row in canonical_rows
            ],
        }
        selector.validate_val_inference_lineage(
            lineage_path,
            lineage_artifact["sha256"],
            epoch=args.epoch,
            expected_candidate=candidate,
            val_inputs_artifact=preflight["val_inputs_receipt"],
            pipeline_artifact=preflight["pipeline_receipt"],
            expected_coverage=expected_coverage,
        )
        return lineage_artifact


def _add_split(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("val",), required=True)


def _add_preflight_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", allow_abbrev=False)
    _add_split(prepare_parser)
    prepare_parser.add_argument("--candidate-manifest", type=Path, required=True)
    prepare_parser.add_argument(
        "--expected-candidate-manifest-sha256",
        required=True,
    )
    prepare_parser.add_argument("--candidate-status", type=Path, required=True)
    prepare_parser.add_argument(
        "--expected-candidate-status-sha256",
        required=True,
    )
    prepare_parser.add_argument("--frozen-inputs", type=Path, required=True)
    prepare_parser.add_argument(
        "--expected-frozen-inputs-sha256",
        required=True,
    )
    prepare_parser.add_argument("--val-inputs", type=Path, required=True)
    prepare_parser.add_argument("--expected-val-inputs-sha256", required=True)
    prepare_parser.add_argument("--pipeline", type=Path, required=True)
    prepare_parser.add_argument("--expected-pipeline-sha256", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)

    shard_parser = commands.add_parser("shard", allow_abbrev=False)
    _add_split(shard_parser)
    _add_preflight_inputs(shard_parser)
    shard_parser.add_argument("--epoch", type=_epoch, required=True)
    shard_parser.add_argument("--output-root", type=Path, required=True)
    shard_parser.add_argument("--num-shards", type=_num_shards, required=True)
    shard_parser.add_argument("--shard-id", type=int, required=True)
    shard_parser.add_argument("--device", required=True)
    shard_parser.add_argument("--seed", type=int, default=20260731)
    shard_parser.add_argument("--progress-every", type=int, default=20)

    finalize_parser = commands.add_parser("finalize", allow_abbrev=False)
    _add_split(finalize_parser)
    _add_preflight_inputs(finalize_parser)
    finalize_parser.add_argument("--epoch", type=_epoch, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--num-shards", type=_num_shards, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for value in vars(args).values():
        _reject_forbidden(value, "command argument")
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "shard":
        result = run_shard(args)
    elif args.command == "finalize":
        result = finalize(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
