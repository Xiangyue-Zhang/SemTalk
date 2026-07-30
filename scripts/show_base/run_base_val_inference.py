#!/usr/bin/env python3
"""Produce frozen SemTalk Base predictions for the SHOW validation split.

This entry point is intentionally separate from ``run_base_inference.py``.
The latter is a test-only eight-shard program; this program accepts only an
explicit ``--split val`` and never exposes a test input.

The transaction has three small, explicit phases:

* ``prepare`` validates the complete seven-candidate bundle, the exact 1,715
  clip validation cache, and the frozen downstream pipeline once.
* ``shard`` runs one deterministic modulo shard.  ``--num-shards`` is
  configurable (including 16 shards across two eight-GPU workers).
* ``finalize`` is CPU-only and single-writer.  It validates exact-once shard
  closure, independently copies every result into an unpublished generation,
  writes the selector-compatible lineage, and publishes the whole generation
  with one directory rename.

Only Base epochs 1/2/4/8/16/32/40 are accepted.  Withdrawn e30/epoch-30 and
Speaker2 labels, as well as every test-labelled path, fail before model load.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
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

from scripts.show_base import select_base_official_adapt as selector  # noqa: E402


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


def _sha256_file(path: Path) -> str:
    return selector.sha256_file(path)


def _artifact(path: Path, *, payload_sha: str | None = None) -> dict[str, str]:
    result = {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path),
    }
    if payload_sha is not None:
        result["receipt_payload_sha256"] = payload_sha
    return result


def _output_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
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
    expected = selector.require_sha256(expected_sha, f"{label} SHA")
    resolved = _regular_file(path, label)
    payload = resolved.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected:
        raise ValInferenceContractError(
            f"{label} SHA mismatch: {observed} != {expected}"
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
    source = _regular_file(source, "shard output")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as input_handle, destination.open("xb") as output:
            for block in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                copied += len(block)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    observed = digest.hexdigest()
    if copied != expected_bytes or observed != expected_sha:
        destination.unlink(missing_ok=True)
        raise ValInferenceContractError(
            f"copied shard output changed: {source}"
        )
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (
        source_stat.st_dev == destination_stat.st_dev
        and source_stat.st_ino == destination_stat.st_ino
    ):
        destination.unlink(missing_ok=True)
        raise ValInferenceContractError("final output must not hardlink a shard")
    return {
        "path": str(destination.resolve(strict=True)),
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
            "epoch must be one of 1,2,4,8,16,32,40; e30 is withdrawn"
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
        or payload["pipeline_source"] != selector.VAL_INFERENCE_SOURCE
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
    bundle = selector.validate_candidate_bundle(
        manifest_path=args.candidate_manifest,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
        status_path=args.candidate_status,
        expected_status_sha256=args.expected_candidate_status_sha256,
        frozen_inputs_path=args.frozen_inputs,
        expected_frozen_inputs_sha256=args.expected_frozen_inputs_sha256,
    )
    val_artifact, coverage = selector.validate_val_inputs(
        args.val_inputs,
        args.expected_val_inputs_sha256,
    )
    pipeline_artifact, pipeline = selector.validate_pipeline(
        args.pipeline,
        args.expected_pipeline_sha256,
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
    entrypoint = pipeline["inference_entrypoint"]
    path = _regular_file(
        Path(entrypoint["path"]),
        "pinned validation inference helper",
    )
    _verified_bytes(
        path,
        entrypoint["sha256"],
        "pinned validation inference helper",
    )
    if entrypoint["sha256"] != selector.VAL_INFERENCE_SOURCE[
        "entrypoint_sha256"
    ]:
        raise ValInferenceContractError("unpinned inference helper")
    name = f"_semtalk_val_helper_{entrypoint['sha256']}"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ValInferenceContractError(f"cannot import helper {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    required = set(selector.INFERENCE_HELPERS) | {
        "_validate_official_transfer_checkpoint",
        "_load_released_model_state_only",
        "_validate_released_model_state_schema",
        "_strict_load_freeze_eval",
        "_normalize_data_parallel_state",
        "_read_verified_checkpoint_snapshot",
        "_torch_load_checkpoint",
        "_finite_state_dict",
        "_validate_base_model_state_schema",
        "_expected_released_representation_schemas",
        "_model_args",
        "_joint_masks",
        "deterministic_npz_bytes",
        "RELEASED_ALL_SPEAKERS_MODELS",
        "OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise ValInferenceContractError(
            f"pinned inference helper lacks {missing}"
        )
    return module


@contextmanager
def _pinned_meta_schema_cuda_compat() -> Iterable[None]:
    """Keep the pinned helper's legacy ``.cuda()`` call on the meta device."""

    import torch

    original_cuda = torch.Tensor.cuda

    def meta_only_cuda(
        tensor: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if (
            args
            or kwargs
            or getattr(getattr(tensor, "device", None), "type", None)
            != "meta"
        ):
            raise ValInferenceContractError(
                "pinned helper schema construction attempted a non-meta "
                "or parameterized Tensor.cuda call"
            )
        return tensor

    torch.Tensor.cuda = meta_only_cuda
    try:
        yield
    finally:
        torch.Tensor.cuda = original_cuda


def _prime_pinned_released_schema_cache(helper: ModuleType) -> None:
    with _pinned_meta_schema_cuda_compat():
        schemas = helper._expected_released_representation_schemas()
    expected_stages = {"face", "global", "hands", "upper", "lower"}
    if (
        not isinstance(schemas, dict)
        or set(schemas) != expected_stages
        or any(
            not isinstance(schemas[stage], dict) or not schemas[stage]
            for stage in expected_stages
        )
    ):
        raise ValInferenceContractError(
            "pinned helper returned an invalid released schema cache"
        )


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
    pipeline = _verified_json(
        Path(pipeline_artifact["path"]),
        pipeline_artifact["sha256"],
        "frozen validation pipeline",
    )
    if (
        val_inputs.get("split") != "val"
        or val_inputs.get("test_visible") is not False
        or pipeline.get("split") != "val"
        or pipeline.get("test_visible") is not False
        or pipeline.get("source") != preflight["pipeline_source"]
        or pipeline.get("inference_entrypoint")
        != preflight["inference_entrypoint"]
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


def _transfer_payload_expectations(
    helper: ModuleType,
    checkpoint: Path,
    _expected_sha: str,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    resolved = _regular_file(
        checkpoint,
        f"official-adapt {stage} checkpoint",
    )
    summary = _regular_file(
        resolved.parent / "summary.json",
        f"{stage} transfer summary",
    )
    summary_payload = _strict_json_bytes(
        summary.read_bytes(),
        f"{stage} transfer summary",
    )
    if not isinstance(summary_payload, dict):
        raise ValInferenceContractError(
            f"{stage} transfer summary must be an object"
        )
    _reject_absolute_paths_in_tree(
        summary_payload,
        f"{stage} transfer summary",
    )
    source = summary_payload.get("source_receipt")
    cache = summary_payload.get("cache_receipt")
    if not isinstance(source, dict) or not isinstance(cache, dict):
        raise ValInferenceContractError(
            f"{stage} transfer source/cache receipt is missing"
        )
    canonical = cache.get("canonical_receipt")
    if not isinstance(canonical, dict):
        raise ValInferenceContractError(
            f"{stage} transfer canonical receipt is missing"
        )
    return source, canonical, summary, _sha256_file(summary)


def _load_models(
    helper: ModuleType,
    *,
    epoch: int,
    preflight: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from models.motion_representation import VAEConvZero
    from models.rvq import RVQVAE
    from models.semtalk import semtalk_base

    _prime_pinned_released_schema_cache(helper)
    candidate = preflight["candidate_bundle"]["candidates"][str(epoch)]
    bundle = preflight["candidate_bundle"]
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
        != epoch * selector.EXPECTED_UPDATES_PER_EPOCH
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
    base = semtalk_base(helper._model_args()).to(device)
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
            "candidate_epoch": epoch,
            "frozen_receipt_sha256": audit["frozen_receipt_sha256"],
        }
    }
    fixed = pipeline["fixed_checkpoints"]

    for stage in ("face", "global"):
        checkpoint = Path(fixed[stage]["path"])
        source, canonical, summary, summary_sha = (
            _transfer_payload_expectations(
                helper,
                checkpoint,
                fixed[stage]["sha256"],
                stage,
            )
        )
        payload, receipt = helper._validate_official_transfer_checkpoint(
            path=checkpoint,
            formal_stage=stage,
            expected_sha256=fixed[stage]["sha256"],
            status_path=summary,
            expected_status_sha256=summary_sha,
            expected_source_receipt=source,
            expected_canonical_receipt=canonical,
        )
        if stage == "face":
            specification = helper.RELEASED_ALL_SPEAKERS_MODELS["face"]
            model = RVQVAE(
                SimpleNamespace(
                    vae_test_dim=specification["vae_test_dim"],
                    vae_layer=specification["vae_layer"],
                    vae_length=256,
                )
            ).to(device)
        else:
            model = VAEConvZero(
                SimpleNamespace(
                    vae_test_dim=61,
                    vae_layer=4,
                    vae_length=256,
                )
            ).to(device)
        helper._strict_load_freeze_eval(
            model,
            helper._normalize_data_parallel_state(
                payload["model_state"],
                checkpoint,
            ),
            path=checkpoint,
        )
        models[stage] = model
        receipts[stage] = {
            "path": receipt["path"],
            "sha256": receipt["sha256"],
            "summary": {
                "path": receipt["formal_training_status"],
                "sha256": receipt["formal_training_status_sha256"],
            },
        }

    for stage in ("hands", "upper", "lower"):
        entry = fixed[stage]
        specification = helper.RELEASED_ALL_SPEAKERS_MODELS[stage]
        state, resolved, snapshot, observed = (
            helper._load_released_model_state_only(
                Path(entry["path"]),
                expected_filename=specification["filename"],
                expected_sha256=entry["sha256"],
            )
        )
        helper._validate_released_model_state_schema(
            state,
            formal_stage=stage,
            path=resolved,
        )
        model = RVQVAE(
            SimpleNamespace(
                vae_test_dim=specification["vae_test_dim"],
                vae_layer=specification["vae_layer"],
                vae_length=256,
            )
        ).to(device)
        helper._strict_load_freeze_eval(model, state, path=resolved)
        models[stage] = model
        receipts[stage] = {
            "path": str(resolved),
            "sha256": observed,
            "bytes": len(snapshot),
            "source": "released_all_speakers_v1",
        }

    for model in models.values():
        model.eval()
        model.requires_grad_(False)
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
        models, model_receipts = _load_models(
            helper,
            epoch=args.epoch,
            preflight=preflight,
            pipeline=pipeline,
            device=args.device,
        )
        masks = helper._joint_masks(__import__("torch").device(args.device))
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
            prediction = helper._infer_clip(
                pose=canonical["pose"],
                trans=canonical["trans"],
                beat=audio["beat"],
                hubert=audio["hubert"],
                speaker_id=helper.SHOW_SPEAKER_IDS[canonical_row["speaker"]],
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
    canonical_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = output_root / SHARDS_DIRECTORY / _shard_name(shard_id, num_shards)
    root = _directory(root, f"shard {shard_id}")
    receipt_path = _regular_file(
        root / SHARD_RECEIPT_FILENAME,
        f"shard {shard_id} receipt",
    )
    receipt = _strict_json_bytes(
        receipt_path.read_bytes(),
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
    _val_inputs, _pipeline, canonical_rows, _audio = (
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
            clip_stage = stage / "diffsheg_eval_clip_ids.txt"
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
                    "diffsheg_clip_manifest_sha256": coverage[
                        "diffsheg_clip_manifest_sha256"
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
