#!/usr/bin/env python3
"""Fail closed over the sole combined SemTalk SHOW final-test result.

The metric producer returns an external SHA/byte/canonical-payload pin for
``final_metrics.json``.  This independent terminal validator freshly replays
the full final-test authority and frozen preflight, then binds those external
pins to the exclusive one-shot claim, the exact PASPA report, exactly seven
finite DiffSHEG metrics, and the TalkSHOW released2/paper16 body plus released
face report.  Both reports must bind the same finalized predictions and the
same authority; neither is valid as a separate test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import evaluate_diffsheg_final_test as producer
from scripts.show_base import run_base_final_test as final_test


FORMAT = "semtalk_show_combined_final_result_validation_v1"


class FinalResultValidationError(RuntimeError):
    """Raised when the externally pinned final result is not authoritative."""


def _sha256(value: Any, label: str) -> str:
    try:
        return producer.require_sha256(value, label)
    except producer.FinalDiffSHEGError as error:
        raise FinalResultValidationError(str(error)) from error


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise FinalResultValidationError(f"{label} must be a positive integer")
    return value


def _safe_snapshot(
    value: Any,
    label: str,
    *,
    expected_sha256: Optional[Any] = None,
    expected_bytes: Optional[Any] = None,
    exclusive_claim: bool = False,
) -> tuple[Path, bytes, str, os.stat_result]:
    if not isinstance(value, (str, os.PathLike)):
        raise FinalResultValidationError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FinalResultValidationError(f"{label} must be absolute")
    try:
        before = os.lstat(path)
    except OSError as error:
        raise FinalResultValidationError(
            f"cannot inspect {label}: {path}"
        ) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FinalResultValidationError(
            f"{label} must be a regular non-symlink file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise FinalResultValidationError(f"{label} path is not canonical")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(resolved)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    identities = [
        tuple(int(getattr(item, field)) for field in fields)
        for item in (before, opened_before, opened_after, after)
    ]
    if len(set(identities)) != 1:
        raise FinalResultValidationError(f"{label} changed while read")
    if exclusive_claim and (
        stat.S_IMODE(after.st_mode) != 0o600 or after.st_nlink != 1
    ):
        raise FinalResultValidationError(
            "one-shot claim lacks exclusive 0600/single-link publication"
        )
    payload = b"".join(chunks)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha256 != _sha256(
        expected_sha256, f"{label} expected SHA-256"
    ):
        raise FinalResultValidationError(f"{label} SHA-256 mismatch")
    if expected_bytes is not None and len(payload) != _positive_int(
        expected_bytes, f"{label} expected bytes"
    ):
        raise FinalResultValidationError(f"{label} byte count mismatch")
    if not payload:
        raise FinalResultValidationError(f"{label} is empty")
    return resolved, payload, observed_sha256, after


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = producer._strict_json_bytes(payload, label)
    except producer.FinalDiffSHEGError as error:
        raise FinalResultValidationError(str(error)) from error
    if type(value) is not dict:
        raise FinalResultValidationError(f"{label} must be a JSON object")
    return value


def _validate_preflight(
    args: argparse.Namespace,
    current: Mapping[str, Any],
) -> tuple[Path, bytes, str, os.stat_result]:
    path, payload, observed_sha256, metadata = _safe_snapshot(
        args.preflight_json,
        "formal DiffSHEG preflight receipt",
        expected_sha256=args.expected_preflight_sha256,
    )
    stored = _strict_json(payload, "formal DiffSHEG preflight receipt")
    if stored != current:
        raise FinalResultValidationError(
            "fresh authority/preflight replay differs from frozen preflight"
        )
    if (
        stored.get("format") != producer.PREFLIGHT_FORMAT
        or stored.get("status") != "formal_ready"
        or producer.canonical_json_sha256(
            {
                key: value
                for key, value in stored.items()
                if key != "receipt_payload_sha256"
            }
        )
        != stored.get("receipt_payload_sha256")
    ):
        raise FinalResultValidationError("frozen preflight payload is invalid")
    return path, payload, observed_sha256, metadata


def _recheck_snapshot(
    path: Path,
    payload: bytes,
    observed_sha256: str,
    metadata: os.stat_result,
    label: str,
    *,
    exclusive_claim: bool = False,
) -> None:
    again_path, again_payload, again_sha256, again_metadata = _safe_snapshot(
        path,
        label,
        expected_sha256=observed_sha256,
        expected_bytes=len(payload),
        exclusive_claim=exclusive_claim,
    )
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        again_path != path
        or again_payload != payload
        or again_sha256 != observed_sha256
        or tuple(int(getattr(again_metadata, field)) for field in identity_fields)
        != tuple(int(getattr(metadata, field)) for field in identity_fields)
    ):
        raise FinalResultValidationError(f"{label} changed before completion")


def _validate_diffsheg_metrics(value: Any, label: str) -> dict[str, float]:
    if type(value) is not dict or set(value) != set(producer.EXPECTED_METRICS):
        raise FinalResultValidationError(
            f"{label} must contain exactly the seven DiffSHEG metrics"
        )
    metrics: dict[str, float] = {}
    for key in producer.EXPECTED_METRICS:
        item = value[key]
        if isinstance(item, bool) or type(item) not in {int, float}:
            raise FinalResultValidationError(f"{label}.{key} is not numeric")
        converted = float(item)
        if not math.isfinite(converted):
            raise FinalResultValidationError(f"{label}.{key} is not finite")
        metrics[key] = converted
    return metrics


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalResultValidationError(
            f"cannot verify terminal validator source closure: {error}"
        ) from error


def _validate_validator_source_closure(
    current_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    authority = current_preflight.get("authority")
    source = authority.get("source") if type(authority) is dict else None
    if type(source) is not dict:
        raise FinalResultValidationError(
            "fresh authority lacks the validator source closure"
        )
    root = Path(str(source.get("source_root", "")))
    if (
        not root.is_absolute()
        or root.resolve() != PROJECT_ROOT.resolve()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise FinalResultValidationError("validator source root changed")
    commit = producer.require_git_oid(
        source.get("commit"), "validator source commit"
    )
    tree = producer.require_git_oid(source.get("tree"), "validator source tree")
    head = _git(root, "rev-parse", "HEAD^{commit}").stdout.decode().strip()
    live_tree = _git(root, "rev-parse", "HEAD^{tree}").stdout.decode().strip()
    origin = _git(root, "remote", "get-url", "origin").stdout.decode().strip()
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    symbolic = _git(root, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    branches = _git(
        root, "for-each-ref", "--format=%(refname)", "refs/heads"
    ).stdout.splitlines()
    if (
        head != commit
        or live_tree != tree
        or origin != source.get("origin")
        or status
        or symbolic.returncode == 0
        or symbolic.stdout.strip()
        or branches
        or source.get("clean") is not True
        or source.get("detached") is not True
        or source.get("local_branches_at_commit") != []
    ):
        raise FinalResultValidationError(
            "terminal validator is not in the clean detached authorized tree"
        )
    validator_path, validator_payload, validator_sha256, _metadata = (
        _safe_snapshot(Path(__file__).resolve(), "terminal validator source")
    )
    relative = validator_path.relative_to(root).as_posix()
    entry = _git(root, "ls-tree", "-z", tree, "--", relative).stdout
    if not entry.endswith(b"\0") or entry.count(b"\0") != 1:
        raise FinalResultValidationError(
            "terminal validator has no unique source-tree blob"
        )
    metadata, separator, observed_path = entry[:-1].partition(b"\t")
    fields = metadata.decode("ascii", errors="strict").split()
    if (
        not separator
        or observed_path.decode("utf-8", errors="strict") != relative
        or len(fields) != 3
        or fields[0] not in {"100644", "100755"}
        or fields[1] != "blob"
        or len(fields[2]) != 40
    ):
        raise FinalResultValidationError(
            "terminal validator tree entry changed"
        )
    blob_oid = _git(
        root, "hash-object", "--no-filters", "--", str(validator_path)
    ).stdout.decode().strip()
    if blob_oid != fields[2]:
        raise FinalResultValidationError(
            "terminal validator bytes differ from the authorized Git blob"
        )
    return {
        "source_root": str(root),
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "path": str(validator_path),
        "relative_path": relative,
        "git_blob_oid": blob_oid,
        "sha256": validator_sha256,
        "bytes": len(validator_payload),
        "clean": True,
        "detached": True,
        "local_branches": [],
    }


def validate_final_result(
    args: argparse.Namespace,
    current_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    validator_source = _validate_validator_source_closure(current_preflight)
    (
        preflight_path,
        preflight_payload,
        preflight_sha256,
        preflight_metadata,
    ) = _validate_preflight(args, current_preflight)
    final_path, final_payload, final_sha256, final_metadata = _safe_snapshot(
        args.final_metrics_json,
        "externally pinned final_metrics.json",
        expected_sha256=args.expected_final_metrics_sha256,
        expected_bytes=args.expected_final_metrics_bytes,
    )
    if final_path.name != "final_metrics.json":
        raise FinalResultValidationError("final metric filename changed")
    formal_output_root = final_path.parent
    if formal_output_root.is_symlink() or not formal_output_root.is_dir():
        raise FinalResultValidationError("formal metric output root is unsafe")
    final_result = _strict_json(final_payload, "final_metrics.json")
    expected_payload_sha256 = _sha256(
        args.expected_final_metrics_canonical_payload_sha256,
        "external final metric canonical payload SHA-256",
    )
    canonical_payload_sha256 = producer.canonical_json_sha256(
        {
            key: value
            for key, value in final_result.items()
            if key != "receipt_payload_sha256"
        }
    )
    if (
        final_result.get("receipt_payload_sha256")
        != expected_payload_sha256
        or canonical_payload_sha256 != expected_payload_sha256
    ):
        raise FinalResultValidationError(
            "final metric canonical payload SHA-256 mismatch"
        )
    expected_result_keys = {
        "format",
        "status",
        "authority",
        "one_shot_claim",
        "preflight",
        "paspa_report",
        "talkshow_suite_start",
        "talkshow_report",
        "evaluator_command_sha256",
        "talkshow_evaluator_call_sha256",
        "protocol",
        "assets",
        "runtime",
        "coverage",
        "metrics",
        "all_metrics_finite",
        "test_evaluations",
        "validation_only_for_selection",
        "test_feedback_into_selection",
        "final_metric_event",
        "receipt_payload_sha256",
    }
    if (
        set(final_result) != expected_result_keys
        or final_result.get("format") != producer.RESULT_FORMAT
        or final_result.get("status") != "complete"
        or final_result.get("authority") != current_preflight.get("authority")
        or final_result.get("protocol") != current_preflight.get("protocol")
        or final_result.get("assets") != current_preflight.get("assets")
        or final_result.get("runtime") != current_preflight.get("runtime")
        or final_result.get("all_metrics_finite") is not True
        or final_result.get("test_evaluations") != 1
        or final_result.get("validation_only_for_selection") is not True
        or final_result.get("test_feedback_into_selection") is not False
        or final_result.get("final_metric_event")
        != producer.final_test.final_authority.FINAL_METRIC_EVENT
        or current_preflight["protocol"].get("final_metric_event")
        != producer.final_test.final_authority.FINAL_METRIC_EVENT
    ):
        raise FinalResultValidationError("final metric authority/schema changed")
    _sha256(
        final_result.get("evaluator_command_sha256"),
        "evaluator command SHA-256",
    )
    _sha256(
        final_result.get("talkshow_evaluator_call_sha256"),
        "TalkSHOW evaluator call SHA-256",
    )
    expected_preflight_receipt = {
        "path": str(preflight_path),
        "sha256": preflight_sha256,
        "receipt_payload_sha256": current_preflight[
            "receipt_payload_sha256"
        ],
    }
    if final_result.get("preflight") != expected_preflight_receipt:
        raise FinalResultValidationError("final metric preflight receipt changed")

    inference = current_preflight.get("inference")
    if type(inference) is not dict:
        raise FinalResultValidationError("fresh inference receipt is missing")
    expected_coverage = {
        "clip_count": producer.EXPECTED_TEST_CLIPS,
        "exact_once": True,
        "num_generation_shards": producer.EXPECTED_NUM_SHARDS,
        "window_length": producer.WINDOW_LENGTH,
        "window_stride": producer.WINDOW_STRIDE,
        "window_count": inference.get("window_count"),
        "generation_passes": 1,
        "shared_prediction_bundle": True,
    }
    if final_result.get("coverage") != expected_coverage:
        raise FinalResultValidationError("final metric coverage changed")

    inference_root = Path(str(inference.get("root", "")))
    if not inference_root.is_absolute() or inference_root.resolve() != inference_root:
        raise FinalResultValidationError("fresh inference root is invalid")
    claim_path = producer._claim_path(inference_root)
    claim_path, claim_payload, claim_sha256, claim_metadata = _safe_snapshot(
        claim_path,
        "exclusive combined one-shot claim",
        exclusive_claim=True,
    )
    claim = _strict_json(claim_payload, "exclusive combined one-shot claim")
    expected_claim = {
        "format": producer.CLAIM_FORMAT,
        "status": "claimed",
        "authority": current_preflight["authority"],
        "preflight": expected_preflight_receipt,
        "output_root": str(formal_output_root.resolve()),
        "test_evaluations": 1,
        "test_feedback_into_selection": False,
        "final_metric_event": producer.final_test.final_authority.FINAL_METRIC_EVENT,
        "shared_predictions": {
            "manifest": current_preflight["assets"]["talkshow_metrics"][
                "prediction_manifest"
            ],
            "lineage": current_preflight["assets"]["talkshow_metrics"][
                "prediction_lineage"
            ],
        },
        "claim_consumed_before_metrics": True,
        "failure_consumes_claim": True,
        "retry_allowed": False,
        "input_set_sha256": inference.get("input_set_sha256"),
    }
    if claim != expected_claim:
        raise FinalResultValidationError(
            "exclusive one-shot claim is stale or malformed"
        )
    if final_result.get("one_shot_claim") != {
        "path": str(claim_path),
        "sha256": claim_sha256,
    }:
        raise FinalResultValidationError("final metric claim receipt changed")

    paspa_receipt = final_result.get("paspa_report")
    if type(paspa_receipt) is not dict or set(paspa_receipt) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise FinalResultValidationError("PASPA report receipt schema changed")
    expected_paspa_path = formal_output_root / "paspa_diffsheg_show_metrics.json"
    if Path(str(paspa_receipt.get("path", ""))) != expected_paspa_path:
        raise FinalResultValidationError("PASPA report path changed")
    frozen_device = current_preflight["assets"]["talkshow_metrics"].get(
        "device"
    )
    if frozen_device != "cuda:0":
        raise FinalResultValidationError(
            "combined metric device is not the frozen cuda:0 device"
        )
    expected_paspa_command = producer._paspa_command(
        argparse.Namespace(device=frozen_device, batch_size=args.batch_size),
        current_preflight,
        expected_paspa_path,
    )
    expected_paspa_command_sha256 = hashlib.sha256(
        b"\0".join(
            argument.encode("utf-8") for argument in expected_paspa_command
        )
    ).hexdigest()
    if (
        final_result["evaluator_command_sha256"]
        != expected_paspa_command_sha256
    ):
        raise FinalResultValidationError(
            "PASPA evaluator command receipt changed"
        )
    paspa_path, paspa_payload, paspa_sha256, paspa_metadata = _safe_snapshot(
        expected_paspa_path,
        "pinned PASPA DiffSHEG report",
        expected_sha256=paspa_receipt["sha256"],
        expected_bytes=paspa_receipt["bytes"],
    )
    paspa_report = _strict_json(paspa_payload, "pinned PASPA DiffSHEG report")
    try:
        paspa_metrics = producer._validate_paspa_report(
            paspa_report, current_preflight
        )
    except producer.FinalDiffSHEGError as error:
        raise FinalResultValidationError(str(error)) from error
    metric_bundle = final_result.get("metrics")
    if type(metric_bundle) is not dict or set(metric_bundle) != {
        "diffsheg",
        "talkshow",
    }:
        raise FinalResultValidationError(
            "combined metric bundle must contain both required suites"
        )
    final_diffsheg_metrics = _validate_diffsheg_metrics(
        metric_bundle["diffsheg"],
        "metrics.diffsheg",
    )
    if final_diffsheg_metrics != paspa_metrics:
        raise FinalResultValidationError(
            "final DiffSHEG metrics differ from the pinned PASPA report"
        )

    suite_start_receipt = final_result.get("talkshow_suite_start")
    if type(suite_start_receipt) is not dict or set(suite_start_receipt) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise FinalResultValidationError(
            "TalkSHOW suite-start receipt schema changed"
        )
    expected_suite_start_path = (
        formal_output_root
        / producer.talkshow_metrics.COMBINED_TALKSHOW_START_NAME
    )
    if Path(str(suite_start_receipt.get("path", ""))) != (
        expected_suite_start_path
    ):
        raise FinalResultValidationError(
            "TalkSHOW suite-start marker path changed"
        )
    (
        suite_start_path,
        suite_start_payload,
        suite_start_sha256,
        suite_start_metadata,
    ) = _safe_snapshot(
        expected_suite_start_path,
        "exclusive TalkSHOW suite-start marker",
        expected_sha256=suite_start_receipt["sha256"],
        expected_bytes=suite_start_receipt["bytes"],
        exclusive_claim=True,
    )
    suite_start = _strict_json(
        suite_start_payload,
        "exclusive TalkSHOW suite-start marker",
    )
    claim_artifact = {
        "path": str(claim_path),
        "sha256": claim_sha256,
        "bytes": len(claim_payload),
    }
    paspa_artifact = {
        "path": str(paspa_path),
        "sha256": paspa_sha256,
        "bytes": len(paspa_payload),
    }
    talkshow_suite = current_preflight["assets"]["talkshow_metrics"]
    expected_suite_start = {
        "format": producer.talkshow_metrics.COMBINED_TALKSHOW_START_FORMAT,
        "status": "started",
        "combined_claim": claim_artifact,
        "paspa_report": paspa_artifact,
        "test_authority": talkshow_suite["test_authority"],
        "prediction_manifest": {
            "path": talkshow_suite["prediction_manifest"]["path"],
            "sha256": talkshow_suite["prediction_manifest"]["sha256"],
        },
        "prediction_lineage": {
            "path": talkshow_suite["prediction_lineage"]["path"],
            "sha256": talkshow_suite["prediction_lineage"]["sha256"],
        },
        "final_metric_event": (
            producer.final_test.final_authority.FINAL_METRIC_EVENT
        ),
        "created_before_talkshow_metrics": True,
        "failure_consumes_event": True,
        "retry_allowed": False,
    }
    if suite_start != expected_suite_start:
        raise FinalResultValidationError(
            "TalkSHOW suite-start marker binding changed"
        )
    combined_event = {
        "claim": claim_artifact,
        "paspa_report": paspa_artifact,
        "suite_start": {
            "path": str(suite_start_path),
            "sha256": suite_start_sha256,
            "bytes": len(suite_start_payload),
        },
    }

    talkshow_receipt = final_result.get("talkshow_report")
    if type(talkshow_receipt) is not dict or set(talkshow_receipt) != {
        "path",
        "sha256",
        "bytes",
        "report_payload_sha256",
    }:
        raise FinalResultValidationError(
            "TalkSHOW report receipt schema changed"
        )
    expected_talkshow_path = (
        formal_output_root / "talkshow_show_body_face_metrics.json"
    )
    if Path(str(talkshow_receipt.get("path", ""))) != expected_talkshow_path:
        raise FinalResultValidationError("TalkSHOW report path changed")
    (
        talkshow_path,
        talkshow_payload,
        talkshow_sha256,
        talkshow_metadata,
    ) = _safe_snapshot(
        expected_talkshow_path,
        "pinned TalkSHOW body/face report",
        expected_sha256=talkshow_receipt["sha256"],
        expected_bytes=talkshow_receipt["bytes"],
    )
    talkshow_report = _strict_json(
        talkshow_payload,
        "pinned TalkSHOW body/face report",
    )
    expected_talkshow_call_sha256 = producer.canonical_json_sha256(
        producer._talkshow_call_receipt(
            current_preflight,
            frozen_device,
            combined_event,
        )
    )
    if (
        final_result["talkshow_evaluator_call_sha256"]
        != expected_talkshow_call_sha256
    ):
        raise FinalResultValidationError(
            "TalkSHOW evaluator call receipt changed"
        )
    expected_prediction_manifest = {
        key: talkshow_suite["prediction_manifest"][key]
        for key in ("path", "sha256", "bytes")
    }
    expected_selection_protocol = {
        "primary_metric": producer.talkshow_metrics.PRIMARY_METRIC_PATH,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 1,
    }
    try:
        producer.talkshow_metrics.validate_report(
            talkshow_report,
            expected_split="test",
            expected_clip_count=producer.EXPECTED_TEST_CLIPS,
            expected_prediction_manifest=expected_prediction_manifest,
            expected_distribution_receipt=talkshow_suite[
                "distribution_receipt"
            ],
            expected_selection_protocol=expected_selection_protocol,
            expected_test_authority=talkshow_suite["test_authority"],
            expected_combined_event=combined_event,
        )
    except producer.talkshow_metrics.MetricAdapterContractError as error:
        raise FinalResultValidationError(
            f"TalkSHOW report fresh validation failed: {error}"
        ) from error
    if (
        talkshow_receipt["report_payload_sha256"]
        != talkshow_report.get("report_payload_sha256")
        or metric_bundle["talkshow"]
        != {
            "body": talkshow_report.get("body"),
            "face": talkshow_report.get("face"),
            "rs": talkshow_report.get("rs"),
        }
        or talkshow_report.get("inputs", {}).get("prediction_manifest", {}).get(
            "sha256"
        )
        != inference["manifest"]["sha256"]
        or talkshow_report.get("inputs", {}).get("prediction_lineage", {}).get(
            "sha256"
        )
        != inference["lineage"]["sha256"]
    ):
        raise FinalResultValidationError(
            "TalkSHOW report does not share the claimed prediction bundle"
        )

    _recheck_snapshot(
        preflight_path,
        preflight_payload,
        preflight_sha256,
        preflight_metadata,
        "formal DiffSHEG preflight receipt",
    )
    _recheck_snapshot(
        claim_path,
        claim_payload,
        claim_sha256,
        claim_metadata,
        "exclusive combined one-shot claim",
        exclusive_claim=True,
    )
    _recheck_snapshot(
        paspa_path,
        paspa_payload,
        paspa_sha256,
        paspa_metadata,
        "pinned PASPA DiffSHEG report",
    )
    _recheck_snapshot(
        suite_start_path,
        suite_start_payload,
        suite_start_sha256,
        suite_start_metadata,
        "exclusive TalkSHOW suite-start marker",
        exclusive_claim=True,
    )
    _recheck_snapshot(
        talkshow_path,
        talkshow_payload,
        talkshow_sha256,
        talkshow_metadata,
        "pinned TalkSHOW body/face report",
    )
    _recheck_snapshot(
        final_path,
        final_payload,
        final_sha256,
        final_metadata,
        "externally pinned final_metrics.json",
    )
    if _validate_validator_source_closure(current_preflight) != validator_source:
        raise FinalResultValidationError(
            "terminal validator source closure changed before completion"
        )

    validation: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "fresh_authority": current_preflight["authority"],
        "validator_source": validator_source,
        "preflight": expected_preflight_receipt,
        "one_shot_claim": {
            "path": str(claim_path),
            "sha256": claim_sha256,
            "bytes": len(claim_payload),
            "exclusive_0600_single_link": True,
        },
        "paspa_report": {
            "path": str(paspa_path),
            "sha256": paspa_sha256,
            "bytes": len(paspa_payload),
        },
        "talkshow_suite_start": {
            "path": str(suite_start_path),
            "sha256": suite_start_sha256,
            "bytes": len(suite_start_payload),
            "exclusive_0600_single_link": True,
        },
        "talkshow_report": {
            "path": str(talkshow_path),
            "sha256": talkshow_sha256,
            "bytes": len(talkshow_payload),
            "report_payload_sha256": talkshow_report[
                "report_payload_sha256"
            ],
        },
        "final_metrics": {
            "path": str(final_path),
            "sha256": final_sha256,
            "bytes": len(final_payload),
            "canonical_payload_sha256": canonical_payload_sha256,
        },
        "metrics": metric_bundle,
        "metric_suites": [
            "paspa_diffsheg_show_seven",
            "talkshow_show_body_face",
        ],
        "diffsheg_metric_names": list(producer.EXPECTED_METRICS),
        "diffsheg_metric_count": len(producer.EXPECTED_METRICS),
        "all_metrics_finite": True,
        "test_evaluations": 1,
        "test_feedback_into_selection": False,
        "shared_prediction_bundle": expected_prediction_manifest,
    }
    validation["receipt_payload_sha256"] = producer.canonical_json_sha256(
        validation
    )
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    final_test._authority_options(parser)
    parser.add_argument("--inference-final-root", type=Path, required=True)
    parser.add_argument("--paspa-root", type=Path, required=True)
    parser.add_argument("--diffsheg-root", type=Path, required=True)
    parser.add_argument("--talkshow-root", type=Path, required=True)
    parser.add_argument("--source-audio-root", type=Path, required=True)
    parser.add_argument("--smplx-path", type=Path, required=True)
    parser.add_argument("--talkshow-metric-root", type=Path, required=True)
    parser.add_argument(
        "--talkshow-feature-extractor", type=Path, required=True
    )
    parser.add_argument(
        "--talkshow-validation-gate-json", type=Path, required=True
    )
    parser.add_argument(
        "--expected-talkshow-validation-gate-sha256", required=True
    )
    parser.add_argument(
        "--expected-talkshow-validation-gate-receipt-payload-sha256",
        required=True,
    )
    parser.add_argument("--talkshow-torch-threads", type=int, default=1)
    parser.add_argument("--expected-audio-set-sha256", required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--final-metrics-json", type=Path, required=True)
    parser.add_argument("--expected-final-metrics-sha256", required=True)
    parser.add_argument("--expected-final-metrics-bytes", type=int, required=True)
    parser.add_argument(
        "--expected-final-metrics-canonical-payload-sha256", required=True
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    # The producer preflight schema includes the formal device.  Terminal
    # replay is CPU-only but must reconstruct that immutable CUDA identity;
    # it is deliberately a non-overridable parser default.
    parser.set_defaults(device="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1
        or args.talkshow_torch_threads < 1
        or args.seed < 0
    ):
        parser.error(
            "--batch-size/--talkshow-torch-threads must be positive and "
            "--seed non-negative"
        )
    prepared_values = (
        args.prepared_authority,
        args.expected_prepared_authority_sha256,
        args.expected_prepared_authority_bytes,
        args.expected_prepared_authority_receipt_payload_sha256,
    )
    if any(value is not None for value in prepared_values):
        parser.error(
            "terminal validator forbids prepared authority; fresh full "
            "authority replay is mandatory"
        )
    try:
        args.expected_audio_set_sha256 = _sha256(
            args.expected_audio_set_sha256,
            "--expected-audio-set-sha256",
        )
        args.expected_preflight_sha256 = _sha256(
            args.expected_preflight_sha256,
            "--expected-preflight-sha256",
        )
        args.expected_talkshow_validation_gate_sha256 = _sha256(
            args.expected_talkshow_validation_gate_sha256,
            "--expected-talkshow-validation-gate-sha256",
        )
        args.expected_talkshow_validation_gate_receipt_payload_sha256 = (
            _sha256(
                args.expected_talkshow_validation_gate_receipt_payload_sha256,
                "--expected-talkshow-validation-gate-receipt-payload-sha256",
            )
        )
        current = producer.build_preflight(args)
        validation = validate_final_result(args, current)
        output = producer._atomic_new(
            args.output_report,
            validation,
            "terminal DiffSHEG final result validation",
        )
        output_payload = output.read_bytes()
        result = {
            "status": "complete",
            "output": str(output),
            "sha256": hashlib.sha256(output_payload).hexdigest(),
            "bytes": len(output_payload),
            "canonical_payload_sha256": validation[
                "receipt_payload_sha256"
            ],
        }
    except (FinalResultValidationError, producer.FinalDiffSHEGError) as error:
        parser.exit(2, f"[abort] {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
