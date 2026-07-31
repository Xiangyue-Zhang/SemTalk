#!/usr/bin/env python3
"""Seal two static DiffSHEG validation partitions and their 22-way union.

This module is CPU-only.  It never launches inference or metric evaluation.
The guarded worker uses it to create one immutable receipt for its exact
11-candidate partition; the CPU finalizer uses it to recover the exact
22-measurement order consumed by ``select_base_official_adapt_long.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.show_base import base_long_val_contract as long_contract  # noqa: E402
from scripts.show_base import select_base_official_adapt_long as selector  # noqa: E402


SOURCE_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
PARTITION_COUNT = 2
SHARDS_PER_CANDIDATE = 8
CLIPS_PER_CANDIDATE = 1_715
PARTITION_FORMAT = "semtalk_show_base_diffsheg_val_partition_v1"
UNION_FORMAT = "semtalk_show_base_diffsheg_val_partition_union_v1"
EVALUATOR_BUNDLE_FORMAT = "semtalk_show_diffsheg_val_evaluator_bundle_v1"
PREFLIGHT_FORMAT = "semtalk_show_base_official_adapt_val_inference_preflight_v1"
LINEAGE_FILENAME = "val-inference-lineage.json"
EXPECTED_HOSTS = {
    0: (
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0"
    ),
    1: (
        "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0"
    ),
}
EXPECTED_EPOCHS = tuple(long_contract.EXPECTED_CANDIDATE_EPOCHS)
PARTITION_EPOCHS = {
    partition_id: tuple(
        epoch
        for index, epoch in enumerate(EXPECTED_EPOCHS)
        if index % PARTITION_COUNT == partition_id
    )
    for partition_id in range(PARTITION_COUNT)
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OID_RE = re.compile(r"[0-9a-f]{40}")
E30_RE = re.compile(r"(^|[^a-z0-9])e(?:poch)?[-_]?0*30([^a-z0-9]|$)")
SPEAKER2_RE = re.compile(r"(^|[^a-z0-9])speaker[-_]?0*2([^a-z0-9]|$)")


class PartitionContractError(RuntimeError):
    """Raised when the finite static validation transaction is not exact."""


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


def _payload_sha(value: Mapping[str, Any]) -> str:
    return long_contract.canonical_json_sha256(dict(value))


def _with_payload_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_payload_sha256"] = _payload_sha(result)
    return result


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PartitionContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_oid(value: object, label: str) -> str:
    if not isinstance(value, str) or OID_RE.fullmatch(value) is None:
        raise PartitionContractError(f"{label} must be a lowercase Git OID")
    return value


def _reject_forbidden_path(path: Path, label: str) -> None:
    for component in path.parts:
        normalized = component.casefold()
        if "test" in normalized:
            raise PartitionContractError(
                f"{label} exposes a test-labelled path: {path}"
            )
        if E30_RE.search(normalized):
            raise PartitionContractError(
                f"{label} exposes withdrawn e30: {path}"
            )
        if SPEAKER2_RE.search(normalized):
            raise PartitionContractError(
                f"{label} exposes forbidden Speaker2: {path}"
            )


def _absolute(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise PartitionContractError(f"{label} must be absolute: {path}")
    _reject_forbidden_path(path, label)
    return path


def _canonical_directory(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    try:
        resolved = path.resolve(strict=True)
        public = os.lstat(path)
    except OSError as error:
        raise PartitionContractError(f"{label} is unavailable: {path}") from error
    if (
        resolved != path
        or stat.S_ISLNK(public.st_mode)
        or not stat.S_ISDIR(public.st_mode)
    ):
        raise PartitionContractError(
            f"{label} must be a canonical non-symlink directory: {path}"
        )
    return resolved


def _canonical_file(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    try:
        resolved = path.resolve(strict=True)
        public = os.lstat(path)
    except OSError as error:
        raise PartitionContractError(f"{label} is unavailable: {path}") from error
    if (
        resolved != path
        or stat.S_ISLNK(public.st_mode)
        or not stat.S_ISREG(public.st_mode)
    ):
        raise PartitionContractError(
            f"{label} must be a canonical regular file: {path}"
        )
    return resolved


def _snapshot(
    path: Path,
    expected_sha256: str | None,
    label: str,
) -> tuple[Path, bytes, str]:
    path = _canonical_file(path, label)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise PartitionContractError(f"{label} changed while read: {path}")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        expected = _require_sha(expected_sha256, f"{label} expected SHA-256")
        if observed != expected:
            raise PartitionContractError(
                f"{label} SHA-256 mismatch: {observed} != {expected}"
            )
    return path, payload, observed


def _json_artifact(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    payload_receipt: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    resolved, payload, observed = _snapshot(path, expected_sha256, label)
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PartitionContractError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise PartitionContractError(f"{label} must be a JSON object")
    artifact = {"path": str(resolved), "sha256": observed}
    if payload_receipt:
        claimed = _require_sha(
            value.get("receipt_payload_sha256"),
            f"{label} payload SHA-256",
        )
        unsigned = dict(value)
        unsigned.pop("receipt_payload_sha256", None)
        if _payload_sha(unsigned) != claimed:
            raise PartitionContractError(f"{label} payload SHA mismatch")
        artifact["receipt_payload_sha256"] = claimed
    return artifact, value


def _write_new_json(path: Path, value: Mapping[str, Any], label: str) -> Path:
    path = _absolute(path, label)
    parent = _canonical_directory(path.parent, f"{label} parent")
    output = parent / path.name
    if output != path or os.path.lexists(output):
        raise PartitionContractError(f"refusing to overwrite {label}: {path}")
    payload = _canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            raise PartitionContractError(
                f"refusing to overwrite {label}: {output}"
            ) from None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _artifact_fields(path: Path, payload_receipt: bool) -> tuple[str, int, str]:
    resolved, payload, observed = _snapshot(path, None, "output artifact")
    claimed = ""
    if payload_receipt:
        artifact, _value = _json_artifact(
            resolved, observed, "output artifact", payload_receipt=True
        )
        claimed = artifact["receipt_payload_sha256"]
    return observed, len(payload), claimed


def _build_evaluator_bundle(paspa_root: Path, diffsheg_root: Path) -> dict[str, Any]:
    # Keep the receipt/union CLI usable in a minimal CPU Python.  The formal
    # evaluator environment imports numpy only for this one asset preflight.
    from scripts.show_base import evaluate_diffsheg_val_fgd as evaluator

    _reject_forbidden_path(_absolute(paspa_root, "PASPA root"), "PASPA root")
    _reject_forbidden_path(
        _absolute(diffsheg_root, "DiffSHEG root"), "DiffSHEG root"
    )
    paspa, paspa_git = evaluator._verify_git_checkout(
        paspa_root,
        label="pinned PASPA evaluator repository",
        expected_head=evaluator.PASPA_COMMIT,
        expected_tree=evaluator.PASPA_TREE,
        expected_origin=evaluator.PASPA_ORIGIN,
    )
    evaluator_path, evaluator_sha = evaluator._verify_file_sha256(
        paspa / evaluator.PASPA_EVALUATOR_RELATIVE,
        evaluator.PASPA_EVALUATOR_SHA256,
        "pinned PASPA DiffSHEG evaluator",
        val_only=False,
    )
    (
        diffsheg,
        stats_path,
        gesture_path,
        assets,
    ) = evaluator._verify_diffsheg_assets(diffsheg_root)
    return _with_payload_sha(
        {
            "format": EVALUATOR_BUNDLE_FORMAT,
            "status": "verified",
            "split": "val",
            "test_visible": False,
            "metric": "fgd",
            "protocol": {
                "name": evaluator.PROTOCOL_NAME,
                "version": evaluator.PROTOCOL_VERSION,
                "window_length": evaluator.WINDOW_LENGTH,
                "window_stride": evaluator.WINDOW_STRIDE,
                "precision": evaluator.PRECISION,
            },
            "paspa": {
                "root": str(paspa),
                "origin": paspa_git["origin"],
                "commit": paspa_git["git_head"],
                "tree": paspa_git["git_tree"],
                "evaluator": {
                    "path": str(evaluator_path),
                    "sha256": evaluator_sha,
                },
            },
            "diffsheg": {
                "root": str(diffsheg),
                "commit": assets["diffsheg_root"]["git_head"],
                "stats": {
                    "path": str(stats_path),
                    "sha256": evaluator.DIFFSHEG_STATS_SHA256,
                },
                "gesture_autoencoder": {
                    "path": str(gesture_path),
                    "sha256": evaluator.DIFFSHEG_GESTURE_AE_SHA256,
                    "input_dim": evaluator.GESTURE_DIM,
                    "latent_dim": evaluator.LATENT_DIM,
                },
            },
        }
    )


def _validate_evaluator_bundle(
    path: Path, expected_sha256: str
) -> tuple[dict[str, str], dict[str, Any]]:
    artifact, value = _json_artifact(
        path,
        expected_sha256,
        "DiffSHEG evaluator bundle",
        payload_receipt=True,
    )
    if value.get("format") != EVALUATOR_BUNDLE_FORMAT:
        raise PartitionContractError("evaluator bundle format changed")
    try:
        replayed = _build_evaluator_bundle(
            Path(value["paspa"]["root"]),
            Path(value["diffsheg"]["root"]),
        )
    except (KeyError, TypeError) as error:
        raise PartitionContractError("evaluator bundle schema changed") from error
    if value != replayed:
        raise PartitionContractError("evaluator bundle replay changed")
    return artifact, value


def _candidate_context(args: argparse.Namespace) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    val_artifact, val_coverage = long_contract.validate_val_inputs(
        args.val_inputs_json, args.expected_val_inputs_sha256
    )
    pipeline_artifact, pipeline = long_contract.validate_pipeline(
        args.pipeline_json, args.expected_pipeline_sha256
    )
    expected_selected = {
        stage: pipeline["fixed_checkpoints"][stage]["sha256"]
        for stage in ("face", "hands", "upper", "lower", "global")
    }
    candidate_bundle = long_contract.validate_candidate_bundle(
        manifest_path=args.base_candidate_manifest,
        expected_manifest_sha256=args.expected_base_candidate_manifest_sha256,
        status_path=args.base_status_json,
        expected_status_sha256=args.expected_base_formal_status_sha256,
        frozen_inputs_path=args.base_frozen_inputs_json,
        expected_frozen_inputs_sha256=args.expected_base_frozen_inputs_sha256,
        expected_selected_prerequisite_sha256=expected_selected,
    )
    return candidate_bundle, val_artifact, val_coverage, pipeline_artifact


def _input_artifacts(
    candidate_bundle: Mapping[str, Any],
    val_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_manifest": dict(candidate_bundle["manifest"]),
        "candidate_status": dict(candidate_bundle["status"]),
        "frozen_inputs": dict(candidate_bundle["frozen_inputs"]),
        "val_inputs": dict(val_artifact),
        "pipeline": dict(pipeline_artifact),
    }


def _validate_preflight(
    args: argparse.Namespace,
    candidate_bundle: Mapping[str, Any],
    val_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _json_artifact(
        args.preflight_json,
        args.expected_preflight_sha256,
        "validation inference preflight",
        payload_receipt=True,
    )
    if artifact["receipt_payload_sha256"] != _require_sha(
        args.expected_preflight_payload_sha256,
        "preflight payload SHA-256",
    ):
        raise PartitionContractError("preflight payload SHA mismatch")
    expected_keys = {
        "format", "status", "split", "test_visible", "candidate_epochs",
        "candidate_bundle", "val_inputs_receipt", "pipeline_receipt",
        "pipeline_source", "inference_entrypoint", "coverage",
        "receipt_payload_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("format") != PREFLIGHT_FORMAT
        or value.get("status") != "complete"
        or value.get("split") != "val"
        or value.get("test_visible") is not False
        or value.get("candidate_epochs") != list(EXPECTED_EPOCHS)
        or value.get("pipeline_source", {}).get("origin") != SOURCE_ORIGIN
        or value.get("pipeline_source", {}).get("clean") is not True
        or value.get("pipeline_source", {}).get("detached") is not True
        or value.get("pipeline_source", {}).get("local_branches_at_commit") != []
    ):
        raise PartitionContractError("preflight is not the common val-only receipt")
    preflight_bundle = value["candidate_bundle"]
    expected_candidates = {
        str(epoch): candidate_bundle["candidates"][epoch]
        for epoch in EXPECTED_EPOCHS
    }
    for role in ("manifest", "status", "frozen_inputs"):
        if preflight_bundle.get(role) != candidate_bundle[role]:
            raise PartitionContractError(f"preflight {role} changed")
    if (
        preflight_bundle.get("candidates") != expected_candidates
        or value.get("val_inputs_receipt") != val_artifact
        or value.get("pipeline_receipt") != pipeline_artifact
    ):
        raise PartitionContractError("preflight common inputs changed")
    return artifact, value


def _measurement_rows(
    args: argparse.Namespace,
    *,
    candidate_bundle: Mapping[str, Any],
    val_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
    expected_epochs: Sequence[int],
    run_root: Path,
) -> list[dict[str, Any]]:
    if (
        len(args.measurement_json) != len(expected_epochs)
        or len(args.expected_measurement_sha256) != len(expected_epochs)
    ):
        raise PartitionContractError(
            f"exactly {len(expected_epochs)} measurement path/SHA pairs are required"
        )
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_shas: set[str] = set()
    seen_lineage: set[str] = set()
    seen_report: set[str] = set()
    for epoch, path, digest in zip(
        expected_epochs,
        args.measurement_json,
        args.expected_measurement_sha256,
    ):
        expected_path = run_root / "candidates" / f"e{epoch}" / (
            "diffsheg-val-measurement.json"
        )
        if path != expected_path:
            raise PartitionContractError(
                f"measurement e{epoch} escaped its new run root"
            )
        artifact, measurement = _json_artifact(
            path,
            digest,
            f"Base e{epoch} measurement",
            payload_receipt=True,
        )
        audited, row = selector.validate_measurement(
            measurement_path=path,
            expected_measurement_sha256=digest,
            candidate_bundle=candidate_bundle,
        )
        fgd = float(row["metrics"]["fgd"])
        if (
            audited != artifact
            or row.get("epoch") != epoch
            or measurement.get("val_inputs_receipt") != val_artifact
            or measurement.get("pipeline_receipt") != pipeline_artifact
            or not math.isfinite(fgd)
            or fgd < 0.0
        ):
            raise PartitionContractError(f"measurement e{epoch} changed")
        lineage_path = row["inference_lineage"]["path"]
        report_path = row["diffsheg_report"]["path"]
        candidate_root = run_root / "candidates" / f"e{epoch}"
        if (
            Path(lineage_path)
            != candidate_root / "final" / LINEAGE_FILENAME
            or Path(report_path) != candidate_root / "diffsheg-val-fgd.json"
            or row["inference_outputs"].get("prediction_dir")
            != str(candidate_root / "final" / "predictions" / "val")
            or row["inference_outputs"].get("ground_truth_dir")
            != str(candidate_root / "final" / "ground-truth" / "val")
        ):
            raise PartitionContractError(
                f"measurement e{epoch} output lineage escaped its candidate root"
            )
        if (
            artifact["path"] in seen_paths
            or artifact["sha256"] in seen_shas
            or lineage_path in seen_lineage
            or report_path in seen_report
        ):
            raise PartitionContractError("candidate measurement outputs were reused")
        seen_paths.add(artifact["path"])
        seen_shas.add(artifact["sha256"])
        seen_lineage.add(lineage_path)
        seen_report.add(report_path)
        rows.append(
            {
                "epoch": epoch,
                "fgd": fgd,
                "measurement": artifact,
                "inference_lineage": dict(row["inference_lineage"]),
                "diffsheg_report": dict(row["diffsheg_report"]),
                "prediction_dir": row["inference_outputs"]["prediction_dir"],
                "ground_truth_dir": row["inference_outputs"]["ground_truth_dir"],
                "exact_once": row["inference_outputs"].get("exact_once") is True,
                "finite": row["inference_outputs"].get("finite") is True,
            }
        )
    if any(not row["exact_once"] or not row["finite"] for row in rows):
        raise PartitionContractError("candidate inference is not exact-once and finite")
    return rows


def seal_partition(args: argparse.Namespace) -> dict[str, Any]:
    if args.partition_count != PARTITION_COUNT or args.partition_id not in (0, 1):
        raise PartitionContractError("formal partition must be exactly 0/2 or 1/2")
    expected_epochs = PARTITION_EPOCHS[args.partition_id]
    if args.formal_host != EXPECTED_HOSTS[args.partition_id]:
        raise PartitionContractError("formal host does not own this static partition")
    source_commit = _require_oid(args.source_commit, "source commit")
    source_tree = _require_oid(args.source_tree, "source tree")
    run_root = _canonical_directory(args.run_root, "partition run root")
    candidate_bundle, val_artifact, _coverage, pipeline_artifact = (
        _candidate_context(args)
    )
    preflight_artifact, _preflight = _validate_preflight(
        args, candidate_bundle, val_artifact, pipeline_artifact
    )
    evaluator_artifact, evaluator_bundle = _validate_evaluator_bundle(
        args.evaluator_bundle_json,
        args.expected_evaluator_bundle_sha256,
    )
    if evaluator_artifact["receipt_payload_sha256"] != _require_sha(
        args.expected_evaluator_bundle_payload_sha256,
        "evaluator bundle payload SHA-256",
    ):
        raise PartitionContractError("evaluator bundle payload root changed")
    rows = _measurement_rows(
        args,
        candidate_bundle=candidate_bundle,
        val_artifact=val_artifact,
        pipeline_artifact=pipeline_artifact,
        expected_epochs=expected_epochs,
        run_root=run_root,
    )
    receipt = _with_payload_sha(
        {
            "format": PARTITION_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "selection_metric": "fgd",
            "partition_id": args.partition_id,
            "partition_count": PARTITION_COUNT,
            "formal_host": args.formal_host,
            "source": {
                "origin": SOURCE_ORIGIN,
                "commit": source_commit,
                "tree": source_tree,
                "clean": True,
                "detached": True,
                "local_branches": [],
            },
            "run_root": str(run_root),
            "candidate_epochs": list(expected_epochs),
            "candidate_count": len(expected_epochs),
            "shards_per_candidate": SHARDS_PER_CANDIDATE,
            "clip_evaluations": len(expected_epochs) * CLIPS_PER_CANDIDATE,
            "inputs": _input_artifacts(
                candidate_bundle, val_artifact, pipeline_artifact
            ),
            "preflight": preflight_artifact,
            "evaluator_bundle": evaluator_artifact,
            "evaluator_bundle_identity": {
                "paspa_commit": evaluator_bundle["paspa"]["commit"],
                "paspa_tree": evaluator_bundle["paspa"]["tree"],
                "paspa_evaluator_sha256": evaluator_bundle["paspa"]["evaluator"]["sha256"],
                "diffsheg_commit": evaluator_bundle["diffsheg"]["commit"],
                "stats_sha256": evaluator_bundle["diffsheg"]["stats"]["sha256"],
                "gesture_autoencoder_sha256": evaluator_bundle["diffsheg"]["gesture_autoencoder"]["sha256"],
            },
            "measurements": rows,
            "exact_once": True,
            "finite": True,
        }
    )
    expected_output = run_root / "partition-receipt.json"
    if args.output_json != expected_output:
        raise PartitionContractError("partition receipt escaped its run root")
    _write_new_json(args.output_json, receipt, "partition receipt")
    return receipt


def _partition_receipt(
    path: Path,
    expected_sha256: str,
    expected_payload_sha256: str,
    partition_id: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    artifact, value = _json_artifact(
        path,
        expected_sha256,
        f"partition {partition_id} receipt",
        payload_receipt=True,
    )
    if artifact["receipt_payload_sha256"] != _require_sha(
        expected_payload_sha256,
        f"partition {partition_id} external payload SHA-256",
    ):
        raise PartitionContractError("partition external payload root changed")
    exact_keys = {
        "format", "status", "split", "test_visible", "selection_metric",
        "partition_id", "partition_count", "formal_host", "source", "run_root",
        "candidate_epochs", "candidate_count", "shards_per_candidate",
        "clip_evaluations", "inputs", "preflight", "evaluator_bundle",
        "evaluator_bundle_identity", "measurements", "exact_once", "finite",
        "receipt_payload_sha256",
    }
    if (
        set(value) != exact_keys
        or value.get("format") != PARTITION_FORMAT
        or value.get("status") != "complete"
        or value.get("split") != "val"
        or value.get("test_visible") is not False
        or value.get("selection_metric") != "fgd"
        or value.get("partition_id") != partition_id
        or value.get("partition_count") != PARTITION_COUNT
        or value.get("formal_host") != EXPECTED_HOSTS[partition_id]
        or value.get("candidate_epochs") != list(PARTITION_EPOCHS[partition_id])
        or value.get("candidate_count") != len(PARTITION_EPOCHS[partition_id])
        or value.get("shards_per_candidate") != SHARDS_PER_CANDIDATE
        or value.get("clip_evaluations")
        != len(PARTITION_EPOCHS[partition_id]) * CLIPS_PER_CANDIDATE
        or value.get("exact_once") is not True
        or value.get("finite") is not True
    ):
        raise PartitionContractError(f"partition {partition_id} receipt changed")
    if Path(artifact["path"]) != Path(value["run_root"]) / "partition-receipt.json":
        raise PartitionContractError("partition receipt path/run-root mismatch")
    measurements = value.get("measurements")
    if (
        not isinstance(measurements, list)
        or [row.get("epoch") for row in measurements]
        != list(PARTITION_EPOCHS[partition_id])
        or any(
            not isinstance(row, dict)
            or row.get("exact_once") is not True
            or row.get("finite") is not True
            or not isinstance(row.get("fgd"), (int, float))
            or isinstance(row.get("fgd"), bool)
            or not math.isfinite(float(row["fgd"]))
            or float(row["fgd"]) < 0.0
            for row in measurements
        )
    ):
        raise PartitionContractError("partition measurements are not finite/exact")
    return artifact, value


def build_union(args: argparse.Namespace) -> dict[str, Any]:
    if args.partition_count != PARTITION_COUNT:
        raise PartitionContractError("formal union requires exactly two partitions")
    if not (
        len(args.partition_receipt) == PARTITION_COUNT
        and len(args.expected_partition_receipt_sha256) == PARTITION_COUNT
        and len(args.expected_partition_receipt_payload_sha256) == PARTITION_COUNT
    ):
        raise PartitionContractError("union requires two ordered partition triples")
    source = {
        "origin": SOURCE_ORIGIN,
        "commit": _require_oid(args.source_commit, "source commit"),
        "tree": _require_oid(args.source_tree, "source tree"),
        "clean": True,
        "detached": True,
        "local_branches": [],
    }
    candidate_bundle, val_artifact, _coverage, pipeline_artifact = (
        _candidate_context(args)
    )
    expected_inputs = _input_artifacts(
        candidate_bundle, val_artifact, pipeline_artifact
    )
    partition_artifacts: list[dict[str, str]] = []
    partitions: list[dict[str, Any]] = []
    for partition_id in range(PARTITION_COUNT):
        artifact, value = _partition_receipt(
            args.partition_receipt[partition_id],
            args.expected_partition_receipt_sha256[partition_id],
            args.expected_partition_receipt_payload_sha256[partition_id],
            partition_id,
        )
        if value.get("source") != source or value.get("inputs") != expected_inputs:
            raise PartitionContractError("partition source/input roots disagree")
        partition_artifacts.append(artifact)
        partitions.append(value)
    if len({row["run_root"] for row in partitions}) != PARTITION_COUNT:
        raise PartitionContractError("partition run roots must be distinct")
    preflight_roots = {
        (
            row["preflight"].get("sha256"),
            row["preflight"].get("receipt_payload_sha256"),
        )
        for row in partitions
    }
    evaluator_roots = {
        (
            row["evaluator_bundle"].get("sha256"),
            row["evaluator_bundle"].get("receipt_payload_sha256"),
            json.dumps(row["evaluator_bundle_identity"], sort_keys=True),
        )
        for row in partitions
    }
    if len(preflight_roots) != 1 or len(evaluator_roots) != 1:
        raise PartitionContractError("partitions do not share one common preflight/bundle")
    by_epoch: dict[int, dict[str, Any]] = {}
    seen_measurement_paths: set[str] = set()
    seen_measurement_shas: set[str] = set()
    seen_lineage: set[str] = set()
    seen_report: set[str] = set()
    seen_prediction: set[str] = set()
    for partition in partitions:
        for row in partition["measurements"]:
            epoch = row["epoch"]
            measurement = row.get("measurement")
            if (
                epoch in by_epoch
                or not isinstance(measurement, dict)
                or set(measurement) != {"path", "sha256", "receipt_payload_sha256"}
                or measurement["path"] in seen_measurement_paths
                or measurement["sha256"] in seen_measurement_shas
                or row["inference_lineage"]["path"] in seen_lineage
                or row["diffsheg_report"]["path"] in seen_report
                or row["prediction_dir"] in seen_prediction
            ):
                raise PartitionContractError("partition candidate outputs overlap")
            _require_sha(measurement["sha256"], "measurement SHA-256")
            _require_sha(
                measurement["receipt_payload_sha256"],
                "measurement payload SHA-256",
            )
            seen_measurement_paths.add(measurement["path"])
            seen_measurement_shas.add(measurement["sha256"])
            seen_lineage.add(row["inference_lineage"]["path"])
            seen_report.add(row["diffsheg_report"]["path"])
            seen_prediction.add(row["prediction_dir"])
            by_epoch[epoch] = row
    if tuple(by_epoch) == EXPECTED_EPOCHS:
        # Modulo partitions intentionally interleave; insertion order must not
        # accidentally become the selector order.
        raise PartitionContractError("partition rows unexpectedly claimed global order")
    if set(by_epoch) != set(EXPECTED_EPOCHS):
        raise PartitionContractError("partition union is not exact 22-way coverage")
    measurements = [
        {
            "epoch": epoch,
            "fgd": float(by_epoch[epoch]["fgd"]),
            "measurement": dict(by_epoch[epoch]["measurement"]),
        }
        for epoch in EXPECTED_EPOCHS
    ]
    return _with_payload_sha(
        {
            "format": UNION_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "selection_metric": "fgd",
            "selection_ordering": ["fgd", "epoch"],
            "source": source,
            "inputs": expected_inputs,
            "partition_count": PARTITION_COUNT,
            "candidate_epochs": list(EXPECTED_EPOCHS),
            "candidate_count": len(EXPECTED_EPOCHS),
            "shards_per_candidate": SHARDS_PER_CANDIDATE,
            "total_shards": len(EXPECTED_EPOCHS) * SHARDS_PER_CANDIDATE,
            "clip_evaluations": len(EXPECTED_EPOCHS) * CLIPS_PER_CANDIDATE,
            "common_preflight": {
                "sha256": next(iter(preflight_roots))[0],
                "receipt_payload_sha256": next(iter(preflight_roots))[1],
            },
            "evaluator_bundle_identity": dict(
                partitions[0]["evaluator_bundle_identity"]
            ),
            "partitions": partition_artifacts,
            "measurements": measurements,
            "exact_once": True,
            "finite": True,
        }
    )


def _validate_union(
    path: Path, expected_sha256: str, expected_payload_sha256: str
) -> tuple[dict[str, str], dict[str, Any]]:
    artifact, value = _json_artifact(
        path, expected_sha256, "partition union", payload_receipt=True
    )
    if artifact["receipt_payload_sha256"] != _require_sha(
        expected_payload_sha256, "union payload SHA-256"
    ):
        raise PartitionContractError("union payload root changed")
    measurements = value.get("measurements")
    if (
        value.get("format") != UNION_FORMAT
        or value.get("status") != "complete"
        or value.get("split") != "val"
        or value.get("test_visible") is not False
        or value.get("selection_metric") != "fgd"
        or value.get("selection_ordering") != ["fgd", "epoch"]
        or value.get("candidate_epochs") != list(EXPECTED_EPOCHS)
        or value.get("candidate_count") != len(EXPECTED_EPOCHS)
        or value.get("partition_count") != PARTITION_COUNT
        or value.get("total_shards") != len(EXPECTED_EPOCHS) * SHARDS_PER_CANDIDATE
        or value.get("exact_once") is not True
        or value.get("finite") is not True
        or not isinstance(measurements, list)
        or [row.get("epoch") for row in measurements] != list(EXPECTED_EPOCHS)
    ):
        raise PartitionContractError("partition union changed")
    return artifact, value


def _add_candidate_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-candidate-manifest-sha256", required=True)
    parser.add_argument("--base-status-json", type=Path, required=True)
    parser.add_argument("--expected-base-formal-status-sha256", required=True)
    parser.add_argument("--base-frozen-inputs-json", type=Path, required=True)
    parser.add_argument("--expected-base-frozen-inputs-sha256", required=True)
    parser.add_argument("--val-inputs-json", type=Path, required=True)
    parser.add_argument("--expected-val-inputs-sha256", required=True)
    parser.add_argument("--pipeline-json", type=Path, required=True)
    parser.add_argument("--expected-pipeline-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create-run-root", allow_abbrev=False)
    create.add_argument("--path", type=Path, required=True)
    create.add_argument("--subdirectory", action="append", default=[])

    fields = commands.add_parser("artifact-fields", allow_abbrev=False)
    fields.add_argument("--artifact-path", type=Path, required=True)
    fields.add_argument("--payload-receipt", action="store_true")

    epochs = commands.add_parser("partition-epochs", allow_abbrev=False)
    epochs.add_argument("--partition-id", type=int, required=True)
    epochs.add_argument("--partition-count", type=int, required=True)

    bundle = commands.add_parser("evaluator-preflight", allow_abbrev=False)
    bundle.add_argument("--paspa-root", type=Path, required=True)
    bundle.add_argument("--diffsheg-root", type=Path, required=True)
    bundle.add_argument("--output-json", type=Path, required=True)

    seal = commands.add_parser("seal-partition", allow_abbrev=False)
    seal.add_argument("--partition-id", type=int, required=True)
    seal.add_argument("--partition-count", type=int, required=True)
    seal.add_argument("--formal-host", required=True)
    seal.add_argument("--run-root", type=Path, required=True)
    seal.add_argument("--source-commit", required=True)
    seal.add_argument("--source-tree", required=True)
    _add_candidate_inputs(seal)
    seal.add_argument("--preflight-json", type=Path, required=True)
    seal.add_argument("--expected-preflight-sha256", required=True)
    seal.add_argument("--expected-preflight-payload-sha256", required=True)
    seal.add_argument("--evaluator-bundle-json", type=Path, required=True)
    seal.add_argument("--expected-evaluator-bundle-sha256", required=True)
    seal.add_argument("--expected-evaluator-bundle-payload-sha256", required=True)
    seal.add_argument("--measurement-json", action="append", type=Path, default=[])
    seal.add_argument("--expected-measurement-sha256", action="append", default=[])
    seal.add_argument("--output-json", type=Path, required=True)

    union = commands.add_parser("union-partitions", allow_abbrev=False)
    union.add_argument("--partition-count", type=int, required=True)
    union.add_argument("--source-commit", required=True)
    union.add_argument("--source-tree", required=True)
    _add_candidate_inputs(union)
    union.add_argument("--partition-receipt", action="append", type=Path, default=[])
    union.add_argument("--expected-partition-receipt-sha256", action="append", default=[])
    union.add_argument(
        "--expected-partition-receipt-payload-sha256", action="append", default=[]
    )
    union.add_argument("--output-json", type=Path, required=True)

    extract = commands.add_parser("extract-union", allow_abbrev=False)
    extract.add_argument("--union-json", type=Path, required=True)
    extract.add_argument("--expected-union-sha256", required=True)
    extract.add_argument("--expected-union-payload-sha256", required=True)
    return parser


def _create_run_root(args: argparse.Namespace) -> None:
    path = _absolute(args.path, "run root")
    parent = _canonical_directory(path.parent, "run-root parent")
    if path != parent / path.name or os.path.lexists(path):
        raise PartitionContractError("formal run root must be new and canonical")
    names = args.subdirectory
    if len(names) != len(set(names)) or any(
        re.fullmatch(r"[a-z][a-z0-9-]*", name or "") is None for name in names
    ):
        raise PartitionContractError("unsafe or duplicate run-root subdirectory")
    os.mkdir(path, 0o700)
    try:
        for name in names:
            os.mkdir(path / name, 0o700)
    except BaseException:
        for name in reversed(names):
            try:
                os.rmdir(path / name)
            except OSError:
                pass
        try:
            os.rmdir(path)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create-run-root":
            _create_run_root(args)
            result: Any = {"status": "created", "path": str(args.path)}
        elif args.command == "artifact-fields":
            digest, size, payload = _artifact_fields(
                args.artifact_path, args.payload_receipt
            )
            sys.stdout.buffer.write(
                digest.encode() + b"\0" + str(size).encode() + b"\0" + payload.encode() + b"\0"
            )
            return 0
        elif args.command == "partition-epochs":
            if args.partition_count != PARTITION_COUNT or args.partition_id not in (0, 1):
                raise PartitionContractError("formal partition must be exactly 0/2 or 1/2")
            for epoch in PARTITION_EPOCHS[args.partition_id]:
                sys.stdout.buffer.write(str(epoch).encode() + b"\0")
            return 0
        elif args.command == "evaluator-preflight":
            result = _build_evaluator_bundle(args.paspa_root, args.diffsheg_root)
            _write_new_json(args.output_json, result, "evaluator bundle")
        elif args.command == "seal-partition":
            result = seal_partition(args)
        elif args.command == "union-partitions":
            result = build_union(args)
            _write_new_json(args.output_json, result, "partition union")
        elif args.command == "extract-union":
            _artifact, result = _validate_union(
                args.union_json,
                args.expected_union_sha256,
                args.expected_union_payload_sha256,
            )
            for row in result["measurements"]:
                measurement = row["measurement"]
                sys.stdout.buffer.write(
                    measurement["path"].encode()
                    + b"\0"
                    + measurement["sha256"].encode()
                    + b"\0"
                )
            return 0
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (
        PartitionContractError,
        long_contract.SelectionContractError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
