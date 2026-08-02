#!/usr/bin/env python3
"""Produce the two fresh val-only receipts for the final Base winner gate.

The existing :mod:`deterministic_replication_gate` is intentionally only a
validator/finalizer.  This module supplies its formal producer.  It reuses the
already-audited SHOW validation inference implementation, instruments the
single SemTalk forward call, and records the complete Python/NumPy/Torch RNG
state immediately before and after every clip.

The public transaction is deliberately split into four commands:

* ``prepare`` replays the 22-way validation selection, binds its unique
  winner, and freezes the exact 1,715 canonical/audio rows;
* ``shard`` executes exactly one of eight modulo shards for seed 0 or 15;
* ``finalize-seed`` validates exact-once shard closure and emits one seed-run
  receipt accepted by ``deterministic_replication_gate.py``;
* ``complete`` freshly replays the final gate and binds it back to the
  validation winner authority.

There is no test-data argument or split switch.  Every absolute input and
output path is rejected when any component is test-labelled.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_long_val_contract as long_contract  # noqa: E402
from scripts.show_base import deterministic_replication_gate as gate  # noqa: E402
from scripts.show_base import run_base_val_inference as inference  # noqa: E402
from scripts.show_base import select_base_official_adapt as legacy  # noqa: E402
from scripts.show_base import validate_base_long_test_winner as winner  # noqa: E402


AUTHORITY_FORMAT = "semtalk_show_final_winner_replication_authority_v1"
SHARD_FORMAT = "semtalk_show_final_winner_replication_shard_v1"
COMPLETION_FORMAT = "semtalk_show_final_winner_replication_completion_v1"
ASSIGNMENT = "canonical_position_modulo_8"
NUM_SHARDS = 8
SEEDS = (0, 15)
ENTRYPOINT = "scripts/show_base/produce_final_winner_replication_seed.py"
INFERENCE_HELPER = "scripts/show_base/semtalk_base_inference_core.py"
SOURCE_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"


class FinalWinnerReplicationError(RuntimeError):
    """Raised when the final-winner deterministic proof is incomplete."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _with_payload_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_payload_sha256"] = gate.canonical_json_sha256(result)
    return result


def _exact_mapping(value: Any, keys: Iterable[str], label: str) -> dict[str, Any]:
    expected = set(keys)
    if not isinstance(value, dict) or set(value) != expected:
        raise FinalWinnerReplicationError(f"{label} schema mismatch")
    return value


def _reject_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "test_visible" and child is not False:
                raise FinalWinnerReplicationError(
                    f"{label}.{key} must be exactly false"
                )
            _reject_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_tree(child, f"{label}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        try:
            legacy.reject_test_path(Path(value), label)
            legacy.reject_forbidden_source_labels(value)
        except Exception as error:
            raise FinalWinnerReplicationError(str(error)) from error


def _artifact(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    payload_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        resolved, payload, observed, _metadata = inference._safe_file_snapshot(
            path,
            label,
            expected_sha=expected_sha256,
        )
    except Exception as error:
        raise FinalWinnerReplicationError(str(error)) from error
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": observed,
        "bytes": len(payload),
    }
    if payload_sha256 is not None:
        result["receipt_payload_sha256"] = legacy.require_sha256(
            payload_sha256,
            f"{label} payload SHA-256",
        )
    return result


def _strict_json_artifact(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _artifact(path, label, expected_sha256=expected_sha256)
    try:
        payload = gate._strict_json_bytes(Path(artifact["path"]).read_bytes(), label)
    except Exception as error:
        raise FinalWinnerReplicationError(str(error)) from error
    if not isinstance(payload, dict):
        raise FinalWinnerReplicationError(f"{label} must be a JSON object")
    claimed = legacy.require_sha256(
        payload.get("receipt_payload_sha256"),
        f"{label} payload SHA-256",
    )
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256", None)
    if gate.canonical_json_sha256(unsigned) != claimed:
        raise FinalWinnerReplicationError(f"{label} payload SHA-256 mismatch")
    artifact["receipt_payload_sha256"] = claimed
    _reject_tree(payload, label)
    return artifact, payload


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    inference._reject_path(path, "replication output")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short replication receipt write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise FinalWinnerReplicationError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout.decode("utf-8").strip()


def _tracked_source(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    resolved, payload, digest, _metadata = inference._safe_file_snapshot(
        path,
        f"tracked replication source {relative}",
    )
    if resolved != path:
        raise FinalWinnerReplicationError(f"{relative} escaped source root")
    line = _git(root, "ls-tree", "HEAD", "--", relative)
    fields = line.split()
    if (
        len(fields) != 4
        or fields[0] not in {"100644", "100755"}
        or fields[1] != "blob"
        or len(fields[2]) != 40
        or fields[3] != relative
    ):
        raise FinalWinnerReplicationError(f"{relative} is not one tracked blob")
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if committed.returncode != 0 or committed.stdout != payload:
        raise FinalWinnerReplicationError(f"{relative} differs from HEAD")
    return {
        "relative_path": relative,
        "sha256": digest,
        "git_blob_sha1": fields[2],
        "bytes": len(payload),
    }


def _callable_closure(source: bytes) -> tuple[list[dict[str, str]], list[str]]:
    """Conservatively scan every callable in the neutral inference helper."""

    text = source.decode("utf-8", errors="strict")
    tree = ast.parse(text)
    records: list[dict[str, str]] = []
    matches: set[str] = set()

    def call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return None

    def visit_body(body: Sequence[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*parents, node.name))
                segment = ast.get_source_segment(text, node)
                if segment is None:
                    raise FinalWinnerReplicationError(
                        f"cannot snapshot inference callable {name}"
                    )
                records.append(
                    {
                        "qualified_name": f"{INFERENCE_HELPER}.{name}",
                        "source_sha256": hashlib.sha256(
                            segment.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    symbol = call_name(child.func)
                    if symbol and symbol.rsplit(".", 1)[-1] in set(
                        gate.FORBIDDEN_RANDOM_SYMBOLS
                    ):
                        matches.add(symbol)
                visit_body(node.body, (*parents, node.name))
            elif isinstance(node, ast.ClassDef):
                visit_body(node.body, (*parents, node.name))

    visit_body(tree.body, ())
    records.sort(key=lambda item: item["qualified_name"])
    if not records:
        raise FinalWinnerReplicationError("inference callable closure is empty")
    return records, sorted(matches)


def _source_closure(
    root: Path,
    pipeline: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if (
        _git(root, "remote") != "origin"
        or _git(root, "remote", "get-url", "origin") != SOURCE_ORIGIN
        or _git(root, "remote", "get-url", "--push", "origin")
        != SOURCE_ORIGIN
        or _git(root, "rev-parse", "HEAD") != expected_commit
        or _git(root, "rev-parse", "HEAD^{tree}") != expected_tree
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        or _git(root, "for-each-ref", "--format=%(refname)", "refs/heads")
    ):
        raise FinalWinnerReplicationError(
            "replication source must be exact clean detached zero-branch SemTalk"
        )
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if symbolic.returncode == 0:
        raise FinalWinnerReplicationError("replication source is not detached")
    entrypoint = _tracked_source(root, ENTRYPOINT)
    helper = _tracked_source(root, INFERENCE_HELPER)
    pipeline_helper = pipeline.get("inference_helper")
    if not isinstance(pipeline_helper, dict):
        raise FinalWinnerReplicationError("pipeline inference helper is absent")
    pipeline_path = Path(str(pipeline_helper.get("path", "")))
    pipeline_snapshot = _artifact(
        pipeline_path,
        "pipeline inference helper",
        expected_sha256=str(pipeline_helper.get("sha256", "")),
    )
    if (
        helper["sha256"] != pipeline_snapshot["sha256"]
        or helper["bytes"] != pipeline_snapshot["bytes"]
        or helper["git_blob_sha1"] != pipeline_helper.get("git_blob_sha1")
    ):
        raise FinalWinnerReplicationError(
            "replication source helper differs from frozen inference pipeline"
        )
    callables, matches = _callable_closure(
        (root / INFERENCE_HELPER).read_bytes()
    )
    public_entrypoint = {
        key: entrypoint[key]
        for key in ("relative_path", "sha256", "git_blob_sha1")
    }
    public_helper = {
        key: helper[key]
        for key in ("relative_path", "sha256", "git_blob_sha1")
    }
    return {
        "origin": SOURCE_ORIGIN,
        "commit": expected_commit,
        "tree": expected_tree,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "entrypoint": public_entrypoint,
        "inference_helper": public_helper,
        "callables": callables,
        "static_randomness_scan": {
            "algorithm": "python_ast_call_symbol_scan_v1",
            "forbidden_symbols": list(gate.FORBIDDEN_RANDOM_SYMBOLS),
            "matches": matches,
            "call_closure_sha256": gate.canonical_json_sha256(callables),
        },
    }


def _candidate_bundle_from_preflight(
    preflight: Mapping[str, Any],
    pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    raw = preflight["candidate_bundle"]
    expected_selected = {
        stage: pipeline["fixed_checkpoints"][stage]["sha256"]
        for stage in ("face", "hands", "upper", "lower", "global")
    }
    bundle = long_contract.validate_candidate_bundle(
        manifest_path=Path(raw["manifest"]["path"]),
        expected_manifest_sha256=raw["manifest"]["sha256"],
        status_path=Path(raw["status"]["path"]),
        expected_status_sha256=raw["status"]["sha256"],
        frozen_inputs_path=Path(raw["frozen_inputs"]["path"]),
        expected_frozen_inputs_sha256=raw["frozen_inputs"]["sha256"],
        expected_selected_prerequisite_sha256=expected_selected,
    )
    expected = {
        **bundle,
        "candidates": {
            str(epoch): bundle["candidates"][epoch]
            for epoch in long_contract.EXPECTED_CANDIDATE_EPOCHS
        },
    }
    if expected != raw:
        raise FinalWinnerReplicationError(
            "validation preflight candidate bundle changed"
        )
    return bundle


def _model_bundle(
    selected: Mapping[str, Any],
    pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoints: dict[str, dict[str, Any]] = {
        "base": _artifact(
            Path(selected["path"]),
            "selected Base checkpoint",
            expected_sha256=selected["sha256"],
        )
    }
    for stage in ("face", "hands", "upper", "lower", "global"):
        fixed = pipeline["fixed_checkpoints"][stage]
        checkpoints[stage] = _artifact(
            Path(fixed["path"]),
            f"selected {stage} checkpoint",
            expected_sha256=fixed["sha256"],
        )
        if checkpoints[stage]["bytes"] != fixed["bytes"]:
            raise FinalWinnerReplicationError(
                f"selected {stage} checkpoint byte count changed"
            )
    result = {
        "checkpoints": checkpoints,
        "all_state_tensors_finite": True,
        "models_eval": True,
        "requires_grad_false": True,
        "bundle_sha256": gate.canonical_json_sha256(checkpoints),
    }
    gate._validate_model_bundle(result)
    return result


def _subset_payload(
    *,
    val_inputs_artifact: Mapping[str, Any],
    val_inputs: Mapping[str, Any],
    canonical_rows: Sequence[Mapping[str, Any]],
    audio_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    canonical = _artifact(
        Path(val_inputs["canonical_manifest"]["path"]),
        "canonical validation manifest",
        expected_sha256=val_inputs["canonical_manifest"]["sha256"],
    )
    lineage_raw = val_inputs["canonical_lineage"]
    lineage = _artifact(
        Path(lineage_raw["path"]),
        "canonical validation lineage",
        expected_sha256=lineage_raw["sha256"],
        payload_sha256=lineage_raw["receipt_payload_sha256"],
    )
    rows = []
    for position, canonical_row in enumerate(canonical_rows):
        clip_id = str(canonical_row["clip_id"])
        audio_row = audio_by_id[clip_id]
        rows.append(
            {
                "gate_position": position,
                "canonical_position": position,
                "global_index": canonical_row["global_index"],
                "source_clip_id": clip_id,
                "canonical_clip_id": legacy.canonical_clip_id(clip_id),
                "speaker": clip_id.split("/", 1)[0],
                "frames": canonical_row["frames"],
                "canonical_row_sha256": gate.canonical_json_sha256(
                    canonical_row
                ),
                "audio_row_sha256": gate.canonical_json_sha256(audio_row),
            }
        )
    return _with_payload_sha(
        {
            "format": gate.SUBSET_FORMAT,
            "payload_hash_algorithm": gate.PAYLOAD_HASH_ALGORITHM,
            "status": "frozen",
            "split": "val",
            "test_visible": False,
            "parent_authority": {
                "val_inputs_receipt": dict(val_inputs_artifact),
                "canonical_manifest": canonical,
                "canonical_lineage": lineage,
                "expected_clip_count": gate.EXPECTED_VAL_CLIPS,
                "global_index_start": gate.EXPECTED_VAL_GLOBAL_INDEX_START,
                "global_index_stop_exclusive": (
                    gate.EXPECTED_VAL_GLOBAL_INDEX_STOP
                ),
            },
            "selection_algorithm": "full_frozen_val_canonical_order_v1",
            "rows": rows,
        }
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    inference._reject_path(args.output_root, "replication authority root")
    if os.path.lexists(args.output_root):
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    parent = args.output_root.parent.resolve(strict=True)
    if args.output_root != parent / args.output_root.name:
        raise FinalWinnerReplicationError("authority root must be canonical")
    preflight_artifact, preflight = inference._preflight_artifact(
        args.preflight,
        args.expected_preflight_sha256,
    )
    val_inputs, pipeline, canonical_rows, audio_by_id = (
        inference._load_preflight_children(preflight)
    )
    bundle = _candidate_bundle_from_preflight(preflight, pipeline)
    selection_path, selection_payload, selection_sha = legacy._verified_json(
        args.winner_selection,
        args.expected_winner_selection_sha256,
        "final-winner validation selection",
    )
    _reject_tree(selection_payload, "final-winner validation selection")
    selected = selection_payload.get("selected")
    checkpoint = selected.get("candidate_checkpoint") if isinstance(
        selected, dict
    ) else None
    if not isinstance(checkpoint, dict):
        raise FinalWinnerReplicationError("winner selection lacks checkpoint")
    validated_winner = winner.validate_test_winner(
        selection_path=selection_path,
        expected_selection_sha256=selection_sha,
        checkpoint_path=Path(checkpoint["path"]),
        expected_checkpoint_sha256=checkpoint["sha256"],
        candidate_bundle=bundle,
    )
    epoch = validated_winner["selected_epoch"]
    selected_checkpoint = validated_winner["selected_checkpoint"]
    if preflight["candidate_bundle"]["candidates"][str(epoch)] != (
        selected_checkpoint
    ):
        raise FinalWinnerReplicationError(
            "validation winner differs from inference preflight candidate"
        )
    val_inputs_artifact = _artifact(
        Path(preflight["val_inputs_receipt"]["path"]),
        "frozen validation inputs",
        expected_sha256=preflight["val_inputs_receipt"]["sha256"],
        payload_sha256=preflight["val_inputs_receipt"][
            "receipt_payload_sha256"
        ],
    )
    pipeline_artifact = _artifact(
        Path(preflight["pipeline_receipt"]["path"]),
        "frozen validation pipeline",
        expected_sha256=preflight["pipeline_receipt"]["sha256"],
        payload_sha256=preflight["pipeline_receipt"][
            "receipt_payload_sha256"
        ],
    )
    preflight_full = _artifact(
        Path(preflight_artifact["path"]),
        "validation inference preflight",
        expected_sha256=preflight_artifact["sha256"],
        payload_sha256=preflight_artifact["receipt_payload_sha256"],
    )
    selection_artifact = _artifact(
        selection_path,
        "final-winner validation selection",
        expected_sha256=selection_sha,
        payload_sha256=selection_payload["receipt_payload_sha256"],
    )
    source = _source_closure(
        args.repo_root,
        pipeline,
        expected_commit=args.source_commit,
        expected_tree=args.source_tree,
    )
    if source["static_randomness_scan"]["matches"]:
        raise FinalWinnerReplicationError(
            "neutral inference callable closure contains random operations"
        )
    models = _model_bundle(selected_checkpoint, pipeline)
    subset = _subset_payload(
        val_inputs_artifact=val_inputs_artifact,
        val_inputs=val_inputs,
        canonical_rows=canonical_rows,
        audio_by_id=audio_by_id,
    )
    stage = parent / f".{args.output_root.name}.partial-{os.getpid()}"
    inference._reject_path(stage, "replication authority staging root")
    stage.mkdir()
    final_subset = args.output_root / "subset-manifest.json"
    try:
        stage_subset = stage / final_subset.name
        _atomic_json_new(stage_subset, subset)
        subset_artifact = {
            "path": str(final_subset),
            "sha256": hashlib.sha256(stage_subset.read_bytes()).hexdigest(),
            "bytes": stage_subset.stat().st_size,
        }
        audio_artifacts = [
            _artifact(
                Path(item["path"]),
                "frozen validation audio manifest",
                expected_sha256=item["sha256"],
            )
            for item in val_inputs["audio_manifests"]
        ]
        authority = _with_payload_sha(
            {
                "format": AUTHORITY_FORMAT,
                "status": "frozen",
                "split": "val",
                "test_visible": False,
                "scope": "final_winner",
                "coverage_mode": "full_frozen_val_1715",
                "seeds": list(SEEDS),
                "num_shards": NUM_SHARDS,
                "assignment": ASSIGNMENT,
                "preflight": preflight_full,
                "winner_selection": selection_artifact,
                "winner_authorization": validated_winner,
                "selected_epoch": epoch,
                "selected_checkpoint": selected_checkpoint,
                "val_inputs": val_inputs_artifact,
                "pipeline": pipeline_artifact,
                "canonical_manifest": subset[
                    "parent_authority"
                ]["canonical_manifest"],
                "canonical_lineage": subset[
                    "parent_authority"
                ]["canonical_lineage"],
                "audio_manifests": audio_artifacts,
                "audio_rows_sha256": gate.canonical_json_sha256(
                    [row["audio_row_sha256"] for row in subset["rows"]]
                ),
                "subset_manifest": subset_artifact,
                "source_closure": source,
                "model_bundle": models,
            }
        )
        _reject_tree(authority, "replication authority")
        stage_authority = stage / "replication-authority.json"
        _atomic_json_new(stage_authority, authority)
        inference._publish_directory(stage, args.output_root)
    except BaseException:
        if stage.exists():
            import shutil

            shutil.rmtree(stage)
        raise
    artifact = _artifact(
        args.output_root / "replication-authority.json",
        "replication authority",
        payload_sha256=authority["receipt_payload_sha256"],
    )
    _load_authority(Path(artifact["path"]), artifact["sha256"])
    return artifact


def _load_authority(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _strict_json_artifact(
        path,
        expected_sha256,
        "final-winner replication authority",
    )
    expected_keys = {
        "format",
        "status",
        "split",
        "test_visible",
        "scope",
        "coverage_mode",
        "seeds",
        "num_shards",
        "assignment",
        "preflight",
        "winner_selection",
        "winner_authorization",
        "selected_epoch",
        "selected_checkpoint",
        "val_inputs",
        "pipeline",
        "canonical_manifest",
        "canonical_lineage",
        "audio_manifests",
        "audio_rows_sha256",
        "subset_manifest",
        "source_closure",
        "model_bundle",
        "receipt_payload_sha256",
    }
    _exact_mapping(value, expected_keys, "replication authority")
    if (
        value["format"] != AUTHORITY_FORMAT
        or value["status"] != "frozen"
        or value["split"] != "val"
        or value["test_visible"] is not False
        or value["scope"] != "final_winner"
        or value["coverage_mode"] != "full_frozen_val_1715"
        or value["seeds"] != list(SEEDS)
        or value["num_shards"] != NUM_SHARDS
        or value["assignment"] != ASSIGNMENT
    ):
        raise FinalWinnerReplicationError("replication authority is not formal")
    for key in (
        "preflight",
        "winner_selection",
        "val_inputs",
        "pipeline",
        "canonical_lineage",
    ):
        expected = value[key]
        observed = _artifact(
            Path(expected["path"]),
            f"authority {key}",
            expected_sha256=expected["sha256"],
            payload_sha256=expected["receipt_payload_sha256"],
        )
        if observed != expected:
            raise FinalWinnerReplicationError(f"authority {key} changed")
    canonical = value["canonical_manifest"]
    if _artifact(
        Path(canonical["path"]),
        "authority canonical manifest",
        expected_sha256=canonical["sha256"],
    ) != canonical:
        raise FinalWinnerReplicationError("canonical manifest changed")
    audio = [
        _artifact(
            Path(item["path"]),
            "authority audio manifest",
            expected_sha256=item["sha256"],
        )
        for item in value["audio_manifests"]
    ]
    if audio != value["audio_manifests"] or len(audio) != NUM_SHARDS:
        raise FinalWinnerReplicationError("audio manifest pins changed")
    gate._validate_source_closure(value["source_closure"])
    gate._validate_model_bundle(value["model_bundle"])
    subset_artifact, subset = gate._validate_subset_manifest(
        value["subset_manifest"]
    )
    if subset_artifact != value["subset_manifest"]:
        raise FinalWinnerReplicationError("subset manifest changed")
    if gate.canonical_json_sha256(
        [row["audio_row_sha256"] for row in subset]
    ) != value["audio_rows_sha256"]:
        raise FinalWinnerReplicationError("audio-row authority changed")
    preflight_artifact, preflight = inference._preflight_artifact(
        Path(value["preflight"]["path"]),
        value["preflight"]["sha256"],
    )
    if any(
        preflight_artifact.get(key) != value["preflight"].get(key)
        for key in ("path", "sha256", "receipt_payload_sha256")
    ):
        raise FinalWinnerReplicationError("preflight authority changed")
    val_inputs, pipeline, canonical_rows, audio_by_id = (
        inference._load_preflight_children(preflight)
    )
    recomputed_subset = _subset_payload(
        val_inputs_artifact=value["val_inputs"],
        val_inputs=val_inputs,
        canonical_rows=canonical_rows,
        audio_by_id=audio_by_id,
    )
    subset_payload = gate._strict_json_bytes(
        Path(value["subset_manifest"]["path"]).read_bytes(),
        "replication subset manifest",
    )
    if recomputed_subset != subset_payload:
        raise FinalWinnerReplicationError(
            "canonical/audio subset differs from fresh val replay"
        )
    current_source = _source_closure(
        PROJECT_ROOT,
        pipeline,
        expected_commit=value["source_closure"]["commit"],
        expected_tree=value["source_closure"]["tree"],
    )
    if current_source != value["source_closure"]:
        raise FinalWinnerReplicationError("replication source changed")
    selected_candidate = preflight["candidate_bundle"]["candidates"].get(
        str(value["selected_epoch"])
    )
    if selected_candidate != value["selected_checkpoint"]:
        raise FinalWinnerReplicationError("selected Base checkpoint changed")
    authorization = value["winner_authorization"]
    if (
        not isinstance(authorization, dict)
        or authorization.get("status") != "validated"
        or authorization.get("selection_split") != "val"
        or authorization.get("test_visible_during_selection") is not False
        or authorization.get("authorized_test_evaluations") != 0
        or authorization.get("selected_epoch") != value["selected_epoch"]
        or authorization.get("selected_checkpoint")
        != value["selected_checkpoint"]
        or value["model_bundle"]["checkpoints"]["base"]
        != value["selected_checkpoint"]
    ):
        raise FinalWinnerReplicationError("winner authorization changed")
    _reject_tree(value, "replication authority")
    return artifact, value


def _bytes_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rng_state(device: str) -> dict[str, str]:
    import torch

    numpy_state = np.random.get_state()
    numpy_payload = b"\0".join(
        (
            str(numpy_state[0]).encode("ascii"),
            np.asarray(numpy_state[1]).tobytes(order="C"),
            str(numpy_state[2]).encode("ascii"),
            str(numpy_state[3]).encode("ascii"),
            repr(numpy_state[4]).encode("ascii"),
        )
    )
    torch_cpu = torch.get_rng_state().detach().cpu().contiguous().numpy()
    torch_cuda = (
        torch.cuda.get_rng_state(torch.device(device))
        .detach()
        .cpu()
        .contiguous()
        .numpy()
    )
    values = {
        "python": _bytes_sha(repr(random.getstate()).encode("utf-8")),
        "numpy": _bytes_sha(numpy_payload),
        "torch_cpu": _bytes_sha(torch_cpu.tobytes(order="C")),
        "torch_cuda": _bytes_sha(torch_cuda.tobytes(order="C")),
    }
    return {
        **values,
        "rng_state_sha256": gate.canonical_json_sha256(values),
    }


def _process_identity(proc_root: Path = Path("/proc")) -> str:
    stat_path = proc_root / "self" / "stat"
    try:
        line = stat_path.read_text(encoding="utf-8")
    except OSError as error:
        raise FinalWinnerReplicationError("Linux /proc identity unavailable") from error
    if ") " not in line:
        raise FinalWinnerReplicationError("Linux /proc identity malformed")
    fields = line.rsplit(") ", 1)[1].split()
    if len(fields) < 20 or not fields[19].isdigit() or fields[0] == "Z":
        raise FinalWinnerReplicationError("Linux process identity is unsafe")
    return f"{socket.gethostname()}:pid={os.getpid()}:start={fields[19]}"


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.seed not in SEEDS
        or args.num_shards != NUM_SHARDS
        or not 0 <= args.shard_id < NUM_SHARDS
        or args.device != f"cuda:{args.shard_id}"
    ):
        raise FinalWinnerReplicationError(
            "replication shard requires seeds 0/15 and exact cuda shard 0..7"
        )
    authority_artifact, authority = _load_authority(
        args.authority,
        args.expected_authority_sha256,
    )
    seed_root = inference._directory(
        args.seed_root,
        "replication seed root",
        create=True,
    )
    original_loader = inference._load_pinned_helper
    captures: list[dict[str, Any]] = []

    def instrumented_loader(pipeline: Mapping[str, Any]) -> Any:
        helper = original_loader(pipeline)
        original_infer = helper._infer_clip

        def instrumented_infer(**kwargs: Any) -> Any:
            before = _rng_state(args.device)
            result = original_infer(**kwargs)
            after = _rng_state(args.device)
            if before != after:
                raise FinalWinnerReplicationError(
                    "SemTalk forward consumed RNG state"
                )
            captures.append(
                {
                    "before": before,
                    "after": after,
                    "seed_consumed": False,
                }
            )
            return result

        helper._infer_clip = instrumented_infer
        return helper

    inference._load_pinned_helper = instrumented_loader
    inference_root = seed_root / "inference"
    try:
        inference.run_shard(
            SimpleNamespace(
                split="val",
                preflight=Path(authority["preflight"]["path"]),
                expected_preflight_sha256=authority["preflight"]["sha256"],
                epoch=authority["selected_epoch"],
                output_root=inference_root,
                num_shards=NUM_SHARDS,
                shard_id=args.shard_id,
                device=args.device,
                seed=args.seed,
                progress_every=args.progress_every,
            )
        )
    finally:
        inference._load_pinned_helper = original_loader
    preflight_artifact, preflight = inference._preflight_artifact(
        Path(authority["preflight"]["path"]),
        authority["preflight"]["sha256"],
    )
    _val, pipeline, canonical_rows, _audio = inference._load_preflight_children(
        preflight
    )
    candidate = preflight["candidate_bundle"]["candidates"][
        str(authority["selected_epoch"])
    ]
    inference_receipt, rows = inference._validate_shard(
        output_root=inference_root,
        shard_id=args.shard_id,
        num_shards=NUM_SHARDS,
        epoch=authority["selected_epoch"],
        candidate=candidate,
        preflight_artifact=preflight_artifact,
        preflight=preflight,
        pipeline=pipeline,
        canonical_rows=canonical_rows,
    )
    if len(captures) != len(rows):
        raise FinalWinnerReplicationError("instrumented forward count changed")
    subset_payload = gate._strict_json_bytes(
        Path(authority["subset_manifest"]["path"]).read_bytes(),
        "replication subset manifest",
    )
    subset_rows = subset_payload["rows"]
    clips = []
    for row, rng in zip(rows, captures):
        position = row["canonical_position"]
        subset = subset_rows[position]
        if (
            row["canonical_clip_id"] != subset["canonical_clip_id"]
            or row["frames"] != subset["frames"]
        ):
            raise FinalWinnerReplicationError("instrumented clip identity changed")
        clips.append(
            {
                "gate_position": position,
                "canonical_clip_id": row["canonical_clip_id"],
                "frames": row["frames"],
                "input_binding": {
                    "canonical_row_sha256": subset[
                        "canonical_row_sha256"
                    ],
                    "audio_row_sha256": subset["audio_row_sha256"],
                },
                "prediction": dict(row["prediction"]),
                "rng": rng,
            }
        )
    inference_receipt_path = (
        inference_root
        / inference.SHARDS_DIRECTORY
        / inference._shard_name(args.shard_id, NUM_SHARDS)
        / inference.SHARD_RECEIPT_FILENAME
    )
    inference_receipt_artifact = _artifact(
        inference_receipt_path,
        "instrumented inference shard receipt",
        payload_sha256=inference_receipt["receipt_payload_sha256"],
    )
    result = _with_payload_sha(
        {
            "format": SHARD_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "seed": args.seed,
            "assignment": ASSIGNMENT,
            "shard_id": args.shard_id,
            "num_shards": NUM_SHARDS,
            "authority": authority_artifact,
            "selected_epoch": authority["selected_epoch"],
            "selected_checkpoint": authority["selected_checkpoint"],
            "process_identity": _process_identity(),
            "inference_shard_receipt": inference_receipt_artifact,
            "clip_count": len(clips),
            "clips": clips,
            "exact_once": True,
            "finite": True,
            "rng_unchanged": True,
        }
    )
    output = (
        seed_root
        / "replication-shards"
        / f"shard-{args.shard_id:05d}-of-{NUM_SHARDS:05d}.json"
    )
    _atomic_json_new(output, result)
    return _artifact(
        output,
        "replication shard receipt",
        payload_sha256=result["receipt_payload_sha256"],
    )


def _load_shard(
    path: Path,
    *,
    authority_artifact: Mapping[str, Any],
    authority: Mapping[str, Any],
    seed: int,
    shard_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _strict_json_artifact(
        path,
        gate.sha256_file(path),
        f"seed {seed} shard {shard_id}",
    )
    _exact_mapping(
        value,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "seed",
            "assignment",
            "shard_id",
            "num_shards",
            "authority",
            "selected_epoch",
            "selected_checkpoint",
            "process_identity",
            "inference_shard_receipt",
            "clip_count",
            "clips",
            "exact_once",
            "finite",
            "rng_unchanged",
            "receipt_payload_sha256",
        },
        "replication shard receipt",
    )
    if (
        value["format"] != SHARD_FORMAT
        or value["status"] != "complete"
        or value["split"] != "val"
        or value["test_visible"] is not False
        or value["seed"] != seed
        or value["assignment"] != ASSIGNMENT
        or value["shard_id"] != shard_id
        or value["num_shards"] != NUM_SHARDS
        or value["authority"] != authority_artifact
        or value["selected_epoch"] != authority["selected_epoch"]
        or value["selected_checkpoint"] != authority["selected_checkpoint"]
        or value["exact_once"] is not True
        or value["finite"] is not True
        or value["rng_unchanged"] is not True
        or not isinstance(value["process_identity"], str)
        or not value["process_identity"]
    ):
        raise FinalWinnerReplicationError("replication shard identity changed")
    inference_artifact = value["inference_shard_receipt"]
    if not isinstance(inference_artifact, dict):
        raise FinalWinnerReplicationError("inference shard receipt is absent")
    if _artifact(
        Path(inference_artifact["path"]),
        "bound inference shard receipt",
        expected_sha256=inference_artifact["sha256"],
        payload_sha256=inference_artifact["receipt_payload_sha256"],
    ) != inference_artifact:
        raise FinalWinnerReplicationError("inference shard receipt changed")
    clips = value["clips"]
    expected_positions = list(range(shard_id, gate.EXPECTED_VAL_CLIPS, NUM_SHARDS))
    if (
        not isinstance(clips, list)
        or value["clip_count"] != len(expected_positions)
        or [clip.get("gate_position") for clip in clips] != expected_positions
    ):
        raise FinalWinnerReplicationError("replication shard coverage changed")
    for clip in clips:
        rng = clip.get("rng")
        if (
            not isinstance(rng, dict)
            or rng.get("seed_consumed") is not False
            or rng.get("before") != rng.get("after")
        ):
            raise FinalWinnerReplicationError("replication shard consumed RNG")
        prediction = clip.get("prediction")
        if not isinstance(prediction, dict):
            raise FinalWinnerReplicationError("replication prediction missing")
        if _artifact(
            Path(prediction["path"]),
            "replication prediction",
            expected_sha256=prediction["sha256"],
        ) != prediction:
            raise FinalWinnerReplicationError("replication prediction changed")
    _reject_tree(value, "replication shard")
    return artifact, value


def finalize_seed(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in SEEDS:
        raise FinalWinnerReplicationError("seed receipt must use 0 or 15")
    authority_artifact, authority = _load_authority(
        args.authority,
        args.expected_authority_sha256,
    )
    seed_root = inference._directory(args.seed_root, "replication seed root")
    preflight_artifact, preflight = inference._preflight_artifact(
        Path(authority["preflight"]["path"]),
        authority["preflight"]["sha256"],
    )
    _val, pipeline, canonical_rows, _audio = inference._load_preflight_children(
        preflight
    )
    candidate = preflight["candidate_bundle"]["candidates"][
        str(authority["selected_epoch"])
    ]
    shard_artifacts = []
    shards = []
    model_receipt_digests: set[str] = set()
    runtime_receipt_digests: set[str] = set()
    for shard_id in range(NUM_SHARDS):
        path = (
            seed_root
            / "replication-shards"
            / f"shard-{shard_id:05d}-of-{NUM_SHARDS:05d}.json"
        )
        artifact, value = _load_shard(
            path,
            authority_artifact=authority_artifact,
            authority=authority,
            seed=args.seed,
            shard_id=shard_id,
        )
        replayed_inference, replayed_rows = inference._validate_shard(
            output_root=seed_root / "inference",
            shard_id=shard_id,
            num_shards=NUM_SHARDS,
            epoch=authority["selected_epoch"],
            candidate=candidate,
            preflight_artifact=preflight_artifact,
            preflight=preflight,
            pipeline=pipeline,
            canonical_rows=canonical_rows,
        )
        expected_inference = value["inference_shard_receipt"]
        replayed_path = Path(expected_inference["path"])
        replayed_artifact = _artifact(
            replayed_path,
            "replayed inference shard receipt",
            expected_sha256=expected_inference["sha256"],
            payload_sha256=replayed_inference["receipt_payload_sha256"],
        )
        if (
            replayed_artifact != expected_inference
            or len(replayed_rows) != value["clip_count"]
        ):
            raise FinalWinnerReplicationError(
                "instrumented shard differs from fresh inference replay"
            )
        model_receipt_digests.add(replayed_inference["model_receipts_sha256"])
        runtime_receipt_digests.add(
            replayed_inference["runtime_contract_sha256"]
        )
        shard_artifacts.append(artifact)
        shards.append(value)
    if len(model_receipt_digests) != 1 or len(runtime_receipt_digests) != 1:
        raise FinalWinnerReplicationError(
            "seed shards disagree on loaded models or runtime"
        )
    identities = [value["process_identity"] for value in shards]
    if len(set(identities)) != NUM_SHARDS:
        raise FinalWinnerReplicationError("shards did not run in 8 processes")
    by_position: dict[int, dict[str, Any]] = {}
    for value in shards:
        for clip in value["clips"]:
            position = clip["gate_position"]
            if position in by_position:
                raise FinalWinnerReplicationError("duplicate replication clip")
            by_position[position] = clip
    if set(by_position) != set(range(gate.EXPECTED_VAL_CLIPS)):
        raise FinalWinnerReplicationError("replication clips are not exact-once")
    process_identity = _process_identity()
    run_nonce = gate.canonical_json_sha256(
        {
            "seed": args.seed,
            "process_identity": process_identity,
            "shard_receipts": shard_artifacts,
        }
    )
    receipt = _with_payload_sha(
        {
            "format": gate.SEED_RUN_FORMAT,
            "payload_hash_algorithm": gate.PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "seed": args.seed,
            "coverage_mode": "full_frozen_val_1715",
            "run_nonce": run_nonce,
            "process_identity": process_identity,
            "independent_process": True,
            "subset_manifest": authority["subset_manifest"],
            "source_closure": authority["source_closure"],
            "model_bundle": authority["model_bundle"],
            "runtime": {
                "torch_inference_mode": True,
                "deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "models_eval": True,
                "requires_grad_false": True,
                "discrete_decoding": "logits.argmax(dim=2)",
                "devices": [f"cuda:{index}" for index in range(NUM_SHARDS)],
            },
            "clips": [by_position[index] for index in range(gate.EXPECTED_VAL_CLIPS)],
        }
    )
    _atomic_json_new(args.output_json, receipt)
    digest = gate.sha256_file(args.output_json)
    gate._validate_seed_run(args.output_json, digest)
    return _artifact(
        args.output_json,
        f"seed {args.seed} receipt",
        payload_sha256=receipt["receipt_payload_sha256"],
    )


def complete(args: argparse.Namespace) -> dict[str, Any]:
    authority_artifact, authority = _load_authority(
        args.authority,
        args.expected_authority_sha256,
    )
    gate_artifact, gate_value = gate.load_gate(
        args.gate_json,
        args.expected_gate_sha256,
        expected_scope="final_winner",
    )
    preflight = inference._verified_json(
        Path(authority["preflight"]["path"]),
        authority["preflight"]["sha256"],
        "completion validation preflight",
    )
    _val, pipeline, _canonical, _audio = inference._load_preflight_children(
        preflight
    )
    bundle = _candidate_bundle_from_preflight(preflight, pipeline)
    replayed_winner = winner.validate_test_winner(
        selection_path=Path(authority["winner_selection"]["path"]),
        expected_selection_sha256=authority["winner_selection"]["sha256"],
        checkpoint_path=Path(authority["selected_checkpoint"]["path"]),
        expected_checkpoint_sha256=authority["selected_checkpoint"]["sha256"],
        candidate_bundle=bundle,
    )
    if (
        gate_value["subset_manifest"] != authority["subset_manifest"]
        or gate_value["source_closure"] != authority["source_closure"]
        or gate_value["model_bundle"] != authority["model_bundle"]
        or gate_value["proof"]["clip_count"] != gate.EXPECTED_VAL_CLIPS
        or gate_value["proof"]["runtime_no_rng_consumption"] is not True
        or gate_value["proof"]["byte_exact"] is not True
        or gate_value["proof"]["array_exact"] is not True
        or replayed_winner != authority["winner_authorization"]
    ):
        raise FinalWinnerReplicationError(
            "final replication gate differs from winner authority"
        )
    runner_sha = legacy.require_sha256(
        args.guarded_runner_sha256,
        "guarded runner SHA-256",
    )
    result = _with_payload_sha(
        {
            "format": COMPLETION_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "scope": "final_winner",
            "replication_authority": authority_artifact,
            "winner_selection": authority["winner_selection"],
            "selected_epoch": authority["selected_epoch"],
            "selected_checkpoint": authority["selected_checkpoint"],
            "canonical_manifest": authority["canonical_manifest"],
            "canonical_lineage": authority["canonical_lineage"],
            "audio_manifests": authority["audio_manifests"],
            "audio_rows_sha256": authority["audio_rows_sha256"],
            "seed_runs": gate_value["seed_runs"],
            "seed_processes": 2,
            "shards_per_seed": NUM_SHARDS,
            "total_shard_jobs": len(SEEDS) * NUM_SHARDS,
            "exact_once": True,
            "rng_unchanged_per_clip": True,
            "prediction_bytes_equal": True,
            "prediction_fields_equal": True,
            "guarded_runner": {
                "path": "/tmp/globaldiff_guarded_runner.py",
                "sha256": runner_sha,
                "gpus": list(range(NUM_SHARDS)),
            },
            "replication_gate": gate_artifact,
        }
    )
    _reject_tree(result, "replication completion")
    _atomic_json_new(args.output_json, result)
    return _artifact(
        args.output_json,
        "replication completion",
        payload_sha256=result["receipt_payload_sha256"],
    )


def _sha(value: str) -> str:
    return legacy.require_sha256(value, "SHA-256 argument")


def _git_oid(value: str) -> str:
    return legacy.require_git_oid(value, "Git object ID argument")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", allow_abbrev=False)
    prepare_parser.add_argument("--repo-root", type=Path, required=True)
    prepare_parser.add_argument("--preflight", type=Path, required=True)
    prepare_parser.add_argument(
        "--expected-preflight-sha256", type=_sha, required=True
    )
    prepare_parser.add_argument(
        "--winner-selection", type=Path, required=True
    )
    prepare_parser.add_argument(
        "--expected-winner-selection-sha256", type=_sha, required=True
    )
    prepare_parser.add_argument("--source-commit", type=_git_oid, required=True)
    prepare_parser.add_argument("--source-tree", type=_git_oid, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)

    shard_parser = commands.add_parser("shard", allow_abbrev=False)
    shard_parser.add_argument("--authority", type=Path, required=True)
    shard_parser.add_argument(
        "--expected-authority-sha256", type=_sha, required=True
    )
    shard_parser.add_argument("--seed-root", type=Path, required=True)
    shard_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    shard_parser.add_argument("--num-shards", type=int, required=True)
    shard_parser.add_argument("--shard-id", type=int, required=True)
    shard_parser.add_argument("--device", required=True)
    shard_parser.add_argument("--progress-every", type=int, default=20)

    seed_parser = commands.add_parser("finalize-seed", allow_abbrev=False)
    seed_parser.add_argument("--authority", type=Path, required=True)
    seed_parser.add_argument(
        "--expected-authority-sha256", type=_sha, required=True
    )
    seed_parser.add_argument("--seed-root", type=Path, required=True)
    seed_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    seed_parser.add_argument("--output-json", type=Path, required=True)

    complete_parser = commands.add_parser("complete", allow_abbrev=False)
    complete_parser.add_argument("--authority", type=Path, required=True)
    complete_parser.add_argument(
        "--expected-authority-sha256", type=_sha, required=True
    )
    complete_parser.add_argument("--gate-json", type=Path, required=True)
    complete_parser.add_argument(
        "--expected-gate-sha256", type=_sha, required=True
    )
    complete_parser.add_argument(
        "--guarded-runner-sha256", type=_sha, required=True
    )
    complete_parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _reject_tree(vars(args), "command arguments")
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "shard":
        result = run_shard(args)
    elif args.command == "finalize-seed":
        result = finalize_seed(args)
    elif args.command == "complete":
        result = complete(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
