#!/usr/bin/env python3
"""Freeze and validate the inputs authorized for one SemTalk Base SHOW test.

This module is intentionally neutral.  It does not launch inference, choose a
metric, select a checkpoint, or import any evaluator.  The validation-only
orchestrator calls :func:`build_test_authority` after it has frozen the Base
winner.  A test consumer externally pins the resulting file's SHA-256, byte
count, and payload SHA-256 before any output is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from typing import Any, Mapping, Sequence


FORMAT = "semtalk_show_base_final_test_authority_v2"
INPUTS_FORMAT = "semtalk_show_base_final_authority_inputs_v2"
PAYLOAD_HASH_ALGORITHM = "canonical_json_utf8_sorted_compact_newline_v1"
BASE_SELECTION_METRIC = "validation.diffsheg.metrics.fgd"
BASE_SELECTION_PROTOCOL = "diffsheg_show_validation_fgd_v1"
ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
OFFICIAL_BASELINE_COMMIT = "806b008c97bf51fce203e54109e4c22325253618"
CHECKPOINT_STAGES = ("base", "face", "hands", "upper", "lower", "global")
REPRESENTATION_STAGES = ("face", "hands", "upper", "lower", "global")
INITIAL_PREREQUISITE_SELECTION_FORMAT = (
    "semtalk_show_prerequisite_val_selection_v1"
)
PER_STAGE_PREREQUISITE_SELECTION_FORMAT = (
    "semtalk_show_prerequisite_val_selection_v2"
)
TEST_GLOBAL_START = 15_402
TEST_GLOBAL_STOP = 17_110
TEST_CLIPS = TEST_GLOBAL_STOP - TEST_GLOBAL_START
NUM_SHARDS = 8
MIN_FRAMES_EXCLUSIVE = 60
POSE_FPS = 30
SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
FINAL_METRIC_EVENT = {
    "format": "semtalk_show_combined_final_metric_event_v1",
    "authorized_events": 1,
    "generation_passes": 1,
    "shared_prediction_bundle": True,
    "single_claim_required": True,
    "claim_consumed_before_metrics": True,
    "failure_consumes_claim": True,
    "retry_allowed": False,
    "all_suites_required": True,
    "metric_suites": [
        {
            "name": "paspa_diffsheg_show_seven",
            "metrics": [
                "fmd",
                "fed",
                "expression_diversity",
                "fgd",
                "ba",
                "pcm",
                "gesture_diversity",
            ],
        },
        {
            "name": "talkshow_show_body_face",
            "body_protocols": ["released2", "paper16"],
            "body_metrics": ["FGD", "Variation", "BC"],
            "face_protocol": "released_face_sample0",
            "face_metrics": [
                "jaw_l1",
                "landmark_l1",
                "LVD",
                "face_l2_combined",
            ],
            "rs": "N/A/unreleased",
        },
    ],
}
FINAL_TEST_POLICY = {
    "test_evaluations": 1,
    "test_feedback_into_selection": False,
    "final_metric_event": FINAL_METRIC_EVENT,
}
ARTIFACT_KEYS = {"path", "sha256", "bytes"}
SOURCE_INPUT_KEYS = {
    "source_root",
    "origin",
    "commit",
    "tree",
    "clean",
    "entrypoint",
    "entrypoint_sha256",
}
SOURCE_DERIVED_KEYS = {
    "official_baseline_commit",
    "official_baseline_ancestor",
    "pinned_origin",
    "pinned_commit",
    "pinned_tree",
    "publication_remote_ref",
    "publication_live_main_check_required",
    "detached",
    "local_branches_at_commit",
}
PRODUCER_SOURCE_DERIVED_KEYS = {
    "source_root",
    "pinned_inference_commit",
    "pinned_inference_tree",
    "inference_source_ancestor",
    "official_baseline_ancestor",
    "detached",
    "local_branches_at_commit",
}
CANONICAL_ROOT_RECEIPT_KEYS = {
    "manifest",
    "summary",
    "lineage",
    "source_commit",
    "source_tree",
}
FORBIDDEN_IDENTITIES = (
    "speaker2",
    "sparse",
    "semgate",
    "globaldiff",
)
GENERATOR_SEMANTIC_KEYS = {
    "generator",
    "generator_identity",
    "generator_name",
    "generator_source",
    "model",
    "model_identity",
    "model_name",
    "model_source",
    "checkpoint",
    "checkpoint_path",
    "checkpoint_source",
    "candidate_checkpoint",
    "selected_checkpoint",
    "fixed_checkpoints",
}
EVALUATOR_ASSET_KEYS = {
    "evaluator",
    "evaluator_asset",
    "feature_extractor",
    "metric_asset",
    "smplx",
    "smplx_asset",
    "talkshow",
    "talkshow_asset",
}
BASE_LONG_ARTIFACT_ROLES = ("manifest", "status", "frozen_inputs")
BASE_LONG_BUNDLE_KEYS = {
    "artifacts",
    "manifest",
    "status",
    "frozen_inputs",
    "producer_source",
    "selected_prerequisite_sha256",
    "selected_topology",
    "selected_dataset",
    "candidate_epochs",
    "updates_per_epoch",
    "candidates",
}
BASE_LONG_DATASET_KEYS = {
    "format",
    "lmdb",
    "summary",
    "summary_sha256",
    "lineage",
    "lineage_sha256",
    "entries",
    "train_clips",
    "split",
    "test_visible",
    "data_mdb_sha256",
    "lock_mdb_sha256",
    "prerequisite_selection",
    "selected_prerequisite_sha256",
    "lmdb_binding_scope",
    "node_lmdb_inode_bindings",
}
_CONTROL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "gate_released_all_speakers_on_show": (),
    "prerequisite_val_contract": (),
    "merge_prerequisite_val_shards": (
        "gate_released_all_speakers_on_show",
        "prerequisite_val_contract",
    ),
    "select_prerequisite_candidates": (
        "gate_released_all_speakers_on_show",
        "prerequisite_val_contract",
        "merge_prerequisite_val_shards",
    ),
    "selected_prerequisites": (
        "prerequisite_val_contract",
        "merge_prerequisite_val_shards",
    ),
    "decide_prerequisite_continuation": (
        "prerequisite_val_contract",
        "selected_prerequisites",
    ),
    "prerequisite_boundary_state": (),
    "prerequisite_continuation_wave": (
        "prerequisite_val_contract",
        "merge_prerequisite_val_shards",
        "selected_prerequisites",
        "decide_prerequisite_continuation",
        "prerequisite_boundary_state",
    ),
    "gate_task_space_on_show_v2": (
        "gate_released_all_speakers_on_show",
    ),
    "talkshow_base_val_contract": (
        "gate_released_all_speakers_on_show",
        "gate_task_space_on_show_v2",
        "prerequisite_val_contract",
        "merge_prerequisite_val_shards",
        "selected_prerequisites",
    ),
    "select_base_official_adapt": (
        "gate_task_space_on_show_v2",
        "selected_prerequisites",
    ),
    "base_long_val_contract": (
        "gate_released_all_speakers_on_show",
        "gate_task_space_on_show_v2",
        "prerequisite_val_contract",
        "merge_prerequisite_val_shards",
        "selected_prerequisites",
        "select_base_official_adapt",
    ),
    "select_base_official_adapt_long": (
        "select_base_official_adapt",
        "base_long_val_contract",
    ),
    "validate_base_long_test_winner": (
        "select_base_official_adapt",
        "base_long_val_contract",
        "select_base_official_adapt_long",
    ),
}


class BaseFinalAuthorityError(RuntimeError):
    """Raised when the pre-inference Base authority is not immutable."""


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


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaseFinalAuthorityError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _git_oid(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaseFinalAuthorityError(
            f"{label} must be a lowercase Git object ID"
        )
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BaseFinalAuthorityError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _canonical_regular_path(path: Any, label: str) -> Path:
    if type(path) is not str or not Path(path).is_absolute():
        raise BaseFinalAuthorityError(
            f"{label} path must be an absolute string"
        )
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BaseFinalAuthorityError(
            f"{label} must be a canonical regular file"
        ) from exc
    if str(candidate) != path or resolved != candidate:
        raise BaseFinalAuthorityError(
            f"{label} must be a canonical regular file"
        )
    return candidate


def _safe_file_snapshot(path: Any, label: str) -> tuple[Path, bytes]:
    """Read one canonical file through a no-symlink descriptor walk.

    The directory descriptors close the ancestor-symlink race; ``O_NOFOLLOW``
    closes the final-component race.  Stable pre/post ``fstat`` identities
    ensure the bytes were not changed while they were read.
    """

    candidate = _canonical_regular_path(path, label)
    parts = candidate.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise BaseFinalAuthorityError(
            f"{label} must be below the filesystem root"
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
        file_fd = os.open(
            parts[-1],
            file_flags,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise BaseFinalAuthorityError(
                f"{label} must be a regular non-symlink file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise BaseFinalAuthorityError(
                f"{label} changed while it was read"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise BaseFinalAuthorityError(
                f"{label} size changed while it was read"
            )
        return candidate, payload
    except BaseFinalAuthorityError:
        raise
    except OSError as exc:
        raise BaseFinalAuthorityError(
            f"cannot safely read {label}: {candidate}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _regular_file(path: Any, label: str) -> Path:
    candidate, _payload = _safe_file_snapshot(path, label)
    return candidate


def _verified_artifact(value: Any, label: str) -> tuple[dict[str, Any], bytes]:
    if type(value) is not dict or set(value) != ARTIFACT_KEYS:
        raise BaseFinalAuthorityError(f"{label} artifact schema mismatch")
    path, payload = _safe_file_snapshot(value["path"], label)
    digest = hashlib.sha256(payload).hexdigest()
    expected_digest = _sha256(value["sha256"], f"{label} SHA")
    expected_bytes = _exact_int(value["bytes"], f"{label} bytes", minimum=1)
    if digest != expected_digest or len(payload) != expected_bytes:
        raise BaseFinalAuthorityError(f"{label} artifact changed")
    return dict(value), payload


def _json_artifact(
    value: Any,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, payload = _verified_artifact(value, label)
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaseFinalAuthorityError(f"{label} is invalid JSON") from exc
    if type(decoded) is not dict:
        raise BaseFinalAuthorityError(f"{label} must contain a JSON object")
    return artifact, decoded


def _jsonl_artifact(
    value: Any,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact, payload = _verified_artifact(value, label)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BaseFinalAuthorityError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if type(row) is not dict:
            raise BaseFinalAuthorityError(
                f"{label} line {line_number} is not an object"
            )
        rows.append(row)
    if not rows:
        raise BaseFinalAuthorityError(f"{label} is empty")
    return artifact, rows


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaseFinalAuthorityError(
            f"cannot attest SemTalk source root {root}"
        ) from exc


def _validate_source(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != SOURCE_INPUT_KEYS:
        raise BaseFinalAuthorityError("inference source schema mismatch")
    if (
        value["origin"] != ORIGIN
        or value["clean"] is not True
        or value["entrypoint"]
        != "scripts/show_base/semtalk_base_inference_core.py"
    ):
        raise BaseFinalAuthorityError(
            "test authority is not SemTalk official Base inference"
        )
    root_value = value["source_root"]
    if type(root_value) is not str or not Path(root_value).is_absolute():
        raise BaseFinalAuthorityError("source_root must be absolute")
    root = Path(root_value).resolve()
    if str(root) != root_value or not root.is_dir() or root.is_symlink():
        raise BaseFinalAuthorityError("source_root is not canonical")
    commit = _git_oid(value["commit"], "source commit")
    tree = _git_oid(value["tree"], "source tree")
    local_branches = [
        line
        for line in _git(
            root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
        ).splitlines()
        if line
    ]
    detached = (
        subprocess.run(
            ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).returncode
        != 0
    )
    baseline_ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                OFFICIAL_BASELINE_COMMIT,
                "HEAD",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).returncode
        == 0
    )
    if (
        _git(root, "remote", "get-url", "origin") != ORIGIN
        or _git(root, "rev-parse", "HEAD^{commit}") != commit
        or _git(root, "rev-parse", "HEAD^{tree}") != tree
        or _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        != ""
        or not detached
        or local_branches
        or not baseline_ancestor
    ):
        raise BaseFinalAuthorityError(
            "SemTalk inference source is not the exact detached official "
            "baseline descendant with zero local branches"
        )
    entrypoint_value = str(root / value["entrypoint"])
    try:
        _entrypoint, entrypoint_payload = _safe_file_snapshot(
            entrypoint_value,
            "SemTalk official Base entrypoint",
        )
    except BaseFinalAuthorityError as exc:
        raise BaseFinalAuthorityError(
            "SemTalk official Base entrypoint changed"
        ) from exc
    if hashlib.sha256(entrypoint_payload).hexdigest() != _sha256(
        value["entrypoint_sha256"],
        "entrypoint SHA",
    ):
        raise BaseFinalAuthorityError(
            "SemTalk official Base entrypoint changed"
        )
    return {
        **dict(value),
        "official_baseline_commit": OFFICIAL_BASELINE_COMMIT,
        "official_baseline_ancestor": True,
        "pinned_origin": ORIGIN,
        "pinned_commit": commit,
        "pinned_tree": tree,
        "publication_remote_ref": "refs/heads/main",
        "publication_live_main_check_required": True,
        "detached": True,
        "local_branches_at_commit": [],
    }


def _reject_forbidden(
    value: Any,
    label: str,
    *,
    semantic_identity: bool = False,
) -> None:
    """Reject forbidden generator identities without banning evaluator assets.

    The released TalkSHOW evaluator and pinned SMPL-X assets may legitimately
    live under historical paths containing ``globaldiff``.  Only values whose
    schema keys identify a generator, model, or checkpoint are identity
    bearing.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = str(key).casefold()
            child_semantic = (
                semantic_identity
                or key_name in GENERATOR_SEMANTIC_KEYS
                or key_name.endswith("_checkpoint")
                or key_name.endswith("_checkpoint_path")
                or key_name.endswith("_checkpoint_source")
            )
            if (
                key_name in EVALUATOR_ASSET_KEYS
                or key_name.endswith("_evaluator")
                or key_name.endswith("_metric_asset")
                or key_name.endswith("_smplx_asset")
            ):
                child_semantic = False
            _reject_forbidden(
                child,
                f"{label}.{key}",
                semantic_identity=child_semantic,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(
                child,
                f"{label}[{index}]",
                semantic_identity=semantic_identity,
            )
    elif isinstance(value, str) and semantic_identity:
        normalized = value.casefold()
        if any(token in normalized for token in FORBIDDEN_IDENTITIES):
            raise BaseFinalAuthorityError(
                f"{label} contains forbidden generator identity"
            )


def _canonical_authority(
    manifest_receipt: Mapping[str, Any],
    summary_receipt: Mapping[str, Any],
    lineage_receipt: Mapping[str, Any],
    root_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest, rows = _jsonl_artifact(
        manifest_receipt,
        "canonical manifest",
    )
    summary, summary_payload = _json_artifact(
        summary_receipt,
        "canonical summary",
    )
    lineage, lineage_payload = _json_artifact(
        lineage_receipt,
        "canonical lineage",
    )
    if (
        type(root_receipt) is not dict
        or set(root_receipt) != CANONICAL_ROOT_RECEIPT_KEYS
        or root_receipt["manifest"] != manifest
        or root_receipt["summary"] != summary
        or root_receipt["lineage"] != lineage
    ):
        raise BaseFinalAuthorityError(
            "canonical root receipt does not exactly bind its three artifacts"
        )
    expected_source_commit = _git_oid(
        root_receipt["source_commit"],
        "canonical source commit",
    )
    expected_source_tree = _git_oid(
        root_receipt["source_tree"],
        "canonical source tree",
    )
    if (
        summary_payload.get("status") != "complete"
        or summary_payload.get("manifest_sha256") != manifest["sha256"]
        or summary_payload.get("finite") is not True
        or summary_payload.get("exact_once") is not True
        or summary_payload.get("split_disjoint") is not True
        or lineage_payload.get("final_manifest_sha256")
        != manifest["sha256"]
        or summary_payload.get("lineage_sha256")
        != canonical_json_sha256(lineage_payload)
    ):
        raise BaseFinalAuthorityError(
            "canonical root summary/lineage is not strictly complete"
        )
    contract = lineage_payload.get("lineage_contract")
    contract_sha = lineage_payload.get("lineage_contract_sha256")
    if (
        type(contract) is not dict
        or canonical_json_sha256(contract) != contract_sha
        or summary_payload.get("lineage_contract_sha256") != contract_sha
        or type(contract.get("source_receipt")) is not dict
        or contract["source_receipt"].get("origin") != ORIGIN
        or contract["source_receipt"].get("commit")
        != expected_source_commit
        or contract["source_receipt"].get("tree")
        != expected_source_tree
        or summary_payload.get("source_receipt_sha256")
        != canonical_json_sha256(contract["source_receipt"])
    ):
        raise BaseFinalAuthorityError(
            "canonical root/source lineage contract is invalid"
        )
    selected = [row for row in rows if row.get("split") == "test"]
    observed_indices = [row.get("global_index") for row in selected]
    if observed_indices != list(range(TEST_GLOBAL_START, TEST_GLOBAL_STOP)):
        raise BaseFinalAuthorityError(
            "canonical test rows are not the exact frozen ordered domain"
        )
    by_id: dict[str, dict[str, Any]] = {}
    row_hashes: list[str] = []
    for row in selected:
        clip_id = row.get("clip_id")
        speaker = row.get("speaker")
        if (
            type(clip_id) is not str
            or clip_id in by_id
            or speaker not in SHOW_SPEAKER_IDS
            or clip_id.split("/", 1)[0] != speaker
            or row.get("speaker_id") != SHOW_SPEAKER_IDS[speaker]
            or row.get("pose_fps") != POSE_FPS
            or _exact_int(
                row.get("frames"),
                f"{clip_id} frames",
                minimum=MIN_FRAMES_EXCLUSIVE + 1,
            )
            <= MIN_FRAMES_EXCLUSIVE
            or row.get("lineage_contract_sha256") != contract_sha
        ):
            raise BaseFinalAuthorityError(
                f"canonical test row {clip_id!r} is invalid"
            )
        _sha256(row.get("canonical_npz_sha256"), f"{clip_id} NPZ SHA")
        _sha256(row.get("source_wav_sha256"), f"{clip_id} WAV SHA")
        by_id[clip_id] = row
        row_hashes.append(canonical_json_sha256(row))
    if len(by_id) != TEST_CLIPS or set(
        row["speaker"] for row in selected
    ) != set(SHOW_SPEAKER_IDS):
        raise BaseFinalAuthorityError(
            "canonical test rows do not cover SHOW exactly once"
        )
    return (
        {
            "root_receipt": {
                **dict(root_receipt),
                "source_commit": expected_source_commit,
                "source_tree": expected_source_tree,
            },
            "manifest": manifest,
            "summary": summary,
            "lineage": lineage,
            "lineage_contract_sha256": contract_sha,
            "source_receipt": contract["source_receipt"],
            "test_global_start": TEST_GLOBAL_START,
            "test_global_stop": TEST_GLOBAL_STOP,
            "test_clips": TEST_CLIPS,
            "ordered_row_sha256": row_hashes,
        },
        by_id,
    )


def _audio_authority(
    values: Sequence[Mapping[str, Any]],
    *,
    canonical: Mapping[str, Any],
    canonical_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if type(values) not in {list, tuple} or len(values) != NUM_SHARDS:
        raise BaseFinalAuthorityError(
            "audio authority requires exactly eight shards"
        )
    result: list[dict[str, Any]] = []
    covered: set[str] = set()
    feature_paths: set[str] = set()
    source_receipt: Any = None
    runtime_receipt: Any = None
    for expected_shard, value in enumerate(values):
        if type(value) is not dict or set(value) != {
            "shard_id",
            "manifest",
            "summary",
            "lineage",
        }:
            raise BaseFinalAuthorityError(
                "audio authority shard schema mismatch"
            )
        shard_id = _exact_int(
            value["shard_id"],
            "audio shard_id",
        )
        if shard_id != expected_shard:
            raise BaseFinalAuthorityError(
                "audio authorities are not in shard order 0..7"
            )
        manifest, rows = _jsonl_artifact(
            value["manifest"],
            f"audio shard {shard_id} manifest",
        )
        summary, summary_payload = _json_artifact(
            value["summary"],
            f"audio shard {shard_id} summary",
        )
        lineage, lineage_payload = _json_artifact(
            value["lineage"],
            f"audio shard {shard_id} lineage",
        )
        if (
            summary_payload.get("status") != "complete"
            or lineage_payload.get("status") != "complete"
            or summary_payload.get("output_manifest_sha256")
            != manifest["sha256"]
            or lineage_payload.get("output_manifest_sha256")
            != manifest["sha256"]
            or summary_payload.get("lineage_json_sha256")
            != lineage["sha256"]
            or lineage_payload.get("canonical_manifest_sha256")
            != {canonical["manifest"]["path"]: canonical["manifest"]["sha256"]}
            or lineage_payload.get("canonical_receipt") is None
            or lineage_payload.get("lineage_contract_sha256")
            != canonical["lineage_contract_sha256"]
            or lineage_payload.get("shard_id") != shard_id
            or lineage_payload.get("num_shards") != NUM_SHARDS
        ):
            raise BaseFinalAuthorityError(
                f"audio shard {shard_id} root lineage mismatch"
            )
        current_source = lineage_payload.get("source_receipt")
        current_runtime = lineage_payload.get("runtime")
        if (
            type(current_source) is not dict
            or current_source.get("origin") != ORIGIN
            or type(current_runtime) is not dict
        ):
            raise BaseFinalAuthorityError(
                f"audio shard {shard_id} source/runtime receipt invalid"
            )
        if source_receipt is None:
            source_receipt = current_source
            runtime_receipt = current_runtime
        elif (
            current_source != source_receipt
            or current_runtime != runtime_receipt
        ):
            raise BaseFinalAuthorityError(
                "audio shards used different source/runtime receipts"
            )
        row_hashes: list[str] = []
        feature_artifact_receipts: list[dict[str, Any]] = []
        for row in rows:
            clip_id = row.get("clip_id")
            canonical_row = canonical_by_id.get(str(clip_id))
            if (
                canonical_row is None
                or clip_id in covered
                or row.get("split") != "test"
                or row.get("shard_id") != shard_id
                or row.get("num_shards") != NUM_SHARDS
                or row.get("canonical_npz_sha256")
                != canonical_row["canonical_npz_sha256"]
                or row.get("source_wav_sha256")
                != canonical_row["source_wav_sha256"]
                or row.get("frames") != canonical_row["frames"]
                or row.get("lineage_contract_sha256")
                != canonical["lineage_contract_sha256"]
            ):
                raise BaseFinalAuthorityError(
                    f"audio shard {shard_id} row {clip_id!r} is unbound"
                )
            feature_path, feature_payload = _safe_file_snapshot(
                row.get("audio_feature_npz"),
                f"audio feature {clip_id}",
            )
            feature_sha = _sha256(
                row.get("audio_feature_npz_sha256"),
                f"audio feature {clip_id} SHA",
            )
            if (
                hashlib.sha256(feature_payload).hexdigest() != feature_sha
                or str(feature_path) in feature_paths
            ):
                raise BaseFinalAuthorityError(
                    f"audio feature artifact {clip_id!r} changed or repeats"
                )
            feature_paths.add(str(feature_path))
            feature_artifact_receipts.append(
                {
                    "clip_id": str(clip_id),
                    "path": str(feature_path),
                    "sha256": feature_sha,
                    "bytes": len(feature_payload),
                }
            )
            covered.add(str(clip_id))
            row_hashes.append(canonical_json_sha256(row))
        if lineage_payload.get("shard_clips") != len(rows):
            raise BaseFinalAuthorityError(
                f"audio shard {shard_id} row count mismatch"
            )
        result.append(
            {
                "shard_id": shard_id,
                "manifest": manifest,
                "summary": summary,
                "lineage": lineage,
                "rows": len(rows),
                "ordered_row_sha256": row_hashes,
                "audio_feature_artifacts": {
                    "count": len(feature_artifact_receipts),
                    "bytes": sum(
                        artifact["bytes"]
                        for artifact in feature_artifact_receipts
                    ),
                    "ordered_receipt_sha256": canonical_json_sha256(
                        feature_artifact_receipts
                    ),
                },
                "source_receipt": current_source,
                "runtime": current_runtime,
                "hubert_model_tree_sha256": _sha256(
                    lineage_payload.get("hubert_model_tree_sha256"),
                    f"audio shard {shard_id} HuBERT tree SHA",
                ),
            }
        )
    if (
        covered != set(canonical_by_id)
        or len(feature_paths) != TEST_CLIPS
    ):
        raise BaseFinalAuthorityError(
            "eight audio shards/features do not cover canonical test exactly "
            "once"
        )
    return result


def _checkpoint_bundle(value: Any) -> dict[str, Any]:
    if type(value) is not dict or tuple(sorted(value)) != tuple(
        sorted(CHECKPOINT_STAGES)
    ):
        raise BaseFinalAuthorityError(
            "checkpoint bundle must contain Base and five representations"
        )
    result: dict[str, Any] = {}
    seen_paths: set[str] = set()
    seen_digests: set[str] = set()
    for stage in CHECKPOINT_STAGES:
        artifact, _payload = _verified_artifact(
            value[stage],
            f"{stage} checkpoint",
        )
        _reject_forbidden(
            artifact,
            f"{stage} checkpoint",
            semantic_identity=True,
        )
        if (
            artifact["path"] in seen_paths
            or artifact["sha256"] in seen_digests
        ):
            raise BaseFinalAuthorityError(
                "different checkpoint stages cannot share a path or content"
            )
        seen_paths.add(artifact["path"])
        seen_digests.add(artifact["sha256"])
        result[stage] = artifact
    return result


def _validate_official_training_source(
    value: Any,
    *,
    inference_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that Base-long was produced by reachable official SemTalk code."""

    expected_keys = {
        "origin",
        "commit",
        "tree",
        "branch",
        "clean",
        "entrypoint",
        "entrypoint_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value["origin"] != ORIGIN
        or value["branch"] is not None
        or value["clean"] is not True
    ):
        raise BaseFinalAuthorityError(
            "Base-long producer source receipt schema/identity mismatch"
        )
    commit = _git_oid(value["commit"], "Base-long producer commit")
    tree = _git_oid(value["tree"], "Base-long producer tree")
    entrypoint, entrypoint_payload = _safe_file_snapshot(
        value["entrypoint"],
        "Base-long producer entrypoint",
    )
    expected_suffix = Path(
        "scripts/show_base/train_base_official_adapt_long.py"
    )
    if (
        tuple(entrypoint.parts[-len(expected_suffix.parts) :])
        != expected_suffix.parts
        or hashlib.sha256(entrypoint_payload).hexdigest()
        != _sha256(
            value["entrypoint_sha256"],
            "Base-long producer entrypoint SHA",
        )
    ):
        raise BaseFinalAuthorityError(
            "Base-long producer entrypoint is not the official long trainer"
        )
    producer_root = Path(
        _git(entrypoint.parent, "rev-parse", "--show-toplevel")
    ).resolve()
    inference_root = Path(inference_source["source_root"]).resolve()
    inference_commit = _git_oid(
        inference_source["commit"],
        "pinned inference source commit",
    )
    inference_tree = _git_oid(
        inference_source["tree"],
        "pinned inference source tree",
    )
    symbolic = subprocess.run(
        ["git", "-C", str(producer_root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    local_branches = [
        line
        for line in _git(
            producer_root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
        ).splitlines()
        if line
    ]
    try:
        producer_entrypoint = entrypoint.relative_to(producer_root)
    except ValueError as exc:
        raise BaseFinalAuthorityError(
            "Base-long producer entrypoint escapes its source root"
        ) from exc
    if (
        producer_entrypoint != expected_suffix
        or _git(producer_root, "remote", "get-url", "origin") != ORIGIN
        or _git(producer_root, "rev-parse", "HEAD^{commit}") != commit
        or _git(producer_root, "rev-parse", "HEAD^{tree}") != tree
        or symbolic.returncode != 1
        or local_branches
        or _git(
            producer_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        != ""
        or _git(inference_root, "rev-parse", f"{commit}^{{commit}}")
        != commit
        or _git(inference_root, "rev-parse", f"{commit}^{{tree}}")
        != tree
        or subprocess.run(
            [
                "git",
                "-C",
                str(inference_root),
                "merge-base",
                "--is-ancestor",
                OFFICIAL_BASELINE_COMMIT,
                commit,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).returncode
        != 0
        or subprocess.run(
            [
                "git",
                "-C",
                str(inference_root),
                "merge-base",
                "--is-ancestor",
                commit,
                inference_commit,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).returncode
        != 0
    ):
        raise BaseFinalAuthorityError(
            "Base-long producer commit/tree is not a clean reachable "
            "pinned inference-source ancestor"
        )
    return {
        **dict(value),
        "source_root": str(producer_root),
        "pinned_inference_commit": inference_commit,
        "pinned_inference_tree": inference_tree,
        "inference_source_ancestor": True,
        "official_baseline_ancestor": True,
        "detached": True,
        "local_branches_at_commit": [],
    }


def _replay_base_long_candidate_bundle(
    value: Any,
    *,
    inference_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Fresh replay the sole 22-candidate/e400 Base training transaction."""

    if (
        type(value) is not dict
        or set(value) != set(BASE_LONG_ARTIFACT_ROLES)
    ):
        raise BaseFinalAuthorityError(
            "Base-long external artifact bundle schema mismatch"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for role in BASE_LONG_ARTIFACT_ROLES:
        artifact, _payload = _verified_artifact(
            value[role],
            f"Base-long {role}",
        )
        artifacts[role] = artifact
    module = _control_module("base_long_val_contract")
    validator = getattr(module, "validate_candidate_bundle", None)
    epochs = getattr(module, "EXPECTED_CANDIDATE_EPOCHS", None)
    if (
        not callable(validator)
        or type(epochs) not in {tuple, list}
    ):
        raise BaseFinalAuthorityError(
            "Base-long neutral candidate validator ABI mismatch"
        )
    try:
        replayed = validator(
            manifest_path=Path(artifacts["manifest"]["path"]),
            expected_manifest_sha256=artifacts["manifest"]["sha256"],
            status_path=Path(artifacts["status"]["path"]),
            expected_status_sha256=artifacts["status"]["sha256"],
            frozen_inputs_path=Path(artifacts["frozen_inputs"]["path"]),
            expected_frozen_inputs_sha256=artifacts["frozen_inputs"][
                "sha256"
            ],
        )
    except Exception as exc:
        raise BaseFinalAuthorityError(
            f"Base-long neutral candidate replay failed: {exc}"
        ) from exc
    expected_epochs = tuple(epochs)
    if (
        expected_epochs
        != (
            1,
            2,
            4,
            8,
            16,
            32,
            40,
            50,
            60,
            70,
            80,
            100,
            120,
            140,
            160,
            180,
            200,
            240,
            280,
            320,
            360,
            400,
        )
        or type(replayed) is not dict
        or set(replayed)
        != {
            "manifest",
            "status",
            "frozen_inputs",
            "producer_source",
            "selected_prerequisite_sha256",
            "selected_topology",
            "selected_dataset",
            "updates_per_epoch",
            "candidates",
        }
        or replayed["manifest"]
        != {
            "path": artifacts["manifest"]["path"],
            "sha256": artifacts["manifest"]["sha256"],
        }
        or replayed["status"]
        != {
            "path": artifacts["status"]["path"],
            "sha256": artifacts["status"]["sha256"],
        }
        or type(replayed["frozen_inputs"]) is not dict
        or set(replayed["frozen_inputs"])
        != {"path", "sha256", "receipt_sha256"}
        or replayed["frozen_inputs"]["path"]
        != artifacts["frozen_inputs"]["path"]
        or replayed["frozen_inputs"]["sha256"]
        != artifacts["frozen_inputs"]["sha256"]
        or type(replayed["candidates"]) is not dict
        or tuple(replayed["candidates"]) != expected_epochs
        or type(replayed["selected_prerequisite_sha256"]) is not dict
        or set(replayed["selected_prerequisite_sha256"])
        != set(REPRESENTATION_STAGES)
        or replayed.get("updates_per_epoch") not in {62, 124, 248, 1988}
        or not isinstance(replayed.get("selected_topology"), dict)
        or replayed["selected_topology"].get("updates_per_epoch")
        != replayed["updates_per_epoch"]
        or type(replayed.get("selected_dataset")) is not dict
        or set(replayed["selected_dataset"]) != BASE_LONG_DATASET_KEYS
        or replayed["selected_dataset"].get("split") != "train"
        or replayed["selected_dataset"].get("test_visible") is not False
        or replayed["selected_dataset"].get(
            "selected_prerequisite_sha256"
        )
        != replayed["selected_prerequisite_sha256"]
    ):
        raise BaseFinalAuthorityError(
            "Base-long neutral candidate replay returned a non-canonical "
            "22-candidate authority"
        )
    candidates: list[dict[str, Any]] = []
    updates_per_epoch = replayed["updates_per_epoch"]
    for epoch in expected_epochs:
        checkpoint, _payload = _verified_artifact(
            replayed["candidates"][epoch],
            f"Base-long e{epoch} candidate",
        )
        candidates.append(
            {
                "epoch": epoch,
                "optimizer_updates": epoch * updates_per_epoch,
                "candidate_checkpoint": checkpoint,
            }
        )
    producer_source = _validate_official_training_source(
        replayed["producer_source"],
        inference_source=inference_source,
    )
    return {
        "artifacts": artifacts,
        "manifest": dict(replayed["manifest"]),
        "status": dict(replayed["status"]),
        "frozen_inputs": dict(replayed["frozen_inputs"]),
        "producer_source": producer_source,
        "selected_prerequisite_sha256": dict(
            replayed["selected_prerequisite_sha256"]
        ),
        "selected_topology": dict(replayed["selected_topology"]),
        "selected_dataset": dict(replayed["selected_dataset"]),
        "candidate_epochs": list(expected_epochs),
        "updates_per_epoch": updates_per_epoch,
        "candidates": candidates,
    }


def _control_module(name: str) -> Any:
    """Compile a fresh local validator and its local dependency closure.

    No object from ``sys.modules`` and no bytecode cache is allowed to satisfy
    a formal control import.  Dependencies are installed only for the duration
    of this source compilation, then the caller's module table is restored.
    """

    if name not in _CONTROL_DEPENDENCIES:
        raise BaseFinalAuthorityError(
            f"unknown required fresh control validator {name}"
        )
    source_root = Path(__file__).resolve().parent
    package = sys.modules.get("scripts.show_base")
    if package is None:
        package = __import__("scripts.show_base", fromlist=["*"])
    package_source = getattr(package, "__file__", None)
    expected_package = source_root / "__init__.py"
    if (
        type(package_source) is not str
        or Path(package_source).resolve() != expected_package
    ):
        raise BaseFinalAuthorityError(
            "scripts.show_base package is not this source tree"
        )
    _safe_file_snapshot(str(expected_package), "scripts.show_base package")

    missing = object()
    saved_modules: dict[str, Any] = {}
    saved_attributes: dict[str, Any] = {}
    loaded: dict[str, Any] = {}

    def load(module_name: str) -> Any:
        if module_name in loaded:
            return loaded[module_name]
        for dependency in _CONTROL_DEPENDENCIES[module_name]:
            load(dependency)
        expected = source_root / f"{module_name}.py"
        expected_path, payload = _safe_file_snapshot(
            str(expected),
            f"control validator {module_name}",
        )
        full_name = f"scripts.show_base.{module_name}"
        saved_modules.setdefault(full_name, sys.modules.get(full_name, missing))
        saved_attributes.setdefault(
            module_name,
            getattr(package, module_name, missing),
        )
        module = types.ModuleType(full_name)
        module.__file__ = str(expected_path)
        module.__package__ = "scripts.show_base"
        module.__loader__ = None
        sys.modules[full_name] = module
        setattr(package, module_name, module)
        loaded[module_name] = module
        try:
            code = compile(
                payload,
                str(expected_path),
                "exec",
                dont_inherit=True,
            )
            exec(code, module.__dict__)
        except BaseException:
            loaded.pop(module_name, None)
            raise
        return module

    try:
        return load(name)
    except BaseFinalAuthorityError:
        raise
    except BaseException as exc:
        raise BaseFinalAuthorityError(
            f"cannot source-load required fresh control validator {name}"
        ) from exc
    finally:
        for full_name, previous in saved_modules.items():
            if previous is missing:
                sys.modules.pop(full_name, None)
            else:
                sys.modules[full_name] = previous
        for attribute, previous in saved_attributes.items():
            if previous is missing:
                try:
                    delattr(package, attribute)
                except AttributeError:
                    pass
            else:
                setattr(package, attribute, previous)


def _replay_prerequisite_selection(binding: Mapping[str, Any]) -> dict[str, Any]:
    if type(binding) is not dict or set(binding) != {
        "path",
        "sha256",
        "receipt_payload_sha256",
    }:
        raise BaseFinalAuthorityError(
            "prerequisite selection binding schema mismatch"
        )
    expected_payload_sha = _sha256(
        binding["receipt_payload_sha256"],
        "prerequisite selection payload SHA",
    )
    module = _control_module("select_prerequisite_candidates")
    try:
        replayed = module.replay_selection(
            selection_path=Path(binding["path"]),
            expected_selection_sha256=_sha256(
                binding["sha256"],
                "prerequisite selection file SHA",
            ),
        )
    except Exception as exc:
        raise BaseFinalAuthorityError(
            f"prerequisite selection fresh replay failed: {exc}"
        ) from exc
    if (
        type(replayed) is not dict
        or replayed.get("receipt_payload_sha256") != expected_payload_sha
    ):
        raise BaseFinalAuthorityError(
            "prerequisite selection replay/payload binding mismatch"
        )
    return replayed


def _replay_prerequisite_bridge(
    binding: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    module = _control_module("selected_prerequisites")
    try:
        bridge = module.load_selected_prerequisites(
            binding["path"],
            binding["sha256"],
        )
    except Exception as exc:
        raise BaseFinalAuthorityError(
            f"selected-five bridge fresh replay failed: {exc}"
        ) from exc
    if (
        type(bridge) is not dict
        or bridge.get("format")
        != "semtalk_show_selected_prerequisite_bridge_v1"
        or bridge.get("selection") != binding
        or bridge.get("test_visible") is not False
        or bridge.get("global_verified_not_consumed") is not True
    ):
        raise BaseFinalAuthorityError(
            "selected-five bridge protocol/binding mismatch"
        )
    expected = {
        stage["stage"]: stage["candidate_checkpoint"]
        for stage in selection["stages"]
    }
    selected = bridge.get("selected")
    if (
        type(selected) is not dict
        or set(selected) != set(REPRESENTATION_STAGES)
        or {
            stage: selected[stage].get("candidate_checkpoint")
            for stage in REPRESENTATION_STAGES
        }
        != expected
    ):
        raise BaseFinalAuthorityError(
            "selected-five bridge differs from selector replay"
        )
    return bridge


def _replay_continuation(
    artifact: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    module = _control_module("decide_prerequisite_continuation")
    try:
        replayed = module.replay_decision(
            Path(artifact["path"]),
            artifact["sha256"],
        )
    except Exception as exc:
        raise BaseFinalAuthorityError(
            f"continuation decision fresh replay failed: {exc}"
        ) from exc
    if replayed != payload:
        raise BaseFinalAuthorityError(
            "continuation decision differs from fresh replay"
        )
    return replayed


def _replay_continuation_waves(
    values: Any,
    *,
    candidate_epochs_by_stage: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Fresh-replay the monotone active subsets of mixed-stage v2 waves."""

    mandatory = tuple(range(20, 201, 20))
    if type(candidate_epochs_by_stage) is not dict or set(
        candidate_epochs_by_stage
    ) != set(REPRESENTATION_STAGES):
        raise BaseFinalAuthorityError(
            "prerequisite per-stage candidate schedules are incomplete"
        )
    schedules: dict[str, tuple[int, ...]] = {}
    for stage in REPRESENTATION_STAGES:
        schedule = tuple(candidate_epochs_by_stage[stage])
        if (
            len(schedule) < len(mandatory)
            or schedule[: len(mandatory)] != mandatory
            or any(
                epoch != schedule[index - 1] + 20
                for index, epoch in enumerate(schedule)
                if index
            )
        ):
            raise BaseFinalAuthorityError(
                f"{stage} prerequisite schedule is not the exact +20 chain"
            )
        schedules[stage] = schedule
    if type(values) is not list:
        raise BaseFinalAuthorityError(
            "continuation wave inventory must be a JSON list"
        )
    normalized: list[dict[str, Any]] = []
    positions = {stage: len(mandatory) for stage in REPRESENTATION_STAGES}
    previous_bindings: dict[str, dict[str, Any] | None] = {
        stage: None for stage in REPRESENTATION_STAGES
    }
    eligible = set(REPRESENTATION_STAGES)
    for index, raw_value in enumerate(values):
        artifact = _payload_artifact_from_binding(
            raw_value,
            f"continuation wave {index}",
        )
        module = _control_module("prerequisite_continuation_wave")
        replay = getattr(module, "replay_wave_file", None)
        if not callable(replay):
            raise BaseFinalAuthorityError(
                "continuation wave fresh replay ABI mismatch"
            )
        try:
            receipt = replay(
                Path(artifact["path"]),
                artifact["sha256"],
            )
        except Exception as exc:
            raise BaseFinalAuthorityError(
                f"continuation wave {index} fresh replay failed: {exc}"
            ) from exc
        if (
            type(receipt) is not dict
            or receipt.get("format")
            != "semtalk_show_prerequisite_continuation_wave_v2"
            or receipt.get("status") != "authorized"
            or receipt.get("test_visible") is not False
            or receipt.get("receipt_payload_sha256")
            != artifact["receipt_payload_sha256"]
            or type(receipt.get("trigger_stages")) is not list
            or not receipt["trigger_stages"]
            or type(receipt.get("stages")) is not list
            or len(receipt["stages"]) != len(receipt["trigger_stages"])
        ):
            raise BaseFinalAuthorityError(
                f"continuation wave {index} identity/boundary mismatch"
            )
        triggers = list(receipt["trigger_stages"])
        if (
            triggers
            != [stage for stage in REPRESENTATION_STAGES if stage in set(triggers)]
            or not set(triggers).issubset(eligible)
        ):
            raise BaseFinalAuthorityError(
                f"continuation wave {index} reactivates a frozen stage"
            )
        eligible = set(triggers)
        transitions = []
        current_binding = {
            key: artifact[key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        }
        for expected_stage, stage in zip(triggers, receipt["stages"]):
            old_segment = stage.get("old_segment") if type(stage) is dict else None
            position = positions[expected_stage]
            if position >= len(schedules[expected_stage]):
                raise BaseFinalAuthorityError(
                    f"continuation wave {index} exceeds {expected_stage} schedule"
                )
            boundary = schedules[expected_stage][position - 1]
            target = schedules[expected_stage][position]
            if (
                type(stage) is not dict
                or stage.get("stage") != expected_stage
                or stage.get("boundary_epoch") != boundary
                or stage.get("target_epoch") != target
                or target != boundary + 20
                or type(old_segment) is not dict
                or old_segment.get("predecessor_wave")
                != previous_bindings[expected_stage]
            ):
                raise BaseFinalAuthorityError(
                    f"continuation wave {index} stage chain mismatch"
                )
            positions[expected_stage] += 1
            previous_bindings[expected_stage] = current_binding
            transitions.append(
                {
                    "stage": expected_stage,
                    "boundary_epoch": boundary,
                    "target_epoch": target,
                    "cap_epoch": stage.get("cap_epoch"),
                }
            )
        normalized.append(
            {
                **artifact,
                "canonical_payload_sha256": canonical_json_sha256(
                    receipt
                ),
                "decision": dict(receipt["decision"]),
                "trigger_stages": triggers,
                "stage_transitions": transitions,
            }
        )
    if any(
        positions[stage] != len(schedules[stage])
        for stage in REPRESENTATION_STAGES
    ):
        raise BaseFinalAuthorityError(
            "continuation waves do not cover every stage-local appended boundary"
        )
    return normalized


def _prerequisite_candidate_schedules(
    selection: Mapping[str, Any],
) -> dict[str, list[int]]:
    """Normalize the replayed v1 common or v2 per-stage schedule."""

    mandatory = tuple(range(20, 201, 20))

    def normalize(value: Any, label: str) -> list[int]:
        if type(value) is not list or any(
            type(epoch) is not int for epoch in value
        ):
            raise BaseFinalAuthorityError(f"{label} is not an integer list")
        schedule = list(value)
        if (
            len(schedule) < len(mandatory)
            or tuple(schedule[: len(mandatory)]) != mandatory
            or any(
                epoch != schedule[index - 1] + 20
                for index, epoch in enumerate(schedule)
                if index
            )
        ):
            raise BaseFinalAuthorityError(
                f"{label} is not the exact append-only +20 schedule"
            )
        return schedule

    protocol = selection.get("protocol")
    selection_format = selection.get("format")
    if type(protocol) is not dict:
        raise BaseFinalAuthorityError(
            "selected-five prerequisite protocol is missing"
        )
    common = normalize(
        protocol.get("candidate_epochs"),
        "selected-five common prerequisite schedule",
    )
    if selection_format == INITIAL_PREREQUISITE_SELECTION_FORMAT:
        if (
            tuple(common) != mandatory
            or "candidate_epochs_by_stage" in protocol
            or protocol.get("candidates_per_stage") != len(common)
        ):
            raise BaseFinalAuthorityError(
                "initial-v1 prerequisite schedule is not the mandatory prefix"
            )
        return {stage: list(common) for stage in REPRESENTATION_STAGES}
    if selection_format != PER_STAGE_PREREQUISITE_SELECTION_FORMAT:
        raise BaseFinalAuthorityError(
            "selected-five prerequisite selection format mismatch"
        )
    raw_schedules = protocol.get("candidate_epochs_by_stage")
    if type(raw_schedules) is not dict or set(raw_schedules) != set(
        REPRESENTATION_STAGES
    ):
        raise BaseFinalAuthorityError(
            "selected-five per-stage prerequisite schedules are missing"
        )
    schedules = {
        stage: normalize(
            raw_schedules[stage],
            f"selected-five {stage} prerequisite schedule",
        )
        for stage in REPRESENTATION_STAGES
    }
    union = sorted(
        {
            epoch
            for schedule in schedules.values()
            for epoch in schedule
        }
    )
    if (
        common != union
        or protocol.get("candidates_per_stage")
        != {
            stage: len(schedules[stage])
            for stage in REPRESENTATION_STAGES
        }
    ):
        raise BaseFinalAuthorityError(
            "selected-five per-stage prerequisite schedule union mismatch"
        )
    return schedules


def _payload_artifact_from_binding(
    value: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Add a fresh byte count to one externally SHA/payload-pinned file."""

    allowed = {
        "path",
        "sha256",
        "receipt_payload_sha256",
    }
    if type(value) is not dict or frozenset(value) not in {
        frozenset(allowed),
        frozenset(allowed | {"bytes"}),
    }:
        raise BaseFinalAuthorityError(
            f"{label} payload-artifact binding schema mismatch"
        )
    path, payload = _safe_file_snapshot(value["path"], label)
    expected_sha = _sha256(value["sha256"], f"{label} file SHA")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise BaseFinalAuthorityError(f"{label} artifact changed")
    observed_bytes = len(payload)
    if (
        "bytes" in value
        and _exact_int(value["bytes"], f"{label} bytes", minimum=1)
        != observed_bytes
    ):
        raise BaseFinalAuthorityError(f"{label} byte count changed")
    return {
        "path": str(path),
        "sha256": expected_sha,
        "bytes": observed_bytes,
        "receipt_payload_sha256": _sha256(
            value["receipt_payload_sha256"],
            f"{label} payload SHA",
        ),
    }


def _replay_long_diffsheg_test_winner_claim(
    artifact: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    expected_output_root: Path,
    candidate_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Fresh-replay the one-shot claim for the 22-way DiffSHEG winner."""

    module = _control_module("validate_base_long_test_winner")
    validator = getattr(
        module,
        "validate_published_test_winner_claim",
        None,
    )
    expected_format = getattr(module, "AUTHORIZATION_FORMAT", None)
    if (
        not callable(validator)
        or type(expected_format) is not str
        or payload.get("format") != expected_format
    ):
        raise BaseFinalAuthorityError(
            "formal Base authority requires the long DiffSHEG one-shot "
            "claim protocol"
        )
    payload_sha = _sha256(
        payload.get("receipt_payload_sha256"),
        "long DiffSHEG test-winner claim payload SHA",
    )
    try:
        validated = validator(
            Path(artifact["path"]),
            expected_claim_sha256=artifact["sha256"],
            expected_claim_bytes=artifact["bytes"],
            expected_claim_payload_sha256=payload_sha,
            expected_output_root=expected_output_root,
            candidate_bundle=candidate_bundle,
        )
    except Exception as exc:
        raise BaseFinalAuthorityError(
            f"long DiffSHEG test-winner replay failed: {exc}"
        ) from exc
    expected_keys = {
        "claim_artifact",
        "receipt_payload_sha256",
        "winner_selection",
        "selected_base_checkpoint",
        "selected_epoch",
        "selected_fgd",
        "selected_diffsheg_report",
        "expected_output_root",
        "test_policy",
    }
    if type(validated) is not dict or set(validated) != expected_keys:
        raise BaseFinalAuthorityError(
            "long DiffSHEG test-winner validator schema mismatch"
        )
    if (
        validated["claim_artifact"] != artifact
        or validated["receipt_payload_sha256"] != payload_sha
        or validated["expected_output_root"] != str(expected_output_root)
        or validated["test_policy"]
        != {
            "authorized_evaluations": 1,
            "one_shot_claim_required": True,
            "selection_feedback": False,
        }
    ):
        raise BaseFinalAuthorityError(
            "long DiffSHEG test-winner authority changed"
        )
    return validated


def _control_authority(
    *,
    expected_output_root: Path,
    base_long_candidate_bundle: Mapping[str, Any],
    winner_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    continuation_waves: Sequence[Mapping[str, Any]],
    winner_full_metric_closure: Mapping[str, Any],
    test_claim: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    continuation, continuation_payload = _json_artifact(
        continuation_decision,
        "continuation decision",
    )
    replayed_continuation = _replay_continuation(
        continuation,
        continuation_payload,
    )
    selection_binding = replayed_continuation.get("inputs", {}).get(
        "selection"
    )
    if (
        replayed_continuation.get("format")
        != "semtalk_show_prerequisite_continuation_decision_v2"
        or replayed_continuation.get("status") != "complete"
        or replayed_continuation.get("decision") != "stop"
        or replayed_continuation.get("test_visible") is not False
        or type(selection_binding) is not dict
    ):
        raise BaseFinalAuthorityError(
            "final Base authority requires the fresh continuation-stop "
            "decision"
        )
    prerequisite_selection = _replay_prerequisite_selection(
        selection_binding
    )
    stages = prerequisite_selection.get("stages")
    if (
        prerequisite_selection.get("format")
        not in {
            INITIAL_PREREQUISITE_SELECTION_FORMAT,
            PER_STAGE_PREREQUISITE_SELECTION_FORMAT,
        }
        or prerequisite_selection.get("status") != "selected"
        or prerequisite_selection.get("split") != "val"
        or prerequisite_selection.get("test_visible") is not False
        or type(stages) is not list
        or len(stages) != len(REPRESENTATION_STAGES)
    ):
        raise BaseFinalAuthorityError(
            "selected-five prerequisite replay protocol mismatch"
        )
    explicit_stages: dict[str, dict[str, Any]] = {}
    for stage_receipt in stages:
        stage = stage_receipt.get("stage") if type(stage_receipt) is dict else None
        if (
            stage not in REPRESENTATION_STAGES
            or stage in explicit_stages
            or stage_receipt.get("candidate_checkpoint")
            != checkpoints[stage]
        ):
            raise BaseFinalAuthorityError(
                f"selected-five replay does not bind the {stage!r} checkpoint"
            )
        pinned_measurement = _payload_artifact_from_binding(
            stage_receipt.get("measurement_receipt"),
            f"selected-five {stage} measurement receipt",
        )
        explicit_stages[stage] = {
            "stage": stage,
            "epoch": stage_receipt.get("epoch"),
            "optimizer_updates": stage_receipt.get(
                "optimizer_updates"
            ),
            "selection_metric": stage_receipt.get("selection_metric"),
            "selection_score": stage_receipt.get("selection_score"),
            "candidate_checkpoint": dict(
                stage_receipt["candidate_checkpoint"]
            ),
            "measurement_receipt": pinned_measurement,
        }
    if set(explicit_stages) != set(REPRESENTATION_STAGES):
        raise BaseFinalAuthorityError(
            "selected-five replay stage coverage mismatch"
        )
    if base_long_candidate_bundle.get(
        "selected_prerequisite_sha256"
    ) != {
        stage: explicit_stages[stage]["candidate_checkpoint"]["sha256"]
        for stage in REPRESENTATION_STAGES
    }:
        raise BaseFinalAuthorityError(
            "Base frozen inputs differ from the selected five SHOW winners"
        )
    candidate_epochs_by_stage = _prerequisite_candidate_schedules(
        prerequisite_selection
    )
    decision_stages = replayed_continuation.get("stages")
    caps = {
        "face": 600,
        "hands": 500,
        "upper": 500,
        "lower": 600,
        "global": 1700,
    }
    if (
        type(decision_stages) is not list
        or len(decision_stages) != len(REPRESENTATION_STAGES)
    ):
        raise BaseFinalAuthorityError(
            "final continuation-stop decision stage coverage mismatch"
        )
    for stage, terminal in zip(REPRESENTATION_STAGES, decision_stages):
        selected_stage = explicit_stages[stage]
        schedule = candidate_epochs_by_stage[stage]
        if (
            type(terminal) is not dict
            or terminal.get("stage") != stage
            or type(schedule) is not list
            or not schedule
            or terminal.get("latest_epoch") != schedule[-1]
            or terminal.get("winner_epoch") != selected_stage["epoch"]
            or terminal.get("frozen_winner_epoch") != selected_stage["epoch"]
            or terminal.get("cap_epoch") != caps[stage]
            or terminal.get("action") not in {"freeze", "capped"}
            or terminal.get("target_epoch") is not None
            or terminal.get("requests_continuation") is not False
            or (
                terminal.get("action") == "capped"
                and terminal.get("latest_epoch") != caps[stage]
            )
            or (
                terminal.get("action") == "freeze"
                and terminal.get("latest_epoch") >= caps[stage]
            )
        ):
            raise BaseFinalAuthorityError(
                f"final {stage} continuation action is not frozen/capped"
            )
    replayed_waves = _replay_continuation_waves(
        list(continuation_waves),
        candidate_epochs_by_stage=candidate_epochs_by_stage,
    )
    pinned_continuation_waves = [
        {
            key: wave[key]
            for key in (
                "path",
                "sha256",
                "bytes",
                "receipt_payload_sha256",
            )
        }
        for wave in replayed_waves
    ]
    prerequisite_bridge = _replay_prerequisite_bridge(
        selection_binding,
        prerequisite_selection,
    )
    pinned_prerequisite_selection = _payload_artifact_from_binding(
        selection_binding,
        "prerequisite selection",
    )
    pinned_continuation_decision = _payload_artifact_from_binding(
        {
            **continuation,
            "receipt_payload_sha256": replayed_continuation[
                "receipt_payload_sha256"
            ],
        },
        "continuation decision",
    )

    selection, selection_payload = _json_artifact(
        winner_selection,
        "winner selection",
    )
    pinned_winner_selection = _payload_artifact_from_binding(
        {
            **selection,
            "receipt_payload_sha256": _sha256(
                selection_payload.get("receipt_payload_sha256"),
                "winner selection payload SHA",
            ),
        },
        "winner selection",
    )
    if selection_payload.get("format") != (
        "semtalk_show_base_official_adapt_long_selection_v1"
    ):
        raise BaseFinalAuthorityError(
            "formal Base authority requires the 22-way DiffSHEG "
            "validation selection"
        )
    if selection_payload.get("format") == (
        "semtalk_show_base_official_adapt_long_selection_v1"
    ):
        full_closure, full_closure_payload = _json_artifact(
            winner_full_metric_closure,
            "selected DiffSHEG validation report",
        )
        claim, claim_payload = _json_artifact(
            test_claim,
            "long DiffSHEG one-shot test claim",
        )
        raw_candidate_bundle = {
            "manifest": dict(base_long_candidate_bundle["manifest"]),
            "status": dict(base_long_candidate_bundle["status"]),
            "frozen_inputs": dict(
                base_long_candidate_bundle["frozen_inputs"]
            ),
            "candidates": {
                row["epoch"]: dict(row["candidate_checkpoint"])
                for row in base_long_candidate_bundle["candidates"]
            },
        }
        published = _replay_long_diffsheg_test_winner_claim(
            claim,
            claim_payload,
            expected_output_root=expected_output_root,
            candidate_bundle=raw_candidate_bundle,
        )
        published_rows = selection_payload.get("candidate_metrics")
        long_rows = base_long_candidate_bundle["candidates"]
        if (
            type(published_rows) is not list
            or len(published_rows) != len(long_rows)
        ):
            raise BaseFinalAuthorityError(
                "DiffSHEG winner selection does not cover the exact "
                "Base-long candidate family"
            )
        normalized_published_checkpoints: list[dict[str, Any]] = []
        for long_row, published_row in zip(long_rows, published_rows):
            if (
                type(published_row) is not dict
                or published_row.get("epoch") != long_row["epoch"]
                or published_row.get("candidate_checkpoint")
                != long_row["candidate_checkpoint"]
            ):
                raise BaseFinalAuthorityError(
                    f"DiffSHEG Base e{long_row['epoch']} candidate is not "
                    "the fresh Base-long training artifact"
                )
            normalized_published_checkpoints.append(
                {
                    "epoch": long_row["epoch"],
                    "optimizer_updates": long_row["optimizer_updates"],
                    "candidate_checkpoint": dict(
                        long_row["candidate_checkpoint"]
                    ),
                }
            )
        matching_winners = [
            row
            for row in long_rows
            if row["candidate_checkpoint"]
            == published["selected_base_checkpoint"]
        ]
        selected_report = published["selected_diffsheg_report"]
        if (
            published["winner_selection"] != pinned_winner_selection
            or published["selected_base_checkpoint"] != checkpoints["base"]
            or len(matching_winners) != 1
            or published["selected_epoch"] != matching_winners[0]["epoch"]
            or type(published["selected_fgd"]) not in {int, float}
            or not isinstance(selected_report, dict)
            or set(selected_report) != {"path", "sha256"}
            or {
                key: full_closure[key]
                for key in ("path", "sha256")
            }
            != selected_report
        ):
            raise BaseFinalAuthorityError(
                "DiffSHEG winner does not bind the selected Base/report"
            )
        for label, payload in (
            ("winner selection", selection_payload),
            ("continuation decision", replayed_continuation),
            ("continuation waves", replayed_waves),
            ("prerequisite selection", prerequisite_selection),
            ("selected DiffSHEG validation report", full_closure_payload),
            ("one-shot test claim", claim_payload),
        ):
            _reject_forbidden(payload, label)
        return {
            "winner_selection": {
                **pinned_winner_selection,
                "canonical_payload_sha256": canonical_json_sha256(
                    selection_payload
                ),
                "selection_protocol": BASE_SELECTION_PROTOCOL,
                "selection_metric": BASE_SELECTION_METRIC,
                "candidate_checkpoints": normalized_published_checkpoints,
                "selected_epoch": matching_winners[0]["epoch"],
                "selected_optimizer_updates": matching_winners[0][
                    "optimizer_updates"
                ],
                "selected_fgd": float(published["selected_fgd"]),
                "selected_checkpoint": dict(
                    published["selected_base_checkpoint"]
                ),
                "selected_validation_report": dict(selected_report),
                "fixed_checkpoints": {
                    stage: dict(checkpoints[stage])
                    for stage in REPRESENTATION_STAGES
                },
            },
            "continuation_decision": {
                **pinned_continuation_decision,
                "canonical_payload_sha256": canonical_json_sha256(
                    replayed_continuation
                ),
                "decision": "stop",
                "prerequisite_selection": dict(
                    pinned_prerequisite_selection
                ),
            },
            "continuation_waves": pinned_continuation_waves,
            "winner_full_metric_closure": {
                **full_closure,
                "canonical_payload_sha256": canonical_json_sha256(
                    full_closure_payload
                ),
                "selection_role": "selected_validation_diffsheg_report",
            },
            "prerequisite_selection": {
                **pinned_prerequisite_selection,
                "canonical_payload_sha256": canonical_json_sha256(
                    prerequisite_selection
                ),
                "bridge_format": prerequisite_bridge["format"],
                "prerequisite_consumption": {
                    "base_training_feature_graph": {
                        "live_prerequisite_models": [],
                        "consumed_precomputed_selected_outputs": [
                            "face",
                            "hands",
                            "upper",
                            "lower",
                        ],
                        "global_model_consumed": False,
                    },
                    "official_base_inference": {
                        "strict_loaded_models": list(
                            REPRESENTATION_STAGES
                        ),
                        "decoded_models": list(REPRESENTATION_STAGES),
                        "global_translation_reconstruction": True,
                    },
                },
                "stages": explicit_stages,
            },
            "test_claim": {
                **claim,
                "canonical_payload_sha256": canonical_json_sha256(
                    claim_payload
                ),
                "receipt_payload_sha256": published[
                    "receipt_payload_sha256"
                ],
                "test_policy": {
                    **dict(published["test_policy"]),
                    "num_shards": NUM_SHARDS,
                    "canonical_test_clips": TEST_CLIPS,
                    "final_metric_event": FINAL_METRIC_EVENT,
                },
            },
            "base_long_candidate_bundle": dict(
                base_long_candidate_bundle
            ),
        }


def _authority_inputs(
    *,
    expected_output_root: str | Path,
    canonical_manifest: Mapping[str, Any],
    canonical_summary: Mapping[str, Any],
    canonical_lineage: Mapping[str, Any],
    canonical_root_receipt: Mapping[str, Any],
    audio_authorities: Sequence[Mapping[str, Any]],
    base_long_candidate_artifacts: Mapping[str, Mapping[str, Any]],
    winner_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    continuation_waves: Sequence[Mapping[str, Any]],
    winner_full_metric_closure: Mapping[str, Any],
    test_claim: Mapping[str, Any],
    inference_source: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
    require_output_absent: bool,
) -> dict[str, Any]:
    root_value = Path(expected_output_root).expanduser()
    if not root_value.is_absolute():
        raise BaseFinalAuthorityError(
            "expected_output_root must be absolute"
        )
    output_root = root_value.resolve()
    if require_output_absent and output_root.exists():
        raise BaseFinalAuthorityError(
            "pre-inference expected output root already exists"
        )
    source = _validate_source(inference_source)
    canonical, canonical_by_id = _canonical_authority(
        canonical_manifest,
        canonical_summary,
        canonical_lineage,
        canonical_root_receipt,
    )
    audio = _audio_authority(
        audio_authorities,
        canonical=canonical,
        canonical_by_id=canonical_by_id,
    )
    base_long_candidate_bundle = _replay_base_long_candidate_bundle(
        base_long_candidate_artifacts,
        inference_source=source,
    )
    checkpoint_bundle = _checkpoint_bundle(checkpoints)
    control = _control_authority(
        expected_output_root=output_root,
        base_long_candidate_bundle=base_long_candidate_bundle,
        winner_selection=winner_selection,
        continuation_decision=continuation_decision,
        continuation_waves=continuation_waves,
        winner_full_metric_closure=winner_full_metric_closure,
        test_claim=test_claim,
        checkpoints=checkpoint_bundle,
    )
    if (
        control["winner_selection"].get("selection_protocol")
        != BASE_SELECTION_PROTOCOL
        or control["winner_selection"].get("selection_metric")
        != BASE_SELECTION_METRIC
    ):
        raise BaseFinalAuthorityError(
            "final Base authority selection protocol changed"
        )
    return {
        "expected_output_root": str(output_root),
        "canonical": canonical,
        "audio": audio,
        **control,
        "inference_source": source,
        "checkpoints": checkpoint_bundle,
        "contract": {
            "generator": "SemTalk Base-only",
            "split": "test",
            "test_clips": TEST_CLIPS,
            "test_global_start": TEST_GLOBAL_START,
            "test_global_stop": TEST_GLOBAL_STOP,
            "num_shards": NUM_SHARDS,
            "expected_shard_ids": list(range(NUM_SHARDS)),
            "exact_once": True,
            "deterministic": True,
            "whole_candidate_per_gpu": True,
            "base_candidate_epochs": base_long_candidate_bundle[
                "candidate_epochs"
            ],
            "base_updates_per_epoch": base_long_candidate_bundle[
                "updates_per_epoch"
            ],
            "selected_base_epoch": control["winner_selection"][
                "selected_epoch"
            ],
            "base_selection_protocol": BASE_SELECTION_PROTOCOL,
            "base_selection_metric": BASE_SELECTION_METRIC,
            "selection_split": "val",
            "test_evaluations": 1,
            "test_feedback_into_selection": False,
            "final_metric_event": FINAL_METRIC_EVENT,
            "forbidden_generator_identities": list(
                FORBIDDEN_IDENTITIES
            ),
        },
    }


def _build_test_authority(
    *,
    expected_output_root: str | Path,
    canonical_manifest: Mapping[str, Any],
    canonical_summary: Mapping[str, Any],
    canonical_lineage: Mapping[str, Any],
    canonical_root_receipt: Mapping[str, Any],
    audio_authorities: Sequence[Mapping[str, Any]],
    base_long_candidate_artifacts: Mapping[str, Mapping[str, Any]],
    winner_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    continuation_waves: Sequence[Mapping[str, Any]],
    winner_full_metric_closure: Mapping[str, Any],
    test_claim: Mapping[str, Any],
    inference_source: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
    require_output_absent: bool,
) -> dict[str, Any]:
    inputs = _authority_inputs(
        expected_output_root=expected_output_root,
        canonical_manifest=canonical_manifest,
        canonical_summary=canonical_summary,
        canonical_lineage=canonical_lineage,
        canonical_root_receipt=canonical_root_receipt,
        audio_authorities=audio_authorities,
        base_long_candidate_artifacts=base_long_candidate_artifacts,
        winner_selection=winner_selection,
        continuation_decision=continuation_decision,
        continuation_waves=continuation_waves,
        winner_full_metric_closure=winner_full_metric_closure,
        test_claim=test_claim,
        inference_source=inference_source,
        checkpoints=checkpoints,
        require_output_absent=require_output_absent,
    )
    authority: dict[str, Any] = {
        "format": FORMAT,
        "status": "authorized_pre_inference",
        "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
        **inputs,
    }
    authority["receipt_payload_sha256"] = canonical_json_sha256(authority)
    return authority


def build_test_authority(
    *,
    expected_output_root: str | Path,
    canonical_manifest: Mapping[str, Any],
    canonical_summary: Mapping[str, Any],
    canonical_lineage: Mapping[str, Any],
    canonical_root_receipt: Mapping[str, Any],
    audio_authorities: Sequence[Mapping[str, Any]],
    base_long_candidate_artifacts: Mapping[str, Mapping[str, Any]],
    winner_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    continuation_waves: Sequence[Mapping[str, Any]],
    winner_full_metric_closure: Mapping[str, Any],
    test_claim: Mapping[str, Any],
    inference_source: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one immutable pre-inference authority from explicit inputs."""

    return _build_test_authority(
        expected_output_root=expected_output_root,
        canonical_manifest=canonical_manifest,
        canonical_summary=canonical_summary,
        canonical_lineage=canonical_lineage,
        canonical_root_receipt=canonical_root_receipt,
        audio_authorities=audio_authorities,
        base_long_candidate_artifacts=base_long_candidate_artifacts,
        winner_selection=winner_selection,
        continuation_decision=continuation_decision,
        continuation_waves=continuation_waves,
        winner_full_metric_closure=winner_full_metric_closure,
        test_claim=test_claim,
        inference_source=inference_source,
        checkpoints=checkpoints,
        require_output_absent=True,
    )


def _replay_authority(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "format",
        "status",
        "payload_hash_algorithm",
        "expected_output_root",
        "canonical",
        "audio",
        "winner_selection",
        "continuation_decision",
        "continuation_waves",
        "winner_full_metric_closure",
        "prerequisite_selection",
        "test_claim",
        "base_long_candidate_bundle",
        "inference_source",
        "checkpoints",
        "contract",
        "receipt_payload_sha256",
    }:
        raise BaseFinalAuthorityError("test authority schema mismatch")
    claimed = _sha256(
        value["receipt_payload_sha256"],
        "test authority payload SHA",
    )
    unsigned = dict(value)
    del unsigned["receipt_payload_sha256"]
    if (
        canonical_json_sha256(unsigned) != claimed
        or value["format"] != FORMAT
        or value["status"] != "authorized_pre_inference"
        or value["payload_hash_algorithm"] != PAYLOAD_HASH_ALGORITHM
    ):
        raise BaseFinalAuthorityError(
            "test authority payload hash/identity mismatch"
        )
    canonical = value["canonical"]
    audio = value["audio"]
    if (
        type(value["inference_source"]) is not dict
        or set(value["inference_source"])
        != SOURCE_INPUT_KEYS | SOURCE_DERIVED_KEYS
    ):
        raise BaseFinalAuthorityError(
            "test authority inference source proof schema mismatch"
        )
    if (
        type(value["base_long_candidate_bundle"]) is not dict
        or set(value["base_long_candidate_bundle"])
        != BASE_LONG_BUNDLE_KEYS
        or type(
            value["base_long_candidate_bundle"].get("artifacts")
        )
        is not dict
        or set(value["base_long_candidate_bundle"]["artifacts"])
        != set(BASE_LONG_ARTIFACT_ROLES)
    ):
        raise BaseFinalAuthorityError(
            "test authority Base-long bundle proof schema mismatch"
        )
    producer_source = value["base_long_candidate_bundle"].get(
        "producer_source"
    )
    if (
        type(producer_source) is not dict
        or set(producer_source)
        != {
            "origin",
            "commit",
            "tree",
            "branch",
            "clean",
            "entrypoint",
            "entrypoint_sha256",
        }
        | PRODUCER_SOURCE_DERIVED_KEYS
    ):
        raise BaseFinalAuthorityError(
            "test authority Base-long producer proof schema mismatch"
        )
    if type(value["continuation_waves"]) is not list:
        raise BaseFinalAuthorityError(
            "test authority continuation wave inventory mismatch"
        )
    rebuilt = _build_test_authority(
        expected_output_root=value["expected_output_root"],
        canonical_manifest=canonical["manifest"],
        canonical_summary=canonical["summary"],
        canonical_lineage=canonical["lineage"],
        canonical_root_receipt=canonical["root_receipt"],
        audio_authorities=[
            {
                "shard_id": shard["shard_id"],
                "manifest": shard["manifest"],
                "summary": shard["summary"],
                "lineage": shard["lineage"],
            }
            for shard in audio
        ],
        base_long_candidate_artifacts=value[
            "base_long_candidate_bundle"
        ]["artifacts"],
        winner_selection={
            key: value["winner_selection"][key]
            for key in ARTIFACT_KEYS
        },
        continuation_decision={
            key: value["continuation_decision"][key]
            for key in ARTIFACT_KEYS
        },
        continuation_waves=[
            {
                key: wave[key]
                for key in (
                    "path",
                    "sha256",
                    "bytes",
                    "receipt_payload_sha256",
                )
            }
            for wave in value["continuation_waves"]
        ],
        winner_full_metric_closure={
            key: value["winner_full_metric_closure"][key]
            for key in ARTIFACT_KEYS
        },
        test_claim={
            key: value["test_claim"][key]
            for key in ARTIFACT_KEYS
        },
        inference_source={
            key: value["inference_source"][key]
            for key in SOURCE_INPUT_KEYS
        },
        checkpoints=value["checkpoints"],
        require_output_absent=False,
    )
    if rebuilt != value:
        raise BaseFinalAuthorityError(
            "test authority differs from fresh replay"
        )
    return dict(value)


def validate_test_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_bytes: int,
    expected_receipt_payload_sha256: str,
) -> dict[str, Any]:
    """Fresh-validate an externally pinned authority file."""

    artifact = {
        "path": str(Path(path).expanduser().resolve()),
        "sha256": _sha256(
            expected_file_sha256,
            "test authority file SHA",
        ),
        "bytes": _exact_int(
            expected_bytes,
            "test authority bytes",
            minimum=1,
        ),
    }
    _verified, payload = _verified_artifact(
        artifact,
        "test authority",
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaseFinalAuthorityError(
            "test authority is invalid JSON"
        ) from exc
    authority = _replay_authority(value)
    expected_payload = _sha256(
        expected_receipt_payload_sha256,
        "expected test authority payload SHA",
    )
    if authority["receipt_payload_sha256"] != expected_payload:
        raise BaseFinalAuthorityError(
            "test authority external payload pin mismatch"
        )
    return authority


def atomic_write_new(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create a canonical authority without clobbering an existing file."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("xb") as handle:
            created = True
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def _load_authority_inputs(
    path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    resolved, payload = _safe_file_snapshot(
        str(path),
        "final authority inputs",
    )
    if hashlib.sha256(payload).hexdigest() != _sha256(
        expected_file_sha256,
        "final authority inputs file SHA",
    ):
        raise BaseFinalAuthorityError(
            "final authority inputs external file pin mismatch"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaseFinalAuthorityError(
            "final authority inputs are invalid JSON"
        ) from exc
    expected_keys = {
        "format",
        "status",
        "selection_protocol",
        "test_policy",
        "expected_output_root",
        "canonical_manifest",
        "canonical_summary",
        "canonical_lineage",
        "canonical_root_receipt",
        "audio_authorities",
        "base_long_candidate_artifacts",
        "winner_selection",
        "continuation_decision",
        "continuation_waves",
        "winner_validation_metric_closure",
        "test_claim",
        "inference_source",
        "checkpoints",
        "receipt_payload_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise BaseFinalAuthorityError(
            "final authority inputs schema mismatch"
        )
    claimed = _sha256(
        value["receipt_payload_sha256"],
        "final authority inputs payload SHA",
    )
    unsigned = dict(value)
    del unsigned["receipt_payload_sha256"]
    if (
        canonical_json_sha256(unsigned) != claimed
        or value["format"] != INPUTS_FORMAT
        or value["status"] != "ready"
        or value["selection_protocol"] != BASE_SELECTION_PROTOCOL
        or value["test_policy"] != FINAL_TEST_POLICY
    ):
        raise BaseFinalAuthorityError(
            "final authority inputs identity/policy mismatch"
        )
    if str(resolved) != str(path):
        raise BaseFinalAuthorityError(
            "final authority inputs path is not canonical"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the sole final-test authority from a hash-pinned "
            "DiffSHEG validation winner"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs_path = args.inputs_json.expanduser()
    if not inputs_path.is_absolute():
        raise BaseFinalAuthorityError("--inputs-json must be absolute")
    output_path = args.output_json.expanduser()
    if not output_path.is_absolute():
        raise BaseFinalAuthorityError("--output-json must be absolute")
    canonical_output = output_path.parent.resolve() / output_path.name
    if canonical_output != output_path or os.path.lexists(output_path):
        raise BaseFinalAuthorityError(
            "final authority output must be canonical and absent"
        )
    inputs = _load_authority_inputs(
        inputs_path,
        expected_file_sha256=args.expected_inputs_sha256,
    )
    authority = build_test_authority(
        expected_output_root=inputs["expected_output_root"],
        canonical_manifest=inputs["canonical_manifest"],
        canonical_summary=inputs["canonical_summary"],
        canonical_lineage=inputs["canonical_lineage"],
        canonical_root_receipt=inputs["canonical_root_receipt"],
        audio_authorities=inputs["audio_authorities"],
        base_long_candidate_artifacts=inputs[
            "base_long_candidate_artifacts"
        ],
        winner_selection=inputs["winner_selection"],
        continuation_decision=inputs["continuation_decision"],
        continuation_waves=inputs["continuation_waves"],
        winner_full_metric_closure=inputs[
            "winner_validation_metric_closure"
        ],
        test_claim=inputs["test_claim"],
        inference_source=inputs["inference_source"],
        checkpoints=inputs["checkpoints"],
    )
    atomic_write_new(output_path, authority)
    output_payload = output_path.read_bytes()
    output_sha = hashlib.sha256(output_payload).hexdigest()
    try:
        validate_test_authority(
            output_path,
            expected_file_sha256=output_sha,
            expected_bytes=len(output_payload),
            expected_receipt_payload_sha256=authority[
                "receipt_payload_sha256"
            ],
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "authorized_pre_inference",
                "output": str(output_path),
                "sha256": output_sha,
                "bytes": len(output_payload),
                "receipt_payload_sha256": authority[
                    "receipt_payload_sha256"
                ],
                "selection_protocol": BASE_SELECTION_PROTOCOL,
                "selection_metric": BASE_SELECTION_METRIC,
                "test_evaluations": 1,
                "test_feedback_into_selection": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
