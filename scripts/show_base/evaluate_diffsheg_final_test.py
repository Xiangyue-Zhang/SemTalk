#!/usr/bin/env python3
"""Run the sole combined SemTalk SHOW final-test metric event.

This is an audit bridge, not another metric implementation.  It:

* fully replays the current immutable final-test authority, including its
  validation-only Base winner and selected-five prerequisite receipts;
* binds the selected checkpoint to the already-finalized eight-shard test
  inference bundle;
* requires exactly 1,708 canonical SHOW test clips, exactly once;
* pins PASPA's reconstructed DiffSHEG evaluator, DiffSHEG statistics, all
  three published SHOW autoencoders, and the sealed exact-byte original-WAV
  test-layout view produced by ``prepare_diffsheg_audio_view.py``;
* pins TalkSHOW and SMPL-X by commit/SHA;
* atomically consumes one non-retryable combined claim inside the finalized
  test namespace before either metric suite executes;
* accepts exactly the seven DiffSHEG paper-facing metrics; and
* evaluates TalkSHOW ``released2``/``paper16`` body plus released face by
  directly calling the pinned metric module over those same prediction
  artifacts in this same process and event.

There is no separately authorized TalkSHOW test evaluation.  A failure in
either suite leaves the claim consumed and cannot be retried.  GPU execution
of formal mode must be launched under ``/tmp/globaldiff_guarded_runner.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence
import wave


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import run_base_final_test as final_test
from scripts.show_base import evaluate_talkshow_show_metrics as talkshow_metrics


EXPECTED_TEST_CLIPS = 1_708
EXPECTED_NUM_SHARDS = 8
WINDOW_LENGTH = 88
WINDOW_STRIDE = 88
EXPECTED_METRICS = (
    "fmd",
    "fed",
    "expression_diversity",
    "fgd",
    "ba",
    "pcm",
    "gesture_diversity",
)
METRIC_DEFINITIONS = {
    "fmd": {
        "space": (
            "DiffSHEG-normalized gesture[129]+expression[103], "
            "gesture_expression AE latent Frechet distance"
        ),
        "trend": "lower",
    },
    "fed": {
        "space": (
            "DiffSHEG-normalized jaw[3]+expression[100], "
            "expression AE latent Frechet distance"
        ),
        "trend": "lower",
    },
    "expression_diversity": {
        "space": "generated normalized expression 88-frame windows, pairwise L1",
        "trend": "match_ground_truth_distribution",
    },
    "fgd": {
        "space": (
            "DiffSHEG-normalized gesture[129], gesture AE latent "
            "Frechet distance"
        ),
        "trend": "lower",
    },
    "ba": {
        "space": "TalkSHOW SMPL-X motion beats versus original-audio onsets",
        "trend": "higher",
    },
    "pcm": {
        "space": (
            "normalized gesture reshaped to 43 axis-angle 3-vectors; "
            "fraction with joint L2 error < 1.0"
        ),
        "trend": "higher",
    },
    "gesture_diversity": {
        "space": "generated normalized gesture 88-frame windows, pairwise L1",
        "trend": "match_ground_truth_distribution",
    },
}
SHOW_SPEAKERS = frozenset({"oliver", "chemistry", "seth", "conan"})
VALIDATION_PRIMARY_METRIC = "validation.diffsheg.metrics.fgd"
VALIDATION_SELECTION_PROTOCOL = "diffsheg_show_validation_fgd_v1"
INFERENCE_SUMMARY_FORMAT = "semtalk_show_base_inference_final_summary_v1"
INFERENCE_LINEAGE_FORMAT = "semtalk_show_base_inference_final_lineage_v1"
PREFLIGHT_FORMAT = "semtalk_show_combined_full_test_preflight_v3"
CLAIM_FORMAT = "semtalk_show_combined_full_test_claim_v3"
RESULT_FORMAT = "semtalk_show_combined_full_test_result_v3"
AUDIO_VIEW_FORMAT = "semtalk_show_diffsheg_audio_view_receipt_v1"
AUDIO_VIEW_MANIFEST_NAME = "diffsheg_audio_view_manifest.jsonl"
AUDIO_VIEW_RECEIPT_NAME = "diffsheg_audio_view_receipt.json"
CLAIM_NAME = "diffsheg-full-test-one-shot.claim.json"

PASPA_ORIGIN = "git@github.com:Ly403/PASPA.git"
PASPA_COMMIT = "0df27e6cab4b5ced19cc923afe352f77d547924b"
PASPA_TREE = "574662eaf3122beb5631c02c847456e752d77f0b"
PASPA_EVALUATOR_RELATIVE = Path("scripts/diffsheg_show_eval.py")
PASPA_EVALUATOR_SHA256 = (
    "21fa84fdb9c3f64eb2920714e1d27a685210a225bcffa78503452c4018a53f8c"
)
PASPA_PROTOCOL_DOC_RELATIVE = Path("configs/talkshow_profile/DIFFSHEG_EVAL_zh.md")
PASPA_PROTOCOL_DOC_SHA256 = (
    "c6ec216581a519e9f9ef45e8d280b796ece289164af770086b8a319fa22541ce"
)

DIFFSHEG_COMMIT = "3ebf3058f48cba3da9146afb7623e9ec1ab9e9a5"
DIFFSHEG_TREE = "b2b81733b02c04738f2fbe2ec6314480681b9cbf"
DIFFSHEG_ORIGIN = "https://github.com/JeremyCJM/DiffSHEG.git"
DIFFSHEG_STATS_RELATIVE = Path("data/SHOW/talkshow_mean_std.npy")
DIFFSHEG_STATS_SHA256 = (
    "b90320eba94d0777e7160fd31d0fe6f04a7c86822ac875fb7db5cf58d298cef0"
)
DIFFSHEG_AE_PINS = {
    "fmd": {
        "filename": "gesture_expression.pth.tar",
        "sha256": (
            "1f2c0003389e03a727e91ebced3eeafd7d04f6782b81cf2062e9a5642441627f"
        ),
        "input_dim": 232,
    },
    "fed": {
        "filename": "expression.pth.tar",
        "sha256": (
            "ebb75ecd2eaf36e52c7684a1767b889e4a28b41ea900ae6e036ff1be5d1aa707"
        ),
        "input_dim": 103,
    },
    "fgd": {
        "filename": "gesture.pth.tar",
        "sha256": (
            "5eaf9b882a5ccd5f6eb4385aaadf3d28f3ee4382360ecb13c12f4904b3c3216e"
        ),
        "input_dim": 129,
    },
}
TALKSHOW_COMMIT = "9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
TALKSHOW_ORIGIN = "https://github.com/yhw-yhw/TalkSHOW.git"
SMPLX_NEUTRAL_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)


class FinalDiffSHEGError(RuntimeError):
    """Raised when the formal full-test contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FinalDiffSHEGError(f"{label} must be a lowercase SHA-256")
    return value


def require_git_oid(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FinalDiffSHEGError(f"{label} must be a lowercase Git object ID")
    return value


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise FinalDiffSHEGError(
                    f"{label} has duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FinalDiffSHEGError(
                    f"{label} contains non-finite JSON token {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalDiffSHEGError(f"cannot parse {label}: {error}") from error


def _strict_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise FinalDiffSHEGError(f"cannot read {label}: {error}") from error
    if not payload.endswith(b"\n"):
        raise FinalDiffSHEGError(f"{label} must end with one newline")
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise FinalDiffSHEGError(
                f"{label}:{line_number} is unexpectedly empty"
            )
        value = _strict_json_bytes(line, f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise FinalDiffSHEGError(
                f"{label}:{line_number} must be an object"
            )
        rows.append(value)
    return rows


def _absolute(value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise FinalDiffSHEGError(f"{label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FinalDiffSHEGError(f"{label} must be absolute: {path}")
    return path


def _regular_file(value: Any, label: str) -> Path:
    path = _absolute(value, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FinalDiffSHEGError(f"{label} does not exist: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise FinalDiffSHEGError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return path.resolve(strict=True)


def _directory(value: Any, label: str) -> Path:
    path = _absolute(value, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FinalDiffSHEGError(f"{label} does not exist: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise FinalDiffSHEGError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _verified_file(
    value: Any,
    expected_sha256: Any,
    label: str,
) -> tuple[Path, bytes, str]:
    expected = require_sha256(expected_sha256, f"{label} expected SHA-256")
    path = _regular_file(value, label)
    payload = path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise FinalDiffSHEGError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return path, payload, observed


def _git_value(root: Path, arguments: Sequence[str], label: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalDiffSHEGError(
            f"cannot inspect {label} checkout {root}: {error}"
        ) from error
    value = result.stdout.strip()
    if not value:
        raise FinalDiffSHEGError(f"{label} Git query returned empty output")
    return value


def _load_pinned_paspa(value: Any) -> tuple[ModuleType, dict[str, Any]]:
    root = _directory(value, "PASPA checkout")
    head = _git_value(root, ["rev-parse", "HEAD^{commit}"], "PASPA")
    tree = _git_value(root, ["rev-parse", "HEAD^{tree}"], "PASPA")
    origin = _git_value(root, ["remote", "get-url", "origin"], "PASPA")
    if (head, tree, origin) != (PASPA_COMMIT, PASPA_TREE, PASPA_ORIGIN):
        raise FinalDiffSHEGError(
            "PASPA checkout is not the exact audited commit/tree/origin"
        )
    evaluator_path = _regular_file(
        root / PASPA_EVALUATOR_RELATIVE,
        "PASPA DiffSHEG evaluator",
    )
    evaluator_sha = sha256_file(evaluator_path)
    if evaluator_sha != PASPA_EVALUATOR_SHA256:
        raise FinalDiffSHEGError("PASPA DiffSHEG evaluator bytes changed")
    protocol_doc_path = _regular_file(
        root / PASPA_PROTOCOL_DOC_RELATIVE,
        "PASPA DiffSHEG protocol document",
    )
    protocol_doc_sha = sha256_file(protocol_doc_path)
    if protocol_doc_sha != PASPA_PROTOCOL_DOC_SHA256:
        raise FinalDiffSHEGError("PASPA DiffSHEG protocol document changed")
    name = "_semtalk_final_pinned_paspa_diffsheg_show_eval"
    spec = importlib.util.spec_from_file_location(name, evaluator_path)
    if spec is None or spec.loader is None:
        raise FinalDiffSHEGError(f"cannot import {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    expected = {
        "PROTOCOL_NAME": "diffsheg_show_reconstructed",
        "PROTOCOL_VERSION": 1,
        "DIFFSHEG_REFERENCE_COMMIT": DIFFSHEG_COMMIT,
        "WINDOW_LENGTH": WINDOW_LENGTH,
        "DEFAULT_WINDOW_STRIDE": WINDOW_STRIDE,
    }
    if any(getattr(module, key, None) != expected_value for key, expected_value in expected.items()):
        raise FinalDiffSHEGError("PASPA DiffSHEG protocol constants changed")
    expected_specs = {
        (
            metric,
            {"fmd": "holistic", "fed": "expression", "fgd": "gesture"}[metric],
            pin["input_dim"],
            pin["filename"],
        )
        for metric, pin in DIFFSHEG_AE_PINS.items()
    }
    if set(getattr(module, "FEATURE_SPECS", ())) != expected_specs:
        raise FinalDiffSHEGError("PASPA three-autoencoder feature specs changed")
    return module, {
        "root": str(root),
        "origin": origin,
        "commit": head,
        "tree": tree,
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": evaluator_sha,
        },
        "protocol_document": {
            "path": str(protocol_doc_path),
            "sha256": protocol_doc_sha,
        },
    }


def _load_current_authority(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the current final authority and expose only immutable pins."""

    try:
        authority = final_test._validated_authority(args)
    except Exception as error:
        raise FinalDiffSHEGError(
            f"fresh final-test authority replay failed: {error}"
        ) from error
    final_root = _directory(
        args.inference_final_root,
        "finalized Base inference root",
    )
    expected_root, _shards_root = final_test._output_roots(authority)
    if expected_root != final_root:
        raise FinalDiffSHEGError(
            "finalized Base inference root differs from fresh authority"
        )
    winner = authority.get("winner_selection")
    test_claim = authority.get("test_claim")
    checkpoints = authority.get("checkpoints")
    source = authority.get("inference_source")
    if (
        not isinstance(winner, dict)
        or not isinstance(test_claim, dict)
        or not isinstance(checkpoints, dict)
        or not isinstance(source, dict)
    ):
        raise FinalDiffSHEGError("fresh authority control schema mismatch")
    if Path(str(source.get("source_root", ""))).resolve() != PROJECT_ROOT.resolve():
        raise FinalDiffSHEGError(
            "DiffSHEG adapter is not running from the inference source tree"
        )
    selected = winner.get("selected_checkpoint")
    if not isinstance(selected, dict) or set(selected) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise FinalDiffSHEGError("selected Base checkpoint receipt changed")
    selected_path, _selected_payload, selected_sha = _verified_file(
        selected["path"],
        selected["sha256"],
        "selected Base checkpoint",
    )
    if selected_path.stat().st_size != selected["bytes"]:
        raise FinalDiffSHEGError("selected Base checkpoint byte count changed")
    expected_policy = {
        "authorized_evaluations": 1,
        "one_shot_claim_required": True,
        "selection_feedback": False,
        "num_shards": EXPECTED_NUM_SHARDS,
        "canonical_test_clips": EXPECTED_TEST_CLIPS,
        "final_metric_event": final_test.final_authority.FINAL_METRIC_EVENT,
    }
    if (
        test_claim.get("test_policy") != expected_policy
        or authority.get("contract", {}).get("final_metric_event")
        != final_test.final_authority.FINAL_METRIC_EVENT
    ):
        raise FinalDiffSHEGError(
            "fresh authority does not authorize the sole combined final "
            "metric event"
        )
    authority_artifact = final_test._authority_artifact(args)
    receipt = {
        "fresh_test_authority": authority_artifact,
        "winner_selection": {
            key: winner[key]
            for key in (
                "path",
                "sha256",
                "bytes",
                "receipt_payload_sha256",
                "canonical_payload_sha256",
                "selected_epoch",
                "selected_optimizer_updates",
            )
        },
        "selected_checkpoint": {
            "path": str(selected_path),
            "sha256": selected_sha,
            "bytes": selected["bytes"],
        },
        "selected_prerequisite_sha256": {
            stage: checkpoints[stage]["sha256"]
            for stage in final_test.REPRESENTATION_STAGES
        },
        "selection": {
            "split": "val",
            "primary_metric": VALIDATION_PRIMARY_METRIC,
            "protocol": VALIDATION_SELECTION_PROTOCOL,
            "validation_only_for_selection": True,
            "test_visible_during_selection": False,
            "test_feedback_into_selection": False,
        },
        "test_policy": expected_policy,
        "final_metric_event": final_test.final_authority.FINAL_METRIC_EVENT,
        "source": {
            key: source[key]
            for key in (
                "source_root",
                "origin",
                "commit",
                "tree",
                "entrypoint",
                "entrypoint_sha256",
                "clean",
                "detached",
                "local_branches_at_commit",
            )
        },
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    return authority, receipt

def _artifact_receipt(
    value: Any,
    expected_name: str,
    expected_parent: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise FinalDiffSHEGError(f"{label} receipt schema mismatch")
    path = _regular_file(value["path"], label)
    if path.parent != expected_parent or path.name != expected_name:
        raise FinalDiffSHEGError(f"{label} escapes the finalized NPZ root")
    expected_sha = require_sha256(value["sha256"], f"{label} SHA")
    if (
        isinstance(value["bytes"], bool)
        or not isinstance(value["bytes"], int)
        or value["bytes"] < 1
        or path.stat().st_size != value["bytes"]
        or sha256_file(path) != expected_sha
    ):
        raise FinalDiffSHEGError(f"{label} bytes/SHA changed")
    return path, {
        "path": str(path),
        "bytes": value["bytes"],
        "sha256": expected_sha,
    }


def _validate_inference_bundle(
    args: argparse.Namespace,
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Bind the exact finalized 8-shard output to PASPA's input contract."""

    try:
        manifest, rows, lineage = final_test._load_final_rows(authority)
        expected_contract = final_test._contract(authority, args)
    except Exception as error:
        raise FinalDiffSHEGError(
            f"final inference replay failed: {error}"
        ) from error
    final_root, _shards_root = final_test._output_roots(authority)
    summary_path = _regular_file(
        final_root / "final_summary.json",
        "final inference summary",
    )
    lineage_path = _regular_file(
        final_root / "final_lineage.json",
        "final inference lineage",
    )
    manifest_path = _regular_file(
        final_root / "final_manifest.jsonl",
        "final inference manifest",
    )
    clip_manifest_path = _regular_file(
        final_root / "diffsheg_eval_clip_ids.txt",
        "DiffSHEG test clip manifest",
    )
    npz_root = _directory(final_root / "npz" / "test", "final NPZ root")
    summary = _strict_json_bytes(
        summary_path.read_bytes(),
        "final inference summary",
    )
    if not isinstance(summary, dict):
        raise FinalDiffSHEGError("final summary must be a JSON object")
    expected_summary_keys = {
        "format",
        "status",
        "generator",
        "test_clips",
        "prediction_files",
        "ground_truth_files",
        "num_shards",
        "npz_root",
        "manifest_sha256",
        "clip_manifest_sha256",
        "lineage_sha256",
        "contract_sha256",
        "runtime_sha256",
        "finite",
        "exact_once",
        "split_disjoint",
        "test_evaluations",
        "test_feedback_into_selection",
    }
    if (
        set(summary) != expected_summary_keys
        or summary["format"] != INFERENCE_SUMMARY_FORMAT
        or summary["status"] != "complete"
        or summary["generator"] != "SemTalk Base-only"
        or summary["test_clips"] != EXPECTED_TEST_CLIPS
        or summary["prediction_files"] != EXPECTED_TEST_CLIPS
        or summary["ground_truth_files"] != EXPECTED_TEST_CLIPS
        or summary["num_shards"] != EXPECTED_NUM_SHARDS
        or Path(str(summary["npz_root"])).resolve() != npz_root
        or summary["finite"] is not True
        or summary["exact_once"] is not True
        or summary["split_disjoint"] is not True
        or summary["test_evaluations"] != 0
        or summary["test_feedback_into_selection"] is not False
    ):
        raise FinalDiffSHEGError("final inference summary contract mismatch")
    if (
        lineage.get("format") != INFERENCE_LINEAGE_FORMAT
        or lineage.get("status") != "complete"
        or lineage.get("contract") != expected_contract
        or lineage.get("contract_sha256")
        != final_test._canonical_json_sha256(expected_contract)
        or summary["contract_sha256"] != lineage["contract_sha256"]
        or summary["runtime_sha256"] != lineage["runtime_sha256"]
        or len(lineage.get("shards", ())) != EXPECTED_NUM_SHARDS
    ):
        raise FinalDiffSHEGError("final inference lineage/authority mismatch")
    selection_policy = expected_contract.get("selection_policy")
    if selection_policy != {
        "primary_metric": VALIDATION_PRIMARY_METRIC,
        "protocol": VALIDATION_SELECTION_PROTOCOL,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 1,
        "test_feedback_into_selection": False,
    }:
        raise FinalDiffSHEGError(
            "final inference does not bind validation-only selection"
        )
    selected = authority["winner_selection"]["selected_checkpoint"]
    base_checkpoint = expected_contract.get("checkpoints", {}).get("base")
    if (
        not isinstance(base_checkpoint, dict)
        or Path(str(base_checkpoint.get("path", ""))).resolve()
        != Path(selected["path"]).resolve()
        or base_checkpoint.get("expected_sha256") != selected["sha256"]
        or expected_contract.get("test_clips") != EXPECTED_TEST_CLIPS
        or expected_contract.get("num_shards") != EXPECTED_NUM_SHARDS
        or expected_contract.get("split") != "test"
        or expected_contract.get("exact_once") is not True
        or expected_contract.get("physical_predictions_per_clip") != 1
    ):
        raise FinalDiffSHEGError(
            "final inference is not bound to the val-selected Base winner"
        )
    clip_payload = clip_manifest_path.read_bytes()
    if not clip_payload.endswith(b"\n"):
        raise FinalDiffSHEGError("DiffSHEG clip manifest lacks final newline")
    clip_ids = clip_payload.decode("utf-8", errors="strict").splitlines()
    observed_ids: list[str] = []
    observed_speakers: set[str] = set()
    input_rows: list[dict[str, Any]] = []
    for evaluation_index, row in enumerate(rows):
        output_id = row.get("canonical_clip_id")
        frames = row.get("frames")
        if (
            row.get("evaluation_index") != evaluation_index
            or not isinstance(output_id, str)
            or not output_id
            or "/" in output_id
            or "\\" in output_id
            or "__" not in output_id
            or type(frames) is not int
            or frames < WINDOW_LENGTH
        ):
            raise FinalDiffSHEGError(
                f"invalid or sub-{WINDOW_LENGTH}-frame test row at "
                f"position {evaluation_index}"
            )
        speaker = output_id.split("__", 1)[0]
        if speaker not in SHOW_SPEAKERS or row.get("speaker") != speaker:
            raise FinalDiffSHEGError(f"non-canonical SHOW clip {output_id}")
        observed_ids.append(output_id)
        observed_speakers.add(speaker)
        input_rows.append(
            {
                "evaluation_index": evaluation_index,
                "global_index": row["global_index"],
                "canonical_clip_id": output_id,
                "frames": frames,
                "prediction_sha256": row["prediction"]["sha256"],
                "prediction_bytes": row["prediction"]["bytes"],
                "ground_truth_sha256": row["ground_truth"]["sha256"],
                "ground_truth_bytes": row["ground_truth"]["bytes"],
            }
        )
    if (
        len(rows) != EXPECTED_TEST_CLIPS
        or observed_ids != sorted(observed_ids)
        or len(set(observed_ids)) != EXPECTED_TEST_CLIPS
        or clip_ids != observed_ids
        or observed_speakers != SHOW_SPEAKERS
    ):
        raise FinalDiffSHEGError(
            "final inference clip IDs/order/coverage are not canonical exact-once"
        )
    manifest_sha = sha256_file(manifest_path)
    clip_manifest_sha = sha256_file(clip_manifest_path)
    lineage_sha = sha256_file(lineage_path)
    if (
        manifest_sha != manifest["sha256"]
        or manifest_sha != lineage["final_manifest_sha256"]
        or manifest_sha != summary["manifest_sha256"]
        or clip_manifest_sha != lineage["clip_manifest_sha256"]
        or clip_manifest_sha != summary["clip_manifest_sha256"]
        or final_test._canonical_json_sha256(lineage) != summary["lineage_sha256"]
    ):
        raise FinalDiffSHEGError("final inference metadata SHA closure failed")
    input_set_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(row) for row in input_rows)
    ).hexdigest()
    return {
        "root": str(final_root),
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        },
        "lineage": {
            "path": str(lineage_path),
            "sha256": lineage_sha,
            "bytes": lineage_path.stat().st_size,
            "contract_sha256": lineage["contract_sha256"],
            "runtime_sha256": lineage["runtime_sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "bytes": manifest_path.stat().st_size,
        },
        "clip_manifest": {
            "path": str(clip_manifest_path),
            "sha256": clip_manifest_sha,
            "bytes": clip_manifest_path.stat().st_size,
        },
        "npz_root": str(npz_root),
        "input_set_sha256": input_set_sha,
        "clip_count": EXPECTED_TEST_CLIPS,
        "prediction_files": EXPECTED_TEST_CLIPS,
        "ground_truth_files": EXPECTED_TEST_CLIPS,
        "minimum_frames": min(row["frames"] for row in input_rows),
        "exact_once": True,
        "finite": True,
        "window_length": WINDOW_LENGTH,
        "window_stride": WINDOW_STRIDE,
    }, observed_ids

def _validate_assets(
    evaluator: ModuleType,
    diffsheg_root_value: Any,
    talkshow_root_value: Any,
    smplx_path_value: Any,
) -> dict[str, Any]:
    diffsheg_input = _absolute(diffsheg_root_value, "DiffSHEG checkout")
    diffsheg_root = _directory(diffsheg_input, "DiffSHEG checkout")
    if diffsheg_input != diffsheg_root:
        raise FinalDiffSHEGError("DiffSHEG checkout must be canonical")
    diffsheg_head = _git_value(
        diffsheg_root,
        ["rev-parse", "HEAD^{commit}"],
        "DiffSHEG",
    )
    if diffsheg_head != DIFFSHEG_COMMIT:
        raise FinalDiffSHEGError(
            f"DiffSHEG HEAD {diffsheg_head} != {DIFFSHEG_COMMIT}"
        )
    diffsheg_origin = _git_value(
        diffsheg_root,
        ["remote", "get-url", "origin"],
        "DiffSHEG",
    )
    if diffsheg_origin != DIFFSHEG_ORIGIN:
        raise FinalDiffSHEGError("DiffSHEG origin changed")
    diffsheg_tree = _git_value(
        diffsheg_root,
        ["rev-parse", "HEAD^{tree}"],
        "DiffSHEG",
    )
    detached = subprocess.run(
        ["git", "-C", str(diffsheg_root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).returncode != 0
    local_branches = _git_value(
        diffsheg_root,
        ["for-each-ref", "--format=%(refname)", "refs/heads"],
        "DiffSHEG",
    ).splitlines()
    dirty = _git_value(
        diffsheg_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        "DiffSHEG",
    )
    if (
        diffsheg_tree != DIFFSHEG_TREE
        or not detached
        or local_branches
        or dirty
    ):
        raise FinalDiffSHEGError(
            "DiffSHEG checkout is not the clean detached pinned bundle"
        )
    stats = _regular_file(
        diffsheg_root / DIFFSHEG_STATS_RELATIVE,
        "DiffSHEG SHOW statistics",
    )
    if sha256_file(stats) != DIFFSHEG_STATS_SHA256:
        raise FinalDiffSHEGError("DiffSHEG SHOW statistics bytes changed")
    _no_symlink_below(diffsheg_root, stats, "DiffSHEG SHOW statistics")
    autoencoders: dict[str, Any] = {}
    for metric, pin in DIFFSHEG_AE_PINS.items():
        path = _regular_file(
            diffsheg_root / "data" / "SHOW" / "ae_weights" / pin["filename"],
            f"DiffSHEG {metric} autoencoder",
        )
        observed = sha256_file(path)
        if observed != pin["sha256"]:
            raise FinalDiffSHEGError(
                f"DiffSHEG {metric} autoencoder bytes changed"
            )
        _no_symlink_below(
            diffsheg_root,
            path,
            f"DiffSHEG {metric} autoencoder",
        )
        autoencoders[metric] = {
            "path": str(path),
            "sha256": observed,
            "input_dim": pin["input_dim"],
        }
    talkshow_root = _directory(talkshow_root_value, "TalkSHOW checkout")
    talkshow_head = _git_value(
        talkshow_root,
        ["rev-parse", "HEAD^{commit}"],
        "TalkSHOW",
    )
    talkshow_origin = _git_value(
        talkshow_root,
        ["remote", "get-url", "origin"],
        "TalkSHOW",
    )
    talkshow_tree = _git_value(
        talkshow_root,
        ["rev-parse", "HEAD^{tree}"],
        "TalkSHOW",
    )
    if talkshow_head != TALKSHOW_COMMIT or talkshow_origin != TALKSHOW_ORIGIN:
        raise FinalDiffSHEGError(
            "TalkSHOW checkout is not the exact pinned official source"
        )
    smplx_path = _absolute(smplx_path_value, "SMPL-X model path")
    try:
        smplx_asset = evaluator._resolve_smplx_asset(smplx_path.resolve())
    except evaluator.ProtocolError as error:
        raise FinalDiffSHEGError(str(error)) from error
    smplx_asset = _regular_file(smplx_asset, "SMPL-X neutral asset")
    observed_smplx = sha256_file(smplx_asset)
    if observed_smplx != SMPLX_NEUTRAL_SHA256:
        raise FinalDiffSHEGError(
            "SMPL-X neutral asset SHA changed"
        )
    return {
        "diffsheg": {
            "root": str(diffsheg_root),
            "origin": diffsheg_origin,
            "commit": diffsheg_head,
            "tree": diffsheg_tree,
            "detached": True,
            "local_branches": [],
            "clean": True,
            "stats": {"path": str(stats), "sha256": DIFFSHEG_STATS_SHA256},
            "autoencoders": autoencoders,
        },
        "talkshow": {
            "root": str(talkshow_root),
            "origin": talkshow_origin,
            "commit": talkshow_head,
            "tree": talkshow_tree,
        },
        "smplx": {
            "model_path": str(smplx_path.resolve()),
            "asset_path": str(smplx_asset),
            "sha256": observed_smplx,
        },
    }


def _wav_metadata(payload: bytes, label: str) -> dict[str, Any]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            metadata = {
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "frames": handle.getnframes(),
                "compression": handle.getcomptype(),
            }
    except (OSError, EOFError, wave.Error) as error:
        raise FinalDiffSHEGError(f"{label} is incomplete: {error}") from error
    if (
        metadata["channels"] < 1
        or metadata["sample_width"] < 1
        or metadata["sample_rate"] < 1
        or metadata["frames"] < 1
        or metadata["compression"] != "NONE"
    ):
        raise FinalDiffSHEGError(f"{label} has invalid PCM metadata")
    return metadata


def _view_audio_path(
    root: Path,
    source_clip_id: Any,
    canonical_clip_id: Any,
) -> tuple[Path, str]:
    if type(source_clip_id) is not str or type(canonical_clip_id) is not str:
        raise FinalDiffSHEGError("audio-view clip IDs must be strings")
    pieces = source_clip_id.split("/")
    if (
        len(pieces) != 3
        or any(not piece or piece in {".", ".."} for piece in pieces)
        or any("/" in piece or "\\" in piece for piece in pieces)
        or final_test.canonical_clip_id(source_clip_id) != canonical_clip_id
    ):
        raise FinalDiffSHEGError(
            f"invalid source/canonical audio binding {source_clip_id!r} -> "
            f"{canonical_clip_id!r}"
        )
    speaker, video, sequence = pieces
    relative = Path(speaker) / video / "test" / sequence / f"{sequence}.wav"
    return root / relative, relative.as_posix()


def _no_symlink_below(root: Path, path: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise FinalDiffSHEGError(f"{label} escapes its sealed root") from error
    cursor = root
    for piece in relative.parts:
        cursor = cursor / piece
        try:
            mode = os.lstat(cursor).st_mode
        except FileNotFoundError:
            raise FinalDiffSHEGError(f"{label} is missing: {cursor}") from None
        if stat.S_ISLNK(mode):
            raise FinalDiffSHEGError(f"{label} contains a symlink: {cursor}")


def _audio_set_sha(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    payloads: list[bytes] = []
    for row in rows:
        item = row[key]
        payloads.append(
            canonical_json_bytes(
                {
                    "clip_id": row["clip_id"],
                    "source_clip_id": row["source_clip_id"],
                    "relative_path": item["relative_path"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "wav": row["wav"],
                }
            )
        )
    return hashlib.sha256(b"".join(payloads)).hexdigest()


def _validate_audio_view(
    inference: Mapping[str, Any],
    clip_ids: Sequence[str],
    root_value: Any,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    """Validate the CPU-prepared, exact-once TalkSHOW ``test`` WAV view."""

    root_input = _absolute(root_value, "sealed DiffSHEG source-audio view")
    root = _directory(root_input, "sealed DiffSHEG source-audio view")
    if root_input != root:
        raise FinalDiffSHEGError("DiffSHEG source-audio view is not canonical")
    receipt_path = _regular_file(
        root / AUDIO_VIEW_RECEIPT_NAME,
        "DiffSHEG source-audio view receipt",
    )
    manifest_path = _regular_file(
        root / AUDIO_VIEW_MANIFEST_NAME,
        "DiffSHEG source-audio view manifest",
    )
    _no_symlink_below(root, receipt_path, "audio-view receipt")
    _no_symlink_below(root, manifest_path, "audio-view manifest")
    receipt = _strict_json_bytes(
        receipt_path.read_bytes(),
        "DiffSHEG source-audio view receipt",
    )
    expected_receipt_keys = {
        "format",
        "status",
        "source_root",
        "output_root",
        "canonical_manifest",
        "inference_manifest_sha256",
        "clip_manifest_sha256",
        "clip_count",
        "exact_once",
        "symlinks",
        "padding",
        "truncation",
        "fabrication",
        "manifest",
        "source_ordered_set_sha256",
        "view_ordered_set_sha256",
        "receipt_payload_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_receipt_keys
        or receipt["format"] != AUDIO_VIEW_FORMAT
        or receipt["status"] != "complete"
        or Path(str(receipt["output_root"])).resolve() != root
        or receipt["inference_manifest_sha256"]
        != inference["manifest"]["sha256"]
        or receipt["clip_manifest_sha256"]
        != inference["clip_manifest"]["sha256"]
        or receipt["clip_count"] != EXPECTED_TEST_CLIPS
        or receipt["exact_once"] is not True
        or receipt["symlinks"] != "forbidden"
        or receipt["padding"] != "forbidden"
        or receipt["truncation"] != "forbidden"
        or receipt["fabrication"] != "forbidden"
        or canonical_json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_payload_sha256"
            }
        )
        != receipt["receipt_payload_sha256"]
    ):
        raise FinalDiffSHEGError("DiffSHEG source-audio view receipt mismatch")
    canonical_manifest = receipt["canonical_manifest"]
    if (
        not isinstance(canonical_manifest, dict)
        or not isinstance(canonical_manifest.get("path"), str)
        or type(canonical_manifest.get("bytes")) is not int
        or canonical_manifest["bytes"] < 1
        or require_sha256(
            canonical_manifest.get("sha256"),
            "canonical test manifest SHA",
        )
        != canonical_manifest["sha256"]
    ):
        raise FinalDiffSHEGError("audio view lacks canonical manifest authority")
    source_root = _directory(
        receipt["source_root"],
        "sealed audio-view original source root",
    )
    if Path(str(receipt["source_root"])) != source_root:
        raise FinalDiffSHEGError("audio-view original source root is not canonical")
    manifest_receipt = receipt["manifest"]
    if (
        not isinstance(manifest_receipt, dict)
        or set(manifest_receipt) != {"path", "bytes", "sha256"}
        or Path(str(manifest_receipt["path"])).resolve() != manifest_path
        or manifest_receipt["bytes"] != manifest_path.stat().st_size
        or require_sha256(
            manifest_receipt["sha256"], "audio-view manifest SHA"
        )
        != sha256_file(manifest_path)
    ):
        raise FinalDiffSHEGError("DiffSHEG audio-view manifest receipt changed")
    inference_rows = _strict_jsonl(
        Path(inference["manifest"]["path"]),
        "final inference manifest for audio view",
    )
    expected = [
        (
            row.get("canonical_clip_id"),
            row.get("source_clip_id"),
        )
        for row in inference_rows
    ]
    if [item[0] for item in expected] != list(clip_ids):
        raise FinalDiffSHEGError("audio view is not bound to final clip order")
    rows = _strict_jsonl(manifest_path, "DiffSHEG source-audio view manifest")
    expected_row_keys = {
        "evaluation_index",
        "clip_id",
        "source_clip_id",
        "source",
        "view",
        "wav",
    }
    paths: dict[str, Path] = {}
    expected_wavs: set[Path] = set()
    for index, (row, binding) in enumerate(zip(rows, expected)):
        if (
            not isinstance(row, dict)
            or set(row) != expected_row_keys
            or row["evaluation_index"] != index
            or (row["clip_id"], row["source_clip_id"]) != binding
            or set(row["source"])
            != {"path", "relative_path", "bytes", "sha256"}
            or set(row["view"])
            != {"path", "relative_path", "bytes", "sha256", "materialization"}
            or set(row["wav"])
            != {"channels", "sample_width", "sample_rate", "frames", "compression"}
        ):
            raise FinalDiffSHEGError(f"audio-view row {index} schema mismatch")
        expected_path, expected_relative = _view_audio_path(
            root,
            row["source_clip_id"],
            row["clip_id"],
        )
        view = row["view"]
        source = row["source"]
        source_pieces = row["source_clip_id"].split("/")
        source_relative = (
            Path(*source_pieces)
            / f"{source_pieces[-1]}.wav"
        ).as_posix()
        expected_source_path = source_root / source_relative
        _no_symlink_below(
            source_root,
            expected_source_path,
            f"{row['clip_id']} original source WAV",
        )
        source_path = _regular_file(
            expected_source_path,
            f"{row['clip_id']} original source WAV",
        )
        _no_symlink_below(root, expected_path, f"{row['clip_id']} audio-view WAV")
        path = _regular_file(expected_path, f"{row['clip_id']} audio-view WAV")
        if (
            view["relative_path"] != expected_relative
            or Path(str(view["path"])).resolve() != path
            or source["relative_path"] != source_relative
            or Path(str(source["path"])).resolve() != source_path
            or view["materialization"] not in {"hardlink", "copy"}
            or type(view["bytes"]) is not int
            or view["bytes"] < 1
            or path.stat().st_size != view["bytes"]
            or require_sha256(view["sha256"], f"{row['clip_id']} view SHA")
            != sha256_file(path)
            or source["bytes"] != view["bytes"]
            or source["sha256"] != view["sha256"]
            or source_path.stat().st_size != source["bytes"]
            or sha256_file(source_path) != source["sha256"]
            or _wav_metadata(
                source_path.read_bytes(),
                f"{row['clip_id']} original source WAV",
            )
            != row["wav"]
            or _wav_metadata(path.read_bytes(), f"{row['clip_id']} audio-view WAV")
            != row["wav"]
        ):
            raise FinalDiffSHEGError(f"{row['clip_id']} audio-view bytes changed")
        paths[row["clip_id"]] = path
        expected_wavs.add(path)
    if (
        len(rows) != EXPECTED_TEST_CLIPS
        or len(paths) != EXPECTED_TEST_CLIPS
        or list(paths) != list(clip_ids)
        or _audio_set_sha(rows, "source")
        != receipt["source_ordered_set_sha256"]
        or _audio_set_sha(rows, "view") != receipt["view_ordered_set_sha256"]
    ):
        raise FinalDiffSHEGError("DiffSHEG audio-view exact-once closure failed")
    observed_files: set[Path] = set()
    observed_regulars: set[Path] = set()
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            child = current / name
            if stat.S_ISLNK(os.lstat(child).st_mode):
                raise FinalDiffSHEGError(f"audio view contains symlink {child}")
        for name in filenames:
            child = current / name
            if stat.S_ISLNK(os.lstat(child).st_mode):
                raise FinalDiffSHEGError(f"audio view contains symlink {child}")
            observed_regulars.add(child.resolve())
            if child.suffix == ".wav":
                observed_files.add(child.resolve())
    expected_regulars = expected_wavs | {receipt_path, manifest_path}
    if observed_files != expected_wavs or observed_regulars != expected_regulars:
        raise FinalDiffSHEGError("audio view contains missing or extra files")
    return root, paths, {
        "receipt": {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": sha256_file(receipt_path),
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        },
        "manifest": dict(manifest_receipt),
        "source_root": receipt["source_root"],
        "source_ordered_set_sha256": receipt["source_ordered_set_sha256"],
        "view_ordered_set_sha256": receipt["view_ordered_set_sha256"],
    }


def _validate_protocol_inputs(
    evaluator: ModuleType,
    inference: Mapping[str, Any],
    clip_ids: Sequence[str],
    assets: Mapping[str, Any],
    source_audio_root_value: Any,
    expected_audio_set_sha256: Any | None,
) -> tuple[Any, dict[str, Any]]:
    clip_manifest = Path(inference["clip_manifest"]["path"])
    audio_root, prepared_audio_paths, audio_view = _validate_audio_view(
        inference,
        clip_ids,
        source_audio_root_value,
    )
    try:
        validation = evaluator.validate_inputs(
            prediction_dir=Path(inference["npz_root"]),
            ground_truth_dir=Path(inference["npz_root"]),
            clip_manifest=clip_manifest,
            window_stride=WINDOW_STRIDE,
            require_betas=True,
        )
        evaluator.load_normalization_stats(
            Path(assets["diffsheg"]["stats"]["path"])
        )
        audio_paths, audio_protocol = evaluator._resolve_audio_paths(
            validation,
            audio_dir=None,
            source_audio_root=audio_root,
        )
    except evaluator.ProtocolError as error:
        raise FinalDiffSHEGError(str(error)) from error
    if (
        len(validation.clips) != EXPECTED_TEST_CLIPS
        or tuple(clip.clip_id for clip in validation.clips) != tuple(clip_ids)
        or validation.clip_order != f"explicit_manifest:{clip_manifest}"
        or audio_protocol != "talkshow_original_source"
        or set(audio_paths) != set(clip_ids)
        or {
            clip_id: Path(path).resolve()
            for clip_id, path in audio_paths.items()
        }
        != prepared_audio_paths
    ):
        raise FinalDiffSHEGError(
            "PASPA validation did not preserve canonical test/audio coverage"
        )
    audio_rows: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        path = _regular_file(audio_paths[clip_id], f"{clip_id} source WAV")
        wav_metadata = _wav_metadata(
            path.read_bytes(),
            f"{clip_id} source WAV",
        )
        audio_rows.append(
            {
                "clip_id": clip_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "wav": wav_metadata,
            }
        )
    audio_set_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(row) for row in audio_rows)
    ).hexdigest()
    if expected_audio_set_sha256 is not None:
        expected_audio = require_sha256(
            expected_audio_set_sha256,
            "expected original source-audio set SHA",
        )
        if audio_set_sha != expected_audio:
            raise FinalDiffSHEGError(
                f"source-audio set SHA {audio_set_sha} != {expected_audio}"
            )
    return validation, {
        "protocol": "talkshow_original_source",
        "root": str(audio_root),
        "clip_count": len(audio_rows),
        "ordered_manifest_sha256": audio_set_sha,
        "sealed_view": audio_view,
    }


def _talkshow_metric_preflight(
    args: argparse.Namespace,
    authority: Mapping[str, Any],
    inference: Mapping[str, Any],
    shared_assets: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the second suite without consuming a test evaluation.

    The deterministic distribution receipt contains no generated sample.  It
    only binds the one already-finalized prediction artifact per clip to the
    released logical-slot interpretation.  Building it during CPU preflight
    therefore cannot create a second inference or metric observation.
    """

    gate_path, gate_payload, gate_sha = _verified_file(
        args.talkshow_validation_gate_json,
        args.expected_talkshow_validation_gate_sha256,
        "TalkSHOW final-winner replication gate",
    )
    gate = _strict_json_bytes(
        gate_payload,
        "TalkSHOW final-winner replication gate",
    )
    expected_gate_payload_sha = require_sha256(
        args.expected_talkshow_validation_gate_receipt_payload_sha256,
        "TalkSHOW final-winner gate payload SHA",
    )
    if (
        type(gate) is not dict
        or gate.get("receipt_payload_sha256") != expected_gate_payload_sha
    ):
        raise FinalDiffSHEGError(
            "TalkSHOW final-winner gate payload pin changed"
        )
    gate_artifact = {
        "path": str(gate_path),
        "sha256": gate_sha,
        "bytes": len(gate_payload),
        "receipt_payload_sha256": expected_gate_payload_sha,
    }
    try:
        observed_gate_artifact, _observed_gate = (
            talkshow_metrics._validate_external_validation_gate(
                gate_artifact,
                expected_scope="final_winner",
            )
        )
        metric_root = talkshow_metrics.validate_talkshow_metric_root(
            args.talkshow_metric_root
        )
    except talkshow_metrics.MetricAdapterContractError as error:
        raise FinalDiffSHEGError(
            f"TalkSHOW combined preflight failed: {error}"
        ) from error
    if observed_gate_artifact != gate_artifact:
        raise FinalDiffSHEGError(
            "TalkSHOW final-winner gate artifact changed during replay"
        )

    feature_path = _regular_file(
        args.talkshow_feature_extractor,
        "TalkSHOW released body feature extractor",
    )
    feature_sha = sha256_file(feature_path)
    if feature_sha != talkshow_metrics.FEATURE_EXTRACTOR_SHA256:
        raise FinalDiffSHEGError(
            "TalkSHOW released body feature extractor bytes changed"
        )
    smplx_asset = _regular_file(
        shared_assets["smplx"]["asset_path"],
        "TalkSHOW SMPL-X neutral asset",
    )
    if sha256_file(smplx_asset) != talkshow_metrics.SMPLX_SHA256:
        raise FinalDiffSHEGError("TalkSHOW SMPL-X asset bytes changed")

    manifest_path = Path(inference["manifest"]["path"])
    rows = _strict_jsonl(manifest_path, "shared final prediction manifest")
    prediction_records = [
        {
            "canonical_clip_id": row["canonical_clip_id"],
            "prediction_sha256": row["prediction"]["sha256"],
            "prediction_bytes": row["prediction"]["bytes"],
        }
        for row in rows
    ]
    if len(prediction_records) != EXPECTED_TEST_CLIPS:
        raise FinalDiffSHEGError(
            "TalkSHOW preflight prediction coverage changed"
        )
    prediction_manifest = dict(inference["manifest"])
    try:
        distribution = talkshow_metrics.build_distribution_receipt(
            prediction_manifest_artifact=prediction_manifest,
            prediction_artifacts=prediction_records,
            validation_gate=gate_artifact,
            expected_scope="final_winner",
        )
    except talkshow_metrics.MetricAdapterContractError as error:
        raise FinalDiffSHEGError(
            f"TalkSHOW distribution preflight failed: {error}"
        ) from error
    canonical = authority.get("canonical", {}).get("manifest")
    if type(canonical) is not dict or set(canonical) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise FinalDiffSHEGError(
            "combined authority canonical manifest receipt changed"
        )
    return {
        "validation_gate": gate_artifact,
        "distribution_receipt": distribution,
        "canonical_manifest": dict(canonical),
        "prediction_manifest": prediction_manifest,
        "prediction_lineage": dict(inference["lineage"]),
        "test_authority": dict(
            final_test._authority_artifact(args)
        ),
        "metric_root": metric_root,
        "feature_extractor": {
            "path": str(feature_path),
            "sha256": feature_sha,
            "bytes": feature_path.stat().st_size,
        },
        "smplx_asset": {
            "path": str(smplx_asset),
            "sha256": talkshow_metrics.SMPLX_SHA256,
            "bytes": smplx_asset.stat().st_size,
        },
        "device": args.device,
        "torch_threads": args.talkshow_torch_threads,
    }


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    authority, authority_receipt = _load_current_authority(args)
    inference, clip_ids = _validate_inference_bundle(
        args,
        authority,
    )
    evaluator, paspa = _load_pinned_paspa(args.paspa_root)
    assets = _validate_assets(
        evaluator,
        args.diffsheg_root,
        args.talkshow_root,
        args.smplx_path,
    )
    validation, audio = _validate_protocol_inputs(
        evaluator,
        inference,
        clip_ids,
        assets,
        args.source_audio_root,
        args.expected_audio_set_sha256,
    )
    talkshow_suite = _talkshow_metric_preflight(
        args,
        authority,
        inference,
        assets,
    )
    preflight: dict[str, Any] = {
        "format": PREFLIGHT_FORMAT,
        "status": "formal_ready",
        "authority": authority_receipt,
        "inference": {
            **inference,
            "frame_count": int(validation.total_frames),
            "window_count": int(validation.total_windows),
            "uncovered_tail_frames": int(
                validation.uncovered_tail_frames
            ),
            "paspa_clip_manifest_sha256": (
                validation.clip_manifest_sha256
            ),
        },
        "protocol": {
            "name": "diffsheg_show_reconstructed",
            "version": 1,
            "split": "test",
            "primary_metric_family": "DiffSHEG SHOW seven-metric protocol",
            "final_metric_event": (
                final_test.final_authority.FINAL_METRIC_EVENT
            ),
            "test_evaluations": 1,
            "test_feedback_into_selection": False,
            "validation_selection": authority_receipt["selection"],
            "clip_count": EXPECTED_TEST_CLIPS,
            "window_length": WINDOW_LENGTH,
            "window_stride": WINDOW_STRIDE,
            "tail_policy": "drop_incomplete_tail",
            "metrics": list(EXPECTED_METRICS),
            "metric_definitions": METRIC_DEFINITIONS,
            "metric_count": len(EXPECTED_METRICS),
            "padding": "forbidden",
            "truncation": "forbidden",
            "fabrication": "forbidden",
            "input": "full canonical SHOW NPZ",
            "compatibility_reports": {
                "talkshow_body_face": {
                    "status": "required_same_event",
                    "primary": False,
                    "selection_feedback": False,
                    "metric_spaces_must_not_be_mixed": True,
                    "shared_prediction_bundle": True,
                    "separate_test_evaluation": False,
                }
            },
        },
        "assets": {
            "paspa": paspa,
            **assets,
            "audio": audio,
            "talkshow_metrics": talkshow_suite,
        },
        "runtime": {
            "device": args.device,
            "paspa_batch_size": args.batch_size,
            "talkshow_torch_threads": args.talkshow_torch_threads,
        },
    }
    preflight["receipt_payload_sha256"] = canonical_json_sha256(preflight)
    return preflight


def _atomic_new(path_value: Any, payload: Mapping[str, Any], label: str) -> Path:
    path = _absolute(path_value, label)
    parent = _directory(path.parent, f"{label} parent")
    if path.parent.resolve() != parent:
        raise FinalDiffSHEGError(f"{label} parent contains a symlink")
    encoded = canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FinalDiffSHEGError(
                f"refusing to overwrite {label}: {path}"
            ) from None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve(strict=True)


def _claim_path(final_root: Path) -> Path:
    # Keep the original DiffSHEG claim leaf as the canonical namespace lock.
    # Old and combined producers must race on one O_EXCL target; using a new
    # filename here would let one producer of each generation consume the same
    # finalized prediction bundle concurrently.
    return final_root / CLAIM_NAME


def _validate_preflight_file(
    value: Any,
    expected_sha256: Any,
    current: Mapping[str, Any],
) -> tuple[Path, str]:
    path, payload, observed = _verified_file(
        value,
        expected_sha256,
        "formal DiffSHEG preflight receipt",
    )
    stored = _strict_json_bytes(payload, "formal DiffSHEG preflight receipt")
    if stored != current:
        raise FinalDiffSHEGError(
            "formal inputs changed since the frozen preflight receipt"
        )
    if (
        stored.get("format") != PREFLIGHT_FORMAT
        or stored.get("status") != "formal_ready"
        or canonical_json_sha256(
            {
                key: value
                for key, value in stored.items()
                if key != "receipt_payload_sha256"
            }
        )
        != stored.get("receipt_payload_sha256")
    ):
        raise FinalDiffSHEGError("formal preflight payload is invalid")
    return path, observed


def _paspa_command(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    output: Path,
) -> list[str]:
    evaluator = preflight["assets"]["paspa"]["evaluator"]["path"]
    return [
        sys.executable,
        evaluator,
        "--pred-dir",
        preflight["inference"]["npz_root"],
        "--gt-dir",
        preflight["inference"]["npz_root"],
        "--clip-manifest",
        preflight["inference"]["clip_manifest"]["path"],
        "--diffsheg-root",
        preflight["assets"]["diffsheg"]["root"],
        "--talkshow-root",
        preflight["assets"]["talkshow"]["root"],
        "--source-audio-root",
        preflight["assets"]["audio"]["root"],
        "--smplx-path",
        preflight["assets"]["smplx"]["model_path"],
        "--window-stride",
        str(WINDOW_STRIDE),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--output",
        str(output),
    ]


def _validate_paspa_report(
    report: Any,
    preflight: Mapping[str, Any],
) -> dict[str, float]:
    if not isinstance(report, dict) or report.get("status") != "ok":
        raise FinalDiffSHEGError("PASPA full-test report is not complete")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(EXPECTED_METRICS):
        raise FinalDiffSHEGError(
            "PASPA report must expose exactly the seven DiffSHEG metrics"
        )
    result: dict[str, float] = {}
    for key in EXPECTED_METRICS:
        value = metrics[key]
        if isinstance(value, bool) or type(value) not in {int, float}:
            raise FinalDiffSHEGError(f"metric {key} is not a JSON number")
        converted = float(value)
        if not math.isfinite(converted):
            raise FinalDiffSHEGError(f"metric {key} is not finite")
        result[key] = converted
    protocol = report.get("protocol")
    inputs = report.get("inputs")
    provenance = report.get("provenance")
    if (
        not isinstance(protocol, dict)
        or protocol.get("name") != "diffsheg_show_reconstructed"
        or protocol.get("version") != 1
        or protocol.get("status")
        != "reconstructed_from_public_components"
        or protocol.get("diffsheg_reference_commit") != DIFFSHEG_COMMIT
        or protocol.get("window_length") != WINDOW_LENGTH
        or protocol.get("window_stride") != WINDOW_STRIDE
        or protocol.get("overlap_length") != 0
        or protocol.get("precision") != "float32 AE inference; no autocast"
        or not isinstance(inputs, dict)
        or inputs.get("clip_count") != EXPECTED_TEST_CLIPS
        or inputs.get("window_count")
        != preflight["inference"]["window_count"]
        or inputs.get("frame_count")
        != preflight["inference"]["frame_count"]
        or inputs.get("uncovered_tail_frames")
        != preflight["inference"]["uncovered_tail_frames"]
        or inputs.get("clip_manifest_sha256")
        != preflight["inference"]["paspa_clip_manifest_sha256"]
        or inputs.get("audio_protocol") != "talkshow_original_source"
        or not isinstance(inputs.get("stats"), dict)
        or inputs["stats"].get("sha256") != DIFFSHEG_STATS_SHA256
        or {
            metric: Path(path).resolve()
            for metric, path in inputs.get("checkpoint_paths", {}).items()
        }
        != {
            metric: Path(specification["path"]).resolve()
            for metric, specification in preflight["assets"]["diffsheg"][
                "autoencoders"
            ].items()
        }
        or not isinstance(provenance, dict)
    ):
        raise FinalDiffSHEGError("PASPA protocol/input receipt mismatch")
    evaluator = provenance.get("evaluator")
    if (
        not isinstance(evaluator, dict)
        or evaluator.get("sha256") != PASPA_EVALUATOR_SHA256
        or evaluator.get("repository_git_head") != PASPA_COMMIT
    ):
        raise FinalDiffSHEGError("PASPA evaluator provenance mismatch")
    observed_aes = provenance.get("autoencoders")
    if not isinstance(observed_aes, dict) or set(observed_aes) != set(
        DIFFSHEG_AE_PINS
    ):
        raise FinalDiffSHEGError("PASPA autoencoder coverage mismatch")
    for metric, pin in DIFFSHEG_AE_PINS.items():
        observed = observed_aes[metric]
        expected = preflight["assets"]["diffsheg"]["autoencoders"][metric]
        if (
            not isinstance(observed, dict)
            or observed.get("sha256") != pin["sha256"]
            or observed.get("input_dim") != pin["input_dim"]
            or Path(str(observed.get("path", ""))).resolve()
            != Path(expected["path"])
            or observed.get("feature_count")
            != preflight["inference"]["window_count"]
        ):
            raise FinalDiffSHEGError(
                f"PASPA {metric} autoencoder provenance mismatch"
            )
    ba = provenance.get("ba")
    if (
        not isinstance(ba, dict)
        or ba.get("talkshow_git_head")
        != preflight["assets"]["talkshow"]["commit"]
        or ba.get("smplx_neutral_asset_sha256")
        != preflight["assets"]["smplx"]["sha256"]
    ):
        raise FinalDiffSHEGError("PASPA BA provenance mismatch")
    return result


def _run_talkshow_suite(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    output_root: Path,
    combined_claim: Mapping[str, Any],
    paspa_report: Mapping[str, Any],
) -> tuple[Path, bytes, dict[str, Any], str, dict[str, Any]]:
    suite = preflight["assets"]["talkshow_metrics"]
    manifest = {
        key: suite["prediction_manifest"][key]
        for key in ("path", "sha256", "bytes")
    }
    lineage = suite["prediction_lineage"]
    selection_protocol = {
        "primary_metric": talkshow_metrics.PRIMARY_METRIC_PATH,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 1,
    }
    try:
        backend = talkshow_metrics.TalkShowCudaMetricBackend(
            talkshow_root=suite["metric_root"]["path"],
            feature_extractor=suite["feature_extractor"]["path"],
            smplx_asset=suite["smplx_asset"]["path"],
            device=args.device,
            torch_threads=suite["torch_threads"],
        )
        report = talkshow_metrics.evaluate_canonical_bundle(
            canonical_manifest=suite["canonical_manifest"]["path"],
            expected_canonical_manifest_sha256=(
                suite["canonical_manifest"]["sha256"]
            ),
            prediction_manifest=manifest["path"],
            expected_prediction_manifest_sha256=manifest["sha256"],
            prediction_lineage=lineage["path"],
            expected_prediction_lineage_sha256=lineage["sha256"],
            validation_gate=suite["validation_gate"],
            distribution_declaration=suite["distribution_receipt"],
            backend=backend,
            split="test",
            expected_clip_count=EXPECTED_TEST_CLIPS,
            test_authority=suite["test_authority"],
            combined_claim=combined_claim,
            paspa_report=paspa_report,
        )
        combined_event = report.get("inputs", {}).get("combined_event")
        talkshow_metrics.validate_report(
            report,
            expected_split="test",
            expected_clip_count=EXPECTED_TEST_CLIPS,
            expected_prediction_manifest=manifest,
            expected_distribution_receipt=suite[
                "distribution_receipt"
            ],
            expected_selection_protocol=selection_protocol,
            expected_test_authority=suite["test_authority"],
            expected_combined_event=combined_event,
        )
    except talkshow_metrics.MetricAdapterContractError as error:
        raise FinalDiffSHEGError(
            f"TalkSHOW body/face suite failed after claim consumption: {error}"
        ) from error
    report_path = _atomic_new(
        output_root / "talkshow_show_body_face_metrics.json",
        report,
        "TalkSHOW body/face report",
    )
    payload = report_path.read_bytes()
    call_receipt = _talkshow_call_receipt(
        preflight,
        args.device,
        combined_event,
    )
    return (
        report_path,
        payload,
        report,
        canonical_json_sha256(call_receipt),
        combined_event,
    )


def _talkshow_call_receipt(
    preflight: Mapping[str, Any],
    device: str,
    combined_event: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize every input to the in-process TalkSHOW suite call."""

    suite = preflight["assets"]["talkshow_metrics"]
    manifest = {
        key: suite["prediction_manifest"][key]
        for key in ("path", "sha256", "bytes")
    }
    lineage = suite["prediction_lineage"]
    return {
        "callable": (
            "scripts.show_base.evaluate_talkshow_show_metrics."
            "evaluate_canonical_bundle"
        ),
        "canonical_manifest": suite["canonical_manifest"],
        "prediction_manifest": manifest,
        "prediction_lineage": {
            key: lineage[key] for key in ("path", "sha256", "bytes")
        },
        "validation_gate": suite["validation_gate"],
        "distribution_receipt_payload_sha256": suite[
            "distribution_receipt"
        ]["receipt_payload_sha256"],
        "test_authority": suite["test_authority"],
        "combined_event": dict(combined_event),
        "metric_root": suite["metric_root"],
        "feature_extractor": suite["feature_extractor"],
        "smplx_asset": suite["smplx_asset"],
        "device": device,
        "torch_threads": suite["torch_threads"],
        "split": "test",
        "expected_clip_count": EXPECTED_TEST_CLIPS,
    }


def run_formal(args: argparse.Namespace, preflight: Mapping[str, Any]) -> dict[str, Any]:
    preflight_path, preflight_file_sha = _validate_preflight_file(
        args.preflight_json,
        args.expected_preflight_sha256,
        preflight,
    )
    output_root = _absolute(args.output_root, "formal evaluation output root")
    parent = _directory(output_root.parent, "formal evaluation parent")
    if output_root.parent.resolve() != parent:
        raise FinalDiffSHEGError("formal output parent contains a symlink")
    expected_runtime = {
        "device": args.device,
        "paspa_batch_size": args.batch_size,
        "talkshow_torch_threads": args.talkshow_torch_threads,
    }
    if preflight.get("runtime") != expected_runtime:
        raise FinalDiffSHEGError(
            "combined metric runtime differs from frozen preflight"
        )
    expected_device = expected_runtime["device"]
    if expected_device != "cuda:0":
        raise FinalDiffSHEGError(
            "combined suites must share the frozen cuda:0 device"
        )
    claim_path = _claim_path(Path(preflight["inference"]["root"]))
    if os.path.lexists(claim_path):
        raise FinalDiffSHEGError(
            f"combined one-shot test claim already exists: {claim_path}"
        )
    if os.path.lexists(output_root):
        raise FinalDiffSHEGError(
            f"refusing to reuse formal output root: {output_root}"
        )
    claim = {
        "format": CLAIM_FORMAT,
        "status": "claimed",
        "authority": preflight["authority"],
        "preflight": {
            "path": str(preflight_path),
            "sha256": preflight_file_sha,
            "receipt_payload_sha256": preflight[
                "receipt_payload_sha256"
            ],
        },
        "output_root": str(output_root.resolve()),
        "test_evaluations": 1,
        "test_feedback_into_selection": False,
        "final_metric_event": preflight["protocol"]["final_metric_event"],
        "shared_predictions": {
            "manifest": preflight["assets"]["talkshow_metrics"][
                "prediction_manifest"
            ],
            "lineage": preflight["assets"]["talkshow_metrics"][
                "prediction_lineage"
            ],
        },
        "claim_consumed_before_metrics": True,
        "failure_consumes_claim": True,
        "retry_allowed": False,
        "input_set_sha256": preflight["inference"]["input_set_sha256"],
    }
    # The non-retryable claim is the first formal mutation.  Even failure to
    # create the output directory after this point consumes the authority;
    # there is deliberately no cleanup path for the claim.
    claim_resolved = _atomic_new(
        claim_path,
        claim,
        "combined one-shot test claim",
    )
    combined_claim_artifact = {
        "path": str(claim_resolved),
        "sha256": sha256_file(claim_resolved),
        "bytes": claim_resolved.stat().st_size,
    }
    output_root.mkdir(mode=0o700)
    evaluator_output = output_root / "paspa_diffsheg_show_metrics.json"
    command = _paspa_command(args, preflight, evaluator_output)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise FinalDiffSHEGError(
            f"PASPA DiffSHEG evaluator failed with rc={result.returncode}; "
            f"one-shot claim remains at {claim_resolved}"
        )
    evaluator_path = _regular_file(
        evaluator_output,
        "PASPA DiffSHEG full-test report",
    )
    evaluator_payload = evaluator_path.read_bytes()
    evaluator_report = _strict_json_bytes(
        evaluator_payload,
        "PASPA DiffSHEG full-test report",
    )
    metrics = _validate_paspa_report(evaluator_report, preflight)
    paspa_report_artifact = {
        "path": str(evaluator_path),
        "sha256": hashlib.sha256(evaluator_payload).hexdigest(),
        "bytes": len(evaluator_payload),
    }
    (
        talkshow_path,
        talkshow_payload,
        talkshow_report,
        talkshow_call_sha256,
        combined_event,
    ) = _run_talkshow_suite(
        args,
        preflight,
        output_root,
        combined_claim_artifact,
        paspa_report_artifact,
    )
    talkshow_metrics_snapshot = {
        "body": talkshow_report["body"],
        "face": talkshow_report["face"],
        "rs": talkshow_report["rs"],
    }
    completion: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "status": "complete",
        "authority": preflight["authority"],
        "one_shot_claim": {
            "path": str(claim_resolved),
            "sha256": sha256_file(claim_resolved),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": preflight_file_sha,
            "receipt_payload_sha256": preflight[
                "receipt_payload_sha256"
            ],
        },
        "paspa_report": paspa_report_artifact,
        "talkshow_suite_start": combined_event["suite_start"],
        "talkshow_report": {
            "path": str(talkshow_path),
            "sha256": hashlib.sha256(talkshow_payload).hexdigest(),
            "bytes": len(talkshow_payload),
            "report_payload_sha256": talkshow_report[
                "report_payload_sha256"
            ],
        },
        "evaluator_command_sha256": hashlib.sha256(
            b"\0".join(argument.encode("utf-8") for argument in command)
        ).hexdigest(),
        "talkshow_evaluator_call_sha256": talkshow_call_sha256,
        "protocol": preflight["protocol"],
        "assets": preflight["assets"],
        "runtime": preflight["runtime"],
        "coverage": {
            "clip_count": EXPECTED_TEST_CLIPS,
            "exact_once": True,
            "num_generation_shards": EXPECTED_NUM_SHARDS,
            "window_length": WINDOW_LENGTH,
            "window_stride": WINDOW_STRIDE,
            "window_count": preflight["inference"]["window_count"],
            "generation_passes": 1,
            "shared_prediction_bundle": True,
        },
        "metrics": {
            "diffsheg": metrics,
            "talkshow": talkshow_metrics_snapshot,
        },
        "all_metrics_finite": True,
        "test_evaluations": 1,
        "validation_only_for_selection": True,
        "test_feedback_into_selection": False,
        "final_metric_event": preflight["protocol"]["final_metric_event"],
    }
    completion["receipt_payload_sha256"] = canonical_json_sha256(completion)
    completion_path = _atomic_new(
        output_root / "final_metrics.json",
        completion,
        "formal DiffSHEG completion marker",
    )
    completion_payload = completion_path.read_bytes()
    completion_file_sha256 = hashlib.sha256(completion_payload).hexdigest()
    completion_bytes = len(completion_payload)
    completion_pin = {
        "path": str(completion_path),
        "sha256": completion_file_sha256,
        "bytes": completion_bytes,
        "canonical_payload_sha256": completion[
            "receipt_payload_sha256"
        ],
    }
    return {
        "status": "complete",
        "output": str(completion_path),
        "sha256": completion_file_sha256,
        "bytes": completion_bytes,
        "metrics": completion["metrics"],
        "receipt_payload_sha256": completion[
            "receipt_payload_sha256"
        ],
        "canonical_payload_sha256": completion[
            "receipt_payload_sha256"
        ],
        "final_metrics": completion_pin,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    final_test._authority_options(parser)
    parser.add_argument("--inference-final-root", type=Path, required=True)
    parser.add_argument("--paspa-root", type=Path, required=True)
    parser.add_argument("--diffsheg-root", type=Path, required=True)
    parser.add_argument("--talkshow-root", type=Path, required=True)
    parser.add_argument(
        "--source-audio-root",
        type=Path,
        required=True,
        help=(
            "sealed test-layout view produced by "
            "prepare_diffsheg_audio_view.py (never the raw TalkSHOW root)"
        ),
    )
    parser.add_argument("--smplx-path", type=Path, required=True)
    parser.add_argument(
        "--talkshow-metric-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--talkshow-feature-extractor",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--talkshow-validation-gate-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-talkshow-validation-gate-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-talkshow-validation-gate-receipt-payload-sha256",
        required=True,
    )
    parser.add_argument(
        "--talkshow-torch-threads",
        type=int,
        default=1,
    )
    parser.add_argument("--expected-audio-set-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-json", type=Path)
    parser.add_argument("--expected-preflight-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.talkshow_torch_threads < 1:
        parser.error("--batch-size/--talkshow-torch-threads must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    prepared_values = (
        args.prepared_authority,
        args.expected_prepared_authority_sha256,
        args.expected_prepared_authority_bytes,
        args.expected_prepared_authority_receipt_payload_sha256,
    )
    if any(value is not None for value in prepared_values) and any(
        value is None for value in prepared_values
    ):
        parser.error("prepared authority requires path/SHA/bytes/payload SHA")
    if args.expected_audio_set_sha256 is not None:
        args.expected_audio_set_sha256 = require_sha256(
            args.expected_audio_set_sha256,
            "--expected-audio-set-sha256",
        )
    args.expected_talkshow_validation_gate_sha256 = require_sha256(
        args.expected_talkshow_validation_gate_sha256,
        "--expected-talkshow-validation-gate-sha256",
    )
    args.expected_talkshow_validation_gate_receipt_payload_sha256 = (
        require_sha256(
            args.expected_talkshow_validation_gate_receipt_payload_sha256,
            "--expected-talkshow-validation-gate-receipt-payload-sha256",
        )
    )
    try:
        preflight = build_preflight(args)
        if args.preflight_only:
            if any(
                value is not None
                for value in (
                    args.preflight_json,
                    args.expected_preflight_sha256,
                    args.output_root,
                )
            ):
                parser.error(
                    "--preflight-only forbids formal-run output arguments"
                )
            if args.output_report is None:
                parser.error("--preflight-only requires --output-report")
            output = _atomic_new(
                args.output_report,
                preflight,
                "formal DiffSHEG preflight report",
            )
            print(
                json.dumps(
                    {
                        "status": "formal_ready",
                        "output": str(output),
                        "sha256": sha256_file(output),
                        "audio_set_sha256": preflight["assets"]["audio"][
                            "ordered_manifest_sha256"
                        ],
                        "receipt_payload_sha256": preflight[
                            "receipt_payload_sha256"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.output_report is not None:
            parser.error("formal mode forbids --output-report")
        if (
            args.preflight_json is None
            or args.expected_preflight_sha256 is None
            or args.output_root is None
            or args.expected_audio_set_sha256 is None
        ):
            parser.error(
                "formal mode requires --preflight-json, "
                "--expected-preflight-sha256, --expected-audio-set-sha256, "
                "and --output-root"
            )
        args.expected_preflight_sha256 = require_sha256(
            args.expected_preflight_sha256,
            "--expected-preflight-sha256",
        )
        result = run_formal(args, preflight)
    except FinalDiffSHEGError as error:
        parser.exit(2, f"[abort] {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
