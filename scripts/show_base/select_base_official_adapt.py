#!/usr/bin/env python3
"""Select one official-adapt Base checkpoint using validation FGD only.

This entry point is deliberately decision-only and CPU-only.  It consumes
seven immutable measurement receipts produced by the exact final inference
pipeline on a frozen, validation-only SHOW snapshot.  There is no split
argument and no test input.  The selected checkpoint is the minimum
``(FGD, epoch)`` among epochs 1/2/4/8/16/32/40.

The measurement receipts bind:

* the complete official-adapt candidate manifest, status, and frozen inputs;
* an exact 1,715-clip canonical validation manifest and eight audio shards;
* the fixed Face/Hands/Upper/Lower/Global downstream pipeline; and
* the hash-pinned reconstructed DiffSHEG SHOW evaluator and assets.

The selector never runs inference or an evaluator.  It validates their frozen
receipts, recomputes the winner, and atomically creates one selection receipt.
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
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import gate_task_space_on_show_v2 as _receipt


EXPECTED_CANDIDATE_EPOCHS = (1, 2, 4, 8, 16, 32, 40)
EXPECTED_VAL_CLIPS = 1_715
EXPECTED_AUDIO_SHARDS = 8
EXPECTED_UPDATES_PER_EPOCH = 248
DIFFSHEG_WINDOW = 88
DIFFSHEG_STRIDE = 88
SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}

CANDIDATE_MANIFEST_FORMAT = (
    "semtalk_show_base_official_adapt_manifest_v1"
)
CANDIDATE_STATUS_FORMAT = "semtalk_show_base_official_adapt_status_v1"
THROUGHPUT_GATE_FORMAT = (
    "semtalk_show_base_official_adapt_throughput_gate_v1"
)
FROZEN_INPUTS_FORMAT = (
    "semtalk_show_base_official_adapt_frozen_inputs_v1"
)
VAL_INPUTS_FORMAT = "semtalk_show_base_official_adapt_val_inputs_v1"
VAL_CANONICAL_SUMMARY_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_summary_v1"
)
VAL_CANONICAL_LINEAGE_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_lineage_v1"
)
PIPELINE_FORMAT = "semtalk_show_base_official_adapt_val_pipeline_v1"
VAL_INFERENCE_LINEAGE_FORMAT = (
    "semtalk_show_base_official_adapt_val_inference_lineage_v1"
)
MEASUREMENT_FORMAT = (
    "semtalk_show_base_official_adapt_val_measurement_v1"
)
SELECTION_FORMAT = "semtalk_show_base_official_adapt_selection_v1"

VAL_METRIC_KEYS = ("fgd",)
INFERENCE_HELPERS = (
    "_load_canonical_clip",
    "_load_audio_features",
    "_infer_clip",
    "_inference_only_auxiliary_loss_bypass",
    "_inference_auxiliary_loss_bypass_receipt",
    "_output_arrays",
)
# Match reserved labels, not incidental substrings such as Latest or contest.
_TEST_PATH_LABEL_TOKENS = frozenset(
    {"test", "tests", "testset", "testsets"}
)

DIFFSHEG_PINNED_RECEIPT: dict[str, Any] = {
    "protocol": "diffsheg_show_reconstructed",
    "protocol_version": 1,
    "paspa": {
        "origin": "git@github.com:Ly403/PASPA.git",
        "commit": "0df27e6cab4b5ced19cc923afe352f77d547924b",
        "tree": "574662eaf3122beb5631c02c847456e752d77f0b",
        "evaluator_sha256": (
            "21fa84fdb9c3f64eb2920714e1d27a685210a225bcffa78503452c4018a53f8c"
        ),
    },
    "diffsheg_reference_commit": (
        "3ebf3058f48cba3da9146afb7623e9ec1ab9e9a5"
    ),
    "stats_sha256": (
        "b90320eba94d0777e7160fd31d0fe6f04a7c86822ac875fb7db5cf58d298cef0"
    ),
    "autoencoders": {
        "fgd": {
            "filename": "gesture.pth.tar",
            "sha256": (
                "5eaf9b882a5ccd5f6eb4385aaadf3d28f3ee4382360ecb13c12f4904b3c3216e"
            ),
            "input_dim": 129,
        },
    },
    "window_length": DIFFSHEG_WINDOW,
    "window_stride": DIFFSHEG_STRIDE,
    "precision": "float32 AE inference; no autocast",
    "selection_metric": "fgd",
    "ba_during_selection": False,
}

FIXED_OFFICIAL_CHECKPOINTS = {
    "hands": {
        "filename": "rvq_hands_500.bin",
        "sha256": (
            "08f887aac60d5a2102dce7c57559a6b3d9b7f56e3d4a38055ca47a539b03e436"
        ),
    },
    "upper": {
        "filename": "rvq_upper_500.bin",
        "sha256": (
            "05101461e75b4e9b687ef30437585d56969c6a13d0047b91000b31d88d08ac17"
        ),
    },
    "lower": {
        "filename": "rvq_lower_600.bin",
        "sha256": (
            "2bb43d10e5f32d13d21e6b85580a1b70d36e407c8552a7e62f99c171ae4efce8"
        ),
    },
}
OFFICIAL_BASE_CHECKPOINT = {
    "filename": "best_semtalk_base.bin",
    "sha256": (
        "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
    ),
}
BASE_PRODUCER_SOURCE = {
    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
    "commit": "5b5c8dc72f0e171c52cb4729314ed977c02c1376",
    "tree": "a2f738b9b33f1b70444e4c3876701876da9919ea",
    "entrypoint": "train_base_official_adapt.py",
    "entrypoint_sha256": (
        "6e2eef748f8b1fee3e5c5ddf54dd71fb3b3f3a575975c0cea24cf11973031534"
    ),
}
VAL_INFERENCE_SOURCE = {
    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
    "commit": "8da5c0dc4e9694319d742609ae145476c5408399",
    "tree": "3e3a451eda15f28b3309a66faba6d55ccdb58e6f",
    "entrypoint": "run_base_inference.py",
    "entrypoint_sha256": (
        "d1f5da319d76c0b0190db479327c8c31774efe3f5e1222ad0f35e556d78d59ab"
    ),
}


class SelectionContractError(RuntimeError):
    """Raised when validation-only selection cannot be proven."""


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionContractError(
            f"{label} must be a canonical lowercase SHA-256"
        )
    return value


def require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionContractError(
            f"{label} must be a canonical lowercase Git object ID"
        )
    return value


def require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionContractError(f"{label} must be an exact integer")
    return value


def require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise SelectionContractError(f"{label} must be a JSON number")
    converted = float(value)
    if not math.isfinite(converted):
        raise SelectionContractError(f"{label} must be finite")
    return converted


def require_exact_keys(
    value: Any,
    keys: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SelectionContractError(f"{label} schema mismatch")
    return value


def reject_forbidden_source_labels(*values: object) -> None:
    """Hard reject withdrawn e30 and forbidden Speaker2 labels/paths."""

    for value in values:
        for component in str(value).replace("\\", "/").split("/"):
            stem = component.rsplit(".", 1)[0]
            if len(stem) in {40, 64} and all(
                character in "0123456789abcdefABCDEF"
                for character in stem
            ):
                continue
            normalized = component.casefold()
            if (
                re.search(r"(^|[^a-z0-9])e[-_]?30([^a-z0-9]|$)", normalized)
                or re.search(
                    r"(^|[^a-z0-9])speaker[-_]?2([^a-z0-9]|$)",
                    normalized,
                )
            ):
                raise SelectionContractError(
                    "withdrawn e30 or forbidden Speaker2 input: "
                    f"{value}"
                )


def require_absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SelectionContractError(f"{label} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        raise SelectionContractError(f"{label} must be absolute")
    reject_forbidden_source_labels(path)
    return path


def require_val_only_path(value: Any, label: str) -> Path:
    path = require_absolute_path(value, label)
    reject_test_path(path, label)
    return path


def reject_test_path(path: Path, label: str) -> None:
    for piece in path.parts:
        tokens = set(re.findall(r"[a-z0-9]+", piece.casefold()))
        if tokens & _TEST_PATH_LABEL_TOKENS:
            raise SelectionContractError(
                f"{label} must not expose a test-labeled path: {path}"
            )


def require_directory(value: Any, label: str) -> Path:
    path = require_val_only_path(value, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SelectionContractError(
            f"{label} must be a non-symlink directory: {path}"
        )
    resolved = path.resolve(strict=True)
    reject_forbidden_source_labels(resolved)
    reject_test_path(resolved, label)
    return resolved


def canonical_clip_id(source_clip_id: str) -> str:
    """Mirror the final inference output-ID mapping without importing numpy."""

    pieces = source_clip_id.split("/")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        raise SelectionContractError(
            f"invalid canonical SHOW source clip ID: {source_clip_id!r}"
        )
    speaker, _video, sequence = pieces
    if speaker not in SHOW_SPEAKER_IDS:
        raise SelectionContractError(f"unknown SHOW speaker {speaker!r}")
    if (
        "__" in speaker
        or "/" in sequence
        or sequence in {"", ".", ".."}
        or "\x00" in sequence
    ):
        raise SelectionContractError(
            f"unsafe canonical SHOW source clip ID: {source_clip_id!r}"
        )
    return f"{speaker}__{sequence}"


def _regular_file(path: Path, label: str) -> Path:
    reject_forbidden_source_labels(path)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SelectionContractError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    reject_forbidden_source_labels(resolved)
    return resolved


def _verified_bytes(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes, str]:
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    resolved = _regular_file(path, label)
    payload = resolved.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise SelectionContractError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return resolved, payload, observed


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return _receipt.strict_json_loads(payload, label)
    except (TypeError, ValueError, RuntimeError) as error:
        raise SelectionContractError(str(error)) from error


def _verified_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    resolved, payload, observed = _verified_bytes(
        path,
        expected_sha256,
        label,
    )
    value = _strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise SelectionContractError(f"{label} must be a JSON object")
    return resolved, value, observed


def _strict_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SelectionContractError(
            f"{label} is not UTF-8: {error}"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_json_bytes(
            line.encode("utf-8"),
            f"{label}:{line_number}",
        )
        if not isinstance(value, dict):
            raise SelectionContractError(
                f"{label}:{line_number} must be a JSON object"
            )
        rows.append(value)
    return rows


def _payload_hash_without(
    payload: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    claimed = require_sha256(payload.get(field), f"{label}.{field}")
    body = dict(payload)
    body.pop(field, None)
    observed = canonical_json_sha256(body)
    if observed != claimed:
        raise SelectionContractError(
            f"{label} payload hash mismatch: {observed} != {claimed}"
        )
    return claimed


def _artifact_fields(
    value: Any,
    label: str,
    *,
    payload_hash: bool = False,
) -> tuple[Path, str, str | None]:
    expected = {"path", "sha256"}
    if payload_hash:
        expected.add("receipt_payload_sha256")
    value = require_exact_keys(value, expected, label)
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise SelectionContractError(f"{label}.path must be nonempty")
    path = Path(raw_path)
    if not path.is_absolute():
        raise SelectionContractError(f"{label}.path must be absolute")
    reject_forbidden_source_labels(path)
    digest = require_sha256(value["sha256"], f"{label}.sha256")
    receipt_hash = (
        require_sha256(
            value["receipt_payload_sha256"],
            f"{label}.receipt_payload_sha256",
        )
        if payload_hash
        else None
    )
    return path, digest, receipt_hash


def _verify_artifact(
    value: Any,
    label: str,
    *,
    payload_hash: bool = False,
) -> tuple[dict[str, Any], Path, bytes]:
    path, digest, receipt_hash = _artifact_fields(
        value,
        label,
        payload_hash=payload_hash,
    )
    resolved, payload, _ = _verified_bytes(path, digest, label)
    receipt = {"path": str(resolved), "sha256": digest}
    if receipt_hash is not None:
        receipt["receipt_payload_sha256"] = receipt_hash
    return receipt, resolved, payload


def _validate_frozen_inputs(payload: Any) -> str:
    payload = require_exact_keys(
        payload,
        {
            "format",
            "source",
            "official_base",
            "speaker_initialization",
            "dataset",
            "protocol",
            "receipt_sha256",
        },
        "Base frozen inputs",
    )
    receipt_sha = _payload_hash_without(
        payload,
        "receipt_sha256",
        "Base frozen inputs",
    )
    source = require_exact_keys(
        payload["source"],
        {
            "origin",
            "commit",
            "tree",
            "branch",
            "clean",
            "entrypoint",
            "entrypoint_sha256",
        },
        "Base producer source",
    )
    source_entrypoint = require_absolute_path(
        source["entrypoint"],
        "Base producer source entrypoint",
    )
    if (
        source["origin"] != BASE_PRODUCER_SOURCE["origin"]
        or require_git_oid(
            source["commit"],
            "Base producer source commit",
        )
        != BASE_PRODUCER_SOURCE["commit"]
        or require_git_oid(
            source["tree"],
            "Base producer source tree",
        )
        != BASE_PRODUCER_SOURCE["tree"]
        or source["branch"] is not None
        or source["clean"] is not True
        or source_entrypoint.name != BASE_PRODUCER_SOURCE["entrypoint"]
        or require_sha256(
            source["entrypoint_sha256"],
            "Base producer entrypoint SHA",
        )
        != BASE_PRODUCER_SOURCE["entrypoint_sha256"]
    ):
        raise SelectionContractError(
            "Base frozen inputs do not bind the pinned detached producer"
        )
    protocol = payload["protocol"]
    candidate_epochs = (
        protocol.get("candidate_epochs")
        if isinstance(protocol, dict)
        else None
    )
    optimizer = protocol.get("optimizer") if isinstance(protocol, dict) else None
    protocol_learning_rate = (
        require_finite_number(
            optimizer.get("learning_rate"),
            "Base protocol learning_rate",
        )
        if isinstance(optimizer, dict)
        else math.nan
    )
    if (
        payload["format"] != FROZEN_INPUTS_FORMAT
        or not isinstance(protocol, dict)
        or protocol.get("format")
        != "semtalk_show_base_official_adapt_protocol_v1"
        or protocol.get("target_dataset") != "SHOW"
        or protocol.get("target_speaker_scope") != "All"
        or not isinstance(candidate_epochs, list)
        or any(type(epoch) is not int for epoch in candidate_epochs)
        or candidate_epochs != list(EXPECTED_CANDIDATE_EPOCHS)
        or require_exact_int(
            protocol.get("epochs"),
            "Base protocol epochs",
        )
        != 40
        or require_exact_int(
            protocol.get("expected_updates_per_epoch"),
            "Base protocol updates per epoch",
        )
        != EXPECTED_UPDATES_PER_EPOCH
        or protocol.get("vq_models_in_training_graph") is not False
        or protocol.get("precision") not in {"bf16", "fp32"}
        or not isinstance(optimizer, dict)
        or optimizer.get("name") != "Adam"
        or not 0.0 < protocol_learning_rate <= 3e-4
        or protocol.get("throughput_gate")
        != {"warmup_updates": 20, "timed_updates": 50}
        or protocol.get("initialization", {}).get("forbidden_sources")
        != ["e30", "Speaker2"]
    ):
        raise SelectionContractError(
            "Base frozen inputs do not preserve the official adaptation "
            "training contract"
        )
    dataset = payload["dataset"]
    if (
        not isinstance(dataset, dict)
        or require_exact_int(
            dataset.get("entries"),
            "Base training dataset entries",
        )
        != 127_286
        or require_exact_int(
            dataset.get("train_clips"),
            "Base training dataset clips",
        )
        != 13_687
        or dataset.get("vq_models_in_training_graph") is not False
    ):
        raise SelectionContractError("Base frozen training dataset mismatch")
    official_base = payload["official_base"]
    if (
        not isinstance(official_base, dict)
        or official_base.get("source") != "released_all_speakers_v1"
        or official_base.get("filename")
        != OFFICIAL_BASE_CHECKPOINT["filename"]
        or official_base.get("sha256")
        != OFFICIAL_BASE_CHECKPOINT["sha256"]
        or official_base.get("speaker_scope") != "All-Speakers"
        or official_base.get("all_model_state_tensors_finite") is not True
        or official_base.get("strict_state_dict_load") is not True
    ):
        raise SelectionContractError(
            "Base frozen inputs do not bind the exact official initialization"
        )
    for field in ("path", "filename"):
        if field in official_base:
            reject_forbidden_source_labels(official_base[field])
    return receipt_sha


def _validate_throughput_gate(
    value: Any,
    *,
    frozen_receipt_sha256: str,
    expected_precision: str,
    expected_learning_rate: float,
) -> dict[str, Any]:
    value = require_exact_keys(
        value,
        {"path", "sha256", "samples_per_second", "seconds_per_update"},
        "Base throughput gate receipt",
    )
    path = require_val_only_path(
        value["path"],
        "Base throughput gate path",
    )
    resolved, report, observed_sha = _verified_json(
        path,
        require_sha256(
            value["sha256"],
            "Base throughput gate artifact SHA",
        ),
        "Base throughput gate",
    )
    claimed_receipt_sha = require_sha256(
        report.get("receipt_sha256"),
        "Base throughput gate payload SHA",
    )
    report_body = dict(report)
    report_body.pop("receipt_sha256", None)
    samples_per_second = require_finite_number(
        report.get("samples_per_second"),
        "Base throughput samples_per_second",
    )
    seconds_per_update = require_finite_number(
        report.get("seconds_per_update"),
        "Base throughput seconds_per_update",
    )
    elapsed_seconds = require_finite_number(
        report.get("elapsed_seconds"),
        "Base throughput elapsed_seconds",
    )
    if (
        canonical_json_sha256(report_body) != claimed_receipt_sha
        or report.get("format") != THROUGHPUT_GATE_FORMAT
        or report.get("status") != "pass"
        or report.get("frozen_receipt_sha256") != frozen_receipt_sha256
        or require_exact_int(
            report.get("world_size"),
            "Base throughput world_size",
        )
        != 8
        or require_exact_int(
            report.get("local_batch_size"),
            "Base throughput local_batch_size",
        )
        != 64
        or require_exact_int(
            report.get("global_batch_size"),
            "Base throughput global_batch_size",
        )
        != 512
        or require_exact_int(
            report.get("warmup_updates"),
            "Base throughput warmup_updates",
        )
        != 20
        or require_exact_int(
            report.get("timed_updates"),
            "Base throughput timed_updates",
        )
        != 50
        or report.get("precision") != expected_precision
        or require_finite_number(
            report.get("learning_rate"),
            "Base throughput learning_rate",
        )
        != expected_learning_rate
        or report.get("all_losses_finite") is not True
        or require_exact_int(
            report.get("optimizer_updates"),
            "Base throughput optimizer_updates",
        )
        != 70
        or samples_per_second <= 0.0
        or seconds_per_update <= 0.0
        or elapsed_seconds <= 0.0
        or not math.isclose(
            seconds_per_update,
            elapsed_seconds / 50.0,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        or not math.isclose(
            samples_per_second,
            512.0 * 50.0 / elapsed_seconds,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        or require_finite_number(
            value["samples_per_second"],
            "Base throughput receipt samples_per_second",
        )
        != samples_per_second
        or require_finite_number(
            value["seconds_per_update"],
            "Base throughput receipt seconds_per_update",
        )
        != seconds_per_update
    ):
        raise SelectionContractError(
            "Base throughput gate does not bind the exact training protocol"
        )
    return {
        "path": str(resolved),
        "sha256": observed_sha,
        "samples_per_second": samples_per_second,
        "seconds_per_update": seconds_per_update,
    }


def validate_candidate_bundle(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    status_path: Path,
    expected_status_sha256: str,
    frozen_inputs_path: Path,
    expected_frozen_inputs_sha256: str,
) -> dict[str, Any]:
    """Verify the exact seven-candidate producer transaction without torch."""

    frozen_resolved, frozen, frozen_file_sha = _verified_json(
        frozen_inputs_path,
        expected_frozen_inputs_sha256,
        "Base frozen inputs",
    )
    frozen_receipt_sha = _validate_frozen_inputs(frozen)
    manifest_resolved, manifest, manifest_file_sha = _verified_json(
        manifest_path,
        expected_manifest_sha256,
        "Base candidate manifest",
    )
    status_resolved, status, status_file_sha = _verified_json(
        status_path,
        expected_status_sha256,
        "Base training status",
    )
    if not (
        frozen_resolved.parent
        == manifest_resolved.parent
        == status_resolved.parent
    ):
        raise SelectionContractError(
            "Base manifest, status, and frozen inputs must share one run root"
        )

    manifest = require_exact_keys(
        manifest,
        {
            "format",
            "status",
            "candidate_epochs",
            "frozen_receipt_sha256",
            "throughput_gate",
            "entries",
            "entries_sha256",
            "completed_epochs",
            "optimizer_updates",
        },
        "Base candidate manifest",
    )
    entries = manifest["entries"]
    manifest_candidate_epochs = manifest["candidate_epochs"]
    if (
        manifest["format"] != CANDIDATE_MANIFEST_FORMAT
        or manifest["status"] != "complete"
        or not isinstance(manifest_candidate_epochs, list)
        or any(type(epoch) is not int for epoch in manifest_candidate_epochs)
        or manifest_candidate_epochs != list(EXPECTED_CANDIDATE_EPOCHS)
        or manifest["frozen_receipt_sha256"] != frozen_receipt_sha
        or require_exact_int(
            manifest["completed_epochs"],
            "Base manifest completed_epochs",
        )
        != 40
        or require_exact_int(
            manifest["optimizer_updates"],
            "Base manifest optimizer_updates",
        )
        != 40 * EXPECTED_UPDATES_PER_EPOCH
        or not isinstance(entries, list)
        or len(entries) != len(EXPECTED_CANDIDATE_EPOCHS)
        or manifest["entries_sha256"] != canonical_json_sha256(entries)
    ):
        raise SelectionContractError(
            "Base candidate manifest is not the exact complete epoch set"
        )
    throughput_receipt = _validate_throughput_gate(
        manifest["throughput_gate"],
        frozen_receipt_sha256=frozen_receipt_sha,
        expected_precision=str(frozen["protocol"].get("precision")),
        expected_learning_rate=require_finite_number(
            frozen["protocol"].get("optimizer", {}).get("learning_rate"),
            "Base protocol learning_rate",
        ),
    )

    candidate_receipts: dict[int, dict[str, Any]] = {}
    expected_candidate_paths: set[Path] = set()
    entry_keys = {
        "epoch",
        "optimizer_updates",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_container_schema",
        "model_state_tensors",
        "model_state_schema_sha256",
        "all_model_state_tensors_finite",
        "frozen_receipt_sha256",
    }
    for expected_epoch, raw_entry in zip(
        EXPECTED_CANDIDATE_EPOCHS,
        entries,
    ):
        entry = require_exact_keys(
            raw_entry,
            entry_keys,
            f"Base candidate epoch {expected_epoch}",
        )
        relative = entry["checkpoint"]
        entry_epoch = require_exact_int(
            entry["epoch"],
            f"epoch {expected_epoch} manifest epoch",
        )
        entry_updates = require_exact_int(
            entry["optimizer_updates"],
            f"epoch {expected_epoch} optimizer updates",
        )
        expected_relative = (
            f"candidates/base_official_adapt_epoch_{expected_epoch:02d}.bin"
        )
        if (
            entry_epoch != expected_epoch
            or entry_updates != expected_epoch * EXPECTED_UPDATES_PER_EPOCH
            or relative != expected_relative
            or entry["checkpoint_container_schema"]
            != ["audit", "model_state"]
            or require_exact_int(
                entry["model_state_tensors"],
                f"epoch {expected_epoch} model_state_tensors",
            )
            <= 0
            or entry["all_model_state_tensors_finite"] is not True
            or entry["frozen_receipt_sha256"] != frozen_receipt_sha
        ):
            raise SelectionContractError(
                f"invalid Base candidate receipt at epoch {expected_epoch}"
            )
        require_sha256(
            entry["model_state_schema_sha256"],
            f"epoch {expected_epoch} model-state schema SHA",
        )
        checkpoint_sha = require_sha256(
            entry["checkpoint_sha256"],
            f"epoch {expected_epoch} checkpoint SHA",
        )
        reject_forbidden_source_labels(relative)
        candidate_path = manifest_resolved.parent / relative
        resolved_checkpoint = _regular_file(
            candidate_path,
            f"Base candidate epoch {expected_epoch}",
        )
        try:
            resolved_checkpoint.relative_to(manifest_resolved.parent)
        except ValueError as error:
            raise SelectionContractError(
                "Base candidate escapes its immutable run root"
            ) from error
        observed_bytes = resolved_checkpoint.stat().st_size
        if (
            require_exact_int(
                entry["checkpoint_bytes"],
                f"epoch {expected_epoch} checkpoint bytes",
            )
            != observed_bytes
            or sha256_file(resolved_checkpoint) != checkpoint_sha
        ):
            raise SelectionContractError(
                f"Base candidate bytes changed at epoch {expected_epoch}"
            )
        candidate_receipts[expected_epoch] = {
            "path": str(resolved_checkpoint),
            "sha256": checkpoint_sha,
            "bytes": observed_bytes,
        }
        expected_candidate_paths.add(resolved_checkpoint)

    candidate_directory = manifest_resolved.parent / "candidates"
    if candidate_directory.is_symlink() or not candidate_directory.is_dir():
        raise SelectionContractError(
            "Base candidate directory is missing or unsafe"
        )
    children = list(candidate_directory.iterdir())
    actual_candidate_paths = {
        child.resolve()
        for child in children
        if child.is_file() and not child.is_symlink()
    }
    if (
        len(children) != len(EXPECTED_CANDIDATE_EPOCHS)
        or actual_candidate_paths != expected_candidate_paths
    ):
        raise SelectionContractError(
            "Base candidate directory is not the exact seven-epoch cover"
        )

    status = require_exact_keys(
        status,
        {
            "format",
            "status",
            "completed_epochs",
            "optimizer_updates",
            "updates_per_epoch",
            "candidate_manifest_sha256",
            "frozen_receipt_sha256",
            "throughput_gate",
            "world_size",
            "local_batch_size",
            "global_batch_size",
            "all_training_state_finite",
            "started_unix",
            "completed_unix",
        },
        "Base training status",
    )
    completed_epochs = require_exact_int(
        status["completed_epochs"],
        "Base status completed_epochs",
    )
    optimizer_updates = require_exact_int(
        status["optimizer_updates"],
        "Base status optimizer_updates",
    )
    updates_per_epoch = require_exact_int(
        status["updates_per_epoch"],
        "Base status updates_per_epoch",
    )
    world_size = require_exact_int(
        status["world_size"],
        "Base status world_size",
    )
    local_batch_size = require_exact_int(
        status["local_batch_size"],
        "Base status local_batch_size",
    )
    global_batch_size = require_exact_int(
        status["global_batch_size"],
        "Base status global_batch_size",
    )
    if (
        status["format"] != CANDIDATE_STATUS_FORMAT
        or status["status"] != "complete"
        or completed_epochs != 40
        or optimizer_updates != 40 * EXPECTED_UPDATES_PER_EPOCH
        or updates_per_epoch != EXPECTED_UPDATES_PER_EPOCH
        or status["candidate_manifest_sha256"] != manifest_file_sha
        or status["frozen_receipt_sha256"] != frozen_receipt_sha
        or world_size != 8
        or local_batch_size != 64
        or global_batch_size != 512
        or status["all_training_state_finite"] is not True
        or status["throughput_gate"] != throughput_receipt
    ):
        raise SelectionContractError(
            "Base official adaptation did not finalize exactly"
        )
    started = require_finite_number(status["started_unix"], "started_unix")
    completed = require_finite_number(
        status["completed_unix"],
        "completed_unix",
    )
    if started < 0.0 or completed < started:
        raise SelectionContractError("invalid Base training timestamps")
    return {
        "manifest": {
            "path": str(manifest_resolved),
            "sha256": manifest_file_sha,
        },
        "status": {
            "path": str(status_resolved),
            "sha256": status_file_sha,
        },
        "frozen_inputs": {
            "path": str(frozen_resolved),
            "sha256": frozen_file_sha,
            "receipt_sha256": frozen_receipt_sha,
        },
        "candidates": candidate_receipts,
    }


def _canonical_coverage(
    payload: bytes,
    label: str,
) -> tuple[set[str], dict[str, Any]]:
    rows = _strict_jsonl(payload, label)
    if len(rows) != EXPECTED_VAL_CLIPS:
        raise SelectionContractError(
            f"canonical validation rows {len(rows)} != {EXPECTED_VAL_CLIPS}"
        )
    seen_indices: set[int] = set()
    seen_source_ids: set[str] = set()
    seen_output_ids: set[str] = set()
    ordered: list[tuple[int, str, int]] = []
    ordered_clips: list[dict[str, Any]] = []
    speakers: set[str] = set()
    for row in rows:
        if row.get("split") != "val":
            raise SelectionContractError(
                "canonical selection manifest must contain val rows only"
            )
        index = require_exact_int(
            row.get("global_index"),
            "canonical validation global_index",
        )
        frames = require_exact_int(
            row.get("frames"),
            "canonical validation frames",
        )
        clip_id = row.get("clip_id")
        if (
            index < 0
            or index in seen_indices
            or frames < DIFFSHEG_WINDOW
            or not isinstance(clip_id, str)
            or not clip_id
            or clip_id in seen_source_ids
        ):
            raise SelectionContractError(
                f"invalid canonical validation row {clip_id!r}"
            )
        output_id = canonical_clip_id(clip_id)
        if output_id in seen_output_ids:
            raise SelectionContractError(
                f"canonical validation output-ID collision: {output_id}"
            )
        seen_indices.add(index)
        seen_source_ids.add(clip_id)
        seen_output_ids.add(output_id)
        speakers.add(clip_id.split("/", 1)[0])
        ordered.append((index, output_id, frames))
        ordered_clips.append(
            {
                "global_index": index,
                "source_clip_id": clip_id,
                "canonical_clip_id": output_id,
                "frames": frames,
            }
        )
    if [item[0] for item in ordered] != sorted(seen_indices):
        raise SelectionContractError(
            "canonical validation manifest is not in global-index order"
        )
    if speakers != set(SHOW_SPEAKER_IDS):
        raise SelectionContractError(
            "canonical validation coverage is not the four SHOW speakers"
        )
    clip_digest = hashlib.sha256()
    diffsheg_digest = hashlib.sha256()
    frame_count = 0
    window_count = 0
    uncovered_tail_frames = 0
    for _, output_id, frames in ordered:
        clip_digest.update(output_id.encode("utf-8"))
        clip_digest.update(b"\n")
        starts = tuple(
            range(0, frames - DIFFSHEG_WINDOW + 1, DIFFSHEG_STRIDE)
        )
        if not starts:
            raise AssertionError("frame lower bound did not provide a window")
        frame_count += frames
        window_count += len(starts)
        uncovered_tail_frames += frames - (
            starts[-1] + DIFFSHEG_WINDOW
        )
        diffsheg_digest.update(output_id.encode("utf-8"))
        diffsheg_digest.update(b"\0")
        diffsheg_digest.update(str(frames).encode("ascii"))
        diffsheg_digest.update(b"\0")
        diffsheg_digest.update(
            ",".join(str(value) for value in starts).encode("ascii")
        )
        diffsheg_digest.update(b"\n")
    return seen_source_ids, {
        "split": "val",
        "clip_count": EXPECTED_VAL_CLIPS,
        "frame_count": frame_count,
        "window_count": window_count,
        "uncovered_tail_frames": uncovered_tail_frames,
        "clip_ids_sha256": clip_digest.hexdigest(),
        "diffsheg_clip_manifest_sha256": diffsheg_digest.hexdigest(),
        "_ordered_clips": ordered_clips,
    }


def public_val_coverage(coverage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: coverage[key]
        for key in (
            "split",
            "clip_count",
            "frame_count",
            "window_count",
            "uncovered_tail_frames",
            "clip_ids_sha256",
            "diffsheg_clip_manifest_sha256",
        )
    }


def _audio_coverage(
    manifest_receipts: Sequence[Any],
    summary_receipts: Sequence[Any],
    lineage_receipts: Sequence[Any],
    canonical_ids: set[str],
) -> dict[str, Any]:
    if not (
        len(manifest_receipts)
        == len(summary_receipts)
        == len(lineage_receipts)
        == EXPECTED_AUDIO_SHARDS
    ):
        raise SelectionContractError(
            "validation audio inputs require exactly eight "
            "manifest/summary/lineage receipts"
        )

    manifests: dict[int, dict[str, Any]] = {}
    audio_ids: set[str] = set()
    audio_paths: set[Path] = set()
    for receipt_value in manifest_receipts:
        artifact, resolved, payload = _verify_artifact(
            receipt_value,
            "validation audio manifest",
        )
        reject_test_path(resolved, "validation audio manifest")
        rows = _strict_jsonl(payload, f"validation audio manifest {resolved}")
        shard_ids: set[int] = set()
        for row in rows:
            clip_id = row.get("clip_id")
            shard_id = require_exact_int(
                row.get("shard_id"),
                "validation audio shard_id",
            )
            num_shards = require_exact_int(
                row.get("num_shards"),
                "validation audio num_shards",
            )
            if (
                row.get("format") != "semtalk_show_audio_clip_v1"
                or row.get("split") != "val"
                or num_shards != EXPECTED_AUDIO_SHARDS
                or not isinstance(clip_id, str)
                or not clip_id
                or clip_id in audio_ids
            ):
                raise SelectionContractError(
                    f"invalid validation audio row {clip_id!r}"
                )
            frames = require_exact_int(
                row.get("frames"),
                f"{clip_id} audio frames",
            )
            if (
                frames < DIFFSHEG_WINDOW
                or row.get("beat_shape") != [frames, 3]
                or row.get("hubert_shape") != [frames, 1024]
            ):
                raise SelectionContractError(
                    f"{clip_id}: invalid audio feature shapes"
                )
            audio_sha = require_sha256(
                row.get("audio_feature_npz_sha256"),
                f"{clip_id} audio feature SHA",
            )
            audio_path = require_val_only_path(
                row.get("audio_feature_npz"),
                f"{clip_id} audio feature path",
            )
            resolved_audio = _regular_file(
                audio_path,
                f"{clip_id} audio feature",
            )
            reject_test_path(
                resolved_audio,
                f"{clip_id} audio feature",
            )
            if (
                resolved_audio.name
                != f"{hashlib.sha256(clip_id.encode('utf-8')).hexdigest()}.npz"
                or resolved_audio in audio_paths
                or sha256_file(resolved_audio) != audio_sha
            ):
                raise SelectionContractError(
                    f"{clip_id}: audio feature receipt mismatch"
                )
            shard_ids.add(shard_id)
            audio_ids.add(clip_id)
            audio_paths.add(resolved_audio)
        if len(shard_ids) != 1:
            raise SelectionContractError(
                f"{resolved}: audio manifest must contain exactly one shard"
            )
        shard_id = shard_ids.pop()
        if shard_id in manifests:
            raise SelectionContractError(
                f"duplicate validation audio shard {shard_id}"
            )
        manifests[shard_id] = {
            "artifact": artifact,
            "rows": len(rows),
            "path": str(resolved),
        }
    if set(manifests) != set(range(EXPECTED_AUDIO_SHARDS)):
        raise SelectionContractError(
            "validation audio manifests do not cover shards 0..7"
        )
    if audio_ids != canonical_ids or len(audio_ids) != EXPECTED_VAL_CLIPS:
        raise SelectionContractError(
            "validation audio/canonical clip coverage is not exact"
        )

    summaries: dict[int, dict[str, Any]] = {}
    for receipt_value in summary_receipts:
        artifact, resolved, payload = _verify_artifact(
            receipt_value,
            "validation audio summary",
        )
        reject_test_path(resolved, "validation audio summary")
        summary = _strict_json_bytes(
            payload,
            f"validation audio summary {resolved}",
        )
        if not isinstance(summary, dict):
            raise SelectionContractError("audio summary must be an object")
        shard_id = require_exact_int(
            summary.get("shard_id"),
            "validation audio summary shard_id",
        )
        if (
            summary.get("format") != "semtalk_show_audio_summary_v1"
            or summary.get("status") != "complete"
            or summary.get("num_shards") != EXPECTED_AUDIO_SHARDS
            or summary.get("full_split_clips") != EXPECTED_VAL_CLIPS
            or shard_id not in manifests
            or shard_id in summaries
            or summary.get("output_manifest_sha256")
            != manifests[shard_id]["artifact"]["sha256"]
            or summary.get("shard_clips") != manifests[shard_id]["rows"]
        ):
            raise SelectionContractError(
                f"{resolved}: invalid validation audio summary"
            )
        summaries[shard_id] = artifact

    lineages: dict[int, dict[str, Any]] = {}
    for receipt_value in lineage_receipts:
        artifact, resolved, payload = _verify_artifact(
            receipt_value,
            "validation audio lineage",
        )
        reject_test_path(resolved, "validation audio lineage")
        lineage = _strict_json_bytes(
            payload,
            f"validation audio lineage {resolved}",
        )
        if not isinstance(lineage, dict):
            raise SelectionContractError("audio lineage must be an object")
        shard_id = require_exact_int(
            lineage.get("shard_id"),
            "validation audio lineage shard_id",
        )
        protocol = lineage.get("protocol")
        if (
            lineage.get("format") != "semtalk_show_audio_lineage_v1"
            or lineage.get("status") != "complete"
            or lineage.get("num_shards") != EXPECTED_AUDIO_SHARDS
            or lineage.get("full_split_clips") != EXPECTED_VAL_CLIPS
            or lineage.get("shard_clips") != manifests.get(
                shard_id,
                {},
            ).get("rows")
            or lineage.get("output_manifest_sha256")
            != manifests.get(shard_id, {}).get("artifact", {}).get("sha256")
            or not isinstance(protocol, dict)
            or protocol.get("split") != "val"
            or shard_id in lineages
        ):
            raise SelectionContractError(
                f"{resolved}: invalid validation audio lineage"
            )
        lineages[shard_id] = artifact
    if (
        set(summaries) != set(range(EXPECTED_AUDIO_SHARDS))
        or set(lineages) != set(range(EXPECTED_AUDIO_SHARDS))
    ):
        raise SelectionContractError(
            "validation audio summary/lineage coverage is incomplete"
        )
    return {
        "manifests": [manifests[index]["artifact"] for index in range(8)],
        "summaries": [summaries[index] for index in range(8)],
        "lineages": [lineages[index] for index in range(8)],
        "num_shards": EXPECTED_AUDIO_SHARDS,
        "clip_count": len(audio_ids),
        "exact_once": True,
    }


def _validate_val_canonical_receipts(
    *,
    summary_value: Any,
    lineage_value: Any,
    canonical_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_artifact, summary_path, summary_payload = _verify_artifact(
        summary_value,
        "validation canonical summary",
    )
    lineage_artifact, lineage_path, lineage_payload = _verify_artifact(
        lineage_value,
        "validation canonical lineage",
    )
    reject_test_path(summary_path, "validation canonical summary")
    reject_test_path(lineage_path, "validation canonical lineage")
    summary = require_exact_keys(
        _strict_json_bytes(summary_payload, str(summary_path)),
        {
            "format",
            "status",
            "split",
            "test_visible",
            "clip_count",
            "manifest_sha256",
            "lineage_sha256",
            "lineage_contract_sha256",
            "receipt_payload_sha256",
        },
        "validation canonical summary",
    )
    lineage = require_exact_keys(
        _strict_json_bytes(lineage_payload, str(lineage_path)),
        {
            "format",
            "status",
            "split",
            "test_visible",
            "clip_count",
            "manifest_sha256",
            "lineage_contract_sha256",
            "projection",
            "source_receipt",
            "receipt_payload_sha256",
        },
        "validation canonical lineage",
    )
    _payload_hash_without(
        summary,
        "receipt_payload_sha256",
        "validation canonical summary",
    )
    _payload_hash_without(
        lineage,
        "receipt_payload_sha256",
        "validation canonical lineage",
    )
    projection = require_exact_keys(
        lineage["projection"],
        {"operation", "split", "test_rows_materialized"},
        "validation canonical projection",
    )
    source = require_exact_keys(
        lineage["source_receipt"],
        {
            "origin",
            "commit",
            "tree",
            "full_manifest_sha256",
            "full_summary_sha256",
            "full_lineage_sha256",
        },
        "validation canonical source receipt",
    )
    if (
        summary["format"] != VAL_CANONICAL_SUMMARY_FORMAT
        or lineage["format"] != VAL_CANONICAL_LINEAGE_FORMAT
        or summary["status"] != "complete"
        or lineage["status"] != "complete"
        or summary["split"] != "val"
        or lineage["split"] != "val"
        or summary["test_visible"] is not False
        or lineage["test_visible"] is not False
        or require_exact_int(
            summary["clip_count"],
            "validation canonical summary clip_count",
        )
        != EXPECTED_VAL_CLIPS
        or require_exact_int(
            lineage["clip_count"],
            "validation canonical lineage clip_count",
        )
        != EXPECTED_VAL_CLIPS
        or summary["manifest_sha256"] != canonical_manifest_sha256
        or lineage["manifest_sha256"] != canonical_manifest_sha256
        or summary["lineage_sha256"] != lineage_artifact["sha256"]
        or require_sha256(
            summary["lineage_contract_sha256"],
            "validation canonical summary lineage contract SHA",
        )
        != require_sha256(
            lineage["lineage_contract_sha256"],
            "validation canonical lineage contract SHA",
        )
        or projection
        != {
            "operation": "filter_exact_split",
            "split": "val",
            "test_rows_materialized": False,
        }
        or source["origin"] != BASE_PRODUCER_SOURCE["origin"]
    ):
        raise SelectionContractError(
            "canonical val-view summary/lineage binding mismatch"
        )
    require_git_oid(
        source["commit"],
        "canonical val-view source commit",
    )
    require_git_oid(
        source["tree"],
        "canonical val-view source tree",
    )
    for key in (
        "full_manifest_sha256",
        "full_summary_sha256",
        "full_lineage_sha256",
    ):
        require_sha256(source[key], f"canonical val-view source {key}")
    return summary_artifact, lineage_artifact


def validate_val_inputs(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, inputs, file_sha = _verified_json(
        path,
        expected_sha256,
        "Base validation inputs",
    )
    reject_test_path(resolved, "Base validation inputs")
    inputs = require_exact_keys(
        inputs,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "expected_clip_count",
            "canonical_manifest",
            "canonical_summary",
            "canonical_lineage",
            "audio_manifests",
            "audio_summaries",
            "audio_lineages",
            "clip_ids_sha256",
            "diffsheg_clip_manifest_sha256",
            "receipt_payload_sha256",
        },
        "Base validation inputs",
    )
    receipt_payload_sha = _payload_hash_without(
        inputs,
        "receipt_payload_sha256",
        "Base validation inputs",
    )
    if (
        inputs["format"] != VAL_INPUTS_FORMAT
        or inputs["status"] != "frozen"
        or inputs["split"] != "val"
        or inputs["test_visible"] is not False
        or inputs["expected_clip_count"] != EXPECTED_VAL_CLIPS
    ):
        raise SelectionContractError(
            "Base selector accepts only the frozen validation split"
        )
    canonical_artifact, canonical_path, canonical_payload = _verify_artifact(
        inputs["canonical_manifest"],
        "validation canonical manifest",
    )
    reject_test_path(canonical_path, "validation canonical manifest")
    canonical_ids, coverage = _canonical_coverage(
        canonical_payload,
        str(canonical_path),
    )
    if (
        require_sha256(
            inputs["clip_ids_sha256"],
            "validation clip ID SHA",
        )
        != coverage["clip_ids_sha256"]
        or require_sha256(
            inputs["diffsheg_clip_manifest_sha256"],
            "validation DiffSHEG clip-manifest SHA",
        )
        != coverage["diffsheg_clip_manifest_sha256"]
    ):
        raise SelectionContractError(
            "validation input coverage digest mismatch"
        )
    canonical_summary, canonical_lineage = _validate_val_canonical_receipts(
        summary_value=inputs["canonical_summary"],
        lineage_value=inputs["canonical_lineage"],
        canonical_manifest_sha256=canonical_artifact["sha256"],
    )
    audio = _audio_coverage(
        inputs["audio_manifests"],
        inputs["audio_summaries"],
        inputs["audio_lineages"],
        canonical_ids,
    )
    artifact = {
        "path": str(resolved),
        "sha256": file_sha,
        "receipt_payload_sha256": receipt_payload_sha,
    }
    return artifact, {
        **coverage,
        "canonical_manifest": canonical_artifact,
        "canonical_summary": canonical_summary,
        "canonical_lineage": canonical_lineage,
        "audio": audio,
    }


def validate_pipeline(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, pipeline, file_sha = _verified_json(
        path,
        expected_sha256,
        "Base validation pipeline",
    )
    reject_test_path(resolved, "Base validation pipeline")
    pipeline = require_exact_keys(
        pipeline,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "mode",
            "base_candidate_variable_only",
            "source",
            "inference_entrypoint",
            "inference_helpers",
            "fixed_checkpoints",
            "diffsheg",
            "receipt_payload_sha256",
        },
        "Base validation pipeline",
    )
    payload_sha = _payload_hash_without(
        pipeline,
        "receipt_payload_sha256",
        "Base validation pipeline",
    )
    source = require_exact_keys(
        pipeline["source"],
        {"origin", "commit", "tree", "entrypoint", "entrypoint_sha256"},
        "validation inference source",
    )
    if (
        pipeline["format"] != PIPELINE_FORMAT
        or pipeline["status"] != "frozen"
        or pipeline["split"] != "val"
        or pipeline["test_visible"] is not False
        or pipeline["mode"] != "official_show_adapt_v1"
        or pipeline["base_candidate_variable_only"] is not True
        or pipeline["inference_helpers"] != list(INFERENCE_HELPERS)
        or pipeline["diffsheg"] != DIFFSHEG_PINNED_RECEIPT
        or source != VAL_INFERENCE_SOURCE
    ):
        raise SelectionContractError(
            "validation pipeline is not the frozen final-pipeline contract"
        )
    entrypoint_artifact, entrypoint_path, _entrypoint_bytes = (
        _verify_artifact(
            pipeline["inference_entrypoint"],
            "validation inference entrypoint",
        )
    )
    if (
        entrypoint_path.name != "run_base_inference.py"
        or entrypoint_artifact["sha256"] != source["entrypoint_sha256"]
    ):
        raise SelectionContractError(
            "validation inference entrypoint source/hash mismatch"
        )
    fixed = pipeline["fixed_checkpoints"]
    if not isinstance(fixed, dict) or set(fixed) != {
        "face",
        "hands",
        "upper",
        "lower",
        "global",
    }:
        raise SelectionContractError(
            "validation pipeline fixed-checkpoint coverage mismatch"
        )
    for stage in ("face", "global"):
        entry = require_exact_keys(
            fixed[stage],
            {"path", "sha256", "selection_split", "test_visible"},
            f"fixed {stage} checkpoint",
        )
        reject_forbidden_source_labels(entry["path"])
        require_val_only_path(
            entry["path"],
            f"fixed {stage} checkpoint path",
        )
        require_sha256(entry["sha256"], f"fixed {stage} checkpoint SHA")
        expected_filename = (
            "best_face_transfer.bin"
            if stage == "face"
            else "best_global_transfer.bin"
        )
        if (
            Path(str(entry["path"])).name != expected_filename
            or entry["selection_split"] != "val"
            or entry["test_visible"] is not False
        ):
            raise SelectionContractError(
                f"fixed {stage} is not independently val-selected"
            )
    for stage, specification in FIXED_OFFICIAL_CHECKPOINTS.items():
        entry = require_exact_keys(
            fixed[stage],
            {"path", "sha256", "source"},
            f"fixed {stage} checkpoint",
        )
        reject_forbidden_source_labels(entry["path"])
        require_val_only_path(
            entry["path"],
            f"fixed {stage} checkpoint path",
        )
        if (
            Path(str(entry["path"])).name != specification["filename"]
            or entry["sha256"] != specification["sha256"]
            or entry["source"] != "released_all_speakers_v1"
        ):
            raise SelectionContractError(
                f"fixed {stage} is not the exact official checkpoint"
            )
    return {
        "path": str(resolved),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
    }, pipeline


def _validate_output_file_receipt(
    value: Any,
    *,
    expected_directory: Path,
    expected_filename: str,
    label: str,
) -> dict[str, Any]:
    value = require_exact_keys(
        value,
        {"path", "sha256", "bytes"},
        label,
    )
    path = require_val_only_path(value["path"], f"{label}.path")
    resolved = _regular_file(path, label)
    reject_test_path(resolved, label)
    if (
        resolved.parent != expected_directory
        or resolved.name != expected_filename
    ):
        raise SelectionContractError(f"{label} path mismatch")
    size = require_exact_int(value["bytes"], f"{label}.bytes")
    digest = require_sha256(value["sha256"], f"{label}.sha256")
    if (
        size <= 0
        or resolved.stat().st_size != size
        or sha256_file(resolved) != digest
    ):
        raise SelectionContractError(
            f"{label} bytes do not match the bound output artifact"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "bytes": size,
    }


def validate_val_inference_lineage(
    path: Path,
    expected_sha256: str,
    *,
    epoch: int,
    expected_candidate: Mapping[str, Any],
    val_inputs_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
    expected_coverage: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one candidate's exact val output manifest without importing torch."""

    resolved, lineage, file_sha = _verified_json(
        path,
        expected_sha256,
        f"epoch {epoch} validation inference lineage",
    )
    reject_test_path(
        resolved,
        f"epoch {epoch} validation inference lineage",
    )
    lineage = require_exact_keys(
        lineage,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "epoch",
            "candidate_checkpoint",
            "val_inputs_receipt",
            "pipeline_receipt",
            "prediction_dir",
            "ground_truth_dir",
            "final_manifest",
            "clip_manifest",
            "clip_count",
            "frame_count",
            "window_count",
            "uncovered_tail_frames",
            "clip_ids_sha256",
            "diffsheg_clip_manifest_sha256",
            "prediction_files",
            "ground_truth_files",
            "exact_once",
            "finite",
            "receipt_payload_sha256",
        },
        f"epoch {epoch} validation inference lineage",
    )
    payload_sha = _payload_hash_without(
        lineage,
        "receipt_payload_sha256",
        f"epoch {epoch} validation inference lineage",
    )
    lineage_epoch = require_exact_int(
        lineage["epoch"],
        "validation inference lineage epoch",
    )
    candidate_path, candidate_sha, _ = _artifact_fields(
        lineage["candidate_checkpoint"],
        f"epoch {epoch} lineage candidate checkpoint",
    )
    if (
        lineage["format"] != VAL_INFERENCE_LINEAGE_FORMAT
        or lineage["status"] != "complete"
        or lineage["split"] != "val"
        or lineage["test_visible"] is not False
        or lineage_epoch != epoch
        or candidate_path.resolve() != Path(expected_candidate["path"])
        or candidate_sha != expected_candidate["sha256"]
        or lineage["val_inputs_receipt"] != val_inputs_artifact
        or lineage["pipeline_receipt"] != pipeline_artifact
        or lineage["exact_once"] is not True
        or lineage["finite"] is not True
    ):
        raise SelectionContractError(
            f"epoch {epoch} inference lineage is not candidate-bound val-only"
        )

    prediction_dir = require_directory(
        lineage["prediction_dir"],
        f"epoch {epoch} prediction directory",
    )
    ground_truth_dir = require_directory(
        lineage["ground_truth_dir"],
        f"epoch {epoch} ground-truth directory",
    )
    final_artifact, final_path, final_payload = _verify_artifact(
        lineage["final_manifest"],
        f"epoch {epoch} final inference manifest",
    )
    clip_artifact, clip_path, clip_payload = _verify_artifact(
        lineage["clip_manifest"],
        f"epoch {epoch} DiffSHEG clip manifest",
    )
    reject_test_path(final_path, "final inference manifest")
    reject_test_path(clip_path, "DiffSHEG clip manifest")
    if final_path.name != "final_manifest.jsonl":
        raise SelectionContractError("final inference manifest basename mismatch")
    if clip_path.name != "diffsheg_eval_clip_ids.txt":
        raise SelectionContractError("DiffSHEG clip manifest basename mismatch")

    expected_rows = expected_coverage.get("_ordered_clips")
    if (
        not isinstance(expected_rows, list)
        or len(expected_rows) != EXPECTED_VAL_CLIPS
    ):
        raise SelectionContractError("canonical val row coverage is unavailable")
    rows = _strict_jsonl(
        final_payload,
        f"epoch {epoch} final inference manifest {final_path}",
    )
    if len(rows) != EXPECTED_VAL_CLIPS:
        raise SelectionContractError(
            f"epoch {epoch} final inference rows must equal "
            f"{EXPECTED_VAL_CLIPS}"
        )
    prediction_paths: set[str] = set()
    ground_truth_paths: set[str] = set()
    row_keys = {
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
    for row_number, (raw_row, canonical_row) in enumerate(
        zip(rows, expected_rows),
    ):
        row = require_exact_keys(
            raw_row,
            row_keys,
            f"epoch {epoch} inference row {row_number}",
        )
        output_id = canonical_row["canonical_clip_id"]
        if (
            require_exact_int(
                row["global_index"],
                f"epoch {epoch} inference row global_index",
            )
            != canonical_row["global_index"]
            or row["split"] != "val"
            or row["source_clip_id"] != canonical_row["source_clip_id"]
            or row["canonical_clip_id"] != output_id
            or require_exact_int(
                row["frames"],
                f"epoch {epoch} inference row frames",
            )
            != canonical_row["frames"]
            or require_exact_int(
                row["epoch"],
                f"epoch {epoch} inference row candidate epoch",
            )
            != epoch
            or row["candidate_checkpoint_sha256"]
            != expected_candidate["sha256"]
        ):
            raise SelectionContractError(
                f"epoch {epoch} inference row {row_number} "
                "does not match canonical val/candidate lineage"
            )
        prediction = _validate_output_file_receipt(
            row["prediction"],
            expected_directory=prediction_dir,
            expected_filename=f"res_{output_id}.npz",
            label=f"epoch {epoch} {output_id} prediction",
        )
        ground_truth = _validate_output_file_receipt(
            row["ground_truth"],
            expected_directory=ground_truth_dir,
            expected_filename=f"gt_{output_id}.npz",
            label=f"epoch {epoch} {output_id} ground truth",
        )
        if (
            prediction["path"] in prediction_paths
            or ground_truth["path"] in ground_truth_paths
        ):
            raise SelectionContractError(
                f"epoch {epoch} inference file coverage is not exact-once"
            )
        prediction_paths.add(prediction["path"])
        ground_truth_paths.add(ground_truth["path"])

    for directory, expected_paths, role in (
        (prediction_dir, prediction_paths, "prediction"),
        (ground_truth_dir, ground_truth_paths, "ground-truth"),
    ):
        children = list(directory.iterdir())
        actual_paths = {
            str(child.resolve())
            for child in children
            if child.is_file() and not child.is_symlink()
        }
        if (
            len(children) != EXPECTED_VAL_CLIPS
            or actual_paths != expected_paths
        ):
            raise SelectionContractError(
                f"epoch {epoch} {role} directory is not the exact "
                "1,715-file cover"
            )

    expected_clip_payload = "".join(
        f"{row['canonical_clip_id']}\n" for row in expected_rows
    ).encode("utf-8")
    public_coverage = public_val_coverage(expected_coverage)
    counts = {
        key: require_exact_int(
            lineage[key],
            f"validation inference lineage {key}",
        )
        for key in (
            "clip_count",
            "frame_count",
            "window_count",
            "uncovered_tail_frames",
        )
    }
    if (
        clip_payload != expected_clip_payload
        or clip_artifact["sha256"] != public_coverage["clip_ids_sha256"]
        or counts
        != {
            key: public_coverage[key]
            for key in (
                "clip_count",
                "frame_count",
                "window_count",
                "uncovered_tail_frames",
            )
        }
        or require_exact_int(
            lineage["prediction_files"],
            "validation inference prediction_files",
        )
        != EXPECTED_VAL_CLIPS
        or require_exact_int(
            lineage["ground_truth_files"],
            "validation inference ground_truth_files",
        )
        != EXPECTED_VAL_CLIPS
        or lineage["clip_ids_sha256"]
        != public_coverage["clip_ids_sha256"]
        or lineage["diffsheg_clip_manifest_sha256"]
        != public_coverage["diffsheg_clip_manifest_sha256"]
    ):
        raise SelectionContractError(
            f"epoch {epoch} inference lineage does not exactly cover "
            "the 1,715 canonical val clips"
        )
    artifact = {
        "path": str(resolved),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
    }
    return artifact, {
        "prediction_dir": str(prediction_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "final_manifest": final_artifact,
        "clip_manifest": clip_artifact,
        "coverage": public_coverage,
    }


def validate_diffsheg_report(
    report: Any,
    *,
    expected_coverage: Mapping[str, Any],
    inference_lineage: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    if not isinstance(report, dict) or report.get("status") != "ok":
        raise SelectionContractError("DiffSHEG validation report is incomplete")
    protocol = report.get("protocol")
    inputs = report.get("inputs")
    metrics = report.get("metrics")
    provenance = report.get("provenance")
    if not all(
        isinstance(value, dict)
        for value in (protocol, inputs, metrics, provenance)
    ):
        raise SelectionContractError("DiffSHEG report schema is incomplete")
    assert isinstance(protocol, dict)
    assert isinstance(inputs, dict)
    assert isinstance(metrics, dict)
    assert isinstance(provenance, dict)
    pins = DIFFSHEG_PINNED_RECEIPT
    public_coverage = public_val_coverage(expected_coverage)
    expected_clip_manifest_path = (
        inference_lineage["clip_manifest"]["path"]
        if inference_lineage is not None
        else None
    )
    expected_clip_manifest_sha256 = public_coverage[
        "diffsheg_clip_manifest_sha256"
    ]
    expected_clip_order = (
        f"explicit_manifest:{expected_clip_manifest_path}"
        if expected_clip_manifest_path is not None
        else None
    )
    if (
        protocol.get("name") != pins["protocol"]
        or protocol.get("version") != pins["protocol_version"]
        or protocol.get("status")
        != "reconstructed_from_public_components"
        or protocol.get("diffsheg_reference_commit")
        != pins["diffsheg_reference_commit"]
        or protocol.get("window_length") != DIFFSHEG_WINDOW
        or protocol.get("window_stride") != DIFFSHEG_STRIDE
        or protocol.get("precision") != pins["precision"]
        or protocol.get("ba") is not None
        or (
            protocol.get("clip_order") != expected_clip_order
            if expected_clip_order is not None
            else not str(protocol.get("clip_order", "")).startswith(
                "explicit_manifest:"
            )
        )
    ):
        raise SelectionContractError("DiffSHEG validation protocol mismatch")
    clip_count = require_exact_int(
        inputs.get("clip_count"),
        "DiffSHEG validation clip_count",
    )
    frame_count = require_exact_int(
        inputs.get("frame_count"),
        "DiffSHEG validation frame_count",
    )
    window_count = require_exact_int(
        inputs.get("window_count"),
        "DiffSHEG validation window_count",
    )
    uncovered = require_exact_int(
        inputs.get("uncovered_tail_frames"),
        "DiffSHEG validation uncovered_tail_frames",
    )
    if (
        clip_count != public_coverage["clip_count"]
        or frame_count != public_coverage["frame_count"]
        or window_count != public_coverage["window_count"]
        or uncovered != public_coverage["uncovered_tail_frames"]
        or inputs.get("clip_manifest_sha256")
        != expected_clip_manifest_sha256
        or not isinstance(inputs.get("stats"), dict)
        or inputs["stats"].get("sha256") != pins["stats_sha256"]
    ):
        raise SelectionContractError(
            "DiffSHEG report does not exactly cover frozen validation"
        )
    for field in (
        "prediction_dir",
        "ground_truth_dir",
        "clip_manifest",
        "weights_dir",
    ):
        require_val_only_path(
            inputs.get(field),
            f"DiffSHEG inputs.{field}",
        )
    if inference_lineage is not None:
        if (
            str(
                require_val_only_path(
                    inputs["prediction_dir"],
                    "DiffSHEG inputs.prediction_dir",
                ).resolve()
            )
            != inference_lineage["prediction_dir"]
            or str(
                require_val_only_path(
                    inputs["ground_truth_dir"],
                    "DiffSHEG inputs.ground_truth_dir",
                ).resolve()
            )
            != inference_lineage["ground_truth_dir"]
            or str(
                require_val_only_path(
                    inputs["clip_manifest"],
                    "DiffSHEG inputs.clip_manifest",
                ).resolve()
            )
            != expected_clip_manifest_path
        ):
            raise SelectionContractError(
                "DiffSHEG report paths do not bind the candidate inference "
                "lineage"
            )
    require_absolute_path(
        inputs["stats"].get("path"),
        "DiffSHEG inputs.stats.path",
    )
    if set(metrics) != set(VAL_METRIC_KEYS):
        raise SelectionContractError(
            "validation metric coverage must be exactly FGD-only"
        )
    validated_metrics = {
        key: require_finite_number(metrics[key], f"DiffSHEG metric {key}")
        for key in VAL_METRIC_KEYS
    }
    if validated_metrics["fgd"] < 0.0:
        raise SelectionContractError("DiffSHEG metric fgd is negative")

    evaluator = provenance.get("evaluator")
    diffsheg_root = provenance.get("diffsheg_root")
    autoencoders = provenance.get("autoencoders")
    if (
        not isinstance(evaluator, dict)
        or evaluator.get("sha256")
        != pins["paspa"]["evaluator_sha256"]
        or evaluator.get("repository_git_head")
        != pins["paspa"]["commit"]
        or not isinstance(diffsheg_root, dict)
        or diffsheg_root.get("git_head")
        != pins["diffsheg_reference_commit"]
        or not isinstance(autoencoders, dict)
        or set(autoencoders) != {"fgd"}
    ):
        raise SelectionContractError("DiffSHEG evaluator provenance mismatch")
    require_absolute_path(
        evaluator.get("path"),
        "DiffSHEG evaluator path",
    )
    require_absolute_path(
        evaluator.get("repository_root"),
        "DiffSHEG evaluator repository root",
    )
    require_absolute_path(
        diffsheg_root.get("path"),
        "DiffSHEG reference checkout",
    )
    for metric_name, specification in pins["autoencoders"].items():
        observed = autoencoders[metric_name]
        if (
            not isinstance(observed, dict)
            or observed.get("sha256") != specification["sha256"]
            or observed.get("input_dim") != specification["input_dim"]
            or Path(str(observed.get("path", ""))).name
            != specification["filename"]
        ):
            raise SelectionContractError(
                f"DiffSHEG {metric_name} autoencoder provenance mismatch"
            )
        require_absolute_path(
            observed["path"],
            f"DiffSHEG {metric_name} autoencoder path",
        )
    return validated_metrics, {
        "clip_count": clip_count,
        "frame_count": frame_count,
        "window_count": window_count,
        "uncovered_tail_frames": uncovered,
    }


def select_minimum_fgd(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the strict minimum ``(fgd, epoch)`` from the exact epoch set."""

    epochs = [
        require_exact_int(row.get("epoch"), "selection epoch")
        for row in rows
    ]
    if epochs != list(EXPECTED_CANDIDATE_EPOCHS):
        raise SelectionContractError(
            "selection rows must exactly cover epochs "
            "1/2/4/8/16/32/40 in order"
        )
    for row in rows:
        metrics = row.get("metrics")
        require_exact_keys(
            dict(metrics) if isinstance(metrics, Mapping) else metrics,
            {"fgd"},
            "selection row metrics",
        )
        require_finite_number(metrics["fgd"], "selection FGD")
    return min(
        rows,
        key=lambda row: (
            float(row["metrics"]["fgd"]),
            int(row["epoch"]),
        ),
    )


def _measurement_artifact(
    path: Path,
    expected_sha256: str,
    *,
    candidates: Mapping[int, Mapping[str, Any]],
    common_val_inputs: dict[str, Any] | None,
    common_pipeline: dict[str, Any] | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    resolved, measurement, file_sha = _verified_json(
        path,
        expected_sha256,
        "Base validation measurement",
    )
    reject_test_path(resolved, "Base validation measurement")
    measurement = require_exact_keys(
        measurement,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "selection_eligible",
            "epoch",
            "candidate_checkpoint",
            "val_inputs_receipt",
            "pipeline_receipt",
            "inference_lineage",
            "diffsheg_report",
            "receipt_payload_sha256",
        },
        "Base validation measurement",
    )
    payload_sha = _payload_hash_without(
        measurement,
        "receipt_payload_sha256",
        "Base validation measurement",
    )
    epoch = require_exact_int(measurement["epoch"], "measurement epoch")
    if (
        measurement["format"] != MEASUREMENT_FORMAT
        or measurement["status"] != "complete"
        or measurement["split"] != "val"
        or measurement["test_visible"] is not False
        or measurement["selection_eligible"] is not True
        or epoch not in candidates
    ):
        raise SelectionContractError(
            "Base measurement is not validation-only and selection-eligible"
        )
    candidate_path, candidate_sha, _ = _artifact_fields(
        measurement["candidate_checkpoint"],
        f"epoch {epoch} candidate checkpoint",
    )
    expected_candidate = candidates[epoch]
    if (
        candidate_path.resolve() != Path(expected_candidate["path"])
        or candidate_sha != expected_candidate["sha256"]
    ):
        raise SelectionContractError(
            f"measurement epoch {epoch} does not bind its candidate"
        )

    val_path, val_sha, val_payload_sha = _artifact_fields(
        measurement["val_inputs_receipt"],
        "measurement val-input receipt",
        payload_hash=True,
    )
    if common_val_inputs is None:
        val_artifact, val_coverage = validate_val_inputs(val_path, val_sha)
        if val_artifact["receipt_payload_sha256"] != val_payload_sha:
            raise SelectionContractError(
                "measurement val-input payload hash mismatch"
            )
        common_val_inputs = {
            "artifact": val_artifact,
            "coverage": val_coverage,
        }
    elif measurement["val_inputs_receipt"] != common_val_inputs["artifact"]:
        raise SelectionContractError(
            "all candidates must use one frozen validation input receipt"
        )

    pipeline_path, pipeline_sha, pipeline_payload_sha = _artifact_fields(
        measurement["pipeline_receipt"],
        "measurement pipeline receipt",
        payload_hash=True,
    )
    if common_pipeline is None:
        pipeline_artifact, pipeline = validate_pipeline(
            pipeline_path,
            pipeline_sha,
        )
        if (
            pipeline_artifact["receipt_payload_sha256"]
            != pipeline_payload_sha
        ):
            raise SelectionContractError(
                "measurement pipeline payload hash mismatch"
            )
        common_pipeline = {
            "artifact": pipeline_artifact,
            "payload": pipeline,
        }
    elif measurement["pipeline_receipt"] != common_pipeline["artifact"]:
        raise SelectionContractError(
            "all candidates must use one frozen validation pipeline"
        )

    (
        inference_path,
        inference_sha,
        inference_payload_sha,
    ) = _artifact_fields(
        measurement["inference_lineage"],
        f"epoch {epoch} inference lineage",
        payload_hash=True,
    )
    inference_artifact, inference = validate_val_inference_lineage(
        inference_path,
        inference_sha,
        epoch=epoch,
        expected_candidate=expected_candidate,
        val_inputs_artifact=common_val_inputs["artifact"],
        pipeline_artifact=common_pipeline["artifact"],
        expected_coverage=common_val_inputs["coverage"],
    )
    if (
        inference_artifact["receipt_payload_sha256"]
        != inference_payload_sha
    ):
        raise SelectionContractError(
            f"epoch {epoch} inference lineage payload hash mismatch"
        )

    report_artifact, report_path, report_bytes = _verify_artifact(
        measurement["diffsheg_report"],
        f"epoch {epoch} DiffSHEG report",
    )
    reject_test_path(report_path, f"epoch {epoch} DiffSHEG report")
    report = _strict_json_bytes(
        report_bytes,
        f"epoch {epoch} DiffSHEG report {report_path}",
    )
    metrics, coverage = validate_diffsheg_report(
        report,
        expected_coverage=common_val_inputs["coverage"],
        inference_lineage=inference,
    )
    row = {
        "epoch": epoch,
        "candidate_checkpoint": dict(expected_candidate),
        "inference_lineage": inference_artifact,
        "inference_outputs": inference,
        "diffsheg_report": report_artifact,
        "metrics": metrics,
        "coverage": coverage,
    }
    artifact = {
        "path": str(resolved),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
    }
    return artifact, row, common_val_inputs, common_pipeline


def build_selection(
    *,
    candidate_bundle: Mapping[str, Any],
    measurement_paths: Sequence[Path],
    expected_measurement_sha256: Sequence[str],
) -> dict[str, Any]:
    if (
        len(measurement_paths) != len(EXPECTED_CANDIDATE_EPOCHS)
        or len(expected_measurement_sha256)
        != len(EXPECTED_CANDIDATE_EPOCHS)
    ):
        raise SelectionContractError(
            "exactly seven measurement paths and seven external SHA roots "
            "are required"
        )
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    common_val_inputs: dict[str, Any] | None = None
    common_pipeline: dict[str, Any] | None = None
    candidates = candidate_bundle["candidates"]
    resolved_measurement_paths = []
    for path in measurement_paths:
        resolved_path = Path(path).resolve()
        reject_forbidden_source_labels(resolved_path)
        if any("test" in piece.casefold() for piece in resolved_path.parts):
            raise SelectionContractError(
                "measurement paths must not expose test artifacts"
            )
        resolved_measurement_paths.append(str(resolved_path))
    if (
        len(set(resolved_measurement_paths))
        != len(EXPECTED_CANDIDATE_EPOCHS)
        or len(set(expected_measurement_sha256))
        != len(EXPECTED_CANDIDATE_EPOCHS)
    ):
        raise SelectionContractError(
            "the seven externally rooted measurement receipts must be distinct"
        )
    for path, digest in zip(
        measurement_paths,
        expected_measurement_sha256,
    ):
        artifact, row, common_val_inputs, common_pipeline = (
            _measurement_artifact(
                path,
                digest,
                candidates=candidates,
                common_val_inputs=common_val_inputs,
                common_pipeline=common_pipeline,
            )
        )
        artifacts.append(artifact)
        rows.append(row)
    selected = select_minimum_fgd(rows)
    coverage_values = [row["coverage"] for row in rows]
    if any(value != coverage_values[0] for value in coverage_values[1:]):
        raise SelectionContractError(
            "candidate DiffSHEG reports do not share exact validation coverage"
        )
    if common_val_inputs is None or common_pipeline is None:
        raise AssertionError("seven measurements did not bind common receipts")
    unique_fields = (
        ("inference_lineage", "path"),
        ("inference_lineage", "sha256"),
        ("diffsheg_report", "path"),
        ("diffsheg_report", "sha256"),
        ("inference_outputs", "prediction_dir"),
    )
    for outer, inner in unique_fields:
        if len({row[outer][inner] for row in rows}) != len(rows):
            raise SelectionContractError(
                f"candidate rows must bind seven distinct {outer}.{inner} "
                "values"
            )
    full_coverage = common_val_inputs["coverage"]
    selection_coverage = {
        **public_val_coverage(full_coverage),
        "canonical_manifest": full_coverage["canonical_manifest"],
        "canonical_summary": full_coverage["canonical_summary"],
        "canonical_lineage": full_coverage["canonical_lineage"],
        "audio": full_coverage["audio"],
    }
    selection = {
        "format": SELECTION_FORMAT,
        "status": "selected",
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "selection_policy": {
            "candidate_epochs": list(EXPECTED_CANDIDATE_EPOCHS),
            "metric": "FGD",
            "metric_report_key": "fgd",
            "operator": "min",
            "tie_break": "lowest_epoch",
            "ordering": ["fgd", "epoch"],
            "test_feedback_into_selection": False,
        },
        "candidate_bundle": {
            key: candidate_bundle[key]
            for key in ("manifest", "status", "frozen_inputs")
        },
        "source_roles": {
            "base_candidate_producer": BASE_PRODUCER_SOURCE,
            "val_inference_helpers": VAL_INFERENCE_SOURCE,
        },
        "val_inputs_receipt": common_val_inputs["artifact"],
        "val_coverage": selection_coverage,
        "pipeline_receipt": common_pipeline["artifact"],
        "diffsheg_pinned_receipt": DIFFSHEG_PINNED_RECEIPT,
        "measurement_receipts": artifacts,
        "candidate_metrics": rows,
        "selected": {
            "epoch": selected["epoch"],
            "candidate_checkpoint": selected["candidate_checkpoint"],
            "fgd": selected["metrics"]["fgd"],
            "inference_lineage": selected["inference_lineage"],
            "diffsheg_report": selected["diffsheg_report"],
        },
        "test_policy": {
            "authorized_evaluations": 1,
            "one_shot_claim_required": True,
            "selection_feedback": False,
        },
    }
    selection["receipt_payload_sha256"] = canonical_json_sha256(selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select official-adapt SemTalk Base from frozen SHOW validation "
            "FGD only"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--base-candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-base-candidate-manifest-sha256",
        required=True,
    )
    parser.add_argument("--base-status-json", type=Path, required=True)
    parser.add_argument(
        "--expected-base-formal-status-sha256",
        required=True,
    )
    parser.add_argument("--base-frozen-inputs-json", type=Path, required=True)
    parser.add_argument(
        "--expected-base-frozen-inputs-sha256",
        required=True,
    )
    parser.add_argument(
        "--measurement-json",
        action="append",
        type=Path,
        required=True,
        help=(
            "repeat exactly seven times in epoch order "
            "1,2,4,8,16,32,40"
        ),
    )
    parser.add_argument(
        "--expected-measurement-sha256",
        action="append",
        required=True,
        help="repeat exactly seven times in matching epoch order",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reject_forbidden_source_labels(args.output_json)
    resolved_output = args.output_json.resolve()
    reject_forbidden_source_labels(resolved_output)
    reject_test_path(resolved_output, "selection output")
    if args.output_json.is_symlink() or args.output_json.exists():
        raise SelectionContractError(
            f"refusing to overwrite existing selection: {args.output_json}"
        )
    candidate_bundle = validate_candidate_bundle(
        manifest_path=args.base_candidate_manifest,
        expected_manifest_sha256=(
            args.expected_base_candidate_manifest_sha256
        ),
        status_path=args.base_status_json,
        expected_status_sha256=args.expected_base_formal_status_sha256,
        frozen_inputs_path=args.base_frozen_inputs_json,
        expected_frozen_inputs_sha256=(
            args.expected_base_frozen_inputs_sha256
        ),
    )
    selection = build_selection(
        candidate_bundle=candidate_bundle,
        measurement_paths=args.measurement_json,
        expected_measurement_sha256=args.expected_measurement_sha256,
    )
    try:
        _receipt.atomic_json_new(args.output_json, selection)
    except (OSError, ValueError, RuntimeError) as error:
        raise SelectionContractError(str(error)) from error
    print(
        json.dumps(
            {
                "status": "selected",
                "split": "val",
                "test_visible": False,
                "selected_epoch": selection["selected"]["epoch"],
                "selected_fgd": selection["selected"]["fgd"],
                "output": str(args.output_json.resolve()),
                "receipt_payload_sha256": selection[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
