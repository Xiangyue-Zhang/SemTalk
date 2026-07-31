#!/usr/bin/env python3
"""Wait for one frozen SemTalk long-run validation wave to become immutable.

This is deliberately CPU-only.  It watches only the exact candidate receipt
paths named by the frozen schedule and never infers readiness from a checkpoint
file that may still be growing.  A successful invocation publishes one
write-once wave receipt which can be passed to a guarded GPU wave launcher.
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
import tempfile
import time
from typing import Any, Mapping, Sequence


SCHEDULE_SHA256 = (
    "013f8ade256f20579f9d545ac44da681238ed56af1cb687fc6fe9356c0bbefa3"
)
TRAJECTORY_ANCHOR_SHA256 = (
    "e27c27a0da2793f44608b618d08356df0b60f55ae039434a228c50ff73028cf2"
)
TRAINER_ENTRYPOINT_SHA256 = (
    "0970d1200aa779ec4b7a308c406f61f43c93a303ca487aa32f3ca944ee450ad7"
)
READY_FORMAT = "semtalk_show_base_official_adapt_long_candidate_ready_v1"
MANIFEST_FORMAT = "semtalk_show_base_official_adapt_long_manifest_v1"
FROZEN_INPUTS_FORMAT = "semtalk_show_base_official_adapt_frozen_inputs_v1"
PROTOCOL_FORMAT = "semtalk_show_base_official_adapt_long_protocol_v1"
THROUGHPUT_GATE_FORMAT = (
    "semtalk_show_base_official_adapt_long_throughput_gate_v1"
)
WAVE_FORMAT = "semtalk_show_base_long_val_wave_ready_v1"
MODEL_STATE_SCHEMA_SHA256 = (
    "e082b2699fe3bf676731f1ba1263562b965aeefb700c53d02eb99e71b1ac906a"
)
CANDIDATE_EPOCHS = (
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
VALIDATION_WAVES = (
    (1, 2, 4, 8),
    (16, 32, 40, 50),
    (60, 70, 80, 100),
    (120, 140, 160, 180),
    (200, 240, 280, 320),
    (360, 400),
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_COMPONENT = re.compile(
    r"(^|[^a-z0-9])(?:e(?:poch)?[-_]?30|speaker[-_]?2|semgate|sparse)"
    r"([^a-z0-9]|$)",
    re.IGNORECASE,
)
ANCHOR_EPOCHS = frozenset((1, 2, 4, 8, 16, 32, 40))
READY_KEYS = frozenset(
    (
        "format",
        "status",
        "selection_eligible",
        "test_visible",
        "epoch",
        "optimizer_updates",
        "candidate_checkpoint",
        "candidate_manifest",
        "frozen_inputs",
        "protocol",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "trajectory_anchor_match",
        "published_unix",
        "receipt_payload_sha256",
    )
)
CANDIDATE_CHECKPOINT_KEYS = frozenset(
    (
        "path",
        "relative_path",
        "sha256",
        "bytes",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
    )
)
CANDIDATE_MANIFEST_KEYS = frozenset(
    (
        "path",
        "sha256_at_ready",
        "entries_sha256_at_ready",
        "immutable_snapshot",
        "live_path",
    )
)
FROZEN_INPUT_KEYS = frozenset(
    ("path", "sha256", "receipt_payload_sha256")
)
PROTOCOL_KEYS = frozenset(("format", "payload_sha256"))
MANIFEST_KEYS = frozenset(
    (
        "format",
        "status",
        "candidate_epochs",
        "frozen_receipt_sha256",
        "schedule_sha256",
        "trajectory_anchor_sha256",
        "throughput_gate",
        "entries",
        "entries_sha256",
    )
)
MANIFEST_ENTRY_KEYS = frozenset(
    (
        "epoch",
        "optimizer_updates",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_container_schema",
        "model_state_tensors",
        "model_state_schema_sha256",
        "model_state_semantic_sha256",
        "all_model_state_tensors_finite",
        "frozen_receipt_sha256",
        "trajectory_anchor_match",
    )
)


class WatchContractError(RuntimeError):
    """Raised when an immutable candidate receipt fails validation."""


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if newline:
        encoded += "\n"
    return encoded.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, newline=False)
    ).hexdigest()


def payload_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_payload_sha256", None)
    return canonical_json_sha256(body)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise WatchContractError(f"{label} is not a canonical SHA-256")
    return value


def require_exact_keys(
    value: object, expected: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WatchContractError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise WatchContractError(
            f"{label} keys changed; missing={missing}, extra={extra}"
        )
    return value


def require_strict_bool(value: object, expected: bool, label: str) -> None:
    if value is not expected:
        raise WatchContractError(f"{label} must be {expected!r}")


def require_relative_file(
    value: object, expected: str, label: str
) -> str:
    if not isinstance(value, str) or value != expected:
        raise WatchContractError(f"{label} mismatch: {value!r} != {expected!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise WatchContractError(f"{label} is not a safe relative path")
    reject_forbidden(relative, label)
    return value


def reject_forbidden(value: object, label: str) -> None:
    for component in str(value).replace("\\", "/").split("/"):
        lowered = component.casefold()
        if "test" in lowered or FORBIDDEN_COMPONENT.search(lowered):
            raise WatchContractError(
                f"{label} has a forbidden component: {value}"
            )


def absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise WatchContractError(f"{label} must be a path")
    path = Path(value)
    if not path.is_absolute():
        raise WatchContractError(f"{label} must be absolute: {path}")
    reject_forbidden(path, label)
    return path


def regular_file(value: object, label: str) -> Path:
    path = absolute_path(value, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise WatchContractError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise WatchContractError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    reject_forbidden(resolved, label)
    return resolved


def require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        raise WatchContractError(
            f"{label} escapes the training root: {path}"
        ) from None


def directory(value: object, label: str) -> Path:
    path = absolute_path(value, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise WatchContractError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise WatchContractError(
            f"{label} must be a regular non-symlink directory: {path}"
        )
    resolved = path.resolve(strict=True)
    reject_forbidden(resolved, label)
    return resolved


def verified_file(value: object, expected_sha: str, label: str) -> Path:
    expected = require_sha256(expected_sha, f"{label} expected SHA")
    path = regular_file(value, label)
    observed = sha256_file(path)
    if observed != expected:
        raise WatchContractError(
            f"{label} SHA mismatch: {observed} != {expected}"
        )
    return path


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise WatchContractError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8", errors="strict"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            WatchContractError(f"non-finite JSON token {token!r} in {path}")
        ),
    )
    if not isinstance(value, dict):
        raise WatchContractError(f"expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_json(path_value: object, value: Mapping[str, Any]) -> Path:
    path = absolute_path(path_value, "wave receipt output")
    parent = directory(path.parent, "wave receipt parent")
    output = parent / path.name
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.partial-", suffix=".json", dir=parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {output}") from None
        fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_schedule(
    schedule_value: object, expected_sha: str
) -> tuple[Path, dict[str, Any]]:
    if expected_sha != SCHEDULE_SHA256:
        raise WatchContractError("watcher only accepts the frozen schedule SHA")
    schedule = verified_file(schedule_value, expected_sha, "long schedule")
    value = strict_json(schedule)
    if (
        value.get("format") != "semtalk_show_base_long_schedule_v1"
        or tuple(value.get("candidate_epochs", ())) != CANDIDATE_EPOCHS
        or tuple(tuple(wave) for wave in value.get("validation_waves", ()))
        != VALIDATION_WAVES
        or value.get("training", {}).get("updates_per_epoch") != 248
        or value.get("training", {}).get("total_epochs") != 400
        or value.get("selection", {}).get("test_visible") is not False
        or value.get("selection", {}).get("ordering") != ["FGD", "epoch"]
        or value.get("selection", {}).get("direction") != ["min", "min"]
    ):
        raise WatchContractError("invalid frozen long-run schedule")
    return schedule, value


def validate_anchor(
    anchor_value: object, expected_sha: str
) -> tuple[Path, dict[str, Any]]:
    if expected_sha != TRAJECTORY_ANCHOR_SHA256:
        raise WatchContractError(
            "watcher only accepts the frozen trajectory-anchor SHA"
        )
    anchor = verified_file(anchor_value, expected_sha, "trajectory anchor")
    value = strict_json(anchor)
    if (
        value.get("format") != "semtalk_show_base_long_trajectory_anchor_v1"
        or set(value.get("entries", {})) != {
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
            "40",
        }
    ):
        raise WatchContractError("invalid trajectory anchor")
    for epoch_text, entry in value["entries"].items():
        if (
            not isinstance(entry, dict)
            or type(entry.get("tensor_count")) is not int
            or entry["tensor_count"] != 1_790
        ):
            raise WatchContractError(
                f"invalid trajectory anchor entry e{epoch_text}"
            )
        require_sha256(
            entry.get("checkpoint_sha256"),
            f"trajectory anchor e{epoch_text} checkpoint SHA",
        )
        require_sha256(
            entry.get("model_state_semantic_sha256"),
            f"trajectory anchor e{epoch_text} semantic SHA",
        )
    return anchor, value


def validate_frozen_inputs(
    *,
    train_root: Path,
    value: object,
    expected_receipt_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = require_exact_keys(
        value, FROZEN_INPUT_KEYS, "candidate frozen-input reference"
    )
    expected_path = train_root / "frozen_inputs.json"
    path = regular_file(receipt["path"], "candidate frozen inputs")
    if path != expected_path:
        raise WatchContractError(
            f"frozen-input path mismatch: {path} != {expected_path}"
        )
    claimed_sha = require_sha256(
        receipt["sha256"], "candidate frozen-input file SHA"
    )
    if sha256_file(path) != claimed_sha:
        raise WatchContractError("candidate frozen-input file changed")
    reference_payload_sha = require_sha256(
        receipt["receipt_payload_sha256"],
        "candidate frozen-input payload SHA",
    )
    expected_receipt_sha = require_sha256(
        expected_receipt_sha, "ready frozen-receipt SHA"
    )
    if reference_payload_sha != expected_receipt_sha:
        raise WatchContractError(
            "candidate frozen-input reference payload SHA changed"
        )

    frozen = strict_json(path)
    expected_frozen_keys = {
        "format",
        "run_purpose",
        "target_epochs",
        "source",
        "official_base",
        "speaker_initialization",
        "dataset",
        "protocol",
        "long_contract",
        "receipt_sha256",
    }
    if set(frozen) != expected_frozen_keys:
        raise WatchContractError("frozen-input receipt schema changed")
    if frozen.get("format") != FROZEN_INPUTS_FORMAT:
        raise WatchContractError("frozen-input receipt format changed")
    if (
        frozen.get("run_purpose") != "formal_training"
        or frozen.get("target_epochs") != list(CANDIDATE_EPOCHS)
    ):
        raise WatchContractError(
            "frozen-input receipt is not the formal 400-epoch run"
        )
    frozen_payload_sha = require_sha256(
        frozen.get("receipt_sha256"), "frozen-input receipt payload SHA"
    )
    frozen_body = dict(frozen)
    frozen_body.pop("receipt_sha256")
    if (
        canonical_json_sha256(frozen_body) != frozen_payload_sha
        or frozen_payload_sha != expected_receipt_sha
    ):
        raise WatchContractError("frozen-input receipt payload mismatch")

    source = frozen.get("source")
    if (
        not isinstance(source, dict)
        or source.get("origin")
        != "git@github.com:Xiangyue-Zhang/SemTalk.git"
        or source.get("clean") is not True
        or source.get("branch") is not None
    ):
        raise WatchContractError(
            "frozen source is not the clean detached SemTalk trust root"
        )
    entrypoint_sha = require_sha256(
        source.get("entrypoint_sha256"), "trainer entrypoint SHA"
    )
    if entrypoint_sha != TRAINER_ENTRYPOINT_SHA256:
        raise WatchContractError(
            "frozen trainer entrypoint is not the audited long-run source"
        )
    for name in ("commit", "tree"):
        identifier = source.get(name)
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[0-9a-f]{40}", identifier) is None
        ):
            raise WatchContractError(f"frozen source {name} is invalid")
    reject_forbidden(source.get("entrypoint", ""), "trainer entrypoint")

    official = frozen.get("official_base")
    if (
        not isinstance(official, dict)
        or official.get("source") != "released_all_speakers_v1"
        or official.get("sha256")
        != "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
        or official.get("speaker_scope") != "All-Speakers"
        or official.get("training_dataset") != "BEAT2"
        or official.get("model_state_tensors") != 1_790
        or official.get("model_state_schema_sha256")
        != MODEL_STATE_SCHEMA_SHA256
        or official.get("all_model_state_tensors_finite") is not True
        or official.get("strict_state_dict_load") is not True
    ):
        raise WatchContractError(
            "frozen official Base checkpoint is not All-Speakers"
        )
    reject_forbidden(official.get("path", ""), "official Base checkpoint")

    protocol = frozen.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("scope") != "SemTalk Base only"
        or protocol.get("target_dataset") != "SHOW"
        or protocol.get("target_speaker_scope") != "All"
        or protocol.get("world_size") != 8
        or protocol.get("local_batch_size") != 64
        or protocol.get("global_batch_size") != 512
        or protocol.get("expected_updates_per_epoch") != 248
        or protocol.get("epochs") != 400
        or protocol.get("candidate_epochs") != list(CANDIDATE_EPOCHS)
        or protocol.get("trajectory_anchor_epochs")
        != sorted(ANCHOR_EPOCHS)
        or protocol.get("precision") != "bf16"
        or protocol.get("vq_models_in_training_graph") is not False
    ):
        raise WatchContractError("frozen long-run protocol changed")
    initialization = protocol.get("initialization")
    if (
        not isinstance(initialization, dict)
        or initialization.get("source") != "released_all_speakers_v1"
        or initialization.get("speaker_scope") != "All-Speakers"
        or initialization.get("sha256")
        != "52999373a2c6bb6252c1153317116bb226d115c0a81d61362029ed3cc1d89603"
        or set(initialization.get("forbidden_sources", ()))
        != {"e30", "Speaker2"}
    ):
        raise WatchContractError("frozen initialization protocol changed")
    forward = protocol.get("forward_contract")
    if (
        not isinstance(forward, dict)
        or forward.get("forwards_per_optimizer_step") != 1
        or forward.get("audio_conditioned_main_forward") is not True
        or forward.get("masked_self_forward") is not False
        or forward.get("word_auxiliary_forward") is not False
    ):
        raise WatchContractError("frozen Base forward contract changed")
    schedule = protocol.get("schedule")
    trajectory_anchor = protocol.get("trajectory_anchor")
    if (
        not isinstance(schedule, dict)
        or schedule.get("sha256") != SCHEDULE_SHA256
        or not isinstance(trajectory_anchor, dict)
        or trajectory_anchor.get("sha256") != TRAJECTORY_ANCHOR_SHA256
    ):
        raise WatchContractError("frozen protocol schedule/anchor changed")

    long_contract = frozen.get("long_contract")
    if (
        not isinstance(long_contract, dict)
        or long_contract.get("format")
        != "semtalk_show_base_long_contract_receipts_v1"
    ):
        raise WatchContractError("frozen long-contract receipt changed")
    long_schedule = long_contract.get("schedule")
    long_anchor = long_contract.get("trajectory_anchor")
    if (
        not isinstance(long_schedule, dict)
        or long_schedule.get("sha256") != SCHEDULE_SHA256
        or not isinstance(long_anchor, dict)
        or long_anchor.get("sha256") != TRAJECTORY_ANCHOR_SHA256
    ):
        raise WatchContractError("frozen long-contract schedule/anchor mismatch")

    return (
        {
            "path": str(path),
            "sha256": claimed_sha,
            "receipt_payload_sha256": frozen_payload_sha,
        },
        {
            "format": protocol["format"],
            "payload_sha256": canonical_json_sha256(protocol),
            "trainer_entrypoint_sha256": entrypoint_sha,
            "source_commit": source["commit"],
            "source_tree": source["tree"],
        },
    )


def validate_manifest_snapshot(
    *,
    train_root: Path,
    epoch: int,
    value: object,
    ready_checkpoint: Mapping[str, Any],
    frozen_receipt_sha: str,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    reference = require_exact_keys(
        value, CANDIDATE_MANIFEST_KEYS, f"candidate e{epoch} manifest reference"
    )
    require_strict_bool(
        reference["immutable_snapshot"],
        True,
        f"candidate e{epoch} immutable manifest marker",
    )
    expected_snapshot = (
        train_root
        / "candidate_manifest_snapshots"
        / f"epoch-{epoch:04d}.json"
    )
    snapshot_path = regular_file(
        reference["path"], f"candidate e{epoch} manifest snapshot"
    )
    if snapshot_path != expected_snapshot:
        raise WatchContractError(
            f"candidate e{epoch} manifest snapshot path changed"
        )
    snapshot_sha = require_sha256(
        reference["sha256_at_ready"],
        f"candidate e{epoch} manifest snapshot SHA",
    )
    if sha256_file(snapshot_path) != snapshot_sha:
        raise WatchContractError(
            f"candidate e{epoch} manifest snapshot changed"
        )
    live_path = regular_file(
        reference["live_path"], f"candidate e{epoch} live manifest"
    )
    if live_path != train_root / "candidate_manifest.json":
        raise WatchContractError(
            f"candidate e{epoch} live manifest path changed"
        )

    snapshot = strict_json(snapshot_path)
    require_exact_keys(
        snapshot, MANIFEST_KEYS, f"candidate e{epoch} manifest snapshot"
    )
    entries = snapshot["entries"]
    prefix_length = CANDIDATE_EPOCHS.index(epoch) + 1
    expected_epochs = list(CANDIDATE_EPOCHS[:prefix_length])
    if (
        snapshot["format"] != MANIFEST_FORMAT
        or snapshot["status"] != "running"
        or snapshot["candidate_epochs"] != list(CANDIDATE_EPOCHS)
        or snapshot["frozen_receipt_sha256"] != frozen_receipt_sha
        or snapshot["schedule_sha256"] != SCHEDULE_SHA256
        or snapshot["trajectory_anchor_sha256"]
        != TRAJECTORY_ANCHOR_SHA256
        or not isinstance(entries, list)
        or len(entries) != prefix_length
    ):
        raise WatchContractError(
            f"candidate e{epoch} immutable manifest contract changed"
        )
    entries_sha = require_sha256(
        snapshot["entries_sha256"],
        f"candidate e{epoch} manifest entries SHA",
    )
    reference_entries_sha = require_sha256(
        reference["entries_sha256_at_ready"],
        f"candidate e{epoch} ready manifest entries SHA",
    )
    if (
        canonical_json_sha256(entries) != entries_sha
        or reference_entries_sha != entries_sha
    ):
        raise WatchContractError(
            f"candidate e{epoch} manifest entries changed"
        )

    throughput = snapshot["throughput_gate"]
    if (
        not isinstance(throughput, dict)
        or set(throughput)
        != {
            "path",
            "sha256",
            "samples_per_second",
            "seconds_per_update",
        }
        or not isinstance(throughput["path"], str)
    ):
        raise WatchContractError(
            f"candidate e{epoch} throughput-gate reference changed"
        )
    reject_forbidden(throughput["path"], "throughput-gate path")
    require_sha256(
        throughput["sha256"], f"candidate e{epoch} throughput-gate SHA"
    )
    for field in ("samples_per_second", "seconds_per_update"):
        number = throughput[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) <= 0
        ):
            raise WatchContractError(
                f"candidate e{epoch} invalid throughput {field}"
            )

    observed_epochs: list[int] = []
    for index, entry_value in enumerate(entries):
        entry_epoch = expected_epochs[index]
        entry = require_exact_keys(
            entry_value,
            MANIFEST_ENTRY_KEYS,
            f"candidate e{epoch} manifest entry {index}",
        )
        relative_checkpoint = (
            f"candidates/base_official_adapt_epoch_{entry_epoch:02d}.bin"
        )
        require_relative_file(
            entry["checkpoint"],
            relative_checkpoint,
            f"candidate e{entry_epoch} manifest checkpoint",
        )
        entry_sha = require_sha256(
            entry["checkpoint_sha256"],
            f"candidate e{entry_epoch} manifest checkpoint SHA",
        )
        entry_semantic_sha = require_sha256(
            entry["model_state_semantic_sha256"],
            f"candidate e{entry_epoch} manifest semantic SHA",
        )
        expected_anchor = anchor["entries"].get(str(entry_epoch))
        expected_anchor_match = True if expected_anchor is not None else None
        if (
            entry["epoch"] != entry_epoch
            or entry["optimizer_updates"] != entry_epoch * 248
            or type(entry["checkpoint_bytes"]) is not int
            or entry["checkpoint_bytes"] <= 0
            or entry["checkpoint_container_schema"]
            != ["audit", "model_state"]
            or entry["model_state_tensors"] != 1_790
            or entry["model_state_schema_sha256"]
            != MODEL_STATE_SCHEMA_SHA256
            or entry["all_model_state_tensors_finite"] is not True
            or entry["frozen_receipt_sha256"] != frozen_receipt_sha
            or entry["trajectory_anchor_match"] is not expected_anchor_match
        ):
            raise WatchContractError(
                f"candidate e{entry_epoch} manifest entry contract changed"
            )
        if expected_anchor is not None and (
            entry_sha != expected_anchor["checkpoint_sha256"]
            or entry_semantic_sha
            != expected_anchor["model_state_semantic_sha256"]
        ):
            raise WatchContractError(
                f"candidate e{entry_epoch} manifest violates anchor"
            )
        observed_epochs.append(entry_epoch)
    if observed_epochs != expected_epochs:
        raise WatchContractError(
            f"candidate e{epoch} manifest prefix is not exact"
        )

    current = entries[-1]
    if (
        current["checkpoint"]
        != ready_checkpoint["relative_path"]
        or current["checkpoint_sha256"] != ready_checkpoint["sha256"]
        or current["checkpoint_bytes"] != ready_checkpoint["bytes"]
        or current["model_state_tensors"]
        != ready_checkpoint["model_state_tensors"]
        or current["model_state_schema_sha256"]
        != ready_checkpoint["model_state_schema_sha256"]
        or current["model_state_semantic_sha256"]
        != ready_checkpoint["model_state_semantic_sha256"]
    ):
        raise WatchContractError(
            f"candidate e{epoch} ready/checkpoint manifest mismatch"
        )
    return {
        "path": str(snapshot_path),
        "sha256": snapshot_sha,
        "entries_sha256": entries_sha,
        "entry_count": len(entries),
        "live_path": str(live_path),
    }


def validate_ready(
    *,
    train_root: Path,
    epoch: int,
    schedule_path: Path,
    anchor_path: Path,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    train_root = directory(train_root, "long training run root")
    receipt_path = regular_file(
        train_root / "candidate_receipts" / f"epoch-{epoch:04d}.json",
        f"candidate e{epoch} ready receipt",
    )
    value = strict_json(receipt_path)
    require_exact_keys(value, READY_KEYS, f"candidate e{epoch} ready receipt")
    published = value.get("published_unix")
    expected_anchor_match = True if epoch in ANCHOR_EPOCHS else None
    if (
        value.get("format") != READY_FORMAT
        or value.get("status") != "ready"
        or value.get("selection_eligible") is not False
        or value.get("test_visible") is not False
        or value.get("epoch") != epoch
        or value.get("optimizer_updates") != epoch * 248
        or value.get("schedule_sha256") != SCHEDULE_SHA256
        or value.get("trajectory_anchor_sha256")
        != TRAJECTORY_ANCHOR_SHA256
        or value.get("trajectory_anchor_match")
        is not expected_anchor_match
        or isinstance(published, bool)
        or not isinstance(published, (int, float))
        or not math.isfinite(float(published))
        or float(published) <= 0
        or payload_sha256(value) != value.get("receipt_payload_sha256")
    ):
        raise WatchContractError(f"candidate e{epoch} ready receipt is invalid")

    frozen_receipt_sha = require_sha256(
        value.get("frozen_receipt_sha256"),
        f"candidate e{epoch} frozen-receipt SHA",
    )
    checkpoint = require_exact_keys(
        value.get("candidate_checkpoint"),
        CANDIDATE_CHECKPOINT_KEYS,
        f"candidate e{epoch} checkpoint reference",
    )
    relative_checkpoint = (
        f"candidates/base_official_adapt_epoch_{epoch:02d}.bin"
    )
    require_relative_file(
        checkpoint.get("relative_path"),
        relative_checkpoint,
        f"candidate e{epoch} checkpoint relative path",
    )
    checkpoint_path = regular_file(
        checkpoint.get("path"), f"candidate e{epoch} checkpoint"
    )
    candidates_root = directory(train_root / "candidates", "candidate directory")
    expected_checkpoint_path = train_root / relative_checkpoint
    if (
        checkpoint_path.parent != candidates_root
        or checkpoint_path != expected_checkpoint_path
        or checkpoint.get("model_state_tensors") != 1_790
        or checkpoint.get("model_state_schema_sha256")
        != MODEL_STATE_SCHEMA_SHA256
        or type(checkpoint.get("bytes")) is not int
        or checkpoint["bytes"] < 1
        or checkpoint_path.stat().st_size != checkpoint["bytes"]
    ):
        raise WatchContractError(f"candidate e{epoch} checkpoint contract")
    checkpoint_sha = require_sha256(
        checkpoint.get("sha256"), f"candidate e{epoch} checkpoint SHA"
    )
    semantic_sha = require_sha256(
        checkpoint.get("model_state_semantic_sha256"),
        f"candidate e{epoch} semantic SHA",
    )
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise WatchContractError(f"candidate e{epoch} checkpoint changed")

    frozen_artifact, frozen_protocol = validate_frozen_inputs(
        train_root=train_root,
        value=value.get("frozen_inputs"),
        expected_receipt_sha=frozen_receipt_sha,
    )
    ready_protocol = require_exact_keys(
        value.get("protocol"),
        PROTOCOL_KEYS,
        f"candidate e{epoch} protocol reference",
    )
    if (
        ready_protocol["format"] != PROTOCOL_FORMAT
        or require_sha256(
            ready_protocol["payload_sha256"],
            f"candidate e{epoch} protocol payload SHA",
        )
        != frozen_protocol["payload_sha256"]
    ):
        raise WatchContractError(
            f"candidate e{epoch} protocol payload mismatch"
        )

    anchored = str(epoch) in anchor["entries"]
    if anchored:
        expected = anchor["entries"][str(epoch)]
        if (
            checkpoint_sha != expected["checkpoint_sha256"]
            or semantic_sha != expected["model_state_semantic_sha256"]
        ):
            raise WatchContractError(
                f"candidate e{epoch} violates the frozen trajectory anchor"
            )
    manifest_artifact = validate_manifest_snapshot(
        train_root=train_root,
        epoch=epoch,
        value=value.get("candidate_manifest"),
        ready_checkpoint=checkpoint,
        frozen_receipt_sha=frozen_receipt_sha,
        anchor=anchor,
    )
    return {
        "epoch": epoch,
        "optimizer_updates": epoch * 248,
        "ready_receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "receipt_payload_sha256": value["receipt_payload_sha256"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "relative_path": relative_checkpoint,
            "sha256": checkpoint_sha,
            "bytes": checkpoint["bytes"],
            "model_state_tensors": checkpoint["model_state_tensors"],
            "model_state_schema_sha256": checkpoint[
                "model_state_schema_sha256"
            ],
            "model_state_semantic_sha256": semantic_sha,
        },
        "candidate_manifest": manifest_artifact,
        "frozen_inputs": frozen_artifact,
        "protocol": {
            "format": PROTOCOL_FORMAT,
            "payload_sha256": frozen_protocol["payload_sha256"],
        },
        "source": {
            "commit": frozen_protocol["source_commit"],
            "tree": frozen_protocol["source_tree"],
            "trainer_entrypoint_sha256": frozen_protocol[
                "trainer_entrypoint_sha256"
            ],
            "clean": True,
            "branch": None,
        },
        "frozen_receipt_sha256": frozen_receipt_sha,
        "schedule_sha256": SCHEDULE_SHA256,
        "trajectory_anchor_sha256": TRAJECTORY_ANCHOR_SHA256,
        "trajectory_anchor_exact": anchored,
        "published_unix": float(published),
    }


def validate_wave_consistency(
    candidates: Sequence[Mapping[str, Any]]
) -> None:
    if not candidates:
        raise WatchContractError("validation wave has no candidates")
    first = candidates[0]
    immutable_fields = (
        "frozen_receipt_sha256",
        "frozen_inputs",
        "protocol",
        "source",
        "schedule_sha256",
        "trajectory_anchor_sha256",
    )
    for candidate in candidates[1:]:
        for field in immutable_fields:
            if candidate[field] != first[field]:
                raise WatchContractError(
                    f"validation wave candidates disagree on {field}"
                )

    previous_entries: list[Any] = []
    previous_published = -math.inf
    for candidate in candidates:
        published = float(candidate["published_unix"])
        if published < previous_published:
            raise WatchContractError(
                "candidate ready publication times are not monotonic"
            )
        previous_published = published
        snapshot = strict_json(Path(candidate["candidate_manifest"]["path"]))
        entries = snapshot["entries"]
        if entries[: len(previous_entries)] != previous_entries:
            raise WatchContractError(
                "immutable candidate-manifest snapshots are not monotonic"
            )
        previous_entries = entries


def wait_wave(args: argparse.Namespace) -> dict[str, Any]:
    schedule_path, _schedule = validate_schedule(
        args.schedule, args.expected_schedule_sha256
    )
    anchor_path, anchor = validate_anchor(
        args.trajectory_anchor, args.expected_trajectory_anchor_sha256
    )
    train_root = directory(args.train_root, "long training run root")
    if not 1 <= args.wave_index <= len(VALIDATION_WAVES):
        raise WatchContractError("wave-index is outside the frozen schedule")
    wave = VALIDATION_WAVES[args.wave_index - 1]
    deadline = time.monotonic() + args.timeout_seconds
    candidates: list[dict[str, Any]] = []
    while True:
        candidates = []
        missing = []
        for epoch in wave:
            path = (
                train_root
                / "candidate_receipts"
                / f"epoch-{epoch:04d}.json"
            )
            if not path.is_file() or path.is_symlink():
                missing.append(str(path))
                continue
            candidates.append(
                validate_ready(
                    train_root=train_root,
                    epoch=epoch,
                    schedule_path=schedule_path,
                    anchor_path=anchor_path,
                    anchor=anchor,
                )
            )
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for frozen validation wave "
                f"{args.wave_index}: {missing}"
            )
        time.sleep(args.poll_seconds)
    validate_wave_consistency(candidates)
    body = {
        "format": WAVE_FORMAT,
        "status": "ready",
        "split": "val",
        "test_visible": False,
        "selection_eligible": False,
        "wave_index": args.wave_index,
        "epochs": list(wave),
        "schedule": {"path": str(schedule_path), "sha256": SCHEDULE_SHA256},
        "trajectory_anchor": {
            "path": str(anchor_path),
            "sha256": TRAJECTORY_ANCHOR_SHA256,
        },
        "train_root": str(train_root),
        "candidates": candidates,
    }
    receipt = {**body, "receipt_payload_sha256": payload_sha256(body)}
    output = write_new_json(args.output, receipt)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "wave_index": args.wave_index,
        "epochs": list(wave),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--trajectory-anchor", type=Path, required=True)
    parser.add_argument("--expected-trajectory-anchor-sha256", required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--wave-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=24 * 60 * 60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not math.isfinite(args.poll_seconds)
        or args.poll_seconds <= 0
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        raise WatchContractError("poll/timeout seconds must be positive")
    result = wait_wave(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
