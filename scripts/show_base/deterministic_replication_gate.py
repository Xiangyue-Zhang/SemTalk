#!/usr/bin/env python3
"""Fail-closed proof for deterministic SemTalk logical sample replication.

SemTalk Base inference has no sampling argument.  That fact is useful only
after it has been proven for the exact source, checkpoint bundle, runtime and
frozen SHOW validation inputs used by a formal run.  This module merges two
*independent-process* validation runs (seeds 0 and 15), verifies that:

* four frozen clips cover Oliver/Chemistry/Seth/Conan and multiple lengths;
* every forward leaves Python/NumPy/Torch CPU/Torch CUDA RNG states unchanged;
* the two runs produce byte-identical NPZ files and array-identical fields;
* source call-closure, eval/argmax/deterministic flags, Base checkpoint and the
  five representation checkpoints are exactly identical.

Only a passing receipt authorizes
``deterministic_replication_of_single_prediction_v1``.  There is deliberately
no "best effort" result: any mismatch raises :class:`ReplicationGateError`, so
the caller must execute 16 real inferences instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True

FORMAT = "semtalk_show_deterministic_replication_gate_v1"
SEED_RUN_FORMAT = "semtalk_show_deterministic_seed_run_v1"
DISTRIBUTION_FORMAT = "semtalk_show_deterministic_distribution_receipt_v1"
PROTOCOL = "deterministic_replication_of_single_prediction_v1"
REPLICATION_ALGORITHM = "logical_reference_v1"
EXPECTED_SEEDS = (0, 15)
EXPECTED_SPEAKERS = ("oliver", "chemistry", "seth", "conan")
EXPECTED_NPZ_FIELDS = (
    "betas",
    "poses",
    "expressions",
    "trans",
    "model",
    "gender",
    "mocap_frame_rate",
)
MODEL_STAGES = ("base", "face", "hands", "upper", "lower", "global")
LOGICAL_SLOTS = tuple(range(16))
RELEASED2_SLOTS = (0, 1)
PAPER16_SLOTS = LOGICAL_SLOTS
FACE_SLOT = 0
DIFFSHEG_SLOT = 0
RNG_FIELDS = ("python", "numpy", "torch_cpu", "torch_cuda")
FORBIDDEN_RANDOM_SYMBOLS = (
    "bernoulli",
    "dropout",
    "multinomial",
    "normal",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "random",
    "sample",
)


class ReplicationGateError(RuntimeError):
    """Raised when deterministic replication cannot be proven."""


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplicationGateError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplicationGateError(f"{label} must be a Git object ID")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplicationGateError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _exact_mapping(
    value: Any,
    keys: Iterable[str],
    label: str,
) -> dict[str, Any]:
    expected = set(keys)
    if not isinstance(value, dict) or set(value) != expected:
        raise ReplicationGateError(
            f"{label} schema mismatch: "
            f"{sorted(value) if isinstance(value, dict) else type(value)} "
            f"!= {sorted(expected)}"
        )
    return value


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReplicationGateError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ReplicationGateError):
            raise
        raise ReplicationGateError(f"{label} is not strict JSON: {error}") from error


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ReplicationGateError(f"{label} must be absolute")
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ReplicationGateError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ReplicationGateError(
            f"{label} must be a regular non-symlink file"
        )
    return path.resolve(strict=True)


def _artifact(
    value: Any,
    label: str,
    *,
    payload_sha: bool = False,
) -> tuple[dict[str, Any], Path]:
    keys = {"path", "sha256", "bytes"}
    if payload_sha:
        keys.add("receipt_payload_sha256")
    artifact = _exact_mapping(value, keys, label)
    path = _regular_file(Path(artifact["path"]), label)
    digest = _require_sha256(artifact["sha256"], f"{label} SHA-256")
    size = _require_int(artifact["bytes"], f"{label} bytes", minimum=1)
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ReplicationGateError(f"{label} artifact changed")
    if payload_sha:
        _require_sha256(
            artifact["receipt_payload_sha256"],
            f"{label} payload SHA-256",
        )
    return dict(artifact), path


def _payload_hash(value: Mapping[str, Any], label: str) -> str:
    claimed = _require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256", None)
    observed = canonical_json_sha256(unsigned)
    if observed != claimed:
        raise ReplicationGateError(
            f"{label} payload SHA-256 mismatch: {observed} != {claimed}"
        )
    return claimed


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    header = json.dumps(
        {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def _npz_field_receipt(path: Path, label: str) -> dict[str, str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(EXPECTED_NPZ_FIELDS):
                raise ReplicationGateError(
                    f"{label} NPZ fields differ from canonical SemTalk output"
                )
            result = {
                field: _array_sha256(np.asarray(archive[field]))
                for field in EXPECTED_NPZ_FIELDS
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ReplicationGateError):
            raise
        raise ReplicationGateError(f"cannot load {label} NPZ: {error}") from error
    return result


def _validate_rng_state(value: Any, label: str) -> dict[str, Any]:
    state = _exact_mapping(
        value,
        {*RNG_FIELDS, "rng_state_sha256"},
        label,
    )
    for field in RNG_FIELDS:
        _require_sha256(state[field], f"{label}.{field}")
    claimed = _require_sha256(
        state["rng_state_sha256"],
        f"{label}.rng_state_sha256",
    )
    observed = canonical_json_sha256(
        {field: state[field] for field in RNG_FIELDS}
    )
    if observed != claimed:
        raise ReplicationGateError(f"{label} aggregate SHA-256 mismatch")
    return dict(state)


def _validate_source_closure(value: Any) -> dict[str, Any]:
    closure = _exact_mapping(
        value,
        {
            "origin",
            "commit",
            "tree",
            "clean",
            "detached",
            "local_branches_at_commit",
            "entrypoint",
            "inference_helper",
            "callables",
            "static_randomness_scan",
        },
        "source closure",
    )
    if (
        closure["origin"]
        != "git@github.com:Xiangyue-Zhang/SemTalk.git"
        or closure["clean"] is not True
        or closure["detached"] is not True
        or closure["local_branches_at_commit"] != []
    ):
        raise ReplicationGateError(
            "source closure must be a clean detached SemTalk checkout "
            "without local heads at the formal commit"
        )
    _require_git_oid(closure["commit"], "source closure commit")
    _require_git_oid(closure["tree"], "source closure tree")
    for key in ("entrypoint", "inference_helper"):
        artifact = _exact_mapping(
            closure[key],
            {"relative_path", "sha256", "git_blob_sha1"},
            f"source closure {key}",
        )
        if (
            not isinstance(artifact["relative_path"], str)
            or not artifact["relative_path"]
            or Path(artifact["relative_path"]).is_absolute()
            or ".." in Path(artifact["relative_path"]).parts
        ):
            raise ReplicationGateError(
                f"source closure {key} relative path is invalid"
            )
        _require_sha256(
            artifact["sha256"],
            f"source closure {key} SHA-256",
        )
        if (
            not isinstance(artifact["git_blob_sha1"], str)
            or len(artifact["git_blob_sha1"]) != 40
        ):
            raise ReplicationGateError(
                f"source closure {key} Git blob SHA-1 is invalid"
            )
    callables = closure["callables"]
    if (
        not isinstance(callables, list)
        or not callables
        or any(
            not isinstance(item, dict)
            or set(item) != {"qualified_name", "source_sha256"}
            or not isinstance(item["qualified_name"], str)
            or not item["qualified_name"]
            for item in callables
        )
    ):
        raise ReplicationGateError("source call-closure is incomplete")
    for item in callables:
        _require_sha256(
            item["source_sha256"],
            f"{item['qualified_name']} source SHA-256",
        )
    names = [item["qualified_name"] for item in callables]
    if len(names) != len(set(names)):
        raise ReplicationGateError("source call-closure has duplicate names")
    scan = _exact_mapping(
        closure["static_randomness_scan"],
        {
            "algorithm",
            "forbidden_symbols",
            "matches",
            "call_closure_sha256",
        },
        "static randomness scan",
    )
    if (
        scan["algorithm"] != "python_ast_call_symbol_scan_v1"
        or scan["forbidden_symbols"] != list(FORBIDDEN_RANDOM_SYMBOLS)
        or scan["matches"] != []
        or scan["call_closure_sha256"]
        != canonical_json_sha256(callables)
    ):
        raise ReplicationGateError(
            "static source closure contains or fails to exclude random ops"
        )
    return dict(closure)


def _validate_model_bundle(value: Any) -> dict[str, Any]:
    bundle = _exact_mapping(
        value,
        {
            "checkpoints",
            "all_state_tensors_finite",
            "models_eval",
            "requires_grad_false",
            "bundle_sha256",
        },
        "model bundle",
    )
    checkpoints = bundle["checkpoints"]
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(MODEL_STAGES):
        raise ReplicationGateError(
            "model bundle must contain Base plus five prerequisites"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for stage in MODEL_STAGES:
        artifact, _ = _artifact(
            checkpoints[stage],
            f"{stage} checkpoint",
        )
        normalized[stage] = artifact
    if (
        bundle["all_state_tensors_finite"] is not True
        or bundle["models_eval"] is not True
        or bundle["requires_grad_false"] is not True
        or _require_sha256(
            bundle["bundle_sha256"],
            "model bundle SHA-256",
        )
        != canonical_json_sha256(normalized)
    ):
        raise ReplicationGateError(
            "model bundle lacks finite/frozen/eval closure"
        )
    return {
        **dict(bundle),
        "checkpoints": normalized,
    }


def _validate_subset_manifest(
    artifact_value: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact, path = _artifact(
        artifact_value,
        "replication-gate subset manifest",
    )
    payload = _strict_json_bytes(path.read_bytes(), str(path))
    if not isinstance(payload, list) or len(payload) != len(EXPECTED_SPEAKERS):
        raise ReplicationGateError(
            "gate subset must contain exactly four validation clips"
        )
    expected_keys = {
        "gate_position",
        "canonical_position",
        "global_index",
        "source_clip_id",
        "canonical_clip_id",
        "speaker",
        "frames",
        "canonical_row_sha256",
        "audio_row_sha256",
    }
    rows: list[dict[str, Any]] = []
    for expected_position, row in enumerate(payload):
        item = _exact_mapping(
            row,
            expected_keys,
            f"gate subset row {expected_position}",
        )
        if (
            _require_int(
                item["gate_position"],
                "gate position",
            )
            != expected_position
            or item["speaker"] != EXPECTED_SPEAKERS[expected_position]
            or _require_int(
                item["canonical_position"],
                "canonical position",
            )
            < 0
            or _require_int(item["global_index"], "global index") < 0
            or _require_int(item["frames"], "gate frames", minimum=4) < 4
            or not isinstance(item["source_clip_id"], str)
            or not item["source_clip_id"]
            or not isinstance(item["canonical_clip_id"], str)
            or not item["canonical_clip_id"]
        ):
            raise ReplicationGateError(
                f"gate subset row {expected_position} is invalid"
            )
        _require_sha256(
            item["canonical_row_sha256"],
            "canonical row SHA-256",
        )
        _require_sha256(item["audio_row_sha256"], "audio row SHA-256")
        rows.append(dict(item))
    if len({row["frames"] for row in rows}) < 2:
        raise ReplicationGateError(
            "gate subset must cover multiple clip lengths"
        )
    return artifact, rows


def _validate_seed_run(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _regular_file(path, "seed-run receipt")
    expected = _require_sha256(expected_sha256, "seed-run receipt SHA-256")
    observed = sha256_file(resolved)
    if observed != expected:
        raise ReplicationGateError(
            f"seed-run receipt SHA-256 mismatch: {observed} != {expected}"
        )
    payload = _strict_json_bytes(resolved.read_bytes(), str(resolved))
    run = _exact_mapping(
        payload,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "seed",
            "run_nonce",
            "process_identity",
            "independent_process",
            "subset_manifest",
            "source_closure",
            "model_bundle",
            "runtime",
            "clips",
            "receipt_payload_sha256",
        },
        "seed-run receipt",
    )
    claimed_payload = _payload_hash(run, "seed-run receipt")
    if (
        run["format"] != SEED_RUN_FORMAT
        or run["status"] != "complete"
        or run["split"] != "val"
        or run["test_visible"] is not False
        or run["seed"] not in EXPECTED_SEEDS
        or run["independent_process"] is not True
        or not isinstance(run["run_nonce"], str)
        or len(run["run_nonce"]) < 16
        or not isinstance(run["process_identity"], str)
        or not run["process_identity"]
    ):
        raise ReplicationGateError(
            "seed-run is not an independent frozen validation run"
        )
    subset_artifact, subset = _validate_subset_manifest(
        run["subset_manifest"]
    )
    source = _validate_source_closure(run["source_closure"])
    model_bundle = _validate_model_bundle(run["model_bundle"])
    runtime = _exact_mapping(
        run["runtime"],
        {
            "torch_inference_mode",
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "models_eval",
            "requires_grad_false",
            "discrete_decoding",
            "device",
        },
        "seed-run runtime",
    )
    if (
        runtime["torch_inference_mode"] is not True
        or runtime["deterministic_algorithms"] is not True
        or runtime["cudnn_deterministic"] is not True
        or runtime["cudnn_benchmark"] is not False
        or runtime["models_eval"] is not True
        or runtime["requires_grad_false"] is not True
        or runtime["discrete_decoding"] != "logits.argmax(dim=2)"
        or not isinstance(runtime["device"], str)
        or not runtime["device"].startswith("cuda")
    ):
        raise ReplicationGateError(
            "seed-run runtime is not deterministic eval/argmax inference"
        )
    clips = run["clips"]
    if not isinstance(clips, list) or len(clips) != len(subset):
        raise ReplicationGateError("seed-run clip coverage mismatch")
    normalized_clips: list[dict[str, Any]] = []
    for expected_row, value in zip(subset, clips):
        clip = _exact_mapping(
            value,
            {
                "gate_position",
                "canonical_clip_id",
                "frames",
                "input_binding",
                "prediction",
                "rng",
            },
            f"seed {run['seed']} clip",
        )
        if (
            clip["gate_position"] != expected_row["gate_position"]
            or clip["canonical_clip_id"]
            != expected_row["canonical_clip_id"]
            or clip["frames"] != expected_row["frames"]
        ):
            raise ReplicationGateError(
                "seed-run clip order/input differs from subset manifest"
            )
        input_binding = _exact_mapping(
            clip["input_binding"],
            {"canonical_row_sha256", "audio_row_sha256"},
            "seed-run clip input binding",
        )
        if input_binding != {
            "canonical_row_sha256": expected_row["canonical_row_sha256"],
            "audio_row_sha256": expected_row["audio_row_sha256"],
        }:
            raise ReplicationGateError("seed-run clip input binding changed")
        prediction, prediction_path = _artifact(
            clip["prediction"],
            "gate prediction",
        )
        field_receipts = prediction.get("field_sha256")
        # ``_artifact`` intentionally validates only the file envelope.  The
        # field receipt is carried beside it in seed-run schema below.
        if field_receipts is not None:  # pragma: no cover - schema rejects it
            raise ReplicationGateError(
                "prediction field digests must not be embedded in artifact"
            )
        rng = _exact_mapping(
            clip["rng"],
            {"before", "after", "seed_consumed"},
            "seed-run clip RNG proof",
        )
        before = _validate_rng_state(rng["before"], "RNG before")
        after = _validate_rng_state(rng["after"], "RNG after")
        if rng["seed_consumed"] is not False or before != after:
            raise ReplicationGateError(
                "SemTalk forward consumed RNG state; logical replication "
                "is forbidden"
            )
        field_sha = _npz_field_receipt(
            prediction_path,
            f"seed {run['seed']} {clip['canonical_clip_id']}",
        )
        normalized_clips.append(
            {
                **dict(clip),
                "prediction": prediction,
                "prediction_field_sha256": field_sha,
                "rng": {
                    "before": before,
                    "after": after,
                    "seed_consumed": False,
                },
            }
        )
    return (
        {
            "path": str(resolved),
            "sha256": observed,
            "bytes": resolved.stat().st_size,
            "receipt_payload_sha256": claimed_payload,
        },
        {
            **dict(run),
            "subset_manifest": subset_artifact,
            "source_closure": source,
            "model_bundle": model_bundle,
            "runtime": dict(runtime),
            "clips": normalized_clips,
        },
    )


def _same_npz_arrays(first: Path, second: Path) -> bool:
    with (
        np.load(first, allow_pickle=False) as first_archive,
        np.load(second, allow_pickle=False) as second_archive,
    ):
        if (
            set(first_archive.files) != set(EXPECTED_NPZ_FIELDS)
            or set(second_archive.files) != set(EXPECTED_NPZ_FIELDS)
        ):
            return False
        for field in EXPECTED_NPZ_FIELDS:
            first_value = np.asarray(first_archive[field])
            second_value = np.asarray(second_archive[field])
            if (
                first_value.dtype != second_value.dtype
                or first_value.shape != second_value.shape
                or not np.array_equal(first_value, second_value)
            ):
                return False
    return True


def build_gate(
    *,
    seed_run_paths: Sequence[Path],
    expected_seed_run_sha256: Sequence[str],
    scope: str,
) -> dict[str, Any]:
    if scope not in {"validation_candidate_family", "final_winner"}:
        raise ReplicationGateError("unsupported replication-gate scope")
    if len(seed_run_paths) != 2 or len(expected_seed_run_sha256) != 2:
        raise ReplicationGateError(
            "exactly two independently rooted seed runs are required"
        )
    loaded = [
        _validate_seed_run(path, digest)
        for path, digest in zip(
            seed_run_paths,
            expected_seed_run_sha256,
        )
    ]
    loaded.sort(key=lambda item: item[1]["seed"])
    artifacts = [item[0] for item in loaded]
    runs = [item[1] for item in loaded]
    if [run["seed"] for run in runs] != list(EXPECTED_SEEDS):
        raise ReplicationGateError("gate requires exactly seeds 0 and 15")
    if (
        runs[0]["run_nonce"] == runs[1]["run_nonce"]
        or runs[0]["process_identity"] == runs[1]["process_identity"]
    ):
        raise ReplicationGateError(
            "seed 0 and seed 15 must run in independent processes"
        )
    for field in (
        "subset_manifest",
        "source_closure",
        "model_bundle",
        "runtime",
    ):
        if runs[0][field] != runs[1][field]:
            raise ReplicationGateError(
                f"seed runs differ in frozen {field}"
            )
    compared_clips: list[dict[str, Any]] = []
    for first, second in zip(runs[0]["clips"], runs[1]["clips"]):
        comparable_keys = {
            "gate_position",
            "canonical_clip_id",
            "frames",
            "input_binding",
        }
        if any(first[key] != second[key] for key in comparable_keys):
            raise ReplicationGateError("seed-run clip identity changed")
        first_prediction, first_path = _artifact(
            first["prediction"],
            "seed 0 prediction",
        )
        second_prediction, second_path = _artifact(
            second["prediction"],
            "seed 15 prediction",
        )
        if (
            first_prediction["sha256"] != second_prediction["sha256"]
            or first_prediction["bytes"] != second_prediction["bytes"]
            or first_path.read_bytes() != second_path.read_bytes()
        ):
            raise ReplicationGateError(
                f"{first['canonical_clip_id']}: seed outputs are not "
                "byte-identical"
            )
        if (
            first["prediction_field_sha256"]
            != second["prediction_field_sha256"]
            or not _same_npz_arrays(first_path, second_path)
        ):
            raise ReplicationGateError(
                f"{first['canonical_clip_id']}: seed outputs are not "
                "array-identical"
            )
        compared_clips.append(
            {
                "gate_position": first["gate_position"],
                "canonical_clip_id": first["canonical_clip_id"],
                "frames": first["frames"],
                "seed0_prediction_sha256": first_prediction["sha256"],
                "seed15_prediction_sha256": second_prediction["sha256"],
                "field_sha256": first["prediction_field_sha256"],
                "byte_exact": True,
                "array_exact": True,
                "seed0_rng_unchanged": True,
                "seed15_rng_unchanged": True,
            }
        )
    subset_path = Path(runs[0]["subset_manifest"]["path"])
    subset = _strict_json_bytes(subset_path.read_bytes(), str(subset_path))
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "pass",
        "split": "val",
        "test_visible": False,
        "scope": scope,
        "gate_protocol": PROTOCOL,
        "seeds": list(EXPECTED_SEEDS),
        "seed_runs": artifacts,
        "subset_manifest": runs[0]["subset_manifest"],
        "source_closure": runs[0]["source_closure"],
        "model_bundle": runs[0]["model_bundle"],
        "proof": {
            "independent_processes": True,
            "static_no_random_ops": True,
            "runtime_no_rng_consumption": True,
            "byte_exact": True,
            "array_exact": True,
            "speakers": [row["speaker"] for row in subset],
            "frame_lengths": [row["frames"] for row in subset],
            "multi_length": True,
            "clips": compared_clips,
        },
        "replication_authorization": {
            "protocol": PROTOCOL,
            "independent_samples": False,
            "deterministic_delta_distribution": True,
            "replication_algorithm": REPLICATION_ALGORITHM,
            "seed_consumed": False,
            "physical_samples_per_clip": 1,
            "logical_slots": list(LOGICAL_SLOTS),
            "released2_slots": list(RELEASED2_SLOTS),
            "paper16_slots": list(PAPER16_SLOTS),
            "face_slot": FACE_SLOT,
            "diffsheg_slot": DIFFSHEG_SLOT,
            "variation_exact_zero": True,
            "failure_fallback": "sixteen_independent_physical_inferences",
        },
    }
    result["receipt_payload_sha256"] = canonical_json_sha256(result)
    return result


def load_gate(
    path: Path,
    expected_sha256: str,
    *,
    expected_scope: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _regular_file(path, "replication gate")
    expected = _require_sha256(expected_sha256, "replication gate SHA-256")
    observed = sha256_file(resolved)
    if observed != expected:
        raise ReplicationGateError("replication gate file SHA-256 mismatch")
    payload = _strict_json_bytes(resolved.read_bytes(), str(resolved))
    gate = _exact_mapping(
        payload,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "scope",
            "gate_protocol",
            "seeds",
            "seed_runs",
            "subset_manifest",
            "source_closure",
            "model_bundle",
            "proof",
            "replication_authorization",
            "receipt_payload_sha256",
        },
        "replication gate",
    )
    claimed = _payload_hash(gate, "replication gate")
    if (
        gate["format"] != FORMAT
        or gate["status"] != "pass"
        or gate["split"] != "val"
        or gate["test_visible"] is not False
        or gate["gate_protocol"] != PROTOCOL
        or gate["seeds"] != list(EXPECTED_SEEDS)
        or (
            expected_scope is not None
            and gate["scope"] != expected_scope
        )
    ):
        raise ReplicationGateError("replication gate protocol mismatch")
    rebuilt = build_gate(
        seed_run_paths=[
            Path(item["path"]) for item in gate["seed_runs"]
        ],
        expected_seed_run_sha256=[
            item["sha256"] for item in gate["seed_runs"]
        ],
        scope=gate["scope"],
    )
    if rebuilt != gate:
        raise ReplicationGateError(
            "replication gate differs from fresh seed-run replay"
        )
    return (
        {
            "path": str(resolved),
            "sha256": observed,
            "bytes": resolved.stat().st_size,
            "receipt_payload_sha256": claimed,
        },
        gate,
    )


def prediction_artifact_manifest(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(records):
        row = _exact_mapping(
            dict(value),
            {
                "canonical_clip_id",
                "prediction_sha256",
                "prediction_bytes",
            },
            f"prediction artifact row {index}",
        )
        clip_id = row["canonical_clip_id"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise ReplicationGateError(
                "prediction artifact IDs must be unique/non-empty"
            )
        seen.add(clip_id)
        normalized.append(
            {
                "canonical_clip_id": clip_id,
                "prediction_sha256": _require_sha256(
                    row["prediction_sha256"],
                    f"{clip_id} prediction SHA-256",
                ),
                "prediction_bytes": _require_int(
                    row["prediction_bytes"],
                    f"{clip_id} prediction bytes",
                    minimum=1,
                ),
            }
        )
    if not normalized:
        raise ReplicationGateError("prediction artifact manifest is empty")
    return normalized, canonical_json_sha256(normalized)


def build_distribution_receipt(
    *,
    gate_path: Path,
    expected_gate_sha256: str,
    prediction_manifest_artifact: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    expected_scope: str,
) -> dict[str, Any]:
    gate_artifact, _gate = load_gate(
        gate_path,
        expected_gate_sha256,
        expected_scope=expected_scope,
    )
    manifest_artifact, _manifest_path = _artifact(
        prediction_manifest_artifact,
        "prediction manifest",
    )
    _normalized, prediction_digest = prediction_artifact_manifest(
        prediction_records
    )
    logical_bindings = [
        {
            "slot": slot,
            "prediction_artifact_manifest_sha256": prediction_digest,
        }
        for slot in LOGICAL_SLOTS
    ]
    result: dict[str, Any] = {
        "format": DISTRIBUTION_FORMAT,
        "protocol": PROTOCOL,
        "independent_samples": False,
        "deterministic_delta_distribution": True,
        "replication_algorithm": REPLICATION_ALGORITHM,
        "seed_consumed": False,
        "physical_samples_per_clip": 1,
        "logical_slots": list(LOGICAL_SLOTS),
        "released2_slots": list(RELEASED2_SLOTS),
        "paper16_slots": list(PAPER16_SLOTS),
        "face_slot": FACE_SLOT,
        "diffsheg_slot": DIFFSHEG_SLOT,
        "slot_artifact_policy": "same_prediction_sha256",
        "prediction_manifest": manifest_artifact,
        "prediction_artifact_manifest_sha256": prediction_digest,
        "logical_slot_bindings": logical_bindings,
        "validation_gate": gate_artifact,
        "variation_exact_zero": True,
    }
    result["receipt_payload_sha256"] = canonical_json_sha256(result)
    return result


def validate_distribution_receipt(
    value: Any,
    *,
    expected_gate_artifact: Mapping[str, Any],
    expected_prediction_manifest: Mapping[str, Any],
    expected_prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = _exact_mapping(
        value,
        {
            "format",
            "protocol",
            "independent_samples",
            "deterministic_delta_distribution",
            "replication_algorithm",
            "seed_consumed",
            "physical_samples_per_clip",
            "logical_slots",
            "released2_slots",
            "paper16_slots",
            "face_slot",
            "diffsheg_slot",
            "slot_artifact_policy",
            "prediction_manifest",
            "prediction_artifact_manifest_sha256",
            "logical_slot_bindings",
            "validation_gate",
            "variation_exact_zero",
            "receipt_payload_sha256",
        },
        "distribution receipt",
    )
    _payload_hash(receipt, "distribution receipt")
    _normalized, expected_prediction_digest = prediction_artifact_manifest(
        expected_prediction_records
    )
    expected_bindings = [
        {
            "slot": slot,
            "prediction_artifact_manifest_sha256": (
                expected_prediction_digest
            ),
        }
        for slot in LOGICAL_SLOTS
    ]
    exact = {
        "format": DISTRIBUTION_FORMAT,
        "protocol": PROTOCOL,
        "independent_samples": False,
        "deterministic_delta_distribution": True,
        "replication_algorithm": REPLICATION_ALGORITHM,
        "seed_consumed": False,
        "physical_samples_per_clip": 1,
        "logical_slots": list(LOGICAL_SLOTS),
        "released2_slots": list(RELEASED2_SLOTS),
        "paper16_slots": list(PAPER16_SLOTS),
        "face_slot": FACE_SLOT,
        "diffsheg_slot": DIFFSHEG_SLOT,
        "slot_artifact_policy": "same_prediction_sha256",
        "prediction_manifest": dict(expected_prediction_manifest),
        "prediction_artifact_manifest_sha256": (
            expected_prediction_digest
        ),
        "logical_slot_bindings": expected_bindings,
        "validation_gate": dict(expected_gate_artifact),
        "variation_exact_zero": True,
    }
    unsigned = dict(receipt)
    unsigned.pop("receipt_payload_sha256")
    if unsigned != exact:
        raise ReplicationGateError(
            "distribution receipt differs from exact deterministic delta "
            "contract"
        )
    return dict(receipt)


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("finalize", allow_abbrev=False)
    finalize.add_argument(
        "--seed-run-json",
        action="append",
        type=Path,
        required=True,
    )
    finalize.add_argument(
        "--expected-seed-run-sha256",
        action="append",
        required=True,
    )
    finalize.add_argument(
        "--scope",
        choices=("validation_candidate_family", "final_winner"),
        required=True,
    )
    finalize.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "finalize":  # pragma: no cover
        raise AssertionError(args.command)
    result = build_gate(
        seed_run_paths=args.seed_run_json,
        expected_seed_run_sha256=args.expected_seed_run_sha256,
        scope=args.scope,
    )
    _atomic_json_new(args.output_json, result)
    print(
        json.dumps(
            {
                "status": "pass",
                "scope": args.scope,
                "output": str(args.output_json.resolve()),
                "receipt_payload_sha256": result[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
