#!/usr/bin/env python3
"""Fresh validation for one published SemTalk SHOW Base test winner.

This module is intentionally self-contained and standard-library only.  It
does not trust caller-supplied checkpoint mappings.  The Base checkpoint is
recomputed from the frozen validation rows and the five representation
checkpoints are recovered from the separately pinned prerequisite selection.
Only then is the claim allowed to authorize one test evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any, Iterable, Mapping, Sequence


CLAIM_FORMAT = "semtalk_show_base_published_test_winner_claim_v1"
FRESH_CLAIM_FORMAT = "semtalk_show_base_fresh_val_published_test_winner_claim_v3"
AUTHORIZATION_FORMAT = (
    "semtalk_show_base_published_test_winner_authorization_v1"
)
WINNER_SELECTION_FORMAT = (
    "semtalk_show_base_talkshow_released2_fgd_selection_v1"
)
FRESH_WINNER_SELECTION_FORMAT = (
    "semtalk_show_base_fresh_val_released2_fgd_selection_v2"
)
PREREQUISITE_SELECTION_FORMAT = (
    "semtalk_show_prerequisite_val_selection_v2"
)
INITIAL_PREREQUISITE_SELECTION_FORMAT = (
    "semtalk_show_prerequisite_val_selection_v1"
)
CONTINUATION_DECISION_FORMAT = (
    "semtalk_show_prerequisite_continuation_decision_v2"
)
VAL_LINEAGE_FORMAT = "semtalk_show_base_val_inference_lineage_v2"
FORMAL_VAL_LINEAGE_FORMAT = (
    "semtalk_show_base_talkshow_val_inference_lineage_v2"
)
FRESH_VAL_TRANSACTION_FORMAT = (
    "semtalk_show_base_fresh_val_candidate_transaction_v1"
)
FRESH_VAL_FAILURE_FORMAT = (
    "semtalk_show_base_fresh_val_failure_manifest_v1"
)
FRESH_VAL_PREFLIGHT_FORMAT = (
    "semtalk_show_base_official_adapt_val_inference_preflight_v1"
)
FRESH_VAL_SHARD_FORMAT = (
    "semtalk_show_base_official_adapt_val_inference_shard_v1"
)
FRESH_VAL_ASSIGNMENT = "canonical_position_modulo_num_shards"
FRESH_VAL_GENERATOR_MODULE = "models.semtalk.semtalk_base"
FRESH_VAL_FORMAL_HOSTS = (
    "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-master-0",
    "iannnzhang-aws-28data2-m2d-iannnzhang-28data-2x8-worker-0",
)
METRIC_REPORT_FORMAT = "semtalk_show_talkshow_metrics_v1"
PRIMARY_SCREEN_FORMAT = "semtalk_show_released2_primary_screen_v1"
WINNER_FULL_CLOSURE_FORMAT = (
    "semtalk_show_base_fresh_val_winner_full_metric_closure_v1"
)
METRIC_REPORT_HASH_ALGORITHM = (
    "canonical_json_utf8_sorted_compact_newline_v1"
)
DISTRIBUTION_FORMAT = (
    "semtalk_show_deterministic_distribution_receipt_v2"
)
PAYLOAD_HASH_ALGORITHM = "canonical_json_utf8_sorted_compact_v1"
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_SCOPE = "all_speakers_0_1_2_3"
EXPECTED_DATASET = {
    "name": "SHOW",
    "release": "ReleaseV1.0",
    "selection": ">2s",
    "speakers": {
        "oliver": 0,
        "chemistry": 1,
        "seth": 2,
        "conan": 3,
    },
}
STAGES = ("face", "hands", "upper", "lower", "global")
STAGE_SELECTION_METRICS = {
    "face": "face_geometry_expression_objective_v1",
    "hands": "hands_rotation_geometry_objective_v1",
    "upper": "upper_rotation_geometry_objective_v1",
    "lower": "lower_rotation_contact_objective_v1",
    "global": "global_root_contact_objective_v1",
}
# Mandatory prefix retained for compatibility/documentation only.  Formal
# receipts may append e220, e240, ...; every consumer below derives the actual
# inventory through prerequisite_val_contract.validate_candidate_epochs().
PREREQUISITE_CANDIDATE_EPOCHS = tuple(range(20, 201, 20))
PREREQUISITE_UPDATES_PER_EPOCH_BY_STAGE = {
    "face": 497,
    "hands": 497,
    "upper": 497,
    "lower": 497,
    "global": 1_988,
}
FRESH_PREREQUISITE_CONSUMPTION = {
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
        "strict_loaded_models": list(STAGES),
        "decoded_models": list(STAGES),
        "global_translation_reconstruction": True,
    },
}
BASE_CANDIDATE_EPOCHS = (
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
BASE_UPDATES_PER_EPOCH = 248
EXPECTED_VAL_CLIPS = 1_715
EXPECTED_VAL_GLOBAL_INDEX_START = 13_687
EXPECTED_VAL_GLOBAL_INDEX_STOP = (
    EXPECTED_VAL_GLOBAL_INDEX_START + EXPECTED_VAL_CLIPS
)
EXPECTED_SHARDS = 8
EXPECTED_TEST_CLIPS = 1_708
PRIMARY_METRIC = "body.released2.metrics.FGD"
ARTIFACT_KEYS = {"path", "sha256", "bytes"}
PAYLOAD_ARTIFACT_KEYS = {
    "path",
    "sha256",
    "bytes",
    "receipt_payload_sha256",
}
TEST_POLICY = {
    "authorized_evaluations": 1,
    "one_shot_claim_required": True,
    "selection_feedback": False,
    "num_shards": EXPECTED_SHARDS,
    "canonical_test_clips": EXPECTED_TEST_CLIPS,
}
_LOCAL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "gate_released_all_speakers_on_show": (),
    "prerequisite_val_contract": (),
    "merge_prerequisite_val_shards": (
        "gate_released_all_speakers_on_show",
        "prerequisite_val_contract",
    ),
    "selected_prerequisites": (
        "prerequisite_val_contract",
        "merge_prerequisite_val_shards",
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
    "base_long_val_contract": (
        "talkshow_base_val_contract",
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
    "deterministic_replication_gate": (),
    "base_final_authority": (),
    "evaluate_talkshow_show_metrics": (),
}

_FORBIDDEN_COMPACT_TOKENS = (
    "speaker2",
    "semgate",
    "sparse",
    "diff" + "sheg",
)
_GLOBAL_MODEL_TOKEN = "globaldiff"


class PublishedWinnerClaimError(RuntimeError):
    """Raised when a published winner claim is not self-contained and exact."""


def _canonical_regular_path(path_value: Any, label: str) -> Path:
    if not isinstance(path_value, (str, os.PathLike)):
        raise PublishedWinnerClaimError(f"{label} path must be path-like")
    path = Path(path_value)
    if not path.is_absolute():
        raise PublishedWinnerClaimError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PublishedWinnerClaimError(
            f"cannot resolve {label}: {path}"
        ) from error
    if resolved != path:
        raise PublishedWinnerClaimError(
            f"{label} path must be canonical and contain no symlink"
        )
    return path


def _safe_file_snapshot(
    path_value: Any,
    label: str,
) -> tuple[Path, bytes]:
    path = _canonical_regular_path(path_value, label)
    try:
        resolved_identity = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise PublishedWinnerClaimError(
            f"cannot stat {label}: {path}"
        ) from error
    if not stat.S_ISREG(resolved_identity.st_mode):
        raise PublishedWinnerClaimError(
            f"{label} must be a regular non-symlink file"
        )
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise PublishedWinnerClaimError(
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
            raise PublishedWinnerClaimError(
                f"{label} must be a regular non-symlink file"
            )
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(resolved_identity, field) != getattr(before, field)
            for field in stable_fields
        ):
            raise PublishedWinnerClaimError(
                f"{label} changed before it was opened"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        final_path = os.stat(path, follow_symlinks=False)
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(final_path, field)
            for field in stable_fields
        ):
            raise PublishedWinnerClaimError(
                f"{label} changed while it was read"
            )
        try:
            final_resolved = path.resolve(strict=True)
        except OSError as error:
            raise PublishedWinnerClaimError(
                f"{label} path changed while it was read"
            ) from error
        if final_resolved != path:
            raise PublishedWinnerClaimError(
                f"{label} acquired a symlink while it was read"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise PublishedWinnerClaimError(
                f"{label} size changed while it was read"
            )
        return path, payload
    except PublishedWinnerClaimError:
        raise
    except OSError as error:
        raise PublishedWinnerClaimError(
            f"cannot safely read {label}: {path}"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _fresh_local_module(name: str) -> Any:
    if name not in _LOCAL_DEPENDENCIES:
        raise PublishedWinnerClaimError(
            f"unknown local validator {name}"
        )
    source_root = Path(__file__).resolve().parent
    package = sys.modules.get("scripts.show_base")
    if package is None:
        package = __import__("scripts.show_base", fromlist=["*"])
    expected_package = source_root / "__init__.py"
    package_source = getattr(package, "__file__", None)
    if (
        type(package_source) is not str
        or Path(package_source).resolve() != expected_package
    ):
        raise PublishedWinnerClaimError(
            "scripts.show_base package is not this source tree"
        )
    _safe_file_snapshot(
        str(expected_package),
        "scripts.show_base package",
    )

    missing = object()
    saved_modules: dict[str, Any] = {}
    saved_attributes: dict[str, Any] = {}
    loaded: dict[str, Any] = {}

    def load(module_name: str) -> Any:
        if module_name in loaded:
            return loaded[module_name]
        for dependency in _LOCAL_DEPENDENCIES[module_name]:
            load(dependency)
        source_path, source = _safe_file_snapshot(
            str(source_root / f"{module_name}.py"),
            f"local validator {module_name}",
        )
        full_name = f"scripts.show_base.{module_name}"
        saved_modules.setdefault(full_name, sys.modules.get(full_name, missing))
        saved_attributes.setdefault(
            module_name,
            getattr(package, module_name, missing),
        )
        module = types.ModuleType(full_name)
        module.__file__ = str(source_path)
        module.__package__ = "scripts.show_base"
        module.__loader__ = None
        sys.modules[full_name] = module
        setattr(package, module_name, module)
        loaded[module_name] = module
        code = compile(
            source,
            str(source_path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        return module

    try:
        return load(name)
    except PublishedWinnerClaimError:
        raise
    except BaseException as error:
        raise PublishedWinnerClaimError(
            f"cannot source-load local validator {name}"
        ) from error
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


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            "receipt is not canonical finite JSON"
        ) from error
    return (payload + ("\n" if newline else "")).encode("utf-8")


def canonical_json_sha256(value: Any, *, newline: bool = False) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(value, newline=newline)
    ).hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublishedWinnerClaimError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except PublishedWinnerClaimError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            f"{label} is not strict JSON: {error}"
        ) from error


def _exact_mapping(
    value: Any,
    keys: Iterable[str],
    label: str,
) -> dict[str, Any]:
    expected = set(keys)
    if not isinstance(value, dict) or set(value) != expected:
        raise PublishedWinnerClaimError(f"{label} schema mismatch")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublishedWinnerClaimError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PublishedWinnerClaimError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise PublishedWinnerClaimError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PublishedWinnerClaimError(
            f"{label} must be finite and nonnegative"
        )
    return result


def _regular_file(path_value: Any, label: str) -> Path:
    path, _payload = _safe_file_snapshot(path_value, label)
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_artifact(
    value: Any,
    label: str,
    *,
    with_payload: bool,
) -> tuple[dict[str, Any], bytes]:
    keys = PAYLOAD_ARTIFACT_KEYS if with_payload else ARTIFACT_KEYS
    artifact = _exact_mapping(value, keys, f"{label} artifact")
    path, payload = _safe_file_snapshot(artifact["path"], label)
    expected_sha = _require_sha256(
        artifact["sha256"], f"{label} file SHA-256"
    )
    expected_bytes = _require_int(
        artifact["bytes"], f"{label} bytes", minimum=1
    )
    if _sha256_bytes(payload) != expected_sha or len(payload) != expected_bytes:
        raise PublishedWinnerClaimError(f"{label} artifact changed")
    normalized = {
        "path": str(path),
        "sha256": expected_sha,
        "bytes": expected_bytes,
    }
    if with_payload:
        normalized["receipt_payload_sha256"] = _require_sha256(
            artifact["receipt_payload_sha256"],
            f"{label} payload SHA-256",
        )
    if normalized != artifact:
        raise PublishedWinnerClaimError(
            f"{label} artifact path is not canonical"
        )
    return normalized, payload


def _verify_compact_receipt(
    artifact: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized, payload = _normalize_artifact(
        artifact, label, with_payload=True
    )
    value = _strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise PublishedWinnerClaimError(f"{label} must be a JSON object")
    claimed = _require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} embedded payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256")
    if (
        canonical_json_sha256(unsigned) != claimed
        or normalized["receipt_payload_sha256"] != claimed
    ):
        raise PublishedWinnerClaimError(f"{label} payload hash mismatch")
    return normalized, value


def _verify_compact_reference(
    artifact_value: Any,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _exact_mapping(
        artifact_value,
        {"path", "sha256", "receipt_payload_sha256"},
        f"{label} artifact",
    )
    path, payload = _safe_file_snapshot(artifact["path"], label)
    file_sha = _require_sha256(artifact["sha256"], f"{label} file SHA-256")
    payload_sha = _require_sha256(
        artifact["receipt_payload_sha256"], f"{label} payload SHA-256"
    )
    if _sha256_bytes(payload) != file_sha:
        raise PublishedWinnerClaimError(f"{label} artifact changed")
    value = _strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise PublishedWinnerClaimError(f"{label} must be a JSON object")
    embedded = _require_sha256(
        value.get("receipt_payload_sha256"),
        f"{label} embedded payload SHA-256",
    )
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256")
    normalized = {
        "path": str(path),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
    }
    if (
        normalized != artifact
        or embedded != payload_sha
        or canonical_json_sha256(unsigned) != embedded
    ):
        raise PublishedWinnerClaimError(f"{label} payload hash mismatch")
    return normalized, value


def _compact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _reject_forbidden_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "origin" and child != EXPECTED_ORIGIN:
                raise PublishedWinnerClaimError(
                    f"{label}.{key} has a foreign repository origin"
                )
            _reject_forbidden_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_tree(child, f"{label}[{index}]")
    elif isinstance(value, str):
        compact = _compact_token(value)
        if any(token in compact for token in _FORBIDDEN_COMPACT_TOKENS):
            raise PublishedWinnerClaimError(
                f"{label} contains a forbidden generator dependency"
            )


def _validate_checkpoint(value: Any, label: str) -> dict[str, Any]:
    artifact, _payload = _normalize_artifact(
        value, label, with_payload=False
    )
    compact_path = _compact_token(artifact["path"])
    if (
        any(token in compact_path for token in _FORBIDDEN_COMPACT_TOKENS)
        or _GLOBAL_MODEL_TOKEN in compact_path
    ):
        raise PublishedWinnerClaimError(
            f"{label} is not a SemTalk checkpoint dependency"
        )
    return artifact


def _prerequisite_candidate_schedules(
    selection: Mapping[str, Any],
    prerequisite_contract: Any,
) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]]]:
    """Strictly normalize either formal prerequisite selection protocol."""

    protocol = selection.get("protocol")
    if not isinstance(protocol, dict):
        raise PublishedWinnerClaimError(
            "prerequisite selection protocol mismatch"
        )
    try:
        candidate_epochs = prerequisite_contract.validate_candidate_epochs(
            protocol.get("candidate_epochs")
        )
        if selection.get("format") == INITIAL_PREREQUISITE_SELECTION_FORMAT:
            if candidate_epochs != PREREQUISITE_CANDIDATE_EPOCHS:
                raise ValueError(
                    "initial-v1 schedule must be the mandatory prefix"
                )
            candidate_epochs_by_stage = {
                stage: candidate_epochs for stage in STAGES
            }
            expected_protocol = {
                "name": "five_independent_show_prerequisite_validation_v1",
                "candidate_epochs": list(candidate_epochs),
                "candidates_per_stage": len(candidate_epochs),
                "clips_per_candidate": EXPECTED_VAL_CLIPS,
                "shards_per_candidate": EXPECTED_SHARDS,
                "window_length": 64,
                "window_stride": 20,
                "full_base_fgd_used": False,
            }
        elif selection.get("format") == PREREQUISITE_SELECTION_FORMAT:
            raw_schedules = protocol.get("candidate_epochs_by_stage")
            if not isinstance(raw_schedules, dict) or set(
                raw_schedules
            ) != set(STAGES):
                raise ValueError("per-stage schedule coverage mismatch")
            candidate_epochs_by_stage = {
                stage: prerequisite_contract.validate_candidate_epochs(
                    raw_schedules[stage]
                )
                for stage in STAGES
            }
            if tuple(
                sorted(
                    {
                        epoch
                        for values in candidate_epochs_by_stage.values()
                        for epoch in values
                    }
                )
            ) != candidate_epochs:
                raise ValueError("per-stage schedule union mismatch")
            expected_protocol = {
                "name": "five_independent_show_prerequisite_validation_v2",
                "candidate_epochs": list(candidate_epochs),
                "candidate_epochs_by_stage": {
                    stage: list(candidate_epochs_by_stage[stage])
                    for stage in STAGES
                },
                "candidates_per_stage": {
                    stage: len(candidate_epochs_by_stage[stage])
                    for stage in STAGES
                },
                "clips_per_candidate": EXPECTED_VAL_CLIPS,
                "shards_per_candidate": EXPECTED_SHARDS,
                "window_length": 64,
                "window_stride": 20,
                "full_base_fgd_used": False,
            }
        else:
            raise ValueError("selection format mismatch")
    except Exception as error:
        raise PublishedWinnerClaimError(
            f"prerequisite candidate schedule is invalid: {error}"
        ) from error
    if protocol != expected_protocol:
        raise PublishedWinnerClaimError(
            "prerequisite selection protocol mismatch"
        )
    return candidate_epochs, candidate_epochs_by_stage


def _validate_prerequisite_selection(
    artifact_value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    artifact, selection = _verify_compact_receipt(
        artifact_value, "prerequisite selection"
    )
    _exact_mapping(
        selection,
        {
            "format",
            "status",
            "target_dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "protocol",
            "selection_policy",
            "canonical_view",
            "producer_sources",
            "training_sources",
            "config_sha256",
            "candidate_index_receipt",
            "measurement_index_receipt",
            "stages",
            "receipt_payload_sha256",
        },
        "prerequisite selection",
    )
    if (
        selection["format"]
        not in {
            INITIAL_PREREQUISITE_SELECTION_FORMAT,
            PREREQUISITE_SELECTION_FORMAT,
        }
        or selection["status"] != "selected"
        or selection["target_dataset"] != "SHOW"
        or selection["target_speaker_scope"] != EXPECTED_SCOPE
        or selection["split"] != "val"
        or selection["test_visible"] is not False
    ):
        raise PublishedWinnerClaimError(
            "prerequisite selection identity mismatch"
        )
    prerequisite_contract = _fresh_local_module(
        "prerequisite_val_contract"
    )
    if (
        getattr(prerequisite_contract, "REQUIRED_CANDIDATE_EPOCHS", None)
        != PREREQUISITE_CANDIDATE_EPOCHS
        or not callable(
            getattr(prerequisite_contract, "updates_per_epoch", None)
        )
        or not callable(
            getattr(
                prerequisite_contract,
                "validate_candidate_epochs",
                None,
            )
        )
        or not callable(
            getattr(prerequisite_contract, "updates_per_epoch", None)
        )
    ):
        raise PublishedWinnerClaimError(
            "prerequisite schedule validator ABI mismatch"
        )
    try:
        stage_updates_per_epoch = {
            stage: prerequisite_contract.updates_per_epoch(stage)
            for stage in STAGES
        }
    except Exception as error:
        raise PublishedWinnerClaimError(
            f"prerequisite stage update topology is invalid: {error}"
        ) from error
    if stage_updates_per_epoch != PREREQUISITE_UPDATES_PER_EPOCH_BY_STAGE:
        raise PublishedWinnerClaimError(
            "prerequisite stage update topology validator ABI mismatch"
        )
    _candidate_epochs, candidate_epochs_by_stage = (
        _prerequisite_candidate_schedules(
            selection,
            prerequisite_contract,
        )
    )
    if selection["selection_policy"] != {
        "per_stage_independent": True,
        "ordering": [
            "selection_score",
            "epoch",
            "optimizer_updates",
            "checkpoint_sha256",
        ],
        "test_feedback_into_selection": False,
    }:
        raise PublishedWinnerClaimError(
            "prerequisite selection policy mismatch"
        )
    _require_sha256(selection["config_sha256"], "prerequisite config SHA")
    _reject_forbidden_tree(selection, "prerequisite selection")
    for role in ("candidate_index_receipt", "measurement_index_receipt"):
        _verify_compact_reference(selection[role], f"prerequisite {role}")
    stages = selection["stages"]
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise PublishedWinnerClaimError(
            "prerequisite selection must contain exactly five stages"
        )
    fixed: dict[str, dict[str, Any]] = {}
    for expected_stage, raw_stage in zip(STAGES, stages):
        stage = _exact_mapping(
            raw_stage,
            {
                "stage",
                "selection_metric",
                "epoch",
                "optimizer_updates",
                "candidate_index",
                "selection_score",
                "candidate_checkpoint",
                "measurement_receipt",
                "coverage",
            },
            f"prerequisite {expected_stage}",
        )
        epoch = _require_int(
            stage["epoch"], f"prerequisite {expected_stage} epoch", minimum=1
        )
        if (
            stage["stage"] != expected_stage
            or stage["selection_metric"]
            != STAGE_SELECTION_METRICS[expected_stage]
            or epoch not in candidate_epochs_by_stage[expected_stage]
            or stage["optimizer_updates"]
            != epoch * stage_updates_per_epoch[expected_stage]
            or not isinstance(stage["candidate_index"], int)
            or isinstance(stage["candidate_index"], bool)
            or stage["candidate_index"]
            != candidate_epochs_by_stage[expected_stage].index(epoch)
            or _require_number(
                stage["selection_score"],
                f"prerequisite {expected_stage} selection score",
            )
            < 0.0
            or stage["coverage"]
            != {
                "split": "val",
                "test_visible": False,
                "clips": EXPECTED_VAL_CLIPS,
                "shards": EXPECTED_SHARDS,
                "exact_once": True,
                "all_finite": True,
            }
        ):
            raise PublishedWinnerClaimError(
                f"prerequisite {expected_stage} selection changed"
            )
        _verify_compact_reference(
            stage["measurement_receipt"],
            f"prerequisite {expected_stage} measurement",
        )
        fixed[expected_stage] = _validate_checkpoint(
            stage["candidate_checkpoint"],
            f"prerequisite {expected_stage} checkpoint",
        )
    return artifact, selection, fixed


def _validate_continuation_decision(
    artifact_value: Any,
    *,
    prerequisite_artifact: Mapping[str, Any],
    prerequisite_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, decision = _verify_compact_receipt(
        artifact_value, "prerequisite continuation decision"
    )
    _exact_mapping(
        decision,
        {
            "format",
            "status",
            "decision",
            "test_visible",
            "protocol",
            "inputs",
            "stages",
            "receipt_payload_sha256",
        },
        "prerequisite continuation decision",
    )
    if (
        decision["format"] != CONTINUATION_DECISION_FORMAT
        or decision["status"] != "complete"
        or decision["decision"] != "stop"
        or decision["test_visible"] is not False
    ):
        raise PublishedWinnerClaimError(
            "prerequisite continuation is not a final stop decision"
        )
    if decision["protocol"] != {
        "name": "fresh_replayed_independent_stage_val_improvement_v2",
        "score_direction": "lower_is_better",
        "recent_candidates": 3,
        "relative_improvement_reference": (
            "best_of_preceding_two_recent_candidates"
        ),
        "minimum_relative_improvement": 0.005,
        "interval_epochs": 20,
        "stage_cap_epochs": {
            "face": 600,
            "hands": 500,
            "upper": 500,
            "lower": 600,
            "global": 1700,
        },
        "continue_rule": (
            "each_stage_latest_boundary_is_global_val_winner_and_relative_"
            "improvement_gte_threshold_and_below_stage_cap"
        ),
        "terminal_rule": (
            "otherwise_freeze_global_val_winner;at_cap_mark_capped;"
            "terminal_stages_never_reenter"
        ),
    }:
        raise PublishedWinnerClaimError(
            "prerequisite continuation policy mismatch"
        )
    inputs = _exact_mapping(
        decision["inputs"],
        {"selection", "measurement_index", "stage_measurements"},
        "prerequisite continuation inputs",
    )
    expected_selection_binding = {
        key: prerequisite_artifact[key]
        for key in ("path", "sha256", "receipt_payload_sha256")
    }
    if inputs["selection"] != expected_selection_binding:
        raise PublishedWinnerClaimError(
            "continuation decision does not bind the pinned selection"
        )
    prerequisite_contract = _fresh_local_module(
        "prerequisite_val_contract"
    )
    _candidate_epochs, candidate_epochs_by_stage = (
        _prerequisite_candidate_schedules(
            prerequisite_selection,
            prerequisite_contract,
        )
    )
    stage_rows = prerequisite_selection["stages"]
    decision_stages = decision["stages"]
    if not isinstance(decision_stages, list) or len(decision_stages) != len(
        STAGES
    ):
        raise PublishedWinnerClaimError(
            "continuation decision stage coverage mismatch"
        )
    for expected_stage, selected, raw_decision in zip(
        STAGES, stage_rows, decision_stages
    ):
        stage = _exact_mapping(
            raw_decision,
            {
                "stage",
                "recent_candidate_epochs",
                "recent_selection_scores",
                "winner_epoch",
                "latest_epoch",
                "previous_best_score",
                "latest_score",
                "relative_improvement",
                "latest_is_winner",
                "meets_relative_improvement_threshold",
                "cap_epoch",
                "action",
                "target_epoch",
                "frozen_winner_epoch",
                "requests_continuation",
            },
            f"continuation {expected_stage}",
        )
        candidate_epochs = candidate_epochs_by_stage[expected_stage]
        cap = decision["protocol"]["stage_cap_epochs"][expected_stage]
        if (
            stage["stage"] != expected_stage
            or stage["winner_epoch"] != selected["epoch"]
            or stage["latest_epoch"] != candidate_epochs[-1]
            or stage["frozen_winner_epoch"] != selected["epoch"]
            or stage["cap_epoch"] != cap
            or stage["action"] not in {"freeze", "capped"}
            or stage["target_epoch"] is not None
            or stage["requests_continuation"] is not False
            or (
                stage["action"] == "capped"
                and stage["latest_epoch"] != cap
            )
            or (
                stage["action"] == "freeze"
                and stage["latest_epoch"] >= cap
            )
        ):
            raise PublishedWinnerClaimError(
                f"continuation {expected_stage} contradicts selection"
            )
        recent_epochs = stage["recent_candidate_epochs"]
        recent_scores = stage["recent_selection_scores"]
        if (
            not isinstance(recent_epochs, list)
            or recent_epochs != list(candidate_epochs[-3:])
            or not all(
                isinstance(epoch, int) and not isinstance(epoch, bool)
                for epoch in recent_epochs
            )
            or not isinstance(recent_scores, list)
            or len(recent_scores) != 3
        ):
            raise PublishedWinnerClaimError(
                f"continuation {expected_stage} recent evidence changed"
            )
        for index, score in enumerate(recent_scores):
            _require_number(
                score, f"continuation {expected_stage} score {index}"
            )
        _require_number(
            stage["previous_best_score"],
            f"continuation {expected_stage} previous score",
        )
        _require_number(
            stage["latest_score"],
            f"continuation {expected_stage} latest score",
        )
        relative = stage["relative_improvement"]
        if (
            isinstance(relative, bool)
            or type(relative) not in {int, float}
            or not math.isfinite(float(relative))
        ):
            raise PublishedWinnerClaimError(
                f"continuation {expected_stage} improvement is invalid"
            )
        for key in (
            "latest_is_winner",
            "meets_relative_improvement_threshold",
        ):
            if not isinstance(stage[key], bool):
                raise PublishedWinnerClaimError(
                    f"continuation {expected_stage} {key} is invalid"
                )
    _reject_forbidden_tree(decision, "prerequisite continuation")
    return artifact, decision


def _replay_continuation_wave_file(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    module = _fresh_local_module("prerequisite_continuation_wave")
    replay = getattr(module, "replay_wave_file", None)
    if not callable(replay):
        raise PublishedWinnerClaimError(
            "continuation wave fresh replay ABI mismatch"
        )
    try:
        value = replay(Path(artifact["path"]), artifact["sha256"])
    except Exception as error:
        raise PublishedWinnerClaimError(
            f"continuation wave fresh replay failed: {error}"
        ) from error
    if type(value) is not dict:
        raise PublishedWinnerClaimError(
            "continuation wave fresh replay returned no receipt"
        )
    return value


def _validate_continuation_waves(
    values: Any,
    *,
    prerequisite_selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prerequisite_contract = _fresh_local_module(
        "prerequisite_val_contract"
    )
    _candidate_epochs, schedules = _prerequisite_candidate_schedules(
        prerequisite_selection,
        prerequisite_contract,
    )
    if type(values) is not list:
        raise PublishedWinnerClaimError(
            "continuation waves must be a JSON list"
        )
    result: list[dict[str, Any]] = []
    artifact_paths: set[str] = set()
    positions = {
        stage: len(PREREQUISITE_CANDIDATE_EPOCHS) for stage in STAGES
    }
    predecessors: dict[str, dict[str, Any] | None] = {
        stage: None for stage in STAGES
    }
    eligible = set(STAGES)
    for index, raw_artifact in enumerate(values):
        artifact, _payload = _normalize_artifact(
            raw_artifact,
            f"continuation wave {index}",
            with_payload=True,
        )
        if artifact["path"] in artifact_paths:
            raise PublishedWinnerClaimError(
                f"continuation wave {index} artifact path was reused"
            )
        artifact_paths.add(artifact["path"])
        receipt = _replay_continuation_wave_file(artifact)
        if (
            receipt.get("format")
            != "semtalk_show_prerequisite_continuation_wave_v2"
            or receipt.get("status") != "authorized"
            or receipt.get("test_visible") is not False
            or receipt.get("receipt_payload_sha256")
            != artifact["receipt_payload_sha256"]
            or not isinstance(receipt.get("trigger_stages"), list)
            or not receipt["trigger_stages"]
            or not isinstance(receipt.get("stages"), list)
            or len(receipt["stages"]) != len(receipt["trigger_stages"])
        ):
            raise PublishedWinnerClaimError(
                f"continuation wave {index} identity/boundary mismatch"
            )
        triggers = list(receipt["trigger_stages"])
        if (
            triggers != [stage for stage in STAGES if stage in set(triggers)]
            or not set(triggers).issubset(eligible)
        ):
            raise PublishedWinnerClaimError(
                f"continuation wave {index} reactivates a frozen stage"
            )
        eligible = set(triggers)
        binding = {
            key: artifact[key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        }
        for expected_stage, stage in zip(triggers, receipt["stages"]):
            old = stage.get("old_segment") if isinstance(stage, dict) else None
            position = positions[expected_stage]
            if position >= len(schedules[expected_stage]):
                raise PublishedWinnerClaimError(
                    f"continuation wave {index} exceeds {expected_stage} schedule"
                )
            boundary = schedules[expected_stage][position - 1]
            target = schedules[expected_stage][position]
            if (
                not isinstance(stage, dict)
                or stage.get("stage") != expected_stage
                or stage.get("boundary_epoch") != boundary
                or stage.get("target_epoch") != target
                or target != boundary + 20
                or not isinstance(old, dict)
                or old.get("predecessor_wave")
                != predecessors[expected_stage]
            ):
                raise PublishedWinnerClaimError(
                    f"continuation wave {index} predecessor chain mismatch"
                )
            positions[expected_stage] += 1
            predecessors[expected_stage] = binding
        result.append(artifact)
    if any(positions[stage] != len(schedules[stage]) for stage in STAGES):
        raise PublishedWinnerClaimError(
            "continuation waves do not cover every stage-local boundary"
        )
    return result


def _validate_distribution(
    artifact_value: Any,
    *,
    prediction_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, receipt = _verify_compact_receipt(
        artifact_value, "candidate distribution receipt"
    )
    if (
        receipt.get("format") != DISTRIBUTION_FORMAT
        or receipt.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM
        or receipt.get("protocol")
        != "deterministic_replication_of_single_prediction_v1"
        or receipt.get("physical_samples_per_clip") != 1
        or receipt.get("independent_samples") is not False
        or receipt.get("deterministic_delta_distribution") is not True
        or receipt.get("seed_consumed") is not False
        or receipt.get("logical_slots") != list(range(16))
        or receipt.get("released2_slots") != [0, 1]
        or receipt.get("paper16_slots") != list(range(16))
        or receipt.get("face_slot") != 0
        or receipt.get("prediction_manifest") != prediction_manifest
    ):
        raise PublishedWinnerClaimError(
            "candidate distribution receipt changed"
        )
    variation = receipt.get("variation_policy")
    if (
        not isinstance(variation, dict)
        or variation.get("reported_statistic")
        != "raw_metric_primitive_v1"
        or variation.get("exact_zero_claim") is not False
        or variation.get("public_value_transform")
        != "identity_no_clamp_no_round_v1"
    ):
        raise PublishedWinnerClaimError(
            "candidate Variation policy does not expose the raw primitive"
        )
    return artifact, receipt


def _validate_lineage(
    artifact_value: Any,
    *,
    epoch: int,
    updates: int,
    checkpoint: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    distribution_artifact: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, lineage = _verify_compact_receipt(
        artifact_value, f"Base e{epoch} inference lineage"
    )
    _exact_mapping(
        lineage,
        {
            "format",
            "status",
            "generator",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "epoch",
            "optimizer_updates",
            "candidate_checkpoint",
            "prediction_manifest",
            "distribution_receipt",
            "prerequisite_selection",
            "continuation_decision",
            "receipt_payload_sha256",
        },
        f"Base e{epoch} inference lineage",
    )
    if lineage != {
        "format": VAL_LINEAGE_FORMAT,
        "status": "complete",
        "generator": "SemTalk Base Motion Generation",
        "dataset": "SHOW",
        "target_speaker_scope": EXPECTED_SCOPE,
        "split": "val",
        "test_visible": False,
        "epoch": epoch,
        "optimizer_updates": updates,
        "candidate_checkpoint": checkpoint,
        "prediction_manifest": prediction_manifest,
        "distribution_receipt": distribution_artifact,
        "prerequisite_selection": prerequisite_artifact,
        "continuation_decision": continuation_artifact,
        "receipt_payload_sha256": lineage["receipt_payload_sha256"],
    }:
        raise PublishedWinnerClaimError(
            f"Base e{epoch} inference lineage changed"
        )
    _reject_forbidden_tree(lineage, f"Base e{epoch} inference lineage")
    return artifact, lineage


def _strict_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            raise PublishedWinnerClaimError(
                f"{label} contains blank line {line_number}"
            )
        value = _strict_json_bytes(raw_line, f"{label} line {line_number}")
        if not isinstance(value, dict):
            raise PublishedWinnerClaimError(
                f"{label} line {line_number} is not an object"
            )
        rows.append(value)
    return rows


def _reference_matches_payload_artifact(
    reference: Any,
    artifact: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(reference, dict):
        raise PublishedWinnerClaimError(f"{label} reference is not an object")
    for key in ("path", "sha256", "receipt_payload_sha256"):
        if key in reference and reference[key] != artifact.get(key):
            raise PublishedWinnerClaimError(f"{label} reference changed")
    if reference.get("path") != artifact.get("path") or reference.get(
        "sha256"
    ) != artifact.get("sha256"):
        raise PublishedWinnerClaimError(f"{label} reference is incomplete")


def _validate_frozen_val_authority(
    *,
    val_inputs_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        contract = _fresh_local_module("base_long_val_contract")
        validated_val, coverage = contract.validate_val_inputs(
            Path(val_inputs_artifact["path"]),
            val_inputs_artifact["sha256"],
        )
        validated_pipeline, pipeline = contract.validate_pipeline(
            Path(pipeline_artifact["path"]),
            pipeline_artifact["sha256"],
            expected_prerequisite_selection=prerequisite_artifact,
        )
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            f"{label} audited validation authority replay failed"
        ) from error
    for expected, observed, role in (
        (val_inputs_artifact, validated_val, "val inputs"),
        (pipeline_artifact, validated_pipeline, "pipeline"),
    ):
        if any(
            expected.get(key) != observed.get(key)
            for key in ("path", "sha256", "receipt_payload_sha256")
        ):
            raise PublishedWinnerClaimError(
                f"{label} audited {role} binding changed"
            )
    ordered = coverage.get("_ordered_clips")
    canonical = coverage.get("canonical_manifest")
    if (
        not isinstance(ordered, list)
        or len(ordered) != EXPECTED_VAL_CLIPS
        or not isinstance(canonical, dict)
    ):
        raise PublishedWinnerClaimError(
            f"{label} audited validation coverage changed"
        )
    return coverage, dict(pipeline)


def _validate_formal_val_lineage(
    artifact_value: Any,
    *,
    epoch: int,
    checkpoint: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    val_inputs_artifact: Mapping[str, Any],
    pipeline_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    artifact, lineage = _verify_compact_receipt(
        artifact_value, f"Base e{epoch} formal inference lineage"
    )
    _exact_mapping(
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
            "talkshow_window_manifest_sha256",
            "prediction_files",
            "ground_truth_files",
            "exact_once",
            "finite",
            "receipt_payload_sha256",
        },
        f"Base e{epoch} formal inference lineage",
    )
    checkpoint_reference = lineage["candidate_checkpoint"]
    if (
        lineage["format"] != FORMAL_VAL_LINEAGE_FORMAT
        or lineage["status"] != "complete"
        or lineage["split"] != "val"
        or lineage["test_visible"] is not False
        or lineage["epoch"] != epoch
        or checkpoint_reference
        != {"path": checkpoint["path"], "sha256": checkpoint["sha256"]}
        or lineage["clip_count"] != EXPECTED_VAL_CLIPS
        or lineage["prediction_files"] != EXPECTED_VAL_CLIPS
        or lineage["ground_truth_files"] != EXPECTED_VAL_CLIPS
        or lineage["exact_once"] is not True
        or lineage["finite"] is not True
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} formal inference lineage is not exact val-only"
        )
    _reference_matches_payload_artifact(
        lineage["val_inputs_receipt"],
        val_inputs_artifact,
        f"Base e{epoch} val inputs",
    )
    _reference_matches_payload_artifact(
        lineage["pipeline_receipt"],
        pipeline_artifact,
        f"Base e{epoch} pipeline",
    )
    final_manifest = lineage["final_manifest"]
    if (
        not isinstance(final_manifest, dict)
        or final_manifest.get("path") != prediction_manifest["path"]
        or final_manifest.get("sha256") != prediction_manifest["sha256"]
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} final prediction manifest binding changed"
        )
    manifest_artifact, manifest_payload = _normalize_artifact(
        prediction_manifest,
        f"Base e{epoch} prediction manifest",
        with_payload=False,
    )
    rows = _strict_jsonl_bytes(
        manifest_payload, f"Base e{epoch} prediction manifest"
    )
    if len(rows) != EXPECTED_VAL_CLIPS:
        raise PublishedWinnerClaimError(
            f"Base e{epoch} prediction manifest is not 1715 clips"
        )
    expected_row_keys = {
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
    prediction_paths: set[str] = set()
    ground_truth_paths: set[str] = set()
    source_clip_ids: set[str] = set()
    canonical_clip_ids: set[str] = set()
    prediction_dir = Path(lineage["prediction_dir"])
    ground_truth_dir = Path(lineage["ground_truth_dir"])
    if (
        not prediction_dir.is_absolute()
        or not ground_truth_dir.is_absolute()
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} output directories are not absolute"
        )
    for position, row in enumerate(rows):
        source_clip_id = row.get("source_clip_id")
        canonical_clip_id = row.get("canonical_clip_id")
        prediction = row.get("prediction")
        ground_truth = row.get("ground_truth")
        if (
            set(row) != expected_row_keys
            or row.get("global_index")
            != EXPECTED_VAL_GLOBAL_INDEX_START + position
            or row.get("split") != "val"
            or row.get("epoch") != epoch
            or row.get("candidate_checkpoint_sha256")
            != checkpoint["sha256"]
            or not isinstance(source_clip_id, str)
            or not source_clip_id
            or source_clip_id in source_clip_ids
            or not isinstance(canonical_clip_id, str)
            or not canonical_clip_id
            or canonical_clip_id in canonical_clip_ids
            or isinstance(row.get("frames"), bool)
            or not isinstance(row.get("frames"), int)
            or row["frames"] < 1
            or not isinstance(prediction, dict)
            or set(prediction) != ARTIFACT_KEYS
            or not isinstance(ground_truth, dict)
            or set(ground_truth) != ARTIFACT_KEYS
            or prediction.get("path")
            != str(prediction_dir / f"res_{canonical_clip_id}.npz")
            or ground_truth.get("path")
            != str(ground_truth_dir / f"gt_{canonical_clip_id}.npz")
            or prediction.get("path") in prediction_paths
            or ground_truth.get("path") in ground_truth_paths
            or isinstance(prediction.get("bytes"), bool)
            or not isinstance(prediction.get("bytes"), int)
            or prediction["bytes"] < 1
            or isinstance(ground_truth.get("bytes"), bool)
            or not isinstance(ground_truth.get("bytes"), int)
            or ground_truth["bytes"] < 1
        ):
            raise PublishedWinnerClaimError(
                f"Base e{epoch} prediction row {position} is not exact-once"
            )
        _require_sha256(
            prediction.get("sha256"),
            f"Base e{epoch} prediction row {position} SHA-256",
        )
        _require_sha256(
            ground_truth.get("sha256"),
            f"Base e{epoch} ground-truth row {position} SHA-256",
        )
        source_clip_ids.add(source_clip_id)
        canonical_clip_ids.add(canonical_clip_id)
        prediction_paths.add(prediction["path"])
        ground_truth_paths.add(ground_truth["path"])
    if (
        [row["global_index"] for row in rows]
        != list(
            range(
                EXPECTED_VAL_GLOBAL_INDEX_START,
                EXPECTED_VAL_GLOBAL_INDEX_STOP,
            )
        )
        or len(prediction_paths) != EXPECTED_VAL_CLIPS
        or len(ground_truth_paths) != EXPECTED_VAL_CLIPS
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} prediction identity coverage changed"
        )
    return artifact, lineage, rows


def _validate_actual_model_receipts(
    value: Any,
    *,
    epoch: int,
    checkpoint: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    label: str,
) -> str:
    receipts = _exact_mapping(
        value,
        {"base", "face", "hands", "upper", "lower", "global"},
        label,
    )
    base = _exact_mapping(
        receipts["base"],
        {
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
        },
        f"{label} Base",
    )
    base_artifact = {
        key: base[key] for key in ("path", "sha256", "bytes")
    }
    frozen_receipt_sha = (
        preflight.get("candidate_bundle", {})
        .get("frozen_inputs", {})
        .get("receipt_sha256")
    )
    updates_per_epoch = (
        preflight.get("candidate_bundle", {}).get("updates_per_epoch")
    )
    if (
        base_artifact != checkpoint
        or base["stage"] != "base"
        or base["candidate_epoch"] != epoch
        or updates_per_epoch not in {62, 124, 248, 1988}
        or base["updates_per_epoch"] != updates_per_epoch
        or base["optimizer_updates"] != epoch * updates_per_epoch
        or base["frozen_receipt_sha256"] != frozen_receipt_sha
        or base["strict_state_dict_load"] is not True
        or base["all_model_state_tensors_finite"] is not True
        or base["frozen_eval"] is not True
    ):
        raise PublishedWinnerClaimError(
            f"{label} Base actual-load receipt changed"
        )
    fixed = pipeline.get("fixed_checkpoints")
    if (
        pipeline.get("prerequisite_consumption")
        != FRESH_PREREQUISITE_CONSUMPTION
        or not isinstance(fixed, dict)
        or set(fixed) != set(STAGES)
    ):
        raise PublishedWinnerClaimError(
            f"{label} fresh pipeline consumption/fixed-five coverage changed"
        )
    expected_model_keys = {
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
    fixed_keys = {
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
    }
    for stage in STAGES:
        actual = _exact_mapping(
            receipts[stage], expected_model_keys, f"{label} {stage}"
        )
        fixed_row = _exact_mapping(
            fixed[stage], fixed_keys, f"{label} fixed {stage}"
        )
        if (
            {key: actual[key] for key in fixed_keys} != fixed_row
            or actual["prerequisite_selection"]
            != pipeline.get("prerequisite_selection")
            or _require_int(
                actual["model_state_tensors"],
                f"{label} {stage} model-state tensors",
                minimum=1,
            )
            < 1
            or _require_sha256(
                actual["model_state_schema_sha256"],
                f"{label} {stage} model-state schema",
            )
            != actual["model_state_schema_sha256"]
            or actual["strict_state_dict_load"] is not True
            or actual["all_model_state_tensors_finite"] is not True
            or actual["frozen_eval"] is not True
        ):
            raise PublishedWinnerClaimError(
                f"{label} {stage} actual load differs from fresh fixed five"
            )
    return canonical_json_sha256(receipts)


def _validate_candidate_transaction(
    artifact_value: Any,
    *,
    epoch: int,
    updates: int,
    checkpoint: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    distribution_artifact: Mapping[str, Any],
    distribution: Mapping[str, Any],
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Fresh-replay one immutable 8-GPU/1,715-clip validation closure."""

    artifact, transaction = _verify_compact_receipt(
        artifact_value, f"Base e{epoch} candidate transaction"
    )
    _exact_mapping(
        transaction,
        {
            "format",
            "payload_hash_algorithm",
            "status",
            "generator",
            "generator_module",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "formal_host",
            "epoch",
            "optimizer_updates",
            "candidate_checkpoint",
            "prerequisite_selection",
            "continuation_decision",
            "preflight_receipt",
            "val_inputs_receipt",
            "pipeline_receipt",
            "prediction_manifest",
            "inference_lineage",
            "distribution_receipt",
            "validation_gate",
            "shards",
            "failure_manifest",
            "source_runtime_input_pins",
            "coverage",
            "receipt_payload_sha256",
        },
        f"Base e{epoch} candidate transaction",
    )
    expected_formal_host = FRESH_VAL_FORMAL_HOSTS[
        BASE_CANDIDATE_EPOCHS.index(epoch) % len(FRESH_VAL_FORMAL_HOSTS)
    ]
    if (
        transaction["format"] != FRESH_VAL_TRANSACTION_FORMAT
        or transaction["payload_hash_algorithm"] != PAYLOAD_HASH_ALGORITHM
        or transaction["status"] != "complete"
        or transaction["generator"] != "SemTalk Base Motion Generation"
        or transaction["generator_module"] != FRESH_VAL_GENERATOR_MODULE
        or transaction["dataset"] != "SHOW"
        or transaction["target_speaker_scope"] != EXPECTED_SCOPE
        or transaction["split"] != "val"
        or transaction["test_visible"] is not False
        or transaction["formal_host"] != expected_formal_host
        or transaction["epoch"] != epoch
        or transaction["optimizer_updates"] != updates
        or transaction["candidate_checkpoint"] != checkpoint
        or transaction["prerequisite_selection"] != prerequisite_artifact
        or transaction["continuation_decision"] != continuation_artifact
        or transaction["prediction_manifest"] != prediction_manifest
        or transaction["distribution_receipt"] != distribution_artifact
        or transaction["validation_gate"]
        != distribution.get("validation_gate")
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} candidate transaction identity changed"
        )

    preflight_artifact, preflight = _verify_compact_receipt(
        transaction["preflight_receipt"],
        f"Base e{epoch} validation preflight",
    )
    if (
        preflight.get("format") != FRESH_VAL_PREFLIGHT_FORMAT
        or preflight.get("status") != "complete"
        or preflight.get("split") != "val"
        or preflight.get("test_visible") is not False
        or preflight.get("candidate_epochs") != list(BASE_CANDIDATE_EPOCHS)
        or preflight.get("coverage", {}).get("clip_count")
        != EXPECTED_VAL_CLIPS
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} validation preflight changed"
        )
    preflight_candidate = (
        preflight.get("candidate_bundle", {})
        .get("candidates", {})
        .get(str(epoch))
    )
    if preflight_candidate != checkpoint:
        raise PublishedWinnerClaimError(
            f"Base e{epoch} preflight checkpoint differs from transaction"
        )
    val_inputs_artifact, _val_inputs = _verify_compact_receipt(
        transaction["val_inputs_receipt"], f"Base e{epoch} val inputs"
    )
    pipeline_artifact, _pipeline = _verify_compact_receipt(
        transaction["pipeline_receipt"], f"Base e{epoch} pipeline"
    )
    _reference_matches_payload_artifact(
        preflight.get("val_inputs_receipt"),
        val_inputs_artifact,
        f"Base e{epoch} preflight val inputs",
    )
    _reference_matches_payload_artifact(
        preflight.get("pipeline_receipt"),
        pipeline_artifact,
        f"Base e{epoch} preflight pipeline",
    )
    validation_coverage, pipeline = _validate_frozen_val_authority(
        val_inputs_artifact=val_inputs_artifact,
        pipeline_artifact=pipeline_artifact,
        prerequisite_artifact=prerequisite_artifact,
        label=f"Base e{epoch}",
    )
    lineage_artifact, lineage, final_rows = _validate_formal_val_lineage(
        transaction["inference_lineage"],
        epoch=epoch,
        checkpoint=checkpoint,
        prediction_manifest=prediction_manifest,
        val_inputs_artifact=val_inputs_artifact,
        pipeline_artifact=pipeline_artifact,
    )
    if transaction["inference_lineage"] != lineage_artifact:
        raise PublishedWinnerClaimError(
            f"Base e{epoch} transaction lineage is not canonical"
        )
    ordered_clips = validation_coverage["_ordered_clips"]
    for position, (final_row, canonical_row) in enumerate(
        zip(final_rows, ordered_clips)
    ):
        if any(
            final_row.get(key) != canonical_row.get(key)
            for key in (
                "global_index",
                "source_clip_id",
                "canonical_clip_id",
                "frames",
            )
        ):
            raise PublishedWinnerClaimError(
                f"Base e{epoch} final row {position} differs from audited val inventory"
            )
    canonical_artifact = validation_coverage["canonical_manifest"]
    canonical_path, canonical_payload = _safe_file_snapshot(
        canonical_artifact["path"],
        f"Base e{epoch} audited canonical manifest",
    )
    if (
        str(canonical_path) != canonical_artifact["path"]
        or _sha256_bytes(canonical_payload) != canonical_artifact["sha256"]
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} audited canonical manifest changed"
        )
    expected_metric_canonical = {
        "path": canonical_artifact["path"],
        "sha256": canonical_artifact["sha256"],
        "bytes": len(canonical_payload),
        "rows": EXPECTED_VAL_CLIPS,
        "selected_rows": EXPECTED_VAL_CLIPS,
    }

    gate_artifact, gate = _verify_compact_receipt(
        transaction["validation_gate"],
        f"Base e{epoch} deterministic validation gate",
    )
    gate_base = (
        gate.get("model_bundle", {}).get("checkpoints", {}).get("base")
    )
    if (
        gate.get("format") != "semtalk_show_deterministic_replication_gate_v2"
        or gate.get("status") != "pass"
        or gate.get("split") != "val"
        or gate.get("test_visible") is not False
        or gate.get("coverage_mode") != "full_frozen_val_1715"
        or gate.get("proof", {}).get("clip_count") != EXPECTED_VAL_CLIPS
        or gate_base != checkpoint
        or gate_artifact != transaction["validation_gate"]
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} deterministic gate is not checkpoint-bound"
        )

    shard_rows = transaction["shards"]
    if not isinstance(shard_rows, list) or len(shard_rows) != EXPECTED_SHARDS:
        raise PublishedWinnerClaimError(
            f"Base e{epoch} transaction must contain exactly eight shards"
        )
    common_model_sha: str | None = None
    common_runtime_sha: str | None = None
    total_clips = 0
    merged_shard_rows: list[dict[str, Any] | None] = [
        None for _ in range(EXPECTED_VAL_CLIPS)
    ]
    receipt_paths: set[str] = set()
    manifest_paths: set[str] = set()
    normalized_shards: list[dict[str, Any]] = []
    compact_preflight = {
        key: preflight_artifact[key]
        for key in ("path", "sha256", "receipt_payload_sha256")
    }
    for expected_shard, raw_shard in enumerate(shard_rows):
        shard = _exact_mapping(
            raw_shard,
            {"shard_id", "receipt", "manifest"},
            f"Base e{epoch} shard {expected_shard} transaction row",
        )
        if shard["shard_id"] != expected_shard:
            raise PublishedWinnerClaimError(
                f"Base e{epoch} shard ordering changed"
            )
        receipt_artifact, receipt = _verify_compact_receipt(
            shard["receipt"], f"Base e{epoch} shard {expected_shard} receipt"
        )
        manifest_artifact, manifest_payload = _normalize_artifact(
            shard["manifest"],
            f"Base e{epoch} shard {expected_shard} manifest",
            with_payload=False,
        )
        manifest_reference = receipt.get("manifest")
        if (
            receipt.get("format") != FRESH_VAL_SHARD_FORMAT
            or receipt.get("status") != "complete"
            or receipt.get("split") != "val"
            or receipt.get("test_visible") is not False
            or receipt.get("epoch") != epoch
            or receipt.get("candidate_checkpoint")
            != {"path": checkpoint["path"], "sha256": checkpoint["sha256"]}
            or receipt.get("preflight_receipt") != compact_preflight
            or receipt.get("assignment") != FRESH_VAL_ASSIGNMENT
            or receipt.get("shard_id") != expected_shard
            or receipt.get("num_shards") != EXPECTED_SHARDS
            or not isinstance(receipt.get("device"), dict)
            or receipt["device"].get("device") != f"cuda:{expected_shard}"
            or receipt.get("exact_once") is not True
            or receipt.get("finite") is not True
            or not isinstance(manifest_reference, dict)
            or manifest_reference.get("path") != manifest_artifact["path"]
            or manifest_reference.get("sha256") != manifest_artifact["sha256"]
        ):
            raise PublishedWinnerClaimError(
                f"Base e{epoch} shard {expected_shard} receipt changed"
            )
        manifest_rows = _strict_jsonl_bytes(
            manifest_payload,
            f"Base e{epoch} shard {expected_shard} manifest",
        )
        expected_positions = list(
            range(expected_shard, EXPECTED_VAL_CLIPS, EXPECTED_SHARDS)
        )
        if (
            [row.get("canonical_position") for row in manifest_rows]
            != expected_positions
            or receipt.get("clip_count") != len(manifest_rows)
            or receipt.get("prediction_files") != len(manifest_rows)
            or receipt.get("ground_truth_files") != len(manifest_rows)
        ):
            raise PublishedWinnerClaimError(
                f"Base e{epoch} shard {expected_shard} coverage changed"
            )
        shard_row_keys = {
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
        for source, position in zip(manifest_rows, expected_positions):
            final = final_rows[position]
            prediction = source.get("prediction")
            ground_truth = source.get("ground_truth")
            if (
                set(source) != shard_row_keys
                or source.get("canonical_position") != position
                or source.get("global_index")
                != EXPECTED_VAL_GLOBAL_INDEX_START + position
                or source.get("split") != "val"
                or source.get("epoch") != epoch
                or source.get("candidate_checkpoint_sha256")
                != checkpoint["sha256"]
                or not isinstance(prediction, dict)
                or set(prediction) != ARTIFACT_KEYS
                or not isinstance(ground_truth, dict)
                or set(ground_truth) != ARTIFACT_KEYS
                or merged_shard_rows[position] is not None
            ):
                raise PublishedWinnerClaimError(
                    f"Base e{epoch} shard row {position} changed"
                )
            source_identity = {
                key: source[key]
                for key in (
                    "global_index",
                    "split",
                    "source_clip_id",
                    "canonical_clip_id",
                    "frames",
                    "epoch",
                    "candidate_checkpoint_sha256",
                )
            }
            final_identity = {
                key: final[key]
                for key in source_identity
            }
            if (
                source_identity != final_identity
                or {
                    key: prediction[key]
                    for key in ("sha256", "bytes")
                }
                != {
                    key: final["prediction"][key]
                    for key in ("sha256", "bytes")
                }
                or {
                    key: ground_truth[key]
                    for key in ("sha256", "bytes")
                }
                != {
                    key: final["ground_truth"][key]
                    for key in ("sha256", "bytes")
                }
            ):
                raise PublishedWinnerClaimError(
                    f"Base e{epoch} shard/final manifest diverged at {position}"
                )
            merged_shard_rows[position] = source
        model_sha = _require_sha256(
            receipt.get("model_receipts_sha256"),
            f"Base e{epoch} shard model-receipt SHA",
        )
        runtime_sha = _require_sha256(
            receipt.get("runtime_contract_sha256"),
            f"Base e{epoch} shard runtime SHA",
        )
        if (
            not isinstance(receipt.get("model_receipts"), dict)
            or not isinstance(receipt.get("runtime_contract"), dict)
            or canonical_json_sha256(receipt["model_receipts"])
            != model_sha
            or canonical_json_sha256(receipt["runtime_contract"])
            != runtime_sha
        ):
            raise PublishedWinnerClaimError(
                f"Base e{epoch} shard model/runtime payload pins changed"
            )
        validated_model_sha = _validate_actual_model_receipts(
            receipt["model_receipts"],
            epoch=epoch,
            checkpoint=checkpoint,
            preflight=preflight,
            pipeline=pipeline,
            label=f"Base e{epoch} shard {expected_shard}",
        )
        if validated_model_sha != model_sha:
            raise PublishedWinnerClaimError(
                f"Base e{epoch} shard {expected_shard} actual-load hash "
                "changed"
            )
        if common_model_sha is None:
            common_model_sha = model_sha
            common_runtime_sha = runtime_sha
        elif model_sha != common_model_sha or runtime_sha != common_runtime_sha:
            raise PublishedWinnerClaimError(
                f"Base e{epoch} shards disagree on model/runtime pins"
            )
        if (
            receipt_artifact["path"] in receipt_paths
            or manifest_artifact["path"] in manifest_paths
        ):
            raise PublishedWinnerClaimError(
                f"Base e{epoch} shard evidence path was reused"
            )
        receipt_paths.add(receipt_artifact["path"])
        manifest_paths.add(manifest_artifact["path"])
        total_clips += len(manifest_rows)
        normalized_shards.append(
            {
                "shard_id": expected_shard,
                "receipt": receipt_artifact,
                "manifest": manifest_artifact,
            }
        )

    failure_artifact, failures = _verify_compact_receipt(
        transaction["failure_manifest"],
        f"Base e{epoch} failure manifest",
    )
    if failures != {
        "format": FRESH_VAL_FAILURE_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "epoch": epoch,
        "failure_count": 0,
        "failures": [],
        "receipt_payload_sha256": failures["receipt_payload_sha256"],
    }:
        raise PublishedWinnerClaimError(
            f"Base e{epoch} failure manifest is not empty"
        )
    coverage = {
        "clip_count": EXPECTED_VAL_CLIPS,
        "num_shards": EXPECTED_SHARDS,
        "exact_once": True,
        "all_finite": True,
        "failure_count": 0,
    }
    pins = {
        "source": preflight.get("candidate_bundle", {}).get(
            "producer_source"
        ),
        "preflight_receipt_payload_sha256": preflight_artifact[
            "receipt_payload_sha256"
        ],
        "val_inputs_receipt": val_inputs_artifact,
        "pipeline_receipt": pipeline_artifact,
        "model_receipts_sha256": common_model_sha,
        "runtime_contract_sha256": common_runtime_sha,
        "prediction_manifest_sha256": prediction_manifest["sha256"],
        "distribution_receipt_payload_sha256": distribution_artifact[
            "receipt_payload_sha256"
        ],
    }
    if (
        total_clips != EXPECTED_VAL_CLIPS
        or any(row is None for row in merged_shard_rows)
        or transaction["coverage"] != coverage
        or transaction["source_runtime_input_pins"] != pins
        or transaction["shards"] != normalized_shards
        or transaction["failure_manifest"] != failure_artifact
    ):
        raise PublishedWinnerClaimError(
            f"Base e{epoch} transaction closure changed"
        )
    _reject_forbidden_tree(
        {
            "generator_module": transaction["generator_module"],
            "source": pins["source"],
            "checkpoint": checkpoint,
        },
        f"Base e{epoch} transaction generator closure",
    )
    return (
        artifact,
        transaction,
        lineage_artifact,
        expected_metric_canonical,
    )


def _validate_metric_report(
    artifact_value: Any,
    *,
    prediction_manifest: Mapping[str, Any],
    lineage_artifact: Mapping[str, Any],
    distribution: Mapping[str, Any],
    expected_canonical_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact, payload = _normalize_artifact(
        artifact_value, "TalkSHOW validation metric report", with_payload=False
    )
    report = _strict_json_bytes(payload, "TalkSHOW validation metric report")
    _exact_mapping(
        report,
        {
            "format",
            "report_payload_hash_algorithm",
            "status",
            "generator",
            "dataset",
            "split",
            "selection_protocol",
            "distribution_receipt",
            "inputs",
            "metric_assets",
            "counts",
            "body",
            "face",
            "rs",
            "runtime",
            "formal_mode",
            "test_only_mode",
            "report_payload_sha256",
        },
        "TalkSHOW validation metric report",
    )
    claimed = _require_sha256(
        report["report_payload_sha256"],
        "TalkSHOW report payload SHA-256",
    )
    unsigned = dict(report)
    unsigned.pop("report_payload_sha256")
    if (
        canonical_json_sha256(unsigned, newline=True) != claimed
        or report["report_payload_hash_algorithm"]
        != METRIC_REPORT_HASH_ALGORITHM
    ):
        raise PublishedWinnerClaimError(
            "TalkSHOW validation report payload hash mismatch"
        )
    expected_protocol = {
        "primary_metric": PRIMARY_METRIC,
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 0,
    }
    if (
        report["format"] != METRIC_REPORT_FORMAT
        or report["status"] != "complete"
        or report["generator"] != "SemTalk Base-only"
        or report["dataset"] != EXPECTED_DATASET
        or report["split"] != "val"
        or report["selection_protocol"] != expected_protocol
        or report["distribution_receipt"] != distribution
        or report["formal_mode"] is not True
        or report["test_only_mode"] is not False
    ):
        raise PublishedWinnerClaimError(
            "TalkSHOW validation report identity changed"
        )
    inputs = report["inputs"]
    if not isinstance(inputs, dict):
        raise PublishedWinnerClaimError("TalkSHOW report inputs are invalid")
    prediction_input = inputs.get("prediction_manifest")
    lineage_input = inputs.get("prediction_lineage")
    canonical_input = inputs.get("canonical_manifest")
    if (
        not isinstance(prediction_input, dict)
        or {
            key: prediction_input.get(key) for key in ARTIFACT_KEYS
        }
        != prediction_manifest
        or not isinstance(lineage_input, dict)
        or {
            key: lineage_input.get(key) for key in ARTIFACT_KEYS
        }
        != {
            key: lineage_artifact[key] for key in ARTIFACT_KEYS
        }
        or lineage_input.get("payload_sha256")
        != lineage_artifact["receipt_payload_sha256"]
    ):
        raise PublishedWinnerClaimError(
            "TalkSHOW report input bindings changed"
        )
    if (
        expected_canonical_manifest is not None
        and canonical_input != dict(expected_canonical_manifest)
    ):
        raise PublishedWinnerClaimError(
            "TalkSHOW report canonical manifest differs from audited val inputs"
        )
    try:
        metric_adapter = _fresh_local_module(
            "evaluate_talkshow_show_metrics"
        )
        validation = metric_adapter.validate_report(
            report,
            expected_split="val",
            expected_clip_count=EXPECTED_VAL_CLIPS,
            expected_prediction_manifest=prediction_manifest,
            expected_distribution_receipt=distribution,
            expected_selection_protocol=expected_protocol,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            "neutral TalkSHOW report replay failed"
        ) from error
    body = report["body"]
    released = body.get("released2") if isinstance(body, dict) else None
    metrics = released.get("metrics") if isinstance(released, dict) else None
    if not isinstance(metrics, dict) or set(metrics) != {
        "FGD",
        "Variation",
        "BC",
    }:
        raise PublishedWinnerClaimError(
            "TalkSHOW released2 metric schema changed"
        )
    report_fgd = _require_number(metrics["FGD"], PRIMARY_METRIC)
    if (
        not isinstance(validation, dict)
        or validation.get("status") != "pass"
        or validation.get("split") != "val"
        or validation.get("clips") != EXPECTED_VAL_CLIPS
        or validation.get("primary_metric_path") != PRIMARY_METRIC
        or _require_number(
            validation.get("primary_metric"),
            "fresh TalkSHOW primary metric",
        )
        != report_fgd
    ):
        raise PublishedWinnerClaimError(
            "neutral TalkSHOW report replay changed the primary metric"
        )
    return artifact, report, validation


def _validate_primary_replay_receipt(
    artifact_value: Any,
    *,
    report: Mapping[str, Any],
    report_validation: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    distribution: Mapping[str, Any],
    expected_real_feature_cache: Mapping[str, Any],
    expected_canonical_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    if expected_canonical_manifest is not None:
        pinned_cache, pinned_cache_payload = _verify_compact_receipt(
            expected_real_feature_cache,
            "released2 real-feature cache",
        )
        if (
            pinned_cache != expected_real_feature_cache
            or pinned_cache_payload.get("canonical_manifest")
            != dict(expected_canonical_manifest)
        ):
            raise PublishedWinnerClaimError(
                "released2 real-feature cache differs from audited val inputs"
            )
    artifact = _exact_mapping(
        artifact_value,
        PAYLOAD_ARTIFACT_KEYS,
        "released2 primary replay artifact",
    )
    try:
        metric_adapter = _fresh_local_module(
            "evaluate_talkshow_show_metrics"
        )
        replay = metric_adapter.validate_released2_primary_replay_receipt(
            artifact,
            expected_report=report,
            expected_prediction_manifest=prediction_manifest,
            expected_distribution_receipt=distribution,
            expected_selection_protocol={
                "primary_metric": PRIMARY_METRIC,
                "mode": "min",
                "validation_only_for_selection": True,
                "test_evaluations": 0,
            },
            expected_split="val",
            expected_clip_count=EXPECTED_VAL_CLIPS,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            "released2 primary fresh replay verification failed"
        ) from error
    replay = _exact_mapping(
        replay,
        {
            "artifact",
            "receipt_payload_sha256",
            "primary_metric_path",
            "primary_metric",
            "report_payload_sha256",
            "prediction_manifest",
            "real_feature_cache",
            "metric_assets",
            "runtime",
        },
        "released2 primary replay validation",
    )
    normalized_artifact, _payload = _normalize_artifact(
        artifact,
        "released2 primary replay",
        with_payload=True,
    )
    cache_artifact, _cache_payload = _normalize_artifact(
        replay["real_feature_cache"],
        "released2 real-feature cache",
        with_payload=True,
    )
    primary = _require_number(
        replay["primary_metric"], "fresh-replayed released2 FGD"
    )
    if (
        replay["artifact"] != normalized_artifact
        or replay["receipt_payload_sha256"]
        != normalized_artifact["receipt_payload_sha256"]
        or replay["primary_metric_path"] != PRIMARY_METRIC
        or replay["report_payload_sha256"]
        != report_validation["report_payload_sha256"]
        or replay["prediction_manifest"] != prediction_manifest
        or replay["real_feature_cache"] != cache_artifact
        or cache_artifact != expected_real_feature_cache
        or not isinstance(replay["metric_assets"], dict)
        or not isinstance(replay["runtime"], dict)
    ):
        raise PublishedWinnerClaimError(
            "released2 primary fresh replay authority changed"
        )
    return normalized_artifact, primary


def _validate_primary_screen_receipt(
    artifact_value: Any,
    *,
    prediction_manifest: Mapping[str, Any],
    distribution_artifact: Mapping[str, Any],
    expected_real_feature_cache: Mapping[str, Any],
    expected_canonical_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    pinned_cache, pinned_cache_payload = _verify_compact_receipt(
        expected_real_feature_cache,
        "released2 real-feature cache",
    )
    if (
        pinned_cache != expected_real_feature_cache
        or pinned_cache_payload.get("canonical_manifest")
        != dict(expected_canonical_manifest)
    ):
        raise PublishedWinnerClaimError(
            "primary screen real-feature cache differs from audited val inputs"
        )
    artifact = _exact_mapping(
        artifact_value,
        PAYLOAD_ARTIFACT_KEYS,
        "released2 primary screen artifact",
    )
    try:
        metric_adapter = _fresh_local_module(
            "evaluate_talkshow_show_metrics"
        )
        validation = metric_adapter.validate_released2_primary_screen_receipt(
            artifact,
            expected_prediction_manifest=prediction_manifest,
            expected_distribution_receipt=distribution_artifact,
            expected_real_feature_cache=pinned_cache,
            expected_canonical_manifest=expected_canonical_manifest,
            expected_selection_protocol={
                "primary_metric": PRIMARY_METRIC,
                "mode": "min",
                "validation_only_for_selection": True,
                "test_evaluations": 0,
            },
            expected_split="val",
            expected_clip_count=EXPECTED_VAL_CLIPS,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            "released2 primary screen verification failed"
        ) from error
    validation = _exact_mapping(
        validation,
        {
            "artifact",
            "primary_metric_path",
            "primary_metric",
            "real_feature_statistics",
            "generated_feature_statistics",
            "metric_assets",
            "runtime",
        },
        "released2 primary screen validation",
    )
    normalized_artifact, screen = _verify_compact_receipt(
        artifact,
        "released2 primary screen",
    )
    primary = _require_number(
        validation["primary_metric"],
        "fresh-screened released2 FGD",
    )
    if (
        screen.get("format") != PRIMARY_SCREEN_FORMAT
        or validation["artifact"] != normalized_artifact
        or validation["primary_metric_path"] != PRIMARY_METRIC
        or not isinstance(validation["real_feature_statistics"], dict)
        or not isinstance(validation["generated_feature_statistics"], dict)
        or not isinstance(validation["metric_assets"], dict)
        or not isinstance(validation["runtime"], dict)
    ):
        raise PublishedWinnerClaimError(
            "released2 primary screen authority changed"
        )
    return normalized_artifact, primary, validation


def _validate_primary_screen_replay_receipt(
    artifact_value: Any,
    *,
    screen_artifact: Mapping[str, Any],
    screen_validation: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    distribution_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    """Validate the independent raw-NPZ replay used for ranking."""

    normalized_screen, screen = _verify_compact_receipt(
        screen_artifact,
        "released2 primary screen",
    )
    artifact = _exact_mapping(
        artifact_value,
        PAYLOAD_ARTIFACT_KEYS,
        "released2 primary screen replay artifact",
    )
    try:
        metric_adapter = _fresh_local_module(
            "evaluate_talkshow_show_metrics"
        )
        validation = (
            metric_adapter.validate_released2_primary_screen_replay_receipt(
                artifact,
                expected_screen_artifact=normalized_screen,
                expected_screen=screen,
                screen_validation=screen_validation,
                expected_prediction_manifest=prediction_manifest,
                expected_distribution_receipt=distribution_artifact,
                expected_selection_protocol={
                    "primary_metric": PRIMARY_METRIC,
                    "mode": "min",
                    "validation_only_for_selection": True,
                    "test_evaluations": 0,
                },
                expected_split="val",
                expected_clip_count=EXPECTED_VAL_CLIPS,
            )
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PublishedWinnerClaimError(
            "released2 primary screen raw replay verification failed"
        ) from error
    validation = _exact_mapping(
        validation,
        {
            "artifact",
            "receipt_payload_sha256",
            "primary_metric_path",
            "primary_metric",
            "report_payload_sha256",
            "prediction_manifest",
            "real_feature_cache",
            "metric_assets",
            "runtime",
        },
        "released2 primary screen replay validation",
    )
    normalized_artifact, replay = _verify_compact_receipt(
        artifact,
        "released2 primary screen raw replay",
    )
    primary = _require_number(
        validation["primary_metric"],
        "fresh-replayed released2 FGD",
    )
    if (
        replay.get("format")
        != "semtalk_show_released2_primary_fresh_replay_v1"
        or validation["artifact"] != normalized_artifact
        or validation["receipt_payload_sha256"]
        != normalized_artifact["receipt_payload_sha256"]
        or validation["primary_metric_path"] != PRIMARY_METRIC
        or validation["report_payload_sha256"]
        != normalized_screen["receipt_payload_sha256"]
        or validation["prediction_manifest"] != prediction_manifest
        or not isinstance(validation["metric_assets"], dict)
        or not isinstance(validation["runtime"], dict)
    ):
        raise PublishedWinnerClaimError(
            "released2 primary screen replay authority changed"
        )
    return normalized_artifact, primary


def _validate_winner_selection(
    artifact_value: Any,
    *,
    prerequisite_artifact: Mapping[str, Any],
    continuation_artifact: Mapping[str, Any],
    expected_real_feature_cache: Mapping[str, Any],
    require_fresh_transaction: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact, selection = _verify_compact_receipt(
        artifact_value, "Base validation winner selection"
    )
    _exact_mapping(
        selection,
        {
            "format",
            "payload_hash_algorithm",
            "status",
            "generator",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "selection_eligible",
            "primary_metric",
            "selection_policy",
            "prerequisite_selection",
            "continuation_decision",
            "real_feature_cache",
            "candidates",
            "selected",
            "test_policy",
            "receipt_payload_sha256",
        },
        "Base validation winner selection",
    )
    if (
        selection["format"]
        != (
            FRESH_WINNER_SELECTION_FORMAT
            if require_fresh_transaction
            else WINNER_SELECTION_FORMAT
        )
        or selection["payload_hash_algorithm"] != PAYLOAD_HASH_ALGORITHM
        or selection["status"] != "selected"
        or selection["generator"] != "SemTalk Base Motion Generation"
        or selection["dataset"] != "SHOW"
        or selection["target_speaker_scope"] != EXPECTED_SCOPE
        or selection["split"] != "val"
        or selection["test_visible"] is not False
        or selection["selection_eligible"] is not True
        or selection["primary_metric"] != PRIMARY_METRIC
        or selection["prerequisite_selection"] != prerequisite_artifact
        or selection["continuation_decision"] != continuation_artifact
        or selection["real_feature_cache"] != expected_real_feature_cache
        or selection["test_policy"] != TEST_POLICY
    ):
        raise PublishedWinnerClaimError(
            "Base validation winner selection identity changed"
        )
    updates_per_epoch = selection.get("selection_policy", {}).get(
        "updates_per_epoch"
    )
    expected_policy = {
        "candidate_epochs": list(BASE_CANDIDATE_EPOCHS),
        "updates_per_epoch": updates_per_epoch,
        "operator": "min",
        "ordering": [PRIMARY_METRIC, "epoch", "optimizer_updates"],
        "test_feedback_into_selection": False,
    }
    if (
        updates_per_epoch not in {62, 124, 248, 1988}
        or selection["selection_policy"] != expected_policy
    ):
        raise PublishedWinnerClaimError(
            "Base validation selection policy changed"
        )
    candidates = selection["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(
        BASE_CANDIDATE_EPOCHS
    ):
        raise PublishedWinnerClaimError(
            "Base validation selection candidate coverage mismatch"
        )
    common_row_keys = {
        "epoch",
        "optimizer_updates",
        "candidate_checkpoint",
        "prediction_manifest",
        "inference_lineage",
        "distribution_receipt",
        "body_released2_fgd",
    }
    replayed: list[tuple[float, int, int, dict[str, Any]]] = []
    checkpoint_paths: set[str] = set()
    for expected_epoch, raw_row in zip(BASE_CANDIDATE_EPOCHS, candidates):
        row_keys = set(common_row_keys)
        has_transaction = isinstance(raw_row, dict) and (
            "candidate_transaction" in raw_row
        )
        if require_fresh_transaction and not has_transaction:
            raise PublishedWinnerClaimError(
                f"Base e{expected_epoch} fresh candidate transaction is mandatory"
            )
        if has_transaction:
            row_keys.add("candidate_transaction")
        if require_fresh_transaction:
            row_keys.update(
                {"primary_screen_receipt", "primary_replay_receipt"}
            )
        else:
            row_keys.update(
                {"talkshow_metric_report", "primary_replay_receipt"}
            )
        row = _exact_mapping(
            raw_row, row_keys, f"Base e{expected_epoch} selection row"
        )
        epoch = _require_int(row["epoch"], "Base candidate epoch", minimum=1)
        updates = _require_int(
            row["optimizer_updates"],
            f"Base e{expected_epoch} optimizer updates",
            minimum=1,
        )
        if (
            epoch != expected_epoch
            or updates != epoch * updates_per_epoch
        ):
            raise PublishedWinnerClaimError(
                f"Base e{expected_epoch} candidate boundary changed"
            )
        checkpoint = _validate_checkpoint(
            row["candidate_checkpoint"], f"Base e{epoch} checkpoint"
        )
        if checkpoint["path"] in checkpoint_paths:
            raise PublishedWinnerClaimError(
                "Base candidate checkpoint path was reused"
            )
        checkpoint_paths.add(checkpoint["path"])
        prediction_manifest, _manifest_payload = _normalize_artifact(
            row["prediction_manifest"],
            f"Base e{epoch} prediction manifest",
            with_payload=False,
        )
        distribution_artifact, distribution = _validate_distribution(
            row["distribution_receipt"],
            prediction_manifest=prediction_manifest,
        )
        transaction_artifact: dict[str, Any] | None = None
        expected_canonical_manifest: dict[str, Any] | None = None
        if has_transaction:
            (
                transaction_artifact,
                _transaction,
                lineage_artifact,
                expected_canonical_manifest,
            ) = _validate_candidate_transaction(
                row["candidate_transaction"],
                epoch=epoch,
                updates=updates,
                checkpoint=checkpoint,
                prediction_manifest=prediction_manifest,
                distribution_artifact=distribution_artifact,
                distribution=distribution,
                prerequisite_artifact=prerequisite_artifact,
                continuation_artifact=continuation_artifact,
            )
            if lineage_artifact != row["inference_lineage"]:
                raise PublishedWinnerClaimError(
                    f"Base e{epoch} transaction lineage differs from row"
                )
        else:
            lineage_artifact, _lineage = _validate_lineage(
                row["inference_lineage"],
                epoch=epoch,
                updates=updates,
                checkpoint=checkpoint,
                prediction_manifest=prediction_manifest,
                distribution_artifact=distribution_artifact,
                prerequisite_artifact=prerequisite_artifact,
                continuation_artifact=continuation_artifact,
            )
        screen_artifact: dict[str, Any] | None = None
        report_artifact: dict[str, Any] | None = None
        replay_artifact: dict[str, Any] | None = None
        if require_fresh_transaction:
            if expected_canonical_manifest is None:
                raise PublishedWinnerClaimError(
                    f"Base e{epoch} fresh canonical authority is missing"
                )
            screen_artifact, screen_fgd, screen_validation = (
                _validate_primary_screen_receipt(
                    row["primary_screen_receipt"],
                    prediction_manifest=prediction_manifest,
                    distribution_artifact=distribution_artifact,
                    expected_real_feature_cache=expected_real_feature_cache,
                    expected_canonical_manifest=expected_canonical_manifest,
                )
            )
            replay_artifact, fgd = _validate_primary_screen_replay_receipt(
                row["primary_replay_receipt"],
                screen_artifact=screen_artifact,
                screen_validation=screen_validation,
                prediction_manifest=prediction_manifest,
                distribution_artifact=distribution_artifact,
            )
            if fgd != screen_fgd:
                raise PublishedWinnerClaimError(
                    f"Base e{epoch} raw replay differs from screen FGD"
                )
        else:
            report_artifact, report, report_validation = (
                _validate_metric_report(
                    row["talkshow_metric_report"],
                    prediction_manifest=prediction_manifest,
                    lineage_artifact=lineage_artifact,
                    distribution=distribution,
                    expected_canonical_manifest=expected_canonical_manifest,
                )
            )
            replay_artifact, fgd = _validate_primary_replay_receipt(
                row["primary_replay_receipt"],
                report=report,
                report_validation=report_validation,
                prediction_manifest=prediction_manifest,
                distribution=distribution,
                expected_real_feature_cache=expected_real_feature_cache,
                expected_canonical_manifest=expected_canonical_manifest,
            )
        if _require_number(
            row["body_released2_fgd"],
            f"Base e{epoch} declared released2 FGD",
        ) != fgd:
            raise PublishedWinnerClaimError(
                f"Base e{epoch} declared FGD differs from primary evidence"
            )
        normalized_row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "candidate_checkpoint": checkpoint,
            "prediction_manifest": prediction_manifest,
            "inference_lineage": lineage_artifact,
            "distribution_receipt": distribution_artifact,
            "body_released2_fgd": fgd,
        }
        if transaction_artifact is not None:
            normalized_row["candidate_transaction"] = transaction_artifact
        if require_fresh_transaction:
            normalized_row["primary_screen_receipt"] = screen_artifact
            normalized_row["primary_replay_receipt"] = replay_artifact
        else:
            normalized_row["talkshow_metric_report"] = report_artifact
            normalized_row["primary_replay_receipt"] = replay_artifact
        if normalized_row != row:
            raise PublishedWinnerClaimError(
                f"Base e{epoch} selection row is not canonical"
            )
        replayed.append((fgd, epoch, updates, normalized_row))
    winner = min(replayed, key=lambda item: (item[0], item[1], item[2]))[3]
    if selection["selected"] != winner:
        raise PublishedWinnerClaimError(
            "published Base winner differs from fresh released2 FGD replay"
        )
    _reject_forbidden_tree(selection, "Base winner selection")
    return artifact, selection, winner


def build_winner_full_metric_closure(
    *,
    winner_selection: Mapping[str, Any],
    prerequisite_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
    talkshow_metric_report: Mapping[str, Any],
    primary_replay_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exactly one full validation report to the selected screen winner."""

    prerequisite, prerequisite_payload, _fixed = (
        _validate_prerequisite_selection(prerequisite_selection)
    )
    continuation, _decision = _validate_continuation_decision(
        continuation_decision,
        prerequisite_artifact=prerequisite,
        prerequisite_selection=prerequisite_payload,
    )
    cache, _cache_payload = _verify_compact_receipt(
        real_feature_cache,
        "winner full-metric released2 real-feature cache",
    )
    selection, _selection_payload, winner = _validate_winner_selection(
        winner_selection,
        prerequisite_artifact=prerequisite,
        continuation_artifact=continuation,
        expected_real_feature_cache=cache,
        require_fresh_transaction=True,
    )
    epoch = winner["epoch"]
    checkpoint = winner["candidate_checkpoint"]
    prediction = winner["prediction_manifest"]
    distribution_artifact, distribution = _validate_distribution(
        winner["distribution_receipt"],
        prediction_manifest=prediction,
    )
    (
        transaction,
        _transaction_payload,
        lineage,
        canonical_manifest,
    ) = _validate_candidate_transaction(
        winner["candidate_transaction"],
        epoch=epoch,
        updates=winner["optimizer_updates"],
        checkpoint=checkpoint,
        prediction_manifest=prediction,
        distribution_artifact=distribution_artifact,
        distribution=distribution,
        prerequisite_artifact=prerequisite,
        continuation_artifact=continuation,
    )
    screen, screen_fgd, screen_validation = (
        _validate_primary_screen_receipt(
            winner["primary_screen_receipt"],
            prediction_manifest=prediction,
            distribution_artifact=distribution_artifact,
            expected_real_feature_cache=cache,
            expected_canonical_manifest=canonical_manifest,
        )
    )
    report, report_payload, report_validation = _validate_metric_report(
        talkshow_metric_report,
        prediction_manifest=prediction,
        lineage_artifact=lineage,
        distribution=distribution,
        expected_canonical_manifest=canonical_manifest,
    )
    replay, replay_fgd = _validate_primary_replay_receipt(
        primary_replay_receipt,
        report=report_payload,
        report_validation=report_validation,
        prediction_manifest=prediction,
        distribution=distribution,
        expected_real_feature_cache=cache,
        expected_canonical_manifest=canonical_manifest,
    )
    _replay_artifact, replay_payload = _verify_compact_receipt(
        replay,
        "winner full-metric released2 replay",
    )
    released2 = report_payload.get("body", {}).get("released2", {})
    report_statistics = released2.get("feature_statistics")
    report_metrics = released2.get("metrics")
    screen_real = screen_validation["real_feature_statistics"]
    screen_generated = screen_validation["generated_feature_statistics"]
    if (
        not isinstance(report_statistics, dict)
        or not isinstance(report_metrics, dict)
        or report_statistics.get("real") != screen_real
        or report_statistics.get("generated") != screen_generated
        or replay_payload.get("real_feature_statistics") != screen_real
        or replay_payload.get("generated_feature_statistics")
        != screen_generated
        or report_metrics.get("FGD") != screen_fgd
        or replay_payload.get("primary_metric") != screen_fgd
        or replay_fgd != screen_fgd
        or report_validation.get("primary_metric") != screen_fgd
    ):
        raise PublishedWinnerClaimError(
            "winner full metrics differ from primary screen exact moments/FGD"
        )
    if len(
        {
            transaction["path"],
            screen["path"],
            report["path"],
            replay["path"],
        }
    ) != 4:
        raise PublishedWinnerClaimError(
            "winner full-metric evidence artifact path was reused"
        )
    closure = {
        "format": WINNER_FULL_CLOSURE_FORMAT,
        "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "full_validation_evaluations": 1,
        "winner_selection": selection,
        "selected_epoch": epoch,
        "optimizer_updates": winner["optimizer_updates"],
        "candidate_checkpoint": checkpoint,
        "candidate_transaction": transaction,
        "prediction_manifest": prediction,
        "inference_lineage": lineage,
        "distribution_receipt": distribution_artifact,
        "primary_screen_receipt": screen,
        "talkshow_metric_report": report,
        "primary_replay_receipt": replay,
        "released2_equivalence": {
            "protocol": "exact_json_feature_moments_and_exact_fgd_v1",
            "real_feature_count": screen_real["count"],
            "generated_feature_count": screen_generated["count"],
            "real_moments_exact": True,
            "generated_moments_exact": True,
            "fgd_exact": True,
        },
        "body_released2_fgd": screen_fgd,
    }
    closure["receipt_payload_sha256"] = canonical_json_sha256(closure)
    return closure


def _validate_winner_full_metric_closure(
    artifact_value: Any,
    *,
    winner_selection: Mapping[str, Any],
    prerequisite_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    real_feature_cache: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, closure = _verify_compact_receipt(
        artifact_value,
        "winner full validation metric closure",
    )
    rebuilt = build_winner_full_metric_closure(
        winner_selection=winner_selection,
        prerequisite_selection=prerequisite_selection,
        continuation_decision=continuation_decision,
        real_feature_cache=real_feature_cache,
        talkshow_metric_report=closure.get("talkshow_metric_report", {}),
        primary_replay_receipt=closure.get("primary_replay_receipt", {}),
    )
    if closure != rebuilt:
        raise PublishedWinnerClaimError(
            "winner full validation metric closure changed"
        )
    return artifact, closure


def _expected_output_path(value: Any, label: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise PublishedWinnerClaimError(f"{label} must be path-like")
    path = Path(value)
    if not path.is_absolute():
        raise PublishedWinnerClaimError(f"{label} must be absolute")
    return str(path.resolve(strict=False))


def build_fresh_published_test_winner_claim(
    *,
    winner_selection: Mapping[str, Any],
    prerequisite_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    continuation_waves: Sequence[Mapping[str, Any]],
    real_feature_cache: Mapping[str, Any],
    winner_full_metric_closure: Mapping[str, Any],
    expected_output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the only claim accepted by the formal Base test authority.

    Every checkpoint and the fresh Base winner are recovered by replaying the
    externally pinned validation receipts.  The output path is an absent,
    absolute one-shot destination; no test artifact is read while publishing.
    """

    raw_output = Path(expected_output_root)
    if raw_output.exists() or raw_output.is_symlink():
        raise PublishedWinnerClaimError(
            "fresh published claim output root must be absent"
        )
    expected_root = _expected_output_path(
        raw_output, "fresh claim expected output root"
    )
    prerequisite_artifact, prerequisite_payload, fixed_checkpoints = (
        _validate_prerequisite_selection(prerequisite_selection)
    )
    continuation_artifact, _continuation_payload = (
        _validate_continuation_decision(
            continuation_decision,
            prerequisite_artifact=prerequisite_artifact,
            prerequisite_selection=prerequisite_payload,
        )
    )
    normalized_waves = _validate_continuation_waves(
        list(continuation_waves),
        prerequisite_selection=prerequisite_payload,
    )
    cache_artifact, _cache_payload = _verify_compact_receipt(
        real_feature_cache,
        "fresh published released2 real-feature cache",
    )
    winner_artifact, _winner_payload, winner = _validate_winner_selection(
        winner_selection,
        prerequisite_artifact=prerequisite_artifact,
        continuation_artifact=continuation_artifact,
        expected_real_feature_cache=cache_artifact,
        require_fresh_transaction=True,
    )
    full_closure_artifact, _full_closure = (
        _validate_winner_full_metric_closure(
            winner_full_metric_closure,
            winner_selection=winner_artifact,
            prerequisite_selection=prerequisite_artifact,
            continuation_decision=continuation_artifact,
            real_feature_cache=cache_artifact,
        )
    )
    claim: dict[str, Any] = {
        "format": FRESH_CLAIM_FORMAT,
        "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
        "status": "authorized",
        "generator": "SemTalk Base Motion Generation",
        "dataset": "SHOW",
        "target_speaker_scope": EXPECTED_SCOPE,
        "winner_selection": winner_artifact,
        "winner_full_metric_closure": full_closure_artifact,
        "prerequisite_selection": prerequisite_artifact,
        "continuation_decision": continuation_artifact,
        "continuation_waves": normalized_waves,
        "real_feature_cache": cache_artifact,
        "selected_base_checkpoint": dict(winner["candidate_checkpoint"]),
        "fixed_checkpoints": {
            stage: dict(fixed_checkpoints[stage]) for stage in STAGES
        },
        "expected_output_root": expected_root,
        "test_policy": dict(TEST_POLICY),
        "test_visible_during_selection": False,
    }
    _reject_forbidden_tree(claim, "fresh published winner claim")
    claim["receipt_payload_sha256"] = canonical_json_sha256(claim)
    return claim


def validate_published_test_winner_claim(
    claim_path: str | os.PathLike[str],
    *,
    expected_claim_sha256: str,
    expected_claim_bytes: int,
    expected_claim_payload_sha256: str,
    expected_output_root: str | os.PathLike[str],
    prerequisite_selection: Mapping[str, Any],
    continuation_decision: Mapping[str, Any],
    continuation_waves: Sequence[Mapping[str, Any]],
    winner_full_metric_closure: Mapping[str, Any] | None = None,
    expected_claim_format: str = CLAIM_FORMAT,
) -> dict[str, Any]:
    """Validate and return one immutable, one-shot SHOW test authorization.

    ``prerequisite_selection``, ``continuation_decision`` and every
    ``continuation_waves`` artifact are externally pinned.  No checkpoint
    mapping is accepted from the caller: all six checkpoints are recovered
    from fresh receipt replay.
    """

    claim_file, claim_payload = _safe_file_snapshot(
        claim_path,
        "published winner claim",
    )
    expected_file_sha = _require_sha256(
        expected_claim_sha256, "published claim file SHA-256"
    )
    expected_size = _require_int(
        expected_claim_bytes, "published claim bytes", minimum=1
    )
    expected_payload_sha = _require_sha256(
        expected_claim_payload_sha256,
        "published claim payload SHA-256",
    )
    if (
        _sha256_bytes(claim_payload) != expected_file_sha
        or len(claim_payload) != expected_size
    ):
        raise PublishedWinnerClaimError("published claim artifact changed")
    claim = _strict_json_bytes(claim_payload, "published winner claim")
    claim_keys = {
        "format",
        "payload_hash_algorithm",
        "status",
        "generator",
        "dataset",
        "target_speaker_scope",
        "winner_selection",
        "prerequisite_selection",
        "continuation_decision",
        "continuation_waves",
        "real_feature_cache",
        "selected_base_checkpoint",
        "fixed_checkpoints",
        "expected_output_root",
        "test_policy",
        "test_visible_during_selection",
        "receipt_payload_sha256",
    }
    if expected_claim_format == FRESH_CLAIM_FORMAT:
        claim_keys.add("winner_full_metric_closure")
    _exact_mapping(
        claim,
        claim_keys,
        "published winner claim",
    )
    embedded_payload_sha = _require_sha256(
        claim["receipt_payload_sha256"],
        "published claim embedded payload SHA-256",
    )
    unsigned_claim = dict(claim)
    unsigned_claim.pop("receipt_payload_sha256")
    if (
        canonical_json_sha256(unsigned_claim) != embedded_payload_sha
        or embedded_payload_sha != expected_payload_sha
    ):
        raise PublishedWinnerClaimError("published claim payload hash mismatch")
    expected_root = _expected_output_path(
        expected_output_root, "expected output root"
    )
    if expected_claim_format not in {CLAIM_FORMAT, FRESH_CLAIM_FORMAT}:
        raise PublishedWinnerClaimError(
            "externally expected claim format is unsupported"
        )
    if (
        claim["format"] != expected_claim_format
        or claim["payload_hash_algorithm"] != PAYLOAD_HASH_ALGORITHM
        or claim["status"] != "authorized"
        or claim["generator"] != "SemTalk Base Motion Generation"
        or claim["dataset"] != "SHOW"
        or claim["target_speaker_scope"] != EXPECTED_SCOPE
        or claim["expected_output_root"] != expected_root
        or claim["test_policy"] != TEST_POLICY
        or claim["test_visible_during_selection"] is not False
    ):
        raise PublishedWinnerClaimError("published claim identity changed")
    prerequisite_artifact, prerequisite_payload, fixed_checkpoints = (
        _validate_prerequisite_selection(prerequisite_selection)
    )
    continuation_artifact, _continuation_payload = (
        _validate_continuation_decision(
            continuation_decision,
            prerequisite_artifact=prerequisite_artifact,
            prerequisite_selection=prerequisite_payload,
        )
    )
    normalized_waves = _validate_continuation_waves(
        list(continuation_waves),
        prerequisite_selection=prerequisite_payload,
    )
    real_feature_cache, _cache_payload = _normalize_artifact(
        claim["real_feature_cache"],
        "published released2 real-feature cache",
        with_payload=True,
    )
    winner_artifact, _winner_payload, winner = _validate_winner_selection(
        claim["winner_selection"],
        prerequisite_artifact=prerequisite_artifact,
        continuation_artifact=continuation_artifact,
        expected_real_feature_cache=real_feature_cache,
        require_fresh_transaction=(
            expected_claim_format == FRESH_CLAIM_FORMAT
        ),
    )
    full_closure_artifact: dict[str, Any] | None = None
    if expected_claim_format == FRESH_CLAIM_FORMAT:
        if winner_full_metric_closure is None:
            raise PublishedWinnerClaimError(
                "fresh published claim requires externally pinned winner full metrics"
            )
        full_closure_artifact, _full_closure = (
            _validate_winner_full_metric_closure(
                winner_full_metric_closure,
                winner_selection=winner_artifact,
                prerequisite_selection=prerequisite_artifact,
                continuation_decision=continuation_artifact,
                real_feature_cache=real_feature_cache,
            )
        )
    if (
        claim["prerequisite_selection"] != prerequisite_artifact
        or claim["continuation_decision"] != continuation_artifact
        or claim["continuation_waves"] != normalized_waves
        or claim["winner_selection"] != winner_artifact
        or claim["selected_base_checkpoint"]
        != winner["candidate_checkpoint"]
        or claim["fixed_checkpoints"] != fixed_checkpoints
        or (
            expected_claim_format == FRESH_CLAIM_FORMAT
            and claim["winner_full_metric_closure"]
            != full_closure_artifact
        )
    ):
        raise PublishedWinnerClaimError(
            "published claim differs from freshly derived checkpoint authority"
        )
    _reject_forbidden_tree(claim, "published winner claim")
    claim_artifact = {
        "path": str(claim_file),
        "sha256": expected_file_sha,
        "bytes": expected_size,
    }
    validated = {
        "claim_artifact": claim_artifact,
        "receipt_payload_sha256": embedded_payload_sha,
        "winner_selection": winner_artifact,
        "selected_base_checkpoint": dict(winner["candidate_checkpoint"]),
        "fixed_checkpoints": {
            stage: dict(fixed_checkpoints[stage]) for stage in STAGES
        },
        "continuation_waves": normalized_waves,
        "expected_output_root": expected_root,
        "test_policy": dict(TEST_POLICY),
    }
    if expected_claim_format == FRESH_CLAIM_FORMAT:
        validated["winner_full_metric_closure"] = full_closure_artifact
    return validated


__all__ = [
    "AUTHORIZATION_FORMAT",
    "BASE_CANDIDATE_EPOCHS",
    "BASE_UPDATES_PER_EPOCH",
    "CLAIM_FORMAT",
    "CONTINUATION_DECISION_FORMAT",
    "DISTRIBUTION_FORMAT",
    "EXPECTED_SCOPE",
    "FRESH_CLAIM_FORMAT",
    "FRESH_WINNER_SELECTION_FORMAT",
    "INITIAL_PREREQUISITE_SELECTION_FORMAT",
    "METRIC_REPORT_FORMAT",
    "PAYLOAD_HASH_ALGORITHM",
    "PREREQUISITE_SELECTION_FORMAT",
    "PublishedWinnerClaimError",
    "STAGES",
    "TEST_POLICY",
    "VAL_LINEAGE_FORMAT",
    "WINNER_SELECTION_FORMAT",
    "build_fresh_published_test_winner_claim",
    "canonical_json_sha256",
    "validate_published_test_winner_claim",
]
