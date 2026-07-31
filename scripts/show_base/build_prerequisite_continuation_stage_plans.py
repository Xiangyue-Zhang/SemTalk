#!/usr/bin/env python3
"""Build deterministic active-stage plans for one prerequisite +20 wave.

The operator first asks ``show_base_train.py`` to print the formal config
receipt for both the immutable old run (first wave only) and each proposed new
run.  This builder then replays the continuation decision, selected bridge,
candidate index and optional predecessor wave, verifies those config receipts,
derives the segment chain, and publishes the exact adapter JSON consumed by
``publish_prerequisite_continuation_wave.py``.

No checkpoint is copied and no GPU process is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from scripts.show_base import decide_prerequisite_continuation as decision_mod
from scripts.show_base import prerequisite_continuation_runtime as runtime
from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract
from scripts.show_base import publish_prerequisite_continuation_wave as publisher
from scripts.show_base import selected_prerequisites


CONFIG_KEYS = {
    "format",
    "formal_stage",
    "hostname",
    "world_size",
    "smplx_asset",
    "config_snapshot",
    "config_sha256",
    "config_semantic_sha256",
    "source_receipt",
    "source_receipt_sha256",
    "receipt_payload_sha256",
}


class StagePlanBuildError(RuntimeError):
    """Raised when a continuation plan cannot be derived exactly."""


def _parse_stage_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise StagePlanBuildError(f"{label} must be STAGE=/absolute/path")
        stage, raw_path = raw.split("=", 1)
        if stage not in wave.STAGES or stage in result:
            raise StagePlanBuildError(f"invalid or duplicate stage {stage!r}")
        path = Path(raw_path)
        if not path.is_absolute() or ".." in path.parts:
            raise StagePlanBuildError(f"{stage} {label} must be absolute")
        result[stage] = path
    return result


def _parse_stage_shas(values: Sequence[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise StagePlanBuildError(f"{label} must be STAGE=SHA256")
        stage, value = raw.split("=", 1)
        if stage not in wave.STAGES or stage in result:
            raise StagePlanBuildError(f"invalid or duplicate stage {stage!r}")
        result[stage] = wave._require_sha256(value, f"{stage} {label}")
    return result


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise StagePlanBuildError(f"{label} must be absolute")
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise StagePlanBuildError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise StagePlanBuildError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise StagePlanBuildError(f"{label} is not canonical")
    return resolved


def _load_config(path: Path, expected_sha: str, *, stage: str, label: str) -> dict[str, Any]:
    resolved = _regular(path, label)
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != wave._require_sha256(
        expected_sha, f"{label} file SHA-256"
    ):
        raise StagePlanBuildError(f"{label} changed")
    try:
        value = contract.strict_json_bytes(payload, str(resolved))
    except contract.ContractError as error:
        raise StagePlanBuildError(str(error)) from error
    if not isinstance(value, dict) or set(value) != CONFIG_KEYS:
        raise StagePlanBuildError(f"{label} schema mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_payload_sha256")
    if (
        value["format"] != runtime.CONFIG_RECEIPT_FORMAT
        or value["formal_stage"] != stage
        or value["world_size"] != wave._topology(stage)["world_size"]
        or value["config_sha256"]
        != hashlib.sha256(
            json.dumps(
                value["config_snapshot"],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        or value["config_semantic_sha256"]
        != runtime.config_semantic_sha256(value["config_snapshot"])
        or value["source_receipt_sha256"]
        != contract.canonical_payload_sha256(value["source_receipt"])
        or claimed != contract.canonical_payload_sha256(unsigned)
    ):
        raise StagePlanBuildError(f"{label} payload mismatch")
    try:
        source = contract.validate_training_audit_source(
            value["source_receipt"], label, reprove_entrypoint=False
        )
        host = wave._validate_host(value["hostname"], f"{label} hostname")
        asset = wave._validate_smplx_asset(
            value["smplx_asset"], stage=stage, host=host, label=f"{label} asset"
        )
    except (contract.ContractError, wave.ContinuationWaveError) as error:
        raise StagePlanBuildError(str(error)) from error
    return {**value, "source_receipt": source, "hostname": host, "smplx_asset": asset}


def _decision_index(
    decision_path: Path,
    decision_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        decision = decision_mod.replay_decision(decision_path, decision_sha)
        selection = wave._validate_payload_binding(
            decision["inputs"]["selection"], "decision selection"
        )
        bridge = selected_prerequisites.load_selected_prerequisites(
            Path(selection["path"]), selection["sha256"]
        )
        index_binding = wave._validate_payload_binding(
            bridge["candidate_index_receipt"], "selected candidate index"
        )
        index, artifact = contract.load_candidate_index(
            Path(index_binding["path"]), index_binding["sha256"]
        )
    except Exception as error:
        raise StagePlanBuildError(f"decision/index replay failed: {error}") from error
    if (
        artifact["path"] != index_binding["path"]
        or artifact["sha256"] != index_binding["sha256"]
        or index["receipt_payload_sha256"]
        != index_binding["receipt_payload_sha256"]
    ):
        raise StagePlanBuildError("candidate index binding changed")
    return decision, bridge, index


def _source_from_frozen(value: Mapping[str, Any], stage: str) -> dict[str, Any]:
    portable = value.get("portable_identity")
    if not isinstance(portable, dict):
        raise StagePlanBuildError(f"{stage} frozen source lacks portable identity")
    return wave._validate_source(
        {
            "commit": portable.get("commit"),
            "tree": portable.get("tree"),
            "source_receipt_sha256": value.get("receipt_payload_sha256"),
        },
        f"{stage} frozen source",
    )


def _source_from_config(value: Mapping[str, Any], stage: str) -> dict[str, Any]:
    source = value["source_receipt"]
    return wave._validate_source(
        {
            "commit": source.get("commit"),
            "tree": source.get("tree"),
            "source_receipt_sha256": value["source_receipt_sha256"],
        },
        f"{stage} new config source",
    )


def _first_wave_old(
    *,
    stage: str,
    boundary: int,
    index: Mapping[str, Any],
    old_config: Mapping[str, Any],
) -> dict[str, Any]:
    entries = index["stages"][stage]
    boundary_entries = [entry for entry in entries if entry["epoch"] == boundary]
    if len(boundary_entries) != 1:
        raise StagePlanBuildError(f"{stage} boundary candidate is not exact-once")
    candidate = wave._catalog_candidate_from_index(
        boundary_entries[0], stage=stage, index=entries.index(boundary_entries[0])
    )
    old_run = str(Path(candidate["checkpoint"]["path"]).parent.parent)
    frozen = index["source_receipts"][stage]
    old_source = _source_from_frozen(frozen, stage)
    if (
        old_config["config_snapshot"].get("epochs") != boundary
        or old_config["config_sha256"] != index["config_sha256"][stage]
        or old_config["source_receipt"] != frozen.get("training_audit")
        or {
            "commit": old_config["source_receipt"].get("commit"),
            "tree": old_config["source_receipt"].get("tree"),
        }
        != {
            "commit": old_source["commit"],
            "tree": old_source["tree"],
        }
    ):
        raise StagePlanBuildError(f"{stage} old config differs from candidate index")
    status_artifact = index["formal_training_status"][stage]
    runtime_evidence = wave._load_old_runtime_evidence(
        stage=stage,
        boundary=boundary,
        old_run_path=old_run,
        old_source_receipt=frozen,
        old_config_sha256=index["config_sha256"][stage],
        old_dataset_receipt_sha256=index["dataset_receipt_sha256"][stage],
        formal_status_artifact=status_artifact,
        boundary_candidate=candidate,
    )
    if (
        old_config["hostname"] != runtime_evidence["old_host"]
        or old_config["smplx_asset"]
        != runtime_evidence["old_smplx_asset"]
    ):
        raise StagePlanBuildError(
            f"{stage} old config host/asset differ from runtime"
        )
    status_path = _regular(Path(status_artifact["path"]), f"{stage} status")
    status = contract.strict_json_bytes(status_path.read_bytes(), str(status_path))
    dataset_semantic = runtime.dataset_semantic_sha256(status["dataset_receipt"])
    chain = [
        wave._make_chain_segment(
            run_path=old_run,
            start_epoch=wave.INTERVAL_EPOCHS,
            end_epoch=boundary,
            predecessor_segment_id=None,
        )
    ]
    return {
        "old_run_path": old_run,
        "old_source": old_source,
        "old_host": runtime_evidence["old_host"],
        "old_smplx_asset": runtime_evidence["old_smplx_asset"],
        "old_config_sha256": index["config_sha256"][stage],
        "old_config_semantic_sha256": old_config["config_semantic_sha256"],
        "old_dataset_semantic_sha256": dataset_semantic,
        "candidate_segment_chain": chain,
        "predecessor_wave": None,
    }


def _later_wave_old(
    *,
    stage: str,
    boundary: int,
    predecessor: Mapping[str, Any],
    predecessor_binding: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        entry
        for entry in predecessor["stages"]
        if entry["stage"] == stage
    ]
    previous_target = (
        matches[0].get("target_epoch")
        if len(matches) == 1
        and predecessor.get("format") == wave.PER_STAGE_FORMAT
        else predecessor.get("target_epoch")
    )
    if len(matches) != 1 or previous_target != boundary:
        raise StagePlanBuildError(f"{stage} predecessor boundary mismatch")
    previous = matches[0]
    old = previous["old_segment"]
    new = previous["new_segment"]
    appended = wave._make_chain_segment(
        run_path=new["run_path"],
        start_epoch=boundary,
        end_epoch=boundary,
        predecessor_segment_id=new["predecessor_segment_id"],
    )
    if appended["segment_id"] != new["chain_segment_id"]:
        raise StagePlanBuildError(f"{stage} predecessor segment identity changed")
    return {
        "old_run_path": new["run_path"],
        "old_source": new["source"],
        "old_host": new["host"],
        "old_smplx_asset": new["smplx_asset"],
        "old_config_sha256": new["config_sha256"],
        "old_config_semantic_sha256": new["config_semantic_sha256"],
        "old_dataset_semantic_sha256": new["dataset_semantic_sha256"],
        "candidate_segment_chain": [*old["candidate_segment_chain"], appended],
        "predecessor_wave": dict(predecessor_binding),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    new_runs = _parse_stage_paths(args.new_run, "new run")
    new_configs = _parse_stage_paths(args.new_config_receipt, "new config receipt")
    new_config_shas = _parse_stage_shas(
        args.expected_new_config_receipt_sha256, "new config receipt SHA-256"
    )
    old_configs = _parse_stage_paths(args.old_config_receipt or [], "old config receipt")
    old_config_shas = _parse_stage_shas(
        args.expected_old_config_receipt_sha256 or [], "old config receipt SHA-256"
    )
    decision, _bridge, index = _decision_index(
        args.decision_json, args.expected_decision_sha256
    )
    active, _decision_payload_sha = wave._validate_per_stage_decision(
        decision
    )
    required = set(active)
    if (
        set(new_runs) != required
        or set(new_configs) != required
        or set(new_config_shas) != required
    ):
        raise StagePlanBuildError(
            "new run/config inputs must cover exactly active stages"
        )

    predecessor = None
    predecessor_binding = None
    if args.predecessor_wave is not None:
        if args.expected_predecessor_wave_sha256 is None:
            raise StagePlanBuildError("predecessor wave SHA-256 is required")
        predecessor = wave.replay_wave_file(
            args.predecessor_wave, args.expected_predecessor_wave_sha256
        )
        predecessor_binding = {
            "path": str(args.predecessor_wave.resolve(strict=True)),
            "sha256": args.expected_predecessor_wave_sha256,
            "receipt_payload_sha256": predecessor["receipt_payload_sha256"],
        }
        if old_configs or old_config_shas:
            raise StagePlanBuildError("later waves derive old config from predecessor")
    elif set(old_configs) != required or set(old_config_shas) != required:
        raise StagePlanBuildError(
            "first wave requires old config receipts for exactly active stages"
        )

    plans: dict[str, Any] = {}
    for stage in wave.STAGES:
        if stage not in active:
            continue
        boundary = active[stage]["boundary_epoch"]
        new_config = _load_config(
            new_configs[stage],
            new_config_shas[stage],
            stage=stage,
            label=f"{stage} new config",
        )
        target = active[stage]["target_epoch"]
        if new_config["config_snapshot"].get("epochs") != target:
            raise StagePlanBuildError(
                f"{stage} new config epochs differ from authorized target"
            )
        old = (
            _later_wave_old(
                stage=stage,
                boundary=boundary,
                predecessor=predecessor,
                predecessor_binding=predecessor_binding,
            )
            if predecessor is not None
            else _first_wave_old(
                stage=stage,
                boundary=boundary,
                index=index,
                old_config=_load_config(
                    old_configs[stage], old_config_shas[stage], stage=stage, label=f"{stage} old config"
                ),
            )
        )
        if (
            new_config["config_semantic_sha256"]
            != old["old_config_semantic_sha256"]
        ):
            raise StagePlanBuildError(f"{stage} config semantics changed")
        new_source = _source_from_config(new_config, stage)
        new_run = new_runs[stage]
        if new_run.exists() or new_run.is_symlink():
            raise StagePlanBuildError(f"{stage} new run path must be absent/nonreused")
        plans[stage] = {
            "stage": stage,
            **old,
            "new_run_path": str(new_run),
            "new_source": new_source,
            "new_host": new_config["hostname"],
            "new_smplx_asset": new_config["smplx_asset"],
            "new_source_repository": str(args.new_source_repository),
            "new_config_sha256": new_config["config_sha256"],
            "new_config_semantic_sha256": new_config[
                "config_semantic_sha256"
            ],
            # Continuation permits only enumerated source-bound dataset fields
            # to change, so the new semantic hash is exactly the old one.
            "new_dataset_semantic_sha256": old[
                "old_dataset_semantic_sha256"
            ],
        }
        if set(plans[stage]) != wave.ADAPTER_STAGE_PLAN_KEYS:
            raise StagePlanBuildError(f"{stage} adapter plan schema mismatch")

    result = {
        "format": publisher.FORMAT,
        "status": "complete",
        "test_visible": False,
        "stages": plans,
    }
    result["receipt_payload_sha256"] = wave.canonical_json_sha256(result)
    # Reuse the publisher-side parser before exposing the artifact.
    contract.atomic_json_new(args.output_json, result)
    observed_sha = contract.sha256_file(args.output_json)
    loaded, _artifact = publisher._load_stage_plans(args.output_json, observed_sha)
    if loaded != plans:
        raise StagePlanBuildError("published stage plans differ from replay")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--decision-json", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--new-source-repository", type=Path, required=True)
    parser.add_argument("--new-run", action="append", required=True)
    parser.add_argument("--new-config-receipt", action="append", required=True)
    parser.add_argument(
        "--expected-new-config-receipt-sha256", action="append", required=True
    )
    parser.add_argument("--old-config-receipt", action="append")
    parser.add_argument("--expected-old-config-receipt-sha256", action="append")
    parser.add_argument("--predecessor-wave", type=Path)
    parser.add_argument("--expected-predecessor-wave-sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build(_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "stages": list(result["stages"]),
                "receipt_payload_sha256": result["receipt_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
