#!/usr/bin/env python3
"""Run canonical SHOW validation for provisional Base e1/e2/e4/e8/e16/e32.

This adapter is deliberately separate from the formal 22-candidate/e400
producer.  It validates the trainer-owned short-quality publications and then
delegates only shard execution/finalization to the unchanged formal inference
engine.  Its preflight schema cannot be consumed by the formal CLI.

Only SHOW ``val`` is accepted.  There is no test argument or fallback path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_long_val_contract as formal_validation
from scripts.show_base import run_base_val_inference as engine
from scripts.show_base import select_base_training_topology as topology
from scripts.show_base import train_base_official_adapt_long as training

FORMAT = "semtalk_show_base_short_quality_val_preflight_v2"
COMPLETION_FORMAT = "semtalk_show_base_short_quality_val_completion_v2"
QUALITY_EPOCHS = topology.QUALITY_EPOCHS
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SOURCE_FILES = (
    "scripts/show_base/base_short_quality_val_adapter.py",
    "scripts/show_base/base_short_quality_val_8shard.sh",
    "scripts/show_base/run_base_val_inference.py",
    "scripts/show_base/semtalk_base_inference_core.py",
    "scripts/show_base/base_long_val_contract.py",
    "scripts/show_base/select_base_official_adapt.py",
    "scripts/show_base/evaluate_diffsheg_val_fgd.py",
)


class ShortQualityValError(RuntimeError):
    """Raised when provisional validation is not fully bound."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _with_payload_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_payload_sha256"] = _payload_sha(result)
    return result


def _sha(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
        raise ShortQualityValError(f"{label} must be lowercase SHA-256")
    return value


def _epoch(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch must be an integer") from error
    if parsed not in QUALITY_EPOCHS:
        choices = ",".join(str(epoch) for epoch in QUALITY_EPOCHS)
        raise argparse.ArgumentTypeError(f"epoch must be one of {choices}")
    return parsed


def _run_git(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise ShortQualityValError(
            "cannot audit adapter source: "
            + process.stderr.decode("utf-8", errors="replace").strip()
        )
    return process.stdout.decode("utf-8").strip()


def _adapter_source_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if root != PROJECT_ROOT.resolve(strict=True):
        raise ShortQualityValError("adapter must execute from its own source root")
    origin = _run_git(root, "remote", "get-url", "origin")
    push_origin = _run_git(root, "remote", "get-url", "--push", "origin")
    commit = _run_git(root, "rev-parse", "HEAD")
    tree = _run_git(root, "rev-parse", "HEAD^{tree}")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    heads = [
        line
        for line in _run_git(
            root,
            "for-each-ref",
            "--format=%(objectname) %(refname:short)",
            "refs/heads",
        ).splitlines()
        if line.startswith(commit + " ")
    ]
    if (
        origin != EXPECTED_ORIGIN
        or push_origin != EXPECTED_ORIGIN
        or status
        or heads
        or _run_git(root, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD"
    ):
        raise ShortQualityValError(
            "adapter source must be clean, detached, branchless at HEAD, "
            "and use the sole SemTalk origin"
        )
    files: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = (root / relative).resolve(strict=True)
        if path != root / relative or path.is_symlink() or not path.is_file():
            raise ShortQualityValError(f"invalid adapter source file {relative}")
        payload = path.read_bytes()
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            check=False,
            capture_output=True,
        )
        if committed.returncode != 0 or committed.stdout != payload:
            raise ShortQualityValError(
                f"adapter source file is untracked or differs from HEAD: {relative}"
            )
        files[relative] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return {
        "origin": origin,
        "source_root": str(root),
        "commit": commit,
        "tree": tree,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "files": files,
    }


def _ready_inputs(
    ready_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _artifact, first = topology._artifact(
        ready_artifacts[0], "e1 candidate-ready", payload=True
    )
    assert first is not None
    frozen_value = first["frozen_inputs"]
    frozen, _path, _sha256 = training._load_json_receipt(
        Path(frozen_value["path"]),
        frozen_value["sha256"],
        "short-quality frozen inputs",
    )
    manifest_path = Path(first["candidate_manifest"]["path"])
    manifest, _manifest_path, _manifest_sha = training._load_json_receipt(
        manifest_path,
        first["candidate_manifest"]["sha256_at_ready"],
        "short-quality ready manifest",
    )
    return first, frozen, manifest


def _selected_five(
    frozen: Mapping[str, Any], pipeline: Mapping[str, Any]
) -> dict[str, str]:
    dataset = frozen.get("dataset")
    selected = (
        dataset.get("selected_prerequisite_sha256")
        if isinstance(dataset, dict)
        else None
    )
    selection = (
        dataset.get("prerequisite_selection") if isinstance(dataset, dict) else None
    )
    expected = {
        stage: pipeline["fixed_checkpoints"][stage]["sha256"]
        for stage in ("face", "hands", "upper", "lower", "global")
    }
    if (
        selected != expected
        or not isinstance(selection, dict)
        or selection.get("sha256") != pipeline["prerequisite_selection"]["sha256"]
    ):
        raise ShortQualityValError(
            "short-quality training features differ from the validation-selected five"
        )
    return expected


def _normalize_ready(values: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    epochs: list[int] = []
    result: list[dict[str, Any]] = []
    for raw_epoch, raw_path, raw_sha in values:
        epoch = _epoch(raw_epoch)
        epochs.append(epoch)
        path = Path(raw_path)
        if not path.is_absolute():
            raise ShortQualityValError("candidate-ready path must be absolute")
        payload = path.read_bytes()
        parsed = json.loads(payload)
        result.append(
            {
                "path": str(path),
                "sha256": _sha(raw_sha, f"e{epoch} candidate-ready SHA"),
                "bytes": len(payload),
                "receipt_payload_sha256": _sha(
                    parsed.get("receipt_payload_sha256"),
                    f"e{epoch} candidate-ready payload SHA",
                ),
            }
        )
    if tuple(epochs) != QUALITY_EPOCHS:
        raise ShortQualityValError(
            "candidate-ready epochs must be e1/e2/e4/e8/e16/e32 in order"
        )
    return result


def _status_artifact(path: Path, expected_sha: str) -> dict[str, Any]:
    payload = path.read_bytes()
    value = json.loads(payload)
    return {
        "path": str(path),
        "sha256": _sha(expected_sha, "short-quality status SHA"),
        "bytes": len(payload),
        "receipt_payload_sha256": _sha(
            value.get("receipt_payload_sha256"),
            "short-quality status payload SHA",
        ),
    }


def _build_bound_payload(
    *,
    mode: str,
    topology_gate_path: Path,
    topology_gate_sha: str,
    quality_gate_path: Path,
    quality_gate_sha: str,
    status_artifact: Mapping[str, Any],
    ready_artifacts: Sequence[Mapping[str, Any]],
    val_inputs_path: Path,
    val_inputs_sha: str,
    pipeline_path: Path,
    pipeline_sha: str,
) -> dict[str, Any]:
    topology_gate = training.validate_topology_gate_spec(
        SimpleNamespace(
            topology_gate_spec=topology_gate_path,
            expected_topology_gate_spec_sha256=topology_gate_sha,
            topology_mode=mode,
        )
    )
    quality_gate = topology.validate_quality_gate_spec(
        quality_gate_path, quality_gate_sha
    )
    status, ready, semantic_sha, checkpoints = (
        topology.validate_short_quality_training_bundle(
            mode,
            status_artifact,
            ready_artifacts,
            topology_gate_spec_sha256=topology_gate["sha256"],
            quality_gate_spec_sha256=quality_gate["sha256"],
        )
    )
    val_artifact, coverage = formal_validation.validate_val_inputs(
        val_inputs_path, val_inputs_sha
    )
    pipeline_artifact, pipeline = formal_validation.validate_pipeline(
        pipeline_path, pipeline_sha
    )
    first, frozen, ready_manifest = _ready_inputs(ready)
    selected = _selected_five(frozen, pipeline)
    source = _adapter_source_receipt(Path(pipeline["source"]["source_root"]))
    if any(
        source[key] != pipeline["source"][key]
        for key in (
            "origin",
            "source_root",
            "commit",
            "tree",
            "clean",
            "detached",
            "local_branches_at_commit",
        )
    ):
        raise ShortQualityValError(
            "adapter and inference pipeline mix source checkouts"
        )
    frozen_source = frozen.get("source")
    if (
        not isinstance(frozen_source, dict)
        or frozen_source.get("origin") != EXPECTED_ORIGIN
    ):
        raise ShortQualityValError("training source is not the official SemTalk origin")
    frozen_value = first["frozen_inputs"]
    status_payload = json.loads(Path(status["path"]).read_text(encoding="utf-8"))
    bundle = {
        "manifest": dict(status_payload["candidate_manifest"]),
        "status": dict(status),
        "frozen_inputs": {
            "path": frozen_value["path"],
            "sha256": frozen_value["sha256"],
            "receipt_sha256": frozen_value["receipt_payload_sha256"],
        },
        "updates_per_epoch": int(training.TOPOLOGY_SPECS[mode]["updates_per_epoch"]),
        "candidates": {
            str(epoch): dict(checkpoints[epoch]) for epoch in QUALITY_EPOCHS
        },
    }
    authority = {
        "topology_mode": mode,
        "topology_gate_spec": {
            "path": topology_gate["path"],
            "sha256": topology_gate["sha256"],
            "payload_sha256": topology_gate["payload_sha256"],
        },
        "quality_gate_spec": {
            "path": quality_gate["path"],
            "sha256": quality_gate["sha256"],
        },
        "short_quality_status": dict(status),
        "candidate_ready_receipts": list(ready),
        "topology_independent_input_sha256": semantic_sha,
        "selected_prerequisite_sha256": selected,
        "throughput_gate": dict(ready_manifest["throughput_gate"]),
        "training_source": dict(frozen_source),
    }
    binding_sha = training.canonical_json_sha256(
        {
            "candidate_bundle": bundle,
            "authority": authority,
            "val_inputs_receipt": val_artifact,
            "pipeline_receipt": pipeline_artifact,
            "adapter_source": source,
        }
    )
    return {
        "format": FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "candidate_epochs": list(QUALITY_EPOCHS),
        "candidate_bundle": bundle,
        "val_inputs_receipt": val_artifact,
        "pipeline_receipt": pipeline_artifact,
        "pipeline_source": pipeline["source"],
        "inference_entrypoint": pipeline["inference_entrypoint"],
        "coverage": engine.selector.public_val_coverage(coverage),
        "short_quality_authority": authority,
        "adapter_source": source,
        "quality_input_binding_sha256": binding_sha,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output.is_absolute() or os.path.lexists(args.output):
        raise FileExistsError(f"refusing non-new absolute output {args.output}")
    ready = _normalize_ready(args.candidate_ready_receipt)
    status = _status_artifact(
        args.short_quality_status, args.expected_short_quality_status_sha256
    )
    payload = _with_payload_sha(
        _build_bound_payload(
            mode=args.mode,
            topology_gate_path=args.topology_gate_spec,
            topology_gate_sha=args.expected_topology_gate_spec_sha256,
            quality_gate_path=args.quality_gate_spec,
            quality_gate_sha=args.expected_quality_gate_spec_sha256,
            status_artifact=status,
            ready_artifacts=ready,
            val_inputs_path=args.val_inputs,
            val_inputs_sha=args.expected_val_inputs_sha256,
            pipeline_path=args.pipeline,
            pipeline_sha=args.expected_pipeline_sha256,
        )
    )
    engine._write_new(args.output, _canonical_bytes(payload))
    return engine._artifact(args.output, payload_sha=payload["receipt_payload_sha256"])


def validate(args: argparse.Namespace) -> dict[str, Any]:
    artifact, _payload = _preflight_artifact(
        args.preflight, args.expected_preflight_sha256
    )
    return artifact


def _preflight_artifact(
    path: Path, expected_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = engine._regular_file(path, "short-quality validation preflight")
    payload = engine._verified_json(
        resolved, expected_sha, "short-quality validation preflight"
    )
    expected_keys = {
        "format",
        "status",
        "split",
        "test_visible",
        "candidate_epochs",
        "candidate_bundle",
        "val_inputs_receipt",
        "pipeline_receipt",
        "pipeline_source",
        "inference_entrypoint",
        "coverage",
        "short_quality_authority",
        "adapter_source",
        "quality_input_binding_sha256",
        "receipt_payload_sha256",
    }
    if set(payload) != expected_keys or payload.get("format") != FORMAT:
        raise ShortQualityValError("short-quality preflight schema mismatch")
    if (
        payload.get("status") != "complete"
        or payload.get("split") != "val"
        or payload.get("test_visible") is not False
        or payload.get("candidate_epochs") != list(QUALITY_EPOCHS)
        or _payload_sha(payload) != payload.get("receipt_payload_sha256")
    ):
        raise ShortQualityValError("short-quality preflight is not frozen val-only")
    authority = payload["short_quality_authority"]
    rebuilt = _build_bound_payload(
        mode=authority["topology_mode"],
        topology_gate_path=Path(authority["topology_gate_spec"]["path"]),
        topology_gate_sha=authority["topology_gate_spec"]["sha256"],
        quality_gate_path=Path(authority["quality_gate_spec"]["path"]),
        quality_gate_sha=authority["quality_gate_spec"]["sha256"],
        status_artifact=authority["short_quality_status"],
        ready_artifacts=authority["candidate_ready_receipts"],
        val_inputs_path=Path(payload["val_inputs_receipt"]["path"]),
        val_inputs_sha=payload["val_inputs_receipt"]["sha256"],
        pipeline_path=Path(payload["pipeline_receipt"]["path"]),
        pipeline_sha=payload["pipeline_receipt"]["sha256"],
    )
    if rebuilt != {
        key: value for key, value in payload.items() if key != "receipt_payload_sha256"
    }:
        raise ShortQualityValError("short-quality preflight differs from fresh replay")
    return (
        {
            "path": str(resolved),
            "sha256": expected_sha,
            "receipt_payload_sha256": payload["receipt_payload_sha256"],
        },
        payload,
    )


@contextmanager
def _engine_adapter() -> Iterable[None]:
    original = engine._preflight_artifact
    engine._preflight_artifact = _preflight_artifact
    try:
        yield
    finally:
        engine._preflight_artifact = original


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    with _engine_adapter():
        return engine.run_shard(args)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    with _engine_adapter():
        return engine.finalize(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", allow_abbrev=False)
    prepare_parser.add_argument("--split", choices=("val",), required=True)
    prepare_parser.add_argument(
        "--mode", choices=list(training.TOPOLOGY_SPECS), required=True
    )
    prepare_parser.add_argument("--topology-gate-spec", type=Path, required=True)
    prepare_parser.add_argument("--expected-topology-gate-spec-sha256", required=True)
    prepare_parser.add_argument("--quality-gate-spec", type=Path, required=True)
    prepare_parser.add_argument("--expected-quality-gate-spec-sha256", required=True)
    prepare_parser.add_argument("--short-quality-status", type=Path, required=True)
    prepare_parser.add_argument("--expected-short-quality-status-sha256", required=True)
    prepare_parser.add_argument(
        "--candidate-ready-receipt",
        nargs=3,
        action="append",
        required=True,
        metavar=("EPOCH", "PATH", "SHA256"),
    )
    prepare_parser.add_argument("--val-inputs", type=Path, required=True)
    prepare_parser.add_argument("--expected-val-inputs-sha256", required=True)
    prepare_parser.add_argument("--pipeline", type=Path, required=True)
    prepare_parser.add_argument("--expected-pipeline-sha256", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)

    validate_parser = commands.add_parser("validate", allow_abbrev=False)
    validate_parser.add_argument("--split", choices=("val",), required=True)
    validate_parser.add_argument("--preflight", type=Path, required=True)
    validate_parser.add_argument("--expected-preflight-sha256", required=True)

    for name in ("shard", "finalize"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--split", choices=("val",), required=True)
        command.add_argument("--preflight", type=Path, required=True)
        command.add_argument("--expected-preflight-sha256", required=True)
        command.add_argument("--epoch", type=_epoch, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--num-shards", type=engine._num_shards, required=True)
        if name == "shard":
            command.add_argument("--shard-id", type=int, required=True)
            command.add_argument("--device", required=True)
            command.add_argument("--seed", type=int, default=20260731)
            command.add_argument("--progress-every", type=int, default=20)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "validate":
        result = validate(args)
    elif args.command == "shard":
        result = run_shard(args)
    elif args.command == "finalize":
        result = finalize(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
