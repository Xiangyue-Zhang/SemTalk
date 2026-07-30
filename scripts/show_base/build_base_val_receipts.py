#!/usr/bin/env python3
"""Build the two immutable receipts consumed by Base validation inference.

This entry point is CPU-only and deliberately has no split switch.  It can
only materialize the frozen 1,715-clip SHOW validation inputs receipt or the
frozen final-pipeline receipt.  Both outputs are created with new-only
semantics and are immediately revalidated by
``select_base_official_adapt.py`` before success is reported.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import select_base_official_adapt as selector


TRANSFER_FORMAT = "semtalk_show_official_transfer_v1"
EXPECTED_HELPER_SOURCE = {
    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
    "commit": "78412d9a4bb349da45c7eb2dc5995021c3688e4f",
    "tree": "1a99575c994d375d9abb02f0ef98577192444ed2",
    "entrypoint": "run_base_inference.py",
    "entrypoint_sha256": (
        "8f634d9fb3e76fa620f3690638d40b5b6bc12dbb3293f148389fc2fdc09f4cf4"
    ),
}
EXPECTED_TRANSFER_SOURCE = {
    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
    "commit": "a8f24eed12529a4014f5f82df5cef40e2eb5d9d7",
    "tree": "b2e941937a135850fd99e1450d0ab1d883c9fbc0",
    "entrypoint": "train_official_transfer.py",
    "entrypoint_sha256": (
        "df60dc664a24cce0e04a6b57d44efaebff8b7936a973c20d2267b6cb4d109d91"
    ),
}


class ReceiptBuildError(RuntimeError):
    """Raised when an immutable receipt cannot be proven."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, label: str) -> tuple[dict[str, str], Path, bytes]:
    raw = selector.require_val_only_path(str(path), label)
    resolved = selector._regular_file(raw, label)
    selector.reject_test_path(resolved, label)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, resolved, payload


def _strict_object(payload: bytes, label: str) -> dict[str, Any]:
    value = selector._strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise ReceiptBuildError(f"{label} must be a JSON object")
    return value


def _payload_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["receipt_payload_sha256"] = selector.canonical_json_sha256(
        receipt
    )
    return receipt


def _transfer_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_transfer_payload_hash(
    payload: Mapping[str, Any],
    label: str,
) -> str:
    claimed = selector.require_sha256(
        payload.get("receipt_sha256"),
        f"{label} receipt SHA",
    )
    body = dict(payload)
    body.pop("receipt_sha256", None)
    observed = _transfer_payload_sha256(body)
    if observed != claimed:
        raise ReceiptBuildError(
            f"{label} receipt payload SHA mismatch: {observed} != {claimed}"
        )
    return claimed


def _prepare_new_output(path: Path, label: str) -> tuple[Path, Path]:
    raw = selector.require_val_only_path(str(path), label)
    if raw.name in {"", ".", ".."}:
        raise ReceiptBuildError(f"{label} has an invalid filename")
    parent = selector.require_directory(str(raw.parent), f"{label} parent")
    if raw.parent != parent:
        raise ReceiptBuildError(
            f"{label} parent must be canonical and contain no symlink: "
            f"{raw.parent}"
        )
    try:
        os.lstat(raw)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"{label} already exists: {raw}")
    return raw, parent


def _atomic_new_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    validator: Callable[[Path, str], Any],
) -> tuple[Path, str]:
    output, parent = _prepare_new_output(path, label)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.",
        dir=parent,
    )
    temporary = Path(temporary_name)
    created = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise FileExistsError(
                    f"{label} already exists: {output}"
                ) from error
            raise
        created = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        observed_sha = hashlib.sha256(encoded).hexdigest()
        # This is intentionally immediate.  A builder success means the
        # decision-side validator accepted the exact bytes just linked.
        validator(output, observed_sha)
        return output, observed_sha
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    canonical, canonical_path, canonical_bytes = _artifact(
        args.canonical_manifest,
        "validation canonical manifest",
    )
    canonical_ids, coverage = selector._canonical_coverage(
        canonical_bytes,
        str(canonical_path),
    )
    canonical_summary, canonical_lineage = (
        selector._validate_val_canonical_receipts(
            summary_value=_artifact(
                args.canonical_summary,
                "validation canonical summary",
            )[0],
            lineage_value=_artifact(
                args.canonical_lineage,
                "validation canonical lineage",
            )[0],
            canonical_manifest_sha256=canonical["sha256"],
        )
    )
    audio = selector._audio_coverage(
        [
            _artifact(path, "validation audio manifest")[0]
            for path in args.audio_manifest
        ],
        [
            _artifact(path, "validation audio summary")[0]
            for path in args.audio_summary
        ],
        [
            _artifact(path, "validation audio lineage")[0]
            for path in args.audio_lineage
        ],
        canonical_ids,
    )
    receipt = _payload_receipt(
        {
            "format": selector.VAL_INPUTS_FORMAT,
            "status": "frozen",
            "split": "val",
            "test_visible": False,
            "expected_clip_count": selector.EXPECTED_VAL_CLIPS,
            "canonical_manifest": canonical,
            "canonical_summary": canonical_summary,
            "canonical_lineage": canonical_lineage,
            "audio_manifests": audio["manifests"],
            "audio_summaries": audio["summaries"],
            "audio_lineages": audio["lineages"],
            "clip_ids_sha256": coverage["clip_ids_sha256"],
            "diffsheg_clip_manifest_sha256": coverage[
                "diffsheg_clip_manifest_sha256"
            ],
        }
    )
    output, output_sha = _atomic_new_json(
        args.output,
        receipt,
        label="validation inputs receipt",
        validator=selector.validate_val_inputs,
    )
    return {
        "status": "complete",
        "kind": "inputs",
        "path": str(output),
        "sha256": output_sha,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "clip_count": coverage["clip_count"],
        "audio_shards": audio["num_shards"],
    }


def _require_exact_int(value: Any, label: str) -> int:
    return selector.require_exact_int(value, label)


def _require_nonnegative_finite(value: Any, label: str) -> float:
    converted = selector.require_finite_number(value, label)
    if converted < 0.0:
        raise ReceiptBuildError(f"{label} must be nonnegative")
    return converted


def _validate_source_receipt(value: Any, label: str) -> dict[str, Any]:
    source = selector.require_exact_keys(
        value,
        {
            "origin",
            "commit",
            "tree",
            "branch",
            "clean",
            "entrypoint",
            "entrypoint_sha256",
        },
        label,
    )
    entrypoint = selector.require_val_only_path(
        source["entrypoint"],
        f"{label} entrypoint",
    )
    resolved_entrypoint = selector._regular_file(
        entrypoint,
        f"{label} entrypoint",
    )
    selector.reject_test_path(resolved_entrypoint, f"{label} entrypoint")
    if (
        source["origin"] != EXPECTED_TRANSFER_SOURCE["origin"]
        or source["commit"] != EXPECTED_TRANSFER_SOURCE["commit"]
        or source["tree"] != EXPECTED_TRANSFER_SOURCE["tree"]
        or source["branch"] is not None
        or source["clean"] is not True
        or resolved_entrypoint.name != EXPECTED_TRANSFER_SOURCE["entrypoint"]
        or source["entrypoint_sha256"]
        != EXPECTED_TRANSFER_SOURCE["entrypoint_sha256"]
        or _sha256_file(resolved_entrypoint)
        != EXPECTED_TRANSFER_SOURCE["entrypoint_sha256"]
    ):
        raise ReceiptBuildError(
            f"{label} is not the exact clean detached a8f24ee source"
        )
    selector.require_git_oid(source["commit"], f"{label} commit")
    selector.require_git_oid(source["tree"], f"{label} tree")
    selector.require_sha256(
        source["entrypoint_sha256"],
        f"{label} entrypoint SHA",
    )
    return source


def _validate_metrics(
    summary: Mapping[str, Any],
    *,
    stage: str,
) -> tuple[Path, list[dict[str, Any]]]:
    path = selector.require_val_only_path(
        summary["metrics_jsonl"],
        f"{stage} transfer metrics",
    )
    resolved = selector._regular_file(path, f"{stage} transfer metrics")
    selector.reject_test_path(resolved, f"{stage} transfer metrics")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != summary["metrics_jsonl_sha256"]:
        raise ReceiptBuildError(f"{stage} transfer metrics SHA mismatch")
    rows = selector._strict_jsonl(payload, f"{stage} transfer metrics")
    epochs = _require_exact_int(summary["epochs"], f"{stage} epochs")
    if len(rows) != epochs or epochs <= 0:
        raise ReceiptBuildError(f"{stage} transfer metrics coverage mismatch")
    totals: list[tuple[float, int]] = []
    for expected_epoch, row in enumerate(rows, 1):
        if set(row) != {
            "epoch",
            "train",
            "val",
            "elapsed_seconds",
            "rank_count",
            "finite",
        }:
            raise ReceiptBuildError(f"{stage} metrics row schema mismatch")
        epoch = _require_exact_int(row["epoch"], f"{stage} metric epoch")
        if epoch != expected_epoch or row["finite"] is not True:
            raise ReceiptBuildError(f"{stage} metrics epochs are not exact")
        if (
            not isinstance(row["train"], dict)
            or not isinstance(row["val"], dict)
            or "total" not in row["train"]
            or "total" not in row["val"]
        ):
            raise ReceiptBuildError(f"{stage} metric mappings are invalid")
        for split in ("train", "val"):
            for name, metric in row[split].items():
                _require_nonnegative_finite(
                    metric,
                    f"{stage} {split} {name}",
                )
        _require_nonnegative_finite(
            row["elapsed_seconds"],
            f"{stage} elapsed seconds",
        )
        if _require_exact_int(row["rank_count"], f"{stage} rank count") <= 0:
            raise ReceiptBuildError(f"{stage} rank count must be positive")
        totals.append((float(row["val"]["total"]), epoch))
    best_total, best_epoch = min(totals, key=lambda item: (item[0], item[1]))
    if (
        summary["best_epoch"] != best_epoch
        or float(summary["best_validation_total"]) != best_total
        or best_epoch == 30
    ):
        raise ReceiptBuildError(
            f"{stage} summary is not the non-e30 validation minimum"
        )
    return resolved, rows


def _validate_transfer(
    *,
    stage: str,
    checkpoint_path: Path,
    summary_path: Path,
) -> dict[str, str]:
    checkpoint, resolved_checkpoint, _ = _artifact(
        checkpoint_path,
        f"best {stage} transfer checkpoint",
    )
    expected_filename = f"best_{stage}_transfer.bin"
    if resolved_checkpoint.name != expected_filename:
        raise ReceiptBuildError(
            f"{stage} checkpoint must be named {expected_filename}"
        )
    _summary_artifact, resolved_summary, summary_bytes = _artifact(
        summary_path,
        f"{stage} transfer summary",
    )
    summary = selector.require_exact_keys(
        _strict_object(summary_bytes, f"{stage} transfer summary"),
        {
            "format",
            "status",
            "stage",
            "epochs",
            "best_epoch",
            "best_validation_total",
            "official_initialization",
            "source_receipt",
            "cache_receipt",
            "trainable_policy",
            "protocol",
            "frozen_encoder_quantizer_sha256",
            "withdrawn_e30_allowed",
            "test_visible",
            "metrics_jsonl",
            "metrics_jsonl_sha256",
            "elapsed_seconds",
            "receipt_sha256",
        },
        f"{stage} transfer summary",
    )
    _verify_transfer_payload_hash(
        summary,
        f"{stage} transfer summary",
    )
    protocol = summary["protocol"]
    if (
        summary["format"] != TRANSFER_FORMAT
        or summary["status"] != "complete"
        or summary["stage"] != stage
        or summary["withdrawn_e30_allowed"] is not False
        or summary["test_visible"] is not False
        or not isinstance(protocol, dict)
        or protocol.get("stage") != stage
        or protocol.get("selection_split") != "val"
        or protocol.get("test_visible") is not False
    ):
        raise ReceiptBuildError(
            f"{resolved_summary}: invalid {stage} validation summary"
        )
    _validate_source_receipt(
        summary["source_receipt"],
        f"{stage} transfer source",
    )
    selector.require_sha256(
        summary["metrics_jsonl_sha256"],
        f"{stage} metrics SHA",
    )
    _require_nonnegative_finite(
        summary["best_validation_total"],
        f"{stage} best validation total",
    )
    _require_nonnegative_finite(
        summary["elapsed_seconds"],
        f"{stage} elapsed seconds",
    )
    _validate_metrics(summary, stage=stage)

    try:
        import torch
    except ImportError as error:
        raise ReceiptBuildError(
            "PyTorch is required to bind transfer checkpoints to summaries"
        ) from error
    payload = torch.load(
        resolved_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {
            "model_state",
            "optimizer_state",
            "epoch",
            "transfer_receipt",
        }
    ):
        raise ReceiptBuildError(f"{stage} transfer checkpoint schema mismatch")
    transfer = payload["transfer_receipt"]
    if not isinstance(transfer, dict):
        raise ReceiptBuildError(f"{stage} transfer receipt is missing")
    _verify_transfer_payload_hash(
        transfer,
        f"{stage} checkpoint transfer receipt",
    )
    validation = transfer.get("validation")
    if (
        transfer.get("format") != TRANSFER_FORMAT
        or transfer.get("stage") != stage
        or transfer.get("epoch") != payload["epoch"]
        or payload["epoch"] != summary["best_epoch"]
        or payload["epoch"] == 30
        or transfer.get("withdrawn_e30_allowed") is not False
        or transfer.get("test_visible") is not False
        or not isinstance(validation, dict)
        or validation.get("total") != summary["best_validation_total"]
    ):
        raise ReceiptBuildError(
            f"{stage} best checkpoint/validation summary mismatch"
        )
    for key in (
        "official_initialization",
        "source_receipt",
        "cache_receipt",
        "trainable_policy",
        "protocol",
        "frozen_encoder_quantizer_sha256",
    ):
        if transfer.get(key) != summary[key]:
            raise ReceiptBuildError(
                f"{stage} checkpoint/summary {key} mismatch"
            )
    return checkpoint


def _validate_official_checkpoint(
    path: Path,
    *,
    stage: str,
) -> dict[str, str]:
    artifact, resolved, _ = _artifact(
        path,
        f"official {stage} checkpoint",
    )
    specification = selector.FIXED_OFFICIAL_CHECKPOINTS[stage]
    if (
        resolved.name != specification["filename"]
        or artifact["sha256"] != specification["sha256"]
    ):
        raise ReceiptBuildError(
            f"{stage} is not the exact released All-Speakers checkpoint"
        )
    return {
        **artifact,
        "source": "released_all_speakers_v1",
    }


def build_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if selector.VAL_INFERENCE_SOURCE != EXPECTED_HELPER_SOURCE:
        raise ReceiptBuildError(
            "selector validation helper pin is not 78412d9/1a995/8f634d"
        )
    helper, resolved_helper, _ = _artifact(
        args.pinned_helper,
        "pinned validation inference helper",
    )
    if (
        resolved_helper.name != EXPECTED_HELPER_SOURCE["entrypoint"]
        or helper["sha256"] != EXPECTED_HELPER_SOURCE[
            "entrypoint_sha256"
        ]
    ):
        raise ReceiptBuildError(
            "validation helper is not the exact 78412d9 pinned entrypoint"
        )
    face = _validate_transfer(
        stage="face",
        checkpoint_path=args.face_checkpoint,
        summary_path=args.face_summary,
    )
    global_checkpoint = _validate_transfer(
        stage="global",
        checkpoint_path=args.global_checkpoint,
        summary_path=args.global_summary,
    )
    diffsheg = selector.DIFFSHEG_PINNED_RECEIPT
    if (
        not isinstance(diffsheg.get("autoencoders"), dict)
        or set(diffsheg["autoencoders"]) != {"fgd"}
        or diffsheg.get("selection_metric") != "fgd"
        or diffsheg.get("ba_during_selection") is not False
    ):
        raise ReceiptBuildError("validation DiffSHEG receipt is not FGD-only")
    pipeline = _payload_receipt(
        {
            "format": selector.PIPELINE_FORMAT,
            "status": "frozen",
            "split": "val",
            "test_visible": False,
            "mode": "official_show_adapt_v1",
            "base_candidate_variable_only": True,
            "source": dict(EXPECTED_HELPER_SOURCE),
            "inference_entrypoint": helper,
            "inference_helpers": list(selector.INFERENCE_HELPERS),
            "fixed_checkpoints": {
                "face": {
                    **face,
                    "selection_split": "val",
                    "test_visible": False,
                },
                "global": {
                    **global_checkpoint,
                    "selection_split": "val",
                    "test_visible": False,
                },
                "hands": _validate_official_checkpoint(
                    args.hands_checkpoint,
                    stage="hands",
                ),
                "upper": _validate_official_checkpoint(
                    args.upper_checkpoint,
                    stage="upper",
                ),
                "lower": _validate_official_checkpoint(
                    args.lower_checkpoint,
                    stage="lower",
                ),
            },
            "diffsheg": diffsheg,
        }
    )
    output, output_sha = _atomic_new_json(
        args.output,
        pipeline,
        label="validation pipeline receipt",
        validator=selector.validate_pipeline,
    )
    return {
        "status": "complete",
        "kind": "pipeline",
        "path": str(output),
        "sha256": output_sha,
        "receipt_payload_sha256": pipeline["receipt_payload_sha256"],
        "face_checkpoint_sha256": face["sha256"],
        "global_checkpoint_sha256": global_checkpoint["sha256"],
        "diffsheg_selection_metric": "fgd",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen Base validation-only receipts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inputs = subparsers.add_parser(
        "inputs",
        help="Build the exact 1,715-clip validation inputs receipt.",
    )
    inputs.add_argument("--output", type=Path, required=True)
    inputs.add_argument("--canonical-manifest", type=Path, required=True)
    inputs.add_argument("--canonical-summary", type=Path, required=True)
    inputs.add_argument("--canonical-lineage", type=Path, required=True)
    inputs.add_argument(
        "--audio-manifest",
        type=Path,
        action="append",
        required=True,
    )
    inputs.add_argument(
        "--audio-summary",
        type=Path,
        action="append",
        required=True,
    )
    inputs.add_argument(
        "--audio-lineage",
        type=Path,
        action="append",
        required=True,
    )
    inputs.set_defaults(handler=build_inputs)

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Build the frozen final-pipeline validation receipt.",
    )
    pipeline.add_argument("--output", type=Path, required=True)
    pipeline.add_argument("--pinned-helper", type=Path, required=True)
    pipeline.add_argument("--face-checkpoint", type=Path, required=True)
    pipeline.add_argument("--face-summary", type=Path, required=True)
    pipeline.add_argument("--global-checkpoint", type=Path, required=True)
    pipeline.add_argument("--global-summary", type=Path, required=True)
    pipeline.add_argument("--hands-checkpoint", type=Path, required=True)
    pipeline.add_argument("--upper-checkpoint", type=Path, required=True)
    pipeline.add_argument("--lower-checkpoint", type=Path, required=True)
    pipeline.set_defaults(handler=build_pipeline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inputs":
        for name in ("audio_manifest", "audio_summary", "audio_lineage"):
            if len(getattr(args, name)) != selector.EXPECTED_AUDIO_SHARDS:
                parser.error(f"--{name.replace('_', '-')} must appear 8 times")
    try:
        report = args.handler(args)
    except (
        ReceiptBuildError,
        selector.SelectionContractError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
        TypeError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
