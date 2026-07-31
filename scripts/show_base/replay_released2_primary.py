#!/usr/bin/env python3
"""Build or replay the fresh released2 FGD authority used for Base selection."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.show_base import evaluate_talkshow_show_metrics as metrics
from scripts.show_base import talkshow_base_val_contract as val_contract


ENTRYPOINT_RELATIVE = "scripts/show_base/replay_released2_primary.py"


def _metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--talkshow-metric-root", type=Path, required=True)
    parser.add_argument("--feature-extractor", type=Path, required=True)
    parser.add_argument("--smplx-asset", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--split", choices=("val",), required=True)
    parser.add_argument("--expected-clip-count", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("build-cache", allow_abbrev=False)
    _metric_arguments(cache)
    cache.add_argument("--canonical-manifest", type=Path, required=True)
    cache.add_argument(
        "--expected-canonical-manifest-sha256",
        required=True,
    )
    cache.add_argument("--canonical-summary", type=Path, required=True)
    cache.add_argument(
        "--expected-canonical-summary-sha256",
        required=True,
    )
    cache.add_argument("--canonical-lineage", type=Path, required=True)
    cache.add_argument(
        "--expected-canonical-lineage-sha256",
        required=True,
    )
    for name in ("manifest", "summary", "lineage"):
        cache.add_argument(
            f"--audio-{name}",
            type=Path,
            action="append",
            required=True,
        )
        cache.add_argument(
            f"--expected-audio-{name}-sha256",
            action="append",
            required=True,
        )
    cache.add_argument("--source-root", type=Path, required=True)
    cache.add_argument("--expected-source-commit", required=True)
    cache.add_argument("--expected-source-tree", required=True)
    cache.add_argument("--expected-entrypoint-sha256", required=True)

    screen = commands.add_parser("screen", allow_abbrev=False)
    _metric_arguments(screen)
    screen.add_argument("--canonical-manifest", type=Path, required=True)
    screen.add_argument(
        "--expected-canonical-manifest-sha256", required=True
    )
    screen.add_argument(
        "--expected-canonical-manifest-bytes", type=int, required=True
    )
    screen.add_argument("--prediction-manifest", type=Path, required=True)
    screen.add_argument(
        "--expected-prediction-manifest-sha256", required=True
    )
    screen.add_argument(
        "--expected-prediction-manifest-bytes", type=int, required=True
    )
    screen.add_argument("--distribution-json", type=Path, required=True)
    screen.add_argument("--expected-distribution-sha256", required=True)
    screen.add_argument(
        "--expected-distribution-bytes", type=int, required=True
    )
    screen.add_argument(
        "--expected-distribution-payload-sha256", required=True
    )
    screen.add_argument(
        "--real-feature-cache-json", type=Path, required=True
    )
    screen.add_argument(
        "--expected-real-feature-cache-sha256", required=True
    )
    screen.add_argument(
        "--expected-real-feature-cache-bytes", type=int, required=True
    )
    screen.add_argument(
        "--expected-real-feature-cache-payload-sha256", required=True
    )

    replay = commands.add_parser("replay", allow_abbrev=False)
    _metric_arguments(replay)
    replay.add_argument("--report-json", type=Path, required=True)
    replay.add_argument("--expected-report-sha256", required=True)
    replay.add_argument("--expected-report-bytes", type=int, required=True)
    replay.add_argument("--prediction-manifest", type=Path, required=True)
    replay.add_argument(
        "--expected-prediction-manifest-sha256",
        required=True,
    )
    replay.add_argument(
        "--expected-prediction-manifest-bytes",
        type=int,
        required=True,
    )
    replay.add_argument("--real-feature-cache-json", type=Path, required=True)
    replay.add_argument(
        "--expected-real-feature-cache-sha256",
        required=True,
    )
    replay.add_argument(
        "--expected-real-feature-cache-bytes",
        type=int,
        required=True,
    )
    replay.add_argument(
        "--expected-real-feature-cache-payload-sha256",
        required=True,
    )
    return parser.parse_args(argv)


def _backend(args: argparse.Namespace) -> metrics.TalkShowCudaMetricBackend:
    return metrics.TalkShowCudaMetricBackend(
        talkshow_root=args.talkshow_metric_root,
        feature_extractor=args.feature_extractor,
        smplx_asset=args.smplx_asset,
        device=args.device,
        torch_threads=args.torch_threads,
    )


def _pinned_artifact(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, str], Path, bytes]:
    expected = val_contract.require_sha256(
        expected_sha256,
        f"{label} expected SHA-256",
    )
    resolved, payload, observed = val_contract._verified_bytes(
        path,
        expected,
        label,
    )
    val_contract.reject_test_path(resolved, label)
    return {"path": str(resolved), "sha256": observed}, resolved, payload


def _source_authority(args: argparse.Namespace) -> dict[str, Any]:
    source_root = val_contract.require_directory(
        str(args.source_root),
        "released2 real-feature cache source root",
    )
    repository_root = REPOSITORY_ROOT.resolve(strict=True)
    if source_root != repository_root:
        raise metrics.MetricAdapterContractError(
            "real-feature cache producer must bind its own source checkout"
        )
    expected_commit = val_contract.require_git_oid(
        args.expected_source_commit,
        "released2 real-feature cache expected source commit",
    )
    expected_tree = val_contract.require_git_oid(
        args.expected_source_tree,
        "released2 real-feature cache expected source tree",
    )
    expected_entrypoint = val_contract.require_sha256(
        args.expected_entrypoint_sha256,
        "released2 real-feature cache expected entrypoint SHA-256",
    )
    source = val_contract.build_fresh_pipeline_source_receipt(source_root)
    if source["commit"] != expected_commit or source["tree"] != expected_tree:
        raise metrics.MetricAdapterContractError(
            "released2 real-feature cache source commit/tree mismatch"
        )
    entrypoint = source["files"].get(ENTRYPOINT_RELATIVE)
    if (
        not isinstance(entrypoint, dict)
        or entrypoint.get("sha256") != expected_entrypoint
    ):
        raise metrics.MetricAdapterContractError(
            "released2 real-feature cache entrypoint SHA-256 mismatch"
        )
    return {
        "origin": source["origin"],
        "source_root": source["source_root"],
        "commit": source["commit"],
        "tree": source["tree"],
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "entrypoint": {
            "path": entrypoint["path"],
            "relative": ENTRYPOINT_RELATIVE,
            "sha256": entrypoint["sha256"],
            "bytes": entrypoint["bytes"],
            "git_mode": entrypoint["git_mode"],
            "git_blob_sha1": entrypoint["git_blob_sha1"],
        },
    }


def _validation_input_authority(args: argparse.Namespace) -> dict[str, Any]:
    canonical, canonical_path, canonical_payload = _pinned_artifact(
        args.canonical_manifest,
        args.expected_canonical_manifest_sha256,
        "released2 real-feature cache canonical manifest",
    )
    canonical_ids, coverage = val_contract._canonical_coverage(
        canonical_payload,
        str(canonical_path),
    )
    canonical_summary = _pinned_artifact(
        args.canonical_summary,
        args.expected_canonical_summary_sha256,
        "released2 real-feature cache canonical summary",
    )[0]
    canonical_lineage = _pinned_artifact(
        args.canonical_lineage,
        args.expected_canonical_lineage_sha256,
        "released2 real-feature cache canonical lineage",
    )[0]
    canonical_summary, canonical_lineage = (
        val_contract._validate_val_canonical_receipts(
            summary_value=canonical_summary,
            lineage_value=canonical_lineage,
            canonical_manifest_sha256=canonical["sha256"],
        )
    )

    audio_receipts: dict[str, list[dict[str, str]]] = {}
    for name in ("manifest", "summary", "lineage"):
        paths = getattr(args, f"audio_{name}")
        expected = getattr(args, f"expected_audio_{name}_sha256")
        if len(paths) != val_contract.EXPECTED_AUDIO_SHARDS or len(
            expected
        ) != val_contract.EXPECTED_AUDIO_SHARDS:
            raise metrics.MetricAdapterContractError(
                f"--audio-{name} and --expected-audio-{name}-sha256 "
                "must each appear exactly eight times"
            )
        audio_receipts[name] = [
            _pinned_artifact(
                path,
                digest,
                f"released2 real-feature cache audio {name} {index}",
            )[0]
            for index, (path, digest) in enumerate(zip(paths, expected))
        ]
    audio = val_contract._audio_coverage(
        audio_receipts["manifest"],
        audio_receipts["summary"],
        audio_receipts["lineage"],
        canonical_ids,
    )
    if args.split != "val" or args.expected_clip_count != len(canonical_ids):
        raise metrics.MetricAdapterContractError(
            "released2 real-feature cache accepts only the exact validation "
            "clip set"
        )
    inputs: dict[str, Any] = {
        "format": val_contract.VAL_INPUTS_FORMAT,
        "status": "frozen",
        "split": "val",
        "test_visible": False,
        "expected_clip_count": val_contract.EXPECTED_VAL_CLIPS,
        "canonical_manifest": canonical,
        "canonical_summary": canonical_summary,
        "canonical_lineage": canonical_lineage,
        "audio_manifests": audio["manifests"],
        "audio_summaries": audio["summaries"],
        "audio_lineages": audio["lineages"],
        "clip_ids_sha256": coverage["clip_ids_sha256"],
        "talkshow_window_manifest_sha256": coverage[
            "talkshow_window_manifest_sha256"
        ],
    }
    inputs["receipt_payload_sha256"] = val_contract.canonical_json_sha256(
        inputs
    )
    return inputs


def _production_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority: dict[str, Any] = {
        "format": (
            metrics.PRIMARY_REAL_FEATURE_CACHE_PRODUCTION_AUTHORITY_FORMAT
        ),
        "status": "frozen",
        "source": _source_authority(args),
        "validation_inputs": _validation_input_authority(args),
    }
    authority["receipt_payload_sha256"] = (
        metrics.compact_canonical_json_sha256(authority)
    )
    return authority


def _prepare_new_output(path: Path) -> Path:
    output = path.expanduser()
    if not output.is_absolute():
        raise metrics.MetricAdapterContractError(
            "released2 real-feature cache output must be absolute"
        )
    val_contract.reject_test_path(
        output,
        "released2 real-feature cache output",
    )
    parent = val_contract.require_directory(
        str(output.parent),
        "released2 real-feature cache output parent",
    )
    if output.parent != parent:
        raise metrics.MetricAdapterContractError(
            "released2 real-feature cache output parent must be canonical"
        )
    source_root = REPOSITORY_ROOT.resolve(strict=True)
    if output == source_root or source_root in output.parents:
        raise metrics.MetricAdapterContractError(
            "released2 real-feature cache output must be outside source root"
        )
    try:
        os.lstat(output)
    except FileNotFoundError:
        return output
    raise FileExistsError(
        f"released2 real-feature cache output already exists: {output}"
    )


def _validate_written_cache(
    output: Path,
    payload: bytes,
    result: dict[str, Any],
) -> None:
    artifact = {
        "path": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "receipt_payload_sha256": result["receipt_payload_sha256"],
    }
    metrics._validate_released2_real_feature_cache(
        result,
        expected_artifact=artifact,
        expected_canonical_manifest=result["canonical_manifest"],
        expected_split="val",
        expected_clip_count=val_contract.EXPECTED_VAL_CLIPS,
        fixture_mode=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-cache":
        output = _prepare_new_output(args.output_json)
        initial_authority = _production_authority(args)
        backend = _backend(args)
        result = metrics.build_released2_real_feature_cache(
            canonical_manifest=args.canonical_manifest,
            expected_canonical_manifest_sha256=(
                args.expected_canonical_manifest_sha256
            ),
            backend=backend,
            split=args.split,
            expected_clip_count=args.expected_clip_count,
            formal_mode=True,
            test_only_allow_four_clip_subset=False,
            production_authority=initial_authority,
        )
        if _production_authority(args) != initial_authority:
            raise metrics.MetricAdapterContractError(
                "released2 real-feature cache source or inputs changed during "
                "production"
            )
    elif args.command == "screen":
        backend = _backend(args)
        cache_artifact = {
            "path": str(args.real_feature_cache_json.resolve()),
            "sha256": args.expected_real_feature_cache_sha256,
            "bytes": args.expected_real_feature_cache_bytes,
            "receipt_payload_sha256": (
                args.expected_real_feature_cache_payload_sha256
            ),
        }
        _cache_artifact, cache = metrics._payload_json_artifact(
            cache_artifact,
            label="primary-screen real-feature cache",
        )
        distribution_artifact = {
            "path": str(args.distribution_json.resolve()),
            "sha256": args.expected_distribution_sha256,
            "bytes": args.expected_distribution_bytes,
            "receipt_payload_sha256": (
                args.expected_distribution_payload_sha256
            ),
        }
        result = metrics.build_released2_primary_screen(
            backend,
            canonical_manifest={
                "path": str(args.canonical_manifest.resolve()),
                "sha256": args.expected_canonical_manifest_sha256,
                "bytes": args.expected_canonical_manifest_bytes,
            },
            prediction_manifest={
                "path": str(args.prediction_manifest.resolve()),
                "sha256": args.expected_prediction_manifest_sha256,
                "bytes": args.expected_prediction_manifest_bytes,
            },
            distribution_receipt=distribution_artifact,
            real_feature_cache=cache,
            expected_real_feature_cache_artifact=cache_artifact,
            expected_selection_protocol={
                "primary_metric": metrics.PRIMARY_METRIC_PATH,
                "mode": "min",
                "validation_only_for_selection": True,
                "test_evaluations": 0,
            },
            expected_split=args.split,
            expected_clip_count=args.expected_clip_count,
        )
    else:
        backend = _backend(args)
        _report_path, report_payload = metrics._verified_file_snapshot(
            args.report_json,
            args.expected_report_sha256,
            "fresh-primary report",
        )
        if len(report_payload) != args.expected_report_bytes:
            raise metrics.MetricAdapterContractError(
                "fresh-primary report byte count changed"
            )
        report = metrics._strict_json_snapshot(
            report_payload,
            "fresh-primary report",
        )
        cache_artifact = {
            "path": str(args.real_feature_cache_json.resolve()),
            "sha256": args.expected_real_feature_cache_sha256,
            "bytes": args.expected_real_feature_cache_bytes,
            "receipt_payload_sha256": (
                args.expected_real_feature_cache_payload_sha256
            ),
        }
        _cache_artifact, cache = metrics._payload_json_artifact(
            cache_artifact,
            label="fresh-primary real-feature cache",
        )
        prediction_artifact = {
            "path": str(args.prediction_manifest.resolve()),
            "sha256": args.expected_prediction_manifest_sha256,
            "bytes": args.expected_prediction_manifest_bytes,
        }
        selection_protocol = {
            "primary_metric": metrics.PRIMARY_METRIC_PATH,
            "mode": "min",
            "validation_only_for_selection": True,
            "test_evaluations": 0,
        }
        if report.get("format") == metrics.PRIMARY_SCREEN_FORMAT:
            report_artifact = {
                "path": str(args.report_json.resolve()),
                "sha256": args.expected_report_sha256,
                "bytes": args.expected_report_bytes,
                "receipt_payload_sha256": report[
                    "receipt_payload_sha256"
                ],
            }
            screen_validation = (
                metrics.validate_released2_primary_screen_receipt(
                    report_artifact,
                    expected_prediction_manifest=prediction_artifact,
                    expected_distribution_receipt=report[
                        "distribution_receipt"
                    ],
                    expected_real_feature_cache=cache_artifact,
                    expected_canonical_manifest=report[
                        "canonical_manifest"
                    ],
                    expected_selection_protocol=selection_protocol,
                    expected_split=args.split,
                    expected_clip_count=args.expected_clip_count,
                )
            )
            result = metrics.fresh_replay_released2_primary_screen(
                report,
                screen_validation,
                backend,
                screen_artifact=report_artifact,
                real_feature_cache=cache,
                expected_real_feature_cache_artifact=cache_artifact,
                expected_prediction_manifest=prediction_artifact,
                expected_distribution_receipt=report[
                    "distribution_receipt"
                ],
                expected_selection_protocol=selection_protocol,
                expected_split=args.split,
                expected_clip_count=args.expected_clip_count,
            )
        else:
            result = metrics.fresh_replay_released2_primary(
                report,
                backend,
                real_feature_cache=cache,
                expected_real_feature_cache_artifact=cache_artifact,
                expected_prediction_manifest=prediction_artifact,
                expected_distribution_receipt=report[
                    "distribution_receipt"
                ],
                expected_selection_protocol=selection_protocol,
                expected_split=args.split,
                expected_clip_count=args.expected_clip_count,
            )
    payload = metrics.canonical_json_bytes(result)
    destination = output if args.command == "build-cache" else args.output_json
    metrics._atomic_write_new(destination, payload)
    if args.command == "build-cache":
        owned = os.lstat(destination)
        try:
            _validate_written_cache(destination, payload, result)
        except BaseException:
            try:
                current = os.lstat(destination)
            except FileNotFoundError:
                pass
            else:
                if (
                    stat.S_ISREG(current.st_mode)
                    and (current.st_dev, current.st_ino)
                    == (owned.st_dev, owned.st_ino)
                ):
                    destination.unlink()
            raise
    print(
        f"released2 primary {args.command} complete: "
        f"split={args.split} clips={args.expected_clip_count} "
        f"sha256={hashlib.sha256(payload).hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
