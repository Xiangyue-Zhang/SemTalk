#!/usr/bin/env python3
"""Build or replay the fresh released2 FGD authority used for Base selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.show_base import evaluate_talkshow_show_metrics as metrics


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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    backend = _backend(args)
    if args.command == "build-cache":
        result = metrics.build_released2_real_feature_cache(
            canonical_manifest=args.canonical_manifest,
            expected_canonical_manifest_sha256=(
                args.expected_canonical_manifest_sha256
            ),
            backend=backend,
            split=args.split,
            expected_clip_count=args.expected_clip_count,
        )
    elif args.command == "screen":
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
    metrics._atomic_write_new(
        args.output_json,
        metrics.canonical_json_bytes(result),
    )
    print(
        f"released2 primary {args.command} complete: "
        f"split={args.split} clips={args.expected_clip_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
