#!/usr/bin/env python3
"""Strict train-only entry point for the SHOW SemTalk Base reproduction.

This deliberately bypasses SemTalk's stock test-during-training loop.  It keeps
the model, trainer, optimizer, scheduler, and epoch update semantics intact,
while adding resumable checkpoints and finite-value audits.  Launch it with
``torchrun`` even for a one-GPU run so rank handling is unambiguous.
"""

from __future__ import annotations

import hashlib
import io
import json
from numbers import Real
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time
import traceback
from typing import Any

# Formal source receipts require the checkout to stay byte-for-byte clean.
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from utils import config, logger_tools, other_tools


FORMAL_SMPLX_FILENAME = "SMPLX_NEUTRAL_2020.npz"
FORMAL_SMPLX_SHA256 = (
    "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74"
)
FORMAL_SMPLX_STAGES = frozenset({"face", "hands", "upper", "lower"})
BASE_CANDIDATE_INTERVAL_EPOCHS = 10
BASE_CANDIDATE_TRANSACTION_FILENAME = "base_candidate_transaction.json"
BASE_CANDIDATE_STAGING_FILENAME = ".base_candidate_checkpoint.staging"


def _require_exact_audit_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(
            f"{label} must be an exact integer, got {value!r}"
        )
    return value


def _require_exact_audit_int_mapping(
    value: Any,
    expected: dict[str, int],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimeError(f"{label} must have exactly {sorted(expected)}")
    for key, expected_value in expected.items():
        if (
            _require_exact_audit_int(value[key], f"{label}.{key}")
            != expected_value
        ):
            raise RuntimeError(
                f"{label}.{key} must equal {expected_value}"
            )
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_torch_save(
    path: Path,
    payload: dict[str, Any],
    *,
    staging_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = (
        staging_path
        if staging_path is not None
        else path.with_name(f".{path.name}.tmp.{os.getpid()}")
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.is_symlink() or (tmp.exists() and not tmp.is_file()):
        raise RuntimeError(
            f"atomic torch staging path must be a regular non-symlink file: {tmp}"
        )
    torch.save(payload, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    for directory in {tmp.parent, path.parent}:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load_candidate_snapshot(
    snapshot: bytes,
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Deserialize exactly the immutable bytes that were hashed."""
    try:
        return torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
        )


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_document_sha256(payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _formal_smplx_asset_receipt(args: Any) -> dict[str, Any] | None:
    if args.formal_stage not in FORMAL_SMPLX_STAGES:
        if args.expected_smplx_asset_sha256:
            raise RuntimeError(
                "--expected_smplx_asset_sha256 is restricted to the four "
                "SMPL-X-backed RVQ stages"
            )
        return None

    expected = str(args.expected_smplx_asset_sha256 or "").strip().lower()
    if expected != FORMAL_SMPLX_SHA256:
        raise RuntimeError(
            "formal face/hands/upper/lower training requires the frozen "
            f"SMPL-X SHA-256 {FORMAL_SMPLX_SHA256}"
        )
    asset_input = (
        Path(args.data_path_1)
        / "smplx_models"
        / "smplx"
        / FORMAL_SMPLX_FILENAME
    )
    try:
        asset_stat = asset_input.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(asset_input) from None
    if stat.S_ISLNK(asset_stat.st_mode) or not stat.S_ISREG(asset_stat.st_mode):
        raise RuntimeError(
            f"formal SMPL-X asset must be a regular non-symlink file: {asset_input}"
        )
    if asset_input.name != FORMAL_SMPLX_FILENAME:
        raise RuntimeError("formal SMPL-X asset basename mismatch")
    asset_path = asset_input.resolve(strict=True)
    observed = _sha256(asset_path)
    if observed != expected:
        raise RuntimeError(
            f"formal SMPL-X asset SHA mismatch: {observed} != {expected}"
        )
    return {
        "format": "semtalk_show_smplx_asset_v1",
        "filename": FORMAL_SMPLX_FILENAME,
        "path": str(asset_path),
        "sha256": observed,
        "bytes": int(asset_stat.st_size),
        "regular_file": True,
        "symlink": False,
    }


def _source_receipt() -> dict[str, str]:
    root = Path(__file__).resolve().parent

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "formal training source must be a clean checkout; first change: "
            f"{status.splitlines()[0]}"
        )
    origin = git("remote", "get-url", "origin")
    expected_origin = "git@github.com:Xiangyue-Zhang/SemTalk.git"
    if origin != expected_origin:
        raise RuntimeError(
            f"formal training origin {origin!r} != {expected_origin!r}"
        )
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "origin": origin,
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": _sha256(Path(__file__).resolve()),
    }


def _config_fingerprint(args: Any) -> str:
    snapshot = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume_state", "local_rank"}
    }
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_receipt(
    args: Any,
    *,
    train_samples: int,
    current_source: dict[str, str],
) -> dict[str, Any]:
    smplx_asset_receipt = _formal_smplx_asset_receipt(args)
    if not args.dataset_summary:
        raise RuntimeError("formal training requires --dataset_summary")
    summary_input = Path(args.dataset_summary)
    lineage_input = Path(args.lineage_manifest)
    if summary_input.is_symlink() or lineage_input.is_symlink():
        raise RuntimeError("dataset summary and lineage must be regular files")
    summary_path = summary_input.resolve()
    lineage_path = lineage_input.resolve()
    if not summary_path.is_file() or not lineage_path.is_file():
        raise FileNotFoundError(
            f"missing dataset receipt: {summary_path} / {lineage_path}"
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with lineage_path.open(encoding="utf-8") as handle:
        lineage = json.load(handle)
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise RuntimeError(f"{summary_path}: dataset summary is not complete")
    if not isinstance(lineage, dict) or lineage.get("status") != "complete":
        raise RuntimeError(f"{lineage_path}: dataset lineage is not complete")

    lmdb_path = Path(args.train_path).resolve()
    data_path = lmdb_path / "data.mdb"
    if not lmdb_path.is_dir() or data_path.is_symlink() or not data_path.is_file():
        raise RuntimeError(f"invalid formal training LMDB: {lmdb_path}")
    data_sha = _sha256(data_path)
    if (
        Path(summary.get("lmdb", "")).resolve() != lmdb_path
        or summary.get("data_mdb_sha256") != data_sha
        or _require_exact_audit_int(
            summary.get("entries"),
            "dataset summary entries",
        )
        != train_samples
        or train_samples != 127_309
        or _require_exact_audit_int(
            summary.get("train_clips"),
            "dataset summary train_clips",
        )
        != 13687
    ):
        raise RuntimeError(
            f"{summary_path}: LMDB/sample/frozen-train receipt mismatch"
        )

    expected_speakers = {
        "oliver": 0,
        "chemistry": 1,
        "seth": 2,
        "conan": 3,
    }
    if args.formal_stage == "base":
        protocol = lineage.get("protocol")
        expected_forbidden = {
            "ASR",
            "TextGrid",
            "vocabulary",
            "CLIP",
            "emotion",
            "semantic",
            "SemGate",
            "Sparse",
        }
        checkpoints = lineage.get("formal_checkpoints")
        if (
            summary.get("format") != "semtalk_show_base_lmdb_summary_v1"
            or lineage.get("format")
            != "semtalk_show_base_feature_lineage_v1"
            or Path(summary.get("lineage_json", "")).resolve() != lineage_path
            or summary.get("lineage_json_sha256") != _sha256(lineage_path)
            or _require_exact_audit_int(
                lineage.get("entries"),
                "Base lineage entries",
            )
            != train_samples
            or _require_exact_audit_int(
                lineage.get("train_clips"),
                "Base lineage train_clips",
            )
            != 13687
            or not isinstance(protocol, dict)
            or protocol.get("scope") != "SemTalk Base only"
            or protocol.get("split") != "train"
            or _require_exact_audit_int_mapping(
                protocol.get("speakers"),
                expected_speakers,
                "Base protocol speakers",
            )
            != expected_speakers
            or _require_exact_audit_int(
                protocol.get("window_length"),
                "Base protocol window_length",
            )
            != 64
            or _require_exact_audit_int(
                protocol.get("stride"),
                "Base protocol stride",
            )
            != 20
            or protocol.get("in_word")
            != "int64_all_zero_unused_placeholder"
            or set(protocol.get("forbidden_components", []))
            != expected_forbidden
            or not isinstance(checkpoints, dict)
            or set(checkpoints) != {
                "face",
                "hands",
                "upper",
                "lower",
                "global",
            }
            or any(
                not isinstance(record, dict)
                or record.get("formal_stage") != stage
                for stage, record in checkpoints.items()
            )
            or {
                key: lineage.get("source_receipt", {}).get(key)
                for key in ("origin", "commit", "tree")
            }
            != {
                key: current_source.get(key)
                for key in ("origin", "commit", "tree")
            }
        ):
            raise RuntimeError(
                "Base dataset summary/lineage are not mutually bound"
            )
    else:
        representation_format = summary.get("format")
        representation_protocol = summary.get("protocol")
        if (
            representation_format
            != "semtalk_show_representation_lmdb_v2_global_foot"
        ):
            raise RuntimeError(
                "formal prerequisites require the Global-foot v2 receipt"
            )
        if (
            not isinstance(representation_protocol, dict)
            or representation_protocol.get("split") != "train"
            or _require_exact_audit_int(
                representation_protocol.get("window_length"),
                "representation protocol window_length",
            )
            != 64
            or _require_exact_audit_int(
                representation_protocol.get("stride"),
                "representation protocol stride",
            )
            != 20
            or _require_exact_audit_int_mapping(
                representation_protocol.get("speaker_map"),
                expected_speakers,
                "representation protocol speaker_map",
            )
            != expected_speakers
        ):
            raise RuntimeError(
                "invalid four-speaker representation training protocol"
            )
        if summary.get("protocol", {}).get(
            "global_foot_fastpath"
        ) != {
            "enabled": True,
            "contract": "semtalk_show_global_foot_fastpath_v1",
            "field": "lower_foot_local",
            "shape": [64, 4, 3],
            "dtype": "float32",
            "activation_env": (
                "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH=1"
            ),
        }:
            raise RuntimeError(
                "invalid Global-foot fastpath representation receipt"
            )
        if {
            key: summary.get("source_receipt", {}).get(key)
            for key in ("origin", "commit", "tree")
        } != {
            key: current_source.get(key)
            for key in ("origin", "commit", "tree")
        }:
            raise RuntimeError(
                "representation/training source receipt mismatch"
            )
        expected_fastpath = (
            representation_format.endswith("_v2_global_foot")
            and args.formal_stage == "global"
        )
        observed_fastpath = os.environ.get(
            "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH",
            "0",
        )
        if observed_fastpath != ("1" if expected_fastpath else "0"):
            raise RuntimeError(
                "Global-foot fastpath environment does not match the "
                "representation receipt and formal stage"
            )
        if (
            summary_path != lineage_path
            or lineage != summary
        ):
            raise RuntimeError(
                "representation training must bind the exact LMDB summary "
                "as its lineage manifest"
            )
    parity_receipt = None
    if args.formal_stage == "global":
        if (
            not args.global_fastpath_parity_bundle
            or not args.expected_global_fastpath_parity_sha256
        ):
            raise RuntimeError("Global training requires a parity bundle")
        parity_input = Path(args.global_fastpath_parity_bundle)
        if parity_input.is_symlink() or not parity_input.is_file():
            raise FileNotFoundError(parity_input)
        parity_path = parity_input.resolve()
        parity_sha = _sha256(parity_path)
        expected_parity_sha = (
            args.expected_global_fastpath_parity_sha256.strip().lower()
        )
        if (
            len(expected_parity_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_parity_sha
            )
        ):
            raise RuntimeError("Global parity expected SHA is not lowercase SHA-256")
        if parity_sha != expected_parity_sha:
            raise RuntimeError("Global parity bundle SHA mismatch")
        with parity_path.open(encoding="utf-8") as handle:
            parity = json.load(handle)
        if (
            parity.get("format")
            != "semtalk_show_global_foot_parity_suite_v1"
            or parity.get("status") != "pass"
            or parity.get("contract")
            != "semtalk_show_global_foot_fastpath_v1"
            or _require_exact_audit_int_mapping(
                parity.get("speakers"),
                expected_speakers,
                "Global parity speakers",
            )
            != expected_speakers
            or parity.get("canonical_receipt")
            != summary.get("canonical_receipt")
            or {
                key: parity.get("source_receipt", {}).get(key)
                for key in ("origin", "commit", "tree")
            }
            != {
                key: current_source.get(key)
                for key in ("origin", "commit", "tree")
            }
        ):
            raise RuntimeError("invalid Global parity bundle")
        checker_path = (
            Path(__file__).resolve().parent
            / "scripts"
            / "show_base"
            / "check_global_foot_fastpath_parity.py"
        )
        if (
            checker_path.is_symlink()
            or not checker_path.is_file()
            or Path(str(parity.get("checker", ""))).resolve()
            != checker_path
            or parity.get("checker_sha256") != _sha256(checker_path)
        ):
            raise RuntimeError("Global parity checker source mismatch")
        smplx_asset = (
            Path(args.data_path_1).resolve()
            / "smplx_models"
            / "smplx"
            / "SMPLX_NEUTRAL_2020.npz"
        )
        if smplx_asset.is_symlink() or not smplx_asset.is_file():
            raise FileNotFoundError(smplx_asset)
        smplx_sha = _sha256(smplx_asset)
        if parity.get("smplx_asset_sha256") != smplx_sha:
            raise RuntimeError("Global parity/SMPL-X asset SHA mismatch")
        reports = parity.get("reports")
        report_speakers = set()
        if isinstance(reports, list):
            for record in reports:
                if not isinstance(record, dict):
                    break
                speaker = record.get("speaker")
                speaker_id = _require_exact_audit_int(
                    record.get("speaker_id"),
                    "Global parity report speaker_id",
                )
                report_speakers.add((speaker, speaker_id))
        if (
            not isinstance(reports, list)
            or len(reports) != 4
            or report_speakers != set(expected_speakers.items())
        ):
            raise RuntimeError(
                "Global parity suite does not cover all four SHOW speakers"
            )
        canonical_receipt = summary.get("canonical_receipt")
        if not isinstance(canonical_receipt, dict):
            raise RuntimeError("Global parity lacks a canonical receipt")
        canonical_input = Path(str(canonical_receipt.get("manifest", "")))
        if canonical_input.is_symlink() or not canonical_input.is_file():
            raise FileNotFoundError(canonical_input)
        canonical_manifest = canonical_input.resolve()
        if _sha256(canonical_manifest) != canonical_receipt.get(
            "manifest_sha256"
        ):
            raise RuntimeError("Global parity canonical manifest SHA mismatch")
        selected_rows: dict[str, dict[str, Any]] = {}
        with canonical_manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"{canonical_manifest}:{line_number}: non-object row"
                    )
                speaker = str(row.get("speaker"))
                speaker_id = _require_exact_audit_int(
                    row.get("speaker_id"),
                    f"{canonical_manifest}:{line_number}: speaker_id",
                )
                frames = _require_exact_audit_int(
                    row.get("frames"),
                    f"{canonical_manifest}:{line_number}: frames",
                )
                if (
                    row.get("split") == "train"
                    and speaker in expected_speakers
                    and speaker_id == expected_speakers[speaker]
                    and frames >= 64
                    and speaker not in selected_rows
                ):
                    selected_rows[speaker] = row
        if set(selected_rows) != set(expected_speakers):
            raise RuntimeError(
                "Global parity canonical manifest lacks four speaker windows"
            )
        expected_thresholds = {
            "contact": 0.95,
            "loss_atol": 2e-6,
            "loss_rtol": 1e-5,
            "gradient_atol": 2e-6,
            "gradient_rtol": 1e-5,
            "cache_atol": 1e-6,
            "cache_rtol": 1e-6,
            "wrong_axis_zero_atol": 2e-7,
        }
        expected_optimizer = {
            "name": "Adam",
            "lr": 1.5e-4,
            "betas": [0.5, 0.999],
            "weight_decay": 0.0,
            "steps": 1,
        }
        expected_loss_checks = {
            "contact",
            "vertex",
            "vertex_velocity_wrong_axis",
            "vertex_acceleration_wrong_axis",
            "foot",
            "total",
        }
        expected_check_keys = {
            "cache",
            "losses",
            "gradient",
            "adam_parameter",
            "adam_exp_avg",
            "adam_exp_avg_sq",
        }

        def validate_close(
            value: Any,
            *,
            atol: float,
            rtol: float,
            label: str,
        ) -> None:
            if not isinstance(value, dict) or set(value) != {
                "max_abs",
                "max_rel",
            }:
                raise RuntimeError(f"{label}: invalid close-check payload")
            max_abs = float(value["max_abs"])
            max_rel = float(value["max_rel"])
            if (
                not np.isfinite(max_abs)
                or not np.isfinite(max_rel)
                or max_abs < 0
                or max_rel < 0
                or (max_abs > atol and max_rel > rtol)
            ):
                raise RuntimeError(
                    f"{label}: parity delta exceeds its recorded tolerance"
                )

        report_receipts = []
        observed_report_paths: set[Path] = set()
        for record in reports:
            assert isinstance(record, dict)
            report_input = Path(str(record.get("report", "")))
            if report_input.is_symlink() or not report_input.is_file():
                raise FileNotFoundError(report_input)
            report_path = report_input.resolve()
            report_sha = _sha256(report_path)
            with report_path.open(encoding="utf-8") as handle:
                report_payload = json.load(handle)
            speaker = str(record["speaker"])
            selected_row = selected_rows[speaker]
            checks = report_payload.get("checks")
            if (
                report_path in observed_report_paths
                or report_sha != record.get("report_sha256")
                or report_payload != record.get("payload")
                or record.get("clip_id") != selected_row.get("clip_id")
                or report_payload.get("status") != "pass"
                or report_payload.get("contract")
                != "semtalk_show_global_foot_fastpath_v1"
                or Path(
                    str(report_payload.get("canonical_npz", ""))
                ).resolve()
                != Path(str(selected_row.get("canonical_npz", ""))).resolve()
                or report_payload.get("canonical_npz_sha256")
                != selected_row.get("canonical_npz_sha256")
                or Path(
                    str(report_payload.get("lower_foot_local", ""))
                ).resolve()
                != Path(
                    str(selected_row.get("lower_foot_local", ""))
                ).resolve()
                or report_payload.get("lower_foot_local_sha256")
                != selected_row.get("lower_foot_local_sha256")
                or report_payload.get("smplx_asset_sha256") != smplx_sha
                or report_payload.get("window")
                != {"start_frame": 0, "frames": 64}
                or report_payload.get("thresholds") != expected_thresholds
                or report_payload.get("optimizer") != expected_optimizer
                or not isinstance(checks, dict)
                or set(checks) != expected_check_keys
                or not isinstance(checks.get("losses"), dict)
                or set(checks["losses"]) != expected_loss_checks
            ):
                raise RuntimeError(
                    f"invalid Global parity report: {report_path}"
                )
            observed_report_paths.add(report_path)
            wrong_axis = float(
                report_payload.get("legacy_wrong_axis_max_loss", float("nan"))
            )
            if (
                not np.isfinite(wrong_axis)
                or wrong_axis < 0
                or wrong_axis > expected_thresholds["wrong_axis_zero_atol"]
            ):
                raise RuntimeError(
                    f"{report_path}: invalid wrong-axis parity loss"
                )
            validate_close(
                checks["cache"],
                atol=expected_thresholds["cache_atol"],
                rtol=expected_thresholds["cache_rtol"],
                label=f"{report_path}:cache",
            )
            for name, value in checks["losses"].items():
                validate_close(
                    value,
                    atol=expected_thresholds["loss_atol"],
                    rtol=expected_thresholds["loss_rtol"],
                    label=f"{report_path}:losses.{name}",
                )
            for name in (
                "gradient",
                "adam_parameter",
                "adam_exp_avg",
                "adam_exp_avg_sq",
            ):
                validate_close(
                    checks[name],
                    atol=expected_thresholds["gradient_atol"],
                    rtol=expected_thresholds["gradient_rtol"],
                    label=f"{report_path}:{name}",
                )
            report_receipts.append(
                {
                    "speaker": record["speaker"],
                    "speaker_id": record["speaker_id"],
                    "path": str(report_path),
                    "sha256": report_sha,
                }
            )
        parity_receipt = {
            "path": str(parity_path),
            "sha256": parity_sha,
            "format": parity["format"],
            "status": parity["status"],
            "speakers": parity["speakers"],
            "checker": str(checker_path),
            "checker_sha256": parity["checker_sha256"],
            "smplx_asset": str(smplx_asset),
            "smplx_asset_sha256": smplx_sha,
            "reports": report_receipts,
        }
    elif (
        args.global_fastpath_parity_bundle is not None
        or args.expected_global_fastpath_parity_sha256 is not None
    ):
        raise RuntimeError(
            "Global parity arguments are forbidden outside the global stage"
        )
    return {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "lineage": str(lineage_path),
        "lineage_sha256": _sha256(lineage_path),
        "lmdb": str(lmdb_path),
        "data_mdb_sha256": data_sha,
        "entries": train_samples,
        "train_clips": 13687,
        "split_label": "SHOW available frozen subset",
        "source_binding": {
            key: current_source[key] for key in ("origin", "commit", "tree")
        },
        "smplx_asset": smplx_asset_receipt,
        "global_fastpath_parity": parity_receipt,
    }


def _finite_tree(value: Any, prefix: str) -> list[str]:
    bad: list[str] = []
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            bad.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            bad.extend(_finite_tree(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            bad.extend(_finite_tree(child, f"{prefix}[{index}]"))
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        bad.append(prefix)
    return bad


def _local_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def _all_rng_states(world_size: int) -> list[dict[str, Any]] | None:
    local = _local_rng_state()
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [state for state in gathered if state is not None]


def _tracker_snapshot(trainer: Any) -> dict[str, dict[str, float | int]]:
    trainer._flush_train_metrics()
    snapshot: dict[str, dict[str, float | int]] = {}
    for name, states in trainer.tracker.loss_meters.items():
        meter = states["train"]
        if meter.count:
            snapshot[name] = {
                "avg": float(meter.avg),
                "count": int(meter.count),
            }
    return snapshot


def _rvq_ema_state(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    """Capture rank-local EMA state omitted by QuantizeEMAReset.state_dict()."""
    state: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "QuantizeEMAReset":
            continue
        initialized = bool(module.init)
        item: dict[str, Any] = {"init": initialized}
        if initialized:
            if module.code_sum is None or module.code_count is None:
                raise RuntimeError(f"initialized RVQ EMA module {name} has empty state")
            item["code_sum"] = module.code_sum.detach().cpu()
            item["code_count"] = module.code_count.detach().cpu()
        state[name] = item
    return state


def _restore_rvq_ema_state(
    model: torch.nn.Module,
    state: dict[str, dict[str, Any]],
) -> None:
    modules = {
        name: module
        for name, module in model.named_modules()
        if module.__class__.__name__ == "QuantizeEMAReset"
    }
    if set(modules) != set(state):
        raise RuntimeError(
            "resume RVQ EMA module set mismatch: "
            f"model={sorted(modules)} checkpoint={sorted(state)}"
        )
    for name, module in modules.items():
        item = state[name]
        if not isinstance(item, dict) or type(item.get("init")) is not bool:
            raise RuntimeError(
                f"resume RVQ EMA module {name} init must be an exact boolean"
            )
        initialized = item["init"]
        expected_keys = (
            {"init", "code_sum", "code_count"}
            if initialized
            else {"init"}
        )
        if set(item) != expected_keys:
            raise RuntimeError(
                f"resume RVQ EMA module {name} has invalid state keys"
            )
        module.init = initialized
        if initialized:
            module.code_sum = item["code_sum"].to(
                device=module.codebook.device,
                dtype=module.codebook.dtype,
            )
            module.code_count = item["code_count"].to(
                device=module.codebook.device,
                dtype=module.codebook.dtype,
            )
        else:
            module.code_sum = None
            module.code_count = None


def _rvq_ema_invariant_errors(model: torch.nn.Module) -> list[str]:
    errors: list[str] = []
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "QuantizeEMAReset"
    ]
    if modules and len(modules) != 6:
        errors.append(f"expected 6 RVQ EMA layers, found {len(modules)}")
    for name, module in modules:
        if not bool(module.init):
            errors.append(f"{name}.init is false")
            continue
        if module.code_sum is None or module.code_count is None:
            errors.append(f"{name} has empty EMA accumulators")
            continue
        if module.code_sum.shape != module.codebook.shape:
            errors.append(f"{name}.code_sum shape mismatch")
        if module.code_count.shape != (module.codebook.shape[0],):
            errors.append(f"{name}.code_count shape mismatch")
        if bool((module.code_count < 0).any().item()):
            errors.append(f"{name}.code_count contains negative values")
        if float(module.code_count.sum().item()) <= 0:
            errors.append(f"{name}.code_count has non-positive total")
    return errors


def _validate_formal_stage(args: Any) -> None:
    common = {
        "dataset": "show_base",
        "training_speakers": [0, 1, 2, 3],
        "ori_joints": "beat_smplx_joints",
        "batch_size": 64,
        "pose_length": 64,
        "pre_frames": 4,
        "stride": 20,
        "opt_betas": [0.5, 0.999],
        "opt": "adam",
        "weight_decay": 0.0,
        "lr_policy": "step",
        "decay_rate": 0.3,
        "warmup_epochs": 0,
        "amsgrad": False,
        "pose_fps": 30,
        "vae_codebook_size": 256,
        "vae_grow": [1, 1, 2, 1],
        "vae_quantizer_lambda": 1.0,
        "variational": False,
        "rot6d": True,
        "dropout_prob": 0.3,
        "pretrain": False,
        "sparse": 0,
        "data_path": "",
        "cache_path": "",
        "e_path": "",
        "e_name": None,
        "test_path": "",
        "word_cache": False,
        "word_rep": "disabled_zero_placeholder",
        "t_pre_encoder": "disabled",
        "word_index_num": 0,
        "word_dims": 0,
        "word_f": 0,
        "freeze_wordembed": True,
        "hubert_mean_path": "",
        "hubert_std_path": "",
        "audio_infer_path": "",
        "base_ckpt": "",
        "test_ckpt": "",
        "deterministic": True,
        "benchmark": True,
        "cudnn_enabled": True,
        "log_period": 1_989,
        "save_every": 5,
    }
    stages: dict[str, dict[str, Any]] = {
        "face": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "aeface",
            "train_rvq": True,
            "tar_joints": "beat_smplx_face",
            "vae_test_dim": 106,
            "vae_layer": 2,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 600,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "rvq_face_600.bin",
        },
        "hands": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "ae",
            "train_rvq": True,
            "tar_joints": "beat_smplx_hands",
            "vae_test_dim": 180,
            "vae_layer": 2,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 500,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "rvq_hands_500.bin",
        },
        "upper": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "ae",
            "train_rvq": True,
            "tar_joints": "beat_smplx_upper",
            "vae_test_dim": 78,
            "vae_layer": 2,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 500,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 9999,
            "final_ckpt_name": "rvq_upper_500.bin",
        },
        "lower": {
            "model": "rvq",
            "g_name": "RVQVAE",
            "trainer": "aelower",
            "train_rvq": True,
            "tar_joints": "beat_smplx_lower",
            "vae_test_dim": 61,
            "vae_layer": 4,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 600,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "rvq_lower_600.bin",
        },
        "global": {
            "model": "motion_representation",
            "g_name": "VAEConvZero",
            "trainer": "aelowerfoot",
            "train_rvq": True,
            "tar_joints": "beat_smplx_lower",
            "vae_test_dim": 61,
            "vae_layer": 4,
            "vae_length": 256,
            "rec_weight": 1.0,
            "rec_pos_weight": 1.0,
            "rec_ver_weight": 1.0,
            "grad_norm": 0.0,
            "epochs": 1700,
            "random_seed": 2021,
            "lr_base": 3e-4,
            "decay_epochs": 780,
            "final_ckpt_name": "last_1700_foot.bin",
        },
        "base": {
            "model": "semtalk",
            "g_name": "semtalk_base",
            "trainer": "semtalk_base",
            "train_rvq": False,
            "tar_joints": "beat_smplx_full",
            "pose_dims": 330,
            "audio_f": 256,
            "motion_f": 256,
            "hidden_size": 768,
            "lf": 3,
            "ll": 3,
            "lu": 3,
            "lh": 3,
            "cf": 1,
            "cl": 1,
            "cu": 1,
            "ch": 1,
            "vae_test_dim": 330,
            "vae_layer": 4,
            "vae_length": 240,
            "rec_weight": 1.0,
            "rec_pos_weight": 0.0,
            "rec_ver_weight": 0.0,
            "grad_norm": 0.99,
            "epochs": 400,
            "random_seed": 43,
            "lr_base": 1e-4,
            "decay_epochs": 999,
            "final_ckpt_name": "semtalk_base_epoch_400.bin",
        },
    }
    if args.formal_stage not in stages:
        raise RuntimeError(
            f"--formal_stage must be one of {sorted(stages)}, got {args.formal_stage!r}"
        )
    expected = {**common, **stages[args.formal_stage]}
    mismatches = []
    for name, value in expected.items():
        actual = getattr(args, name)
        if actual != value:
            mismatches.append(f"{name}={actual!r} expected {value!r}")
    if mismatches:
        raise RuntimeError(
            f"formal stage {args.formal_stage} config mismatch: "
            + "; ".join(mismatches)
        )
    if args.load_ckpt not in {None, ""}:
        raise RuntimeError(
            "formal SHOW training must start from scratch; --load_ckpt is forbidden"
        )
    if args.d_name is not None:
        raise RuntimeError("formal Base-only training forbids a discriminator")


def _load_resume(
    trainer: Any,
    resume_path: Path,
    rank: int,
    world_size: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_summary_sha256: str,
    data_mdb_sha256: str,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    source_receipt_sha256: str,
    candidate_manifest_path: Path,
) -> tuple[
    int,
    dict[str, dict[str, float | int]],
    float,
    str | None,
    int,
    dict[str, Any] | None,
]:
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "semtalk_show_train_resume_v5":
        raise RuntimeError("unsupported or unsafe resume checkpoint format")
    completed_epochs = _require_exact_audit_int(
        payload.get("completed_epochs"),
        "resume completed_epochs",
    )
    optimizer_updates = _require_exact_audit_int(
        payload.get("optimizer_updates"),
        "resume optimizer_updates",
    )
    if (
        _require_exact_audit_int(
            payload.get("world_size"),
            "resume world_size",
        )
        != world_size
    ):
        raise RuntimeError(
            f"resume world_size={payload['world_size']} does not match {world_size}"
        )
    if (
        _require_exact_audit_int(
            payload.get("train_samples"),
            "resume train_samples",
        )
        != len(trainer.train_data)
    ):
        raise RuntimeError("resume train sample count does not match current dataset")
    if (
        _require_exact_audit_int(
            payload.get("updates_per_epoch"),
            "resume updates_per_epoch",
        )
        != trainer.train_length
    ):
        raise RuntimeError("resume updates/epoch does not match current dataloader")
    if (
        _require_exact_audit_int(
            payload.get("batch_size"),
            "resume batch_size",
        )
        != trainer.args.batch_size
    ):
        raise RuntimeError("resume batch size does not match current config")
    if payload["config_sha256"] != config_sha256:
        raise RuntimeError("resume training config fingerprint does not match")
    if payload.get("lineage_manifest_sha256") != lineage_sha256:
        raise RuntimeError("resume lineage manifest fingerprint does not match")
    if payload.get("dataset_summary_sha256") != dataset_summary_sha256:
        raise RuntimeError("resume dataset summary fingerprint does not match")
    if payload.get("data_mdb_sha256") != data_mdb_sha256:
        raise RuntimeError("resume LMDB fingerprint does not match")
    if payload.get("dataset_receipt_sha256") != _payload_sha256(dataset_receipt):
        raise RuntimeError("resume dataset receipt fingerprint does not match")
    if payload.get("smplx_asset_receipt") != dataset_receipt.get("smplx_asset"):
        raise RuntimeError("resume SMPL-X asset receipt does not match")
    if payload.get("source_receipt_sha256") != source_receipt_sha256:
        raise RuntimeError("resume source checkout fingerprint does not match")
    expected_optimizer_updates = completed_epochs * trainer.train_length
    if optimizer_updates != expected_optimizer_updates:
        raise RuntimeError(
            "resume actual optimizer update count does not match completed epochs"
        )
    committed_candidate_count = _require_exact_audit_int(
        payload.get("candidate_manifest_entry_count"),
        "resume candidate_manifest_entry_count",
    )
    expected_committed_count = (
        completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS
        if trainer.args.formal_stage == "base"
        else 0
    )
    if committed_candidate_count != expected_committed_count:
        raise RuntimeError("resume Base candidate committed entry count mismatch")
    transaction = _inspect_base_candidate_transaction(
        candidate_manifest_path,
        formal_stage=trainer.args.formal_stage,
        updates_per_epoch=trainer.train_length,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    current_candidate_entries = _base_candidate_entries(
        candidate_manifest_path
    )
    current_candidate_count = len(current_candidate_entries)
    allowed_candidate_counts = {committed_candidate_count}
    if (
        trainer.args.formal_stage == "base"
        and completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS
        == BASE_CANDIDATE_INTERVAL_EPOCHS // 2
    ):
        allowed_candidate_counts.add(committed_candidate_count + 1)
    if current_candidate_count not in allowed_candidate_counts:
        raise RuntimeError(
            "resume Base candidate manifest has more than one uncommitted "
            "future candidate"
        )
    if transaction is None:
        if current_candidate_count != committed_candidate_count:
            raise RuntimeError(
                "future Base candidate manifest is missing its transaction"
            )
        candidate_manifest_sha = _validate_base_candidate_manifest(
            candidate_manifest_path,
            formal_stage=trainer.args.formal_stage,
            completed_epochs=(
                current_candidate_count * BASE_CANDIDATE_INTERVAL_EPOCHS
            ),
            updates_per_epoch=trainer.train_length,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt_sha256=source_receipt_sha256,
        )
    else:
        candidate_manifest_sha = transaction["manifest_sha256"]
        target_epoch = _require_exact_audit_int(
            transaction.get("target_epoch"),
            "candidate transaction target_epoch",
        )
        stale_committed = (
            target_epoch == completed_epochs
            and transaction["manifested"]
            and transaction["state"] == "manifest_committed"
            and current_candidate_count == committed_candidate_count
        )
        future_transaction = (
            target_epoch
            == completed_epochs + BASE_CANDIDATE_INTERVAL_EPOCHS // 2
            and completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS
            == BASE_CANDIDATE_INTERVAL_EPOCHS // 2
            and _require_exact_audit_int(
                transaction.get("previous_entry_count"),
                "candidate transaction previous_entry_count",
            )
            == committed_candidate_count
        )
        if not stale_committed and not future_transaction:
            raise RuntimeError(
                "Base candidate transaction is not aligned to the exact "
                "resume boundary"
            )
        if (
            current_candidate_count == committed_candidate_count + 1
            and transaction["core"]["previous_manifest_sha256"]
            != payload.get("candidate_manifest_sha256")
        ):
            raise RuntimeError(
                "future Base candidate transaction previous manifest "
                "fingerprint mismatch"
            )
    committed_entries = current_candidate_entries[:committed_candidate_count]
    if payload.get(
        "candidate_manifest_entries_sha256"
    ) != _base_candidate_entries_sha256(committed_entries):
        raise RuntimeError("resume Base candidate committed prefix mismatch")
    if (
        current_candidate_count == committed_candidate_count
        and payload.get("candidate_manifest_sha256")
        != candidate_manifest_sha
    ):
        raise RuntimeError(
            "resume Base candidate manifest fingerprint does not match"
        )
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    ema_modules = _rvq_ema_state(trainer.model)
    if ema_modules:
        if "rvq_ema_state" not in payload:
            raise RuntimeError("RVQ resume checkpoint is missing EMA state")
        _restore_rvq_ema_state(trainer.model, payload["rvq_ema_state"])
    elif payload.get("rvq_ema_state"):
        raise RuntimeError("non-RVQ model received unexpected RVQ EMA state")
    trainer.opt.load_state_dict(payload["optimizer_state"])
    trainer.opt_s.load_state_dict(payload["scheduler_state"])
    trainer._restore_formal_optimizer_updates(optimizer_updates)
    rng_states = payload["rng_states"]
    if len(rng_states) != world_size:
        raise RuntimeError("resume RNG state count does not match world size")
    _restore_rng_state(rng_states[rank])
    bad = _finite_tree(trainer.model.state_dict(), "model")
    bad.extend(_finite_tree(_rvq_ema_state(trainer.model), "rvq_ema"))
    bad.extend(_finite_tree(trainer.opt.state_dict(), "optimizer"))
    bad.extend(_finite_tree(trainer.opt_s.state_dict(), "scheduler"))
    bad.extend(_rvq_ema_invariant_errors(trainer.model))
    if bad:
        raise FloatingPointError(
            "invalid state immediately after resume: " + ", ".join(bad[:20])
        )
    last_metrics = payload.get("last_metrics")
    if not isinstance(last_metrics, dict):
        raise RuntimeError("resume checkpoint is missing last_metrics")
    for metric_name, metric in last_metrics.items():
        average = metric.get("avg") if isinstance(metric, dict) else None
        if (
            not isinstance(metric_name, str)
            or not isinstance(metric, dict)
            or set(metric) != {"avg", "count"}
            or isinstance(average, bool)
            or not isinstance(average, Real)
            or not bool(np.isfinite(float(average)))
            or _require_exact_audit_int(
                metric.get("count"),
                f"resume last_metrics[{metric_name!r}].count",
            )
            < 1
        ):
            raise RuntimeError(
                f"resume checkpoint has invalid metric receipt {metric_name!r}"
            )
    started_unix = float(payload.get("started_unix", 0.0))
    if not np.isfinite(started_unix) or started_unix <= 0:
        raise RuntimeError("resume checkpoint has invalid started_unix")
    return (
        completed_epochs,
        last_metrics,
        started_unix,
        candidate_manifest_sha,
        current_candidate_count,
        transaction,
    )


def _save_resume(
    trainer: Any,
    path: Path,
    completed_epochs: int,
    rng_states: list[dict[str, Any]],
    world_size: int,
    train_samples: int,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_summary_sha256: str,
    data_mdb_sha256: str,
    dataset_receipt: dict[str, Any],
    source_receipt_sha256: str,
    optimizer_updates: int,
    candidate_manifest_sha256: str | None,
    candidate_manifest_entry_count: int,
    candidate_manifest_entries_sha256: str,
    last_metrics: dict[str, dict[str, float | int]],
    started_unix: float,
) -> None:
    completed_epochs = _require_exact_audit_int(
        completed_epochs,
        "resume save completed_epochs",
    )
    world_size = _require_exact_audit_int(
        world_size,
        "resume save world_size",
    )
    train_samples = _require_exact_audit_int(
        train_samples,
        "resume save train_samples",
    )
    updates_per_epoch = _require_exact_audit_int(
        updates_per_epoch,
        "resume save updates_per_epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "resume save optimizer_updates",
    )
    candidate_manifest_entry_count = _require_exact_audit_int(
        candidate_manifest_entry_count,
        "resume save candidate_manifest_entry_count",
    )
    batch_size = _require_exact_audit_int(
        trainer.args.batch_size,
        "resume save batch_size",
    )
    expected_optimizer_updates = completed_epochs * updates_per_epoch
    if (
        optimizer_updates != expected_optimizer_updates
        or trainer.formal_optimizer_updates != expected_optimizer_updates
    ):
        raise RuntimeError(
            "refusing to save resume with a derived optimizer update count"
        )
    expected_candidate_count = (
        completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS
        if trainer.args.formal_stage == "base"
        else 0
    )
    if candidate_manifest_entry_count != expected_candidate_count:
        raise RuntimeError(
            "refusing to save resume with an uncommitted candidate count"
        )
    if (
        not isinstance(candidate_manifest_entries_sha256, str)
        or len(candidate_manifest_entries_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate_manifest_entries_sha256
        )
    ):
        raise RuntimeError("invalid candidate entry-prefix SHA-256")
    if (
        expected_candidate_count == 0
        and candidate_manifest_sha256 is not None
    ) or (
        expected_candidate_count > 0
        and (
            not isinstance(candidate_manifest_sha256, str)
            or len(candidate_manifest_sha256) != 64
        )
    ):
        raise RuntimeError("invalid Base candidate manifest SHA binding")
    _atomic_torch_save(
        path,
        {
            "format": "semtalk_show_train_resume_v5",
            "completed_epochs": completed_epochs,
            "world_size": world_size,
            "train_samples": train_samples,
            "updates_per_epoch": updates_per_epoch,
            "batch_size": batch_size,
            "config_sha256": config_sha256,
            "lineage_manifest_sha256": lineage_sha256,
            "dataset_summary_sha256": dataset_summary_sha256,
            "data_mdb_sha256": data_mdb_sha256,
            "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
            "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
            "source_receipt_sha256": source_receipt_sha256,
            "optimizer_updates": optimizer_updates,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "candidate_manifest_entry_count": candidate_manifest_entry_count,
            "candidate_manifest_entries_sha256": (
                candidate_manifest_entries_sha256
            ),
            "last_metrics": last_metrics,
            "started_unix": started_unix,
            "model_state": trainer.model.state_dict(),
            "rvq_ema_state": _rvq_ema_state(trainer.model),
            "optimizer_state": trainer.opt.state_dict(),
            "scheduler_state": trainer.opt_s.state_dict(),
            "rng_states": rng_states,
        },
    )


def _model_payload(
    trainer: Any,
    *,
    formal_stage: str,
    config_sha256: str,
    lineage_sha256: str,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    optimizer_updates: int,
    candidate_manifest_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "model audit optimizer_updates",
    )
    if candidate_manifest_receipt is not None:
        for key in ("entries", "last_epoch", "last_optimizer_updates"):
            _require_exact_audit_int(
                candidate_manifest_receipt.get(key),
                f"model audit Base candidate manifest {key}",
            )
    return {
        "model_state": trainer.model.state_dict(),
        "audit": {
            "format": "semtalk_show_model_v2",
            "formal_stage": formal_stage,
            "config_sha256": config_sha256,
            "lineage_manifest_sha256": lineage_sha256,
            "dataset_summary_sha256": dataset_receipt["summary_sha256"],
            "data_mdb_sha256": dataset_receipt["data_mdb_sha256"],
            "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
            "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
            "source_receipt": source_receipt,
            "source_receipt_sha256": _payload_sha256(source_receipt),
            "optimizer_updates": optimizer_updates,
            "base_candidate_manifest": candidate_manifest_receipt,
        },
    }


def _base_candidate_audit(
    *,
    epoch: int,
    optimizer_updates: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    epoch = _require_exact_audit_int(
        epoch,
        "Base candidate audit epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "Base candidate audit optimizer_updates",
    )
    return {
        "format": "semtalk_show_base_candidate_model_v1",
        "formal_stage": "base",
        "candidate_epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
    }


def _candidate_filename(epoch: int, optimizer_updates: int) -> str:
    epoch = _require_exact_audit_int(epoch, "Base candidate filename epoch")
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "Base candidate filename optimizer_updates",
    )
    return (
        f"semtalk_base_candidate_epoch_{epoch:04d}"
        f"_step_{optimizer_updates:09d}.bin"
    )


def _candidate_relative_path(epoch: int, optimizer_updates: int) -> str:
    return (
        f"base_candidates/"
        f"{_candidate_filename(epoch, optimizer_updates)}"
    )


def _candidate_transaction_path(manifest_path: Path) -> Path:
    return manifest_path.parent / BASE_CANDIDATE_TRANSACTION_FILENAME


def _candidate_staging_path(manifest_path: Path) -> Path:
    return manifest_path.parent / BASE_CANDIDATE_STAGING_FILENAME


def _load_regular_json(path: Path, label: str) -> dict[str, Any]:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _base_candidate_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if manifest_path.is_symlink():
        raise RuntimeError(
            f"Base candidate manifest must not be a symlink: {manifest_path}"
        )
    if not manifest_path.exists():
        return []
    manifest = _load_regular_json(
        manifest_path,
        "Base candidate manifest",
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("Base candidate manifest entries must be a list")
    return entries


def _base_candidate_entries_sha256(
    entries: list[dict[str, Any]],
) -> str:
    return _payload_sha256(entries)


def _base_candidate_manifest_payload(
    *,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "semtalk_show_base_candidate_manifest_v2",
        "formal_stage": "base",
        "interval_epochs": BASE_CANDIDATE_INTERVAL_EPOCHS,
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
        "entries": entries,
    }


def _base_candidate_transaction_core(
    *,
    epoch: int,
    optimizer_updates: int,
    previous_manifest_sha256: str | None,
    previous_entries: list[dict[str, Any]],
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    epoch = _require_exact_audit_int(
        epoch,
        "Base candidate transaction epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        optimizer_updates,
        "Base candidate transaction optimizer_updates",
    )
    candidate_audit = _base_candidate_audit(
        epoch=epoch,
        optimizer_updates=optimizer_updates,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    return {
        "format": "semtalk_show_base_candidate_transaction_core_v1",
        "formal_stage": "base",
        "candidate_epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "checkpoint": _candidate_relative_path(epoch, optimizer_updates),
        "staging_checkpoint": BASE_CANDIDATE_STAGING_FILENAME,
        "previous_manifest_sha256": previous_manifest_sha256,
        "previous_entry_count": len(previous_entries),
        "previous_entries_sha256": _base_candidate_entries_sha256(
            previous_entries
        ),
        "config_sha256": config_sha256,
        "lineage_manifest_sha256": lineage_sha256,
        "dataset_receipt_sha256": _payload_sha256(dataset_receipt),
        "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
        "source_receipt": source_receipt,
        "source_receipt_sha256": _payload_sha256(source_receipt),
        "model_audit_sha256": _payload_sha256(candidate_audit),
    }


def _base_candidate_transaction_payload(
    core: dict[str, Any],
    *,
    state: str,
    checkpoint_sha256: str | None,
    manifest_sha256: str | None,
) -> dict[str, Any]:
    if state not in {
        "prepared",
        "checkpoint_committed",
        "manifest_committed",
    }:
        raise RuntimeError(f"invalid Base candidate transaction state: {state}")
    return {
        "format": "semtalk_show_base_candidate_transaction_v1",
        "state": state,
        "core": core,
        "core_sha256": _payload_sha256(core),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
    }


def _write_base_candidate_transaction(
    transaction_path: Path,
    core: dict[str, Any],
    *,
    state: str,
    checkpoint_sha256: str | None,
    manifest_sha256: str | None,
) -> dict[str, Any]:
    payload = _base_candidate_transaction_payload(
        core,
        state=state,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
    )
    _atomic_json(transaction_path, payload)
    return payload


def _validate_base_candidate_snapshot(
    manifest_path: Path,
    *,
    expected_sha256: str | None,
    expected_entry_count: int,
    allowed_pending_checkpoint: Path | None = None,
    allow_empty_candidate_dir: bool = False,
) -> list[dict[str, Any]]:
    candidate_dir = manifest_path.parent / "base_candidates"
    if expected_entry_count == 0:
        if expected_sha256 is not None or manifest_path.exists() or (
            manifest_path.is_symlink()
        ):
            raise RuntimeError("unexpected Base candidate snapshot before epoch 10")
        if allowed_pending_checkpoint is None:
            if allow_empty_candidate_dir:
                if candidate_dir.is_symlink() or (
                    candidate_dir.exists()
                    and (
                        not candidate_dir.is_dir()
                        or any(candidate_dir.iterdir())
                    )
                ):
                    raise RuntimeError(
                        "empty transaction candidate directory changed"
                    )
                return []
            if candidate_dir.exists() or candidate_dir.is_symlink():
                raise RuntimeError(
                    "unexpected Base candidate snapshot before epoch 10"
                )
            return []
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            raise RuntimeError("pending Base candidate directory is invalid")
        actual_children = list(candidate_dir.iterdir())
        if len(actual_children) != 1:
            raise RuntimeError("pending Base candidate file set changed")
        child = actual_children[0]
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(
            child_stat.st_mode
        ):
            raise RuntimeError(
                f"pending Base candidate is not a regular file: {child}"
            )
        if child.resolve() != allowed_pending_checkpoint.resolve():
            raise RuntimeError("pending Base candidate path changed")
        return []
    entries = _base_candidate_entries(manifest_path)
    if (
        len(entries) != expected_entry_count
        or expected_sha256 is None
        or _sha256(manifest_path) != expected_sha256
        or candidate_dir.is_symlink()
        or not candidate_dir.is_dir()
    ):
        raise RuntimeError("Base candidate manifest snapshot changed")
    expected_paths = {
        (manifest_path.parent / str(record.get("checkpoint", ""))).resolve()
        for record in entries
        if isinstance(record, dict)
    }
    if allowed_pending_checkpoint is not None:
        expected_paths.add(allowed_pending_checkpoint.resolve())
    actual_children = list(candidate_dir.iterdir())
    for child in actual_children:
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(
            child_stat.st_mode
        ):
            raise RuntimeError(
                f"Base candidate is no longer a regular file: {child}"
            )
    actual_paths = {child.resolve() for child in actual_children}
    expected_file_count = expected_entry_count + int(
        allowed_pending_checkpoint is not None
    )
    if len(expected_paths) != expected_file_count or actual_paths != expected_paths:
        raise RuntimeError("Base candidate file set changed")
    return entries


def _inspect_new_prepared_base_candidate_transaction(
    manifest_path: Path,
    *,
    core: dict[str, Any],
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Strictly verify a transaction just created by this process.

    The recovery inspector intentionally performs a full checkpoint audit,
    because it must distrust artifacts left by an earlier process.  A newly
    prepared transaction has no checkpoint yet, so reloading every historical
    candidate would add quadratic I/O without strengthening this boundary.
    This inspector instead reopens the atomic intent, verifies it byte-for-byte
    at the payload level, and rechecks the immutable manifest/file snapshot.
    """

    transaction_path = _candidate_transaction_path(manifest_path)
    staging_path = _candidate_staging_path(manifest_path)
    expected_payload = _base_candidate_transaction_payload(
        core,
        state="prepared",
        checkpoint_sha256=None,
        manifest_sha256=None,
    )
    transaction = _load_regular_json(
        transaction_path,
        "Base candidate transaction",
    )
    if transaction != expected_payload:
        raise RuntimeError("new Base candidate transaction payload mismatch")

    previous_count = _require_exact_audit_int(
        core.get("previous_entry_count"),
        "new candidate transaction previous_entry_count",
    )
    previous_entries = _validate_base_candidate_snapshot(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_entry_count=previous_count,
    )
    if (
        len(previous_entries) != previous_count
        or _base_candidate_entries_sha256(previous_entries)
        != core["previous_entries_sha256"]
    ):
        raise RuntimeError("new Base candidate transaction prefix mismatch")

    candidate_path = (
        manifest_path.parent / str(core["checkpoint"])
    ).resolve()
    if candidate_path.exists() or candidate_path.is_symlink():
        raise RuntimeError(
            "new Base candidate transaction unexpectedly has a checkpoint"
        )
    if staging_path.exists() or staging_path.is_symlink():
        raise RuntimeError(
            "new Base candidate transaction unexpectedly has a staging file"
        )
    return {
        "path": transaction_path,
        "payload": transaction,
        "core": core,
        "state": "prepared",
        "target_epoch": _require_exact_audit_int(
            core.get("candidate_epoch"),
            "new candidate transaction candidate_epoch",
        ),
        "previous_entry_count": previous_count,
        "current_entry_count": previous_count,
        "candidate_path": candidate_path,
        "candidate_exists": False,
        "checkpoint_sha256": None,
        "manifested": False,
        "manifest_sha256": expected_manifest_sha256,
        "staging_exists": False,
    }


def _validate_base_candidate_manifest(
    manifest_path: Path,
    *,
    formal_stage: str,
    completed_epochs: int,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt_sha256: str,
    allowed_pending_checkpoint: Path | None = None,
    allow_empty_candidate_dir: bool = False,
) -> str | None:
    candidate_dir = manifest_path.parent / "base_candidates"
    expected_epochs = list(
        range(
            BASE_CANDIDATE_INTERVAL_EPOCHS,
            int(completed_epochs) + 1,
            BASE_CANDIDATE_INTERVAL_EPOCHS,
        )
    )
    if formal_stage != "base":
        if (
            manifest_path.exists()
            or manifest_path.is_symlink()
            or candidate_dir.exists()
            or candidate_dir.is_symlink()
        ):
            raise RuntimeError(
                "Base candidate artifacts are forbidden outside formal_stage=base"
            )
        return None
    if not expected_epochs:
        if allowed_pending_checkpoint is not None:
            _validate_base_candidate_snapshot(
                manifest_path,
                expected_sha256=None,
                expected_entry_count=0,
                allowed_pending_checkpoint=allowed_pending_checkpoint,
            )
            return None
        if allow_empty_candidate_dir:
            _validate_base_candidate_snapshot(
                manifest_path,
                expected_sha256=None,
                expected_entry_count=0,
                allow_empty_candidate_dir=True,
            )
            return None
        if (
            manifest_path.exists()
            or manifest_path.is_symlink()
            or candidate_dir.exists()
            or candidate_dir.is_symlink()
        ):
            raise RuntimeError(
                "unexpected Base candidate artifacts before epoch 10"
            )
        return None
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise RuntimeError(
            f"Base candidate directory must be a regular directory: {candidate_dir}"
        )

    manifest = _load_regular_json(
        manifest_path,
        "Base candidate manifest",
    )
    expected_manifest_keys = {
        "format",
        "formal_stage",
        "interval_epochs",
        "config_sha256",
        "lineage_manifest_sha256",
        "dataset_receipt_sha256",
        "smplx_asset_receipt",
        "source_receipt",
        "source_receipt_sha256",
        "entries",
    }
    source_receipt = manifest.get("source_receipt")
    entries = manifest.get("entries")
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("format")
        != "semtalk_show_base_candidate_manifest_v2"
        or manifest.get("formal_stage") != "base"
        or _require_exact_audit_int(
            manifest.get("interval_epochs"),
            "candidate manifest interval_epochs",
        )
        != BASE_CANDIDATE_INTERVAL_EPOCHS
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("lineage_manifest_sha256") != lineage_sha256
        or manifest.get("dataset_receipt_sha256")
        != _payload_sha256(dataset_receipt)
        or manifest.get("smplx_asset_receipt")
        != dataset_receipt.get("smplx_asset")
        or not isinstance(source_receipt, dict)
        or _payload_sha256(source_receipt) != source_receipt_sha256
        or manifest.get("source_receipt_sha256") != source_receipt_sha256
        or not isinstance(entries, list)
        or len(entries) != len(expected_epochs)
    ):
        raise RuntimeError("Base candidate manifest binding mismatch")

    expected_files: set[Path] = set()
    for entry_index, (expected_epoch, record) in enumerate(
        zip(expected_epochs, entries, strict=True)
    ):
        expected_updates = expected_epoch * int(updates_per_epoch)
        expected_filename = _candidate_filename(
            expected_epoch,
            expected_updates,
        )
        expected_relative_path = _candidate_relative_path(
            expected_epoch,
            expected_updates,
        )
        previous_entries = entries[:entry_index]
        if previous_entries:
            previous_manifest = {
                **manifest,
                "entries": previous_entries,
            }
            previous_manifest_sha256 = _json_document_sha256(
                previous_manifest
            )
        else:
            previous_manifest_sha256 = None
        expected_transaction_core = _base_candidate_transaction_core(
            epoch=expected_epoch,
            optimizer_updates=expected_updates,
            previous_manifest_sha256=previous_manifest_sha256,
            previous_entries=previous_entries,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "epoch",
                "optimizer_updates",
                "checkpoint",
                "checkpoint_sha256",
                "model_audit_sha256",
                "transaction_core",
                "transaction_core_sha256",
            }
            or _require_exact_audit_int(
                record.get("epoch"),
                "candidate record epoch",
            )
            != expected_epoch
            or _require_exact_audit_int(
                record.get("optimizer_updates"),
                "candidate record optimizer_updates",
            )
            != expected_updates
            or record.get("checkpoint") != expected_relative_path
            or not isinstance(record.get("transaction_core"), dict)
            or _require_exact_audit_int(
                record["transaction_core"].get("candidate_epoch"),
                "candidate transaction core candidate_epoch",
            )
            != expected_epoch
            or _require_exact_audit_int(
                record["transaction_core"].get("optimizer_updates"),
                "candidate transaction core optimizer_updates",
            )
            != expected_updates
            or _require_exact_audit_int(
                record["transaction_core"].get("previous_entry_count"),
                "candidate transaction core previous_entry_count",
            )
            != entry_index
            or record.get("transaction_core") != expected_transaction_core
            or record.get("transaction_core_sha256")
            != _payload_sha256(expected_transaction_core)
        ):
            raise RuntimeError(
                f"invalid Base candidate record for epoch {expected_epoch}"
            )
        checkpoint_input = candidate_dir / expected_filename
        try:
            checkpoint_stat = checkpoint_input.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(checkpoint_input) from None
        if (
            stat.S_ISLNK(checkpoint_stat.st_mode)
            or not stat.S_ISREG(checkpoint_stat.st_mode)
        ):
            raise RuntimeError(
                "Base candidate checkpoint must be a regular non-symlink "
                f"file: {checkpoint_input}"
            )
        checkpoint_snapshot = checkpoint_input.read_bytes()
        checkpoint_sha = hashlib.sha256(checkpoint_snapshot).hexdigest()
        if checkpoint_sha != record.get("checkpoint_sha256"):
            raise RuntimeError(
                f"Base candidate checkpoint SHA mismatch: {checkpoint_input}"
            )
        checkpoint = _torch_load_candidate_snapshot(
            checkpoint_snapshot,
            checkpoint_path=checkpoint_input,
        )
        expected_audit = _base_candidate_audit(
            epoch=expected_epoch,
            optimizer_updates=expected_updates,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint) != {"model_state", "audit"}
            or checkpoint.get("audit") != expected_audit
            or record.get("model_audit_sha256")
            != _payload_sha256(expected_audit)
            or not isinstance(checkpoint.get("model_state"), dict)
            or not checkpoint["model_state"]
            or _finite_tree(
                checkpoint["model_state"],
                f"candidate_epoch_{expected_epoch}",
            )
        ):
            raise RuntimeError(
                f"invalid Base candidate checkpoint: {checkpoint_input}"
            )
        expected_files.add(checkpoint_input.resolve())

    if allowed_pending_checkpoint is not None:
        pending_input = allowed_pending_checkpoint
        if pending_input.parent.resolve() != candidate_dir.resolve():
            raise RuntimeError("pending Base candidate is outside candidate_dir")
        try:
            pending_stat = pending_input.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(pending_input) from None
        if (
            stat.S_ISLNK(pending_stat.st_mode)
            or not stat.S_ISREG(pending_stat.st_mode)
        ):
            raise RuntimeError(
                "pending Base candidate checkpoint must be a regular "
                f"non-symlink file: {pending_input}"
            )
        expected_files.add(pending_input.resolve())

    actual_children = list(candidate_dir.iterdir())
    for child in actual_children:
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(
            child_stat.st_mode
        ):
            raise RuntimeError(
                f"Base candidate is not a regular non-symlink file: {child}"
            )
    actual_files = {child.resolve() for child in actual_children}
    if actual_files != expected_files:
        raise RuntimeError("Base candidate directory contains missing or extra files")
    return _sha256(manifest_path)


def _load_and_validate_base_candidate_checkpoint(
    candidate_path: Path,
    *,
    expected_audit: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        candidate_stat = candidate_path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(candidate_path) from None
    if (
        stat.S_ISLNK(candidate_stat.st_mode)
        or not stat.S_ISREG(candidate_stat.st_mode)
    ):
        raise RuntimeError(
            "Base candidate checkpoint must be a regular non-symlink file: "
            f"{candidate_path}"
        )
    checkpoint_snapshot = candidate_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_snapshot).hexdigest()
    checkpoint = _torch_load_candidate_snapshot(
        checkpoint_snapshot,
        checkpoint_path=candidate_path,
    )
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"model_state", "audit"}
        or checkpoint.get("audit") != expected_audit
        or not isinstance(checkpoint.get("model_state"), dict)
        or not checkpoint["model_state"]
        or _finite_tree(checkpoint["model_state"], "pending_candidate")
    ):
        raise RuntimeError(
            f"invalid Base candidate checkpoint: {candidate_path}"
        )
    return checkpoint, checkpoint_sha256


def _valid_optional_sha256(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _inspect_base_candidate_transaction(
    manifest_path: Path,
    *,
    formal_stage: str,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
) -> dict[str, Any] | None:
    transaction_path = _candidate_transaction_path(manifest_path)
    staging_path = _candidate_staging_path(manifest_path)
    if transaction_path.is_symlink():
        raise RuntimeError("Base candidate transaction must not be a symlink")
    if not transaction_path.exists():
        if staging_path.exists() or staging_path.is_symlink():
            raise RuntimeError(
                "Base candidate staging file exists without a transaction"
            )
        return None
    if formal_stage != "base":
        raise RuntimeError(
            "Base candidate transaction is forbidden outside formal_stage=base"
        )

    transaction = _load_regular_json(
        transaction_path,
        "Base candidate transaction",
    )
    if (
        set(transaction)
        != {
            "format",
            "state",
            "core",
            "core_sha256",
            "checkpoint_sha256",
            "manifest_sha256",
        }
        or transaction.get("format")
        != "semtalk_show_base_candidate_transaction_v1"
        or transaction.get("state")
        not in {
            "prepared",
            "checkpoint_committed",
            "manifest_committed",
        }
        or not isinstance(transaction.get("core"), dict)
        or transaction.get("core_sha256")
        != _payload_sha256(transaction["core"])
        or not _valid_optional_sha256(
            transaction.get("checkpoint_sha256")
        )
        or not _valid_optional_sha256(transaction.get("manifest_sha256"))
    ):
        raise RuntimeError("invalid Base candidate transaction envelope")

    core = transaction["core"]
    target_epoch = _require_exact_audit_int(
        core.get("candidate_epoch"),
        "candidate transaction candidate_epoch",
    )
    optimizer_updates = _require_exact_audit_int(
        core.get("optimizer_updates"),
        "candidate transaction optimizer_updates",
    )
    previous_count = _require_exact_audit_int(
        core.get("previous_entry_count"),
        "candidate transaction previous_entry_count",
    )
    if (
        target_epoch <= 0
        or target_epoch % BASE_CANDIDATE_INTERVAL_EPOCHS != 0
        or target_epoch
        != (previous_count + 1) * BASE_CANDIDATE_INTERVAL_EPOCHS
        or optimizer_updates != target_epoch * int(updates_per_epoch)
    ):
        raise RuntimeError("invalid Base candidate transaction target")

    entries = _base_candidate_entries(manifest_path)
    current_count = len(entries)
    if current_count not in {previous_count, previous_count + 1}:
        raise RuntimeError(
            "Base candidate transaction/manifest entry count mismatch"
        )
    previous_entries = entries[:previous_count]
    if len(previous_entries) != previous_count:
        raise RuntimeError("Base candidate transaction prefix is incomplete")
    if previous_entries:
        manifest = _load_regular_json(
            manifest_path,
            "Base candidate manifest",
        )
        previous_manifest_sha256 = _json_document_sha256(
            {**manifest, "entries": previous_entries}
        )
    else:
        previous_manifest_sha256 = None
    expected_core = _base_candidate_transaction_core(
        epoch=target_epoch,
        optimizer_updates=optimizer_updates,
        previous_manifest_sha256=previous_manifest_sha256,
        previous_entries=previous_entries,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    if core != expected_core:
        raise RuntimeError("Base candidate transaction binding mismatch")

    candidate_path = (
        manifest_path.parent / str(core["checkpoint"])
    ).resolve()
    expected_candidate_path = (
        manifest_path.parent
        / _candidate_relative_path(target_epoch, optimizer_updates)
    ).resolve()
    if candidate_path != expected_candidate_path:
        raise RuntimeError("Base candidate transaction path mismatch")
    candidate_exists = candidate_path.exists() or candidate_path.is_symlink()
    if candidate_exists and (staging_path.exists() or staging_path.is_symlink()):
        raise RuntimeError(
            "Base candidate final checkpoint and staging file coexist"
        )
    if staging_path.exists() or staging_path.is_symlink():
        staging_stat = staging_path.lstat()
        if stat.S_ISLNK(staging_stat.st_mode) or not stat.S_ISREG(
            staging_stat.st_mode
        ):
            raise RuntimeError(
                "Base candidate staging path must be a regular non-symlink file"
            )

    expected_audit = _base_candidate_audit(
        epoch=target_epoch,
        optimizer_updates=optimizer_updates,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    checkpoint_sha256 = None
    if candidate_exists:
        _, checkpoint_sha256 = _load_and_validate_base_candidate_checkpoint(
            candidate_path,
            expected_audit=expected_audit,
        )

    manifested = current_count == previous_count + 1
    manifest_sha256 = _validate_base_candidate_manifest(
        manifest_path,
        formal_stage="base",
        completed_epochs=(
            current_count * BASE_CANDIDATE_INTERVAL_EPOCHS
        ),
        updates_per_epoch=updates_per_epoch,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt_sha256=_payload_sha256(source_receipt),
        allowed_pending_checkpoint=(
            candidate_path if candidate_exists and not manifested else None
        ),
        allow_empty_candidate_dir=(
            not candidate_exists and not manifested
        ),
    )
    if not manifested and manifest_sha256 != core["previous_manifest_sha256"]:
        raise RuntimeError(
            "Base candidate transaction previous manifest SHA mismatch"
        )
    if manifested and not candidate_exists:
        raise RuntimeError(
            "Base candidate manifest committed without its checkpoint"
        )

    state = transaction["state"]
    recorded_checkpoint_sha256 = transaction["checkpoint_sha256"]
    recorded_manifest_sha256 = transaction["manifest_sha256"]
    if state == "prepared":
        if (
            recorded_checkpoint_sha256 is not None
            or recorded_manifest_sha256 is not None
        ):
            raise RuntimeError("prepared candidate transaction has commit SHAs")
    elif state == "checkpoint_committed":
        if (
            not candidate_exists
            or recorded_checkpoint_sha256 != checkpoint_sha256
            or recorded_manifest_sha256 is not None
        ):
            raise RuntimeError(
                "checkpoint_committed candidate transaction mismatch"
            )
    elif (
        not manifested
        or not candidate_exists
        or recorded_checkpoint_sha256 != checkpoint_sha256
        or recorded_manifest_sha256 != manifest_sha256
    ):
        raise RuntimeError("manifest_committed candidate transaction mismatch")

    return {
        "path": transaction_path,
        "payload": transaction,
        "core": core,
        "state": state,
        "target_epoch": target_epoch,
        "previous_entry_count": previous_count,
        "current_entry_count": current_count,
        "candidate_path": candidate_path,
        "candidate_exists": candidate_exists,
        "checkpoint_sha256": checkpoint_sha256,
        "manifested": manifested,
        "manifest_sha256": manifest_sha256,
        "staging_exists": staging_path.exists(),
    }


def _assert_candidate_matches_model(
    trainer: Any,
    checkpoint: dict[str, Any],
) -> None:
    current_state = trainer.model.state_dict()
    existing_state = checkpoint.get("model_state")
    if (
        not isinstance(existing_state, dict)
        or set(existing_state) != set(current_state)
    ):
        raise RuntimeError(
            "replayed Base candidate model-state keys do not match"
        )
    for name, current_value in current_state.items():
        existing_value = existing_state[name]
        if (
            not torch.is_tensor(existing_value)
            or not torch.is_tensor(current_value)
            or existing_value.dtype != current_value.dtype
            or tuple(existing_value.shape) != tuple(current_value.shape)
            or not torch.equal(
                existing_value,
                current_value.detach().cpu(),
            )
        ):
            raise RuntimeError(
                "replayed Base candidate differs from immutable "
                f"checkpoint at {name}"
            )


def _clear_committed_base_candidate_transaction(
    manifest_path: Path,
    *,
    completed_epochs: int,
    manifest_sha256: str | None,
) -> None:
    transaction_path = _candidate_transaction_path(manifest_path)
    if not transaction_path.exists() and not transaction_path.is_symlink():
        return
    transaction = _load_regular_json(
        transaction_path,
        "Base candidate transaction",
    )
    core = transaction.get("core")
    if (
        transaction.get("state") != "manifest_committed"
        or not isinstance(core, dict)
        or _require_exact_audit_int(
            core.get("candidate_epoch"),
            "committed candidate transaction epoch",
        )
        != completed_epochs
        or transaction.get("manifest_sha256") != manifest_sha256
        or _candidate_staging_path(manifest_path).exists()
        or _candidate_staging_path(manifest_path).is_symlink()
    ):
        raise RuntimeError(
            "refusing to clear an uncommitted Base candidate transaction"
        )
    transaction_path.unlink()
    directory_fd = os.open(transaction_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _save_base_candidate(
    trainer: Any,
    manifest_path: Path,
    *,
    completed_epochs: int,
    updates_per_epoch: int,
    config_sha256: str,
    lineage_sha256: str | None,
    dataset_receipt: dict[str, Any],
    source_receipt: dict[str, str],
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if trainer.args.formal_stage != "base":
        raise RuntimeError("Base candidate save requested for a non-Base stage")
    if (
        completed_epochs <= 0
        or completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS != 0
    ):
        raise RuntimeError("Base candidates may only be saved every 10 epochs")
    optimizer_updates = trainer.formal_optimizer_updates
    expected_updates = completed_epochs * int(updates_per_epoch)
    if optimizer_updates != expected_updates:
        raise RuntimeError("Base candidate actual optimizer update count mismatch")

    transaction_path = _candidate_transaction_path(manifest_path)
    staging_path = _candidate_staging_path(manifest_path)
    candidate_dir = manifest_path.parent / "base_candidates"
    candidate_path = candidate_dir / _candidate_filename(
        completed_epochs,
        optimizer_updates,
    )
    transaction = _inspect_base_candidate_transaction(
        manifest_path,
        formal_stage="base",
        updates_per_epoch=updates_per_epoch,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    if transaction is None:
        previous_count = (
            completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS - 1
        )
        previous_entries = _validate_base_candidate_snapshot(
            manifest_path,
            expected_sha256=expected_manifest_sha256,
            expected_entry_count=previous_count,
        )
        core = _base_candidate_transaction_core(
            epoch=completed_epochs,
            optimizer_updates=optimizer_updates,
            previous_manifest_sha256=expected_manifest_sha256,
            previous_entries=previous_entries,
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        _write_base_candidate_transaction(
            transaction_path,
            core,
            state="prepared",
            checkpoint_sha256=None,
            manifest_sha256=None,
        )
        transaction = _inspect_new_prepared_base_candidate_transaction(
            manifest_path,
            core=core,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        assert transaction is not None
    else:
        if (
            transaction["target_epoch"] != completed_epochs
            or transaction["core"]["optimizer_updates"] != optimizer_updates
            or transaction["manifest_sha256"] != expected_manifest_sha256
        ):
            raise RuntimeError(
                "active Base candidate transaction does not match this epoch "
                "or trusted manifest fingerprint"
            )
    core = transaction["core"]
    previous_count = _require_exact_audit_int(
        core.get("previous_entry_count"),
        "candidate save previous_entry_count",
    )
    previous_entries = _base_candidate_entries(manifest_path)[:previous_count]
    if candidate_dir.exists() and (
        candidate_dir.is_symlink() or not candidate_dir.is_dir()
    ):
        raise RuntimeError(
            f"invalid Base candidate directory: {candidate_dir}"
        )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_audit = _base_candidate_audit(
        epoch=completed_epochs,
        optimizer_updates=optimizer_updates,
        config_sha256=config_sha256,
        lineage_sha256=lineage_sha256,
        dataset_receipt=dataset_receipt,
        source_receipt=source_receipt,
    )
    if not candidate_path.exists() and not candidate_path.is_symlink():
        _atomic_torch_save(
            candidate_path,
            {
                "model_state": trainer.model.state_dict(),
                "audit": candidate_audit,
            },
            staging_path=staging_path,
        )
    checkpoint, checkpoint_sha256 = (
        _load_and_validate_base_candidate_checkpoint(
            candidate_path,
            expected_audit=candidate_audit,
        )
    )
    _assert_candidate_matches_model(trainer, checkpoint)
    if transaction["state"] != "manifest_committed":
        _write_base_candidate_transaction(
            transaction_path,
            core,
            state="checkpoint_committed",
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=None,
        )

    record = {
        "epoch": completed_epochs,
        "optimizer_updates": optimizer_updates,
        "checkpoint": _candidate_relative_path(
            completed_epochs,
            optimizer_updates,
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "model_audit_sha256": _payload_sha256(candidate_audit),
        "transaction_core": core,
        "transaction_core_sha256": _payload_sha256(core),
    }
    current_entries = _base_candidate_entries(manifest_path)
    if len(current_entries) == previous_count:
        manifest_payload = _base_candidate_manifest_payload(
            config_sha256=config_sha256,
            lineage_sha256=lineage_sha256,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
            entries=[*previous_entries, record],
        )
        _atomic_json(manifest_path, manifest_payload)
    elif (
        len(current_entries) != previous_count + 1
        or current_entries[-1] != record
    ):
        raise RuntimeError(
            "Base candidate manifest is not at the recoverable transaction edge"
        )

    manifest_sha = _sha256(manifest_path)
    if manifest_sha != _json_document_sha256(
        _load_regular_json(
            manifest_path,
            "Base candidate manifest",
        )
    ):
        raise RuntimeError("Base candidate manifest serialization mismatch")
    _validate_base_candidate_snapshot(
        manifest_path,
        expected_sha256=manifest_sha,
        expected_entry_count=previous_count + 1,
    )
    _write_base_candidate_transaction(
        transaction_path,
        core,
        state="manifest_committed",
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha,
    )
    return {
        "path": manifest_path.name,
        "sha256": manifest_sha,
        "entries": len(previous_entries) + 1,
        "last_epoch": completed_epochs,
        "last_optimizer_updates": optimizer_updates,
    }


def _base_candidate_manifest_receipt(
    manifest_path: Path,
    manifest_sha256: str | None,
    *,
    completed_epochs: int,
    updates_per_epoch: int,
) -> dict[str, Any] | None:
    if manifest_sha256 is None:
        return None
    completed_epochs = _require_exact_audit_int(
        completed_epochs,
        "Base candidate receipt completed_epochs",
    )
    updates_per_epoch = _require_exact_audit_int(
        updates_per_epoch,
        "Base candidate receipt updates_per_epoch",
    )
    last_epoch = (
        completed_epochs
        // BASE_CANDIDATE_INTERVAL_EPOCHS
        * BASE_CANDIDATE_INTERVAL_EPOCHS
    )
    return {
        "path": manifest_path.name,
        "sha256": manifest_sha256,
        "entries": completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS,
        "last_epoch": last_epoch,
        "last_optimizer_updates": last_epoch * updates_per_epoch,
    }


def _require_trusted_candidate_manifest_sha(
    observed_sha256: str | None,
    trusted_sha256: str | None,
) -> str | None:
    if observed_sha256 != trusted_sha256:
        raise RuntimeError(
            "final Base candidate manifest SHA differs from the trusted "
            "training snapshot"
        )
    return observed_sha256


def _verify_or_write_final(
    path: Path,
    expected_payload: dict[str, Any],
) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"existing final checkpoint is not a regular file: {path}"
            )
        try:
            actual = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            actual = torch.load(path, map_location="cpu")
        if (
            not isinstance(actual, dict)
            or set(actual) != {"model_state", "audit"}
            or actual.get("audit") != (
                expected_payload["audit"]
            )
        ):
            raise RuntimeError(
                "existing final checkpoint audit does not match this run"
            )
        actual_state = actual.get("model_state")
        expected_state = expected_payload["model_state"]
        if (
            not isinstance(actual_state, dict)
            or set(actual_state) != set(expected_state)
        ):
            raise RuntimeError(
                "existing final checkpoint model-state keys do not match"
            )
        for name, expected in expected_state.items():
            actual_value = actual_state[name]
            if (
                not torch.is_tensor(actual_value)
                or not torch.is_tensor(expected)
                or actual_value.dtype != expected.dtype
                or tuple(actual_value.shape) != tuple(expected.shape)
                or not torch.equal(
                    actual_value.detach().cpu(),
                    expected.detach().cpu(),
                )
            ):
                raise RuntimeError(
                    "existing final checkpoint differs from resumed model at "
                    f"{name}"
                )
    else:
        _atomic_torch_save(path, expected_payload)
    return _sha256(path)


def main() -> None:
    args = config.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.local_rank = local_rank
    args.ddp = world_size > 1
    args.gpus = list(range(world_size)) if args.ddp else [0]
    args.skip_test_init = True

    if not args.train_only:
        raise RuntimeError("show_base_train.py requires --train_only true")
    if not args.strict_finite:
        raise RuntimeError("formal training requires --strict_finite true")
    if (
        args.expected_train_samples != 127_309
        or args.expected_updates_per_epoch != 1_989
    ):
        raise RuntimeError(
            "formal training requires exactly 127309 samples and "
            "1989 optimizer updates per epoch"
        )
    if not args.lineage_manifest:
        raise RuntimeError("formal training requires --lineage_manifest")
    if not args.dataset_summary:
        raise RuntimeError("formal training requires --dataset_summary")
    if args.save_every <= 0:
        raise RuntimeError("formal training requires positive --save_every")
    if args.debug or args.inference or args.test_state:
        raise RuntimeError("debug, inference, and test_state are forbidden in formal training")
    if not args.run_name:
        raise RuntimeError("--run_name is required for an auditable run")
    if "/" in args.run_name or args.run_name in {".", ".."}:
        raise RuntimeError("--run_name must be a single path component")
    if not args.final_ckpt_name:
        raise RuntimeError("--final_ckpt_name is required")
    if Path(args.final_ckpt_name).name != args.final_ckpt_name:
        raise RuntimeError("--final_ckpt_name must be a basename")
    if args.final_ckpt_name in {
        "latest_resume.pt",
        "formal_training_status.json",
        "base_candidate_manifest.json",
        BASE_CANDIDATE_TRANSACTION_FILENAME,
        BASE_CANDIDATE_STAGING_FILENAME,
    }:
        raise RuntimeError("--final_ckpt_name collides with a reserved artifact")
    if world_size != 1:
        raise RuntimeError(
            "formal SHOW reproduction training is single-GPU per model: RVQ EMA "
            "is rank-local and Base has unused branches plus multiple forwards"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each formal model task must receive exactly one isolated visible GPU"
        )

    stage_key = (args.model, args.g_name, args.trainer, bool(args.train_rvq))
    allowed_stages = {
        ("rvq", "RVQVAE", "aeface", True),
        ("rvq", "RVQVAE", "ae", True),
        ("rvq", "RVQVAE", "aelower", True),
        ("motion_representation", "VAEConvZero", "aelowerfoot", True),
        ("semtalk", "semtalk_base", "semtalk_base", False),
    }
    if stage_key not in allowed_stages:
        raise RuntimeError(
            "formal SHOW training only permits the five representation models "
            f"and semtalk_base; received {stage_key!r}"
        )
    _validate_formal_stage(args)
    if os.environ.get("PYTHONHASHSEED") != str(args.random_seed):
        raise RuntimeError(
            "PYTHONHASHSEED must be exported before launch and match random_seed"
        )

    source_receipt = _source_receipt()
    source_receipt_sha = _payload_sha256(source_receipt)
    initial_smplx_asset_receipt = _formal_smplx_asset_receipt(args)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)

    trainer = __import__(
        f"{args.trainer}_trainer", fromlist=["something"]
    ).CustomTrainer(args)
    train_samples = len(trainer.train_data)
    if args.expected_train_samples and train_samples != args.expected_train_samples:
        raise RuntimeError(
            f"train sample count {train_samples} != expected {args.expected_train_samples}"
        )
    updates_per_epoch = trainer.train_length
    if train_samples <= 0 or updates_per_epoch <= 0:
        raise RuntimeError("train dataset and dataloader must both be non-empty")
    if (
        args.expected_updates_per_epoch
        and updates_per_epoch != args.expected_updates_per_epoch
    ):
        raise RuntimeError(
            f"updates/epoch {updates_per_epoch} != expected "
            f"{args.expected_updates_per_epoch}"
        )
    dataset_receipt = _dataset_receipt(
        args,
        train_samples=train_samples,
        current_source=source_receipt,
    )
    if dataset_receipt.get("smplx_asset") != initial_smplx_asset_receipt:
        raise RuntimeError("formal SMPL-X asset changed while initializing trainer")

    checkpoint_dir = Path(trainer.checkpoint_path)
    status_path = checkpoint_dir / "formal_training_status.json"
    resume_path = checkpoint_dir / "latest_resume.pt"
    final_path = checkpoint_dir / args.final_ckpt_name
    candidate_manifest_path = checkpoint_dir / "base_candidate_manifest.json"
    lineage_sha = None
    if args.lineage_manifest:
        lineage_path = Path(args.lineage_manifest)
        if not lineage_path.is_file():
            raise FileNotFoundError(lineage_path)
        lineage_sha = _sha256(lineage_path)
    config_sha = _config_fingerprint(args)

    start_epoch = 0
    started_at = time.time()
    last_metrics: dict[str, dict[str, float | int]] = {}
    candidate_manifest_sha: str | None = None
    current_candidate_count = 0
    active_candidate_transaction: dict[str, Any] | None = None
    requested_resume = Path(args.resume_state) if args.resume_state else None
    if requested_resume:
        if requested_resume.is_symlink() or not requested_resume.is_file():
            raise FileNotFoundError(requested_resume)
        (
            start_epoch,
            last_metrics,
            started_at,
            candidate_manifest_sha,
            current_candidate_count,
            active_candidate_transaction,
        ) = _load_resume(
                trainer,
                requested_resume,
                rank=rank,
                world_size=world_size,
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_summary_sha256=dataset_receipt["summary_sha256"],
                data_mdb_sha256=dataset_receipt["data_mdb_sha256"],
                dataset_receipt=dataset_receipt,
                source_receipt=source_receipt,
                source_receipt_sha256=source_receipt_sha,
                candidate_manifest_path=candidate_manifest_path,
        )
    else:
        active_candidate_transaction = _inspect_base_candidate_transaction(
            candidate_manifest_path,
            formal_stage=args.formal_stage,
            updates_per_epoch=updates_per_epoch,
            config_sha256=config_sha,
            lineage_sha256=lineage_sha,
            dataset_receipt=dataset_receipt,
            source_receipt=source_receipt,
        )
        if active_candidate_transaction is not None:
            raise RuntimeError(
                "an uncommitted Base candidate transaction requires "
                "--resume_state"
            )
        current_candidate_count = len(
            _base_candidate_entries(candidate_manifest_path)
        )
        candidate_manifest_sha = _validate_base_candidate_manifest(
            candidate_manifest_path,
            formal_stage=args.formal_stage,
            completed_epochs=(
                current_candidate_count * BASE_CANDIDATE_INTERVAL_EPOCHS
            ),
            updates_per_epoch=updates_per_epoch,
            config_sha256=config_sha,
            lineage_sha256=lineage_sha,
            dataset_receipt=dataset_receipt,
            source_receipt_sha256=source_receipt_sha,
        )
    if start_epoch < 0 or start_epoch > args.epochs:
        raise RuntimeError(f"invalid resume epoch {start_epoch}")
    expected_start_updates = start_epoch * updates_per_epoch
    if trainer.formal_optimizer_updates != expected_start_updates:
        raise RuntimeError("initial actual optimizer update count mismatch")
    committed_candidate_count = (
        start_epoch // BASE_CANDIDATE_INTERVAL_EPOCHS
        if args.formal_stage == "base"
        else 0
    )
    if (
        active_candidate_transaction is not None
        and _require_exact_audit_int(
            active_candidate_transaction["target_epoch"],
            "active candidate transaction target_epoch",
        )
        == start_epoch
    ):
        _clear_committed_base_candidate_transaction(
            candidate_manifest_path,
            completed_epochs=start_epoch,
            manifest_sha256=candidate_manifest_sha,
        )
        active_candidate_transaction = None
    replay_candidate_epoch = None
    if active_candidate_transaction is not None:
        replay_candidate_epoch = _require_exact_audit_int(
            active_candidate_transaction["target_epoch"],
            "active candidate transaction target_epoch",
        )
    elif current_candidate_count != committed_candidate_count:
        raise RuntimeError("initial Base candidate entry count mismatch")
    candidate_manifest_receipt = _base_candidate_manifest_receipt(
        candidate_manifest_path,
        candidate_manifest_sha,
        completed_epochs=(
            current_candidate_count * BASE_CANDIDATE_INTERVAL_EPOCHS
        ),
        updates_per_epoch=updates_per_epoch,
    )

    if rank == 0:
        _atomic_json(
            status_path,
            {
                "status": "running",
                "run_name": args.run_name,
                "model": args.g_name,
                "trainer": args.trainer,
                "formal_stage": args.formal_stage,
                "epochs": args.epochs,
                "start_epoch": start_epoch,
                "world_size": world_size,
                "train_samples": train_samples,
                "updates_per_epoch": updates_per_epoch,
                "optimizer_updates": trainer.formal_optimizer_updates,
                "batch_size": args.batch_size,
                "lineage_manifest_sha256": lineage_sha,
                "dataset_receipt": dataset_receipt,
                "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
                "base_candidate_manifest": candidate_manifest_receipt,
                "source_receipt": source_receipt,
                "source_receipt_sha256": source_receipt_sha,
                "config_sha256": config_sha,
                "argv": sys.argv,
                "started_unix": started_at,
                "resumed_unix": (
                    time.time() if requested_resume is not None else None
                ),
            },
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            if args.ddp:
                trainer.train_loader.sampler.set_epoch(epoch)
            trainer.tracker.reset()
            updates_before_epoch = trainer.formal_optimizer_updates
            trainer.train(epoch)
            updates_after_epoch = trainer.formal_optimizer_updates
            if updates_after_epoch - updates_before_epoch != updates_per_epoch:
                raise RuntimeError(
                    "formal epoch actual optimizer update delta mismatch: "
                    f"epoch={epoch} before={updates_before_epoch} "
                    f"after={updates_after_epoch} expected_delta={updates_per_epoch}"
                )
            last_metrics = _tracker_snapshot(trainer)

            if args.strict_finite:
                bad = _finite_tree(trainer.model.state_dict(), "model")
                bad.extend(_finite_tree(_rvq_ema_state(trainer.model), "rvq_ema"))
                bad.extend(_finite_tree(trainer.opt.state_dict(), "optimizer"))
                bad.extend(_finite_tree(trainer.opt_s.state_dict(), "scheduler"))
                bad.extend(_finite_tree(last_metrics, "metrics"))
                bad.extend(_rvq_ema_invariant_errors(trainer.model))
                if bad:
                    raise FloatingPointError(
                        "non-finite training state: " + ", ".join(bad[:20])
                    )

            completed_epochs = epoch + 1
            expected_total_updates = completed_epochs * updates_per_epoch
            if updates_after_epoch != expected_total_updates:
                raise RuntimeError(
                    "formal cumulative actual optimizer update count mismatch"
                )
            if (
                args.formal_stage == "base"
                and completed_epochs % BASE_CANDIDATE_INTERVAL_EPOCHS == 0
            ):
                candidate_manifest_receipt = _save_base_candidate(
                    trainer,
                    candidate_manifest_path,
                    completed_epochs=completed_epochs,
                    updates_per_epoch=updates_per_epoch,
                    config_sha256=config_sha,
                    lineage_sha256=lineage_sha,
                    dataset_receipt=dataset_receipt,
                    source_receipt=source_receipt,
                    expected_manifest_sha256=candidate_manifest_sha,
                )
                candidate_manifest_sha = candidate_manifest_receipt["sha256"]
                current_candidate_count = _require_exact_audit_int(
                    candidate_manifest_receipt["entries"],
                    "Base candidate manifest receipt entries",
                )
            else:
                expected_candidate_count = (
                    completed_epochs // BASE_CANDIDATE_INTERVAL_EPOCHS
                    if args.formal_stage == "base"
                    else 0
                )
                allowed_pending_checkpoint = None
                allow_empty_candidate_dir = False
                if (
                    replay_candidate_epoch is not None
                    and completed_epochs < replay_candidate_epoch
                ):
                    assert active_candidate_transaction is not None
                    expected_candidate_count = _require_exact_audit_int(
                        active_candidate_transaction["current_entry_count"],
                        "active candidate transaction current_entry_count",
                    )
                    if (
                        active_candidate_transaction["candidate_exists"]
                        and not active_candidate_transaction["manifested"]
                    ):
                        allowed_pending_checkpoint = (
                            active_candidate_transaction["candidate_path"]
                        )
                    allow_empty_candidate_dir = (
                        not active_candidate_transaction["candidate_exists"]
                        and not active_candidate_transaction["manifested"]
                    )
                current_candidate_entries = _validate_base_candidate_snapshot(
                    candidate_manifest_path,
                    expected_sha256=candidate_manifest_sha,
                    expected_entry_count=expected_candidate_count,
                    allowed_pending_checkpoint=allowed_pending_checkpoint,
                    allow_empty_candidate_dir=allow_empty_candidate_dir,
                )
                candidate_manifest_receipt = _base_candidate_manifest_receipt(
                    candidate_manifest_path,
                    candidate_manifest_sha,
                    completed_epochs=(
                        len(current_candidate_entries)
                        * BASE_CANDIDATE_INTERVAL_EPOCHS
                    ),
                    updates_per_epoch=updates_per_epoch,
                )
            should_save = (
                completed_epochs == args.epochs
                or (args.save_every > 0 and completed_epochs % args.save_every == 0)
            )
            if should_save:
                rng_states = _all_rng_states(world_size)
                if rank == 0:
                    assert rng_states is not None
                    committed_candidate_entries = _base_candidate_entries(
                        candidate_manifest_path
                    )
                    _save_resume(
                        trainer,
                        resume_path,
                        completed_epochs=completed_epochs,
                        rng_states=rng_states,
                        world_size=world_size,
                        train_samples=train_samples,
                        updates_per_epoch=updates_per_epoch,
                        config_sha256=config_sha,
                        lineage_sha256=lineage_sha,
                        dataset_summary_sha256=dataset_receipt[
                            "summary_sha256"
                        ],
                        data_mdb_sha256=dataset_receipt[
                            "data_mdb_sha256"
                        ],
                        dataset_receipt=dataset_receipt,
                        source_receipt_sha256=source_receipt_sha,
                        optimizer_updates=trainer.formal_optimizer_updates,
                        candidate_manifest_sha256=candidate_manifest_sha,
                        candidate_manifest_entry_count=len(
                            committed_candidate_entries
                        ),
                        candidate_manifest_entries_sha256=(
                            _base_candidate_entries_sha256(
                                committed_candidate_entries
                            )
                        ),
                        last_metrics=last_metrics,
                        started_unix=started_at,
                    )
                    if (
                        args.formal_stage == "base"
                        and completed_epochs
                        % BASE_CANDIDATE_INTERVAL_EPOCHS
                        == 0
                    ):
                        _clear_committed_base_candidate_transaction(
                            candidate_manifest_path,
                            completed_epochs=completed_epochs,
                            manifest_sha256=candidate_manifest_sha,
                        )
                        active_candidate_transaction = None
                        replay_candidate_epoch = None
                    _atomic_json(
                        status_path,
                        {
                            "status": "running",
                            "run_name": args.run_name,
                            "model": args.g_name,
                            "trainer": args.trainer,
                            "formal_stage": args.formal_stage,
                            "epochs": args.epochs,
                            "completed_epochs": completed_epochs,
                            "world_size": world_size,
                            "train_samples": train_samples,
                            "updates_per_epoch": updates_per_epoch,
                            "batch_size": args.batch_size,
                            "optimizer_updates": trainer.formal_optimizer_updates,
                            "lineage_manifest_sha256": lineage_sha,
                            "dataset_receipt": dataset_receipt,
                            "smplx_asset_receipt": dataset_receipt.get(
                                "smplx_asset"
                            ),
                            "base_candidate_manifest": (
                                candidate_manifest_receipt
                            ),
                            "source_receipt": source_receipt,
                            "source_receipt_sha256": source_receipt_sha,
                            "config_sha256": config_sha,
                            "latest_resume_sha256": _sha256(resume_path),
                            "last_metrics": last_metrics,
                            "argv": sys.argv,
                            "started_unix": started_at,
                            "updated_unix": time.time(),
                        },
                    )
            if args.ddp:
                dist.barrier()

        if rank == 0:
            bad = _finite_tree(trainer.model.state_dict(), "model")
            bad.extend(_finite_tree(_rvq_ema_state(trainer.model), "rvq_ema"))
            bad.extend(_rvq_ema_invariant_errors(trainer.model))
            if bad:
                raise FloatingPointError(
                    "non-finite final model state: " + ", ".join(bad[:20])
                )
            final_source_receipt = _source_receipt()
            if final_source_receipt != source_receipt:
                raise RuntimeError("formal source changed during training")
            final_dataset_receipt = _dataset_receipt(
                args,
                train_samples=train_samples,
                current_source=final_source_receipt,
            )
            if final_dataset_receipt != dataset_receipt:
                raise RuntimeError(
                    "formal dataset/parity assets changed during training"
                )
            expected_final_updates = args.epochs * updates_per_epoch
            if trainer.formal_optimizer_updates != expected_final_updates:
                raise RuntimeError(
                    "formal final actual optimizer update count mismatch"
                )
            if (
                _candidate_transaction_path(candidate_manifest_path).exists()
                or _candidate_transaction_path(
                    candidate_manifest_path
                ).is_symlink()
                or _candidate_staging_path(candidate_manifest_path).exists()
                or _candidate_staging_path(
                    candidate_manifest_path
                ).is_symlink()
            ):
                raise RuntimeError(
                    "Base candidate transaction/staging remains at final audit"
                )
            observed_candidate_manifest_sha = (
                _validate_base_candidate_manifest(
                candidate_manifest_path,
                formal_stage=args.formal_stage,
                completed_epochs=args.epochs,
                updates_per_epoch=updates_per_epoch,
                config_sha256=config_sha,
                lineage_sha256=lineage_sha,
                dataset_receipt=dataset_receipt,
                source_receipt_sha256=source_receipt_sha,
                )
            )
            _require_trusted_candidate_manifest_sha(
                observed_candidate_manifest_sha,
                candidate_manifest_sha,
            )
            candidate_manifest_receipt = _base_candidate_manifest_receipt(
                candidate_manifest_path,
                observed_candidate_manifest_sha,
                completed_epochs=args.epochs,
                updates_per_epoch=updates_per_epoch,
            )
            assert lineage_sha is not None
            final_sha = _verify_or_write_final(
                final_path,
                _model_payload(
                    trainer,
                    formal_stage=args.formal_stage,
                    config_sha256=config_sha,
                    lineage_sha256=lineage_sha,
                    dataset_receipt=dataset_receipt,
                    source_receipt=source_receipt,
                    optimizer_updates=trainer.formal_optimizer_updates,
                    candidate_manifest_receipt=(
                        candidate_manifest_receipt
                    ),
                ),
            )
            _atomic_json(
                status_path,
                {
                    "status": "complete",
                    "run_name": args.run_name,
                    "model": args.g_name,
                    "trainer": args.trainer,
                    "formal_stage": args.formal_stage,
                    "epochs": args.epochs,
                    "completed_epochs": args.epochs,
                    "world_size": world_size,
                    "train_samples": train_samples,
                    "updates_per_epoch": updates_per_epoch,
                    "batch_size": args.batch_size,
                    "optimizer_updates": trainer.formal_optimizer_updates,
                    "lineage_manifest_sha256": lineage_sha,
                    "dataset_receipt": dataset_receipt,
                    "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
                    "base_candidate_manifest": candidate_manifest_receipt,
                    "source_receipt": source_receipt,
                    "source_receipt_sha256": source_receipt_sha,
                    "config_sha256": config_sha,
                    "final_checkpoint": str(final_path),
                    "final_checkpoint_sha256": final_sha,
                    "latest_resume_sha256": _sha256(resume_path),
                    "last_metrics": last_metrics,
                    "argv": sys.argv,
                    "started_unix": started_at,
                    "completed_unix": time.time(),
                },
            )
            logger.info(f"strict train-only finalize PASS: {final_path} {final_sha}")
        dist.barrier()
    except BaseException as exc:
        if rank == 0:
            _atomic_json(
                status_path,
                {
                    "status": "failed",
                    "run_name": args.run_name,
                    "model": args.g_name,
                    "trainer": args.trainer,
                    "formal_stage": args.formal_stage,
                    "epochs": args.epochs,
                    "world_size": world_size,
                    "train_samples": train_samples,
                    "updates_per_epoch": updates_per_epoch,
                    "optimizer_updates": trainer.formal_optimizer_updates,
                    "batch_size": args.batch_size,
                    "lineage_manifest_sha256": lineage_sha,
                    "dataset_receipt": dataset_receipt,
                    "smplx_asset_receipt": dataset_receipt.get("smplx_asset"),
                    "base_candidate_manifest": candidate_manifest_receipt,
                    "source_receipt": source_receipt,
                    "source_receipt_sha256": source_receipt_sha,
                    "config_sha256": config_sha,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "argv": sys.argv,
                    "started_unix": started_at,
                    "failed_unix": time.time(),
                },
            )
        raise
    finally:
        if rank == 0 and getattr(trainer, "writer", None) is not None:
            trainer.writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
