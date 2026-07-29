#!/usr/bin/env python3
"""Run and freeze one Global-foot fastpath parity gate per SHOW speaker."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

sys.dont_write_bytecode = True


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SPEAKERS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
CONTRACT = "semtalk_show_global_foot_fastpath_v1"
EXPECTED_SPLIT_COUNTS = {
    "train": 13_687,
    "val": 1_715,
    "test": 1_708,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value: str, label: str, lengths: set[int]) -> str:
    normalized = value.strip().lower()
    if len(normalized) not in lengths or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{label} is not a lowercase hexadecimal digest")
    return normalized


def require_exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{label} must be an exact integer")
    return value


def require_exact_int_mapping(
    value: Any,
    expected: dict[str, int],
    label: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimeError(f"{label} must have exactly {sorted(expected)}")
    for key, expected_value in expected.items():
        if (
            require_exact_int(value[key], f"{label}.{key}")
            != expected_value
        ):
            raise RuntimeError(
                f"{label}.{key} must equal {expected_value}"
            )
    return value


def validate_canonical_manifest_rows(rows: list[dict[str, Any]]) -> None:
    expected_total = sum(EXPECTED_SPLIT_COUNTS.values())
    if len(rows) != expected_total:
        raise RuntimeError(
            f"canonical manifest has {len(rows)} rows, expected {expected_total}"
        )
    split_counts: Counter[str] = Counter()
    clip_ids: set[str] = set()
    global_indices: set[int] = set()
    for line_number, row in enumerate(rows, 1):
        split = row.get("split")
        if split not in EXPECTED_SPLIT_COUNTS:
            raise RuntimeError(
                f"canonical manifest row {line_number} has invalid split"
            )
        split_counts[split] += 1
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise RuntimeError(
                f"canonical manifest row {line_number} has invalid clip_id"
            )
        if clip_id in clip_ids:
            raise RuntimeError(f"duplicate canonical clip_id {clip_id!r}")
        clip_ids.add(clip_id)
        global_index = require_exact_int(
            row.get("global_index"),
            f"canonical manifest row {line_number} global_index",
        )
        if global_index < 0 or global_index >= expected_total:
            raise RuntimeError(
                f"canonical manifest row {line_number} global_index out of range"
            )
        if global_index in global_indices:
            raise RuntimeError(
                f"duplicate canonical global_index {global_index}"
            )
        global_indices.add(global_index)
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or speaker not in SPEAKERS:
            raise RuntimeError(
                f"canonical manifest row {line_number} has invalid speaker"
            )
        speaker_id = require_exact_int(
            row.get("speaker_id"),
            f"canonical manifest row {line_number} speaker_id",
        )
        if speaker_id != SPEAKERS[speaker]:
            raise RuntimeError(
                f"canonical manifest row {line_number} speaker/name mismatch"
            )
        if (
            require_exact_int(
                row.get("frames"),
                f"canonical manifest row {line_number} frames",
            )
            <= 0
        ):
            raise RuntimeError(
                f"canonical manifest row {line_number} has invalid frames"
            )
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            f"canonical manifest split counts {dict(split_counts)} "
            f"!= {EXPECTED_SPLIT_COUNTS}"
        )
    if global_indices != set(range(expected_total)):
        raise RuntimeError("canonical manifest global_index is not exact-once")


def canonical_payload_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_receipt(
    expected_commit: str,
    expected_tree: str,
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]

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
            "parity suite requires a clean source checkout; first change: "
            f"{status.splitlines()[0]}"
        )
    receipt = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
    }
    if receipt["origin"] != EXPECTED_ORIGIN:
        raise RuntimeError("unexpected SemTalk origin")
    if receipt["commit"] != expected_commit:
        raise RuntimeError("parity suite source commit mismatch")
    if receipt["tree"] != expected_tree:
        raise RuntimeError("parity suite source tree mismatch")
    return receipt


def load_canonical_receipt(
    manifest: Path,
    summary_path: Path,
    lineage_path: Path,
    expected_canonical_commit: str,
    expected_canonical_tree: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for path in (manifest, summary_path, lineage_path):
        if path.is_symlink():
            raise RuntimeError(f"canonical receipt must not be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = manifest.resolve()
    summary_path = summary_path.resolve()
    lineage_path = lineage_path.resolve()
    manifest_sha = sha256(manifest)
    summary = json.loads(summary_path.read_text())
    lineage = json.loads(lineage_path.read_text())
    if (
        summary.get("status") != "complete"
        or summary.get("schema_name") != "semtalk-show-canonical-motion"
        or require_exact_int(
            summary.get("schema_version"),
            "canonical summary schema_version",
        )
        != 1
        or require_exact_int(
            summary.get("clip_count"),
            "canonical summary clip_count",
        )
        != 17_110
        or summary.get("manifest_sha256") != manifest_sha
        or require_exact_int_mapping(
            summary.get("split_counts"),
            EXPECTED_SPLIT_COUNTS,
            "canonical summary split_counts",
        )
        != EXPECTED_SPLIT_COUNTS
        or summary.get("split_disjoint") is not True
        or summary.get("exact_once") is not True
        or summary.get("finite") is not True
    ):
        raise RuntimeError("invalid canonical summary")
    if (
        lineage.get("final_manifest_sha256") != manifest_sha
        or canonical_payload_sha256(lineage) != summary.get("lineage_sha256")
        or lineage.get("lineage_contract_sha256")
        != summary.get("lineage_contract_sha256")
    ):
        raise RuntimeError("canonical lineage binding mismatch")
    canonical_source = lineage.get(
        "lineage_contract", {}
    ).get("source_receipt")
    if (
        not isinstance(canonical_source, dict)
        or canonical_source.get("origin") != EXPECTED_ORIGIN
        or canonical_source.get("commit") != expected_canonical_commit
        or canonical_source.get("tree") != expected_canonical_tree
        or summary.get("source_receipt_sha256")
        != canonical_payload_sha256(canonical_source)
    ):
        raise RuntimeError("canonical source receipt mismatch")
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{manifest}:{line_number}: non-object row")
            rows.append(row)
    validate_canonical_manifest_rows(rows)
    return rows, {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "lineage": str(lineage_path),
        "lineage_sha256": sha256(lineage_path),
        "lineage_contract_sha256": summary["lineage_contract_sha256"],
        "source_receipt": canonical_source,
    }


def atomic_json_new(
    path: Path,
    payload: dict[str, Any],
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace()
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--expected-smplx-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-canonical-source-commit", required=True)
    parser.add_argument("--expected-canonical-source-tree", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_commit = require_digest(
        args.expected_source_commit,
        "--expected-source-commit",
        {40, 64},
    )
    expected_tree = require_digest(
        args.expected_source_tree,
        "--expected-source-tree",
        {40, 64},
    )
    expected_canonical_commit = require_digest(
        args.expected_canonical_source_commit,
        "--expected-canonical-source-commit",
        {40, 64},
    )
    expected_canonical_tree = require_digest(
        args.expected_canonical_source_tree,
        "--expected-canonical-source-tree",
        {40, 64},
    )
    expected_smplx_sha = require_digest(
        args.expected_smplx_sha256,
        "--expected-smplx-sha256",
        {64},
    )
    root = Path(__file__).resolve().parents[2]
    checker = root / "scripts/show_base/check_global_foot_fastpath_parity.py"
    if checker.is_symlink() or not checker.is_file():
        raise FileNotFoundError(checker)
    checker = checker.resolve()
    checker_sha = sha256(checker)
    source = source_receipt(expected_commit, expected_tree)
    rows, canonical_receipt = load_canonical_receipt(
        args.canonical_manifest,
        args.canonical_summary,
        args.canonical_lineage,
        expected_canonical_commit,
        expected_canonical_tree,
    )
    if args.smplx_model.is_symlink():
        raise RuntimeError(
            f"SMPL-X input must not be a symlink: {args.smplx_model}"
        )
    smplx_model = args.smplx_model.resolve()
    smplx_sha = sha256(smplx_model)
    if smplx_sha != expected_smplx_sha:
        raise RuntimeError("SMPL-X asset SHA mismatch")
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        speaker = str(row.get("speaker"))
        if (
            row.get("split") == "train"
            and speaker in SPEAKERS
            and require_exact_int(
                row.get("speaker_id"),
                f"{row.get('clip_id')}: speaker_id",
            )
            == SPEAKERS[speaker]
            and require_exact_int(
                row.get("frames"),
                f"{row.get('clip_id')}: frames",
            )
            >= 64
            and speaker not in selected
        ):
            selected[speaker] = row
    if set(selected) != set(SPEAKERS):
        raise RuntimeError("cannot select one parity window for every speaker")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    report_records = []
    for speaker in SPEAKERS:
        row = selected[speaker]
        canonical_input = Path(row["canonical_npz"])
        foot_input = Path(row["lower_foot_local"])
        if canonical_input.is_symlink() or foot_input.is_symlink():
            raise RuntimeError(
                f"{speaker}: canonical/foot parity input must not be a symlink"
            )
        canonical = canonical_input.resolve()
        foot = foot_input.resolve()
        report = output_root / f"{speaker}.json"
        command = [
            sys.executable,
            str(checker),
            "--canonical-npz",
            str(canonical),
            "--lower-foot-local",
            str(foot),
            "--smplx-model",
            str(smplx_model),
            "--device",
            args.device,
            "--start-frame",
            "0",
            "--frames",
            "64",
            "--report-json",
            str(report),
        ]
        subprocess.run(command, cwd=root, check=True)
        if report.is_symlink() or not report.is_file():
            raise RuntimeError(f"{speaker}: parity report is not a regular file")
        report_bytes = report.read_bytes()
        report_sha = hashlib.sha256(report_bytes).hexdigest()
        payload = json.loads(report_bytes)
        if (
            payload.get("status") != "pass"
            or payload.get("contract") != CONTRACT
            or payload.get("canonical_npz_sha256")
            != row.get("canonical_npz_sha256")
            or payload.get("lower_foot_local_sha256")
            != row.get("lower_foot_local_sha256")
            or payload.get("smplx_asset_sha256") != expected_smplx_sha
        ):
            raise RuntimeError(f"{speaker}: invalid parity report")
        report_records.append(
            {
                "speaker": speaker,
                "speaker_id": SPEAKERS[speaker],
                "clip_id": row["clip_id"],
                "report": str(report),
                "report_sha256": report_sha,
                "payload": payload,
            }
        )

    def revalidate_inputs() -> None:
        final_source = source_receipt(expected_commit, expected_tree)
        final_rows, final_canonical_receipt = load_canonical_receipt(
            args.canonical_manifest,
            args.canonical_summary,
            args.canonical_lineage,
            expected_canonical_commit,
            expected_canonical_tree,
        )
        if (
            final_source != source
            or final_rows != rows
            or final_canonical_receipt != canonical_receipt
            or checker.is_symlink()
            or not checker.is_file()
            or sha256(checker) != checker_sha
            or args.smplx_model.is_symlink()
            or not smplx_model.is_file()
            or sha256(smplx_model) != smplx_sha
        ):
            raise RuntimeError(
                "formal source/canonical/checker/SMPL-X inputs changed "
                "during parity execution"
            )
        for speaker, row in selected.items():
            for path_key, sha_key in (
                ("canonical_npz", "canonical_npz_sha256"),
                ("lower_foot_local", "lower_foot_local_sha256"),
            ):
                input_path = Path(row[path_key])
                if input_path.is_symlink() or not input_path.is_file():
                    raise RuntimeError(
                        f"{speaker}: parity input changed during execution"
                    )
                if sha256(input_path.resolve()) != row[sha_key]:
                    raise RuntimeError(
                        f"{speaker}: parity input SHA changed during execution"
                    )
        for record in report_records:
            report_path = Path(record["report"])
            if report_path.is_symlink() or not report_path.is_file():
                raise RuntimeError("parity report changed during execution")
            report_bytes = report_path.read_bytes()
            if (
                hashlib.sha256(report_bytes).hexdigest()
                != record["report_sha256"]
                or json.loads(report_bytes) != record["payload"]
            ):
                raise RuntimeError("parity report changed during execution")

    revalidate_inputs()

    bundle = {
        "format": "semtalk_show_global_foot_parity_suite_v1",
        "status": "pass",
        "contract": CONTRACT,
        "speakers": SPEAKERS,
        "canonical_receipt": canonical_receipt,
        "source_receipt": source,
        "checker": str(checker),
        "checker_sha256": checker_sha,
        "smplx_asset_sha256": smplx_sha,
        "reports": report_records,
        "completed_unix": time.time(),
        "argv": sys.argv,
    }
    atomic_json_new(
        output_root / "bundle.json",
        bundle,
        before_replace=revalidate_inputs,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "speakers": list(SPEAKERS),
                "bundle": str(output_root / "bundle.json"),
                "bundle_sha256": sha256(output_root / "bundle.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
