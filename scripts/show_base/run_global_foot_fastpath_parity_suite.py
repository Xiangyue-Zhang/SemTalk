#!/usr/bin/env python3
"""Run and freeze one Global-foot fastpath parity gate per SHOW speaker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.dont_write_bytecode = True


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
SPEAKERS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
CONTRACT = "semtalk_show_global_foot_fastpath_v1"


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
    expected_commit: str,
    expected_tree: str,
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
        or int(summary.get("schema_version", -1)) != 1
        or int(summary.get("clip_count", -1)) != 17_110
        or summary.get("manifest_sha256") != manifest_sha
        or summary.get("split_counts")
        != {"train": 13_687, "val": 1_715, "test": 1_708}
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
        or canonical_source.get("commit") != expected_commit
        or canonical_source.get("tree") != expected_tree
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


def atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
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
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--canonical-lineage", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--expected-smplx-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
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
    expected_smplx_sha = require_digest(
        args.expected_smplx_sha256,
        "--expected-smplx-sha256",
        {64},
    )
    root = Path(__file__).resolve().parents[2]
    checker = root / "scripts/show_base/check_global_foot_fastpath_parity.py"
    if not checker.is_file():
        raise FileNotFoundError(checker)
    source = source_receipt(expected_commit, expected_tree)
    rows, canonical_receipt = load_canonical_receipt(
        args.canonical_manifest,
        args.canonical_summary,
        args.canonical_lineage,
        expected_commit,
        expected_tree,
    )
    if args.smplx_model.is_symlink():
        raise RuntimeError(
            f"SMPL-X input must not be a symlink: {args.smplx_model}"
        )
    smplx_model = args.smplx_model.resolve()
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        speaker = str(row.get("speaker"))
        if (
            row.get("split") == "train"
            and speaker in SPEAKERS
            and int(row.get("speaker_id", -1)) == SPEAKERS[speaker]
            and int(row.get("frames", -1)) >= 64
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
        payload = json.loads(report.read_text())
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
                "report_sha256": sha256(report),
                "payload": payload,
            }
        )

    bundle = {
        "format": "semtalk_show_global_foot_parity_suite_v1",
        "status": "pass",
        "contract": CONTRACT,
        "speakers": SPEAKERS,
        "canonical_receipt": canonical_receipt,
        "source_receipt": source,
        "checker": str(checker),
        "checker_sha256": sha256(checker),
        "smplx_asset_sha256": expected_smplx_sha,
        "reports": report_records,
        "completed_unix": time.time(),
        "argv": sys.argv,
    }
    atomic_json_new(output_root / "bundle.json", bundle)
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
