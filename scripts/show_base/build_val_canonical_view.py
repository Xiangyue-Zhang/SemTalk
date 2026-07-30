#!/usr/bin/env python3
"""Create an immutable validation-only view of the frozen SHOW manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
EXPECTED_VAL_CLIPS = 1_715
SUMMARY_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_summary_v1"
)
LINEAGE_FORMAT = (
    "semtalk_show_base_official_adapt_val_canonical_lineage_v1"
)


class ValViewError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValViewError(f"{label} is not a lowercase SHA-256")
    return value


def _require_git_oid(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValViewError(f"{label} is not a lowercase Git object ID")
    return value


def _strict_object(payload: bytes, label: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValViewError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValViewError(f"{label}: non-finite JSON constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValViewError(f"{label}: expected one JSON object")
    return value


def _compact_payload_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _json_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_verified(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    _require_sha256(expected_sha256, f"{label} expected SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValViewError(f"{label} must be a regular non-symlink: {path}")
    resolved = path.resolve()
    payload = resolved.read_bytes()
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise ValViewError(
            f"{label} SHA-256 mismatch: {actual} != {expected_sha256}"
        )
    return resolved, payload


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(args: argparse.Namespace) -> dict[str, object]:
    source_commit = _require_git_oid(
        args.expected_source_commit,
        "canonical source commit",
    )
    source_tree = _require_git_oid(
        args.expected_source_tree,
        "canonical source tree",
    )
    manifest_path, manifest_payload = _read_verified(
        args.full_manifest,
        args.expected_full_manifest_sha256,
        "full canonical manifest",
    )
    summary_path, summary_payload = _read_verified(
        args.full_summary,
        args.expected_full_summary_sha256,
        "full canonical summary",
    )
    lineage_path, lineage_payload = _read_verified(
        args.full_lineage,
        args.expected_full_lineage_sha256,
        "full canonical lineage",
    )
    full_summary = _strict_object(summary_payload, str(summary_path))
    full_lineage = _strict_object(lineage_payload, str(lineage_path))
    source = (
        full_lineage.get("lineage_contract", {})
        if isinstance(full_lineage.get("lineage_contract"), dict)
        else {}
    ).get("source_receipt")
    split_counts = full_summary.get("split_counts")
    if (
        full_summary.get("status") != "complete"
        or full_summary.get("manifest_sha256")
        != args.expected_full_manifest_sha256
        or full_summary.get("lineage_sha256")
        != args.expected_full_lineage_sha256
        or not isinstance(split_counts, dict)
        or split_counts.get("val") != EXPECTED_VAL_CLIPS
        or not isinstance(source, dict)
        or source.get("origin") != EXPECTED_ORIGIN
        or source.get("commit") != source_commit
        or source.get("tree") != source_tree
    ):
        raise ValViewError("full canonical receipt binding mismatch")
    lineage_contract_sha256 = _require_sha256(
        str(full_lineage.get("lineage_contract_sha256")),
        "canonical lineage contract SHA-256",
    )

    val_rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        manifest_payload.splitlines(),
        1,
    ):
        if not raw_line.strip():
            continue
        row = _strict_object(
            raw_line,
            f"{manifest_path}:{line_number}",
        )
        if row.get("split") != "val":
            continue
        if (
            type(row.get("global_index")) is not int
            or not isinstance(row.get("clip_id"), str)
            or row.get("lineage_contract_sha256")
            != lineage_contract_sha256
        ):
            raise ValViewError(
                f"{manifest_path}:{line_number}: invalid val row"
            )
        val_rows.append(row)
    if len(val_rows) != EXPECTED_VAL_CLIPS:
        raise ValViewError(
            f"val row count {len(val_rows)} != {EXPECTED_VAL_CLIPS}"
        )
    indices = [int(row["global_index"]) for row in val_rows]
    clip_ids = [str(row["clip_id"]) for row in val_rows]
    if (
        indices != sorted(indices)
        or len(indices) != len(set(indices))
        or len(clip_ids) != len(set(clip_ids))
    ):
        raise ValViewError("validation rows are not unique global-index order")

    manifest_bytes = b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in val_rows
    )
    manifest_sha256 = _sha256(manifest_bytes)
    source_receipt = {
        "origin": EXPECTED_ORIGIN,
        "commit": source_commit,
        "tree": source_tree,
        "full_manifest_sha256": args.expected_full_manifest_sha256,
        "full_summary_sha256": args.expected_full_summary_sha256,
        "full_lineage_sha256": args.expected_full_lineage_sha256,
    }
    lineage: dict[str, object] = {
        "format": LINEAGE_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "clip_count": EXPECTED_VAL_CLIPS,
        "manifest_sha256": manifest_sha256,
        "lineage_contract_sha256": lineage_contract_sha256,
        "projection": {
            "operation": "filter_exact_split",
            "split": "val",
            "test_rows_materialized": False,
        },
        "source_receipt": source_receipt,
    }
    lineage["receipt_payload_sha256"] = _compact_payload_sha256(lineage)
    lineage_bytes = _json_file_bytes(lineage)
    lineage_sha256 = _sha256(lineage_bytes)
    summary: dict[str, object] = {
        "format": SUMMARY_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "clip_count": EXPECTED_VAL_CLIPS,
        "manifest_sha256": manifest_sha256,
        "lineage_sha256": lineage_sha256,
        "lineage_contract_sha256": lineage_contract_sha256,
    }
    summary["receipt_payload_sha256"] = _compact_payload_sha256(summary)
    summary_bytes = _json_file_bytes(summary)
    summary_sha256 = _sha256(summary_bytes)

    output_root = args.output_root.resolve()
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing existing output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / (
        f".{output_root.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if os.path.lexists(staging):
        raise FileExistsError(staging)
    staging.mkdir()
    published = False
    try:
        _write_fsync(staging / "manifest.jsonl", manifest_bytes)
        _write_fsync(staging / "lineage.json", lineage_bytes)
        _write_fsync(staging / "summary.json", summary_bytes)
        _fsync_directory(staging)
        _fsync_directory(output_root.parent)
        os.rename(staging, output_root)
        published = True
        _fsync_directory(output_root.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return {
        "manifest": str(output_root / "manifest.jsonl"),
        "manifest_sha256": manifest_sha256,
        "summary": str(output_root / "summary.json"),
        "summary_sha256": summary_sha256,
        "lineage": str(output_root / "lineage.json"),
        "lineage_sha256": lineage_sha256,
        "clip_count": EXPECTED_VAL_CLIPS,
        "test_visible": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    result.add_argument("--full-manifest", type=Path, required=True)
    result.add_argument("--full-summary", type=Path, required=True)
    result.add_argument("--full-lineage", type=Path, required=True)
    result.add_argument(
        "--expected-full-manifest-sha256",
        required=True,
    )
    result.add_argument(
        "--expected-full-summary-sha256",
        required=True,
    )
    result.add_argument(
        "--expected-full-lineage-sha256",
        required=True,
    )
    result.add_argument("--expected-source-commit", required=True)
    result.add_argument("--expected-source-tree", required=True)
    result.add_argument("--output-root", type=Path, required=True)
    return result


def main() -> None:
    receipt = build(parser().parse_args())
    print(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
