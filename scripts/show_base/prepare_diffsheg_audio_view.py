#!/usr/bin/env python3
"""Build the sealed TalkSHOW test-split WAV view required by PASPA.

The official TalkSHOW release stores WAVs as
``<speaker>/<video>/<sequence>/<sequence>.wav``.  PASPA's pinned DiffSHEG
evaluator intentionally resolves the official test layout
``<speaker>/<video>/test/<sequence>/<sequence>.wav``.  This CPU-only tool
materializes that missing directory view without altering or fabricating any
audio.  Every source file is snapshotted, hashed, PCM-validated, linked (or
byte-copied across filesystems), and sealed against the fresh final-test
authority and the already-finalized 1,708-clip inference manifest.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import evaluate_diffsheg_final_test as closure
from scripts.show_base import run_base_final_test as final_test


class AudioViewError(RuntimeError):
    """Raised when the source or output cannot satisfy the sealed contract."""


def _canonical_directory(value: Any, label: str) -> Path:
    absolute = closure._absolute(value, label)
    resolved = closure._directory(absolute, label)
    if absolute != resolved:
        raise AudioViewError(f"{label} must be a canonical non-symlink path")
    return resolved


def _source_path(root: Path, source_clip_id: Any) -> tuple[Path, str]:
    if type(source_clip_id) is not str:
        raise AudioViewError("canonical source clip ID must be a string")
    pieces = source_clip_id.split("/")
    if (
        len(pieces) != 3
        or any(not piece or piece in {".", ".."} for piece in pieces)
        or any("/" in piece or "\\" in piece for piece in pieces)
    ):
        raise AudioViewError(f"unsafe canonical source clip ID {source_clip_id!r}")
    speaker, video, sequence = pieces
    relative = Path(speaker) / video / sequence / f"{sequence}.wav"
    return root / relative, relative.as_posix()


def _snapshot_regular_file(path: Path, root: Path, label: str) -> bytes:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise AudioViewError(f"{label} escapes source root") from error
    cursor = root
    for index, piece in enumerate(relative.parts):
        cursor = cursor / piece
        try:
            observed = os.lstat(cursor)
        except FileNotFoundError:
            raise AudioViewError(f"{label} is missing: {cursor}") from None
        if stat.S_ISLNK(observed.st_mode):
            raise AudioViewError(f"{label} contains a symlink: {cursor}")
        if index + 1 < len(relative.parts) and not stat.S_ISDIR(observed.st_mode):
            raise AudioViewError(f"{label} parent is not a directory: {cursor}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AudioViewError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise AudioViewError(f"{label} must be a non-empty regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    linked = os.lstat(path)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino)
    ):
        raise AudioViewError(f"{label} changed while it was snapshotted")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise AudioViewError(f"{label} short read")
    return payload


def _mkdir_chain(root: Path, parent: Path) -> None:
    relative = parent.relative_to(root)
    cursor = root
    for piece in relative.parts:
        cursor = cursor / piece
        try:
            os.mkdir(cursor, 0o755)
        except FileExistsError:
            observed = os.lstat(cursor)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise AudioViewError(f"unsafe audio-view directory {cursor}")


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise AudioViewError(f"cannot create {path}: {error}") from error
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _link_or_copy(source: Path, target: Path, payload: bytes) -> str:
    try:
        os.link(source, target, follow_symlinks=False)
        mode = "hardlink"
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise AudioViewError(f"cannot hardlink {source} to {target}: {error}") from error
        _write_exclusive(target, payload)
        mode = "copy"
    if closure.sha256_file(target) != hashlib.sha256(payload).hexdigest():
        raise AudioViewError(f"materialized WAV changed: {target}")
    return mode


def _canonical_manifest_receipt(authority: Mapping[str, Any]) -> dict[str, Any]:
    receipt = authority.get("canonical", {}).get("manifest")
    if not isinstance(receipt, dict):
        raise AudioViewError("fresh authority lacks canonical manifest receipt")
    return dict(receipt)


def build_view(args: argparse.Namespace) -> dict[str, Any]:
    try:
        authority, _authority_receipt = closure._load_current_authority(args)
        inference, clip_ids = closure._validate_inference_bundle(args, authority)
        canonical_rows, _canonical_by_id, _audio_by_id = final_test._load_inputs(
            authority
        )
    except Exception as error:
        raise AudioViewError(f"fresh final authority replay failed: {error}") from error
    if len(canonical_rows) != closure.EXPECTED_TEST_CLIPS:
        raise AudioViewError("canonical test manifest is not the exact 1,708 set")
    canonical_by_output: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        source_clip_id = row.get("clip_id")
        try:
            output_id = final_test.canonical_clip_id(str(source_clip_id))
        except Exception as error:
            raise AudioViewError(f"invalid canonical clip {source_clip_id!r}") from error
        if output_id in canonical_by_output:
            raise AudioViewError(f"canonical output collision {output_id}")
        canonical_by_output[output_id] = row
    if set(canonical_by_output) != set(clip_ids):
        raise AudioViewError("canonical manifest and finalized inference differ")

    source_root = _canonical_directory(args.source_audio_root, "TalkSHOW source-audio root")
    output = closure._absolute(args.output_root, "DiffSHEG audio-view output")
    parent = _canonical_directory(output.parent, "DiffSHEG audio-view parent")
    if output.parent != parent or os.path.lexists(output):
        raise AudioViewError("audio-view output must be a new canonical path")
    os.mkdir(output, 0o755)
    output_root = output.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for evaluation_index, clip_id in enumerate(clip_ids):
        canonical = canonical_by_output[clip_id]
        source_clip_id = canonical["clip_id"]
        source_path, source_relative = _source_path(source_root, source_clip_id)
        payload = _snapshot_regular_file(
            source_path,
            source_root,
            f"{source_clip_id} original WAV",
        )
        wav = closure._wav_metadata(payload, f"{source_clip_id} original WAV")
        target_path, target_relative = closure._view_audio_path(
            output_root,
            source_clip_id,
            clip_id,
        )
        _mkdir_chain(output_root, target_path.parent)
        materialization = _link_or_copy(source_path, target_path, payload)
        digest = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "evaluation_index": evaluation_index,
                "clip_id": clip_id,
                "source_clip_id": source_clip_id,
                "source": {
                    "path": str(source_path),
                    "relative_path": source_relative,
                    "bytes": len(payload),
                    "sha256": digest,
                },
                "view": {
                    "path": str(target_path),
                    "relative_path": target_relative,
                    "bytes": len(payload),
                    "sha256": digest,
                    "materialization": materialization,
                },
                "wav": wav,
            }
        )
    if len(rows) != closure.EXPECTED_TEST_CLIPS:
        raise AudioViewError("audio-view build did not cover every test clip")
    manifest_payload = b"".join(closure.canonical_json_bytes(row) for row in rows)
    manifest_path = output_root / closure.AUDIO_VIEW_MANIFEST_NAME
    _write_exclusive(manifest_path, manifest_payload)
    manifest_receipt = {
        "path": str(manifest_path),
        "bytes": len(manifest_payload),
        "sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    receipt: dict[str, Any] = {
        "format": closure.AUDIO_VIEW_FORMAT,
        "status": "complete",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "canonical_manifest": _canonical_manifest_receipt(authority),
        "inference_manifest_sha256": inference["manifest"]["sha256"],
        "clip_manifest_sha256": inference["clip_manifest"]["sha256"],
        "clip_count": closure.EXPECTED_TEST_CLIPS,
        "exact_once": True,
        "symlinks": "forbidden",
        "padding": "forbidden",
        "truncation": "forbidden",
        "fabrication": "forbidden",
        "manifest": manifest_receipt,
        "source_ordered_set_sha256": closure._audio_set_sha(rows, "source"),
        "view_ordered_set_sha256": closure._audio_set_sha(rows, "view"),
    }
    receipt["receipt_payload_sha256"] = closure.canonical_json_sha256(receipt)
    receipt_path = closure._atomic_new(
        output_root / closure.AUDIO_VIEW_RECEIPT_NAME,
        receipt,
        "DiffSHEG audio-view receipt",
    )
    validated_root, paths, _validated = closure._validate_audio_view(
        inference,
        clip_ids,
        output_root,
    )
    if validated_root != output_root or len(paths) != closure.EXPECTED_TEST_CLIPS:
        raise AudioViewError("audio-view self-validation failed")
    return {
        "status": "complete",
        "output_root": str(output_root),
        "receipt": str(receipt_path),
        "receipt_sha256": closure.sha256_file(receipt_path),
        "source_ordered_set_sha256": receipt["source_ordered_set_sha256"],
        "view_ordered_set_sha256": receipt["view_ordered_set_sha256"],
        "clip_count": len(paths),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    final_test._authority_options(parser)
    parser.add_argument("--inference-final-root", type=Path, required=True)
    parser.add_argument("--source-audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prepared = (
        args.prepared_authority,
        args.expected_prepared_authority_sha256,
        args.expected_prepared_authority_bytes,
        args.expected_prepared_authority_receipt_payload_sha256,
    )
    if any(value is not None for value in prepared) and any(
        value is None for value in prepared
    ):
        parser.error("prepared authority requires path/SHA/bytes/payload SHA")
    try:
        result = build_view(args)
    except (AudioViewError, closure.FinalDiffSHEGError) as error:
        parser.exit(2, f"[abort] {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
