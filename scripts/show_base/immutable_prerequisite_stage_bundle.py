#!/usr/bin/env python3
"""Publish and materialize immutable cross-node prerequisite run bundles.

The two SHOW workers do not share ``/local-ssd``.  Validation, however, uses
one canonical absolute run path for all five prerequisite stages.  This
utility provides the missing fail-closed transfer primitive:

* ``pack`` snapshots explicitly named stage runs into a new-only bundle;
* ``stage`` verifies every bundled byte, reserves the destination directory
  with ``mkdir`` (so it can never replace an existing run), copies the exact
  snapshot, and verifies the materialized tree again.

An interrupted operation intentionally leaves an ``.incomplete`` marker and
must not be reused.  That is preferable to silently repairing or overwriting
an authority path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
from typing import Any, Iterable, Mapping, Sequence


FORMAT = "semtalk_show_prerequisite_stage_bundle_v1"
STAGES = ("face", "hands", "upper", "lower", "global")
INCOMPLETE = ".incomplete"
MANIFEST = "bundle_manifest.json"


class BundleError(RuntimeError):
    """Raised when a bundle or destination is not byte-exact."""


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if newline:
        encoded += "\n"
    return encoded.encode("utf-8")


def _payload_sha(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BundleError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise BundleError(f"{label} must be an absolute normalized path")
    return path


def _directory(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise BundleError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise BundleError(f"{label} must be a non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise BundleError(f"{label} is not canonical: {path}")
    return resolved


def _regular(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise BundleError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise BundleError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise BundleError(f"{label} is not canonical: {path}")
    return resolved


def _parse_stage_values(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise BundleError(f"{label} must be STAGE=/absolute/path")
        stage, raw_path = raw.split("=", 1)
        if stage not in STAGES or stage in result:
            raise BundleError(f"invalid or duplicate stage {stage!r}")
        result[stage] = _absolute(raw_path, f"{stage} {label}")
    if not result:
        raise BundleError(f"at least one {label} is required")
    return result


def _walk_tree(root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for name in list(dirnames):
            child = current_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise BundleError(f"non-directory/symlink in run tree: {child}")
            relative = child.relative_to(root).as_posix()
            directories.append(relative)
        for name in filenames:
            child = current_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise BundleError(f"non-regular/symlink in run tree: {child}")
            relative = child.relative_to(root).as_posix()
            files.append(
                {
                    "relative_path": relative,
                    "sha256": _sha256_file(child),
                    "bytes": child.stat().st_size,
                    "mode": stat.S_IMODE(mode),
                }
            )
    return sorted(directories), sorted(files, key=lambda row: row["relative_path"])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reserve_new_directory(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    parent = _directory(path.parent, f"{label} parent")
    if path.parent != parent:
        raise BundleError(f"{label} parent is not canonical")
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        raise BundleError(f"refusing to reuse {label}: {path}") from None
    marker = path / INCOMPLETE
    marker.write_text("incomplete\n", encoding="ascii")
    _fsync_directory(path)
    _fsync_directory(parent)
    return path


def _copy_file(source: Path, destination: Path, mode: int) -> None:
    if destination.exists() or destination.is_symlink():
        raise BundleError(f"refusing to overwrite {destination}")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
    )
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            descriptor = -1
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.chmod(destination, mode, follow_symlinks=False)


def _copy_snapshot(
    source: Path,
    destination: Path,
    directories: Iterable[str],
    files: Sequence[Mapping[str, Any]],
) -> None:
    for relative in directories:
        (destination / relative).mkdir(mode=0o700)
    for row in files:
        relative = str(row["relative_path"])
        source_path = _regular(source / relative, f"source file {relative}")
        if (
            source_path.stat().st_size != row["bytes"]
            or _sha256_file(source_path) != row["sha256"]
        ):
            raise BundleError(f"source changed while copying: {source_path}")
        _copy_file(source_path, destination / relative, int(row["mode"]))


def _finish_reserved(root: Path) -> None:
    marker = root / INCOMPLETE
    _regular(marker, "incomplete marker").unlink()
    _fsync_directory(root)
    _fsync_directory(root.parent)


def _manifest_bytes(value: Mapping[str, Any]) -> bytes:
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


def pack(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_host != socket.gethostname():
        raise BundleError("source hostname differs from bundle authority")
    sources = _parse_stage_values(args.source_run, "source run")
    destinations = _parse_stage_values(args.destination_run, "destination run")
    if set(sources) != set(destinations):
        raise BundleError("source and destination stage coverage differs")
    bundle = _reserve_new_directory(args.bundle_root, "bundle root")
    files_root = bundle / "files"
    files_root.mkdir()
    stages: dict[str, Any] = {}
    for stage in STAGES:
        if stage not in sources:
            continue
        source = _directory(sources[stage], f"{stage} source run")
        directories, files = _walk_tree(source)
        if not files:
            raise BundleError(f"{stage} source run is empty")
        stage_destination = files_root / stage
        stage_destination.mkdir()
        _copy_snapshot(source, stage_destination, directories, files)
        copied_directories, copied_files = _walk_tree(stage_destination)
        if copied_directories != directories or copied_files != files:
            raise BundleError(f"{stage} bundled snapshot differs from source")
        stages[stage] = {
            "source_run": str(source),
            "destination_run": str(destinations[stage]),
            "directories": directories,
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(int(row["bytes"]) for row in files),
            "tree_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
        }
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "source_host": args.source_host,
        "destination_host": args.destination_host,
        "stages": stages,
    }
    manifest["receipt_payload_sha256"] = _payload_sha(manifest)
    manifest_path = bundle / MANIFEST
    payload = _manifest_bytes(manifest)
    descriptor = os.open(
        manifest_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _finish_reserved(bundle)
    return {
        "path": str(manifest_path.resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "receipt_payload_sha256": manifest["receipt_payload_sha256"],
    }


def _load_manifest(
    path: Path,
    expected_sha: str,
    expected_payload_sha: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = _regular(path, "bundle manifest")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _require_sha(
        expected_sha, "bundle manifest SHA-256"
    ):
        raise BundleError("bundle manifest file changed")
    value = json.loads(payload.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "format",
            "status",
            "source_host",
            "destination_host",
            "stages",
            "receipt_payload_sha256",
        }
        or value["format"] != FORMAT
        or value["status"] != "complete"
        or value["receipt_payload_sha256"]
        != _require_sha(expected_payload_sha, "bundle payload SHA-256")
        or _payload_sha(value) != value["receipt_payload_sha256"]
    ):
        raise BundleError("bundle manifest protocol mismatch")
    root = resolved.parent
    if (root / INCOMPLETE).exists() or resolved.name != MANIFEST:
        raise BundleError("bundle is incomplete or manifest is misplaced")
    return root, value


def _validate_stage_snapshot(root: Path, stage: str, value: Mapping[str, Any]) -> Path:
    if set(value) != {
        "source_run",
        "destination_run",
        "directories",
        "files",
        "file_count",
        "total_bytes",
        "tree_sha256",
    }:
        raise BundleError(f"{stage} bundle stage schema mismatch")
    snapshot = _directory(root / "files" / stage, f"{stage} bundle snapshot")
    directories, files = _walk_tree(snapshot)
    if (
        directories != value["directories"]
        or files != value["files"]
        or len(files) != value["file_count"]
        or sum(int(row["bytes"]) for row in files) != value["total_bytes"]
        or hashlib.sha256(_canonical_bytes(files)).hexdigest()
        != value["tree_sha256"]
    ):
        raise BundleError(f"{stage} bundle bytes changed")
    return snapshot


def stage(args: argparse.Namespace) -> dict[str, Any]:
    root, manifest = _load_manifest(
        args.bundle_manifest,
        args.expected_bundle_sha256,
        args.expected_bundle_payload_sha256,
    )
    if args.stage not in manifest["stages"]:
        raise BundleError(f"stage {args.stage!r} is absent from bundle")
    if (
        args.destination_host != socket.gethostname()
        or manifest["destination_host"] != args.destination_host
    ):
        raise BundleError("destination hostname differs from bundle authority")
    value = manifest["stages"][args.stage]
    destination = _absolute(args.destination_run, "destination run")
    if str(destination) != value["destination_run"]:
        raise BundleError("destination path differs from bundle authority")
    snapshot = _validate_stage_snapshot(root, args.stage, value)
    reserved = _reserve_new_directory(destination, "destination run")
    _copy_snapshot(
        snapshot,
        reserved,
        value["directories"],
        value["files"],
    )
    directories, files = _walk_tree(reserved)
    # The marker is a regular file until the transaction commits.
    files = [row for row in files if row["relative_path"] != INCOMPLETE]
    if directories != value["directories"] or files != value["files"]:
        raise BundleError("materialized destination differs from bundle")
    _finish_reserved(reserved)
    return {
        "stage": args.stage,
        "destination_run": str(reserved),
        "file_count": value["file_count"],
        "total_bytes": value["total_bytes"],
        "tree_sha256": value["tree_sha256"],
        "bundle_manifest": {
            "path": str(args.bundle_manifest.resolve(strict=True)),
            "sha256": args.expected_bundle_sha256,
            "receipt_payload_sha256": args.expected_bundle_payload_sha256,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack", allow_abbrev=False)
    pack_parser.add_argument("--source-host", required=True)
    pack_parser.add_argument("--destination-host", required=True)
    pack_parser.add_argument("--source-run", action="append", required=True)
    pack_parser.add_argument("--destination-run", action="append", required=True)
    pack_parser.add_argument("--bundle-root", type=Path, required=True)
    stage_parser = commands.add_parser("stage", allow_abbrev=False)
    stage_parser.add_argument("--bundle-manifest", type=Path, required=True)
    stage_parser.add_argument("--expected-bundle-sha256", required=True)
    stage_parser.add_argument(
        "--expected-bundle-payload-sha256", required=True
    )
    stage_parser.add_argument("--destination-host", required=True)
    stage_parser.add_argument("--stage", choices=STAGES, required=True)
    stage_parser.add_argument("--destination-run", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = pack(args) if args.command == "pack" else stage(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
