#!/usr/bin/env python3
"""Build the reproducible metric-only TalkSHOW source root.

The upstream checkout is an input authority only.  A new Git repository is
materialized from its exact detached commit, then ``nets/__init__.py`` is
replaced with one fixed no-eager-import byte string.  The result is accepted
only when the production TalkSHOW validator proves the exact 15-file import
closure and the only dirty paths are the patch and its canonical marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import evaluate_talkshow_show_metrics as metrics


BUILDER_FORMAT = "semtalk_talkshow_metric_root_builder_v1"
EXPECTED_DIRTY = {
    " M nets/__init__.py",
    "?? .paspa_talkshow_patch.json",
}


class TalkShowMetricRootBuildError(RuntimeError):
    """Raised when the deterministic metric root cannot be proven."""


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and process.returncode != 0:
        raise TalkShowMetricRootBuildError(
            f"git {' '.join(arguments)} failed for {root}: "
            + process.stderr.decode("utf-8", errors="replace").strip()
        )
    return process


def _git_text(root: Path, *arguments: str) -> str:
    return _git(root, *arguments).stdout.decode(
        "utf-8", errors="strict"
    ).rstrip("\n")


def _validate_clean_upstream(root_value: Path) -> dict[str, Any]:
    root = metrics._resolved_canonical_directory(
        root_value,
        "TalkSHOW metric-root upstream",
    )
    git_directory = metrics._resolved_canonical_directory(
        root / ".git",
        "TalkSHOW metric-root upstream Git directory",
    )
    if git_directory != root / ".git":
        raise TalkShowMetricRootBuildError(
            "TalkSHOW upstream must use its own non-symlink .git directory"
        )
    remotes = [
        line for line in _git_text(root, "remote").splitlines() if line
    ]
    origin = _git_text(root, "remote", "get-url", "origin")
    push_origin = _git_text(
        root,
        "remote",
        "get-url",
        "--push",
        "origin",
    )
    commit = _git_text(root, "rev-parse", "HEAD^{commit}")
    tree = _git_text(root, "rev-parse", "HEAD^{tree}")
    status = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    symbolic = _git(root, "symbolic-ref", "-q", "HEAD", check=False)
    local_heads = _git_text(
        root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/heads",
    )
    init_path, init_payload = metrics._safe_file_snapshot(
        root / "nets" / "__init__.py",
        "TalkSHOW upstream nets/__init__.py",
    )
    if (
        remotes != ["origin"]
        or origin != metrics.TALKSHOW_METRIC_UPSTREAM_ORIGIN
        or push_origin != origin
        or commit != metrics.TALKSHOW_METRIC_COMMIT
        or tree != metrics.TALKSHOW_METRIC_TREE
        or status
        or symbolic.returncode == 0
        or local_heads
        or init_path != root / "nets" / "__init__.py"
        or hashlib.sha256(init_payload).hexdigest()
        != metrics.TALKSHOW_PATCH_MARKER["source_sha256"]
    ):
        raise TalkShowMetricRootBuildError(
            "TalkSHOW upstream must be the exact clean detached branchless "
            "yhw-yhw/TalkSHOW commit/tree"
        )
    return {
        "path": str(root),
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "clean": True,
        "detached": True,
        "local_branches": [],
        "nets_init_sha256": hashlib.sha256(init_payload).hexdigest(),
    }


def _claim_new_directory(path_value: Path) -> tuple[Path, os.stat_result]:
    output = path_value.expanduser()
    if not output.is_absolute():
        raise TalkShowMetricRootBuildError(
            "TalkSHOW metric-root output must be absolute"
        )
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise TalkShowMetricRootBuildError(
            "TalkSHOW metric-root output parent does not exist"
        ) from error
    if parent != output.parent or not parent.is_dir() or parent.is_symlink():
        raise TalkShowMetricRootBuildError(
            "TalkSHOW metric-root output parent must be canonical"
        )
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        raise FileExistsError(
            f"TalkSHOW metric-root output already exists: {output}"
        ) from error
    claimed = os.lstat(output)
    if not stat.S_ISDIR(claimed.st_mode):
        raise TalkShowMetricRootBuildError(
            "TalkSHOW metric-root create-new claim is not a directory"
        )
    return output, claimed


def _initialize_checkout(upstream: Path, output: Path) -> None:
    _git(output, "init", "-q")
    _git(
        output,
        "remote",
        "add",
        "origin",
        metrics.TALKSHOW_METRIC_UPSTREAM_ORIGIN,
    )
    _git(
        output,
        "remote",
        "set-url",
        "--push",
        "origin",
        metrics.TALKSHOW_METRIC_UPSTREAM_ORIGIN,
    )
    _git(
        output,
        "fetch",
        "--quiet",
        "--no-tags",
        "--no-write-fetch-head",
        str(upstream),
        metrics.TALKSHOW_METRIC_COMMIT,
    )
    _git(
        output,
        "-c",
        "core.autocrlf=false",
        "checkout",
        "--quiet",
        "--detach",
        metrics.TALKSHOW_METRIC_COMMIT,
    )


def _atomic_replace(path: Path, payload: bytes) -> None:
    original = os.lstat(path)
    if not stat.S_ISREG(original.st_mode) or stat.S_ISLNK(original.st_mode):
        raise TalkShowMetricRootBuildError(
            f"TalkSHOW patch target is not a regular file: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), stat.S_IMODE(original.st_mode))
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _apply_metric_only_patch(root: Path) -> None:
    init_path = root / "nets" / "__init__.py"
    _path, upstream_payload = metrics._safe_file_snapshot(
        init_path,
        "TalkSHOW patch source nets/__init__.py",
    )
    if (
        hashlib.sha256(upstream_payload).hexdigest()
        != metrics.TALKSHOW_PATCH_MARKER["source_sha256"]
    ):
        raise TalkShowMetricRootBuildError(
            "TalkSHOW patch source bytes changed"
        )
    _atomic_replace(init_path, metrics.TALKSHOW_METRIC_ONLY_INIT)
    marker_path = root / ".paspa_talkshow_patch.json"
    metrics._atomic_write_new(
        marker_path,
        metrics.canonical_json_bytes(metrics.TALKSHOW_PATCH_MARKER),
    )
    patched_path, patched_payload = metrics._safe_file_snapshot(
        init_path,
        "TalkSHOW patched nets/__init__.py",
    )
    observed_marker_path, marker_payload = metrics._safe_file_snapshot(
        marker_path,
        "TalkSHOW metric-only patch marker",
    )
    if (
        patched_path != init_path
        or patched_payload != metrics.TALKSHOW_METRIC_ONLY_INIT
        or len(patched_payload)
        != metrics.TALKSHOW_PATCH_MARKER["patched_bytes"]
        or hashlib.sha256(patched_payload).hexdigest()
        != metrics.TALKSHOW_PATCH_MARKER["patched_sha256"]
        or observed_marker_path != marker_path
        or marker_payload
        != metrics.canonical_json_bytes(metrics.TALKSHOW_PATCH_MARKER)
    ):
        raise TalkShowMetricRootBuildError(
            "TalkSHOW metric-only patch materialization mismatch"
        )


def _remove_owned_output(
    output: Path,
    claimed: os.stat_result,
) -> None:
    try:
        current = os.lstat(output)
    except FileNotFoundError:
        return
    if (
        stat.S_ISDIR(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and (current.st_dev, current.st_ino)
        == (claimed.st_dev, claimed.st_ino)
    ):
        shutil.rmtree(output)


def build_metric_root(
    upstream_value: Path,
    output_value: Path,
) -> dict[str, Any]:
    upstream = Path(
        _validate_clean_upstream(upstream_value)["path"]
    )
    output, claimed = _claim_new_directory(output_value)
    complete = False
    try:
        _initialize_checkout(upstream, output)
        _validate_clean_upstream(output)
        _apply_metric_only_patch(output)
        dirty = {
            line
            for line in _git_text(
                output,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ).splitlines()
            if line
        }
        if dirty != EXPECTED_DIRTY:
            raise TalkShowMetricRootBuildError(
                "TalkSHOW metric-root dirty-file set changed: "
                f"{sorted(dirty)!r}"
            )
        metric_receipt = metrics.validate_talkshow_metric_root(output)
        if (
            metric_receipt["path"] != str(output)
            or metric_receipt["commit"] != metrics.TALKSHOW_METRIC_COMMIT
            or metric_receipt["tree"] != metrics.TALKSHOW_METRIC_TREE
            or tuple(metric_receipt["files"])
            != metrics.TALKSHOW_FGD_SOURCE_FILES
            or metric_receipt["patch_marker"]
            != metrics.TALKSHOW_PATCH_MARKER
        ):
            raise TalkShowMetricRootBuildError(
                "TalkSHOW metric-root validator receipt mismatch"
            )
        os.chmod(output, 0o755)
        complete = True
        return {
            "format": BUILDER_FORMAT,
            "status": "complete",
            "path": str(output),
            "origin": metrics.TALKSHOW_METRIC_UPSTREAM_ORIGIN,
            "commit": metrics.TALKSHOW_METRIC_COMMIT,
            "tree": metrics.TALKSHOW_METRIC_TREE,
            "patch_marker_sha256": hashlib.sha256(
                metrics.canonical_json_bytes(metrics.TALKSHOW_PATCH_MARKER)
            ).hexdigest(),
            "patched_init_sha256": metrics.TALKSHOW_PATCH_MARKER[
                "patched_sha256"
            ],
            "patched_init_bytes": metrics.TALKSHOW_PATCH_MARKER[
                "patched_bytes"
            ],
            "import_closure": list(metrics.TALKSHOW_FGD_SOURCE_FILES),
        }
    finally:
        if not complete:
            _remove_owned_output(output, claimed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_metric_root(args.upstream_root, args.output_root)
    print(
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
