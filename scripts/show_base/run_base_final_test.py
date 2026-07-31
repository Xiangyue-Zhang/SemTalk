#!/usr/bin/env python3
"""Run the one authorized SemTalk Base inference on the frozen SHOW test.

This entry point is deliberately a *consumer* of
``base_final_authority.py``.  It cannot choose a checkpoint, a split, a
speaker subset, or a metric.  The authority has already frozen the
validation-selected Base winner, the five independently validation-selected
SHOW representation models, the exact 1,708-clip test cache and its eight
audio shards.  This module only performs these irreversible phases:

``context``
    Fresh-replay the externally pinned authority and prove that neither the
    final root nor the sibling shard root exists.
``shard``
    Run one exact ``global_index % 8`` CUDA shard through the neutral official
    inference core.
``finalize``
    Rebuild exact-once closure, copy every shard artifact into a new final
    generation, and publish the generation with one directory rename.
``distribution``
    Bind the already validated ``final_winner`` deterministic-replication
    gate to the physical predictions.
``seal``
    Fresh-validate the sole TalkSHOW test report and publish a terminal
    receipt that explicitly forbids test-to-selection feedback.

No command accepts an epoch, a candidate checkpoint, a split, or a selection
metric.  Test output is therefore incapable of changing the validation
winner by construction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import stat
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_final_authority as final_authority  # noqa: E402


NUM_SHARDS = 8
TEST_CLIPS = 1_708
TEST_GLOBAL_START = 15_402
TEST_GLOBAL_STOP = 17_110
SHARD_SUMMARY_FORMAT = "semtalk_show_base_inference_shard_summary_v1"
SHARD_LINEAGE_FORMAT = "semtalk_show_base_inference_shard_lineage_v1"
FINAL_LINEAGE_FORMAT = "semtalk_show_base_inference_final_lineage_v1"
FINAL_SUMMARY_FORMAT = "semtalk_show_base_inference_final_summary_v1"
COMPLETION_FORMAT = "semtalk_show_base_final_test_completion_v1"
CONTRACT_FORMAT = "semtalk_show_base_final_test_consumer_contract_v1"
RUNTIME_FORMAT = "semtalk_show_base_final_test_runtime_v1"
PREPARED_AUTHORITY_FORMAT = (
    "semtalk_show_base_final_test_prepared_authority_v1"
)
CHECKPOINT_STAGES = ("base", "face", "hands", "upper", "lower", "global")
REPRESENTATION_STAGES = ("face", "hands", "upper", "lower", "global")
SHOW_SPEAKER_IDS = {
    "oliver": 0,
    "chemistry": 1,
    "seth": 2,
    "conan": 3,
}
ARTIFACT_KEYS = {"path", "sha256", "bytes"}
SHARD_ROW_KEYS = {
    "global_index",
    "source_clip_id",
    "canonical_clip_id",
    "speaker",
    "speaker_id",
    "frames",
    "canonical_npz",
    "canonical_npz_sha256",
    "audio_feature_npz",
    "audio_feature_npz_sha256",
    "prediction",
    "ground_truth",
}
FINAL_ROW_KEYS = SHARD_ROW_KEYS | {"evaluation_index"}


class FinalTestContractError(RuntimeError):
    """Raised when the one-shot final-test transaction is not exact."""


class _VerifiedTreeLoader(importlib.abc.Loader):
    def __init__(
        self,
        finder: "_VerifiedTreeFinder",
        fullname: str,
        relative: str,
        is_package: bool,
    ) -> None:
        self.finder = finder
        self.fullname = fullname
        self.relative = relative
        self.is_package = is_package

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        path, payload, receipt = self.finder.source_snapshot(
            self.fullname, self.relative
        )
        module.__file__ = str(path)
        module.__cached__ = None
        if self.is_package:
            module.__path__ = [str(path.parent)]
        module.__verified_source_receipt__ = receipt
        code = compile(payload, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _VerifiedTreeFinder(importlib.abc.MetaPathFinder):
    """Load every imported SemTalk module from its verified Git-tree bytes."""

    def __init__(self, authority: Mapping[str, Any]) -> None:
        source = authority["inference_source"]
        self.root = Path(source["source_root"]).resolve()
        self.tree = str(source["tree"])
        try:
            output = subprocess.run(
                [
                    "git", "-C", str(self.root), "ls-tree", "-r", "-z",
                    "--name-only", self.tree,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FinalTestContractError(
                "cannot enumerate the authorized SemTalk source tree"
            ) from exc
        self.tracked = {
            value.decode("utf-8")
            for value in output.split(b"\0")
            if value and value.endswith(b".py")
        }
        self.receipts: dict[str, dict[str, Any]] = {}

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> Any:
        del path, target
        stem = fullname.replace(".", "/")
        candidates = ((f"{stem}.py", False), (f"{stem}/__init__.py", True))
        for relative, is_package in candidates:
            if relative not in self.tracked:
                continue
            loader = _VerifiedTreeLoader(self, fullname, relative, is_package)
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=str(self.root / relative),
                is_package=is_package,
            )
        return None

    def source_snapshot(
        self, fullname: str, relative: str
    ) -> tuple[Path, bytes, dict[str, Any]]:
        if relative not in self.tracked:
            raise FinalTestContractError(
                f"{fullname} is not tracked by the authorized source tree"
            )
        try:
            expected = subprocess.run(
                [
                    "git", "-C", str(self.root), "show",
                    f"{self.tree}:{relative}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FinalTestContractError(
                f"cannot read authorized Git blob for {fullname}"
            ) from exc
        source_path, observed, observed_sha = _safe_file_snapshot(
            self.root / relative, f"verified source module {fullname}"
        )
        if observed != expected:
            raise FinalTestContractError(
                f"live source bytes for {fullname} differ from the authority tree"
            )
        receipt = {
            "module": fullname,
            "relative_path": relative,
            "path": str(source_path),
            "sha256": observed_sha,
            "bytes": len(observed),
            "source_tree": self.tree,
            "loader": "verified_git_source_snapshot_no_bytecode",
        }
        previous = self.receipts.get(fullname)
        if previous is not None and previous != receipt:
            raise FinalTestContractError(
                f"verified source receipt changed for {fullname}"
            )
        self.receipts[fullname] = receipt
        return source_path, observed, receipt


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FinalTestContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FinalTestContractError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalTestContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FinalTestContractError(
                    f"{label} contains non-finite JSON token {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalTestContractError(f"{label} is invalid JSON") from exc


def _strict_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise FinalTestContractError(f"{label} line {number} is blank")
        value = _strict_json(line, f"{label} line {number}")
        if type(value) is not dict:
            raise FinalTestContractError(
                f"{label} line {number} is not a JSON object"
            )
        rows.append(value)
    if not rows:
        raise FinalTestContractError(f"{label} is empty")
    return rows


def _safe_file_snapshot(
    path_value: str | Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[Path, bytes, str]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise FinalTestContractError(f"{label} path must be absolute")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise FinalTestContractError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FinalTestContractError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise FinalTestContractError(f"{label} path is not canonical: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(resolved)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    identities = [
        tuple(int(getattr(item, field)) for field in identity_fields)
        for item in (before, opened_before, opened_after, after)
    ]
    if len(set(identities)) != 1:
        raise FinalTestContractError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    observed = _sha256_bytes(payload)
    if expected_sha256 is not None and observed != _require_sha256(
        expected_sha256, f"{label} expected SHA"
    ):
        raise FinalTestContractError(f"{label} SHA-256 mismatch")
    if expected_bytes is not None and len(payload) != _require_int(
        expected_bytes, f"{label} expected bytes", minimum=1
    ):
        raise FinalTestContractError(f"{label} byte count mismatch")
    return resolved, payload, observed


def _artifact(path: Path) -> dict[str, Any]:
    resolved, payload, observed = _safe_file_snapshot(path, "artifact")
    return {"path": str(resolved), "sha256": observed, "bytes": len(payload)}


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    created = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_new(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    _source, payload, observed = _safe_file_snapshot(
        source,
        "shard output",
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    _write_new(destination, payload)
    if _artifact(destination) != {
        "path": str(destination.resolve()),
        "sha256": observed,
        "bytes": len(payload),
    }:
        raise FinalTestContractError("published copy changed")


def _authority_artifact(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "path": str(Path(args.fresh_test_authority).expanduser().resolve()),
        "sha256": _require_sha256(
            args.expected_test_authority_sha256,
            "test authority file SHA",
        ),
        "bytes": _require_int(
            args.expected_test_authority_bytes,
            "test authority bytes",
            minimum=1,
        ),
        "receipt_payload_sha256": _require_sha256(
            args.expected_test_authority_receipt_payload_sha256,
            "test authority payload SHA",
        ),
    }


def _prepared_authority_artifact(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    path = getattr(args, "prepared_authority", None)
    if path is None:
        return None
    return {
        "path": str(Path(path).expanduser().resolve()),
        "sha256": _require_sha256(
            args.expected_prepared_authority_sha256,
            "prepared authority file SHA",
        ),
        "bytes": _require_int(
            args.expected_prepared_authority_bytes,
            "prepared authority bytes",
            minimum=1,
        ),
        "receipt_payload_sha256": _require_sha256(
            args.expected_prepared_authority_receipt_payload_sha256,
            "prepared authority payload SHA",
        ),
    }


def _validate_authority_identity(authority: Any) -> dict[str, Any]:
    if (
        type(authority) is not dict
        or authority.get("format") != final_authority.FORMAT
        or authority.get("status") != "authorized_pre_inference"
        or authority.get("contract", {}).get("split") != "test"
        or authority.get("contract", {}).get("test_clips") != TEST_CLIPS
        or authority.get("contract", {}).get("num_shards") != NUM_SHARDS
        or set(authority.get("checkpoints", {})) != set(CHECKPOINT_STAGES)
    ):
        raise FinalTestContractError("test authority identity changed")
    if (
        Path(authority["inference_source"]["source_root"]).resolve()
        != PROJECT_ROOT.resolve()
    ):
        raise FinalTestContractError(
            "final-test adapter is not executing from the authorized "
            "SemTalk source root"
        )
    return authority


def _validate_live_source_boundary(authority: Mapping[str, Any]) -> None:
    """Re-bind the prepared receipt to the clean detached live worktree.

    A prepared authority avoids replaying all 22 Base candidates in every GPU
    shard, but it must not turn the source tree into a cached trust decision.
    This local-only proof is deliberately repeated at each inference boundary.
    The independent formal evaluator and terminal report validator still run
    the full authority validator, including its live-remote proof.
    """

    source = authority.get("inference_source")
    if type(source) is not dict:
        raise FinalTestContractError("prepared authority source is missing")
    root = Path(str(source.get("source_root", "")))
    if (
        not root.is_absolute()
        or root.resolve() != PROJECT_ROOT.resolve()
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise FinalTestContractError("live SemTalk source root changed")

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=check,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FinalTestContractError(
                "cannot prove the live SemTalk source boundary"
            ) from exc

    branches = git(
        "for-each-ref", "--format=%(refname)", "refs/heads"
    ).stdout.strip()
    detached = git("symbolic-ref", "-q", "HEAD", check=False).returncode != 0
    observed = {
        "origin": git("remote", "get-url", "origin").stdout.strip(),
        "commit": git("rev-parse", "HEAD^{commit}").stdout.strip(),
        "tree": git("rev-parse", "HEAD^{tree}").stdout.strip(),
        "status": git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout,
    }
    if (
        observed["origin"] != source.get("origin")
        or observed["commit"] != source.get("commit")
        or observed["tree"] != source.get("tree")
        or observed["status"] != ""
        or not detached
        or branches
    ):
        raise FinalTestContractError(
            "live SemTalk source is not the exact clean detached authority tree"
        )


def _validated_authority(args: argparse.Namespace) -> dict[str, Any]:
    receipt = _authority_artifact(args)
    prepared_receipt = _prepared_authority_artifact(args)
    if prepared_receipt is not None:
        _prepared_path, prepared_payload, _prepared_sha = _safe_file_snapshot(
            prepared_receipt["path"],
            "prepared authority",
            expected_sha256=prepared_receipt["sha256"],
            expected_bytes=prepared_receipt["bytes"],
        )
        prepared = _strict_json(prepared_payload, "prepared authority")
        if type(prepared) is not dict or set(prepared) != {
            "format",
            "status",
            "authority_artifact",
            "authority",
            "producer",
            "receipt_payload_sha256",
        }:
            raise FinalTestContractError("prepared authority schema mismatch")
        claimed = prepared.get("receipt_payload_sha256")
        unsigned = dict(prepared)
        del unsigned["receipt_payload_sha256"]
        if (
            prepared.get("format") != PREPARED_AUTHORITY_FORMAT
            or prepared.get("status") != "complete"
            or claimed != prepared_receipt["receipt_payload_sha256"]
            or _canonical_json_sha256(unsigned) != claimed
            or prepared.get("authority_artifact") != receipt
            or prepared.get("producer")
            != {
                "adapter": str(Path(__file__).resolve()),
                "adapter_sha256": _artifact(Path(__file__).resolve())[
                    "sha256"
                ],
                "source_root": str(PROJECT_ROOT.resolve()),
            }
        ):
            raise FinalTestContractError(
                "prepared authority identity/payload mismatch"
            )
        _authority_path, authority_payload, _authority_sha = (
            _safe_file_snapshot(
                receipt["path"],
                "test authority",
                expected_sha256=receipt["sha256"],
                expected_bytes=receipt["bytes"],
            )
        )
        authority = _strict_json(authority_payload, "test authority")
        if (
            authority != prepared.get("authority")
            or authority.get("receipt_payload_sha256")
            != receipt["receipt_payload_sha256"]
        ):
            raise FinalTestContractError(
                "prepared authority differs from the externally pinned file"
            )
        authority = _validate_authority_identity(authority)
        _validate_live_source_boundary(authority)
        return authority

    try:
        authority = final_authority.validate_test_authority(
            receipt["path"],
            expected_file_sha256=receipt["sha256"],
            expected_bytes=receipt["bytes"],
            expected_receipt_payload_sha256=receipt[
                "receipt_payload_sha256"
            ],
        )
    except Exception as exc:
        raise FinalTestContractError(
            f"fresh test authority replay failed: {exc}"
        ) from exc
    return _validate_authority_identity(authority)


def _output_roots(authority: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(str(authority["expected_output_root"]))
    if not root.is_absolute() or root.resolve() != root:
        raise FinalTestContractError("authorized output root is not canonical")
    return root, root.parent / "shards"


def _require_new_output(authority: Mapping[str, Any]) -> tuple[Path, Path]:
    root, shards = _output_roots(authority)
    existing = [str(path) for path in (root, shards) if os.path.lexists(path)]
    if existing:
        raise FinalTestContractError(
            "one-shot output namespace is already consumed: "
            + ", ".join(existing)
        )
    return root, shards


def canonical_clip_id(source_clip_id: str) -> str:
    pieces = source_clip_id.split("/")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        raise FinalTestContractError(
            f"invalid canonical SHOW clip ID {source_clip_id!r}"
        )
    speaker, _video, sequence = pieces
    if speaker not in SHOW_SPEAKER_IDS or "__" in speaker or sequence in {".", ".."}:
        raise FinalTestContractError(
            f"unsafe canonical SHOW clip ID {source_clip_id!r}"
        )
    return f"{speaker}__{sequence}"


def _load_inputs(
    authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    canonical_receipt = authority["canonical"]["manifest"]
    _path, payload, _sha = _safe_file_snapshot(
        canonical_receipt["path"],
        "authorized canonical manifest",
        expected_sha256=canonical_receipt["sha256"],
        expected_bytes=canonical_receipt["bytes"],
    )
    all_rows = _strict_jsonl(payload, "authorized canonical manifest")
    rows = [dict(row) for row in all_rows if row.get("split") == "test"]
    if (
        len(rows) != TEST_CLIPS
        or [row.get("global_index") for row in rows]
        != list(range(TEST_GLOBAL_START, TEST_GLOBAL_STOP))
        or [_canonical_json_sha256(row) for row in rows]
        != authority["canonical"]["ordered_row_sha256"]
    ):
        raise FinalTestContractError(
            "canonical test rows differ from the fresh authority"
        )
    canonical_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        clip_id = row.get("clip_id")
        speaker = row.get("speaker")
        if (
            type(clip_id) is not str
            or clip_id in canonical_by_id
            or speaker not in SHOW_SPEAKER_IDS
            or row.get("speaker_id") != SHOW_SPEAKER_IDS[speaker]
            or clip_id.split("/", 1)[0] != speaker
            or _require_int(row.get("frames"), f"{clip_id} frames", minimum=61)
            < 61
        ):
            raise FinalTestContractError(f"invalid canonical row {clip_id!r}")
        _require_sha256(row.get("canonical_npz_sha256"), f"{clip_id} canonical SHA")
        canonical_by_id[clip_id] = row

    audio_by_id: dict[str, dict[str, Any]] = {}
    for expected_shard, shard in enumerate(authority["audio"]):
        if shard.get("shard_id") != expected_shard:
            raise FinalTestContractError("audio authority shard order changed")
        receipt = shard["manifest"]
        _audio_path, audio_payload, _audio_sha = _safe_file_snapshot(
            receipt["path"],
            f"audio shard {expected_shard} manifest",
            expected_sha256=receipt["sha256"],
            expected_bytes=receipt["bytes"],
        )
        audio_rows = _strict_jsonl(
            audio_payload, f"audio shard {expected_shard} manifest"
        )
        if (
            len(audio_rows) != shard["rows"]
            or [_canonical_json_sha256(row) for row in audio_rows]
            != shard["ordered_row_sha256"]
        ):
            raise FinalTestContractError(
                f"audio shard {expected_shard} row receipt changed"
            )
        for row in audio_rows:
            clip_id = row.get("clip_id")
            canonical = canonical_by_id.get(str(clip_id))
            if (
                canonical is None
                or clip_id in audio_by_id
                or row.get("split") != "test"
                or row.get("shard_id") != expected_shard
                or row.get("num_shards") != NUM_SHARDS
                or row.get("frames") != canonical["frames"]
                or row.get("canonical_npz_sha256")
                != canonical["canonical_npz_sha256"]
            ):
                raise FinalTestContractError(
                    f"audio row {clip_id!r} is not authority-bound"
                )
            _safe_file_snapshot(
                row["audio_feature_npz"],
                f"audio feature {clip_id}",
                expected_sha256=row["audio_feature_npz_sha256"],
            )
            audio_by_id[str(clip_id)] = row
    if set(audio_by_id) != set(canonical_by_id):
        raise FinalTestContractError(
            "audio features do not cover the canonical test exactly once"
        )
    return rows, canonical_by_id, audio_by_id


def _purge_unverified_inference_modules() -> None:
    exact = {
        "scripts.show_base.semtalk_base_inference_core",
        "scripts.show_base.build_base_features",
        "scripts.show_base.selected_prerequisites",
        "scripts.show_base.merge_prerequisite_val_shards",
        "scripts.show_base.prerequisite_val_contract",
    }
    protected = {
        __name__,
        final_authority.__name__,
        "scripts",
        "scripts.show_base",
    }
    for name, module in tuple(sys.modules.items()):
        local = False
        module_file = getattr(module, "__file__", None)
        if module_file:
            try:
                Path(str(module_file)).resolve().relative_to(PROJECT_ROOT.resolve())
                local = True
            except (OSError, ValueError):
                pass
        if (
            name not in protected
            and (
                local
                or name == "models"
                or name.startswith("models.")
                or name in exact
            )
        ):
            sys.modules.pop(name, None)


def _load_core(
    authority: Mapping[str, Any], finder: _VerifiedTreeFinder
) -> ModuleType:
    source = authority["inference_source"]
    source_root = Path(source["source_root"])
    relative = source["entrypoint"]
    path = source_root / relative
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    name = "scripts.show_base.semtalk_base_inference_core"
    module = importlib.import_module(name)
    receipt = getattr(module, "__verified_source_receipt__", None)
    if (
        type(receipt) is not dict
        or receipt.get("relative_path") != relative
        or receipt.get("path") != str(path.resolve())
        or receipt.get("sha256") != source["entrypoint_sha256"]
        or finder.receipts.get(name) != receipt
    ):
        raise FinalTestContractError("neutral inference core was not source-pinned")
    return module


def _pinned_module(
    authority: Mapping[str, Any],
    finder: _VerifiedTreeFinder,
    name: str,
    relative: str,
) -> ModuleType:
    root = Path(authority["inference_source"]["source_root"])
    module = importlib.import_module(name)
    observed = Path(str(getattr(module, "__file__", ""))).resolve()
    expected = (root / relative).resolve()
    receipt = getattr(module, "__verified_source_receipt__", None)
    if (
        observed != expected
        or type(receipt) is not dict
        or receipt.get("relative_path") != relative
        or receipt.get("path") != str(expected)
        or finder.receipts.get(name) != receipt
    ):
        raise FinalTestContractError(f"{name} was not source-pinned")
    return module


def _checkpoint_receipts(
    authority: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage in CHECKPOINT_STAGES:
        value = authority["checkpoints"][stage]
        path, payload, observed = _safe_file_snapshot(
            value["path"],
            f"{stage} checkpoint",
            expected_sha256=value["sha256"],
            expected_bytes=value["bytes"],
        )
        result[stage] = {
            "formal_stage": stage,
            "path": str(path),
            "sha256": observed,
            "bytes": len(payload),
        }
    return result


def _load_models(
    core: ModuleType,
    authority: Mapping[str, Any],
    finder: _VerifiedTreeFinder,
    *,
    device: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    base_artifact = authority["checkpoints"]["base"]
    base_path, base_snapshot, base_sha = core._read_verified_checkpoint_snapshot(
        Path(base_artifact["path"]),
        base_artifact["sha256"],
        "final-authority Base winner",
    )
    if len(base_snapshot) != base_artifact["bytes"]:
        raise FinalTestContractError("Base winner byte count changed")
    base_payload = core._torch_load_checkpoint(base_snapshot, base_path)
    if type(base_payload) is not dict or set(base_payload) != {"model_state", "audit"}:
        raise FinalTestContractError("Base winner checkpoint envelope changed")
    core._finite_state_dict(base_payload["model_state"], base_path)
    core._validate_base_model_state_schema(base_payload["model_state"], base_path)
    audit = base_payload["audit"]
    selected = authority["winner_selection"]
    bundle = authority["base_long_candidate_bundle"]
    if (
        type(audit) is not dict
        or audit.get("format") != core.OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT
        or audit.get("completed_epochs") != selected["selected_epoch"]
        or audit.get("optimizer_updates")
        != selected["selected_optimizer_updates"]
        or audit.get("frozen_receipt_sha256")
        != bundle["frozen_inputs"]["receipt_sha256"]
        or audit.get("official_base_checkpoint_sha256")
        != core.RELEASED_ALL_SPEAKERS_MODELS["base"]["sha256"]
        or audit.get("speaker_scope") != "SHOW_All"
        or audit.get("speaker_rows") != [0, 1, 2, 3]
        or audit.get("vq_models_in_training_graph") is not False
        or audit.get("all_model_state_tensors_finite") is not True
    ):
        raise FinalTestContractError(
            "Base winner audit differs from the validation-selected authority"
        )
    semtalk_module = _pinned_module(
        authority, finder, "models.semtalk", "models/semtalk.py"
    )
    base = semtalk_module.semtalk_base(core._model_args()).to(device)
    core._strict_load_freeze_eval(
        base,
        core._normalize_data_parallel_state(base_payload["model_state"], base_path),
        path=base_path,
    )
    del base_payload

    feature_builder = _pinned_module(
        authority,
        finder,
        "scripts.show_base.build_base_features",
        "scripts/show_base/build_base_features.py",
    )
    selection = authority["prerequisite_selection"]
    try:
        selected_models, selected_records, bridge = (
            feature_builder.load_val_selected_models(
                SimpleNamespace(
                    prerequisite_selection_json=Path(selection["path"]),
                    expected_prerequisite_selection_sha256=selection["sha256"],
                    device=device,
                ),
                retain_global=True,
            )
        )
    except Exception as exc:
        raise FinalTestContractError(
            "cannot strict-load the validation-selected five SHOW models"
        ) from exc
    if set(selected_models) != set(REPRESENTATION_STAGES) or set(
        selected_records
    ) != set(REPRESENTATION_STAGES):
        raise FinalTestContractError("selected-five model coverage changed")
    bridge_selection = bridge.get("selection")
    if (
        type(bridge_selection) is not dict
        or any(
            bridge_selection.get(key) != selection[key]
            for key in ("path", "sha256", "receipt_payload_sha256")
        )
    ):
        raise FinalTestContractError("selected-five bridge changed")

    explicit = selection["stages"]
    models: dict[str, Any] = {"base": base}
    receipts = _checkpoint_receipts(authority)
    for stage in REPRESENTATION_STAGES:
        record = selected_records[stage]
        expected = explicit[stage]
        checkpoint = authority["checkpoints"][stage]
        measurement = record.get("measurement_receipt")
        if (
            record.get("formal_stage") != stage
            or record.get("path") != checkpoint["path"]
            or record.get("sha256") != checkpoint["sha256"]
            or record.get("bytes") != checkpoint["bytes"]
            or record.get("prerequisite_source") != "show_val_selected_v1"
            or record.get("selection_split") != "val"
            or record.get("test_visible") is not False
            or record.get("selected_epoch") != expected["epoch"]
            or record.get("selected_optimizer_updates")
            != expected["optimizer_updates"]
            or record.get("selection_metric") != expected["selection_metric"]
            or measurement != expected["measurement_receipt"]
            or record.get("strict_state_dict_load") is not True
            or record.get("all_model_state_tensors_finite") is not True
            or record.get("frozen_eval") is not True
        ):
            raise FinalTestContractError(
                f"loaded {stage} differs from the authority-selected winner"
            )
        models[stage] = selected_models[stage]
    for model in models.values():
        model.eval()
        model.requires_grad_(False)
    if set(models) != set(CHECKPOINT_STAGES):
        raise FinalTestContractError("runtime model set is not Base plus five")
    torch.cuda.empty_cache()
    return models, receipts


def _set_deterministic(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _runtime(core: ModuleType, *, device: str, seed: int) -> dict[str, Any]:
    import torch

    parsed = torch.device(device)
    if (
        parsed.type != "cuda"
        or parsed.index not in {None, 0}
        or not torch.cuda.is_available()
    ):
        raise FinalTestContractError(
            "formal shard inference requires isolated device cuda:0"
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        "format": RUNTIME_FORMAT,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "device": "cuda:0",
        "device_name": properties.name,
        "device_capability": [int(properties.major), int(properties.minor)],
        "device_total_memory": int(properties.total_memory),
        "seed": seed,
        "deterministic_algorithms": True,
        "window": int(core.WINDOW),
        "pre_frames": int(core.PRE_FRAMES),
        "stride": int(core.STRIDE),
        "auxiliary_loss_bypass": core._inference_auxiliary_loss_bypass_receipt(),
    }


def _contract(
    authority: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    source = authority["inference_source"]
    selection = authority["winner_selection"]
    prerequisites = authority["prerequisite_selection"]
    receipt = {
        "format": CONTRACT_FORMAT,
        "generator": "SemTalk Base-only",
        "source": dict(source),
        "fresh_test_authority": _authority_artifact(args),
        "canonical_manifest": authority["canonical"]["manifest"]["path"],
        "canonical_manifest_sha256": authority["canonical"]["manifest"]["sha256"],
        "canonical_summary_sha256": authority["canonical"]["summary"]["sha256"],
        "canonical_lineage_sha256": authority["canonical"]["lineage"]["sha256"],
        "audio_manifest_sha256": {
            shard["manifest"]["path"]: shard["manifest"]["sha256"]
            for shard in authority["audio"]
        },
        "audio_summary_sha256": {
            shard["summary"]["path"]: shard["summary"]["sha256"]
            for shard in authority["audio"]
        },
        "audio_lineage_sha256": {
            shard["lineage"]["path"]: shard["lineage"]["sha256"]
            for shard in authority["audio"]
        },
        "checkpoints": {
            stage: {
                "path": authority["checkpoints"][stage]["path"],
                "expected_sha256": authority["checkpoints"][stage]["sha256"],
            }
            for stage in CHECKPOINT_STAGES
        },
        "validation_winner": {
            "selection": {
                key: selection[key]
                for key in ("path", "sha256", "bytes", "receipt_payload_sha256")
            },
            "selected_epoch": selection["selected_epoch"],
            "selected_optimizer_updates": selection["selected_optimizer_updates"],
            "selected_checkpoint_sha256": selection["selected_checkpoint"]["sha256"],
        },
        "validation_selected_five": {
            "selection": {
                key: prerequisites[key]
                for key in ("path", "sha256", "bytes", "receipt_payload_sha256")
            },
            "checkpoint_sha256": {
                stage: authority["checkpoints"][stage]["sha256"]
                for stage in REPRESENTATION_STAGES
            },
        },
        "selection_policy": {
            "primary_metric": "body.released2.metrics.FGD",
            "mode": "min",
            "validation_only_for_selection": True,
            "test_evaluations": 1,
            "test_feedback_into_selection": False,
        },
        "speaker_mapping": dict(SHOW_SPEAKER_IDS),
        "split": "test",
        "test_clips": TEST_CLIPS,
        "num_shards": NUM_SHARDS,
        "seed": args.seed,
        "physical_predictions_per_clip": 1,
        "deterministic": True,
        "exact_once": True,
        "sparse_motion_generation": False,
        "semgate": False,
        "output": {
            "root": authority["expected_output_root"],
            "prediction": "npz/test/res_<speaker>__<sequence>.npz",
            "ground_truth": "npz/test/gt_<speaker>__<sequence>.npz",
        },
    }
    prepared = _prepared_authority_artifact(args)
    if prepared is not None:
        receipt["prepared_authority"] = prepared
    return receipt


def _shard_name(shard_id: int) -> str:
    return f"shard-{shard_id:05d}-of-{NUM_SHARDS:05d}"


def run_context(args: argparse.Namespace) -> dict[str, Any]:
    authority = _validated_authority(args)
    root, shards = _require_new_output(authority)
    result = {
        "expected_output_root": str(root),
        "shards_root": str(shards),
        "canonical_manifest": authority["canonical"]["manifest"]["path"],
        "canonical_manifest_sha256": authority["canonical"]["manifest"]["sha256"],
        "source_commit": authority["inference_source"]["commit"],
        "source_tree": authority["inference_source"]["tree"],
        "selected_base_epoch": authority["winner_selection"]["selected_epoch"],
    }
    prepared_output = getattr(args, "prepared_authority_output", None)
    if prepared_output is not None:
        prepared_path = Path(prepared_output).expanduser()
        if not prepared_path.is_absolute():
            raise FinalTestContractError(
                "prepared authority output must be absolute"
            )
        prepared_path = prepared_path.parent.resolve() / prepared_path.name
        prepared_unsigned = {
            "format": PREPARED_AUTHORITY_FORMAT,
            "status": "complete",
            "authority_artifact": _authority_artifact(args),
            "authority": authority,
            "producer": {
                "adapter": str(Path(__file__).resolve()),
                "adapter_sha256": _artifact(Path(__file__).resolve())[
                    "sha256"
                ],
                "source_root": str(PROJECT_ROOT.resolve()),
            },
        }
        prepared = {
            **prepared_unsigned,
            "receipt_payload_sha256": _canonical_json_sha256(
                prepared_unsigned
            ),
        }
        _write_new(prepared_path, _canonical_json_bytes(prepared))
        prepared_artifact = {
            **_artifact(prepared_path),
            "receipt_payload_sha256": prepared[
                "receipt_payload_sha256"
            ],
        }
        result["prepared_authority"] = prepared_artifact
    if args.nul_context:
        for key in (
            "expected_output_root",
            "shards_root",
            "canonical_manifest",
            "canonical_manifest_sha256",
            "source_commit",
            "source_tree",
        ):
            sys.stdout.buffer.write(str(result[key]).encode("utf-8") + b"\0")
        if prepared_output is not None:
            for key in (
                "path",
                "sha256",
                "bytes",
                "receipt_payload_sha256",
            ):
                sys.stdout.buffer.write(
                    str(result["prepared_authority"][key]).encode("utf-8")
                    + b"\0"
                )
        sys.stdout.buffer.flush()
    else:
        print(_canonical_json_bytes(result).decode("utf-8"), end="")
    return result


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    shard_id = _require_int(args.shard_id, "shard_id")
    if shard_id >= NUM_SHARDS:
        raise FinalTestContractError("shard_id must be in 0..7")
    authority = _validated_authority(args)
    root, shards_root = _output_roots(authority)
    if os.path.lexists(root):
        raise FinalTestContractError("final output already exists")
    shards_root.mkdir(parents=True, exist_ok=True)
    if shards_root.is_symlink() or not shards_root.is_dir():
        raise FinalTestContractError("shards root is unsafe")
    final_root = shards_root / _shard_name(shard_id)
    if os.path.lexists(final_root):
        raise FileExistsError(f"refusing to overwrite {final_root}")
    stage = shards_root / (
        f".{_shard_name(shard_id)}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    stage.mkdir()
    npz_stage = stage / "npz" / "test"
    npz_stage.mkdir(parents=True)
    npz_final = final_root / "npz" / "test"
    finder: _VerifiedTreeFinder | None = None
    try:
        canonical_rows, _canonical_by_id, audio_by_id = _load_inputs(authority)
        finder = _VerifiedTreeFinder(authority)
        _purge_unverified_inference_modules()
        sys.meta_path.insert(0, finder)
        core = _load_core(authority, finder)
        _set_deterministic(args.seed)
        runtime = _runtime(core, device=args.device, seed=args.seed)
        contract = _contract(authority, args)
        contract_sha = _canonical_json_sha256(contract)
        models, checkpoint_receipts = _load_models(
            core, authority, finder, device=args.device
        )
        torch = importlib.import_module("torch")
        masks = core._joint_masks(torch.device(args.device))
        output_rows: list[dict[str, Any]] = []
        for canonical in canonical_rows:
            global_index = _require_int(
                canonical.get("global_index"), "canonical global_index"
            )
            if global_index % NUM_SHARDS != shard_id:
                continue
            clip_id = str(canonical["clip_id"])
            output_id = canonical_clip_id(clip_id)
            canonical_arrays, frames = core._load_canonical_clip(canonical)
            audio_row = audio_by_id[clip_id]
            audio = core._load_audio_features(audio_row, expected_frames=frames)
            expected_calls = max(
                1,
                math.ceil((frames - core.PRE_FRAMES) / core.STRIDE),
            )
            with (
                torch.inference_mode(),
                core._inference_only_auxiliary_loss_bypass(
                    models["base"], expected_calls=expected_calls
                ),
            ):
                prediction = core._infer_clip(
                    pose=canonical_arrays["pose"],
                    trans=canonical_arrays["trans"],
                    beat=audio["beat"],
                    hubert=audio["hubert"],
                    speaker_id=SHOW_SPEAKER_IDS[str(canonical["speaker"])],
                    models=models,
                    masks=masks,
                    device=args.device,
                )
            prediction_arrays = core._output_arrays(
                betas=canonical_arrays["beta"][0],
                poses=prediction["poses"],
                expressions=prediction["expressions"],
                trans=prediction["trans"],
            )
            ground_truth_arrays = core._output_arrays(
                betas=canonical_arrays["beta"][0],
                poses=canonical_arrays["pose"],
                expressions=canonical_arrays["facial"],
                trans=canonical_arrays["trans"],
            )
            prediction_payload = core.deterministic_npz_bytes(prediction_arrays)
            ground_truth_payload = core.deterministic_npz_bytes(
                ground_truth_arrays
            )
            prediction_name = f"res_{output_id}.npz"
            target_name = f"gt_{output_id}.npz"
            _write_new(npz_stage / prediction_name, prediction_payload)
            _write_new(npz_stage / target_name, ground_truth_payload)
            output_rows.append(
                {
                    "global_index": global_index,
                    "source_clip_id": clip_id,
                    "canonical_clip_id": output_id,
                    "speaker": canonical["speaker"],
                    "speaker_id": canonical["speaker_id"],
                    "frames": frames,
                    "canonical_npz": str(Path(canonical["canonical_npz"]).resolve()),
                    "canonical_npz_sha256": canonical["canonical_npz_sha256"],
                    "audio_feature_npz": str(
                        Path(audio_row["audio_feature_npz"]).resolve()
                    ),
                    "audio_feature_npz_sha256": audio_row[
                        "audio_feature_npz_sha256"
                    ],
                    "prediction": {
                        "path": str(npz_final / prediction_name),
                        "sha256": _sha256_bytes(prediction_payload),
                        "bytes": len(prediction_payload),
                    },
                    "ground_truth": {
                        "path": str(npz_final / target_name),
                        "sha256": _sha256_bytes(ground_truth_payload),
                        "bytes": len(ground_truth_payload),
                    },
                }
            )
            if args.progress_every and len(output_rows) % args.progress_every == 0:
                print(
                    f"final test shard {shard_id}: {len(output_rows)} clips",
                    flush=True,
                )
        expected_count = sum(
            1
            for index in range(TEST_GLOBAL_START, TEST_GLOBAL_STOP)
            if index % NUM_SHARDS == shard_id
        )
        if len(output_rows) != expected_count:
            raise FinalTestContractError(
                f"shard {shard_id} clip count {len(output_rows)} != {expected_count}"
            )
        output_rows.sort(key=lambda row: row["global_index"])
        manifest_payload = _canonical_jsonl_bytes(output_rows)
        manifest_sha = _sha256_bytes(manifest_payload)
        runtime["verified_source_modules"] = {
            name: finder.receipts[name]
            for name in sorted(finder.receipts)
        }
        runtime_sha = _canonical_json_sha256(runtime)
        summary = {
            "format": SHARD_SUMMARY_FORMAT,
            "status": "complete",
            "shard_id": shard_id,
            "num_shards": NUM_SHARDS,
            "selected_clips": len(output_rows),
            "expected_test_clips": TEST_CLIPS,
            "manifest_sha256": manifest_sha,
            "contract_sha256": contract_sha,
            "runtime_sha256": runtime_sha,
            "finite": True,
            "exact_once": True,
        }
        lineage = {
            "format": SHARD_LINEAGE_FORMAT,
            "status": "complete",
            "shard_id": shard_id,
            "num_shards": NUM_SHARDS,
            "manifest_sha256": manifest_sha,
            "contract": contract,
            "contract_sha256": contract_sha,
            "runtime": runtime,
            "runtime_sha256": runtime_sha,
            "checkpoints": checkpoint_receipts,
        }
        _write_new(stage / "manifest.jsonl", manifest_payload)
        _write_new(stage / "lineage.json", _canonical_json_bytes(lineage))
        # Completion marker is deliberately last.
        _write_new(stage / "summary.json", _canonical_json_bytes(summary))
        _fsync_directory(npz_stage)
        _fsync_directory(npz_stage.parent)
        _fsync_directory(stage)
        if _validated_authority(args) != authority:
            raise FinalTestContractError("authority changed during shard inference")
        os.rename(stage, final_root)
        _fsync_directory(shards_root)
        return summary
    except BaseException:
        if os.path.lexists(stage):
            shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if finder is not None:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass


def _load_exact_artifact(receipt: Any, *, expected: Path, label: str) -> bytes:
    if type(receipt) is not dict or set(receipt) != ARTIFACT_KEYS:
        raise FinalTestContractError(f"{label} receipt schema mismatch")
    path = Path(str(receipt["path"]))
    if path.resolve() != expected.resolve():
        raise FinalTestContractError(f"{label} escapes its exact path")
    _path, payload, _sha = _safe_file_snapshot(
        path,
        label,
        expected_sha256=receipt["sha256"],
        expected_bytes=receipt["bytes"],
    )
    return payload


@contextmanager
def _finalize_lock(root: Path) -> Iterable[None]:
    lock = root.parent / f".{root.name}.finalize.lock"
    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FinalTestContractError("another finalizer holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_shards(
    authority: Mapping[str, Any], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    root, shards_root = _output_roots(authority)
    if os.path.lexists(root):
        raise FinalTestContractError("final output root already exists")
    canonical_rows, canonical_by_id, audio_by_id = _load_inputs(authority)
    core = _load_core(authority)
    contract = _contract(authority, args)
    contract_sha = _canonical_json_sha256(contract)
    expected_checkpoints = _checkpoint_receipts(authority)
    rows_by_index: dict[int, dict[str, Any]] = {}
    shard_receipts: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    runtime_sha: str | None = None
    for shard_id in range(NUM_SHARDS):
        shard_root = shards_root / _shard_name(shard_id)
        manifest_path = shard_root / "manifest.jsonl"
        summary_path = shard_root / "summary.json"
        lineage_path = shard_root / "lineage.json"
        _manifest_file, manifest_payload, manifest_sha = _safe_file_snapshot(
            manifest_path, f"shard {shard_id} manifest"
        )
        _summary_file, summary_payload, summary_sha = _safe_file_snapshot(
            summary_path, f"shard {shard_id} summary"
        )
        _lineage_file, lineage_payload, lineage_sha = _safe_file_snapshot(
            lineage_path, f"shard {shard_id} lineage"
        )
        summary = _strict_json(summary_payload, f"shard {shard_id} summary")
        lineage = _strict_json(lineage_payload, f"shard {shard_id} lineage")
        rows = _strict_jsonl(manifest_payload, f"shard {shard_id} manifest")
        expected_summary_keys = {
            "format", "status", "shard_id", "num_shards", "selected_clips",
            "expected_test_clips", "manifest_sha256", "contract_sha256",
            "runtime_sha256", "finite", "exact_once",
        }
        expected_lineage_keys = {
            "format", "status", "shard_id", "num_shards", "manifest_sha256",
            "contract", "contract_sha256", "runtime", "runtime_sha256",
            "checkpoints",
        }
        if (
            type(summary) is not dict
            or set(summary) != expected_summary_keys
            or summary.get("format") != SHARD_SUMMARY_FORMAT
            or summary.get("status") != "complete"
            or summary.get("shard_id") != shard_id
            or summary.get("num_shards") != NUM_SHARDS
            or summary.get("selected_clips") != len(rows)
            or summary.get("expected_test_clips") != TEST_CLIPS
            or summary.get("manifest_sha256") != manifest_sha
            or summary.get("contract_sha256") != contract_sha
            or summary.get("finite") is not True
            or summary.get("exact_once") is not True
        ):
            raise FinalTestContractError(f"shard {shard_id} summary mismatch")
        if (
            type(lineage) is not dict
            or set(lineage) != expected_lineage_keys
            or lineage.get("format") != SHARD_LINEAGE_FORMAT
            or lineage.get("status") != "complete"
            or lineage.get("shard_id") != shard_id
            or lineage.get("num_shards") != NUM_SHARDS
            or lineage.get("manifest_sha256") != manifest_sha
            or lineage.get("contract") != contract
            or lineage.get("contract_sha256") != contract_sha
            or lineage.get("checkpoints") != expected_checkpoints
            or _canonical_json_sha256(lineage.get("runtime"))
            != lineage.get("runtime_sha256")
            or summary.get("runtime_sha256") != lineage.get("runtime_sha256")
        ):
            raise FinalTestContractError(f"shard {shard_id} lineage mismatch")
        if runtime is None:
            runtime = dict(lineage["runtime"])
            runtime_sha = str(lineage["runtime_sha256"])
        elif lineage["runtime"] != runtime or lineage["runtime_sha256"] != runtime_sha:
            raise FinalTestContractError("shards used different runtimes")
        expected_files: set[str] = set()
        for row in rows:
            if type(row) is not dict or set(row) != SHARD_ROW_KEYS:
                raise FinalTestContractError(f"shard {shard_id} row schema mismatch")
            index = _require_int(row.get("global_index"), "shard global_index")
            if index % NUM_SHARDS != shard_id or index in rows_by_index:
                raise FinalTestContractError("duplicate or misassigned shard row")
            canonical = canonical_by_id.get(str(row.get("source_clip_id")))
            if canonical is None or canonical.get("global_index") != index:
                raise FinalTestContractError("shard row is not canonical-bound")
            clip_id = str(canonical["clip_id"])
            audio = audio_by_id[clip_id]
            output_id = canonical_clip_id(clip_id)
            expected_bindings = {
                "global_index": index,
                "source_clip_id": clip_id,
                "canonical_clip_id": output_id,
                "speaker": canonical["speaker"],
                "speaker_id": canonical["speaker_id"],
                "frames": canonical["frames"],
                "canonical_npz": str(Path(canonical["canonical_npz"]).resolve()),
                "canonical_npz_sha256": canonical["canonical_npz_sha256"],
                "audio_feature_npz": str(Path(audio["audio_feature_npz"]).resolve()),
                "audio_feature_npz_sha256": audio["audio_feature_npz_sha256"],
            }
            if {key: row.get(key) for key in expected_bindings} != expected_bindings:
                raise FinalTestContractError(f"{output_id} input binding changed")
            npz_root = shard_root / "npz" / "test"
            prediction_name = f"res_{output_id}.npz"
            target_name = f"gt_{output_id}.npz"
            _load_exact_artifact(
                row["prediction"],
                expected=npz_root / prediction_name,
                label=f"{output_id} prediction",
            )
            target_payload = _load_exact_artifact(
                row["ground_truth"],
                expected=npz_root / target_name,
                label=f"{output_id} ground truth",
            )
            arrays, frames = core._load_canonical_clip(canonical)
            core._load_audio_features(audio, expected_frames=frames)
            expected_target = core.deterministic_npz_bytes(
                core._output_arrays(
                    betas=arrays["beta"][0],
                    poses=arrays["pose"],
                    expressions=arrays["facial"],
                    trans=arrays["trans"],
                )
            )
            if target_payload != expected_target:
                raise FinalTestContractError(
                    f"{output_id} ground truth is not canonical"
                )
            expected_files.update((prediction_name, target_name))
            rows_by_index[index] = row
        npz_root = shard_root / "npz" / "test"
        if (
            not npz_root.is_dir()
            or npz_root.is_symlink()
            or {path.name for path in npz_root.iterdir()} != expected_files
            or any(
                path.is_symlink() or not path.is_file()
                for path in npz_root.iterdir()
            )
        ):
            raise FinalTestContractError(f"shard {shard_id} NPZ inventory changed")
        shard_receipts.append(
            {
                "shard_id": shard_id,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "summary": str(summary_path),
                "summary_sha256": summary_sha,
                "lineage": str(lineage_path),
                "lineage_sha256": lineage_sha,
                "clips": len(rows),
            }
        )
    if set(rows_by_index) != set(range(TEST_GLOBAL_START, TEST_GLOBAL_STOP)):
        raise FinalTestContractError("shards do not cover SHOW test exactly once")
    if runtime is None or runtime_sha is None:
        raise FinalTestContractError("missing shard runtime")
    ordered = sorted(rows_by_index.values(), key=lambda row: row["canonical_clip_id"])
    if _validated_authority(args) != authority:
        raise FinalTestContractError("authority changed while shards were validated")
    return ordered, shard_receipts, runtime, runtime_sha


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    authority = _validated_authority(args)
    root, _shards = _output_roots(authority)
    with _finalize_lock(root):
        if os.path.lexists(root):
            raise FileExistsError(f"refusing to overwrite {root}")
        rows, shard_receipts, runtime, runtime_sha = _validate_shards(
            authority, args
        )
        contract = _contract(authority, args)
        contract_sha = _canonical_json_sha256(contract)
        stage = root.parent / (
            f".{root.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        test_stage = stage / "npz" / "test"
        stage.mkdir()
        test_stage.mkdir(parents=True)
        test_final = root / "npz" / "test"
        published = False
        try:
            final_rows: list[dict[str, Any]] = []
            for evaluation_index, row in enumerate(rows):
                output_id = str(row["canonical_clip_id"])
                prediction_name = f"res_{output_id}.npz"
                target_name = f"gt_{output_id}.npz"
                _copy_new(
                    Path(row["prediction"]["path"]),
                    test_stage / prediction_name,
                    expected_sha256=row["prediction"]["sha256"],
                    expected_bytes=row["prediction"]["bytes"],
                )
                _copy_new(
                    Path(row["ground_truth"]["path"]),
                    test_stage / target_name,
                    expected_sha256=row["ground_truth"]["sha256"],
                    expected_bytes=row["ground_truth"]["bytes"],
                )
                final_rows.append(
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key not in {"prediction", "ground_truth"}
                        },
                        "prediction": {
                            "path": str(test_final / prediction_name),
                            "sha256": row["prediction"]["sha256"],
                            "bytes": row["prediction"]["bytes"],
                        },
                        "ground_truth": {
                            "path": str(test_final / target_name),
                            "sha256": row["ground_truth"]["sha256"],
                            "bytes": row["ground_truth"]["bytes"],
                        },
                        "evaluation_index": evaluation_index,
                    }
                )
            if any(set(row) != FINAL_ROW_KEYS for row in final_rows):
                raise FinalTestContractError("final manifest row schema changed")
            clip_payload = "".join(
                f"{row['canonical_clip_id']}\n" for row in final_rows
            ).encode("utf-8")
            manifest_payload = _canonical_jsonl_bytes(final_rows)
            lineage = {
                "format": FINAL_LINEAGE_FORMAT,
                "status": "complete",
                "contract": contract,
                "contract_sha256": contract_sha,
                "runtime": runtime,
                "runtime_sha256": runtime_sha,
                "shards": shard_receipts,
                "final_manifest_sha256": _sha256_bytes(manifest_payload),
                "clip_manifest_sha256": _sha256_bytes(clip_payload),
            }
            summary = {
                "format": FINAL_SUMMARY_FORMAT,
                "status": "complete",
                "generator": "SemTalk Base-only",
                "test_clips": len(final_rows),
                "prediction_files": len(final_rows),
                "ground_truth_files": len(final_rows),
                "num_shards": NUM_SHARDS,
                "npz_root": str(test_final),
                "manifest_sha256": lineage["final_manifest_sha256"],
                "clip_manifest_sha256": lineage["clip_manifest_sha256"],
                "lineage_sha256": _canonical_json_sha256(lineage),
                "contract_sha256": contract_sha,
                "runtime_sha256": runtime_sha,
                "finite": True,
                "exact_once": True,
                "split_disjoint": True,
                "test_evaluations": 0,
                "test_feedback_into_selection": False,
            }
            _write_new(stage / "diffsheg_eval_clip_ids.txt", clip_payload)
            _write_new(stage / "final_manifest.jsonl", manifest_payload)
            _write_new(stage / "final_lineage.json", _canonical_json_bytes(lineage))
            _fsync_directory(test_stage)
            _fsync_directory(test_stage.parent)
            if _validated_authority(args) != authority:
                raise FinalTestContractError("authority changed during finalization")
            # Summary is the final-generation completion marker.
            _write_new(stage / "final_summary.json", _canonical_json_bytes(summary))
            _fsync_directory(stage)
            os.rename(stage, root)
            published = True
            _fsync_directory(root.parent)
            print(
                _canonical_json_bytes(
                    _artifact(root / "final_manifest.jsonl")
                ).decode(),
                end="",
            )
            return summary
        except BaseException:
            if not published:
                shutil.rmtree(stage, ignore_errors=True)
            raise


def _load_final_rows(
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root, _shards = _output_roots(authority)
    manifest = _artifact(root / "final_manifest.jsonl")
    _path, payload, _sha = _safe_file_snapshot(
        manifest["path"],
        "final manifest",
        expected_sha256=manifest["sha256"],
        expected_bytes=manifest["bytes"],
    )
    rows = _strict_jsonl(payload, "final manifest")
    _lineage_path, lineage_payload, _lineage_sha = _safe_file_snapshot(
        root / "final_lineage.json", "final lineage"
    )
    lineage = _strict_json(lineage_payload, "final lineage")
    expected_lineage_keys = {
        "format", "status", "contract", "contract_sha256", "runtime",
        "runtime_sha256", "shards", "final_manifest_sha256",
        "clip_manifest_sha256",
    }
    clip_payload = "".join(
        f"{row.get('canonical_clip_id')}\n" for row in rows
    ).encode("utf-8")
    if (
        len(rows) != TEST_CLIPS
        or any(type(row) is not dict or set(row) != FINAL_ROW_KEYS for row in rows)
        or [row.get("evaluation_index") for row in rows] != list(range(TEST_CLIPS))
        or [row.get("canonical_clip_id") for row in rows]
        != sorted(row.get("canonical_clip_id") for row in rows)
        or sorted(row.get("global_index") for row in rows)
        != list(range(TEST_GLOBAL_START, TEST_GLOBAL_STOP))
        or type(lineage) is not dict
        or set(lineage) != expected_lineage_keys
        or lineage.get("format") != FINAL_LINEAGE_FORMAT
        or lineage.get("status") != "complete"
        or lineage.get("final_manifest_sha256") != manifest["sha256"]
        or _canonical_json_sha256(lineage.get("contract"))
        != lineage.get("contract_sha256")
        or _canonical_json_sha256(lineage.get("runtime"))
        != lineage.get("runtime_sha256")
        or lineage.get("clip_manifest_sha256") != _sha256_bytes(clip_payload)
        or type(lineage.get("shards")) is not list
        or len(lineage["shards"]) != NUM_SHARDS
    ):
        raise FinalTestContractError("final output is not exact-once complete")
    npz_root = root / "npz" / "test"
    expected_files: set[str] = set()
    for row in rows:
        output_id = str(row["canonical_clip_id"])
        prediction_name = f"res_{output_id}.npz"
        target_name = f"gt_{output_id}.npz"
        _load_exact_artifact(
            row["prediction"],
            expected=npz_root / prediction_name,
            label=f"final {output_id} prediction",
        )
        _load_exact_artifact(
            row["ground_truth"],
            expected=npz_root / target_name,
            label=f"final {output_id} ground truth",
        )
        expected_files.update((prediction_name, target_name))
    if (
        not npz_root.is_dir()
        or npz_root.is_symlink()
        or {path.name for path in npz_root.iterdir()} != expected_files
        or any(path.is_symlink() or not path.is_file() for path in npz_root.iterdir())
    ):
        raise FinalTestContractError("final prediction inventory changed")
    return manifest, rows, lineage


def run_distribution(args: argparse.Namespace) -> dict[str, Any]:
    authority = _validated_authority(args)
    manifest, rows, lineage = _load_final_rows(authority)
    if lineage.get("contract") != _contract(authority, args):
        raise FinalTestContractError(
            "final lineage differs from the fresh test authority"
        )
    gate_module = importlib.import_module(
        "scripts.show_base.deterministic_replication_gate"
    )
    gate_artifact, _gate = gate_module.load_gate(
        Path(args.validation_gate_json),
        args.expected_validation_gate_sha256,
        expected_scope="final_winner",
    )
    if (
        gate_artifact["bytes"]
        != _require_int(
            args.expected_validation_gate_bytes,
            "validation gate bytes",
            minimum=1,
        )
        or gate_artifact["receipt_payload_sha256"]
        != _require_sha256(
            args.expected_validation_gate_receipt_payload_sha256,
            "validation gate payload SHA",
        )
    ):
        raise FinalTestContractError("final_winner gate external pins changed")
    predictions = [
        {
            "canonical_clip_id": row["canonical_clip_id"],
            "prediction_sha256": row["prediction"]["sha256"],
            "prediction_bytes": row["prediction"]["bytes"],
        }
        for row in rows
    ]
    declaration = gate_module.build_distribution_receipt_from_validated_artifacts(
        gate_artifact=gate_artifact,
        prediction_manifest_artifact=manifest,
        prediction_records=predictions,
    )
    root, _shards = _output_roots(authority)
    destination = root / "distribution-declaration.json"
    _write_new(destination, _canonical_json_bytes(declaration))
    receipt = _artifact(destination)
    print(_canonical_json_bytes(receipt).decode(), end="")
    return receipt


def run_seal(args: argparse.Namespace) -> dict[str, Any]:
    authority = _validated_authority(args)
    manifest, rows, lineage = _load_final_rows(authority)
    if lineage.get("contract") != _contract(authority, args):
        raise FinalTestContractError(
            "final lineage differs from the fresh test authority"
        )
    root, _shards = _output_roots(authority)
    distribution_path, distribution_payload, distribution_sha = _safe_file_snapshot(
        args.distribution_declaration_json,
        "distribution declaration",
        expected_sha256=args.expected_distribution_declaration_sha256,
    )
    distribution = _strict_json(distribution_payload, "distribution declaration")
    report_path, report_payload, report_sha = _safe_file_snapshot(
        args.metric_report_json,
        "TalkSHOW final test report",
        expected_sha256=args.expected_metric_report_sha256,
    )
    report = _strict_json(report_payload, "TalkSHOW final test report")
    evaluator = importlib.import_module(
        "scripts.show_base.evaluate_talkshow_show_metrics"
    )
    authority_artifact = _authority_artifact(args)
    selection_protocol = {
        "primary_metric": "body.released2.metrics.FGD",
        "mode": "min",
        "validation_only_for_selection": True,
        "test_evaluations": 1,
    }
    evaluator.validate_report(
        report,
        expected_split="test",
        expected_clip_count=TEST_CLIPS,
        expected_prediction_manifest=manifest,
        expected_distribution_receipt=distribution,
        expected_selection_protocol=selection_protocol,
        expected_test_authority=authority_artifact,
    )
    if (
        report.get("selection_protocol") != selection_protocol
        or lineage.get("contract", {}).get("selection_policy", {}).get(
            "test_feedback_into_selection"
        )
        is not False
    ):
        raise FinalTestContractError("test report can flow back into selection")
    completion = {
        "format": COMPLETION_FORMAT,
        "status": "complete",
        "fresh_test_authority": authority_artifact,
        "validation_winner": {
            "selected_epoch": authority["winner_selection"]["selected_epoch"],
            "selected_optimizer_updates": authority["winner_selection"]
            ["selected_optimizer_updates"],
            "checkpoint_sha256": authority["checkpoints"]["base"]["sha256"],
        },
        "validation_selected_five_sha256": {
            stage: authority["checkpoints"][stage]["sha256"]
            for stage in REPRESENTATION_STAGES
        },
        "prediction_manifest": manifest,
        "prediction_lineage": _artifact(root / "final_lineage.json"),
        "distribution_declaration": {
            "path": str(distribution_path),
            "sha256": distribution_sha,
            "bytes": len(distribution_payload),
            "receipt_payload_sha256": distribution[
                "receipt_payload_sha256"
            ],
        },
        "metric_report": {
            "path": str(report_path),
            "sha256": report_sha,
            "bytes": len(report_payload),
            "report_payload_sha256": report["report_payload_sha256"],
        },
        "selection_protocol": selection_protocol,
        "test_evaluations": 1,
        "test_feedback_into_selection": False,
        "training_mutations_after_test": 0,
        "exact_once": True,
        "finite": True,
        "clips": len(rows),
    }
    completion["receipt_payload_sha256"] = _canonical_json_sha256(completion)
    destination = root / "formal_completion.json"
    _write_new(destination, _canonical_json_bytes(completion))
    receipt = _artifact(destination)
    print(_canonical_json_bytes(receipt).decode(), end="")
    return receipt


def _authority_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fresh-test-authority", type=Path, required=True)
    parser.add_argument("--expected-test-authority-sha256", required=True)
    parser.add_argument("--expected-test-authority-bytes", type=int, required=True)
    parser.add_argument(
        "--expected-test-authority-receipt-payload-sha256", required=True
    )
    parser.add_argument("--prepared-authority", type=Path)
    parser.add_argument("--expected-prepared-authority-sha256")
    parser.add_argument("--expected-prepared-authority-bytes", type=int)
    parser.add_argument(
        "--expected-prepared-authority-receipt-payload-sha256"
    )
    parser.add_argument("--seed", type=int, default=20260801)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    context = subparsers.add_parser("context", allow_abbrev=False)
    _authority_options(context)
    context.add_argument("--nul-context", action="store_true")
    context.add_argument("--prepared-authority-output", type=Path)

    shard = subparsers.add_parser("shard", allow_abbrev=False)
    _authority_options(shard)
    shard.add_argument("--shard-id", type=int, required=True)
    shard.add_argument("--device", default="cuda:0")
    shard.add_argument("--progress-every", type=int, default=25)

    finalize = subparsers.add_parser("finalize", allow_abbrev=False)
    _authority_options(finalize)

    distribution = subparsers.add_parser("distribution", allow_abbrev=False)
    _authority_options(distribution)
    distribution.add_argument("--validation-gate-json", type=Path, required=True)
    distribution.add_argument(
        "--expected-validation-gate-sha256", required=True
    )
    distribution.add_argument(
        "--expected-validation-gate-bytes", type=int, required=True
    )
    distribution.add_argument(
        "--expected-validation-gate-receipt-payload-sha256", required=True
    )

    seal = subparsers.add_parser("seal", allow_abbrev=False)
    _authority_options(seal)
    seal.add_argument(
        "--distribution-declaration-json", type=Path, required=True
    )
    seal.add_argument(
        "--expected-distribution-declaration-sha256", required=True
    )
    seal.add_argument("--metric-report-json", type=Path, required=True)
    seal.add_argument("--expected-metric-report-sha256", required=True)

    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be non-negative")
    prepared_values = (
        args.prepared_authority,
        args.expected_prepared_authority_sha256,
        args.expected_prepared_authority_bytes,
        args.expected_prepared_authority_receipt_payload_sha256,
    )
    if any(value is not None for value in prepared_values) and any(
        value is None for value in prepared_values
    ):
        parser.error("prepared authority requires path/SHA/bytes/payload SHA")
    if args.command == "context" and args.prepared_authority is not None:
        parser.error("context must perform a fresh full authority replay")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "context":
        run_context(args)
    elif args.command == "shard":
        run_shard(args)
    elif args.command == "finalize":
        run_finalize(args)
    elif args.command == "distribution":
        run_distribution(args)
    elif args.command == "seal":
        run_seal(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
