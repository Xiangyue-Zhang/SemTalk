#!/usr/bin/env python3
"""Run and seal one measured fresh-Base validation concurrency probe.

This is the only formal producer for a probe run.  Its public command accepts
only frozen input authorities and an unused output directory.  It constructs
the guarded-runner and quality-producer argv itself, observes the process and
GPU runtime, then derives candidate metrics from trainer-native raw feature
and trajectory artifacts.  No public argument accepts elapsed time, memory,
return codes, OOM/cleanup/guard assertions, or metric scalars.

The quality producer is a source-bound executable with a fixed interface.  It
must create ``candidate-ready.json`` below the requested workload directory.
That receipt contains four byte-pinned trainer-native candidate receipts; it
does not contain publishable metric scalars.  This producer reopens the raw
NumPy feature matrices and canonical trajectory JSONL before it can publish a
probe-run receipt.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_fresh_val_orchestrator as orchestrator
from scripts.show_base import evaluate_talkshow_show_metrics as metrics
from scripts.show_base import published_test_winner_claim as authority


EXECUTION_SPEC_FORMAT = "semtalk_show_base_probe_execution_spec_v2"
QUALITY_INPUT_FORMAT = "semtalk_show_base_probe_quality_input_v2"
CANDIDATE_READY_FORMAT = "semtalk_show_base_probe_candidate_ready_v2"
TRAINER_NATIVE_FORMAT = "semtalk_show_base_probe_trainer_native_candidate_v2"
WORKLOAD_EXECUTION_FORMAT = "semtalk_show_base_probe_workload_execution_v1"
PRODUCER_EVIDENCE_FORMAT = "semtalk_show_base_probe_producer_evidence_v1"
MEMORY_TELEMETRY_FORMAT = "semtalk_show_base_probe_memory_telemetry_v1"
EXPECTED_RUNNER = Path("/tmp/globaldiff_guarded_runner.py")
EXPECTED_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
EXPECTED_GPUS = tuple(range(8))
EXPECTED_GPU_ARGUMENT = ",".join(str(index) for index in EXPECTED_GPUS)
EXPECTED_ORIGIN = "git@github.com:Xiangyue-Zhang/SemTalk.git"
OOM_MARKERS = (
    b"cuda out of memory",
    b"cublas_status_alloc_failed",
    b"outofmemoryerror",
    b"std::bad_alloc",
)
PR_SET_CHILD_SUBREAPER = 36
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_UNMOUNT = 0x00002000
IN_Q_OVERFLOW = 0x00004000
IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008


class ProbeProducerError(RuntimeError):
    """A formal probe could not be observed and replayed exactly."""


@dataclass
class _PinnedInput:
    path: str
    descriptor: int | None

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


@dataclass
class _RunRootIdentity:
    """Keep the create-new transaction attached to its mkdirat inode."""

    path: Path
    parent_fd: int
    root_fd: int
    expected_dev: int
    expected_ino: int
    inotify_fd: int | None
    parent_watch: int | None
    root_watch: int | None

    @classmethod
    def acquire(
        cls, path: Path, *, expected_dev: int, expected_ino: int
    ) -> "_RunRootIdentity":
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_fd = os.open(path.parent, flags)
        try:
            root_fd = os.open(path.name, flags, dir_fd=parent_fd)
        except BaseException:
            os.close(parent_fd)
            raise
        inotify_fd: int | None = None
        parent_watch: int | None = None
        root_watch: int | None = None
        try:
            opened = os.fstat(root_fd)
            if (
                opened.st_dev != expected_dev
                or opened.st_ino != expected_ino
                or not stat.S_ISDIR(opened.st_mode)
            ):
                raise ProbeProducerError(
                    "create-new probe root no longer names its mkdirat inode"
                )
            if sys.platform == "linux":
                libc = ctypes.CDLL(None, use_errno=True)
                init = libc.inotify_init1
                init.argtypes = [ctypes.c_int]
                init.restype = ctypes.c_int
                add = libc.inotify_add_watch
                add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
                add.restype = ctypes.c_int
                inotify_fd = init(
                    os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
                )
                if inotify_fd < 0:
                    raise ProbeProducerError(
                        f"cannot monitor probe root: errno={ctypes.get_errno()}"
                    )
                parent_watch = add(
                    inotify_fd,
                    os.fsencode(path.parent),
                    IN_MOVED_FROM | IN_MOVED_TO | IN_CREATE | IN_DELETE,
                )
                root_watch = add(
                    inotify_fd,
                    os.fsencode(path),
                    IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT,
                )
                if parent_watch < 0 or root_watch < 0:
                    raise ProbeProducerError(
                        f"cannot watch probe-root inode: errno={ctypes.get_errno()}"
                    )
            result = cls(
                path=path,
                parent_fd=parent_fd,
                root_fd=root_fd,
                expected_dev=expected_dev,
                expected_ino=expected_ino,
                inotify_fd=inotify_fd,
                parent_watch=parent_watch,
                root_watch=root_watch,
            )
            result.verify()
            return result
        except BaseException:
            if inotify_fd is not None:
                os.close(inotify_fd)
            os.close(root_fd)
            os.close(parent_fd)
            raise

    def _reject_namespace_events(self) -> None:
        if self.inotify_fd is None:
            return
        while True:
            try:
                payload = os.read(self.inotify_fd, 64 * 1024)
            except BlockingIOError:
                return
            if not payload:
                return
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < 16:
                    raise ProbeProducerError("truncated probe-root inotify event")
                watch, mask, _cookie, name_length = struct.unpack_from(
                    "iIII", payload, offset
                )
                offset += 16
                end = offset + name_length
                if end > len(payload):
                    raise ProbeProducerError("invalid probe-root inotify event")
                name = payload[offset:end].split(b"\0", 1)[0]
                offset = end
                if mask & IN_Q_OVERFLOW:
                    raise ProbeProducerError("probe-root event queue overflowed")
                if (
                    watch == self.root_watch
                    or (
                        watch == self.parent_watch
                        and name == os.fsencode(self.path.name)
                    )
                ):
                    raise ProbeProducerError(
                        "probe-root pathname/inode changed during transaction"
                    )

    def verify(self) -> None:
        self._reject_namespace_events()
        opened = os.fstat(self.root_fd)
        linked = os.stat(
            self.path.name,
            dir_fd=self.parent_fd,
            follow_symlinks=False,
        )
        try:
            resolved = self.path.resolve(strict=True)
        except OSError as error:
            raise ProbeProducerError("probe-root pathname disappeared") from error
        for identity in (opened, linked):
            if (
                identity.st_dev != self.expected_dev
                or identity.st_ino != self.expected_ino
                or not stat.S_ISDIR(identity.st_mode)
            ):
                raise ProbeProducerError("probe-root inode binding changed")
        if resolved != self.path:
            raise ProbeProducerError("probe-root canonical pathname changed")

    def close(self) -> None:
        if self.root_fd < 0 and self.parent_fd < 0 and self.inotify_fd is None:
            return
        verify_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            self.verify()
        except BaseException as error:
            verify_error = error
        if self.inotify_fd is not None:
            descriptor = self.inotify_fd
            self.inotify_fd = None
            try:
                os.close(descriptor)
            except BaseException as error:
                close_error = error
        for attribute in ("root_fd", "parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            setattr(self, attribute, -1)
            try:
                os.close(descriptor)
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if verify_error is not None:
            raise verify_error
        if close_error is not None:
            raise close_error


@dataclass
class _InputClosureIdentity:
    """Hold every executable/input inode and detect transient mutation."""

    rows: list[tuple[dict[str, Any], int, os.stat_result]]
    inotify_fd: int | None
    watched: dict[int, str]

    @classmethod
    def acquire(
        cls, artifacts: Sequence[Mapping[str, Any]]
    ) -> "_InputClosureIdentity":
        unique: dict[str, dict[str, Any]] = {}
        for raw in artifacts:
            plain = {
                key: raw[key] for key in ("path", "sha256", "bytes")
            }
            normalized, _payload = _plain_artifact(
                plain, "formal probe input closure artifact"
            )
            prior = unique.get(normalized["path"])
            if prior is not None and prior != normalized:
                raise ProbeProducerError(
                    "probe input closure gives one path conflicting identities"
                )
            unique[normalized["path"]] = normalized
        inotify_fd: int | None = None
        watched: dict[int, str] = {}
        if sys.platform == "linux":
            libc = ctypes.CDLL(None, use_errno=True)
            init = libc.inotify_init1
            init.argtypes = [ctypes.c_int]
            init.restype = ctypes.c_int
            inotify_fd = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
            if inotify_fd < 0:
                raise ProbeProducerError(
                    f"cannot monitor probe inputs: errno={ctypes.get_errno()}"
                )
            add = libc.inotify_add_watch
            add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            add.restype = ctypes.c_int
        else:
            add = None
        rows: list[tuple[dict[str, Any], int, os.stat_result]] = []
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        try:
            for artifact in unique.values():
                descriptor = os.open(artifact["path"], flags)
                identity = os.fstat(descriptor)
                if not stat.S_ISREG(identity.st_mode):
                    os.close(descriptor)
                    raise ProbeProducerError(
                        "probe input closure contains a non-regular file"
                    )
                rows.append((artifact, descriptor, identity))
                if inotify_fd is not None and add is not None:
                    watch = add(
                        inotify_fd,
                        os.fsencode(artifact["path"]),
                        IN_MODIFY
                        | IN_ATTRIB
                        | IN_CLOSE_WRITE
                        | IN_DELETE_SELF
                        | IN_MOVE_SELF
                        | IN_UNMOUNT,
                    )
                    if watch < 0:
                        raise ProbeProducerError(
                            "cannot watch one probe input closure inode"
                        )
                    watched[watch] = artifact["path"]
            result = cls(rows=rows, inotify_fd=inotify_fd, watched=watched)
            result.verify()
            return result
        except BaseException:
            for _artifact, descriptor, _identity in rows:
                os.close(descriptor)
            if inotify_fd is not None:
                os.close(inotify_fd)
            raise

    def _reject_events(self) -> None:
        if self.inotify_fd is None:
            return
        while True:
            try:
                payload = os.read(self.inotify_fd, 64 * 1024)
            except BlockingIOError:
                return
            if not payload:
                return
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < 16:
                    raise ProbeProducerError("truncated probe-input event")
                watch, mask, _cookie, name_length = struct.unpack_from(
                    "iIII", payload, offset
                )
                offset += 16 + name_length
                if offset > len(payload):
                    raise ProbeProducerError("invalid probe-input event")
                if mask & IN_Q_OVERFLOW or watch in self.watched:
                    raise ProbeProducerError(
                        "formal probe input changed during execution"
                    )

    def verify(self) -> None:
        self._reject_events()
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        for artifact, descriptor, expected in self.rows:
            opened = os.fstat(descriptor)
            public = os.stat(artifact["path"], follow_symlinks=False)
            if any(
                getattr(expected, field) != getattr(opened, field)
                or getattr(opened, field) != getattr(public, field)
                for field in stable
            ):
                raise ProbeProducerError(
                    "formal probe input inode/path binding changed"
                )

    def close(self) -> None:
        if not self.rows and self.inotify_fd is None:
            return
        verify_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            self.verify()
        except BaseException as error:
            verify_error = error
        if self.inotify_fd is not None:
            descriptor = self.inotify_fd
            self.inotify_fd = None
            try:
                os.close(descriptor)
            except BaseException as error:
                close_error = error
        rows = self.rows
        self.rows = []
        for _artifact, descriptor, _identity in rows:
            try:
                os.close(descriptor)
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if verify_error is not None:
            raise verify_error
        if close_error is not None:
            raise close_error


def _execution_input_closure(
    spec_artifact: Mapping[str, Any],
    spec: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    _quality_artifact, quality = _payload_artifact(
        spec["quality_input"], "probe quality input closure"
    )
    _pipeline_artifact, pipeline = _payload_artifact(
        binding["pipeline"], "probe pipeline closure"
    )
    _val_artifact, val_inputs = _payload_artifact(
        binding["val_inputs"], "probe validation-input closure"
    )
    source_closure = pipeline.get("source_closure")
    if not isinstance(source_closure, dict):
        raise ProbeProducerError("probe pipeline source closure is absent")
    metric_assets = quality["metric_assets"]
    talkshow_source = metric_assets["talkshow_source"]
    talkshow_root = Path(metric_assets["talkshow_metric_root"])
    talkshow_files = talkshow_source.get("files")
    if not isinstance(talkshow_files, dict):
        raise ProbeProducerError("probe TalkSHOW source closure is absent")
    talkshow_artifacts: list[dict[str, Any]] = []
    for relative, receipt in sorted(talkshow_files.items()):
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(receipt, dict)
            or type(receipt.get("bytes")) is not int
        ):
            raise ProbeProducerError("probe TalkSHOW source closure changed")
        talkshow_artifacts.append(
            {
                "path": str(talkshow_root / relative),
                "sha256": receipt.get("sha256"),
                "bytes": receipt["bytes"],
            }
        )
    marker_path, marker_payload, _marker_stat = _snapshot(
        talkshow_root / ".paspa_talkshow_patch.json",
        "probe TalkSHOW patch marker closure",
    )
    talkshow_artifacts.append(
        {
            "path": str(marker_path),
            "sha256": _sha256(marker_payload),
            "bytes": len(marker_payload),
        }
    )

    def child_artifact(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ProbeProducerError(f"{label} is not an artifact")
        path, payload, _metadata = _snapshot(value.get("path"), label)
        observed = _sha256(payload)
        if observed != _require_sha(value.get("sha256"), f"{label} SHA-256"):
            raise ProbeProducerError(f"{label} artifact changed")
        return {"path": str(path), "sha256": observed, "bytes": len(payload)}

    val_children: list[dict[str, Any]] = []
    for role in ("canonical_manifest", "canonical_summary", "canonical_lineage"):
        val_children.append(child_artifact(val_inputs[role], f"probe val {role}"))
    for role in ("audio_manifests", "audio_summaries", "audio_lineages"):
        rows = val_inputs.get(role)
        if not isinstance(rows, list) or len(rows) != len(EXPECTED_GPUS):
            raise ProbeProducerError(f"probe val {role} coverage changed")
        val_children.extend(
            child_artifact(value, f"probe val {role} {index}")
            for index, value in enumerate(rows)
        )
    canonical_manifest, canonical_manifest_payload = _plain_artifact(
        val_inputs["canonical_manifest"],
        "probe val canonical manifest closure",
    )
    if canonical_manifest not in val_children:
        raise ProbeProducerError("probe val canonical manifest closure changed")
    canonical_rows = authority._strict_jsonl_bytes(
        canonical_manifest_payload,
        "probe val canonical manifest closure",
    )
    if len(canonical_rows) != orchestrator.EXPECTED_CLIPS:
        raise ProbeProducerError("probe val canonical closure changed")
    for index, row in enumerate(canonical_rows):
        val_children.append(
            child_artifact(
                {
                    "path": row.get("canonical_npz"),
                    "sha256": row.get("canonical_npz_sha256"),
                },
                f"probe val canonical NPZ {index}",
            )
        )
    for manifest_value in val_inputs["audio_manifests"]:
        manifest, manifest_payload = _plain_artifact(
            manifest_value, "probe val audio manifest closure"
        )
        if manifest not in val_children:
            raise ProbeProducerError("probe val audio manifest closure changed")
        audio_rows = authority._strict_jsonl_bytes(
            manifest_payload,
            "probe val audio manifest closure",
        )
        for index, row in enumerate(audio_rows):
            val_children.append(
                child_artifact(
                    {
                        "path": row.get("audio_feature_npz"),
                        "sha256": row.get("audio_feature_npz_sha256"),
                    },
                    f"probe val audio feature {index}",
                )
            )
    fixed_checkpoint_artifacts = [
        {
            "path": pipeline["fixed_checkpoints"][stage]["path"],
            "sha256": pipeline["fixed_checkpoints"][stage]["sha256"],
            "bytes": pipeline["fixed_checkpoints"][stage]["bytes"],
        }
        for stage in authority.STAGES
    ]
    candidate_manifest_artifact, candidate_manifest_payload = _plain_artifact(
        quality["candidate_bundle"]["manifest"],
        "probe candidate manifest closure",
    )
    candidate_manifest = _strict_json_document(
        candidate_manifest_payload, "probe candidate manifest closure"
    )
    candidate_entries = (
        candidate_manifest.get("entries")
        if isinstance(candidate_manifest, dict)
        else None
    )
    if not isinstance(candidate_entries, list) or not candidate_entries:
        raise ProbeProducerError("probe candidate checkpoint closure changed")
    candidate_checkpoint_artifacts: list[dict[str, Any]] = []
    candidate_root = Path(candidate_manifest_artifact["path"]).parent
    for index, entry in enumerate(candidate_entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("checkpoint"), str):
            raise ProbeProducerError("probe candidate checkpoint entry changed")
        relative = Path(entry["checkpoint"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ProbeProducerError("probe candidate checkpoint escaped run root")
        candidate_checkpoint_artifacts.append(
            child_artifact(
                {
                    "path": str(candidate_root / relative),
                    "sha256": entry.get("checkpoint_sha256"),
                },
                f"probe candidate checkpoint {index}",
            )
        )
    artifacts: list[Mapping[str, Any]] = [
        spec_artifact,
        spec["quality_input"],
        binding["pipeline"],
        binding["prerequisite_selection"],
        binding["val_inputs"],
        binding["subset_manifest"],
        spec["python"],
        spec["quality_producer"],
        spec["runner"],
        spec["nvidia_smi"],
        *(row["candidate_checkpoint"] for row in binding["candidate_checkpoints"]),
        *quality["candidate_bundle"].values(),
        *quality["representation_lmdb"].values(),
        *fixed_checkpoint_artifacts,
        *candidate_checkpoint_artifacts,
        *val_children,
        quality["training_metrics"],
        metric_assets["feature_extractor"],
        metric_assets["smplx_asset"],
        *talkshow_artifacts,
        *source_closure.values(),
    ]
    return artifacts


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha(value: Any, label: str, *, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ) is None:
        raise ProbeProducerError(f"{label} must be {length} lowercase hex")
    return value


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProbeProducerError(f"{label} schema changed")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise ProbeProducerError("non-finite/non-canonical JSON value") from error


def _strict_json(payload: bytes, label: str) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProbeProducerError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProbeProducerError(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeProducerError(f"{label} is not strict JSON") from error
    if _canonical_bytes(value) != payload:
        raise ProbeProducerError(f"{label} is not canonical JSON")
    return value


def _strict_json_document(payload: bytes, label: str) -> Any:
    """Parse external JSON without requiring this repository's layout."""

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProbeProducerError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProbeProducerError(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeProducerError(f"{label} is not strict JSON") from error


def _snapshot(path_value: Any, label: str) -> tuple[Path, bytes, os.stat_result]:
    path = Path(path_value)
    if not path.is_absolute():
        raise ProbeProducerError(f"{label} path must be absolute")
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ProbeProducerError(f"{label} is unavailable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProbeProducerError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        public_after = os.lstat(path)
        resolved_after = path.resolve(strict=True)
    except OSError as error:
        raise ProbeProducerError(
            f"{label} path disappeared while reading"
        ) from error
    stable = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(after, field)
        or getattr(after, field) != getattr(public_after, field)
        for field in stable
    ) or resolved_after != path:
        raise ProbeProducerError(f"{label} identity changed while reading")
    return resolved_after, payload, after


def _plain_artifact(value: Any, label: str) -> tuple[dict[str, Any], bytes]:
    row = _exact(value, {"path", "sha256", "bytes"}, label)
    path, payload, _metadata = _snapshot(row["path"], label)
    observed = _sha256(payload)
    if (
        str(path) != row["path"]
        or _require_sha(row["sha256"], f"{label} SHA-256") != observed
        or type(row["bytes"]) is not int
        or row["bytes"] != len(payload)
    ):
        raise ProbeProducerError(f"{label} artifact changed")
    return dict(row), payload


def _payload_artifact(value: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _exact(
        value,
        {"path", "sha256", "bytes", "receipt_payload_sha256"},
        label,
    )
    plain, payload = _plain_artifact(
        {key: row[key] for key in ("path", "sha256", "bytes")}, label
    )
    decoded = _strict_json(payload, label)
    if not isinstance(decoded, dict):
        raise ProbeProducerError(f"{label} must contain one JSON object")
    expected_payload = _require_sha(
        row["receipt_payload_sha256"], f"{label} payload SHA-256"
    )
    if decoded.get("receipt_payload_sha256") != expected_payload:
        raise ProbeProducerError(f"{label} payload binding changed")
    body = dict(decoded)
    body.pop("receipt_payload_sha256", None)
    if authority.canonical_json_sha256(body) != expected_payload:
        raise ProbeProducerError(f"{label} payload SHA-256 does not replay")
    return {**plain, "receipt_payload_sha256": expected_payload}, decoded


def _validate_representation_lmdb_authority(
    value: Mapping[str, Any],
    *,
    selected_authority: Mapping[str, Any],
    selected_dataset: Any,
) -> dict[str, dict[str, Any]]:
    """Bind probe LMDB bytes to the candidate trainer's frozen dataset."""

    if not isinstance(selected_dataset, dict):
        raise ProbeProducerError("candidate bundle lacks audited LMDB provenance")
    expected_dataset_keys = {
        "format",
        "lmdb",
        "summary",
        "summary_sha256",
        "lineage",
        "lineage_sha256",
        "entries",
        "train_clips",
        "split",
        "test_visible",
        "data_mdb_sha256",
        "lock_mdb_sha256",
        "prerequisite_selection",
        "selected_prerequisite_sha256",
        "lmdb_binding_scope",
        "node_lmdb_inode_bindings",
    }
    if set(selected_dataset) != expected_dataset_keys:
        raise ProbeProducerError("candidate bundle LMDB provenance schema changed")
    normalized: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for role in ("summary", "data", "lock"):
        normalized[role], payloads[role] = _plain_artifact(
            value[role], f"probe LMDB {role}"
        )
    summary = _strict_json_document(payloads["summary"], "probe LMDB summary")
    lmdb_root = Path(selected_dataset["lmdb"])
    selected_receipt = selected_dataset["prerequisite_selection"]
    compact_keys = {"path", "sha256", "receipt_payload_sha256"}
    if (
        not isinstance(summary, dict)
        or summary.get("format") != "semtalk_show_base_lmdb_summary_v1"
        or summary.get("status") != "complete"
        or summary.get("scope") != "SemTalk Base only"
        or summary.get("entries") != selected_dataset["entries"]
        or summary.get("train_clips") != selected_dataset["train_clips"]
        or summary.get("lmdb") != selected_dataset["lmdb"]
        or summary.get("data_mdb_sha256")
        != selected_dataset["data_mdb_sha256"]
        or summary.get("lock_mdb_sha256")
        != selected_dataset["lock_mdb_sha256"]
        or summary.get("lineage_json") != selected_dataset["lineage"]
        or summary.get("lineage_json_sha256")
        != selected_dataset["lineage_sha256"]
        or normalized["summary"]["path"] != selected_dataset["summary"]
        or normalized["summary"]["sha256"]
        != selected_dataset["summary_sha256"]
        or normalized["data"]["path"] != str(lmdb_root / "data.mdb")
        or normalized["data"]["sha256"]
        != selected_dataset["data_mdb_sha256"]
        or normalized["lock"]["path"] != str(lmdb_root / "lock.mdb")
        or normalized["lock"]["sha256"]
        != selected_dataset["lock_mdb_sha256"]
        or not isinstance(selected_receipt, dict)
        or set(selected_receipt) != compact_keys
        or any(
            selected_receipt.get(key) != selected_authority.get(key)
            for key in compact_keys
        )
    ):
        raise ProbeProducerError(
            "probe LMDB/selected-five authority differs from candidate training"
        )
    return normalized


def _pin_input_artifact(
    artifact: Mapping[str, Any],
    *,
    label: str,
    fallback_path: Path,
) -> _PinnedInput:
    """Give the workload a read-only byte snapshot, never its public input path."""

    _plain, payload = _plain_artifact(
        {key: artifact[key] for key in ("path", "sha256", "bytes")}, label
    )
    if sys.platform == "linux" and hasattr(os, "memfd_create"):
        flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(
            os, "MFD_ALLOW_SEALING", 0
        )
        descriptor = os.memfd_create(f"semtalk-{label}", flags=flags)
        try:
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise ProbeProducerError("short write to sealed input memfd")
                offset += written
            os.lseek(descriptor, 0, os.SEEK_SET)
            seals = (
                getattr(fcntl, "F_SEAL_SEAL", 0)
                | getattr(fcntl, "F_SEAL_SHRINK", 0)
                | getattr(fcntl, "F_SEAL_GROW", 0)
                | getattr(fcntl, "F_SEAL_WRITE", 0)
            )
            add_seals = getattr(fcntl, "F_ADD_SEALS", None)
            get_seals = getattr(fcntl, "F_GET_SEALS", None)
            if not seals or add_seals is None or get_seals is None:
                raise ProbeProducerError("Linux sealed memfd support is required")
            fcntl.fcntl(descriptor, add_seals, seals)
            if fcntl.fcntl(descriptor, get_seals) != seals:
                raise ProbeProducerError("input memfd did not acquire exact seals")
            proc_path = f"/proc/{os.getpid()}/fd/{descriptor}"
            if not Path(proc_path).exists():
                raise ProbeProducerError("sealed input memfd is not proc-visible")
            return _PinnedInput(path=proc_path, descriptor=descriptor)
        except BaseException:
            os.close(descriptor)
            raise
    # CPU-only tests run on macOS.  Formal execution is Linux-only; this
    # fallback remains create-new and is revalidated after the fake backend.
    artifact_copy = _write_bytes_new(fallback_path, payload)
    if artifact_copy["sha256"] != artifact["sha256"]:
        raise ProbeProducerError("CPU fixture input snapshot changed")
    return _PinnedInput(path=artifact_copy["path"], descriptor=None)


def _write_bytes_new(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise ProbeProducerError("short write to CPU input snapshot")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    resolved, observed, _metadata = _snapshot(path, "create-new byte artifact")
    return {
        "path": str(resolved),
        "sha256": _sha256(observed),
        "bytes": len(observed),
    }


def _source_state(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    expected_row = _exact(
        expected,
        {"origin", "commit", "tree", "clean", "detached", "local_branches_at_commit"},
        "probe source",
    )

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def git_optional(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in {0, 1}:
            raise ProbeProducerError(
                f"cannot inspect formal probe source: {' '.join(arguments)}"
            )
        return completed.stdout.strip()

    actual = {
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": not bool(git("status", "--porcelain=v1", "--untracked-files=all")),
        "detached": not bool(
            git_optional("symbolic-ref", "-q", "--short", "HEAD")
        ),
        "local_branches_at_commit": [
            line
            for line in git("for-each-ref", "--format=%(refname)", "refs/heads").splitlines()
            if line
        ],
    }
    if (
        actual != expected_row
        or actual["origin"] != EXPECTED_ORIGIN
        or re.fullmatch(r"[0-9a-f]{40}", actual["commit"]) is None
        or re.fullmatch(r"[0-9a-f]{40}", actual["tree"]) is None
        or actual["clean"] is not True
        or actual["detached"] is not True
        or actual["local_branches_at_commit"] != []
    ):
        raise ProbeProducerError("formal probe source state changed")
    return actual


def _validate_quality_input(
    artifact_value: Any,
    *,
    binding: Mapping[str, Any],
    mode: int,
    execution_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, value = _payload_artifact(artifact_value, "probe quality input")
    _exact(
        value,
        {
            "format",
            "status",
            "dataset",
            "target_speaker_scope",
            "split",
            "test_visible",
            "source",
            "pipeline",
            "selected_five_authority",
            "candidate_bundle",
            "val_inputs",
            "representation_lmdb",
            "training_metrics",
            "metric_assets",
            "topology",
            "receipt_payload_sha256",
        },
        "probe quality input",
    )
    topology = _exact(
        value["topology"],
        {
            "world_size",
            "physical_gpus",
            "candidates_per_wave",
            "execution_mode",
            "seed",
            "clips_per_candidate",
            "shards_per_candidate",
        },
        "probe quality topology",
    )
    lmdb = _exact(
        value["representation_lmdb"], {"summary", "data", "lock"}, "probe LMDB"
    )
    training_metrics, training_metrics_payload = _plain_artifact(
        value["training_metrics"], "probe trainer epoch metrics"
    )
    training_metric_rows = authority._strict_jsonl_bytes(
        training_metrics_payload, "probe trainer epoch metrics"
    )
    if len(training_metric_rows) != 400:
        raise ProbeProducerError("probe trainer epoch metrics coverage changed")
    for expected_epoch, row in enumerate(training_metric_rows, start=1):
        if (
            not isinstance(row, dict)
            or row.get("format")
            != "semtalk_show_base_long_epoch_metric_v1"
            or row.get("epoch") != expected_epoch
            or type(row.get("optimizer_updates")) is not int
            or type(row.get("updates_per_epoch")) is not int
            or row["optimizer_updates"]
            != expected_epoch * row["updates_per_epoch"]
            or not isinstance(row.get("metrics"), dict)
            or type(row["metrics"].get("total")) not in (int, float)
            or type(row["metrics"].get("total")) is bool
            or not math.isfinite(float(row["metrics"]["total"]))
            or row.get("all_finite") is not True
        ):
            raise ProbeProducerError(
                f"probe trainer epoch metric e{expected_epoch} changed"
            )
    metric_assets = _exact(
        value["metric_assets"],
        {
            "talkshow_metric_root",
            "talkshow_source",
            "feature_extractor",
            "smplx_asset",
        },
        "probe TalkSHOW metric assets",
    )
    feature_extractor, _feature_payload = _plain_artifact(
        metric_assets["feature_extractor"],
        "probe TalkSHOW feature extractor",
    )
    smplx_asset, _smplx_payload = _plain_artifact(
        metric_assets["smplx_asset"], "probe SMPL-X asset"
    )
    try:
        talkshow_source = metrics.validate_talkshow_metric_root(
            metric_assets["talkshow_metric_root"]
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ProbeProducerError(
            "probe TalkSHOW metric source failed exact replay"
        ) from error
    if (
        metric_assets["talkshow_source"] != talkshow_source
        or feature_extractor["sha256"] != metrics.FEATURE_EXTRACTOR_SHA256
        or smplx_asset["sha256"] != metrics.SMPLX_SHA256
    ):
        raise ProbeProducerError("probe TalkSHOW metric assets changed")
    selected, selected_payload = _payload_artifact(
        value["selected_five_authority"], "selected-five authority"
    )
    candidate_bundle = _exact(
        value["candidate_bundle"],
        {"manifest", "status", "frozen_inputs"},
        "probe candidate bundle",
    )
    normalized_bundle: dict[str, dict[str, Any]] = {}
    for role, raw in candidate_bundle.items():
        normalized_bundle[role], _payload = _plain_artifact(
            raw, f"probe candidate bundle {role}"
        )
    _status_artifact, status_payload = _plain_artifact(
        normalized_bundle["status"], "probe candidate training status"
    )
    status_value = _strict_json_document(
        status_payload, "probe candidate training status"
    )
    if (
        not isinstance(status_value, dict)
        or status_value.get("status") != "complete"
        or status_value.get("epoch_metrics_jsonl")
        != training_metrics["path"]
        or status_value.get("epoch_metrics_sha256")
        != training_metrics["sha256"]
        or status_value.get("epoch_metrics_records") != 400
    ):
        raise ProbeProducerError(
            "probe trainer metrics are not owned by candidate status"
        )
    _pipeline_artifact, pipeline_payload = _payload_artifact(
        value["pipeline"], "probe pipeline authority"
    )
    fixed = pipeline_payload.get("fixed_checkpoints")
    if not isinstance(fixed, dict) or set(fixed) != set(authority.STAGES):
        raise ProbeProducerError("probe pipeline lacks exact selected-five binding")
    try:
        audited_bundle = orchestrator.val_contract.validate_candidate_bundle(
            manifest_path=Path(normalized_bundle["manifest"]["path"]),
            expected_manifest_sha256=normalized_bundle["manifest"]["sha256"],
            status_path=Path(normalized_bundle["status"]["path"]),
            expected_status_sha256=normalized_bundle["status"]["sha256"],
            frozen_inputs_path=Path(
                normalized_bundle["frozen_inputs"]["path"]
            ),
            expected_frozen_inputs_sha256=normalized_bundle[
                "frozen_inputs"
            ]["sha256"],
            expected_selected_prerequisite_sha256={
                stage: fixed[stage]["sha256"] for stage in authority.STAGES
            },
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ProbeProducerError(
            "probe candidate bundle failed exact training replay"
        ) from error
    normalized_lmdb = _validate_representation_lmdb_authority(
        lmdb,
        selected_authority=selected,
        selected_dataset=audited_bundle.get("selected_dataset"),
    )
    expected_probe_candidates = {
        row["epoch"]: row["candidate_checkpoint"]
        for row in binding["candidate_checkpoints"]
    }
    if (
        value["format"] != QUALITY_INPUT_FORMAT
        or value["status"] != "frozen"
        or value["dataset"] != "SHOW"
        or value["target_speaker_scope"] != authority.EXPECTED_SCOPE
        or value["split"] != "val"
        or value["test_visible"] is not False
        or value["source"] != binding["source"]
        or value["pipeline"] != binding["pipeline"]
        or value["selected_five_authority"] != binding["prerequisite_selection"]
        or normalized_bundle != candidate_bundle
        or normalized_lmdb != lmdb
        or value["val_inputs"] != binding["val_inputs"]
        or any(
            audited_bundle["candidates"].get(epoch) != checkpoint
            for epoch, checkpoint in expected_probe_candidates.items()
        )
        or not isinstance(selected_payload, dict)
        or topology
        != {
            "world_size": 8,
            "physical_gpus": list(EXPECTED_GPUS),
            "candidates_per_wave": mode,
            "execution_mode": execution_mode,
            "seed": binding["seed"],
            "clips_per_candidate": binding["clips_per_candidate"],
            "shards_per_candidate": binding["shards_per_candidate"],
        }
    ):
        raise ProbeProducerError("probe quality input authority/topology changed")
    return artifact, {
        **value,
        "representation_lmdb": normalized_lmdb,
        "_audited_candidate_bundle": audited_bundle,
    }


def validate_execution_spec(
    artifact_value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact, value = _payload_artifact(artifact_value, "probe execution spec")
    _exact(
        value,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "formal_host",
            "candidates_per_wave",
            "execution_mode",
            "probe_binding",
            "quality_input",
            "source_root",
            "python",
            "quality_producer",
            "runner",
            "nvidia_smi",
            "receipt_payload_sha256",
        },
        "probe execution spec",
    )
    binding = orchestrator._probe_binding(value["probe_binding"])
    mode = value["candidates_per_wave"]
    execution_mode = value["execution_mode"]
    if (
        value["format"] != EXECUTION_SPEC_FORMAT
        or value["status"] != "frozen"
        or value["split"] != "val"
        or value["test_visible"] is not False
        or value["formal_host"] not in orchestrator.FORMAL_HOST_BY_PARTITION.values()
        or value["formal_host"] != socket.gethostname()
        or mode not in orchestrator.CONCURRENCY_SELECTION_ORDER
        or execution_mode not in {"serial", "concurrent"}
        or (execution_mode == "serial" and mode != 1)
    ):
        raise ProbeProducerError("probe execution identity changed")
    root = Path(value["source_root"])
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise ProbeProducerError("probe source root must be canonical")
    _source_state(root, binding["source"])
    normalized_executables = {}
    for label in ("python", "quality_producer", "runner", "nvidia_smi"):
        normalized, _payload = _plain_artifact(value[label], f"probe {label}")
        executable = Path(normalized["path"])
        if not os.access(executable, os.X_OK):
            raise ProbeProducerError(f"probe {label} is not executable")
        normalized_executables[label] = executable
    try:
        expected_runner = EXPECTED_RUNNER.resolve(strict=True)
        expected_nvidia_smi = EXPECTED_NVIDIA_SMI.resolve(strict=True)
    except OSError as error:
        raise ProbeProducerError(
            "formal runner or nvidia-smi is unavailable"
        ) from error
    if (
        normalized_executables["python"]
        != Path(sys.executable).resolve(strict=True)
        or normalized_executables["runner"] != expected_runner
        or normalized_executables["nvidia_smi"] != expected_nvidia_smi
    ):
        raise ProbeProducerError("formal probe runtime executable changed")
    producer = Path(value["quality_producer"]["path"])
    try:
        producer.relative_to(root)
    except ValueError as error:
        raise ProbeProducerError("quality producer escaped source checkout") from error
    relative = producer.relative_to(root).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0 or tracked.stdout.strip() != relative:
        raise ProbeProducerError("quality producer is not tracked by exact source")
    _pipeline_artifact, pipeline = _payload_artifact(
        binding["pipeline"], "probe pipeline authority"
    )
    source_closure = pipeline.get("source_closure")
    source_producer = (
        source_closure.get(relative)
        if isinstance(source_closure, dict)
        else None
    )
    if (
        not isinstance(source_producer, dict)
        or any(
            source_producer.get(key) != value["quality_producer"].get(key)
            for key in ("path", "sha256", "bytes")
        )
    ):
        raise ProbeProducerError(
            "quality producer is not byte-bound by the frozen pipeline"
        )
    quality_artifact, _quality = _validate_quality_input(
        value["quality_input"],
        binding=binding,
        mode=mode,
        execution_mode=execution_mode,
    )
    result = dict(value)
    result["quality_input"] = quality_artifact
    return artifact, result, binding


def _proc_identity(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 1:
        raise ProbeProducerError("invalid PID identity")
    proc = Path("/proc") / str(pid)
    try:
        stat_line = (proc / "stat").read_text(encoding="utf-8")
        cmdline = (proc / "cmdline").read_bytes()
    except OSError as error:
        raise ProbeProducerError(f"process {pid} disappeared") from error
    close = stat_line.rfind(")")
    fields = stat_line[close + 2 :].split() if close >= 0 else []
    argv = [
        token.decode("utf-8", "surrogateescape")
        for token in cmdline.split(b"\0")
        if token
    ]
    if len(fields) <= 19 or fields[0] == "Z" or not argv:
        raise ProbeProducerError(f"process {pid} identity is incomplete")
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime_ticks": int(fields[19]),
        "argv": argv,
        "argv_sha256": _sha256(cmdline),
    }


def _children(pid: int) -> list[int]:
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        payload = path.read_text(encoding="ascii").strip()
    except OSError:
        return []
    if not payload:
        return []
    result = []
    for token in payload.split():
        if token.isdigit() and int(token) > 1:
            result.append(int(token))
    return result


def _descendant_identities(root_pid: int) -> list[dict[str, Any]]:
    pending = [root_pid]
    seen: set[int] = set()
    result: list[dict[str, Any]] = []
    while pending:
        parent = pending.pop()
        for child in _children(parent):
            if child in seen:
                continue
            seen.add(child)
            pending.append(child)
            try:
                result.append(_proc_identity(child))
            except ProbeProducerError:
                continue
    return result


def _identity_live(identity: Mapping[str, Any]) -> bool:
    try:
        current = _proc_identity(int(identity["pid"]))
    except ProbeProducerError:
        return False
    return (
        current["starttime_ticks"] == identity["starttime_ticks"]
        and current["argv_sha256"] == identity["argv_sha256"]
    )


def _enable_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise ProbeProducerError(f"cannot enable child subreaper: errno={error}")


def _gpu_memory_sample(nvidia_smi: Path) -> tuple[list[int], list[int]]:
    completed = subprocess.run(
        [
            str(nvidia_smi),
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise ProbeProducerError("nvidia-smi memory row changed")
        index, used_mib, total_mib = map(int, fields)
        rows[index] = (used_mib * 1024 * 1024, total_mib * 1024 * 1024)
    if set(rows) != set(EXPECTED_GPUS):
        raise ProbeProducerError("nvidia-smi did not expose exact GPU0..7")
    return (
        [rows[index][0] for index in EXPECTED_GPUS],
        [rows[index][1] for index in EXPECTED_GPUS],
    )


def _verify_restored_guards(
    status: Mapping[str, Any],
    nvidia_smi: Path,
    *,
    expected_python: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_python, _python_payload = _plain_artifact(
        expected_python, "guard Python executable"
    )
    restored = status.get("restored_guards")
    if (
        not isinstance(restored, dict)
        or set(restored) != {str(index) for index in EXPECTED_GPUS}
        or len(set(restored.values())) != len(EXPECTED_GPUS)
        or any(type(pid) is not int or pid <= 1 for pid in restored.values())
    ):
        raise ProbeProducerError("runner did not restore exact GPU0..7 guards")
    gpu_rows = subprocess.run(
        [
            str(nvidia_smi),
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    uuid_rows = subprocess.run(
        [
            str(nvidia_smi),
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    uuid_to_index: dict[str, int] = {}
    for line in uuid_rows:
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or not fields[0].isdigit():
            raise ProbeProducerError("nvidia-smi GPU UUID row changed")
        uuid_to_index[fields[1]] = int(fields[0])
    observed: dict[int, list[int]] = {index: [] for index in EXPECTED_GPUS}
    for line in gpu_rows:
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or fields[0] not in uuid_to_index or not fields[1].isdigit():
            raise ProbeProducerError("nvidia-smi compute-process row changed")
        index = uuid_to_index[fields[0]]
        if index in observed:
            observed[index].append(int(fields[1]))
    identities: dict[str, dict[str, Any]] = {}
    executables: dict[str, dict[str, Any]] = {}
    for index in EXPECTED_GPUS:
        expected_pid = restored[str(index)]
        if observed[index] != [expected_pid]:
            raise ProbeProducerError(f"GPU{index} lacks exactly one restored guard")
        identity = _proc_identity(expected_pid)
        try:
            executable = (Path("/proc") / str(expected_pid) / "exe").resolve(
                strict=True
            )
        except OSError as error:
            raise ProbeProducerError(
                f"GPU{index} guard executable disappeared"
            ) from error
        argv_text = "\0".join(identity["argv"]).casefold()
        if not (
            str(executable) == normalized_python["path"]
            and identity["ppid"] == os.getpid()
            and "globaldiff_gpu_guard_cnn" in argv_text
            and "torchvision" in argv_text
            and "resnet18" in argv_text
        ):
            raise ProbeProducerError(f"GPU{index} guard is not torchvision ResNet18")
        identities[str(index)] = identity
        executables[str(index)] = normalized_python
    return {
        "restored_guards": dict(restored),
        "guard_identities": identities,
        "guard_executables": executables,
        "nvidia_smi_compute_rows_sha256": authority.canonical_json_sha256(gpu_rows),
        "nvidia_smi_uuid_rows_sha256": authority.canonical_json_sha256(uuid_rows),
    }


def _output_artifact(
    path: Path, label: str, *, allow_empty: bool = False
) -> dict[str, Any]:
    resolved, payload, _metadata = _snapshot(path, label)
    if not allow_empty and not payload:
        raise ProbeProducerError(f"{label} is empty")
    return {"path": str(resolved), "sha256": _sha256(payload), "bytes": len(payload)}


def _cleanup_owned_identities(identities: Sequence[Mapping[str, Any]]) -> None:
    unique = {
        (int(row["pid"]), int(row["starttime_ticks"])): row
        for row in identities
        if isinstance(row, Mapping)
        and type(row.get("pid")) is int
        and type(row.get("starttime_ticks")) is int
    }
    for requested_signal, deadline in (
        (signal.SIGTERM, time.monotonic() + 5.0),
        (signal.SIGKILL, time.monotonic() + 5.0),
    ):
        for identity in unique.values():
            if _identity_live(identity):
                os.kill(int(identity["pid"]), requested_signal)
        while time.monotonic() < deadline:
            if not any(_identity_live(row) for row in unique.values()):
                return
            time.sleep(0.05)
    if any(_identity_live(row) for row in unique.values()):
        raise ProbeProducerError("owned escaped descendants resisted cleanup")


@dataclass(frozen=True)
class ObservedExecution:
    started_monotonic_ns: int
    finished_monotonic_ns: int
    gpu_peak_memory_bytes: list[int]
    gpu_total_memory_bytes: list[int]
    runner_return_code: int
    oom_observed: bool
    descendants_exited: bool
    runner_identity: dict[str, Any]
    observed_descendants: list[dict[str, Any]]
    status_artifact: dict[str, Any]
    log_artifact: dict[str, Any]
    stdout_artifact: dict[str, Any]
    stderr_artifact: dict[str, Any]
    telemetry_artifact: dict[str, Any]
    status: dict[str, Any]
    guard_verification: dict[str, Any]


class FormalExecutionBackend:
    """The non-injectable backend selected by the public CLI."""

    def execute(
        self,
        *,
        runner_argv: Sequence[str],
        workload_argv: Sequence[str],
        root: Path,
        nvidia_smi: Path,
        python_artifact: Mapping[str, Any],
    ) -> ObservedExecution:
        _enable_subreaper()
        status_path = root / "runner-status.json"
        log_path = root / "runner.log"
        stdout_path = root / "runner.stdout"
        stderr_path = root / "runner.stderr"
        telemetry_path = root / "memory-telemetry.json"
        for path in (status_path, log_path, stdout_path, stderr_path, telemetry_path):
            if os.path.lexists(path):
                raise FileExistsError(path)
        stdout_handle = stdout_path.open("xb")
        stderr_handle = stderr_path.open("xb")
        started = time.monotonic_ns()
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                list(runner_argv),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                close_fds=True,
                start_new_session=False,
            )
            runner_identity = _proc_identity(process.pid)
            expected_argv_sha = _sha256(
                b"\0".join(os.fsencode(item) for item in runner_argv) + b"\0"
            )
            if (
                runner_identity["ppid"] != os.getpid()
                or runner_identity["argv_sha256"] != expected_argv_sha
            ):
                raise ProbeProducerError(
                    "spawned runner full argv/parent changed"
                )
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            stdout_handle.close()
            stderr_handle.close()
            raise
        peaks = [0] * len(EXPECTED_GPUS)
        totals: list[int] | None = None
        samples: list[dict[str, Any]] = []
        descendants: dict[tuple[int, int], dict[str, Any]] = {}
        try:
            while process.poll() is None:
                used, observed_totals = _gpu_memory_sample(nvidia_smi)
                if totals is None:
                    totals = observed_totals
                elif totals != observed_totals:
                    raise ProbeProducerError("GPU total memory changed during probe")
                peaks = [max(old, new) for old, new in zip(peaks, used)]
                samples.append(
                    {"monotonic_ns": time.monotonic_ns(), "used_bytes": used}
                )
                for identity in _descendant_identities(process.pid):
                    descendants[(identity["pid"], identity["starttime_ticks"])] = identity
                time.sleep(0.05)
            return_code = process.wait()
        except BaseException:
            if process.poll() is None:
                current = _proc_identity(process.pid)
                if current["starttime_ticks"] == runner_identity["starttime_ticks"]:
                    process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current = _proc_identity(process.pid)
                    if current["starttime_ticks"] == runner_identity["starttime_ticks"]:
                        process.kill()
                    process.wait()
            adopted = []
            for pid in _children(os.getpid()):
                try:
                    adopted.append(_proc_identity(pid))
                except ProbeProducerError:
                    continue
            restored_pids: set[int] = set()
            if status_path.is_file() and not status_path.is_symlink():
                try:
                    _path, status_bytes, _metadata = _snapshot(
                        status_path, "failed guarded-runner status"
                    )
                    failed_status = _strict_json_document(
                        status_bytes, "failed guarded-runner status"
                    )
                    restored = (
                        failed_status.get("restored_guards")
                        if isinstance(failed_status, dict)
                        else None
                    )
                    if isinstance(restored, dict):
                        restored_pids = {
                            pid
                            for pid in restored.values()
                            if type(pid) is int and pid > 1
                        }
                except ProbeProducerError:
                    pass
            _cleanup_owned_identities(
                [
                    row
                    for row in (*descendants.values(), *adopted)
                    if row["pid"] not in restored_pids
                ]
            )
            raise
        finally:
            stdout_handle.flush()
            stderr_handle.flush()
            os.fsync(stdout_handle.fileno())
            os.fsync(stderr_handle.fileno())
            stdout_handle.close()
            stderr_handle.close()
        finished = time.monotonic_ns()
        if totals is None or not samples:
            raise ProbeProducerError("probe produced no GPU memory telemetry")
        escaped_identities = []
        for pid in _children(os.getpid()):
            try:
                escaped_identities.append(_proc_identity(pid))
            except ProbeProducerError:
                continue
        telemetry = orchestrator._with_payload_sha(
            {
                "format": MEMORY_TELEMETRY_FORMAT,
                "status": "complete",
                "gpu_indices": list(EXPECTED_GPUS),
                "samples": samples,
                "peak_memory_bytes": peaks,
                "total_memory_bytes": totals,
            }
        )
        orchestrator._write_new(telemetry_path, telemetry)
        status_artifact = _output_artifact(status_path, "guarded-runner status")
        log_artifact = _output_artifact(log_path, "guarded-runner log")
        stdout_artifact = _output_artifact(
            stdout_path, "guarded-runner stdout", allow_empty=True
        )
        stderr_artifact = _output_artifact(
            stderr_path, "guarded-runner stderr", allow_empty=True
        )
        telemetry_artifact, status_payload = _payload_artifact(
            {
                **_output_artifact(telemetry_path, "memory telemetry"),
                "receipt_payload_sha256": telemetry["receipt_payload_sha256"],
            },
            "memory telemetry",
        )
        del status_payload
        _status_path, raw_status, _metadata = _snapshot(status_path, "guarded-runner status")
        status = _strict_json_document(raw_status, "guarded-runner status")
        if not isinstance(status, dict):
            raise ProbeProducerError("guarded-runner status must be an object")
        expected_status = {
            "state": "finished",
            "return_code": 0,
            "error": None,
            "cleanup_error": None,
            "restore_error": None,
            "received_signal": None,
            "command": list(workload_argv),
            "wrapper_pid": runner_identity["pid"],
        }
        if any(status.get(key) != value for key, value in expected_status.items()):
            raise ProbeProducerError("guarded-runner status/command changed")
        if type(status.get("child_pid")) is not int or status["child_pid"] <= 1:
            raise ProbeProducerError("guarded-runner status lacks exact child PID")
        child_matches = [
            row
            for row in descendants.values()
            if row["pid"] == status["child_pid"]
            and row["ppid"] == runner_identity["pid"]
            and row["argv"] == list(workload_argv)
        ]
        if len(child_matches) != 1:
            raise ProbeProducerError(
                "guarded-runner child PID/full argv was not directly observed"
            )
        restored = status.get("restored_guards")
        restored_pids = (
            set(restored.values()) if isinstance(restored, dict) else set()
        )
        unknown_live = [
            row
            for row in (*descendants.values(), *escaped_identities)
            if row["pid"] not in restored_pids and _identity_live(row)
        ]
        descendants_exited = not unknown_live
        if return_code != 0 or not descendants_exited:
            _cleanup_owned_identities(unknown_live)
            raise ProbeProducerError("runner failed or descendants remained alive")
        combined = b"".join(
            _snapshot(path, label)[1]
            for path, label in (
                (log_path, "guarded-runner log"),
                (stdout_path, "guarded-runner stdout"),
                (stderr_path, "guarded-runner stderr"),
            )
        ).lower()
        oom = any(marker in combined for marker in OOM_MARKERS)
        if oom:
            raise ProbeProducerError("OOM marker observed in owned runner evidence")
        guards = _verify_restored_guards(
            status,
            nvidia_smi,
            expected_python=python_artifact,
        )
        return ObservedExecution(
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
            gpu_peak_memory_bytes=peaks,
            gpu_total_memory_bytes=totals,
            runner_return_code=return_code,
            oom_observed=oom,
            descendants_exited=descendants_exited,
            runner_identity=runner_identity,
            observed_descendants=sorted(
                {
                    (row["pid"], row["starttime_ticks"]): row
                    for row in (*descendants.values(), *escaped_identities)
                }.values(),
                key=lambda row: (row["pid"], row["starttime_ticks"]),
            ),
            status_artifact=status_artifact,
            log_artifact=log_artifact,
            stdout_artifact=stdout_artifact,
            stderr_artifact=stderr_artifact,
            telemetry_artifact=telemetry_artifact,
            status=status,
            guard_verification=guards,
        )


def _load_feature_matrix(
    value: Any, label: str, *, minimum_rows: int
) -> tuple[dict[str, Any], np.ndarray]:
    artifact, payload = _plain_artifact(value, label)
    try:
        import io

        matrix = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ProbeProducerError(f"{label} is not one safe NumPy array") from error
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or matrix.shape[0] < minimum_rows
        or matrix.shape[1] < 1
        or not np.issubdtype(matrix.dtype, np.floating)
        or not np.isfinite(matrix).all()
    ):
        raise ProbeProducerError(f"{label} feature matrix changed")
    return artifact, np.asarray(matrix, dtype=np.float64)


def _feature_manifest(
    value: Any,
    *,
    binding: Mapping[str, Any],
    real_rows: int,
    generated_rows: int,
) -> tuple[dict[str, Any], str]:
    artifact, payload = _plain_artifact(value, "trainer-native feature manifest")
    rows = authority._strict_jsonl_bytes(
        payload, "trainer-native feature manifest"
    )
    _subset, subset_payload = authority._normalize_artifact(
        binding["subset_manifest"],
        "trainer-native feature subset",
        with_payload=False,
    )
    subset_rows = authority._strict_jsonl_bytes(
        subset_payload, "trainer-native feature subset"
    )
    expected_ids = [row.get("canonical_clip_id") for row in subset_rows]
    real_cursor = 0
    generated_cursor = 0
    normalized: list[dict[str, Any]] = []
    for expected_id, raw in zip(expected_ids, rows):
        row = _exact(
            raw,
            {
                "canonical_clip_id",
                "real_start",
                "real_rows",
                "generated_start",
                "generated_rows",
            },
            "trainer-native feature row",
        )
        if (
            row["canonical_clip_id"] != expected_id
            or type(row["real_start"]) is not int
            or row["real_start"] != real_cursor
            or type(row["real_rows"]) is not int
            or row["real_rows"] < 1
            or type(row["generated_start"]) is not int
            or row["generated_start"] != generated_cursor
            or type(row["generated_rows"]) is not int
            or row["generated_rows"] != 2 * row["real_rows"]
        ):
            raise ProbeProducerError("trainer-native feature coverage changed")
        real_cursor += row["real_rows"]
        generated_cursor += row["generated_rows"]
        normalized.append(dict(row))
    if (
        len(rows) != len(expected_ids)
        or real_cursor != real_rows
        or generated_cursor != generated_rows
        or generated_rows != 2 * real_rows
    ):
        raise ProbeProducerError("trainer-native feature matrix coverage changed")
    return artifact, authority.canonical_json_sha256(normalized)


def _trajectory_fingerprint(
    value: Any,
    *,
    epoch: int,
    quality_input: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    artifact, payload = _plain_artifact(value, "trainer-native trajectory")
    rows = authority._strict_jsonl_bytes(payload, "trainer-native trajectory")
    previous = -1
    normalized = []
    for row in rows:
        row = _exact(row, {"optimizer_update", "loss"}, "trajectory row")
        update = row["optimizer_update"]
        loss = row["loss"]
        if (
            type(update) is not int
            or update <= previous
            or type(loss) not in (int, float)
            or type(loss) is bool
            or not math.isfinite(float(loss))
        ):
            raise ProbeProducerError("trainer-native trajectory changed")
        previous = update
        normalized.append({"optimizer_update": update, "loss": float(loss)})
    if not normalized:
        raise ProbeProducerError("trainer-native trajectory is empty")
    _quality_artifact, quality_payload = _payload_artifact(
        quality_input, "trainer trajectory quality input"
    )
    source_artifact, source_payload = _plain_artifact(
        quality_payload["training_metrics"], "trainer epoch-metric source"
    )
    del source_artifact
    source_rows = authority._strict_jsonl_bytes(
        source_payload, "trainer epoch-metric source"
    )
    expected = []
    for expected_epoch, source in enumerate(source_rows[:epoch], start=1):
        if (
            not isinstance(source, dict)
            or source.get("epoch") != expected_epoch
            or type(source.get("optimizer_updates")) is not int
            or not isinstance(source.get("metrics"), dict)
            or type(source["metrics"].get("total")) not in (int, float)
            or type(source["metrics"].get("total")) is bool
            or source.get("all_finite") is not True
        ):
            raise ProbeProducerError("trainer epoch-metric source changed")
        expected.append(
            {
                "optimizer_update": source["optimizer_updates"],
                "loss": float(source["metrics"]["total"]),
            }
        )
    if len(source_rows) != 400 or normalized != expected:
        raise ProbeProducerError(
            "trainer-native trajectory was not derived from training status"
        )
    return artifact, authority.canonical_json_sha256(normalized)


def _prediction_values_fingerprint(
    payload: bytes,
    *,
    epoch: int,
    binding: Mapping[str, Any],
    workload_root: Path,
    full_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Hash prediction identities without embedding run-local paths.

    Serial and concurrent probes must be byte-equivalent even though their
    create-new output roots are necessarily different.  The manifest file
    itself contains those roots, so its raw SHA is not a semantic
    fingerprint.  Replay every referenced prediction and hash only the
    frozen clip order plus prediction bytes.
    """

    rows = authority._strict_jsonl_bytes(
        payload, f"trainer-native Base e{epoch} prediction manifest"
    )
    _subset, subset_payload = authority._normalize_artifact(
        binding["subset_manifest"],
        "trainer-native probe subset",
        with_payload=False,
    )
    subset_rows = authority._strict_jsonl_bytes(
        subset_payload, "trainer-native probe subset"
    )
    expected_ids = [row.get("canonical_clip_id") for row in subset_rows]
    if len(rows) != len(expected_ids):
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} prediction coverage changed"
        )
    fingerprints: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    paths: set[str] = set()
    for position, (row, expected_id, full_row) in enumerate(
        zip(rows, expected_ids, full_rows)
    ):
        if (
            not isinstance(row, dict)
            or set(row) != {"canonical_clip_id", "prediction"}
            or row.get("canonical_clip_id") != expected_id
            or not isinstance(full_row, Mapping)
            or full_row.get("canonical_clip_id") != expected_id
            or full_row.get("epoch") != epoch
        ):
            raise ProbeProducerError(
                f"trainer-native Base e{epoch} prediction order changed"
            )
        prediction, _prediction_payload = authority._normalize_artifact(
            row.get("prediction"),
            f"trainer-native Base e{epoch} prediction {position}",
            with_payload=False,
        )
        if prediction["path"] in paths:
            raise ProbeProducerError(
                f"trainer-native Base e{epoch} reused a prediction path"
            )
        try:
            Path(prediction["path"]).relative_to(workload_root)
        except ValueError as error:
            raise ProbeProducerError(
                f"trainer-native Base e{epoch} prediction escaped the fresh run"
            ) from error
        if prediction != full_row.get("prediction"):
            raise ProbeProducerError(
                f"trainer-native Base e{epoch} projection differs from full lineage"
            )
        paths.add(prediction["path"])
        artifacts.append(prediction)
        fingerprints.append(
            {
                "canonical_clip_id": expected_id,
                "sha256": prediction["sha256"],
                "bytes": prediction["bytes"],
            }
        )
    return authority.canonical_json_sha256(fingerprints), artifacts


def _same_compact_artifact(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("path", "sha256", "receipt_payload_sha256")
    )


def _same_lineage_authority_receipt(
    value: Any, authority_receipt: Mapping[str, Any]
) -> bool:
    """Compare the inference lineage's compact three-field authority pin."""

    keys = {"path", "sha256", "receipt_payload_sha256"}
    return (
        isinstance(value, dict)
        and set(value) == keys
        and _same_compact_artifact(value, authority_receipt)
    )


def _validate_full_inference_lineage(
    *,
    lineage_artifact: Mapping[str, Any],
    lineage: Mapping[str, Any],
    epoch: int,
    checkpoint: Mapping[str, Any],
    binding: Mapping[str, Any],
    workload_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the complete 1,715-file lineage before accepting first-64."""

    try:
        val_artifact, coverage = orchestrator.val_contract.validate_val_inputs(
            Path(binding["val_inputs"]["path"]),
            binding["val_inputs"]["sha256"],
        )
        pipeline_artifact, _pipeline = orchestrator.val_contract.validate_pipeline(
            Path(binding["pipeline"]["path"]),
            binding["pipeline"]["sha256"],
        )
        audited_lineage, validated = (
            orchestrator.val_contract.validate_val_inference_lineage(
                Path(lineage_artifact["path"]),
                lineage_artifact["sha256"],
                epoch=epoch,
                expected_candidate=checkpoint,
                val_inputs_artifact=val_artifact,
                pipeline_artifact=pipeline_artifact,
                expected_coverage=coverage,
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} full lineage failed replay"
        ) from error
    if (
        not _same_compact_artifact(val_artifact, binding["val_inputs"])
        or not _same_compact_artifact(pipeline_artifact, binding["pipeline"])
        or not _same_compact_artifact(audited_lineage, lineage_artifact)
    ):
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} lineage authority changed"
        )
    full_reference = validated.get("final_manifest")
    if not isinstance(full_reference, dict):
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} lacks full manifest"
        )
    full_manifest = _output_artifact(
        Path(full_reference["path"]),
        f"trainer-native Base e{epoch} full prediction manifest",
    )
    if any(
        full_manifest.get(key) != full_reference.get(key)
        for key in ("path", "sha256")
    ):
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} full manifest changed"
        )
    try:
        Path(full_manifest["path"]).relative_to(workload_root)
        Path(lineage["prediction_dir"]).relative_to(workload_root)
        Path(lineage["ground_truth_dir"]).relative_to(workload_root)
    except ValueError as error:
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} full inference escaped workload root"
        ) from error
    replayed_manifest, manifest_payload = _plain_artifact(
        full_manifest,
        f"trainer-native Base e{epoch} full prediction manifest",
    )
    if replayed_manifest != full_manifest:
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} full manifest changed"
        )
    rows = authority._strict_jsonl_bytes(
        manifest_payload,
        f"trainer-native Base e{epoch} full prediction manifest",
    )
    if len(rows) != orchestrator.EXPECTED_CLIPS:
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} full coverage changed"
        )
    return full_manifest, rows


def _full_lineage_values_fingerprint(
    rows: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    checkpoint: Mapping[str, Any],
) -> str:
    """Path-independent semantic fingerprint of the complete val output."""

    normalized: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        prediction = row.get("prediction")
        ground_truth = row.get("ground_truth")
        if (
            not isinstance(prediction, Mapping)
            or not isinstance(ground_truth, Mapping)
            or row.get("epoch") != epoch
            or row.get("candidate_checkpoint_sha256") != checkpoint["sha256"]
            or type(row.get("global_index")) is not int
            or type(row.get("frames")) is not int
            or not isinstance(row.get("canonical_clip_id"), str)
        ):
            raise ProbeProducerError(
                f"trainer-native Base e{epoch} full row {position} changed"
            )
        normalized.append(
            {
                "global_index": row["global_index"],
                "canonical_clip_id": row["canonical_clip_id"],
                "frames": row["frames"],
                "epoch": epoch,
                "candidate_checkpoint_sha256": checkpoint["sha256"],
                "prediction_sha256": prediction.get("sha256"),
                "prediction_bytes": prediction.get("bytes"),
                "ground_truth_sha256": ground_truth.get("sha256"),
                "ground_truth_bytes": ground_truth.get("bytes"),
            }
        )
    if len(normalized) != orchestrator.EXPECTED_CLIPS:
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} full semantic coverage changed"
        )
    return authority.canonical_json_sha256(normalized)


def _validate_native_candidate(
    value: Any,
    *,
    epoch: int,
    checkpoint: Mapping[str, Any],
    binding: Mapping[str, Any],
    quality_input: Mapping[str, Any],
    workload_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact, receipt = _payload_artifact(value, f"trainer-native Base e{epoch}")
    _exact(
        receipt,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "epoch",
            "candidate_checkpoint",
            "subset_manifest",
            "prediction_manifest",
            "quality_input",
            "inference_lineage",
            "real_features",
            "generated_features",
            "feature_manifest",
            "training_trajectory",
            "receipt_payload_sha256",
        },
        f"trainer-native Base e{epoch}",
    )
    prediction, prediction_payload = orchestrator.authority._normalize_artifact(
        receipt["prediction_manifest"],
        f"trainer-native Base e{epoch} prediction manifest",
        with_payload=False,
    )
    if (
        receipt["format"] != TRAINER_NATIVE_FORMAT
        or receipt["status"] != "complete"
        or receipt["split"] != "val"
        or receipt["test_visible"] is not False
        or receipt["epoch"] != epoch
        or receipt["candidate_checkpoint"] != checkpoint
        or receipt["subset_manifest"] != binding["subset_manifest"]
        or receipt["quality_input"] != quality_input
    ):
        raise ProbeProducerError(f"trainer-native Base e{epoch} authority changed")
    lineage_artifact, lineage = _payload_artifact(
        receipt["inference_lineage"],
        f"trainer-native Base e{epoch} inference lineage",
    )
    if (
        lineage.get("format")
        != orchestrator.val_contract.VAL_INFERENCE_LINEAGE_FORMAT
        or lineage.get("status") != "complete"
        or lineage.get("split") != "val"
        or lineage.get("test_visible") is not False
        or lineage.get("epoch") != epoch
        or lineage.get("candidate_checkpoint")
        != {"path": checkpoint["path"], "sha256": checkpoint["sha256"]}
        or not _same_lineage_authority_receipt(
            lineage.get("val_inputs_receipt"), binding["val_inputs"]
        )
        or not _same_lineage_authority_receipt(
            lineage.get("pipeline_receipt"), binding["pipeline"]
        )
        or lineage.get("clip_count") != orchestrator.EXPECTED_CLIPS
        or lineage.get("prediction_files") != orchestrator.EXPECTED_CLIPS
        or lineage.get("ground_truth_files") != orchestrator.EXPECTED_CLIPS
        or lineage.get("exact_once") is not True
        or lineage.get("finite") is not True
    ):
        raise ProbeProducerError(
            f"trainer-native Base e{epoch} inference lineage changed"
        )
    full_manifest, full_rows = _validate_full_inference_lineage(
        lineage_artifact=lineage_artifact,
        lineage=lineage,
        epoch=epoch,
        checkpoint=checkpoint,
        binding=binding,
        workload_root=workload_root,
    )
    full_lineage_values_sha = _full_lineage_values_fingerprint(
        full_rows,
        epoch=epoch,
        checkpoint=checkpoint,
    )
    real_artifact, real = _load_feature_matrix(
        receipt["real_features"],
        "trainer-native real features",
        minimum_rows=binding["clips_per_candidate"],
    )
    generated_artifact, generated = _load_feature_matrix(
        receipt["generated_features"],
        f"trainer-native Base e{epoch} generated features",
        minimum_rows=2 * binding["clips_per_candidate"],
    )
    if (
        real.shape[1] != generated.shape[1]
        or generated.shape[0] != 2 * real.shape[0]
    ):
        raise ProbeProducerError(
            "trainer-native released2 feature shapes differ"
        )
    feature_manifest_artifact, feature_manifest_sha = _feature_manifest(
        receipt["feature_manifest"],
        binding=binding,
        real_rows=real.shape[0],
        generated_rows=generated.shape[0],
    )
    real_moments = metrics.FeatureMoments()
    generated_moments = metrics.FeatureMoments()
    real_moments.update(real)
    generated_moments.update(generated)
    fgd = metrics.frechet_distance(real_moments, generated_moments)
    if not math.isfinite(fgd):
        raise ProbeProducerError("trainer-native FGD replay is non-finite")
    trajectory_artifact, trajectory_sha = _trajectory_fingerprint(
        receipt["training_trajectory"],
        epoch=epoch,
        quality_input=quality_input,
    )
    prediction_values_sha, prediction_artifacts = _prediction_values_fingerprint(
        prediction_payload,
        epoch=epoch,
        binding=binding,
        workload_root=workload_root,
        full_rows=full_rows[: binding["clips_per_candidate"]],
    )
    raw_fingerprint = authority.canonical_json_sha256(
        {
            "candidate_checkpoint": {
                "sha256": checkpoint["sha256"],
                "bytes": checkpoint["bytes"],
            },
            "prediction_values_sha256": prediction_values_sha,
            "full_lineage_values_sha256": full_lineage_values_sha,
            "real_features": {
                "sha256": real_artifact["sha256"],
                "bytes": real_artifact["bytes"],
            },
            "generated_features": {
                "sha256": generated_artifact["sha256"],
                "bytes": generated_artifact["bytes"],
            },
            "feature_manifest": {
                "sha256": feature_manifest_artifact["sha256"],
                "bytes": feature_manifest_artifact["bytes"],
                "values_sha256": feature_manifest_sha,
            },
            "training_trajectory": {
                "sha256": trajectory_artifact["sha256"],
                "bytes": trajectory_artifact["bytes"],
            },
            "trajectory_values_sha256": trajectory_sha,
            "fgd": fgd,
        }
    )
    return artifact, receipt, {
        "prediction_manifest": prediction,
        "inference_lineage": lineage_artifact,
        "full_prediction_manifest": full_manifest,
        "fgd": fgd,
        "raw_fingerprint_sha256": raw_fingerprint,
        "full_lineage_values_sha256": full_lineage_values_sha,
        "trajectory_values_sha256": trajectory_sha,
        "prediction_artifacts": prediction_artifacts,
        "raw_artifacts": {
            "real_features": real_artifact,
            "generated_features": generated_artifact,
            "feature_manifest": feature_manifest_artifact,
            "training_trajectory": trajectory_artifact,
        },
    }


def _prepare_command_argv(
    *,
    execution_spec: Mapping[str, Any],
    quality: Mapping[str, Any],
    binding: Mapping[str, Any],
    preflight_path: Path,
) -> list[str]:
    _pipeline_artifact, pipeline = _payload_artifact(
        binding["pipeline"], "workload execution pipeline"
    )
    bundle = quality["candidate_bundle"]
    return [
        execution_spec["python"]["path"],
        pipeline["inference_entrypoint"]["path"],
        "prepare",
        "--split",
        "val",
        "--candidate-manifest",
        bundle["manifest"]["path"],
        "--expected-candidate-manifest-sha256",
        bundle["manifest"]["sha256"],
        "--candidate-status",
        bundle["status"]["path"],
        "--expected-candidate-status-sha256",
        bundle["status"]["sha256"],
        "--frozen-inputs",
        bundle["frozen_inputs"]["path"],
        "--expected-frozen-inputs-sha256",
        bundle["frozen_inputs"]["sha256"],
        "--val-inputs",
        binding["val_inputs"]["path"],
        "--expected-val-inputs-sha256",
        binding["val_inputs"]["sha256"],
        "--pipeline",
        binding["pipeline"]["path"],
        "--expected-pipeline-sha256",
        binding["pipeline"]["sha256"],
        "--output",
        str(preflight_path),
    ]


def _shard_command_argv(
    *,
    execution_spec: Mapping[str, Any],
    binding: Mapping[str, Any],
    preflight: Mapping[str, Any],
    epoch: int,
    output_root: Path,
    shard_id: int,
) -> list[str]:
    _pipeline_artifact, pipeline = _payload_artifact(
        binding["pipeline"], "workload execution pipeline"
    )
    return [
        execution_spec["python"]["path"],
        pipeline["inference_entrypoint"]["path"],
        "shard",
        "--split",
        "val",
        "--preflight",
        preflight["path"],
        "--expected-preflight-sha256",
        preflight["sha256"],
        "--epoch",
        str(epoch),
        "--output-root",
        str(output_root),
        "--num-shards",
        str(len(EXPECTED_GPUS)),
        "--shard-id",
        str(shard_id),
        "--device",
        f"cuda:{shard_id}",
        "--seed",
        str(binding["seed"]),
        "--progress-every",
        "20",
    ]


def _finalize_command_argv(
    *,
    execution_spec: Mapping[str, Any],
    binding: Mapping[str, Any],
    preflight: Mapping[str, Any],
    epoch: int,
    output_root: Path,
) -> list[str]:
    _pipeline_artifact, pipeline = _payload_artifact(
        binding["pipeline"], "workload execution pipeline"
    )
    return [
        execution_spec["python"]["path"],
        pipeline["inference_entrypoint"]["path"],
        "finalize",
        "--split",
        "val",
        "--preflight",
        preflight["path"],
        "--expected-preflight-sha256",
        preflight["sha256"],
        "--epoch",
        str(epoch),
        "--output-root",
        str(output_root),
        "--num-shards",
        str(len(EXPECTED_GPUS)),
    ]


def _validate_workload_command(
    value: Any,
    *,
    expected_argv: Sequence[str],
    expected_stdout: Path,
    expected_stderr: Path,
    label: str,
    expected_output: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, set[str]]:
    keys = {"argv", "argv_sha256", "return_code", "stdout", "stderr"}
    if expected_output is not None:
        keys.add("output")
    row = _exact(value, keys, label)
    stdout, _stdout_payload = _plain_artifact(row["stdout"], f"{label} stdout")
    stderr, _stderr_payload = _plain_artifact(row["stderr"], f"{label} stderr")
    if (
        row["argv"] != list(expected_argv)
        or row["argv_sha256"]
        != authority.canonical_json_sha256(list(expected_argv))
        or row["return_code"] != 0
        or stdout["path"] != str(expected_stdout)
        or stderr["path"] != str(expected_stderr)
        or stdout == stderr
    ):
        raise ProbeProducerError(f"{label} command evidence changed")
    output: dict[str, Any] | None = None
    if expected_output is not None:
        output, _payload = _payload_artifact(row["output"], f"{label} output")
        if output["path"] != str(expected_output):
            raise ProbeProducerError(f"{label} output path changed")
    paths = {stdout["path"], stderr["path"]}
    if output is not None:
        paths.add(output["path"])
    if len(paths) != 2 + int(output is not None):
        raise ProbeProducerError(f"{label} reused an evidence path")
    return row, output, paths


def _validate_workload_execution(
    value: Any,
    *,
    execution_spec: Mapping[str, Any],
    binding: Mapping[str, Any],
    workload_root: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], set[str]]:
    artifact, receipt = _payload_artifact(value, "probe workload execution")
    _exact(
        receipt,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "formal_host",
            "candidates_per_wave",
            "execution_mode",
            "probe_binding_sha256",
            "quality_input",
            "prepare",
            "candidates",
            "receipt_payload_sha256",
        },
        "probe workload execution",
    )
    _quality_artifact, quality = _payload_artifact(
        execution_spec["quality_input"], "probe workload quality input"
    )
    commands_root = workload_root / "commands"
    preflight_path = workload_root / "preflight.json"
    if (
        artifact["path"] != str(workload_root / "workload-execution.json")
        or receipt["format"] != WORKLOAD_EXECUTION_FORMAT
        or receipt["status"] != "complete"
        or receipt["split"] != "val"
        or receipt["test_visible"] is not False
        or receipt["formal_host"] != execution_spec["formal_host"]
        or receipt["candidates_per_wave"]
        != execution_spec["candidates_per_wave"]
        or receipt["execution_mode"] != execution_spec["execution_mode"]
        or receipt["probe_binding_sha256"]
        != authority.canonical_json_sha256(binding)
        or receipt["quality_input"] != execution_spec["quality_input"]
    ):
        raise ProbeProducerError("probe workload execution identity changed")
    prepare_argv = _prepare_command_argv(
        execution_spec=execution_spec,
        quality=quality,
        binding=binding,
        preflight_path=preflight_path,
    )
    _prepare, preflight, unique_paths = _validate_workload_command(
        receipt["prepare"],
        expected_argv=prepare_argv,
        expected_stdout=commands_root / "prepare.stdout",
        expected_stderr=commands_root / "prepare.stderr",
        expected_output=preflight_path,
        label="probe prepare",
    )
    if preflight is None:  # pragma: no cover - exact schema requires output.
        raise AssertionError("prepare output disappeared")
    unique_paths.add(artifact["path"])
    candidates = receipt["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(orchestrator.PROBE_EPOCHS):
        raise ProbeProducerError("probe workload candidate coverage changed")
    checkpoints = {
        row["epoch"]: row["candidate_checkpoint"]
        for row in binding["candidate_checkpoints"]
    }
    by_epoch: dict[int, dict[str, Any]] = {}
    expected_logs = {
        commands_root / "prepare.stdout",
        commands_root / "prepare.stderr",
    }
    for expected_epoch, raw in zip(orchestrator.PROBE_EPOCHS, candidates):
        row = _exact(
            raw,
            {
                "epoch",
                "candidate_checkpoint",
                "shards",
                "finalize",
                "inference_lineage",
                "full_prediction_manifest",
            },
            f"probe workload Base e{expected_epoch}",
        )
        if (
            row["epoch"] != expected_epoch
            or row["candidate_checkpoint"] != checkpoints[expected_epoch]
            or not isinstance(row["shards"], list)
            or len(row["shards"]) != len(EXPECTED_GPUS)
        ):
            raise ProbeProducerError(
                f"probe workload Base e{expected_epoch} identity changed"
            )
        inference_root = workload_root / f"e{expected_epoch}" / "inference"
        for shard_id, shard_raw in enumerate(row["shards"]):
            shard = _exact(
                shard_raw,
                {
                    "shard_id",
                    "device",
                    "argv",
                    "argv_sha256",
                    "return_code",
                    "stdout",
                    "stderr",
                },
                f"probe Base e{expected_epoch} shard {shard_id}",
            )
            if shard["shard_id"] != shard_id or shard["device"] != f"cuda:{shard_id}":
                raise ProbeProducerError(
                    f"probe Base e{expected_epoch} shard topology changed"
                )
            stdout_path = commands_root / f"e{expected_epoch}-shard-{shard_id}.stdout"
            stderr_path = commands_root / f"e{expected_epoch}-shard-{shard_id}.stderr"
            _record, _output, paths = _validate_workload_command(
                {key: shard[key] for key in shard if key not in {"shard_id", "device"}},
                expected_argv=_shard_command_argv(
                    execution_spec=execution_spec,
                    binding=binding,
                    preflight=preflight,
                    epoch=expected_epoch,
                    output_root=inference_root,
                    shard_id=shard_id,
                ),
                expected_stdout=stdout_path,
                expected_stderr=stderr_path,
                label=f"probe Base e{expected_epoch} shard {shard_id}",
            )
            if unique_paths & paths:
                raise ProbeProducerError("probe workload reused command evidence")
            unique_paths.update(paths)
            expected_logs.update({stdout_path, stderr_path})
        finalize_stdout = commands_root / f"e{expected_epoch}-finalize.stdout"
        finalize_stderr = commands_root / f"e{expected_epoch}-finalize.stderr"
        lineage_reference, _lineage_payload = _payload_artifact(
            row["inference_lineage"],
            f"probe Base e{expected_epoch} execution lineage",
        )
        _finalize, finalize_output, paths = _validate_workload_command(
            row["finalize"],
            expected_argv=_finalize_command_argv(
                execution_spec=execution_spec,
                binding=binding,
                preflight=preflight,
                epoch=expected_epoch,
                output_root=inference_root,
            ),
            expected_stdout=finalize_stdout,
            expected_stderr=finalize_stderr,
            expected_output=Path(lineage_reference["path"]),
            label=f"probe Base e{expected_epoch} finalize",
        )
        if finalize_output != lineage_reference:
            raise ProbeProducerError(
                f"probe Base e{expected_epoch} finalize lineage changed"
            )
        full_manifest = _output_artifact(
            Path(row["full_prediction_manifest"]["path"]),
            f"probe Base e{expected_epoch} execution full manifest",
        )
        if full_manifest != row["full_prediction_manifest"]:
            raise ProbeProducerError(
                f"probe Base e{expected_epoch} full-manifest evidence changed"
            )
        # Lineage/full-manifest paths are intentionally shared with the
        # candidate receipts and are cross-bound below, not counted as unique
        # command artifacts here.
        paths.discard(lineage_reference["path"])
        if unique_paths & paths:
            raise ProbeProducerError("probe workload reused finalize evidence")
        unique_paths.update(paths)
        expected_logs.update({finalize_stdout, finalize_stderr})
        by_epoch[expected_epoch] = {
            "inference_lineage": lineage_reference,
            "full_prediction_manifest": full_manifest,
        }
    if (
        not commands_root.is_dir()
        or commands_root.is_symlink()
        or set(commands_root.iterdir()) != expected_logs
        or len(unique_paths) != 76
    ):
        raise ProbeProducerError("probe workload command closure changed")
    for path in unique_paths:
        try:
            Path(path).relative_to(workload_root)
        except ValueError as error:
            raise ProbeProducerError("probe workload execution escaped root") from error
    return artifact, by_epoch, unique_paths


def _validate_candidate_ready(
    path: Path,
    *,
    execution_spec: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    raw_artifact = _output_artifact(path, "candidate-ready receipt")
    _path, payload, _metadata = _snapshot(path, "candidate-ready receipt")
    decoded = _strict_json(payload, "candidate-ready receipt")
    if not isinstance(decoded, dict):
        raise ProbeProducerError("candidate-ready receipt is not an object")
    artifact, ready = _payload_artifact(
        {**raw_artifact, "receipt_payload_sha256": decoded.get("receipt_payload_sha256")},
        "candidate-ready receipt",
    )
    _exact(
        ready,
        {
            "format",
            "status",
            "split",
            "test_visible",
            "formal_host",
            "candidates_per_wave",
            "execution_mode",
            "probe_binding_sha256",
            "quality_input",
            "workload_execution",
            "candidates",
            "receipt_payload_sha256",
        },
        "candidate-ready receipt",
    )
    if (
        ready["format"] != CANDIDATE_READY_FORMAT
        or ready["status"] != "complete"
        or ready["split"] != "val"
        or ready["test_visible"] is not False
        or ready["formal_host"] != execution_spec["formal_host"]
        or ready["candidates_per_wave"] != execution_spec["candidates_per_wave"]
        or ready["execution_mode"] != execution_spec["execution_mode"]
        or ready["probe_binding_sha256"] != authority.canonical_json_sha256(binding)
        or ready["quality_input"] != execution_spec["quality_input"]
        or not isinstance(ready["candidates"], list)
        or len(ready["candidates"]) != len(orchestrator.PROBE_EPOCHS)
    ):
        raise ProbeProducerError("candidate-ready identity changed")
    execution_artifact, executed, execution_paths = _validate_workload_execution(
        ready["workload_execution"],
        execution_spec=execution_spec,
        binding=binding,
        workload_root=path.parent,
    )
    checkpoints = {
        row["epoch"]: row["candidate_checkpoint"]
        for row in binding["candidate_checkpoints"]
    }
    replayed: list[dict[str, Any]] = []
    paths = {artifact["path"], *execution_paths}
    if len(paths) != 1 + len(execution_paths):
        raise ProbeProducerError("candidate-ready reused execution evidence")
    for epoch, native in zip(orchestrator.PROBE_EPOCHS, ready["candidates"]):
        native_artifact, _native, replay = _validate_native_candidate(
            native,
            epoch=epoch,
            checkpoint=checkpoints[epoch],
            binding=binding,
            quality_input=execution_spec["quality_input"],
            workload_root=path.parent,
        )
        candidate_paths = {
            native_artifact["path"],
            replay["prediction_manifest"]["path"],
            replay["inference_lineage"]["path"],
            replay["full_prediction_manifest"]["path"],
            *(artifact["path"] for artifact in replay["prediction_artifacts"]),
            *(
                raw_artifact["path"]
                for raw_artifact in replay["raw_artifacts"].values()
            ),
        }
        if (
            len(candidate_paths)
            != 8 + binding["clips_per_candidate"]
            or paths & candidate_paths
            or replay["inference_lineage"]
            != executed[epoch]["inference_lineage"]
            or replay["full_prediction_manifest"]
            != executed[epoch]["full_prediction_manifest"]
        ):
            raise ProbeProducerError(
                "trainer-native candidate reused an evidence path"
            )
        for candidate_path in candidate_paths:
            try:
                Path(candidate_path).relative_to(path.parent)
            except ValueError as error:
                raise ProbeProducerError(
                    "trainer-native candidate escaped workload root"
                ) from error
        paths.update(candidate_paths)
        replayed.append(
            {"epoch": epoch, "artifact": native_artifact, **replay}
        )
    return artifact, replayed, {
        "artifact": execution_artifact,
        "paths": execution_paths,
        "candidates": executed,
    }


def _workload_argv(
    spec_artifact: Mapping[str, Any],
    spec: Mapping[str, Any],
    workload_root: Path,
    *,
    execution_spec_path: str | None = None,
    quality_input_path: str | None = None,
) -> list[str]:
    quality = spec["quality_input"]
    return [
        spec["python"]["path"],
        spec["quality_producer"]["path"],
        "run-probe",
        "--execution-spec-path",
        execution_spec_path or spec_artifact["path"],
        "--execution-spec-sha256",
        spec_artifact["sha256"],
        "--execution-spec-payload-sha256",
        spec_artifact["receipt_payload_sha256"],
        "--quality-input-path",
        quality_input_path or quality["path"],
        "--quality-input-sha256",
        quality["sha256"],
        "--quality-input-payload-sha256",
        quality["receipt_payload_sha256"],
        "--formal-host",
        spec["formal_host"],
        "--candidates-per-wave",
        str(spec["candidates_per_wave"]),
        "--execution-mode",
        spec["execution_mode"],
        "--output-root",
        str(workload_root),
        "--candidate-ready-output",
        str(workload_root / "candidate-ready.json"),
    ]


def _runner_argv(
    spec: Mapping[str, Any], root: Path, workload_argv: Sequence[str]
) -> list[str]:
    return [
        spec["python"]["path"],
        spec["runner"]["path"],
        "--gpus",
        EXPECTED_GPU_ARGUMENT,
        "--cwd",
        spec["source_root"],
        "--log",
        str(root / "runner.log"),
        "--status",
        str(root / "runner-status.json"),
        "--",
        *workload_argv,
    ]


def _run_and_seal_bound_probe(
    *,
    execution_spec_value: Mapping[str, Any],
    spec_artifact: Mapping[str, Any],
    spec: Mapping[str, Any],
    binding: Mapping[str, Any],
    root: Path,
    workload_root: Path,
    root_identity: _RunRootIdentity,
    input_identity: _InputClosureIdentity,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Execute a probe while its run root and complete inputs are pinned."""

    execution_snapshot: _PinnedInput | None = None
    quality_snapshot: _PinnedInput | None = None
    try:
        execution_snapshot = _pin_input_artifact(
            spec_artifact,
            label="execution-spec",
            fallback_path=root / "pinned-inputs" / "execution-spec.json",
        )
        quality_snapshot = _pin_input_artifact(
            spec["quality_input"],
            label="quality-input",
            fallback_path=root / "pinned-inputs" / "quality-input.json",
        )
        workload_argv = _workload_argv(
            spec_artifact,
            spec,
            workload_root,
            execution_spec_path=execution_snapshot.path,
            quality_input_path=quality_snapshot.path,
        )
        runner_argv = _runner_argv(spec, root, workload_argv)
        root_identity.verify()
        input_identity.verify()
        execution = FormalExecutionBackend().execute(
            runner_argv=runner_argv,
            workload_argv=workload_argv,
            root=root,
            nvidia_smi=Path(spec["nvidia_smi"]["path"]),
            python_artifact=spec["python"],
        )
        replayed_artifact, replayed_spec, replayed_binding = (
            validate_execution_spec(execution_spec_value)
        )
        if (
            replayed_artifact != spec_artifact
            or replayed_spec != spec
            or replayed_binding != binding
        ):
            raise ProbeProducerError("probe inputs changed during execution")
        root_identity.verify()
        input_identity.verify()
    finally:
        if quality_snapshot is not None:
            quality_snapshot.close()
        if execution_snapshot is not None:
            execution_snapshot.close()
    candidate_ready, candidates, workload_execution = _validate_candidate_ready(
        workload_root / "candidate-ready.json",
        execution_spec=spec,
        binding=binding,
    )
    del workload_execution
    metric_artifacts = []
    candidate_outputs = []
    checkpoint_by_epoch = {
        row["epoch"]: row["candidate_checkpoint"]
        for row in binding["candidate_checkpoints"]
    }
    for candidate in candidates:
        epoch = candidate["epoch"]
        metric = orchestrator._with_payload_sha(
            {
                "format": orchestrator.MULTICANDIDATE_PROBE_METRIC_FORMAT,
                "status": "complete",
                "split": "val",
                "test_visible": False,
                "epoch": epoch,
                "candidate_checkpoint": checkpoint_by_epoch[epoch],
                "subset_manifest": binding["subset_manifest"],
                "prediction_manifest": candidate["prediction_manifest"],
                "trainer_native_candidate": candidate["artifact"],
                "metric_values": {
                    "body.released2.metrics.FGD": candidate["fgd"]
                },
                "trajectory_values_sha256": candidate[
                    "trajectory_values_sha256"
                ],
                "raw_metric_fingerprint_sha256": candidate[
                    "raw_fingerprint_sha256"
                ],
            }
        )
        metric_path = workload_root / f"metric-e{epoch}.json"
        metric_artifact = orchestrator._write_new(metric_path, metric)
        metric_artifacts.append(metric_artifact)
        candidate_outputs.append(
            {
                "epoch": epoch,
                "candidate_checkpoint": checkpoint_by_epoch[epoch],
                "prediction_manifest": candidate["prediction_manifest"],
                "metric_receipt": metric_artifact,
            }
        )
    evidence = orchestrator._with_payload_sha(
        {
            "format": PRODUCER_EVIDENCE_FORMAT,
            "status": "complete",
            "execution_spec": spec_artifact,
            "runner_argv": runner_argv,
            "workload_argv": workload_argv,
            "runner_argv_sha256": authority.canonical_json_sha256(runner_argv),
            "workload_argv_sha256": authority.canonical_json_sha256(workload_argv),
            "started_monotonic_ns": execution.started_monotonic_ns,
            "finished_monotonic_ns": execution.finished_monotonic_ns,
            "gpu_peak_memory_bytes": execution.gpu_peak_memory_bytes,
            "gpu_total_memory_bytes": execution.gpu_total_memory_bytes,
            "runner_return_code": execution.runner_return_code,
            "oom_observed": execution.oom_observed,
            "descendants_exited": execution.descendants_exited,
            "runner_identity": execution.runner_identity,
            "observed_descendants": execution.observed_descendants,
            "runner_status": execution.status_artifact,
            "runner_log": execution.log_artifact,
            "runner_stdout": execution.stdout_artifact,
            "runner_stderr": execution.stderr_artifact,
            "memory_telemetry": execution.telemetry_artifact,
            "guard_verification": execution.guard_verification,
            "candidate_ready": candidate_ready,
            "trainer_native_candidates": [row["artifact"] for row in candidates],
            "derived_metrics": metric_artifacts,
        }
    )
    evidence_path = root / "producer-evidence.json"
    evidence_artifact = orchestrator._write_new(evidence_path, evidence)
    run = orchestrator._with_payload_sha(
        {
            "format": orchestrator.MULTICANDIDATE_PROBE_RUN_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "formal_host": spec["formal_host"],
            "candidates_per_wave": spec["candidates_per_wave"],
            "execution_mode": spec["execution_mode"],
            "probe_binding": binding,
            "candidate_outputs": candidate_outputs,
            "execution_trace": {
                "started_monotonic_ns": execution.started_monotonic_ns,
                "finished_monotonic_ns": execution.finished_monotonic_ns,
                "gpu_peak_memory_bytes": execution.gpu_peak_memory_bytes,
                "gpu_total_memory_bytes": execution.gpu_total_memory_bytes,
            },
            "process_evidence": {
                "runner_rc": execution.runner_return_code,
                "oom": execution.oom_observed,
                "descendants_exited": execution.descendants_exited,
                "guards_restored": True,
            },
            "producer_evidence": evidence_artifact,
        }
    )
    orchestrator._fresh_validate_generated_receipt(
        run,
        label="controlled-probe-run",
        replay=orchestrator._replay_probe_run,
    )
    if output_json is not None:
        if output_json != root / "probe-run.json":
            raise ProbeProducerError(
                "formal probe output must be output-root/probe-run.json"
            )
        root_identity.verify()
        orchestrator._write_new(output_json, run)
        root_identity.verify()
    return run


def run_and_seal_probe(
    execution_spec_value: Mapping[str, Any],
    *,
    output_root: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Execute and return one freshly replayed formal probe-run payload."""

    spec_artifact, spec, binding = validate_execution_spec(execution_spec_value)
    root = Path(output_root)
    try:
        root, device, inode = orchestrator._create_new_directory_tree(
            root,
            subdirectories=("workload", "pinned-inputs"),
        )
    except (OSError, orchestrator.BaseFreshValOrchestratorError) as error:
        raise ProbeProducerError(
            "probe output root must be canonical, create-new, and non-symlink"
        ) from error
    workload_root = root / "workload"
    root_identity: _RunRootIdentity | None = None
    input_identity: _InputClosureIdentity | None = None
    active_error: BaseException | None = None
    try:
        root_identity = _RunRootIdentity.acquire(
            root, expected_dev=device, expected_ino=inode
        )
        input_identity = _InputClosureIdentity.acquire(
            _execution_input_closure(spec_artifact, spec, binding)
        )
        return _run_and_seal_bound_probe(
            execution_spec_value=execution_spec_value,
            spec_artifact=spec_artifact,
            spec=spec,
            binding=binding,
            root=root,
            workload_root=workload_root,
            root_identity=root_identity,
            input_identity=input_identity,
            output_json=output_json,
        )
    except BaseException as error:
        active_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        for identity in (input_identity, root_identity):
            if identity is None:
                continue
            try:
                identity.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if active_error is None and cleanup_error is not None:
            raise cleanup_error


def _validate_probe_run_evidence(run: Mapping[str, Any]) -> dict[str, Any]:
    evidence_artifact, evidence = _payload_artifact(
        run.get("producer_evidence"), "probe producer evidence"
    )
    _exact(
        evidence,
        {
            "format",
            "status",
            "execution_spec",
            "runner_argv",
            "workload_argv",
            "runner_argv_sha256",
            "workload_argv_sha256",
            "started_monotonic_ns",
            "finished_monotonic_ns",
            "gpu_peak_memory_bytes",
            "gpu_total_memory_bytes",
            "runner_return_code",
            "oom_observed",
            "descendants_exited",
            "runner_identity",
            "observed_descendants",
            "runner_status",
            "runner_log",
            "runner_stdout",
            "runner_stderr",
            "memory_telemetry",
            "guard_verification",
            "candidate_ready",
            "trainer_native_candidates",
            "derived_metrics",
            "receipt_payload_sha256",
        },
        "probe producer evidence",
    )
    spec_artifact, spec, binding = validate_execution_spec(evidence["execution_spec"])
    _quality_artifact, quality = _payload_artifact(
        spec["quality_input"], "replayed probe quality input"
    )
    workload_root = Path(evidence["candidate_ready"]["path"]).parent
    root = Path(evidence["runner_status"]["path"]).parent
    expected_paths = {
        "producer_evidence": root / "producer-evidence.json",
        "runner_status": root / "runner-status.json",
        "runner_log": root / "runner.log",
        "runner_stdout": root / "runner.stdout",
        "runner_stderr": root / "runner.stderr",
        "memory_telemetry": root / "memory-telemetry.json",
        "candidate_ready": root / "workload" / "candidate-ready.json",
    }
    if (
        workload_root != root / "workload"
        or evidence_artifact["path"]
        != str(expected_paths["producer_evidence"])
        or any(
            evidence[label]["path"] != str(path)
            for label, path in expected_paths.items()
            if label != "producer_evidence"
        )
    ):
        raise ProbeProducerError("probe evidence escaped its create-new root")
    workload_argv = evidence["workload_argv"]
    runner_argv = evidence["runner_argv"]
    if (
        not isinstance(workload_argv, list)
        or not isinstance(runner_argv, list)
        or any(
            not isinstance(item, str) or not item or "\0" in item
            for item in (*workload_argv, *runner_argv)
        )
        or workload_argv.count("--execution-spec-path") != 1
        or workload_argv.count("--quality-input-path") != 1
    ):
        raise ProbeProducerError("probe argv evidence changed")
    execution_input_path = workload_argv[
        workload_argv.index("--execution-spec-path") + 1
    ]
    quality_input_path = workload_argv[
        workload_argv.index("--quality-input-path") + 1
    ]
    proc_pattern = re.compile(r"/proc/([1-9][0-9]*)/fd/([0-9]+)")
    execution_proc_match = proc_pattern.fullmatch(execution_input_path)
    quality_proc_match = proc_pattern.fullmatch(quality_input_path)
    fallback_execution = root / "pinned-inputs" / "execution-spec.json"
    fallback_quality = root / "pinned-inputs" / "quality-input.json"
    if not (
        (
            execution_proc_match
            and quality_proc_match
        )
        or (
            execution_input_path == str(fallback_execution)
            and quality_input_path == str(fallback_quality)
        )
    ):
        raise ProbeProducerError("probe workload did not use pinned inputs")
    expected_workload_argv = _workload_argv(
        spec_artifact,
        spec,
        workload_root,
        execution_spec_path=execution_input_path,
        quality_input_path=quality_input_path,
    )
    expected_runner_argv = _runner_argv(spec, root, expected_workload_argv)
    if (
        evidence["format"] != PRODUCER_EVIDENCE_FORMAT
        or evidence["status"] != "complete"
        or runner_argv != expected_runner_argv
        or workload_argv != expected_workload_argv
        or evidence["runner_argv_sha256"]
        != authority.canonical_json_sha256(runner_argv)
        or evidence["workload_argv_sha256"]
        != authority.canonical_json_sha256(workload_argv)
        or evidence["runner_return_code"] != 0
        or evidence["oom_observed"] is not False
        or evidence["descendants_exited"] is not True
        or run.get("execution_trace")
        != {
            "started_monotonic_ns": evidence["started_monotonic_ns"],
            "finished_monotonic_ns": evidence["finished_monotonic_ns"],
            "gpu_peak_memory_bytes": evidence["gpu_peak_memory_bytes"],
            "gpu_total_memory_bytes": evidence["gpu_total_memory_bytes"],
        }
        or run.get("process_evidence")
        != {"runner_rc": 0, "oom": False, "descendants_exited": True, "guards_restored": True}
        or run.get("probe_binding") != binding
        or run.get("formal_host") != spec["formal_host"]
        or run.get("candidates_per_wave") != spec["candidates_per_wave"]
        or run.get("execution_mode") != spec["execution_mode"]
    ):
        raise ProbeProducerError("probe run differs from producer observations")
    started = evidence["started_monotonic_ns"]
    finished = evidence["finished_monotonic_ns"]
    peaks = evidence["gpu_peak_memory_bytes"]
    totals = evidence["gpu_total_memory_bytes"]
    if (
        type(started) is not int
        or type(finished) is not int
        or started < 0
        or finished <= started
        or not isinstance(peaks, list)
        or not isinstance(totals, list)
        or len(peaks) != len(EXPECTED_GPUS)
        or len(totals) != len(EXPECTED_GPUS)
        or any(type(value) is not int or value < 0 for value in peaks)
        or any(type(value) is not int or value <= 0 for value in totals)
        or any(peak > total for peak, total in zip(peaks, totals))
    ):
        raise ProbeProducerError("probe timing/memory evidence changed")
    owned_payloads = {}
    for label in ("runner_status", "runner_log", "runner_stdout", "runner_stderr"):
        _artifact, owned_payloads[label] = _plain_artifact(
            evidence[label], f"probe {label}"
        )
    if (
        not owned_payloads["runner_status"]
        or not owned_payloads["runner_log"]
        or any(
            marker
            in b"".join(owned_payloads.values()).lower()
            for marker in OOM_MARKERS
        )
    ):
        raise ProbeProducerError("owned runner evidence is missing or reports OOM")
    _telemetry_artifact, telemetry = _payload_artifact(
        evidence["memory_telemetry"], "probe memory telemetry"
    )
    _exact(
        telemetry,
        {
            "format",
            "status",
            "gpu_indices",
            "samples",
            "peak_memory_bytes",
            "total_memory_bytes",
            "receipt_payload_sha256",
        },
        "probe memory telemetry",
    )
    samples = telemetry["samples"]
    if (
        telemetry["format"] != MEMORY_TELEMETRY_FORMAT
        or telemetry["status"] != "complete"
        or telemetry["gpu_indices"] != list(EXPECTED_GPUS)
        or telemetry["peak_memory_bytes"] != peaks
        or telemetry["total_memory_bytes"] != totals
        or not isinstance(samples, list)
        or not samples
    ):
        raise ProbeProducerError("probe memory telemetry changed")
    previous_sample = started - 1
    replayed_peaks = [0] * len(EXPECTED_GPUS)
    for sample in samples:
        _exact(sample, {"monotonic_ns", "used_bytes"}, "memory sample")
        stamp = sample["monotonic_ns"]
        used = sample["used_bytes"]
        if (
            type(stamp) is not int
            or stamp <= previous_sample
            or stamp < started
            or stamp > finished
            or not isinstance(used, list)
            or len(used) != len(EXPECTED_GPUS)
            or any(type(value) is not int or value < 0 for value in used)
            or any(value > total for value, total in zip(used, totals))
        ):
            raise ProbeProducerError("probe memory sample changed")
        previous_sample = stamp
        replayed_peaks = [
            max(old, current) for old, current in zip(replayed_peaks, used)
        ]
    if replayed_peaks != peaks:
        raise ProbeProducerError("probe peak memory does not replay")
    status_artifact, status_payload = _plain_artifact(
        evidence["runner_status"], "probe runner status"
    )
    del status_artifact
    status = _strict_json_document(status_payload, "probe runner status")
    expected_status = {
        "state": "finished",
        "return_code": 0,
        "error": None,
        "cleanup_error": None,
        "restore_error": None,
        "received_signal": None,
        "command": workload_argv,
        "wrapper_pid": evidence["runner_identity"]["pid"],
    }
    runner_identity = _exact(
        evidence["runner_identity"],
        {
            "pid",
            "ppid",
            "pgid",
            "sid",
            "starttime_ticks",
            "argv",
            "argv_sha256",
        },
        "probe runner identity",
    )
    if (
        not isinstance(status, dict)
        or any(status.get(key) != value for key, value in expected_status.items())
        or runner_identity["argv"] != runner_argv
        or type(runner_identity["pid"]) is not int
        or runner_identity["pid"] <= 1
        or type(runner_identity["ppid"]) is not int
        or runner_identity["ppid"] <= 1
        or _require_sha(
            runner_identity["argv_sha256"], "runner argv SHA-256"
        )
        != _sha256(
            b"\0".join(os.fsencode(item) for item in runner_argv) + b"\0"
        )
        or (
            execution_proc_match is not None
            and (
                int(execution_proc_match.group(1))
                != runner_identity["ppid"]
                or quality_proc_match is None
                or int(quality_proc_match.group(1))
                != runner_identity["ppid"]
                or execution_proc_match.group(2)
                == quality_proc_match.group(2)
            )
        )
    ):
        raise ProbeProducerError("replayed runner status changed")
    child_pid = status.get("child_pid")
    descendants = evidence["observed_descendants"]
    if not isinstance(descendants, list):
        raise ProbeProducerError("observed descendant evidence changed")
    direct_children = []
    for descendant in descendants:
        row = _exact(
            descendant,
            {
                "pid",
                "ppid",
                "pgid",
                "sid",
                "starttime_ticks",
                "argv",
                "argv_sha256",
            },
            "observed descendant",
        )
        if (
            type(row["pid"]) is not int
            or row["pid"] <= 1
            or type(row["ppid"]) is not int
            or type(row["starttime_ticks"]) is not int
            or row["starttime_ticks"] < 1
            or not isinstance(row["argv"], list)
            or _require_sha(row["argv_sha256"], "descendant argv SHA-256")
            != _sha256(
                b"\0".join(os.fsencode(item) for item in row["argv"])
                + b"\0"
            )
        ):
            raise ProbeProducerError("observed descendant identity changed")
        if (
            row["pid"] == child_pid
            and row["ppid"] == runner_identity["pid"]
            and row["argv"] == workload_argv
        ):
            direct_children.append(row)
    if len(direct_children) != 1:
        raise ProbeProducerError("direct guarded workload ancestry changed")
    guard = _exact(
        evidence["guard_verification"],
        {
            "restored_guards",
            "guard_identities",
            "guard_executables",
            "nvidia_smi_compute_rows_sha256",
            "nvidia_smi_uuid_rows_sha256",
        },
        "probe restored guards",
    )
    restored = guard["restored_guards"]
    guard_identities = guard["guard_identities"]
    guard_executables = guard["guard_executables"]
    if (
        restored != status.get("restored_guards")
        or not isinstance(restored, dict)
        or set(restored) != {str(index) for index in EXPECTED_GPUS}
        or len(set(restored.values())) != len(EXPECTED_GPUS)
        or not isinstance(guard_identities, dict)
        or set(guard_identities) != set(restored)
        or not isinstance(guard_executables, dict)
        or set(guard_executables) != set(restored)
        or any(type(pid) is not int or pid <= 1 for pid in restored.values())
    ):
        raise ProbeProducerError("guard verification differs from runner status")
    descendant_pids = [row["pid"] for row in descendants]
    if len(descendant_pids) != len(set(descendant_pids)):
        raise ProbeProducerError("descendant PID identity was reused")
    allowed_chain_parents = {
        runner_identity["pid"],
        runner_identity["ppid"],
        *descendant_pids,
    }
    if any(
        row["pid"] not in set(restored.values())
        and row["ppid"] not in allowed_chain_parents
        for row in descendants
    ):
        raise ProbeProducerError("observed descendant escaped the exact ancestor chain")
    for index in EXPECTED_GPUS:
        identity = _exact(
            guard_identities[str(index)],
            {
                "pid",
                "ppid",
                "pgid",
                "sid",
                "starttime_ticks",
                "argv",
                "argv_sha256",
            },
            f"GPU{index} restored guard",
        )
        argv_text = "\0".join(identity["argv"]).casefold()
        matching_descendants = [
            row
            for row in descendants
            if row["pid"] == identity["pid"]
            and row["starttime_ticks"] == identity["starttime_ticks"]
            and row["argv_sha256"] == identity["argv_sha256"]
        ]
        executable, _python_payload = _plain_artifact(
            guard_executables[str(index)],
            f"GPU{index} restored guard executable",
        )
        if (
            identity["pid"] != restored[str(index)]
            or identity["ppid"] != runner_identity["ppid"]
            or len(matching_descendants) != 1
            or executable != spec["python"]
            or "globaldiff_gpu_guard_cnn" not in argv_text
            or "torchvision" not in argv_text
            or "resnet18" not in argv_text
        ):
            raise ProbeProducerError(f"GPU{index} restored guard changed")
    _require_sha(
        guard["nvidia_smi_compute_rows_sha256"], "compute rows SHA-256"
    )
    _require_sha(guard["nvidia_smi_uuid_rows_sha256"], "UUID rows SHA-256")

    ready_artifact, candidates, workload_execution = _validate_candidate_ready(
        Path(evidence["candidate_ready"]["path"]),
        execution_spec=spec,
        binding=binding,
    )
    if ready_artifact != evidence["candidate_ready"]:
        raise ProbeProducerError("candidate-ready artifact binding changed")
    native_artifacts = [candidate["artifact"] for candidate in candidates]
    derived_artifacts = evidence["derived_metrics"]
    outputs = run.get("candidate_outputs")
    if (
        evidence["trainer_native_candidates"] != native_artifacts
        or not isinstance(derived_artifacts, list)
        or not isinstance(outputs, list)
        or len(derived_artifacts) != len(candidates)
        or len(outputs) != len(candidates)
    ):
        raise ProbeProducerError("candidate/metric evidence coverage changed")
    checkpoint_by_epoch = {
        row["epoch"]: row["candidate_checkpoint"]
        for row in binding["candidate_checkpoints"]
    }
    evidence_paths = {
        ready_artifact["path"],
        evidence["runner_status"]["path"],
        evidence["runner_log"]["path"],
        evidence["runner_stdout"]["path"],
        evidence["runner_stderr"]["path"],
        evidence["memory_telemetry"]["path"],
        *workload_execution["paths"],
    }
    if len(evidence_paths) != 6 + len(workload_execution["paths"]):
        raise ProbeProducerError("probe evidence reused workload execution paths")
    for candidate, metric_value, output in zip(
        candidates, derived_artifacts, outputs
    ):
        epoch = candidate["epoch"]
        metric_artifact, metric = _payload_artifact(
            metric_value, f"derived Base e{epoch} metric"
        )
        _exact(
            metric,
            {
                "format",
                "status",
                "split",
                "test_visible",
                "epoch",
                "candidate_checkpoint",
                "subset_manifest",
                "prediction_manifest",
                "trainer_native_candidate",
                "metric_values",
                "trajectory_values_sha256",
                "raw_metric_fingerprint_sha256",
                "receipt_payload_sha256",
            },
            f"derived Base e{epoch} metric",
        )
        expected_metric = {
            "body.released2.metrics.FGD": candidate["fgd"]
        }
        if (
            metric["format"] != orchestrator.MULTICANDIDATE_PROBE_METRIC_FORMAT
            or metric["status"] != "complete"
            or metric["split"] != "val"
            or metric["test_visible"] is not False
            or metric["epoch"] != epoch
            or metric["candidate_checkpoint"] != checkpoint_by_epoch[epoch]
            or metric["subset_manifest"] != binding["subset_manifest"]
            or metric["prediction_manifest"] != candidate["prediction_manifest"]
            or metric["trainer_native_candidate"] != candidate["artifact"]
            or metric["metric_values"] != expected_metric
            or metric["trajectory_values_sha256"]
            != candidate["trajectory_values_sha256"]
            or metric["raw_metric_fingerprint_sha256"]
            != candidate["raw_fingerprint_sha256"]
            or metric_artifact["path"]
            != str(workload_root / f"metric-e{epoch}.json")
            or output
            != {
                "epoch": epoch,
                "candidate_checkpoint": checkpoint_by_epoch[epoch],
                "prediction_manifest": candidate["prediction_manifest"],
                "metric_receipt": metric_artifact,
            }
        ):
            raise ProbeProducerError(
                f"derived Base e{epoch} metric does not replay from raw artifacts"
            )
        candidate_paths = {
            candidate["artifact"]["path"],
            candidate["prediction_manifest"]["path"],
            candidate["inference_lineage"]["path"],
            candidate["full_prediction_manifest"]["path"],
            metric_artifact["path"],
            *(
                artifact["path"]
                for artifact in candidate["prediction_artifacts"]
            ),
            *(
                artifact["path"]
                for artifact in candidate["raw_artifacts"].values()
            ),
        }
        if (
            len(candidate_paths)
            != 9 + binding["clips_per_candidate"]
            or evidence_paths & candidate_paths
        ):
            raise ProbeProducerError("candidate evidence path was reused")
        evidence_paths.update(candidate_paths)
    validated = dict(evidence)
    validated["_quality_common_binding_sha256"] = (
        authority.canonical_json_sha256(
            {
                "source": quality["source"],
                "pipeline": quality["pipeline"],
                "selected_five_authority": quality[
                    "selected_five_authority"
                ],
                "candidate_bundle": quality["candidate_bundle"],
                "val_inputs": quality["val_inputs"],
                "representation_lmdb": quality["representation_lmdb"],
                "training_metrics": quality["training_metrics"],
                "metric_assets": quality["metric_assets"],
                "seed": quality["topology"]["seed"],
                "clips_per_candidate": quality["topology"][
                    "clips_per_candidate"
                ],
                "shards_per_candidate": quality["topology"][
                    "shards_per_candidate"
                ],
            }
        )
    )
    return validated


def _compact_from_cli(args: Any, prefix: str) -> dict[str, Any]:
    path = getattr(args, f"{prefix}_path")
    _resolved, payload, _metadata = _snapshot(path, prefix)
    if _sha256(payload) != _require_sha(getattr(args, f"{prefix}_sha256"), f"{prefix} SHA"):
        raise ProbeProducerError(f"{prefix} SHA changed")
    value = _strict_json(payload, prefix)
    if not isinstance(value, dict):
        raise ProbeProducerError(f"{prefix} must be an object")
    artifact = {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256(payload),
        "bytes": len(payload),
        "receipt_payload_sha256": _require_sha(
            getattr(args, f"{prefix}_payload_sha256"), f"{prefix} payload SHA"
        ),
    }
    _payload_artifact(artifact, prefix)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-and-seal-probe", allow_abbrev=False)
    run.add_argument("--execution-spec-path", type=Path, required=True)
    run.add_argument("--execution-spec-sha256", required=True)
    run.add_argument("--execution-spec-payload-sha256", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--output-json", type=Path, required=True)
    replay = commands.add_parser("replay-probe-run", allow_abbrev=False)
    replay.add_argument("--probe-run-path", type=Path, required=True)
    replay.add_argument("--probe-run-sha256", required=True)
    replay.add_argument("--probe-run-payload-sha256", required=True)
    parsed = parser.parse_args(argv)
    if parsed.command == "run-and-seal-probe":
        spec = _compact_from_cli(parsed, "execution_spec")
        expected_output = parsed.output_root / "probe-run.json"
        if parsed.output_json != expected_output:
            raise ProbeProducerError("probe output JSON must be output-root/probe-run.json")
        run_and_seal_probe(
            spec,
            output_root=parsed.output_root,
            output_json=parsed.output_json,
        )
    else:
        artifact = _compact_from_cli(parsed, "probe_run")
        orchestrator._replay_probe_run(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
