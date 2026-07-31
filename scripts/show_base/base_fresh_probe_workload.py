#!/usr/bin/env python3
"""Execute the real fresh SemTalk Base validation concurrency workload.

The outer producer measures this program through the guarded eight-GPU
runner.  This program has no metric/timing override surface: it invokes the
source-bound ``run_base_val_inference.py`` prepare/shard/finalize commands,
validates their complete 1,715-clip lineages, projects the frozen first 64
validation clips, and extracts the pinned TalkSHOW released2 raw features.
Only those replayable raw artifacts are exposed to the outer producer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_fresh_probe_producer as producer
from scripts.show_base import base_fresh_val_orchestrator as orchestrator
from scripts.show_base import evaluate_talkshow_show_metrics as metrics
from scripts.show_base import published_test_winner_claim as authority


WORKLOAD_EXECUTION_FORMAT = "semtalk_show_base_probe_workload_execution_v1"
EXPECTED_GPUS = tuple(range(8))
PROGRESS_EVERY = 20
PROC_FD_PATTERN = re.compile(r"/proc/([1-9][0-9]*)/fd/([0-9]+)")


class ProbeWorkloadError(RuntimeError):
    """The real Base probe could not be completed and sealed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise ProbeWorkloadError("non-canonical workload JSON") from error


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) for row in rows)


def _strict_json(payload: bytes, label: str) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProbeWorkloadError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProbeWorkloadError(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeWorkloadError(f"{label} is not strict JSON") from error


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProbeWorkloadError(f"{label} schema changed")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProbeWorkloadError(f"{label} must be 64 lowercase hex")
    return value


def _parent_pid(pid: int) -> int:
    try:
        line = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError as error:
        raise ProbeWorkloadError(f"cannot read process {pid} identity") from error
    close = line.rfind(")")
    fields = line[close + 2 :].split() if close >= 0 else []
    if len(fields) < 2 or not fields[1].isdigit():
        raise ProbeWorkloadError(f"process {pid} identity changed")
    return int(fields[1])


def _read_pinned_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_payload_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Read the producer's sealed snapshot, including its proc-fd form."""

    expected_sha = _require_sha(expected_sha256, f"{label} SHA-256")
    expected_payload = _require_sha(
        expected_payload_sha256, f"{label} payload SHA-256"
    )
    raw = str(path)
    match = PROC_FD_PATTERN.fullmatch(raw)
    if sys.platform == "linux":
        if match is None:
            raise ProbeWorkloadError(f"formal {label} is not a sealed proc fd")
        runner_pid = os.getppid()
        producer_pid = _parent_pid(runner_pid)
        if int(match.group(1)) != producer_pid:
            raise ProbeWorkloadError(f"{label} proc fd is not owned by producer")
    elif match is not None:
        raise ProbeWorkloadError(f"non-Linux {label} cannot use proc fd")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if match is None:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ProbeWorkloadError(f"{label} fallback is not a regular file")
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(raw, flags)
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode):
            raise ProbeWorkloadError(f"{label} snapshot is not regular")
        if match is not None:
            get_seals = getattr(fcntl, "F_GET_SEALS", None)
            expected_seals = (
                getattr(fcntl, "F_SEAL_SEAL", 0)
                | getattr(fcntl, "F_SEAL_SHRINK", 0)
                | getattr(fcntl, "F_SEAL_GROW", 0)
                | getattr(fcntl, "F_SEAL_WRITE", 0)
            )
            if (
                get_seals is None
                or not expected_seals
                or fcntl.fcntl(descriptor, get_seals) != expected_seals
            ):
                raise ProbeWorkloadError(f"{label} proc fd is not fully sealed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before_fd, key) != getattr(after_fd, key) for key in stable):
        raise ProbeWorkloadError(f"{label} snapshot changed while reading")
    if _sha256(payload) != expected_sha:
        raise ProbeWorkloadError(f"{label} bytes differ from producer pin")
    decoded = _strict_json(payload, label)
    if not isinstance(decoded, dict):
        raise ProbeWorkloadError(f"{label} is not an object")
    claimed = decoded.get("receipt_payload_sha256")
    body = dict(decoded)
    body.pop("receipt_payload_sha256", None)
    if (
        claimed != expected_payload
        or authority.canonical_json_sha256(body) != expected_payload
    ):
        raise ProbeWorkloadError(f"{label} payload binding changed")
    return decoded


def _canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ProbeWorkloadError(f"{label} must be absolute/non-symlink")
    try:
        mode = os.lstat(path).st_mode
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProbeWorkloadError(f"{label} is unavailable") from error
    if not stat.S_ISDIR(mode) or resolved != path:
        raise ProbeWorkloadError(f"{label} must be a canonical directory")
    return resolved


def _write_bytes_new(path: Path, payload: bytes) -> dict[str, Any]:
    if not path.is_absolute():
        raise ProbeWorkloadError("workload output must be absolute")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise ProbeWorkloadError("short workload artifact write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    artifact, observed = _artifact_payload(
        path, "new workload artifact", allow_empty=True
    )
    if observed != payload:
        raise ProbeWorkloadError("workload artifact changed after write")
    return artifact


def _write_payload_new(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    payload = orchestrator._with_payload_sha(body)
    artifact = _write_bytes_new(path, _canonical_bytes(payload))
    return {**artifact, "receipt_payload_sha256": payload["receipt_payload_sha256"]}


def _artifact_payload(
    path: Path, label: str, *, allow_empty: bool = False
) -> tuple[dict[str, Any], bytes]:
    try:
        resolved, payload, _metadata = producer._snapshot(path, label)
    except (OSError, producer.ProbeProducerError) as error:
        raise ProbeWorkloadError(f"{label} is unavailable or changed") from error
    if not payload and not allow_empty:
        raise ProbeWorkloadError(f"{label} changed")
    return (
        {
            "path": str(resolved),
            "sha256": _sha256(payload),
            "bytes": len(payload),
        },
        payload,
    )


def _artifact(path: Path, label: str, *, allow_empty: bool = False) -> dict[str, Any]:
    artifact, _payload = _artifact_payload(path, label, allow_empty=allow_empty)
    return artifact


def _same_compact_artifact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = {"path", "sha256", "receipt_payload_sha256"}
    return all(left.get(key) == right.get(key) for key in keys)


def _payload_artifact(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, artifact_payload = _artifact_payload(path, label)
    payload = _strict_json(artifact_payload, label)
    if not isinstance(payload, dict):
        raise ProbeWorkloadError(f"{label} is not an object")
    claimed = _require_sha(
        payload.get("receipt_payload_sha256"), f"{label} payload SHA-256"
    )
    body = dict(payload)
    body.pop("receipt_payload_sha256", None)
    if authority.canonical_json_sha256(body) != claimed:
        raise ProbeWorkloadError(f"{label} payload does not replay")
    return {**artifact, "receipt_payload_sha256": claimed}, payload


@dataclass
class _StartedCommand:
    argv: list[str]
    stdout_path: Path
    stderr_path: Path
    stdout_handle: Any
    stderr_handle: Any
    process: subprocess.Popen[bytes]


def _start_command(argv: Sequence[str], *, stdout_path: Path, stderr_path: Path) -> _StartedCommand:
    if any(not isinstance(item, str) or not item or "\0" in item for item in argv):
        raise ProbeWorkloadError("invalid command argv")
    stdout_handle = stdout_path.open("xb")
    try:
        stderr_handle = stderr_path.open("xb")
    except BaseException:
        stdout_handle.close()
        raise
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            start_new_session=False,
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return _StartedCommand(
        argv=list(argv),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        process=process,
    )


def _close_command(command: _StartedCommand) -> None:
    for handle in (command.stdout_handle, command.stderr_handle):
        if not handle.closed:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()


def _stop_owned(commands: Sequence[_StartedCommand]) -> None:
    live = [command for command in commands if command.process.poll() is None]
    for command in live:
        command.process.send_signal(signal.SIGTERM)
    for command in live:
        try:
            command.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            command.process.send_signal(signal.SIGKILL)
            command.process.wait()
    for command in commands:
        _close_command(command)


def _finish_commands(commands: Sequence[_StartedCommand]) -> list[dict[str, Any]]:
    try:
        return_codes = [command.process.wait() for command in commands]
    except BaseException:
        _stop_owned(commands)
        raise
    finally:
        for command in commands:
            _close_command(command)
    records: list[dict[str, Any]] = []
    for command, return_code in zip(commands, return_codes):
        stdout = _artifact(command.stdout_path, "command stdout", allow_empty=True)
        stderr = _artifact(command.stderr_path, "command stderr", allow_empty=True)
        record = {
            "argv": command.argv,
            "argv_sha256": authority.canonical_json_sha256(command.argv),
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
        }
        records.append(record)
    if any(record["return_code"] != 0 for record in records):
        raise ProbeWorkloadError("real Base validation command failed")
    return records


def _run_one(argv: Sequence[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    return _finish_commands(
        [_start_command(argv, stdout_path=stdout_path, stderr_path=stderr_path)]
    )[0]


def _prepare_argv(
    python: str,
    entrypoint: str,
    *,
    quality: Mapping[str, Any],
    binding: Mapping[str, Any],
    output: Path,
) -> list[str]:
    bundle = quality["candidate_bundle"]
    return [
        python,
        entrypoint,
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
        str(output),
    ]


def _shard_argv(
    python: str,
    entrypoint: str,
    *,
    preflight: Mapping[str, Any],
    epoch: int,
    output_root: Path,
    shard_id: int,
    seed: int,
) -> list[str]:
    return [
        python,
        entrypoint,
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
        str(seed),
        "--progress-every",
        str(PROGRESS_EVERY),
    ]


def _finalize_argv(
    python: str,
    entrypoint: str,
    *,
    preflight: Mapping[str, Any],
    epoch: int,
    output_root: Path,
) -> list[str]:
    return [
        python,
        entrypoint,
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


def _load_jsonl_artifact(value: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    artifact = _exact(value, {"path", "sha256", "bytes"}, label)
    path = Path(artifact["path"])
    observed, payload = _artifact_payload(path, label)
    if observed != artifact:
        raise ProbeWorkloadError(f"{label} changed")
    return authority._strict_jsonl_bytes(payload, label)


def _validate_full_lineage(
    *,
    lineage: Mapping[str, Any],
    epoch: int,
    checkpoint: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
                Path(lineage["path"]),
                lineage["sha256"],
                epoch=epoch,
                expected_candidate=checkpoint,
                val_inputs_artifact=val_artifact,
                pipeline_artifact=pipeline_artifact,
                expected_coverage=coverage,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ProbeWorkloadError(
            f"Base e{epoch} full inference lineage failed replay"
        ) from error
    if any(audited_lineage.get(key) != lineage.get(key) for key in ("path", "sha256", "receipt_payload_sha256")):
        raise ProbeWorkloadError(f"Base e{epoch} lineage artifact changed")
    full_manifest, manifest_payload = _artifact_payload(
        Path(validated["final_manifest"]["path"]),
        f"Base e{epoch} full prediction manifest",
    )
    if full_manifest["sha256"] != validated["final_manifest"]["sha256"]:
        raise ProbeWorkloadError(f"Base e{epoch} full manifest changed")
    rows = authority._strict_jsonl_bytes(
        manifest_payload,
        f"Base e{epoch} full prediction manifest",
    )
    if len(rows) != orchestrator.EXPECTED_CLIPS:
        raise ProbeWorkloadError(f"Base e{epoch} full manifest coverage changed")
    return full_manifest, rows


def _project_predictions(
    *,
    rows: Sequence[Mapping[str, Any]],
    subset_rows: Sequence[Mapping[str, Any]],
    epoch: int,
    checkpoint: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    projected: list[dict[str, Any]] = []
    for position, (full, subset) in enumerate(
        zip(rows[: len(subset_rows)], subset_rows)
    ):
        expected_id = subset["canonical_clip_id"]
        prediction = full.get("prediction")
        if (
            full.get("canonical_clip_id") != expected_id
            or full.get("epoch") != epoch
            or full.get("candidate_checkpoint_sha256") != checkpoint["sha256"]
            or not isinstance(prediction, dict)
            or set(prediction) != {"path", "sha256", "bytes"}
        ):
            raise ProbeWorkloadError(
                f"Base e{epoch} projected prediction {position} changed"
            )
        observed = _artifact(
            Path(prediction["path"]),
            f"Base e{epoch} projected prediction {position}",
        )
        if observed != prediction:
            raise ProbeWorkloadError(
                f"Base e{epoch} projected prediction bytes changed"
            )
        projected.append(
            {"canonical_clip_id": expected_id, "prediction": prediction}
        )
    if len(projected) != len(subset_rows):
        raise ProbeWorkloadError(f"Base e{epoch} projection coverage changed")
    return _write_bytes_new(output, _canonical_jsonl(projected)), projected


def _load_training_metrics(quality: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _load_jsonl_artifact(quality["training_metrics"], "trainer epoch metrics")
    if len(rows) != 400:
        raise ProbeWorkloadError("trainer epoch-metric coverage changed")
    for expected_epoch, row in enumerate(rows, 1):
        if (
            row.get("format") != "semtalk_show_base_long_epoch_metric_v1"
            or row.get("epoch") != expected_epoch
            or type(row.get("optimizer_updates")) is not int
            or not isinstance(row.get("metrics"), dict)
            or type(row["metrics"].get("total")) not in (int, float)
            or type(row["metrics"].get("total")) is bool
            or not math.isfinite(float(row["metrics"]["total"]))
            or row.get("all_finite") is not True
        ):
            raise ProbeWorkloadError(
                f"trainer epoch metric e{expected_epoch} changed"
            )
    return rows


def _trajectory_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    epoch: int,
    output: Path,
) -> dict[str, Any]:
    return _write_bytes_new(
        output,
        _canonical_jsonl(
            {
                "optimizer_update": row["optimizer_updates"],
                "loss": float(row["metrics"]["total"]),
            }
            for row in rows[:epoch]
        ),
    )


def _npy_artifact(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return _write_bytes_new(path, buffer.getvalue())


def _extract_features(
    *,
    backend: metrics.TalkShowCudaMetricBackend,
    canonical_rows: Sequence[Mapping[str, Any]],
    full_rows_by_epoch: Mapping[int, Sequence[Mapping[str, Any]]],
    epochs: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], list[dict[str, Any]]]:
    canonical_arrays: list[dict[str, np.ndarray]] = []
    real_chunks: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    real_cursor = 0
    generated_cursor = 0
    for position, row in enumerate(canonical_rows):
        arrays = metrics._load_canonical_npz(
            row["canonical_npz"],
            expected_sha256=row["canonical_npz_sha256"],
            frames=row["frames"],
            speaker_id=row["speaker_id"],
        )
        canonical_arrays.append(arrays)
        real = metrics._validated_backend_features(
            backend,
            metrics.reorder_to_talkshow(
                arrays["pose"], arrays["facial"]
            )[None],
        )
        if real.ndim != 2 or real.shape[0] != 1:
            raise ProbeWorkloadError(
                f"released2 real feature cardinality changed at {position}"
            )
        real_chunks.append(real)
        feature_rows.append(
            {
                "canonical_clip_id": orchestrator.val_contract.canonical_clip_id(
                    row["clip_id"]
                ),
                "real_start": real_cursor,
                "real_rows": 1,
                "generated_start": generated_cursor,
                "generated_rows": len(metrics.RELEASED2_SLOTS),
            }
        )
        real_cursor += 1
        generated_cursor += len(metrics.RELEASED2_SLOTS)
    generated_by_epoch: dict[int, np.ndarray] = {}
    for epoch in epochs:
        generated_chunks: list[np.ndarray] = []
        for position, (canonical, output_row) in enumerate(
            zip(canonical_arrays, full_rows_by_epoch[epoch])
        ):
            output_id = feature_rows[position]["canonical_clip_id"]
            prediction = metrics._load_output_npz(
                output_row["prediction"],
                frames=canonical_rows[position]["frames"],
                prediction=True,
                expected_name=f"res_{output_id}.npz",
            )
            if not np.array_equal(prediction["betas"], canonical["beta"][0]):
                raise ProbeWorkloadError(
                    f"Base e{epoch} prediction betas changed at {position}"
                )
            generated = metrics._validated_backend_features(
                backend,
                np.repeat(
                    metrics.reorder_to_talkshow(
                        prediction["poses"], prediction["expressions"]
                    )[None],
                    len(metrics.RELEASED2_SLOTS),
                    axis=0,
                ),
            )
            if generated.ndim != 2 or generated.shape[0] != len(metrics.RELEASED2_SLOTS):
                raise ProbeWorkloadError(
                    f"Base e{epoch} released2 feature cardinality changed"
                )
            generated_chunks.append(generated)
        generated_by_epoch[epoch] = np.concatenate(generated_chunks, axis=0)
    return np.concatenate(real_chunks, axis=0), generated_by_epoch, feature_rows


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = _read_pinned_receipt(
        args.execution_spec_path,
        expected_sha256=args.execution_spec_sha256,
        expected_payload_sha256=args.execution_spec_payload_sha256,
        label="execution spec",
    )
    quality = _read_pinned_receipt(
        args.quality_input_path,
        expected_sha256=args.quality_input_sha256,
        expected_payload_sha256=args.quality_input_payload_sha256,
        label="quality input",
    )
    _exact(
        spec,
        {
            "format", "status", "split", "test_visible", "formal_host",
            "candidates_per_wave", "execution_mode", "probe_binding",
            "quality_input", "source_root", "python", "quality_producer",
            "runner", "nvidia_smi", "receipt_payload_sha256",
        },
        "execution spec",
    )
    binding = orchestrator._probe_binding(spec["probe_binding"])
    quality_artifact = spec["quality_input"]
    if (
        spec["format"] != producer.EXECUTION_SPEC_FORMAT
        or spec["status"] != "frozen"
        or spec["split"] != "val"
        or spec["test_visible"] is not False
        or spec["formal_host"] != args.formal_host
        or spec["candidates_per_wave"] != args.candidates_per_wave
        or spec["execution_mode"] != args.execution_mode
        or quality_artifact.get("sha256") != args.quality_input_sha256
        or quality_artifact.get("receipt_payload_sha256")
        != args.quality_input_payload_sha256
        or args.execution_mode not in {"serial", "concurrent"}
        or (args.execution_mode == "serial" and args.candidates_per_wave != 1)
    ):
        raise ProbeWorkloadError("execution spec CLI binding changed")
    normalized_quality_artifact, public_quality = producer._validate_quality_input(
        quality_artifact,
        binding=binding,
        mode=args.candidates_per_wave,
        execution_mode=args.execution_mode,
    )
    public_quality = {
        key: value
        for key, value in public_quality.items()
        if key != "_audited_candidate_bundle"
    }
    if normalized_quality_artifact != quality_artifact or public_quality != quality:
        raise ProbeWorkloadError("sealed/public quality input binding changed")
    return spec, quality, binding, normalized_quality_artifact


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    spec, quality, binding, quality_artifact = _validate_inputs(args)
    output_root = _canonical_directory(args.output_root, "workload output root")
    if args.candidate_ready_output != output_root / "candidate-ready.json":
        raise ProbeWorkloadError("candidate-ready output escaped workload root")
    if any(output_root.iterdir()):
        raise ProbeWorkloadError("workload output root is not create-new empty")
    commands_root = output_root / "commands"
    commands_root.mkdir()
    python = spec["python"]["path"]
    pipeline_artifact, pipeline = orchestrator.val_contract.validate_pipeline(
        Path(binding["pipeline"]["path"]), binding["pipeline"]["sha256"]
    )
    if not _same_compact_artifact(pipeline_artifact, binding["pipeline"]):
        raise ProbeWorkloadError("pipeline artifact changed")
    entrypoint = pipeline["inference_entrypoint"]["path"]
    if Path(entrypoint) != Path(spec["source_root"]) / "scripts/show_base/run_base_val_inference.py":
        raise ProbeWorkloadError("real Base validation entrypoint changed")

    preflight_path = output_root / "preflight.json"
    prepare_argv = _prepare_argv(
        python,
        entrypoint,
        quality=quality,
        binding=binding,
        output=preflight_path,
    )
    prepare_record = _run_one(
        prepare_argv,
        stdout_path=commands_root / "prepare.stdout",
        stderr_path=commands_root / "prepare.stderr",
    )
    preflight_artifact, _preflight = _payload_artifact(
        preflight_path, "real Base preflight"
    )
    prepare_record["output"] = preflight_artifact

    checkpoint_by_epoch = {
        row["epoch"]: row["candidate_checkpoint"]
        for row in binding["candidate_checkpoints"]
    }
    candidate_execution: list[dict[str, Any]] = []
    lineage_by_epoch: dict[int, dict[str, Any]] = {}
    full_manifest_by_epoch: dict[int, dict[str, Any]] = {}
    full_rows_by_epoch: dict[int, list[dict[str, Any]]] = {}
    epochs = list(orchestrator.PROBE_EPOCHS)
    for wave_start in range(0, len(epochs), args.candidates_per_wave):
        wave = epochs[wave_start : wave_start + args.candidates_per_wave]
        shard_commands: list[_StartedCommand] = []
        shard_metadata: list[tuple[int, int]] = []
        for epoch in wave:
            inference_root = output_root / f"e{epoch}" / "inference"
            for shard_id in EXPECTED_GPUS:
                argv = _shard_argv(
                    python,
                    entrypoint,
                    preflight=preflight_artifact,
                    epoch=epoch,
                    output_root=inference_root,
                    shard_id=shard_id,
                    seed=binding["seed"],
                )
                shard_commands.append(
                    _start_command(
                        argv,
                        stdout_path=commands_root / f"e{epoch}-shard-{shard_id}.stdout",
                        stderr_path=commands_root / f"e{epoch}-shard-{shard_id}.stderr",
                    )
                )
                shard_metadata.append((epoch, shard_id))
        shard_records = _finish_commands(shard_commands)
        by_epoch: dict[int, list[dict[str, Any]]] = {epoch: [] for epoch in wave}
        for record, (epoch, shard_id) in zip(shard_records, shard_metadata):
            by_epoch[epoch].append(
                {"shard_id": shard_id, "device": f"cuda:{shard_id}", **record}
            )

        finalizers: list[_StartedCommand] = []
        for epoch in wave:
            inference_root = output_root / f"e{epoch}" / "inference"
            finalizers.append(
                _start_command(
                    _finalize_argv(
                        python,
                        entrypoint,
                        preflight=preflight_artifact,
                        epoch=epoch,
                        output_root=inference_root,
                    ),
                    stdout_path=commands_root / f"e{epoch}-finalize.stdout",
                    stderr_path=commands_root / f"e{epoch}-finalize.stderr",
                )
            )
        finalize_records = _finish_commands(finalizers)
        for epoch, finalize_record in zip(wave, finalize_records):
            lineage_path = output_root / f"e{epoch}" / "inference" / "final" / "lineage.json"
            lineage_artifact, _lineage = _payload_artifact(
                lineage_path, f"Base e{epoch} real inference lineage"
            )
            full_manifest, full_rows = _validate_full_lineage(
                lineage=lineage_artifact,
                epoch=epoch,
                checkpoint=checkpoint_by_epoch[epoch],
                binding=binding,
            )
            lineage_by_epoch[epoch] = lineage_artifact
            full_manifest_by_epoch[epoch] = full_manifest
            full_rows_by_epoch[epoch] = full_rows
            finalize_record["output"] = lineage_artifact
            candidate_execution.append(
                {
                    "epoch": epoch,
                    "candidate_checkpoint": checkpoint_by_epoch[epoch],
                    "shards": by_epoch[epoch],
                    "finalize": finalize_record,
                    "inference_lineage": lineage_artifact,
                    "full_prediction_manifest": full_manifest,
                }
            )

    candidate_execution.sort(key=lambda row: row["epoch"])
    val_artifact, val_coverage = orchestrator.val_contract.validate_val_inputs(
        Path(binding["val_inputs"]["path"]), binding["val_inputs"]["sha256"]
    )
    if not _same_compact_artifact(val_artifact, binding["val_inputs"]):
        raise ProbeWorkloadError("validation input artifact changed")
    canonical_manifest = val_coverage["canonical_manifest"]
    canonical_observed, canonical_payload = _artifact_payload(
        Path(canonical_manifest["path"]),
        "canonical SHOW validation manifest",
    )
    if canonical_observed != canonical_manifest:
        raise ProbeWorkloadError("canonical SHOW validation manifest changed")
    canonical_rows = authority._strict_jsonl_bytes(
        canonical_payload,
        "canonical SHOW validation manifest",
    )
    subset_rows = _load_jsonl_artifact(binding["subset_manifest"], "probe subset")
    selected_canonical = canonical_rows[: len(subset_rows)]
    normalized_prefix = [
        {
            "global_index": row["global_index"],
            "source_clip_id": row["clip_id"],
            "canonical_clip_id": orchestrator.val_contract.canonical_clip_id(row["clip_id"]),
            "frames": row["frames"],
            "split": "val",
        }
        for row in selected_canonical
    ]
    if subset_rows != normalized_prefix or len(subset_rows) != orchestrator.PROBE_CLIPS_PER_CANDIDATE:
        raise ProbeWorkloadError("probe subset is not frozen first-64 val")

    projected_artifacts: dict[int, dict[str, Any]] = {}
    projected_rows: dict[int, list[dict[str, Any]]] = {}
    for epoch in epochs:
        projected_artifacts[epoch], projected_rows[epoch] = _project_predictions(
            rows=full_rows_by_epoch[epoch],
            subset_rows=subset_rows,
            epoch=epoch,
            checkpoint=checkpoint_by_epoch[epoch],
            output=output_root / f"e{epoch}" / "prediction-manifest-first64.jsonl",
        )

    metric_assets = quality["metric_assets"]
    backend = metrics.TalkShowCudaMetricBackend(
        talkshow_root=metric_assets["talkshow_metric_root"],
        feature_extractor=metric_assets["feature_extractor"]["path"],
        smplx_asset=metric_assets["smplx_asset"]["path"],
        device="cuda:0",
        torch_threads=1,
    )
    real_features, generated_by_epoch, feature_rows = _extract_features(
        backend=backend,
        canonical_rows=selected_canonical,
        full_rows_by_epoch={
            epoch: full_rows_by_epoch[epoch][: len(subset_rows)] for epoch in epochs
        },
        epochs=epochs,
    )
    training_rows = _load_training_metrics(quality)

    native_artifacts: list[dict[str, Any]] = []
    for epoch in epochs:
        candidate_root = output_root / f"e{epoch}"
        real_artifact = _npy_artifact(candidate_root / "released2-real-features.npy", real_features)
        generated_artifact = _npy_artifact(
            candidate_root / "released2-generated-features.npy",
            generated_by_epoch[epoch],
        )
        feature_manifest = _write_bytes_new(
            candidate_root / "released2-feature-manifest.jsonl",
            _canonical_jsonl(feature_rows),
        )
        trajectory = _trajectory_artifact(
            rows=training_rows,
            epoch=epoch,
            output=candidate_root / "epoch-mean-training-trajectory.jsonl",
        )
        native_artifacts.append(
            _write_payload_new(
                candidate_root / "trainer-native-candidate.json",
                {
                    "format": producer.TRAINER_NATIVE_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": epoch,
                    "candidate_checkpoint": checkpoint_by_epoch[epoch],
                    "subset_manifest": binding["subset_manifest"],
                    "prediction_manifest": projected_artifacts[epoch],
                    "quality_input": quality_artifact,
                    "inference_lineage": lineage_by_epoch[epoch],
                    "real_features": real_artifact,
                    "generated_features": generated_artifact,
                    "feature_manifest": feature_manifest,
                    "training_trajectory": trajectory,
                },
            )
        )

    execution_artifact = _write_payload_new(
        output_root / "workload-execution.json",
        {
            "format": WORKLOAD_EXECUTION_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "formal_host": args.formal_host,
            "candidates_per_wave": args.candidates_per_wave,
            "execution_mode": args.execution_mode,
            "probe_binding_sha256": authority.canonical_json_sha256(binding),
            "quality_input": quality_artifact,
            "prepare": prepare_record,
            "candidates": candidate_execution,
        },
    )
    candidate_ready = _write_payload_new(
        args.candidate_ready_output,
        {
            "format": producer.CANDIDATE_READY_FORMAT,
            "status": "complete",
            "split": "val",
            "test_visible": False,
            "formal_host": args.formal_host,
            "candidates_per_wave": args.candidates_per_wave,
            "execution_mode": args.execution_mode,
            "probe_binding_sha256": authority.canonical_json_sha256(binding),
            "quality_input": quality_artifact,
            "workload_execution": execution_artifact,
            "candidates": native_artifacts,
        },
    )
    return candidate_ready


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-probe", allow_abbrev=False)
    run.add_argument("--execution-spec-path", type=Path, required=True)
    run.add_argument("--execution-spec-sha256", required=True)
    run.add_argument("--execution-spec-payload-sha256", required=True)
    run.add_argument("--quality-input-path", type=Path, required=True)
    run.add_argument("--quality-input-sha256", required=True)
    run.add_argument("--quality-input-payload-sha256", required=True)
    run.add_argument("--formal-host", required=True)
    run.add_argument(
        "--candidates-per-wave",
        type=int,
        choices=orchestrator.CONCURRENCY_SELECTION_ORDER,
        required=True,
    )
    run.add_argument(
        "--execution-mode", choices=("serial", "concurrent"), required=True
    )
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--candidate-ready-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "run-probe":  # pragma: no cover
        raise AssertionError(args.command)
    result = run_probe(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
