#!/usr/bin/env python3
"""Close the V14 live-validation selection into the formal Base ABI.

The live supervisor publishes an outer
``semtalk_show_base_live_val_22way_selection_v2`` document.  That document is
valuable audit evidence, but it is deliberately not the official long Base
selection consumed by the one-shot final-test authority.  This adapter does
not extract a winner from that outer document.  It freshly revalidates the
e400 reconciliation, rehashes and replays all twenty-two reconciled official
measurements, reruns the official long selector, and requires the nested
official selection to be JSON-equivalent to the rebuilt result.

Two create-new publications result:

* a *pure* ``semtalk_show_base_official_adapt_long_selection_v1`` JSON file;
* a handoff receipt binding the outer selection, e400 reconciliation, all 22
  reconciled measurements, the final candidate bundle, and that pure output.

No test input or test result is accepted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.show_base import base_live_val_consumer_bridge as live_bridge
from scripts.show_base import base_long_val_contract as long_contract
from scripts.show_base import select_base_official_adapt_long as long_selector


OUTER_SELECTION_FORMAT = "semtalk_show_base_live_val_22way_selection_v2"
HANDOFF_FORMAT = "semtalk_show_base_v14_live_selection_handoff_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
PAYLOAD_ARTIFACT_KEYS = frozenset(
    {"path", "sha256", "bytes", "receipt_payload_sha256"}
)
OUTER_KEYS = frozenset(
    {
        "format",
        "status",
        "split",
        "test_visible",
        "selection_eligible",
        "candidate_epochs",
        "reconciliation_receipt",
        "live_measurements",
        "reconciled_measurements",
        "formal_selection",
        "selected",
        "test_evaluations_observed",
        "receipt_payload_sha256",
    }
)
HANDOFF_KEYS = frozenset(
    {
        "format",
        "status",
        "split",
        "test_visible",
        "selection_eligible",
        "test_evaluations_observed",
        "candidate_epochs",
        "outer_selection",
        "reconciliation_receipt",
        "reconciled_measurements",
        "candidate_bundle",
        "formal_selection_output",
        "selected",
        "selection_policy",
        "receipt_payload_sha256",
    }
)


class LiveSelectionHandoffError(RuntimeError):
    """The live selection cannot be closed into the formal selection ABI."""


def canonical_json_bytes(value: Any) -> bytes:
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
        raise LiveSelectionHandoffError("value is not strict JSON") from error


def canonical_json_sha256(value: Any) -> str:
    # Selection/reconciliation payload receipts hash compact JSON without the
    # publication's trailing newline (the file itself remains newline
    # canonical).  This matches both the live bridge and official selector.
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def _payload_sha(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_payload_sha256", None)
    return canonical_json_sha256(unsigned)


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise LiveSelectionHandoffError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LiveSelectionHandoffError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _canonical_file(path_value: Any, label: str) -> Path:
    if type(path_value) is not str or not Path(path_value).is_absolute():
        raise LiveSelectionHandoffError(f"{label} path must be absolute")
    path = Path(path_value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LiveSelectionHandoffError(f"{label} is absent") from error
    if path != resolved:
        raise LiveSelectionHandoffError(f"{label} path is not canonical")
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current /= component
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise LiveSelectionHandoffError(
                    f"{label} contains a symlink component"
                )
    except OSError as error:
        raise LiveSelectionHandoffError(f"cannot attest {label}") from error
    return path


def _snapshot(
    path_value: Any,
    label: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> tuple[Path, bytes]:
    path = _canonical_file(path_value, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LiveSelectionHandoffError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LiveSelectionHandoffError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, key) != getattr(after, key)
            or getattr(before, key) != getattr(current, key)
            for key in stable
        ) or path.resolve(strict=True) != path:
            raise LiveSelectionHandoffError(f"{label} changed while read")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if (
        hashlib.sha256(payload).hexdigest()
        != _sha(expected_sha256, f"{label} SHA-256")
        or len(payload) != _integer(expected_bytes, f"{label} bytes", minimum=1)
    ):
        raise LiveSelectionHandoffError(f"{label} file pin mismatch")
    return path, payload


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in items:
            if key in value:
                raise LiveSelectionHandoffError(
                    f"duplicate JSON key in {label}: {key}"
                )
            value[key] = child
        return value

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LiveSelectionHandoffError(
                    f"non-finite JSON token in {label}: {token}"
                )
            ),
        )
    except LiveSelectionHandoffError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LiveSelectionHandoffError(f"invalid JSON in {label}") from error
    if type(value) is not dict:
        raise LiveSelectionHandoffError(f"{label} must be a JSON object")
    if payload != canonical_json_bytes(value):
        raise LiveSelectionHandoffError(f"{label} is not canonical newline JSON")
    return value


def _artifact(
    value: Any,
    label: str,
    *,
    payload_pin: bool = False,
) -> tuple[dict[str, Any], bytes, dict[str, Any] | None]:
    expected = PAYLOAD_ARTIFACT_KEYS if payload_pin else ARTIFACT_KEYS
    if type(value) is not dict or set(value) != expected:
        raise LiveSelectionHandoffError(f"{label} artifact schema mismatch")
    path, payload = _snapshot(
        value["path"],
        label,
        expected_sha256=value["sha256"],
        expected_bytes=value["bytes"],
    )
    artifact = {
        "path": str(path),
        "sha256": value["sha256"],
        "bytes": value["bytes"],
    }
    parsed: dict[str, Any] | None = None
    if payload_pin:
        parsed = _strict_json(payload, label)
        claimed = _sha(
            value["receipt_payload_sha256"], f"{label} payload SHA-256"
        )
        if parsed.get("receipt_payload_sha256") != claimed or _payload_sha(parsed) != claimed:
            raise LiveSelectionHandoffError(f"{label} payload pin mismatch")
        artifact["receipt_payload_sha256"] = claimed
    return artifact, payload, parsed


def _input_artifact(
    *, path: Path, sha256: str, bytes_count: int, label: str
) -> dict[str, Any]:
    resolved, _payload = _snapshot(
        str(path),
        label,
        expected_sha256=sha256,
        expected_bytes=bytes_count,
    )
    return {"path": str(resolved), "sha256": sha256, "bytes": bytes_count}


def _validate_outer(
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pinned, _payload, outer = _artifact(
        artifact, "live 22-way outer selection", payload_pin=True
    )
    assert outer is not None
    epochs = list(long_contract.EXPECTED_CANDIDATE_EPOCHS)
    if set(outer) != OUTER_KEYS or (
        outer.get("format") != OUTER_SELECTION_FORMAT
        or outer.get("status") != "selected"
        or outer.get("split") != "val"
        or outer.get("test_visible") is not False
        or outer.get("selection_eligible") is not True
        or outer.get("candidate_epochs") != epochs
        or type(outer.get("test_evaluations_observed")) is not int
        or outer["test_evaluations_observed"] != 0
    ):
        raise LiveSelectionHandoffError(
            "outer selection is not the exact val-only 22-way V14 result"
        )
    live = outer.get("live_measurements")
    reconciled = outer.get("reconciled_measurements")
    if (
        not isinstance(live, list)
        or len(live) != len(epochs)
        or any(
            type(item) is not dict or set(item) != PAYLOAD_ARTIFACT_KEYS
            for item in live
        )
        or not isinstance(reconciled, list)
        or len(reconciled) != len(epochs)
        or any(
            type(item) is not dict or set(item) != ARTIFACT_KEYS
            for item in reconciled
        )
        or type(outer.get("formal_selection")) is not dict
        or type(outer.get("selected")) is not dict
    ):
        raise LiveSelectionHandoffError(
            "outer selection evidence inventory is not exact 22-way"
        )
    return pinned, outer


def _candidate_bundle(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(artifacts) is not dict or set(artifacts) != {
        "manifest",
        "status",
        "frozen_inputs",
    }:
        raise LiveSelectionHandoffError("candidate artifact bundle schema mismatch")
    pinned: dict[str, dict[str, Any]] = {}
    for role in ("manifest", "status", "frozen_inputs"):
        item = artifacts[role]
        pinned[role], _payload, _parsed = _artifact(
            item, f"candidate {role}"
        )
    try:
        bundle = long_contract.validate_candidate_bundle(
            manifest_path=Path(pinned["manifest"]["path"]),
            expected_manifest_sha256=pinned["manifest"]["sha256"],
            status_path=Path(pinned["status"]["path"]),
            expected_status_sha256=pinned["status"]["sha256"],
            frozen_inputs_path=Path(pinned["frozen_inputs"]["path"]),
            expected_frozen_inputs_sha256=pinned["frozen_inputs"]["sha256"],
        )
    except Exception as error:
        raise LiveSelectionHandoffError(
            f"candidate bundle fresh replay failed: {error}"
        ) from error
    candidates = bundle.get("candidates")
    epochs = tuple(long_contract.EXPECTED_CANDIDATE_EPOCHS)
    if (
        type(candidates) is not dict
        or tuple(candidates) != epochs
        or bundle.get("candidate_epochs") not in (None, list(epochs))
    ):
        raise LiveSelectionHandoffError(
            "candidate bundle does not cover the formal 22 epochs"
        )
    return pinned, bundle


def _derive_handoff(
    *,
    outer_artifact: Mapping[str, Any],
    candidate_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    outer_pin, outer = _validate_outer(outer_artifact)
    pinned_candidates, candidate_bundle = _candidate_bundle(candidate_artifacts)

    reconciliation_value = outer.get("reconciliation_receipt")
    if type(reconciliation_value) is not dict or set(reconciliation_value) != PAYLOAD_ARTIFACT_KEYS:
        raise LiveSelectionHandoffError("outer reconciliation artifact schema mismatch")
    try:
        reconciliation_pin, reconciliation = live_bridge._validate_reconciliation(
            Path(reconciliation_value["path"]), reconciliation_value["sha256"]
        )
    except Exception as error:
        raise LiveSelectionHandoffError(
            f"e400 reconciliation fresh replay failed: {error}"
        ) from error
    if not _strict_equal(reconciliation_pin, reconciliation_value):
        raise LiveSelectionHandoffError("outer reconciliation artifact changed")
    for role, reconciliation_key in (
        ("manifest", "producer_manifest"),
        ("status", "producer_status"),
        ("frozen_inputs", "frozen_inputs"),
    ):
        bound = reconciliation.get(reconciliation_key)
        if type(bound) is not dict or any(
            bound.get(key) != pinned_candidates[role][key]
            for key in ARTIFACT_KEYS
        ):
            raise LiveSelectionHandoffError(
                f"reconciliation does not bind candidate {role}"
            )

    measurement_values = outer.get("reconciled_measurements")
    epochs = tuple(long_contract.EXPECTED_CANDIDATE_EPOCHS)
    if not isinstance(measurement_values, list) or len(measurement_values) != len(epochs):
        raise LiveSelectionHandoffError(
            "outer selection does not bind exactly 22 reconciled measurements"
        )
    paths: list[Path] = []
    shas: list[str] = []
    pinned_measurements: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for epoch, raw in zip(epochs, measurement_values):
        artifact, _payload, _parsed = _artifact(
            raw, f"reconciled measurement e{epoch}"
        )
        try:
            replayed_artifact, row = long_selector.validate_measurement(
                measurement_path=Path(artifact["path"]),
                expected_measurement_sha256=artifact["sha256"],
                candidate_bundle=candidate_bundle,
            )
        except Exception as error:
            raise LiveSelectionHandoffError(
                f"reconciled measurement e{epoch} fresh replay failed: {error}"
            ) from error
        if (
            any(replayed_artifact.get(key) != artifact[key] for key in ("path", "sha256"))
            or row.get("epoch") != epoch
        ):
            raise LiveSelectionHandoffError(
                f"reconciled measurement e{epoch} identity changed"
            )
        paths.append(Path(artifact["path"]))
        shas.append(artifact["sha256"])
        pinned_measurements.append(artifact)
        rows.append(row)
    if (
        len({item["path"] for item in pinned_measurements}) != len(epochs)
        or len({item["sha256"] for item in pinned_measurements}) != len(epochs)
    ):
        raise LiveSelectionHandoffError(
            "reconciled measurement roots are not exact-once"
        )

    try:
        formal_selection = long_selector.build_selection(
            candidate_bundle=candidate_bundle,
            measurement_paths=paths,
            expected_measurement_sha256=shas,
        )
    except Exception as error:
        raise LiveSelectionHandoffError(
            f"official long selection replay failed: {error}"
        ) from error
    if (
        formal_selection.get("format") != long_contract.SELECTION_FORMAT
        or formal_selection.get("status") != "selected"
        or formal_selection.get("split") != "val"
        or formal_selection.get("test_visible") is not False
        or formal_selection.get("selection_eligible") is not True
        or not _strict_equal(outer.get("formal_selection"), formal_selection)
        or not _strict_equal(outer.get("selected"), formal_selection.get("selected"))
    ):
        raise LiveSelectionHandoffError(
            "outer nested official selection differs from fresh replay"
        )
    candidate_rows = formal_selection.get("candidate_metrics")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(epochs):
        raise LiveSelectionHandoffError("official selection candidate coverage changed")
    ranking: list[tuple[float, int]] = []
    for epoch, row in zip(epochs, candidate_rows):
        metrics = row.get("metrics") if type(row) is dict else None
        fgd = metrics.get(long_contract.PRIMARY_SELECTION_REPORT_KEY) if type(metrics) is dict else None
        if (
            row.get("epoch") != epoch
            or isinstance(fgd, bool)
            or type(fgd) not in (int, float)
            or not math.isfinite(float(fgd))
            or float(fgd) < 0.0
        ):
            raise LiveSelectionHandoffError(f"official e{epoch} FGD row changed")
        ranking.append((float(fgd), epoch))
    winning_fgd, winning_epoch = min(ranking, key=lambda item: (item[0], item[1]))
    selected = formal_selection.get("selected")
    if (
        type(selected) is not dict
        or selected.get("epoch") != winning_epoch
        or float(selected.get("fgd")) != winning_fgd
    ):
        raise LiveSelectionHandoffError(
            "official winner is not strict min(FGD, epoch)"
        )

    bundle_receipt = {
        **pinned_candidates,
        "candidate_epochs": list(epochs),
        "candidate_count": len(epochs),
        "canonical_bundle_sha256": canonical_json_sha256(candidate_bundle),
    }
    return {
        "outer_selection": outer_pin,
        "reconciliation_receipt": reconciliation_pin,
        "reconciled_measurements": pinned_measurements,
        "candidate_bundle": bundle_receipt,
        "formal_selection": formal_selection,
        "selected": dict(selected),
        "selection_policy": dict(formal_selection["selection_policy"]),
    }


def _output_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise LiveSelectionHandoffError(f"{label} must be absolute")
    parent = path.parent.resolve(strict=True)
    if path != parent / path.name or os.path.lexists(path):
        raise LiveSelectionHandoffError(f"{label} must be canonical and absent")
    return path


def _write_new(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = canonical_json_bytes(dict(value))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as error:
        raise LiveSelectionHandoffError(f"refusing to overwrite {path}") from error
    created = True
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short create-new publication write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        if created:
            path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def publish_handoff(
    *,
    outer_artifact: Mapping[str, Any],
    candidate_artifacts: Mapping[str, Mapping[str, Any]],
    formal_selection_output: Path,
    handoff_receipt_output: Path,
) -> dict[str, Any]:
    formal_output = _output_path(formal_selection_output, "formal selection output")
    handoff_output = _output_path(handoff_receipt_output, "handoff receipt output")
    if formal_output == handoff_output:
        raise LiveSelectionHandoffError("formal selection and handoff paths collide")
    derived = _derive_handoff(
        outer_artifact=outer_artifact,
        candidate_artifacts=candidate_artifacts,
    )
    formal_artifact = _write_new(formal_output, derived["formal_selection"])
    formal_artifact["receipt_payload_sha256"] = _sha(
        derived["formal_selection"].get("receipt_payload_sha256"),
        "official formal selection payload SHA-256",
    )
    unsigned = {
        "format": HANDOFF_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "test_evaluations_observed": 0,
        "candidate_epochs": list(long_contract.EXPECTED_CANDIDATE_EPOCHS),
        "outer_selection": derived["outer_selection"],
        "reconciliation_receipt": derived["reconciliation_receipt"],
        "reconciled_measurements": derived["reconciled_measurements"],
        "candidate_bundle": derived["candidate_bundle"],
        "formal_selection_output": formal_artifact,
        "selected": derived["selected"],
        "selection_policy": derived["selection_policy"],
    }
    handoff = {
        **unsigned,
        "receipt_payload_sha256": canonical_json_sha256(unsigned),
    }
    handoff_artifact = _write_new(handoff_output, handoff)
    handoff_artifact["receipt_payload_sha256"] = handoff[
        "receipt_payload_sha256"
    ]
    return {
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "test_evaluations_observed": 0,
        "selected_epoch": derived["selected"]["epoch"],
        "selected_fgd": derived["selected"]["fgd"],
        "formal_selection": formal_artifact,
        "handoff_receipt": handoff_artifact,
    }


def validate_handoff(
    *,
    handoff_artifact: Mapping[str, Any],
    winner_selection: Mapping[str, Any],
    candidate_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    handoff_pin, _raw, handoff = _artifact(
        handoff_artifact, "live-selection handoff", payload_pin=True
    )
    assert handoff is not None
    if set(handoff) != HANDOFF_KEYS or (
        handoff.get("format") != HANDOFF_FORMAT
        or handoff.get("status") != "complete"
        or handoff.get("split") != "val"
        or handoff.get("test_visible") is not False
        or handoff.get("selection_eligible") is not True
        or handoff.get("test_evaluations_observed") != 0
        or handoff.get("candidate_epochs")
        != list(long_contract.EXPECTED_CANDIDATE_EPOCHS)
    ):
        raise LiveSelectionHandoffError("handoff is not exact val-only closure")
    winner_pin, _winner_raw, winner = _artifact(
        winner_selection, "formal winner selection", payload_pin=True
    )
    assert winner is not None
    if not _strict_equal(handoff.get("formal_selection_output"), winner_pin):
        raise LiveSelectionHandoffError(
            "winner_selection is not the handoff formal output"
        )
    derived = _derive_handoff(
        outer_artifact=handoff["outer_selection"],
        candidate_artifacts=candidate_artifacts,
    )
    expected = {
        "format": HANDOFF_FORMAT,
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "selection_eligible": True,
        "test_evaluations_observed": 0,
        "candidate_epochs": list(long_contract.EXPECTED_CANDIDATE_EPOCHS),
        "outer_selection": derived["outer_selection"],
        "reconciliation_receipt": derived["reconciliation_receipt"],
        "reconciled_measurements": derived["reconciled_measurements"],
        "candidate_bundle": derived["candidate_bundle"],
        "formal_selection_output": winner_pin,
        "selected": derived["selected"],
        "selection_policy": derived["selection_policy"],
    }
    expected["receipt_payload_sha256"] = canonical_json_sha256(expected)
    if (
        not _strict_equal(handoff, expected)
        or not _strict_equal(winner, derived["formal_selection"])
    ):
        raise LiveSelectionHandoffError(
            "handoff differs from fresh outer/measurement/selector replay"
        )
    return {
        "handoff": handoff_pin,
        "winner_selection": winner_pin,
        "formal_selection": winner,
        "outer_selection": derived["outer_selection"],
        "reconciliation_receipt": derived["reconciliation_receipt"],
        "reconciled_measurements": derived["reconciled_measurements"],
        "candidate_bundle": derived["candidate_bundle"],
        "selected": derived["selected"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--live-selection-json", type=Path, required=True)
    parser.add_argument("--expected-live-selection-sha256", required=True)
    parser.add_argument("--expected-live-selection-bytes", type=int, required=True)
    parser.add_argument("--expected-live-selection-payload-sha256", required=True)
    parser.add_argument("--base-candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-candidate-manifest-sha256", required=True)
    parser.add_argument("--expected-base-candidate-manifest-bytes", type=int, required=True)
    parser.add_argument("--base-status-json", type=Path, required=True)
    parser.add_argument("--expected-base-formal-status-sha256", required=True)
    parser.add_argument("--expected-base-formal-status-bytes", type=int, required=True)
    parser.add_argument("--base-frozen-inputs-json", type=Path, required=True)
    parser.add_argument("--expected-base-frozen-inputs-sha256", required=True)
    parser.add_argument("--expected-base-frozen-inputs-bytes", type=int, required=True)
    parser.add_argument("--formal-selection-output", type=Path, required=True)
    parser.add_argument("--handoff-receipt-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outer = {
        "path": str(args.live_selection_json),
        "sha256": args.expected_live_selection_sha256,
        "bytes": args.expected_live_selection_bytes,
        "receipt_payload_sha256": args.expected_live_selection_payload_sha256,
    }
    candidates = {
        "manifest": _input_artifact(
            path=args.base_candidate_manifest,
            sha256=args.expected_base_candidate_manifest_sha256,
            bytes_count=args.expected_base_candidate_manifest_bytes,
            label="candidate manifest",
        ),
        "status": _input_artifact(
            path=args.base_status_json,
            sha256=args.expected_base_formal_status_sha256,
            bytes_count=args.expected_base_formal_status_bytes,
            label="candidate status",
        ),
        "frozen_inputs": _input_artifact(
            path=args.base_frozen_inputs_json,
            sha256=args.expected_base_frozen_inputs_sha256,
            bytes_count=args.expected_base_frozen_inputs_bytes,
            label="candidate frozen inputs",
        ),
    }
    result = publish_handoff(
        outer_artifact=outer,
        candidate_artifacts=candidates,
        formal_selection_output=args.formal_selection_output,
        handoff_receipt_output=args.handoff_receipt_output,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
