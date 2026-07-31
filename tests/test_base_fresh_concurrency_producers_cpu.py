from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.show_base import base_fresh_val_orchestrator as ORCHESTRATOR
from scripts.show_base import published_test_winner_claim as AUTHORITY


def _canonical_bytes(value: object) -> bytes:
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


def _write_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _write_json(path: Path, value: object) -> dict[str, object]:
    return _write_bytes(path, _canonical_bytes(value))


def _write_jsonl(
    path: Path, rows: list[dict[str, object]]
) -> dict[str, object]:
    payload = b"".join(_canonical_bytes(row) for row in rows)
    return _write_bytes(path, payload)


def _write_receipt(
    path: Path, value: dict[str, object]
) -> dict[str, object]:
    receipt = copy.deepcopy(value)
    receipt.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = AUTHORITY.canonical_json_sha256(
        receipt
    )
    artifact = _write_json(path, receipt)
    artifact["receipt_payload_sha256"] = receipt[
        "receipt_payload_sha256"
    ]
    return artifact


def _payload_args(prefix: str, artifact: dict[str, object]) -> list[str]:
    return [
        f"--{prefix}-path",
        str(artifact["path"]),
        f"--{prefix}-sha256",
        str(artifact["sha256"]),
        f"--{prefix}-payload-sha256",
        str(artifact["receipt_payload_sha256"]),
    ]


def _plain_args(prefix: str, artifact: dict[str, object]) -> list[str]:
    return [
        f"--{prefix}-path",
        str(artifact["path"]),
        f"--{prefix}-sha256",
        str(artifact["sha256"]),
        f"--{prefix}-bytes",
        str(artifact["bytes"]),
    ]


class ProbeFixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.source = {
            "origin": AUTHORITY.EXPECTED_ORIGIN,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        self.pipeline = _write_receipt(
            self.root / "pipeline.json",
            {
                "format": "semtalk_show_base_probe_pipeline_fixture_v1",
                "status": "frozen",
                "split": "val",
                "test_visible": False,
                "generator_module": ORCHESTRATOR.GENERATOR_MODULE,
            },
        )
        self.prerequisite = _write_receipt(
            self.root / "prerequisite.json",
            {
                "format": "semtalk_show_base_probe_prerequisite_fixture_v1",
                "status": "selected",
                "target_dataset": "SHOW",
                "target_speaker_scope": AUTHORITY.EXPECTED_SCOPE,
                "split": "val",
                "test_visible": False,
            },
        )
        self.val_inputs = _write_receipt(
            self.root / "val-inputs.json",
            {
                "format": "semtalk_show_base_probe_val_inputs_fixture_v1",
                "status": "frozen",
                "dataset": "SHOW",
                "target_speaker_scope": AUTHORITY.EXPECTED_SCOPE,
                "split": "val",
                "test_visible": False,
            },
        )
        self.subset_rows = [
            {
                "canonical_clip_id": f"all-speakers-val-{index:03d}",
                "split": "val",
            }
            for index in range(ORCHESTRATOR.PROBE_CLIPS_PER_CANDIDATE)
        ]
        self.subset = _write_jsonl(
            self.root / "probe-subset.jsonl", self.subset_rows
        )
        self.checkpoints = {
            epoch: _write_bytes(
                self.root / f"base-e{epoch}.pth",
                f"official SemTalk Base SHOW e{epoch}\n".encode("ascii"),
            )
            for epoch in ORCHESTRATOR.PROBE_EPOCHS
        }
        self.binding = {
            "source": copy.deepcopy(self.source),
            "pipeline": copy.deepcopy(self.pipeline),
            "prerequisite_selection": copy.deepcopy(self.prerequisite),
            "val_inputs": copy.deepcopy(self.val_inputs),
            "candidate_checkpoints": [
                {
                    "epoch": epoch,
                    "candidate_checkpoint": copy.deepcopy(
                        self.checkpoints[epoch]
                    ),
                }
                for epoch in ORCHESTRATOR.PROBE_EPOCHS
            ],
            "subset_manifest": copy.deepcopy(self.subset),
            "seed": 20260731,
            "clips_per_candidate": (
                ORCHESTRATOR.PROBE_CLIPS_PER_CANDIDATE
            ),
            "shards_per_candidate": AUTHORITY.EXPECTED_SHARDS,
        }

    def candidate_outputs(
        self, tag: str, *, changed_prediction: bool = False
    ) -> list[dict[str, object]]:
        outputs = []
        for epoch in ORCHESTRATOR.PROBE_EPOCHS:
            rows = []
            for index, subset_row in enumerate(self.subset_rows):
                suffix = "-changed" if changed_prediction and index == 0 else ""
                prediction = _write_bytes(
                    self.root
                    / tag
                    / f"e{epoch}"
                    / f"prediction-{index:03d}.npy",
                    f"e{epoch}-clip-{index:03d}{suffix}\n".encode("ascii"),
                )
                rows.append(
                    {
                        "canonical_clip_id": subset_row[
                            "canonical_clip_id"
                        ],
                        "prediction": prediction,
                    }
                )
            manifest = _write_jsonl(
                self.root / tag / f"prediction-e{epoch}.jsonl", rows
            )
            metric = ORCHESTRATOR.build_probe_metric(
                epoch=epoch,
                checkpoint=self.checkpoints[epoch],
                subset_manifest=self.subset,
                prediction_manifest=manifest,
                metric_values={
                    "body.released2.metrics.FGD": epoch / 100.0,
                    "body.released2.metrics.BC": 1.0 - epoch / 100.0,
                },
            )
            metric_artifact = _write_receipt(
                self.root / tag / f"metric-e{epoch}.json", metric
            )
            outputs.append(
                {
                    "epoch": epoch,
                    "candidate_checkpoint": copy.deepcopy(
                        self.checkpoints[epoch]
                    ),
                    "prediction_manifest": manifest,
                    "metric_receipt": metric_artifact,
                }
            )
        return outputs

    def capture(
        self,
        tag: str,
        *,
        host: str,
        candidates_per_wave: int,
        execution_mode: str,
        elapsed_seconds: int,
        outputs: list[dict[str, object]] | None = None,
        process: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if outputs is None:
            outputs = self.candidate_outputs(tag)
        if process is None:
            process = {
                "runner_rc": 0,
                "oom": False,
                "descendants_exited": True,
                "guards_restored": True,
            }
        captured = {
            "format": ORCHESTRATOR.MULTICANDIDATE_PROBE_RUN_INPUT_FORMAT,
            "status": "captured",
            "split": "val",
            "test_visible": False,
            "formal_host": host,
            "candidates_per_wave": candidates_per_wave,
            "execution_mode": execution_mode,
            "probe_binding": copy.deepcopy(self.binding),
            "candidate_outputs": outputs,
            "execution_trace": {
                "started_monotonic_ns": 10_000_000_000,
                "finished_monotonic_ns": (
                    10_000_000_000 + elapsed_seconds * 1_000_000_000
                ),
                "gpu_peak_memory_bytes": [70] * AUTHORITY.EXPECTED_SHARDS,
                "gpu_total_memory_bytes": [100] * AUTHORITY.EXPECTED_SHARDS,
            },
            "process_evidence": process,
        }
        return _write_receipt(self.root / f"{tag}-capture.json", captured)

    def run(
        self,
        tag: str,
        *,
        host: str,
        candidates_per_wave: int,
        execution_mode: str,
        elapsed_seconds: int,
        outputs: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        captured = self.capture(
            tag,
            host=host,
            candidates_per_wave=candidates_per_wave,
            execution_mode=execution_mode,
            elapsed_seconds=elapsed_seconds,
            outputs=outputs,
        )
        run = ORCHESTRATOR.build_probe_run(captured)
        return _write_receipt(self.root / f"{tag}-run.json", run)

    def comparison_matrix(self) -> list[dict[str, object]]:
        comparisons = []
        concurrent_elapsed = {1: 90, 2: 55, 4: 40}
        for host_index, host in enumerate(
            ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values()
        ):
            serial = self.run(
                f"h{host_index}-serial",
                host=host,
                candidates_per_wave=1,
                execution_mode="serial",
                elapsed_seconds=120,
            )
            for mode in ORCHESTRATOR.CONCURRENCY_SELECTION_ORDER:
                concurrent = self.run(
                    f"h{host_index}-c{mode}",
                    host=host,
                    candidates_per_wave=mode,
                    execution_mode="concurrent",
                    elapsed_seconds=concurrent_elapsed[mode] + host_index,
                )
                comparison = ORCHESTRATOR.build_multicandidate_comparison(
                    serial_run=serial,
                    concurrent_run=concurrent,
                )
                comparisons.append(
                    _write_receipt(
                        self.root
                        / f"h{host_index}-c{mode}-comparison.json",
                        comparison,
                    )
                )
        return comparisons


class BaseFreshConcurrencyProducerCpuTests(unittest.TestCase):
    def test_probe_metric_and_probe_run_cli_are_create_new(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProbeFixture(Path(raw))
            outputs = fixture.candidate_outputs("cli-probe")
            first = outputs[0]
            values = _write_json(
                fixture.root / "cli-metric-values.json",
                {
                    "body.released2.metrics.FGD": 0.01,
                    "body.released2.metrics.BC": 0.99,
                },
            )
            metric_output = fixture.root / "cli-probe-metric.json"
            argv = ["build-probe-metric", "--epoch", "1"]
            argv += _plain_args("checkpoint", fixture.checkpoints[1])
            argv += _plain_args("subset-manifest", fixture.subset)
            argv += _plain_args(
                "prediction-manifest", first["prediction_manifest"]
            )
            argv += _plain_args("metric-values", values)
            argv += ["--output-json", str(metric_output)]
            self.assertEqual(ORCHESTRATOR.main(argv), 0)
            metric_artifact, metric = ORCHESTRATOR._artifact(
                metric_output, payload_receipt=True
            )
            ORCHESTRATOR._probe_metric(
                metric_artifact,
                epoch=1,
                checkpoint=fixture.checkpoints[1],
                prediction_manifest=first["prediction_manifest"],
                subset_manifest=fixture.subset,
            )
            self.assertEqual(metric["split"], "val")
            with self.assertRaises(FileExistsError):
                ORCHESTRATOR.main(argv)

            capture = fixture.capture(
                "cli-run",
                host=next(
                    iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
                ),
                candidates_per_wave=2,
                execution_mode="concurrent",
                elapsed_seconds=60,
            )
            run_output = fixture.root / "cli-probe-run.json"
            run_argv = ["build-probe-run"]
            run_argv += _payload_args("probe-run-input", capture)
            run_argv += ["--output-json", str(run_output)]
            self.assertEqual(ORCHESTRATOR.main(run_argv), 0)
            run_artifact, _ = ORCHESTRATOR._artifact(
                run_output, payload_receipt=True
            )
            _artifact, run, summary = ORCHESTRATOR._replay_probe_run(
                run_artifact
            )
            self.assertEqual(run["status"], "complete")
            self.assertTrue(summary["process_success"])

    def test_full_two_host_matrix_builds_fresh_gate_cli(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProbeFixture(Path(raw))
            comparisons = fixture.comparison_matrix()
            gate = ORCHESTRATOR.build_multicandidate_gate(comparisons)
            self.assertEqual(gate["status"], "pass")
            self.assertEqual(gate["selected_candidates_per_wave"], 4)

            _artifact, first_comparison = (
                AUTHORITY._verify_compact_receipt(
                    comparisons[0], "first fixture comparison"
                )
            )
            comparison_output = (
                fixture.root / "formal-concurrency-comparison.json"
            )
            comparison_argv = ["build-concurrency-comparison"]
            comparison_argv += _payload_args(
                "serial-run", first_comparison["serial_run"]
            )
            comparison_argv += _payload_args(
                "concurrent-run", first_comparison["concurrent_run"]
            )
            comparison_argv += [
                "--output-json",
                str(comparison_output),
            ]
            self.assertEqual(ORCHESTRATOR.main(comparison_argv), 0)
            self.assertEqual(
                json.loads(comparison_output.read_text(encoding="utf-8")),
                first_comparison,
            )

            gate_output = fixture.root / "formal-concurrency-gate.json"
            argv = ["build-concurrency-gate"]
            for comparison in comparisons:
                argv += [
                    "--comparison-path",
                    str(comparison["path"]),
                    "--comparison-sha256",
                    str(comparison["sha256"]),
                    "--comparison-payload-sha256",
                    str(comparison["receipt_payload_sha256"]),
                ]
            argv += ["--output-json", str(gate_output)]
            self.assertEqual(ORCHESTRATOR.main(argv), 0)
            artifact, _ = ORCHESTRATOR._artifact(
                gate_output, payload_receipt=True
            )
            _artifact, replayed = ORCHESTRATOR._validate_multicandidate_gate(
                artifact
            )
            self.assertEqual(replayed, gate)
            with self.assertRaises(FileExistsError):
                ORCHESTRATOR.main(argv)

    def test_probe_producers_fail_closed_on_invalid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProbeFixture(Path(raw))
            manifest = fixture.candidate_outputs("negative-metric")[0][
                "prediction_manifest"
            ]
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_probe_metric(
                    epoch=1,
                    checkpoint=fixture.checkpoints[1],
                    subset_manifest=fixture.subset,
                    prediction_manifest=manifest,
                    metric_values={"body.released2.metrics.FGD": float("nan")},
                )

            outputs = fixture.candidate_outputs("negative-run")
            failed_capture = fixture.capture(
                "failed-with-outputs",
                host=next(
                    iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
                ),
                candidates_per_wave=2,
                execution_mode="concurrent",
                elapsed_seconds=60,
                outputs=outputs,
                process={
                    "runner_rc": 1,
                    "oom": False,
                    "descendants_exited": True,
                    "guards_restored": True,
                },
            )
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_probe_run(failed_capture)

            invalid_serial = fixture.capture(
                "serial-c2",
                host=next(
                    iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
                ),
                candidates_per_wave=2,
                execution_mode="serial",
                elapsed_seconds=60,
            )
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_probe_run(invalid_serial)

            _artifact, captured = AUTHORITY._verify_compact_receipt(
                fixture.capture(
                    "forbidden-binding",
                    host=next(
                        iter(
                            ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values()
                        )
                    ),
                    candidates_per_wave=1,
                    execution_mode="serial",
                    elapsed_seconds=120,
                ),
                "fixture capture",
            )
            captured["probe_binding"]["pipeline"]["path"] = (
                "/tmp/SemGate/foreign-pipeline.json"
            )
            forbidden = _write_receipt(
                fixture.root / "forbidden-capture.json", captured
            )
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_probe_run(forbidden)

    def test_comparison_mismatch_and_incomplete_gate_are_not_publishable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = ProbeFixture(Path(raw))
            host = next(
                iter(ORCHESTRATOR.FORMAL_HOST_BY_PARTITION.values())
            )
            serial = fixture.run(
                "mismatch-serial",
                host=host,
                candidates_per_wave=1,
                execution_mode="serial",
                elapsed_seconds=120,
            )
            changed = fixture.candidate_outputs(
                "mismatch-concurrent", changed_prediction=True
            )
            concurrent = fixture.run(
                "mismatch-concurrent-run",
                host=host,
                candidates_per_wave=2,
                execution_mode="concurrent",
                elapsed_seconds=50,
                outputs=changed,
            )
            comparison = ORCHESTRATOR.build_multicandidate_comparison(
                serial_run=serial, concurrent_run=concurrent
            )
            self.assertEqual(comparison["status"], "fail")
            self.assertFalse(
                comparison["equivalence"]["prediction_sha_bytes_equal"]
            )
            comparison_artifact = _write_receipt(
                fixture.root / "mismatch-comparison.json", comparison
            )
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_multicandidate_gate(
                    [comparison_artifact]
                )

    def test_run_spec_producer_hardcodes_base_identity_and_replays_first(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            source = {
                "origin": AUTHORITY.EXPECTED_ORIGIN,
                "commit": "a" * 40,
                "tree": "b" * 40,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            }
            frozen = {
                "format": ORCHESTRATOR.RUN_SPEC_INPUT_FORMAT,
                "status": "frozen_inputs",
                "split": "val",
                "test_visible": False,
                "source": source,
                "multi_candidate_gate": {"path": "/formal/base-gate.json"},
                "candidate_bundle": {},
                "val_inputs": {"path": "/formal/show-val.json"},
                "pipeline": {"path": "/formal/base-pipeline.json"},
                "prerequisite_selection": {},
                "continuation_decision": {},
                "continuation_waves": [],
                "canonical_manifest": {},
                "validation_gates": [],
                "metric_assets": {},
                "real_feature_cache": {},
            }
            input_artifact = _write_receipt(
                root / "run-spec-input.json", frozen
            )
            observed: list[dict[str, object]] = []

            def replay(
                artifact: dict[str, object],
                *,
                expected_source_commit: str,
                expected_source_tree: str,
            ) -> tuple[dict[str, object], dict[str, object]]:
                normalized, payload = AUTHORITY._verify_compact_receipt(
                    artifact, "generated run spec"
                )
                self.assertTrue(Path(normalized["path"]).is_file())
                self.assertEqual(expected_source_commit, "a" * 40)
                self.assertEqual(expected_source_tree, "b" * 40)
                self.assertEqual(
                    payload["generator_module"],
                    "models.semtalk.semtalk_base",
                )
                self.assertEqual(payload["dataset"], "SHOW")
                self.assertEqual(
                    payload["target_speaker_scope"],
                    AUTHORITY.EXPECTED_SCOPE,
                )
                self.assertEqual(payload["split"], "val")
                self.assertIs(payload["test_visible"], False)
                observed.append(payload)
                return normalized, payload

            output = root / "run-spec.json"
            argv = ["build-run-spec"]
            argv += _payload_args("run-spec-input", input_artifact)
            argv += ["--output-json", str(output)]
            with mock.patch.object(
                ORCHESTRATOR, "validate_run_spec", side_effect=replay
            ):
                self.assertEqual(ORCHESTRATOR.main(argv), 0)
                with self.assertRaises(FileExistsError):
                    ORCHESTRATOR.main(argv)
            self.assertEqual(len(observed), 2)
            published = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                published["generator"], "SemTalk Base Motion Generation"
            )
            self.assertNotIn("semgate", json.dumps(published).casefold())
            self.assertNotIn("sparse", json.dumps(published).casefold())

            failed_output = root / "must-not-exist.json"
            failed_argv = ["build-run-spec"]
            failed_argv += _payload_args("run-spec-input", input_artifact)
            failed_argv += ["--output-json", str(failed_output)]
            with mock.patch.object(
                ORCHESTRATOR,
                "validate_run_spec",
                side_effect=ORCHESTRATOR.BaseFreshValOrchestratorError(
                    "fresh replay failed"
                ),
            ):
                with self.assertRaises(
                    ORCHESTRATOR.BaseFreshValOrchestratorError
                ):
                    ORCHESTRATOR.main(failed_argv)
            self.assertFalse(failed_output.exists())

    def test_run_spec_input_schema_scope_and_cli_abbreviation_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            source = {
                "origin": AUTHORITY.EXPECTED_ORIGIN,
                "commit": "a" * 40,
                "tree": "b" * 40,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            }
            base = {
                "format": ORCHESTRATOR.RUN_SPEC_INPUT_FORMAT,
                "status": "frozen_inputs",
                "split": "val",
                "test_visible": False,
                "source": source,
                "multi_candidate_gate": {},
                "candidate_bundle": {},
                "val_inputs": {},
                "pipeline": {},
                "prerequisite_selection": {},
                "continuation_decision": {},
                "continuation_waves": [],
                "canonical_manifest": {},
                "validation_gates": [],
                "metric_assets": {},
                "real_feature_cache": {},
            }
            malformed = copy.deepcopy(base)
            malformed["unexpected"] = True
            malformed_artifact = _write_receipt(
                root / "malformed.json", malformed
            )
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_run_spec(malformed_artifact)

            forbidden = copy.deepcopy(base)
            forbidden["metric_assets"] = {
                "talkshow_metric_root": "/formal/SemGate/metrics"
            }
            forbidden_artifact = _write_receipt(
                root / "forbidden.json", forbidden
            )
            with self.assertRaises(
                ORCHESTRATOR.BaseFreshValOrchestratorError
            ):
                ORCHESTRATOR.build_run_spec(forbidden_artifact)

            with self.assertRaises(SystemExit):
                ORCHESTRATOR._parse_args(
                    [
                        "build-run-spec",
                        "--run-spec-input-p",
                        str(forbidden_artifact["path"]),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
