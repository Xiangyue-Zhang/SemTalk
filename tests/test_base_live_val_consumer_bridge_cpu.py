from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.show_base import base_live_val_consumer_bridge as BRIDGE


class BaseLiveValConsumerBridgeCpuTest(unittest.TestCase):
    def test_strict_json_rejects_duplicate_and_nonfinite_tokens(self) -> None:
        with self.assertRaisesRegex(BRIDGE.LiveConsumerError, "duplicate JSON"):
            BRIDGE._strict_json_bytes(b'{"epoch":1,"epoch":2}', "duplicate")
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(BRIDGE.LiveConsumerError, "non-finite"):
                    BRIDGE._strict_json_bytes(
                        b'{"metric":' + token + b"}", "nonfinite"
                    )

    def test_exact_types_and_rehashed_execution_tamper_are_rejected(self) -> None:
        with self.assertRaises(BRIDGE.LiveConsumerError):
            BRIDGE._integer(True, "integer")
        with self.assertRaises(BRIDGE.LiveConsumerError):
            BRIDGE._integer(8.0, "integer")
        with self.assertRaises(BRIDGE.LiveConsumerError):
            BRIDGE._finite(True, "number")
        self.assertFalse(BRIDGE._strict_equal(1, True))
        self.assertFalse(BRIDGE._strict_equal(1, 1.0))

        expected = {
            "purpose": "single_candidate_diffsheg_validation_only",
            "split": "val",
            "test_visible": False,
            "expected_shards": 8,
            "candidate_epochs_in_work_item": [1],
            "may_publish_standard_22_candidate_measurement_before_e400": False,
            "may_influence_training": False,
            "requires_guarded_runner": True,
        }
        self.assertEqual(BRIDGE._validate_execution_contract(expected, 1), expected)
        for key, replacement in (
            ("expected_shards", True),
            ("candidate_epochs_in_work_item", [1.0]),
            ("may_influence_training", 0),
            ("requires_guarded_runner", 1),
        ):
            with self.subTest(key=key):
                tampered = copy.deepcopy(expected)
                tampered[key] = replacement
                rehashed = BRIDGE._with_payload_sha(
                    {"execution_contract": tampered}
                )
                self.assertEqual(
                    BRIDGE._payload_sha(rehashed),
                    rehashed["receipt_payload_sha256"],
                )
                with self.assertRaises(BRIDGE.LiveConsumerError):
                    BRIDGE._validate_execution_contract(tampered, 1)

    def test_forbidden_paths_and_cli_test_split_are_rejected(self) -> None:
        for path in (
            "/tmp/formal/test/results.json",
            "/tmp/formal/testing/results.json",
            "/tmp/formal/e30/checkpoint.bin",
            "/tmp/formal/Speaker2/checkpoint.bin",
            "/tmp/formal/SemGate/checkpoint.bin",
            "/tmp/formal/Sparse/checkpoint.bin",
        ):
            with self.subTest(path=path):
                with self.assertRaises(BRIDGE.LiveConsumerError):
                    BRIDGE._canonical_path(path, "formal input", must_exist=False)

        argv = [
            "shard", "--split", "test", "--preflight", "/formal/val/p.json",
            "--expected-preflight-sha256", "a" * 64, "--epoch", "1",
            "--output-root", "/formal/val/e1", "--num-shards", "8",
            "--shard-id", "0", "--device", "cuda:0",
        ]
        with self.assertRaises(SystemExit):
            BRIDGE.build_parser().parse_args(argv)

    def test_frozen_alias_and_schedule_canonical_payload_bindings(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-bridge-artifacts-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            frozen_body = {"format": "frozen", "value": 3}
            frozen = dict(frozen_body)
            frozen["receipt_sha256"] = hashlib.sha256(
                BRIDGE._canonical_json(frozen_body)
            ).hexdigest()
            frozen_path = root / "frozen.json"
            frozen_payload = BRIDGE._canonical_json(frozen, newline=True)
            frozen_path.write_bytes(frozen_payload)
            frozen_ref = {
                "path": str(frozen_path),
                "sha256": hashlib.sha256(frozen_payload).hexdigest(),
                "bytes": len(frozen_payload),
                "receipt_payload_sha256": frozen["receipt_sha256"],
            }
            artifact, _payload = BRIDGE._artifact(
                frozen_ref,
                frozenset(
                    {"path", "sha256", "bytes", "receipt_payload_sha256"}
                ),
                "frozen",
                artifact_payload_key="receipt_payload_sha256",
                document_payload_key="receipt_sha256",
            )
            self.assertEqual(artifact, frozen_ref)
            wrong = dict(frozen_ref)
            wrong["receipt_payload_sha256"] = "f" * 64
            with self.assertRaises(BRIDGE.LiveConsumerError):
                BRIDGE._artifact(
                    wrong,
                    frozenset(
                        {"path", "sha256", "bytes", "receipt_payload_sha256"}
                    ),
                    "frozen",
                    artifact_payload_key="receipt_payload_sha256",
                    document_payload_key="receipt_sha256",
                )

            schedule = {"format": "schedule", "epochs": [1, 2, 4]}
            schedule_payload = BRIDGE._canonical_json(schedule, newline=True)
            schedule_path = root / "schedule.json"
            schedule_path.write_bytes(schedule_payload)
            schedule_ref = {
                "path": str(schedule_path),
                "sha256": hashlib.sha256(schedule_payload).hexdigest(),
                "bytes": len(schedule_payload),
                "payload_sha256": hashlib.sha256(
                    BRIDGE._canonical_json(schedule)
                ).hexdigest(),
            }
            self.assertEqual(
                BRIDGE._artifact(
                    schedule_ref,
                    frozenset({"path", "sha256", "bytes", "payload_sha256"}),
                    "schedule",
                    artifact_payload_key="payload_sha256",
                    canonical_full_payload=True,
                )[0],
                schedule_ref,
            )

    def test_create_new_publication_is_race_safe_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-bridge-race-", dir="/private/tmp"
        ) as raw:
            output = Path(raw) / "claim.json"
            value = BRIDGE._with_payload_sha(
                {"format": BRIDGE.CLAIM_FORMAT, "run_root": "/formal/val/run-a"}
            )
            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(
                    pool.map(
                        lambda _index: BRIDGE._write_new_or_identical(output, value),
                        range(64),
                    )
                )
            self.assertEqual(sum(created for _artifact, created in results), 1)
            self.assertEqual(len({row[0]["sha256"] for row in results}), 1)
            self.assertEqual(output.stat().st_nlink, 1)
            artifact, created = BRIDGE._write_new_or_identical(output, value)
            self.assertFalse(created)
            self.assertEqual(artifact["sha256"], results[0][0]["sha256"])

            conflict = BRIDGE._with_payload_sha(
                {"format": BRIDGE.CLAIM_FORMAT, "run_root": "/formal/val/run-b"}
            )
            with self.assertRaises(BRIDGE.DuplicateConsumptionError):
                BRIDGE._write_new_or_identical(output, conflict)

    def test_copied_authority_uses_the_same_producer_claim_slot(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-bridge-claim-anchor-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            ready_dir = root / "candidate_receipts"
            ready_dir.mkdir()
            ready_path = ready_dir / "epoch-0001.json"
            ready_path.write_text("{}\n", encoding="utf-8")
            candidate_dir = root / "candidates"
            candidate_dir.mkdir()
            candidate_path = candidate_dir / "base_official_adapt_epoch_01.bin"
            candidate_path.write_bytes(b"checkpoint")
            ready = {
                "path": str(ready_path),
                "sha256": "1" * 64,
                "bytes": ready_path.stat().st_size,
                "receipt_payload_sha256": "2" * 64,
            }
            candidate = {
                "path": str(candidate_path),
                "relative_path": "candidates/base_official_adapt_epoch_01.bin",
                "sha256": "3" * 64,
                "bytes": candidate_path.stat().st_size,
                "model_state_tensors": 1,
                "model_state_schema_sha256": "4" * 64,
                "model_state_semantic_sha256": "5" * 64,
            }
            original = root / "authorities" / "epoch-0001.json"
            copied = root / "copied-authorities" / "epoch-0001.json"
            original.parent.mkdir()
            copied.parent.mkdir()
            first = BRIDGE._claim_path(
                producer_ready_receipt=ready,
                candidate_checkpoint=candidate,
                epoch=1,
            )
            second = BRIDGE._claim_path(
                producer_ready_receipt=ready,
                candidate_checkpoint=candidate,
                epoch=1,
            )
            self.assertEqual(first, second)
            self.assertEqual(
                first,
                root / "live_val_consumer_claims" / "epoch-0001.json",
            )

            first_body = BRIDGE._with_payload_sha(
                {
                    "format": BRIDGE.CLAIM_FORMAT,
                    "work_authority": {"path": str(original)},
                    "run_root": "/formal/val/run-a",
                }
            )
            copied_body = BRIDGE._with_payload_sha(
                {
                    "format": BRIDGE.CLAIM_FORMAT,
                    "work_authority": {"path": str(copied)},
                    "run_root": "/formal/val/run-b",
                }
            )
            _artifact, created = BRIDGE._write_new_or_identical(
                first, first_body
            )
            self.assertTrue(created)
            with self.assertRaises(BRIDGE.DuplicateConsumptionError):
                BRIDGE._write_new_or_identical(second, copied_body)

            wrong_dir = root / "other"
            wrong_dir.mkdir()
            wrong_path = wrong_dir / "epoch-0001.json"
            wrong_path.write_text("{}\n", encoding="utf-8")
            wrong_ready = dict(ready)
            wrong_ready["path"] = str(wrong_path)
            with self.assertRaisesRegex(
                BRIDGE.LiveConsumerError, "fixed candidate inventory"
            ):
                BRIDGE._claim_path(
                    producer_ready_receipt=wrong_ready,
                    candidate_checkpoint=candidate,
                    epoch=1,
                )

    def _source_git(
        self,
        *,
        dirty: bool = False,
        attached: bool = False,
        heads: bool = False,
        wrong_origin: bool = False,
        wrong_commit: bool = False,
    ):
        def run(_root: Path, *arguments: str, allow_failure: bool = False):
            del allow_failure
            if arguments == ("remote",):
                return 0, "origin"
            if arguments == ("remote", "get-url", "origin"):
                return 0, ("git@example.invalid/other.git" if wrong_origin else BRIDGE.EXPECTED_ORIGIN)
            if arguments == ("remote", "get-url", "--push", "origin"):
                return 0, ("git@example.invalid/other.git" if wrong_origin else BRIDGE.EXPECTED_ORIGIN)
            if arguments == ("rev-parse", "HEAD"):
                return 0, ("f" * 40 if wrong_commit else BRIDGE.VALIDATION_SOURCE_COMMIT)
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return 0, BRIDGE.VALIDATION_SOURCE_TREE
            if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                return 0, (" M dirty.py" if dirty else "")
            if arguments == ("symbolic-ref", "-q", "HEAD"):
                return (0, "refs/heads/main") if attached else (1, "")
            if arguments == (
                "for-each-ref", "--format=%(refname)", "refs/heads"
            ):
                return 0, ("refs/heads/main" if heads else "")
            raise AssertionError(arguments)

        return run

    def test_validation_source_is_exact_clean_detached_and_branchless(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-source-", dir="/private/tmp"
        ) as raw:
            source = {
                "origin": BRIDGE.EXPECTED_ORIGIN,
                "source_root": raw,
                "commit": BRIDGE.VALIDATION_SOURCE_COMMIT,
                "tree": BRIDGE.VALIDATION_SOURCE_TREE,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            }
            with mock.patch.object(BRIDGE, "_git", side_effect=self._source_git()):
                self.assertEqual(BRIDGE._validate_source(source), Path(raw))
            for variant in (
                {"dirty": True},
                {"attached": True},
                {"heads": True},
                {"wrong_origin": True},
                {"wrong_commit": True},
            ):
                with self.subTest(variant=variant):
                    with mock.patch.object(
                        BRIDGE, "_git", side_effect=self._source_git(**variant)
                    ):
                        with self.assertRaises(BRIDGE.LiveConsumerError):
                            BRIDGE._validate_source(source)
            malformed = dict(source)
            malformed["clean"] = 1
            with mock.patch.object(BRIDGE, "_git", side_effect=self._source_git()):
                with self.assertRaises(BRIDGE.LiveConsumerError):
                    BRIDGE._validate_source(malformed)

    @staticmethod
    def _reconcile_fixture(root: Path):
        epochs = BRIDGE.CANDIDATE_EPOCHS
        val = {
            "path": str(root / "val.json"),
            "sha256": "1" * 64,
            "receipt_payload_sha256": "2" * 64,
        }
        pipeline = {
            "path": str(root / "pipeline.json"),
            "sha256": "3" * 64,
            "receipt_payload_sha256": "4" * 64,
        }
        live_artifacts = []
        live_values = []
        work_authorities = []
        candidates = {}
        for index, epoch in enumerate(epochs):
            authority = {
                "path": str(root / f"authority-{epoch}.json"),
                "sha256": f"{index + 10:064x}",
                "bytes": 100 + index,
                "receipt_payload_sha256": f"{index + 40:064x}",
            }
            work_authorities.append(BRIDGE._artifact_core(authority))
            candidate = {
                "path": str(root / f"candidate-{epoch}.bin"),
                "sha256": f"{index + 80:064x}",
            }
            candidates[epoch] = {**candidate, "bytes": 1}
            live_values.append(
                {
                    "candidate_epoch": epoch,
                    "candidate_checkpoint": candidate,
                    "work_authority": authority,
                    "val_inputs_receipt": val,
                    "pipeline_receipt": pipeline,
                    "inference_lineage": {
                        "path": str(root / f"lineage-{epoch}.json"),
                        "sha256": f"{index + 120:064x}",
                        "receipt_payload_sha256": f"{index + 160:064x}",
                    },
                    "diffsheg_report": {
                        "path": str(root / f"report-{epoch}.json"),
                        "sha256": f"{index + 200:064x}",
                    },
                }
            )
            live_artifacts.append(
                {
                    "path": str(root / f"live-{epoch}.json"),
                    "sha256": f"{index + 240:064x}",
                    "bytes": 200 + index,
                    "receipt_payload_sha256": f"{index + 280:064x}",
                }
            )
        source = {
            "origin": BRIDGE.EXPECTED_ORIGIN,
            "source_root": str(root),
            "commit": BRIDGE.VALIDATION_SOURCE_COMMIT,
            "tree": BRIDGE.VALIDATION_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        reconciliation = {
            "validation_source": source,
            "pipeline_receipt": pipeline,
            "val_inputs_receipt": val,
            "producer_manifest": {
                "path": str(root / "manifest.json"), "sha256": "5" * 64,
            },
            "producer_status": {
                "path": str(root / "status.json"), "sha256": "6" * 64,
            },
            "frozen_inputs": {
                "path": str(root / "frozen.json"), "sha256": "7" * 64,
            },
            "work_authorities": work_authorities,
        }
        return reconciliation, live_artifacts, live_values, candidates

    def test_reconcile_requires_e400_receipt_exact_order_and_distinct_roots(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-reconcile-gates-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            reconciliation, live_artifacts, live_values, _candidates = (
                self._reconcile_fixture(root)
            )
            output = root / "selection"
            triples = [
                (str(epoch), str(root / f"live-{epoch}.json"), "a" * 64)
                for epoch in BRIDGE.CANDIDATE_EPOCHS
            ]
            args = argparse.Namespace(
                reconciliation=root / "reconciliation.json",
                expected_reconciliation_sha256="b" * 64,
                live_measurement=triples[:-1],
                output_root=output,
            )
            with mock.patch.object(
                BRIDGE,
                "_validate_reconciliation",
                return_value=({"path": str(args.reconciliation)}, reconciliation),
            ):
                with self.assertRaisesRegex(BRIDGE.LiveConsumerError, "exactly 22"):
                    BRIDGE.reconcile(args)
            self.assertFalse(output.exists())

            args.live_measurement = list(triples)
            args.live_measurement[0], args.live_measurement[1] = (
                args.live_measurement[1], args.live_measurement[0]
            )
            lookup = {
                epoch: (artifact, value)
                for epoch, artifact, value in zip(
                    BRIDGE.CANDIDATE_EPOCHS, live_artifacts, live_values
                )
            }

            def validate(path: Path, _sha: str):
                epoch = int(path.stem.split("-")[-1])
                return lookup[epoch]

            with (
                mock.patch.object(
                    BRIDGE,
                    "_validate_reconciliation",
                    return_value=({"path": str(args.reconciliation)}, reconciliation),
                ),
                mock.patch.object(
                    BRIDGE, "_validate_live_measurement", side_effect=validate
                ),
            ):
                with self.assertRaises(BRIDGE.LiveConsumerError):
                    BRIDGE.reconcile(args)
            self.assertFalse(output.exists())

            args.live_measurement = list(triples)
            duplicate_artifacts = copy.deepcopy(live_artifacts)
            duplicate_artifacts[1]["path"] = duplicate_artifacts[0]["path"]
            duplicate_lookup = {
                epoch: (artifact, value)
                for epoch, artifact, value in zip(
                    BRIDGE.CANDIDATE_EPOCHS, duplicate_artifacts, live_values
                )
            }

            def validate_duplicate(path: Path, _sha: str):
                epoch = int(path.stem.split("-")[-1])
                return duplicate_lookup[epoch]

            with (
                mock.patch.object(
                    BRIDGE,
                    "_validate_reconciliation",
                    return_value=({"path": str(args.reconciliation)}, reconciliation),
                ),
                mock.patch.object(
                    BRIDGE,
                    "_validate_live_measurement",
                    side_effect=validate_duplicate,
                ),
            ):
                with self.assertRaisesRegex(BRIDGE.LiveConsumerError, "not distinct"):
                    BRIDGE.reconcile(args)
            self.assertFalse(output.exists())

            type_alias_reconciliation = copy.deepcopy(reconciliation)
            type_alias_reconciliation["work_authorities"][0]["bytes"] = float(
                type_alias_reconciliation["work_authorities"][0]["bytes"]
            )
            with (
                mock.patch.object(
                    BRIDGE,
                    "_validate_reconciliation",
                    return_value=(
                        {"path": str(args.reconciliation)},
                        type_alias_reconciliation,
                    ),
                ),
                mock.patch.object(
                    BRIDGE, "_validate_live_measurement", side_effect=validate
                ),
            ):
                with self.assertRaisesRegex(BRIDGE.LiveConsumerError, "do not bind"):
                    BRIDGE.reconcile(args)
            self.assertFalse(output.exists())

    def test_reconcile_happy_path_only_promotes_after_receipt(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-reconcile-happy-", dir="/private/tmp"
        ) as raw:
            root = Path(raw)
            reconciliation, live_artifacts, live_values, candidates = (
                self._reconcile_fixture(root)
            )
            reconciliation_artifact = {
                "path": str(root / "reconciliation.json"),
                "sha256": "b" * 64,
                "bytes": 123,
                "receipt_payload_sha256": "c" * 64,
            }
            lookup = {
                epoch: (artifact, value)
                for epoch, artifact, value in zip(
                    BRIDGE.CANDIDATE_EPOCHS, live_artifacts, live_values
                )
            }

            def validate(path: Path, _sha: str):
                epoch = int(path.stem.split("-")[-1])
                return lookup[epoch]

            contract = SimpleNamespace(
                validate_pipeline=lambda *_args, **_kwargs: (
                    reconciliation["pipeline_receipt"],
                    {
                        "fixed_checkpoints": {
                            stage: {"sha256": stage[0] * 64}
                            for stage in ("face", "hands", "upper", "lower", "global")
                        }
                    },
                ),
                validate_candidate_bundle=lambda **_kwargs: {
                    "candidates": candidates,
                    "producer_source": {"origin": BRIDGE.EXPECTED_ORIGIN},
                },
            )
            producer = SimpleNamespace(
                build_measurement=lambda **kwargs: BRIDGE._with_payload_sha(
                    {
                        "format": "selector-ready-fixture",
                        "epoch": kwargs["epoch"],
                        "split": "val",
                    }
                )
            )
            selector = SimpleNamespace(
                build_selection=lambda **_kwargs: BRIDGE._with_payload_sha(
                    {
                        "format": "formal-selection-fixture",
                        "selected": {"epoch": 200, "fgd": 0.125},
                    }
                )
            )
            modules = {
                "contract": contract,
                "producer": producer,
                "selector": selector,
            }
            triples = [
                (str(epoch), str(root / f"live-{epoch}.json"), "a" * 64)
                for epoch in BRIDGE.CANDIDATE_EPOCHS
            ]
            output = root / "selection"
            args = argparse.Namespace(
                reconciliation=root / "reconciliation.json",
                expected_reconciliation_sha256="b" * 64,
                live_measurement=triples,
                output_root=output,
            )
            with (
                mock.patch.object(
                    BRIDGE,
                    "_validate_reconciliation",
                    return_value=(reconciliation_artifact, reconciliation),
                ),
                mock.patch.object(
                    BRIDGE, "_validate_live_measurement", side_effect=validate
                ),
                mock.patch.object(
                    BRIDGE, "_validate_source", return_value=root
                ),
                mock.patch.object(
                    BRIDGE, "_load_validation_modules", return_value=modules
                ),
            ):
                result = BRIDGE.reconcile(args)
            self.assertTrue(result["selection_eligible"])
            self.assertEqual(result["candidate_count"], 22)
            self.assertEqual(result["selected_epoch"], 200)
            selection = json.loads((output / "selection.json").read_bytes())
            self.assertEqual(
                selection["reconciliation_receipt"], reconciliation_artifact
            )
            self.assertTrue(selection["selection_eligible"])
            self.assertEqual(selection["test_evaluations_observed"], 0)
            self.assertEqual(len(selection["live_measurements"]), 22)
            self.assertEqual(len(selection["reconciled_measurements"]), 22)

    def test_live_measurement_is_never_selection_eligible(self) -> None:
        preflight = {
            "candidate_epochs": [1],
            "candidate_bundle": {
                "candidates": {
                    "1": {"path": "/formal/val/e1.bin", "sha256": "1" * 64}
                }
            },
            "work_authority": {"path": "/formal/val/work.json"},
            "consumer_claim": {"path": "/formal/val/claim.json"},
            "val_inputs_receipt": {"path": "/formal/val/inputs.json"},
            "pipeline_receipt": {"path": "/formal/val/pipeline.json"},
        }
        value = BRIDGE._measurement_value(
            preflight_artifact={"path": "/formal/val/preflight.json"},
            preflight=preflight,
            lineage_artifact={"path": "/formal/val/lineage.json"},
            report_artifact={"path": "/formal/val/report.json"},
            fgd=0.25,
        )
        self.assertFalse(value["selection_eligible"])
        self.assertFalse(value["test_visible"])
        self.assertFalse(value["execution_contract"]["may_influence_training"])
        self.assertTrue(
            value["execution_contract"][
                "requires_e400_reconciliation_for_selection"
            ]
        )

    def test_launcher_is_guarded_exact_eight_shard_and_val_only(self) -> None:
        launcher = (
            Path(BRIDGE.__file__).with_name("run_base_live_val_8shard.sh")
        )
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        source = launcher.read_text()
        self.assertIn("semtalk_require_exact_guarded_runner_all_gpus", source)
        self.assertIn("for shard_id in 0 1 2 3 4 5 6 7", source)
        self.assertIn("--num-shards 8", source)
        self.assertIn("--split val", source)
        self.assertIn("capture_child_identity", source)
        self.assertIn("PROC_PPID", source)
        self.assertIn("PROC_STARTTIME", source)
        self.assertIn("PROC_CMDLINE_SHA256", source)
        self.assertIn("refs/heads", source)
        self.assertIn(
            'semtalk_require_formal_venv_python "$python_bin" semtalk',
            source,
        )
        self.assertNotIn("-L $python_bin", source)
        self.assertNotIn("--split test", source)
        self.assertNotIn("pgrep", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("killall", source)


if __name__ == "__main__":
    unittest.main()
