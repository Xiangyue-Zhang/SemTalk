from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from scripts.show_base import produce_base_topology_short_quality as producer
from scripts.show_base import select_base_training_topology as official_selector
from scripts.show_base import train_base_official_adapt_long as contract
from tests import test_produce_base_topology_short_quality_cpu as quality_fixture_module
from tests import test_train_base_official_adapt_long_cpu as training_fixture_module


QualityFixture = quality_fixture_module.QualityFixture
_FakeFormalValidation = quality_fixture_module._FakeFormalValidation


SCRIPT = REPOSITORY / "scripts/show_base/select_base_v14_two_candidate.py"
SPEC = importlib.util.spec_from_file_location("v14_two_candidate_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V14TwoCandidateSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve(strict=True)
        cls.official_repository = cls.root / "official-runtime-70a70f4"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-local",
                "--no-checkout",
                str(REPOSITORY),
                str(cls.official_repository),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(cls.official_repository),
                "checkout",
                "--quiet",
                "--detach",
                selection.RUNTIME_VALIDATION_SOURCE_COMMIT,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(cls.official_repository),
                "remote",
                "set-url",
                "origin",
                selection.SOURCE_ORIGIN,
            ],
            check=True,
        )
        local_branches = subprocess.run(
            [
                "git",
                "-C",
                str(cls.official_repository),
                "for-each-ref",
                "--format=%(refname)",
                "refs/heads",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for reference in local_branches:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(cls.official_repository),
                    "update-ref",
                    "-d",
                    reference,
                ],
                check=True,
            )
        cls.fixture = QualityFixture(cls.root)
        cls.fixture.frozen_payload["source"].update(
            {
                "origin": selection.SOURCE_ORIGIN,
                "commit": selection.TRAINING_SOURCE_COMMIT,
                "tree": selection.TRAINING_SOURCE_TREE,
                "clean": True,
            }
        )
        unsigned = dict(cls.fixture.frozen_payload)
        unsigned.pop("receipt_sha256")
        cls.fixture.frozen_payload["receipt_sha256"] = (
            contract.canonical_json_sha256(unsigned)
        )
        base_frozen_payload = copy.deepcopy(cls.fixture.frozen_payload)
        cls.report_specs = []
        with cls._formal_patches():
            for index, mode in enumerate(selection.MODES):
                specification = contract.TOPOLOGY_SPECS[mode]
                mode_frozen = copy.deepcopy(base_frozen_payload)
                mode_frozen["protocol"].update(
                    {
                        "node_count": specification["node_count"],
                        "local_world_size": specification["local_world_size"],
                        "world_size": specification["world_size"],
                        "local_batch_size": specification["local_batch_size"],
                        "global_batch_size": specification["global_batch_size"],
                        "expected_updates_per_epoch": specification[
                            "updates_per_epoch"
                        ],
                        "expected_unique_samples_per_epoch": specification[
                            "unique_samples_per_epoch"
                        ],
                        "distributed_topology": {"mode": mode},
                        "optimizer": {
                            "name": "Adam",
                            "learning_rate": specification["learning_rate"],
                        },
                        "precision": specification["precision"],
                    }
                )
                mode_frozen["topology"] = {
                    "topology_mode": mode,
                    "node_count": specification["node_count"],
                    "local_world_size": specification["local_world_size"],
                    "world_size": specification["world_size"],
                    "local_batch_size": specification["local_batch_size"],
                    "global_batch_size": specification["global_batch_size"],
                    "updates_per_epoch": specification["updates_per_epoch"],
                    "unique_samples_per_epoch": specification[
                        "unique_samples_per_epoch"
                    ],
                    "receipt_sha256": "7" * 64,
                }
                unsigned = dict(mode_frozen)
                unsigned.pop("receipt_sha256")
                mode_frozen["receipt_sha256"] = (
                    contract.canonical_json_sha256(unsigned)
                )
                cls.fixture.frozen_payload = mode_frozen
                ready = cls.fixture.candidate_ready_receipts(mode)
                report = cls.root / f"quality-report-{index}.json"
                result = producer.main(
                    cls.fixture.argv(mode, report, ready_override=ready)
                )
                if result != 0:
                    raise AssertionError(f"fixture producer failed for {mode}")
                payload = json.loads(report.read_text(encoding="utf-8"))
                cls.report_specs.append(
                    (
                        mode,
                        report.resolve(),
                        _sha(report),
                        report.stat().st_size,
                        payload["receipt_sha256"],
                    )
                )
        cls.protocol_path = (
            REPOSITORY
            / "configs/show_base/semtalk_v14_fast_selection_protocol_20260802.json"
        ).resolve(strict=True)
        cls.native_probes, cls.native_reports = (
            training_fixture_module._topology_selection_inputs(
                {
                    mode: 1_000.0 + index
                    for index, mode in enumerate(contract.TOPOLOGY_SPECS)
                }
            )
        )
        cls.native_selection = official_selector.select_topology(
            cls.native_probes,
            cls.native_reports,
            gate_spec_sha256=selection.TOPOLOGY_GATE_SHA256,
            quality_gate_spec_sha256=selection.QUALITY_GATE_SHA256,
        )
        cls.native_path = cls.root / "trainer-native-selection.json"
        cls._write_json(cls.native_path, cls.native_selection)
        cls.native_sha = _sha(cls.native_path)
        cls.native_probe_root = cls.root / "native-probe-campaign"
        cls.native_quality_root = cls.root / "native-quality-campaign"
        cls.native_probe_rows = []
        for index, mode in enumerate(contract.TOPOLOGY_SPECS):
            probe_path = (
                cls.native_probe_root
                / "probes"
                / f"slot-{index:02d}"
                / "throughput_gate.json"
            )
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            cls._write_json(probe_path, {"mode": mode, "slot": index})
            cls.native_probe_rows.append(
                {
                    "mode": mode,
                    "path": str(probe_path),
                    "sha256": _sha(probe_path),
                    "bytes": probe_path.stat().st_size,
                }
            )
        cls.native_failure_path = (
            cls.native_quality_root
            / "training"
            / "quality_p1_w1_master"
            / "failure.json"
        )
        cls.native_failure_path.parent.mkdir(parents=True, exist_ok=True)
        cls._write_json(
            cls.native_failure_path,
            {
                "format": contract.SHORT_QUALITY_STATUS_FORMAT,
                "status": "failed",
                "error_type": "AdaptationContractError",
                "error": selection.W1_FAILURE_ERROR,
                "failed_unix": 1_700_000_000.0,
                "run_purpose": contract.RUN_PURPOSE_SHORT_QUALITY,
                "target_epochs": list(official_selector.W1_REFERENCE_EPOCHS),
            },
        )
        cls.native_unavailable = {
            "format": selection.TRAINER_NATIVE_UNAVAILABLE_FORMAT,
            "status": "unavailable_not_authoritative",
            "role": "audit_only",
            "authoritative_for_winner": False,
            "reason_code": selection.TRAINER_NATIVE_UNAVAILABLE_REASON,
            "topology_gate_spec_sha256": selection.TOPOLOGY_GATE_SHA256,
            "quality_gate_spec_sha256": selection.QUALITY_GATE_SHA256,
            "probe_campaign_root": str(cls.native_probe_root),
            "quality_campaign_root": str(cls.native_quality_root),
            "probes": cls.native_probe_rows,
            "blocking_failure": {
                "mode": contract.OFFICIAL_W1_REFERENCE_MODE,
                "path": str(cls.native_failure_path),
                "sha256": _sha(cls.native_failure_path),
                "bytes": cls.native_failure_path.stat().st_size,
            },
        }
        cls.native_unavailable["receipt_sha256"] = selection._canonical_sha(
            cls.native_unavailable
        )
        cls.native_unavailable_path = cls.root / "native-unavailable.json"
        cls._write_json(cls.native_unavailable_path, cls.native_unavailable)
        cls.native_unavailable_sha = _sha(cls.native_unavailable_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def _formal_patches(cls):
        from contextlib import ExitStack

        result = ExitStack()
        result.enter_context(
            mock.patch.object(
                official_selector.formal_validation,
                "validate_val_inputs",
                side_effect=_FakeFormalValidation.validate_val_inputs,
            )
        )
        if hasattr(cls, "native_selection"):
            result.enter_context(
                mock.patch.object(
                    selection,
                    "_validate_trainer_native_selection",
                    return_value=(
                        copy.deepcopy(cls.native_selection),
                        {
                            "path": str(cls.native_path),
                            "sha256": cls.native_sha,
                            "bytes": cls.native_path.stat().st_size,
                        },
                    ),
                )
            )
        result.enter_context(
            mock.patch.object(
                selection,
                "_validation_source_authority",
                return_value={
                    "origin": selection.SOURCE_ORIGIN,
                    "commit": selection.RUNTIME_VALIDATION_SOURCE_COMMIT,
                    "tree": selection.RUNTIME_VALIDATION_SOURCE_TREE,
                    "clean": True,
                    "detached": True,
                    "local_branches_at_commit": [],
                    "selector_sha256": selection.OFFICIAL_SELECTOR_SHA256,
                    "training_contract_sha256": (
                        selection.OFFICIAL_TRAIN_CONTRACT_SHA256
                    ),
                    "validation_contract_sha256": (
                        selection.OFFICIAL_VALIDATION_CONTRACT_SHA256
                    ),
                    "diffsheg_adapter_sha256": (
                        selection.OFFICIAL_DIFFSHEG_ADAPTER_SHA256
                    ),
                },
            )
        )
        result.enter_context(
            mock.patch.object(
                official_selector.formal_validation,
                "validate_pipeline",
                side_effect=_FakeFormalValidation.validate_pipeline,
            )
        )
        result.enter_context(
            mock.patch.object(
                official_selector.formal_validation,
                "validate_val_inference_lineage",
                side_effect=_FakeFormalValidation.validate_val_inference_lineage,
            )
        )
        result.enter_context(
            mock.patch.object(
                official_selector.formal_validation,
                "validate_diffsheg_report",
                side_effect=_FakeFormalValidation.validate_diffsheg_report,
            )
        )
        result.enter_context(
            mock.patch.object(
                selection,
                "_validation_pipeline_authority",
                return_value={
                    "origin": selection.SOURCE_ORIGIN,
                    "source_root": "/verified/validation/4066f20",
                    "commit": selection.PIPELINE_EVIDENCE_SOURCE_COMMIT,
                    "tree": selection.PIPELINE_EVIDENCE_SOURCE_TREE,
                    "clean": True,
                    "detached": True,
                    "local_branches_at_commit": [],
                },
            )
        )
        return result

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.write_bytes(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    def _select(
        self,
        specs,
        *,
        output: Path,
        **kwargs,
    ):
        kwargs.setdefault("project_root", REPOSITORY)
        kwargs.setdefault("selector", official_selector)
        kwargs.setdefault("contract", contract)
        kwargs.setdefault("selection_protocol", self.protocol_path)
        kwargs.setdefault(
            "expected_selection_protocol_sha256", selection.PROTOCOL_SHA256
        )
        kwargs.setdefault("trainer_native_selection", self.native_path)
        kwargs.setdefault(
            "expected_trainer_native_selection_sha256", self.native_sha
        )
        return selection.select_two_reports(specs, output=output, **kwargs)

    def test_positive_replays_two_v3_chains_and_writes_canonical_result(self) -> None:
        output = self.root / "selection-positive.json"
        with self._formal_patches():
            payload = self._select(
                self.report_specs,
                output=output,
            )
        self.assertEqual(
            set(payload),
            {
                "format",
                "status",
                "selection_protocol",
                "selection_tuple",
                "validation_split",
                "test_visible",
                "test_measurements_authorized",
                "training_source_authority",
                "validation_source_authority",
                "validation_pipeline_source",
                "quality_reports",
                "quality_report_sha256",
                "topology_independent_input_sha256",
                "common_authority",
                "results",
                "winner",
                "trainer_native_selection",
                "formal_training",
                "receipt_sha256",
            },
        )
        self.assertEqual(len(payload["results"]), 12)
        self.assertEqual(payload["test_measurements_authorized"], 0)
        self.assertFalse(payload["test_visible"])
        self.assertEqual(payload["formal_training"]["target_epochs"], 400)
        self.assertTrue(payload["formal_training"]["fresh"])
        self.assertEqual(
            payload["training_source_authority"]["commit"],
            selection.TRAINING_SOURCE_COMMIT,
        )
        self.assertEqual(
            payload["validation_source_authority"]["commit"],
            selection.RUNTIME_VALIDATION_SOURCE_COMMIT,
        )
        self.assertNotEqual(
            payload["training_source_authority"]["commit"],
            payload["validation_source_authority"]["commit"],
        )
        self.assertEqual(payload["winner"]["epoch"], 1)
        self.assertEqual(
            payload["winner"]["topology_mode"], selection.MODE_P2
        )
        self.assertEqual(
            payload["trainer_native_selection"],
            {
                "role": "audit_only",
                "authoritative_for_winner": False,
                "status": "available_not_authoritative",
                "path": str(self.native_path),
                "format": contract.TOPOLOGY_SELECTION_FORMAT,
                "receipt_sha256": self.native_selection["receipt_sha256"],
                "selected_mode": self.native_selection["selected"]["mode"],
            },
        )
        loaded = json.loads(output.read_text(encoding="utf-8"))
        claimed = loaded.pop("receipt_sha256")
        self.assertEqual(claimed, selection._canonical_sha(loaded))

    def test_protocol_is_exactly_hash_and_semantic_pinned(self) -> None:
        payload, artifact = selection.load_protocol(
            self.protocol_path, selection.PROTOCOL_SHA256
        )
        self.assertEqual(payload, selection.EXPECTED_PROTOCOL)
        self.assertEqual(artifact["sha256"], selection.PROTOCOL_SHA256)
        self.assertEqual(
            _sha(
                self.official_repository
                / "scripts/show_base/select_base_training_topology.py"
            ),
            selection.OFFICIAL_SELECTOR_SHA256,
        )
        self.assertEqual(
            _sha(
                self.official_repository
                / "scripts/show_base/train_base_official_adapt_long.py"
            ),
            selection.OFFICIAL_TRAIN_CONTRACT_SHA256,
        )
        self.assertEqual(
            _sha(
                self.official_repository
                / "scripts/show_base/base_long_val_contract.py"
            ),
            selection.OFFICIAL_VALIDATION_CONTRACT_SHA256,
        )
        self.assertEqual(
            _sha(
                self.official_repository
                / "scripts/show_base/select_base_official_adapt.py"
            ),
            selection.OFFICIAL_DIFFSHEG_ADAPTER_SHA256,
        )
        self.assertEqual(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPOSITORY),
                    "rev-parse",
            f"{selection.RUNTIME_VALIDATION_SOURCE_COMMIT}^{{tree}}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            selection.RUNTIME_VALIDATION_SOURCE_TREE,
        )

    def test_protocol_path_is_cli_bound_and_linux_portable(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn(
            'PROTOCOL_PATH = Path("/private/tmp/', source
        )
        portable = self.root / "linux-tmp-protocol-copy.json"
        portable.write_bytes(self.protocol_path.read_bytes())
        payload, artifact = selection.load_protocol(
            portable, selection.PROTOCOL_SHA256
        )
        self.assertEqual(payload, selection.EXPECTED_PROTOCOL)
        self.assertEqual(artifact["path"], str(portable))
        with self.assertRaises(selection.SelectionError):
            selection.load_protocol(portable, "0" * 64)
        symlink = self.root / "protocol-symlink.json"
        symlink.symlink_to(portable)
        with self.assertRaises(selection.SelectionError):
            selection.load_protocol(symlink, selection.PROTOCOL_SHA256)
        parser = selection._parser()
        parsed = parser.parse_args(
            [
                "--official-project-root",
                str(REPOSITORY),
                "--selection-protocol",
                str(self.protocol_path),
                "--expected-selection-protocol-sha256",
                selection.PROTOCOL_SHA256,
                "--quality-report",
                *[str(value) for value in self.report_specs[0]],
                "--output",
                str(self.root / "native-optional-cli.json"),
            ]
        )
        self.assertIsNone(parsed.trainer_native_selection)
        self.assertIsNone(parsed.trainer_native_unavailable_receipt)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--official-project-root",
                    str(REPOSITORY),
                    "--trainer-native-selection",
                    str(self.native_path),
                    "--expected-trainer-native-selection-sha256",
                    self.native_sha,
                    "--quality-report",
                    *[str(value) for value in self.report_specs[0]],
                    "--output",
                    str(self.root / "missing-protocol-cli.json"),
                ]
            )

    def test_trainer_native_selection_is_freshly_replayed(self) -> None:
        probe_by_mode = {
            row["mode"]: row for row in self.native_probes
        }
        report_by_mode = {
            row["mode"]: row for row in self.native_reports
        }
        skip_by_mode = {
            row["mode"]: row
            for row in self.native_selection["quality_skips"]
        }

        class ReplaySelector:
            @staticmethod
            def _row(rows, mode, path, expected_sha256, path_key):
                row = rows.get(mode)
                if (
                    row is None
                    or str(path) != row[path_key]
                    or expected_sha256
                    != row[
                        "receipt_sha256"
                        if path_key == "receipt_path"
                        else "report_sha256"
                    ]
                ):
                    raise ValueError("native evidence binding changed")
                return copy.deepcopy(row)

            @classmethod
            def validate_probe(cls, mode, path, expected_sha256, **_kwargs):
                return cls._row(
                    probe_by_mode, mode, path, expected_sha256, "report_path"
                )

            @classmethod
            def validate_quality_report(
                cls, mode, path, expected_sha256, **_kwargs
            ):
                return cls._row(
                    report_by_mode, mode, path, expected_sha256, "report_path"
                )

            @classmethod
            def validate_quality_skip(
                cls, mode, path, expected_sha256, **_kwargs
            ):
                return cls._row(
                    skip_by_mode, mode, path, expected_sha256, "receipt_path"
                )

            select_topology = staticmethod(official_selector.select_topology)

        value, artifact = selection._validate_trainer_native_selection(
            self.native_path,
            self.native_sha,
            selector=ReplaySelector,
            contract=contract,
        )
        self.assertEqual(value, self.native_selection)
        self.assertEqual(artifact["sha256"], self.native_sha)

        for attack in ("missing_probe", "replaced_report", "forged_selected"):
            changed = copy.deepcopy(self.native_selection)
            if attack == "missing_probe":
                changed["probes"].pop()
            elif attack == "replaced_report":
                changed["quality_reports"][1] = copy.deepcopy(
                    changed["quality_reports"][0]
                )
            else:
                selected = copy.deepcopy(changed["selected"])
                selected["mode"] = next(
                    mode
                    for mode in contract.TOPOLOGY_SPECS
                    if mode != selected["mode"]
                )
                changed["selected"] = selected
            unsigned = dict(changed)
            unsigned.pop("receipt_sha256")
            changed["receipt_sha256"] = selection._canonical_sha(unsigned)
            attacked = self.root / f"native-{attack}.json"
            self._write_json(attacked, changed)
            with self.subTest(attack=attack):
                with self.assertRaises(selection.SelectionError):
                    selection._validate_trainer_native_selection(
                        attacked,
                        _sha(attacked),
                        selector=ReplaySelector,
                        contract=contract,
                    )

    def test_unavailable_native_audit_freshly_replays_nine_probes_and_w1_failure(
        self,
    ) -> None:
        rows = {row["mode"]: row for row in self.native_probe_rows}

        class ReplayUnavailableSelector:
            W1_REFERENCE_EPOCHS = official_selector.W1_REFERENCE_EPOCHS

            @staticmethod
            def validate_probe(mode, path, expected_sha256, **kwargs):
                self.assertEqual(
                    kwargs, {"gate_spec_sha256": selection.TOPOLOGY_GATE_SHA256}
                )
                row = rows[mode]
                self.assertEqual(str(path), row["path"])
                self.assertEqual(expected_sha256, row["sha256"])
                return {
                    "mode": mode,
                    "report_path": str(path),
                    "report_sha256": expected_sha256,
                }

        audit, artifact = selection._validate_trainer_native_unavailable_receipt(
            self.native_unavailable_path,
            self.native_unavailable_sha,
            selector=ReplayUnavailableSelector,
            contract=contract,
        )
        self.assertEqual(
            audit,
            {
                "role": "audit_only",
                "authoritative_for_winner": False,
                "status": "unavailable_not_authoritative",
                "reason_code": selection.TRAINER_NATIVE_UNAVAILABLE_REASON,
                "evidence_receipt": {
                    "path": str(self.native_unavailable_path),
                    "sha256": self.native_unavailable_sha,
                    "bytes": self.native_unavailable_path.stat().st_size,
                    "receipt_sha256": self.native_unavailable[
                        "receipt_sha256"
                    ],
                },
                "probe_modes": list(contract.TOPOLOGY_SPECS),
                "blocking_mode": contract.OFFICIAL_W1_REFERENCE_MODE,
            },
        )
        self.assertEqual(artifact["sha256"], self.native_unavailable_sha)

        attacks = {}
        missing = copy.deepcopy(self.native_unavailable)
        missing["probes"].pop()
        attacks["missing_probe"] = missing
        reordered = copy.deepcopy(self.native_unavailable)
        reordered["probes"][0], reordered["probes"][1] = (
            reordered["probes"][1],
            reordered["probes"][0],
        )
        attacks["reordered_probes"] = reordered
        wrong_bytes = copy.deepcopy(self.native_unavailable)
        wrong_bytes["probes"][0]["bytes"] += 1
        attacks["wrong_probe_bytes"] = wrong_bytes
        wrong_failure = copy.deepcopy(self.native_unavailable)
        wrong_failure["blocking_failure"]["sha256"] = "0" * 64
        attacks["wrong_failure_sha"] = wrong_failure
        authoritative = copy.deepcopy(self.native_unavailable)
        authoritative["authoritative_for_winner"] = True
        attacks["authoritative_spoof"] = authoritative
        for name, changed in attacks.items():
            changed.pop("receipt_sha256", None)
            changed["receipt_sha256"] = selection._canonical_sha(changed)
            attacked = self.root / f"native-unavailable-{name}.json"
            self._write_json(attacked, changed)
            with self.subTest(name=name), self.assertRaises(
                selection.SelectionError
            ):
                selection._validate_trainer_native_unavailable_receipt(
                    attacked,
                    _sha(attacked),
                    selector=ReplayUnavailableSelector,
                    contract=contract,
                )

    def test_native_audit_absence_is_explicit_and_cannot_block_or_change_winner(
        self,
    ) -> None:
        available_output = self.root / "native-available-selection.json"
        absent_output = self.root / "native-absent-selection.json"
        with self._formal_patches():
            available = self._select(
                self.report_specs,
                output=available_output,
            )
            absent = self._select(
                self.report_specs,
                output=absent_output,
                trainer_native_selection=None,
                expected_trainer_native_selection_sha256=None,
            )
        self.assertEqual(absent["winner"], available["winner"])
        self.assertEqual(absent["results"], available["results"])
        self.assertEqual(
            absent["trainer_native_selection"],
            {
                "role": "audit_only",
                "authoritative_for_winner": False,
                "status": "unavailable_not_authoritative",
                "reason_code": selection.TRAINER_NATIVE_NOT_PROVIDED_REASON,
                "evidence_receipt": None,
                "probe_modes": [],
                "blocking_mode": None,
            },
        )

    def test_native_audit_alternatives_are_paired_and_mutually_exclusive(
        self,
    ) -> None:
        base = {
            "selector": official_selector,
            "contract": contract,
            "validate_specs": False,
            "trainer_native_selection": None,
            "expected_trainer_native_selection_sha256": None,
            "trainer_native_unavailable_receipt": None,
            "expected_trainer_native_unavailable_receipt_sha256": None,
        }
        for changed in (
            {"trainer_native_selection": self.native_path},
            {
                "expected_trainer_native_unavailable_receipt_sha256": (
                    self.native_unavailable_sha
                )
            },
            {
                "trainer_native_selection": self.native_path,
                "expected_trainer_native_selection_sha256": self.native_sha,
                "trainer_native_unavailable_receipt": (
                    self.native_unavailable_path
                ),
                "expected_trainer_native_unavailable_receipt_sha256": (
                    self.native_unavailable_sha
                ),
            },
        ):
            kwargs = {**base, **changed}
            with self.subTest(changed=sorted(changed)), self.assertRaises(
                selection.SelectionError
            ):
                selection._trainer_native_audit(**kwargs)

    def test_report_file_sha_bytes_and_payload_sha_are_independently_pinned(self) -> None:
        mode, path, digest, size, payload = self.report_specs[0]
        for changed, label in (
            ((mode, path, "0" * 64, size, payload), "file SHA"),
            ((mode, path, digest, size + 1, payload), "byte count"),
            ((mode, path, digest, size, "0" * 64), "payload SHA"),
        ):
            with self.subTest(label=label), self.assertRaises(selection.SelectionError):
                selection._prevalidate_report(*changed)

    def test_embedded_artifact_requires_exact_positive_integer_bytes(self) -> None:
        path = self.root / "embedded-artifact.json"
        self._write_json(path, {"value": 1})
        base = {
            "path": str(path),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
        }
        self.assertEqual(
            selection._artifact_payload(base, "embedded fixture"),
            {"value": 1},
        )
        for value in (None, True, 0, path.stat().st_size + 1):
            changed = dict(base)
            if value is None:
                changed.pop("bytes")
            else:
                changed["bytes"] = value
            with self.subTest(value=value), self.assertRaises(
                selection.SelectionError
            ):
                selection._artifact_payload(changed, "embedded fixture")

    def test_validation_pipeline_source_must_be_exact_4066f20(self) -> None:
        def receipt(commit: str, name: str):
            payload = {
                "format": "fixture-fresh-validation-pipeline",
                "split": "val",
                "test_visible": False,
                "source": {
                    "origin": selection.SOURCE_ORIGIN,
                    "source_root": "/verified/validation/4066f20",
                    "commit": commit,
                    "tree": selection.PIPELINE_EVIDENCE_SOURCE_TREE,
                    "clean": True,
                    "detached": True,
                    "local_branches_at_commit": [],
                },
            }
            payload["receipt_payload_sha256"] = selection._canonical_sha(
                payload
            )
            path = self.root / name
            encoded = json.dumps(
                payload, sort_keys=True, allow_nan=False
            ).encode("utf-8") + b"\n"
            path.write_bytes(encoded)
            return {
                "path": str(path),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
                "receipt_payload_sha256": payload[
                    "receipt_payload_sha256"
                ],
            }

        good = {"pipeline_receipt": receipt(
            selection.PIPELINE_EVIDENCE_SOURCE_COMMIT,
            "pipeline-4066f20.json",
        )}
        authority = selection._validation_pipeline_authority(
            good, selection.MODE_P1
        )
        self.assertEqual(
            authority["commit"], selection.PIPELINE_EVIDENCE_SOURCE_COMMIT
        )
        bad = {"pipeline_receipt": receipt(
            selection.TRAINING_SOURCE_COMMIT, "pipeline-5b84075.json"
        )}
        with self.assertRaises(selection.SelectionError):
            selection._validation_pipeline_authority(
                bad, selection.MODE_P1
            )

    def test_missing_duplicate_or_unknown_topology_is_rejected(self) -> None:
        dummy = types.SimpleNamespace()
        for specs in (
            self.report_specs[:1],
            [self.report_specs[0], self.report_specs[0]],
            [("unknown", *self.report_specs[0][1:]), self.report_specs[1]],
        ):
            with self.subTest(specs=len(specs)), self.assertRaises(selection.SelectionError):
                self._select(
                    specs,
                    output=self.root / f"bad-modes-{len(specs)}.json",
                    selector=dummy,
                    contract=dummy,
                    validate_specs=False,
                )

    def test_nonfinite_fgd_is_rejected_before_output(self) -> None:
        class FakeSelector:
            @staticmethod
            def validate_quality_report(mode, path, expected_sha256, **kwargs):
                del path, expected_sha256, kwargs
                return {
                    "mode": mode,
                    "quality_role": "candidate_quality",
                    "reference_only": False,
                    "report_sha256": next(
                        spec[2] for spec in self.report_specs if spec[0] == mode
                    ),
                    "candidate_fgd": {
                        str(epoch): 1.0 for epoch in selection.CANDIDATE_EPOCHS
                    },
                    "topology_independent_input_sha256": "a" * 64,
                    "val_inputs_receipt": {"same": True},
                    "pipeline_receipt": {"same": True},
                    "candidates": [
                        {
                            "diffsheg_fgd": (
                                float("inf") if epoch == 4 and mode == selection.MODE_P1 else 1.0
                            ),
                            "candidate_checkpoint": {"sha256": f"{epoch:064x}"},
                            "provenance": {
                                "epoch": epoch,
                                "topology_mode": mode,
                                "split": "val",
                                "test_visible": False,
                            },
                        }
                        for epoch in selection.CANDIDATE_EPOCHS
                    ],
                }

        output = self.root / "nonfinite.json"
        with (
            mock.patch.object(selection, "_load_mode_frozen", return_value={}),
            mock.patch.object(
                selection,
                "_authority_projection",
                return_value={"common": "a" * 64},
            ),
            self.assertRaises(selection.SelectionError),
        ):
            self._select(
                self.report_specs,
                output=output,
                selector=FakeSelector(),
                contract=types.SimpleNamespace(),
                validate_specs=False,
            )
        self.assertFalse(output.exists())

    def test_bool_cannot_spoof_candidate_or_provenance_epoch(self) -> None:
        class FakeSelector:
            attack_location = "row"

            @classmethod
            def validate_quality_report(
                cls, mode, path, expected_sha256, **kwargs
            ):
                del path, expected_sha256, kwargs
                candidates = []
                for epoch in selection.CANDIDATE_EPOCHS:
                    row_epoch = epoch
                    provenance_epoch = epoch
                    if epoch == 1 and mode == selection.MODE_P1:
                        if cls.attack_location == "row":
                            row_epoch = True
                        else:
                            provenance_epoch = True
                    candidates.append(
                        {
                            "epoch": row_epoch,
                            "diffsheg_fgd": 1.0,
                            "candidate_checkpoint": {
                                "sha256": f"{epoch:064x}"
                            },
                            "provenance": {
                                "epoch": provenance_epoch,
                                "topology_mode": mode,
                                "split": "val",
                                "test_visible": False,
                            },
                        }
                    )
                return {
                    "mode": mode,
                    "quality_role": "candidate_quality",
                    "reference_only": False,
                    "report_sha256": next(
                        spec[2]
                        for spec in self.report_specs
                        if spec[0] == mode
                    ),
                    "candidate_fgd": {
                        str(epoch): 1.0
                        for epoch in selection.CANDIDATE_EPOCHS
                    },
                    "topology_independent_input_sha256": "a" * 64,
                    "val_inputs_receipt": {"same": True},
                    "pipeline_receipt": {"same": True},
                    "candidates": candidates,
                }

        for location in ("row", "provenance"):
            FakeSelector.attack_location = location
            output = self.root / f"bool-epoch-{location}.json"
            with (
                mock.patch.object(
                    selection, "_load_mode_frozen", return_value={}
                ),
                mock.patch.object(
                    selection,
                    "_authority_projection",
                    return_value={"common": "a" * 64},
                ),
                self.subTest(location=location),
                self.assertRaises(selection.SelectionError),
            ):
                self._select(
                    self.report_specs,
                    output=output,
                    selector=FakeSelector,
                    contract=types.SimpleNamespace(),
                    validate_specs=False,
                )
            self.assertFalse(output.exists())

    def test_rehashed_raw_report_bool_epoch_attack_is_rejected(self) -> None:
        for location in ("row", "provenance"):
            specs = list(self.report_specs)
            mode, source, _digest, _size, _payload_sha = specs[0]
            payload = json.loads(source.read_text(encoding="utf-8"))
            if location == "row":
                payload["candidates"][0]["epoch"] = True
            else:
                payload["candidates"][0]["provenance"]["epoch"] = True
            payload.pop("receipt_sha256")
            payload["receipt_sha256"] = selection._canonical_sha(payload)
            attacked = self.root / f"raw-bool-epoch-{location}.json"
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8") + b"\n"
            attacked.write_bytes(encoded)
            specs[0] = (
                mode,
                attacked.resolve(strict=True),
                hashlib.sha256(encoded).hexdigest(),
                len(encoded),
                payload["receipt_sha256"],
            )
            output = self.root / f"raw-bool-output-{location}.json"
            with (
                self._formal_patches(),
                self.subTest(location=location),
                self.assertRaises(selection.SelectionError),
            ):
                self._select(
                    specs,
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_source_data_vq_init_or_runtime_drift_is_rejected(self) -> None:
        with self._formal_patches():
            validated = {
                mode: official_selector.validate_quality_report(
                    mode,
                    path,
                    digest,
                    quality_gate_spec_sha256=selection.QUALITY_GATE_SHA256,
                    topology_gate_spec_sha256=selection.TOPOLOGY_GATE_SHA256,
                )
                for mode, path, digest, _size, _payload in self.report_specs
            }
        frozen = {
            mode: selection._load_mode_frozen(validated[mode], mode)
            for mode in selection.MODES
        }
        baseline = selection._authority_projection(
            frozen[selection.MODE_P1], selection.MODE_P1, contract
        )
        for section, key, value in (
            ("source", "entrypoint_sha256", "f" * 64),
            ("dataset", "data_mdb_sha256", "e" * 64),
            ("speaker_initialization", "rows", [3, 2, 1, 0]),
            ("protocol", "loader_workers", 999),
        ):
            changed = json.loads(json.dumps(frozen[selection.MODE_P2]))
            changed[section][key] = value
            projection = selection._authority_projection(
                changed, selection.MODE_P2, contract
            )
            with self.subTest(section=section):
                self.assertNotEqual(projection, baseline)

    def test_output_is_create_new_and_never_overwritten(self) -> None:
        output = self.root / "exclusive-output.json"
        selection._write_new_json(output, {"one": 1})
        before = output.read_bytes()
        with self.assertRaises(selection.SelectionError):
            selection._write_new_json(output, {"two": 2})
        self.assertEqual(output.read_bytes(), before)

    def test_loader_rejects_preexisting_wrong_path_contract_module(self) -> None:
        name = "scripts.show_base.train_base_official_adapt_long"
        polluted = types.ModuleType(name)
        polluted.__file__ = str(SCRIPT)
        with (
            mock.patch.dict(sys.modules, {name: polluted}),
            self.assertRaisesRegex(selection.SelectionError, "pollute"),
        ):
            selection.load_official_modules(self.official_repository)

    def test_loader_rejects_dirty_clean_root(self) -> None:
        sentinel = self.official_repository / ".v14-selector-dirty-negative"
        self.assertFalse(sentinel.exists())
        try:
            sentinel.write_text("negative fixture\n", encoding="utf-8")
            with self.assertRaisesRegex(
                selection.SelectionError, "Git source authority"
            ):
                selection.load_official_modules(self.official_repository)
        finally:
            sentinel.unlink(missing_ok=True)
        self.assertEqual(
            selection._git(
                self.official_repository,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            "",
        )

    def test_loader_rejects_wrong_blob_even_if_git_queries_are_clean(self) -> None:
        selector_path = (
            self.official_repository
            / "scripts/show_base/select_base_training_topology.py"
        )
        original = selection._read_regular

        def wrong_selector_bytes(path: Path, label: str):
            resolved, data = original(path, label)
            if resolved == selector_path:
                return resolved, data + b"# forged\n"
            return resolved, data

        with (
            mock.patch.object(
                selection, "_read_regular", side_effect=wrong_selector_bytes
            ),
            self.assertRaisesRegex(selection.SelectionError, "selector source SHA"),
        ):
            selection.load_official_modules(self.official_repository)

    def test_loader_rejects_symbolic_branch_head(self) -> None:
        original = selection._git_result

        def symbolic_branch(root: Path, *args: str):
            if args == ("symbolic-ref", "-q", "HEAD"):
                return subprocess.CompletedProcess(
                    ["git", *args], 0, "refs/heads/forged\n", ""
                )
            return original(root, *args)

        with (
            mock.patch.object(
                selection, "_git_result", side_effect=symbolic_branch
            ),
            self.assertRaisesRegex(selection.SelectionError, "detached HEAD"),
        ):
            selection.load_official_modules(self.official_repository)


if __name__ == "__main__":
    unittest.main()
