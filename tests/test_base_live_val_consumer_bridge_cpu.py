from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from scripts.show_base import base_live_val_consumer_bridge as BRIDGE
from scripts.show_base import supervise_base_v14_live_validation as SUPERVISOR


class BaseLiveValConsumerBridgeCpuTest(unittest.TestCase):
    @staticmethod
    def _git_blob_sha1(payload: bytes) -> str:
        header = f"blob {len(payload)}\0".encode("ascii")
        return hashlib.sha1(header + payload).hexdigest()

    @classmethod
    def _runtime_projection_fixture(cls, root: Path):
        evidence_root = root / "evidence"
        runtime_root = root / "runtime"
        repository = Path(BRIDGE.__file__).resolve().parents[2]
        evidence_files = {}
        runtime_files = {}
        for relative in BRIDGE.RUNTIME_EXECUTION_SOURCE_FILES:
            evidence_payload = subprocess.run(
                [
                    "git", "-C", str(repository), "show",
                    f"{BRIDGE.VALIDATION_EVIDENCE_SOURCE_COMMIT}:{relative}",
                ],
                check=True,
                capture_output=True,
            ).stdout
            runtime_payload = subprocess.run(
                [
                    "git", "-C", str(repository), "show",
                    f"{BRIDGE.RUNTIME_VALIDATION_SOURCE_COMMIT}:{relative}",
                ],
                check=True,
                capture_output=True,
            ).stdout
            if relative in BRIDGE.RUNTIME_EXECUTION_CHANGED_FILES:
                expected_pair = BRIDGE.RUNTIME_EXECUTION_CHANGED_SHA256[relative]
                if (
                    hashlib.sha256(evidence_payload).hexdigest(),
                    hashlib.sha256(runtime_payload).hexdigest(),
                ) != expected_pair:
                    raise AssertionError(f"changed proof pair drifted: {relative}")
            elif evidence_payload != runtime_payload:
                raise AssertionError(f"unchanged fixture bytes drifted: {relative}")
            evidence_path = evidence_root / relative
            runtime_path = runtime_root / relative
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_bytes(evidence_payload)
            runtime_path.write_bytes(runtime_payload)
            evidence_files[relative] = {
                "path": str(evidence_path),
                "sha256": hashlib.sha256(evidence_payload).hexdigest(),
                "bytes": len(evidence_payload),
                "git_mode": "100644",
                "git_blob_sha1": cls._git_blob_sha1(evidence_payload),
            }
            runtime_files[relative] = {
                "path": str(runtime_path),
                "sha256": hashlib.sha256(runtime_payload).hexdigest(),
                "bytes": len(runtime_payload),
                "git_mode": "100644",
                "git_blob_sha1": cls._git_blob_sha1(runtime_payload),
            }

        evidence_source = {
            "origin": BRIDGE.EXPECTED_ORIGIN,
            "source_root": str(evidence_root),
            "commit": BRIDGE.VALIDATION_EVIDENCE_SOURCE_COMMIT,
            "tree": BRIDGE.VALIDATION_EVIDENCE_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        runtime_source = {
            "origin": BRIDGE.EXPECTED_ORIGIN,
            "source_root": str(runtime_root),
            "commit": BRIDGE.RUNTIME_VALIDATION_SOURCE_COMMIT,
            "tree": BRIDGE.RUNTIME_VALIDATION_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        evidence_pipeline = {
            "format": "projection-fixture",
            "source": evidence_source,
            "source_closure": evidence_files,
            "inference_entrypoint": evidence_files[
                "scripts/show_base/run_base_val_inference.py"
            ],
            "inference_helper": evidence_files[
                "scripts/show_base/semtalk_base_inference_core.py"
            ],
        }
        evidence_pipeline["receipt_payload_sha256"] = BRIDGE._payload_sha(
            evidence_pipeline
        )
        pipeline_reference = {
            "path": str(root / "pipeline.json"),
            "sha256": "a" * 64,
            "receipt_payload_sha256": evidence_pipeline[
                "receipt_payload_sha256"
            ],
        }
        runtime_receipt = {**runtime_source, "files": runtime_files}
        authority = {
            "validation_evidence_source": evidence_source,
            "runtime_validation_source": runtime_source,
            "runtime_validation_proof": (
                BRIDGE._expected_runtime_validation_proof()
            ),
            "pipeline_receipt": pipeline_reference,
        }
        contract = SimpleNamespace(
            validate_pipeline=lambda *_args, **_kwargs: (
                pipeline_reference,
                evidence_pipeline,
            )
        )

        def modules_for(receipt):
            def build(observed_root: Path):
                if observed_root != runtime_root:
                    raise AssertionError("runtime builder received another root")
                return copy.deepcopy(receipt)

            return {
                "contract": contract,
                "legacy": SimpleNamespace(
                    DIFFSHEG_PRIMARY_PIPELINE_SOURCE_FILES=(
                        BRIDGE.RUNTIME_EXECUTION_SOURCE_FILES
                    ),
                    build_fresh_pipeline_source_receipt=build,
                ),
            }

        return {
            "evidence_root": evidence_root,
            "runtime_root": runtime_root,
            "evidence_pipeline": evidence_pipeline,
            "runtime_receipt": runtime_receipt,
            "authority": authority,
            "modules_for": modules_for,
        }

    def test_scripts_namespace_is_single_root_and_rejects_preload(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-scripts-namespace-", dir="/private/tmp"
        ) as raw:
            root = Path(raw).resolve()
            scripts_root = root / "scripts"
            scripts_root.mkdir()
            prefixes = ("scripts", "models", "dataloaders", "utils")
            project_names = [
                name
                for name in tuple(sys.modules)
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in prefixes
                )
            ]
            saved = {name: sys.modules.pop(name) for name in project_names}
            try:
                BRIDGE._bind_exact_scripts_namespace(root)
                namespace = sys.modules["scripts"]
                self.assertIsNone(namespace.__file__)
                self.assertEqual(namespace.__path__, [str(scripts_root)])
                self.assertEqual(
                    namespace.__spec__.submodule_search_locations,
                    [str(scripts_root)],
                )
                BRIDGE._project_modules_are_from(root)
                BRIDGE._bind_exact_scripts_namespace(root)

                sys.modules.pop("scripts")
                polluted = ModuleType("scripts")
                polluted.__file__ = None
                polluted.__path__ = [str(root / "other"), "/site/scripts"]
                sys.modules["scripts"] = polluted
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError,
                    "preloaded scripts namespace is not exactly runtime-bound",
                ):
                    BRIDGE._bind_exact_scripts_namespace(root)
            finally:
                for name in tuple(sys.modules):
                    if any(
                        name == prefix or name.startswith(prefix + ".")
                        for prefix in prefixes
                    ):
                        sys.modules.pop(name, None)
                sys.modules.update(saved)

    def test_scripts_namespace_prevents_python312_namespace_merging(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-scripts-merge-", dir="/private/tmp"
        ) as raw:
            root = Path(raw).resolve()
            runtime = root / "runtime"
            launch = root / "launch"
            site = root / "site"
            for source in (runtime, launch, site):
                (source / "scripts").mkdir(parents=True)
            prefixes = ("scripts", "models", "dataloaders", "utils")
            project_names = [
                name
                for name in tuple(sys.modules)
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in prefixes
                )
            ]
            saved = {name: sys.modules.pop(name) for name in project_names}
            old_path = list(sys.path)
            try:
                sys.path[:] = [str(launch), str(site), *old_path]
                vanilla = importlib.import_module("scripts")
                self.assertEqual(
                    list(vanilla.__path__)[:2],
                    [str(launch / "scripts"), str(site / "scripts")],
                )
                self.assertGreaterEqual(len(vanilla.__path__), 2)
                sys.modules.pop("scripts")
                BRIDGE._bind_exact_scripts_namespace(runtime)
                namespace = sys.modules["scripts"]
                self.assertEqual(
                    list(namespace.__path__), [str(runtime / "scripts")]
                )
                BRIDGE._project_modules_are_from(runtime)
            finally:
                sys.path[:] = old_path
                for name in tuple(sys.modules):
                    if any(
                        name == prefix or name.startswith(prefix + ".")
                        for prefix in prefixes
                    ):
                        sys.modules.pop(name, None)
                sys.modules.update(saved)

    def test_loader_binds_scripts_before_its_first_runtime_import(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-loader-order-", dir="/private/tmp"
        ) as raw:
            runtime = Path(raw).resolve()
            scripts_root = runtime / "scripts"
            scripts_root.mkdir()
            prefixes = ("scripts", "models", "dataloaders", "utils")
            project_names = [
                name
                for name in tuple(sys.modules)
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in prefixes
                )
            ]
            saved = {name: sys.modules.pop(name) for name in project_names}
            old_path = list(sys.path)
            observed = []

            def stop_after_binding(name: str):
                namespace = sys.modules["scripts"]
                observed.append(name)
                self.assertEqual(namespace.__path__, [str(scripts_root)])
                self.assertEqual(
                    namespace.__spec__.submodule_search_locations,
                    [str(scripts_root)],
                )
                raise RuntimeError("stop after exact namespace binding")

            try:
                with mock.patch.object(
                    BRIDGE.importlib,
                    "import_module",
                    side_effect=stop_after_binding,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "stop after exact namespace binding"
                    ):
                        BRIDGE._load_validation_modules(runtime)
                self.assertEqual(
                    observed,
                    ["scripts.show_base.base_long_val_contract"],
                )
                self.assertNotEqual(sys.path[0], str(runtime))
            finally:
                sys.path[:] = old_path
                for name in tuple(sys.modules):
                    if any(
                        name == prefix or name.startswith(prefix + ".")
                        for prefix in prefixes
                    ):
                        sys.modules.pop(name, None)
                sys.modules.update(saved)

    def test_successful_loader_keeps_runtime_root_for_lazy_model_imports(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-loader-lazy-", dir="/private/tmp"
        ) as raw:
            runtime = Path(raw).resolve()
            show_base = runtime / "scripts" / "show_base"
            show_base.mkdir(parents=True)
            models = runtime / "models"
            models.mkdir()
            (models / "__init__.py").write_text("", encoding="utf-8")
            (models / "lazy_probe.py").write_text(
                "SOURCE = 'runtime'\n", encoding="utf-8"
            )
            module_names = {
                "contract": "scripts.show_base.base_long_val_contract",
                "engine": "scripts.show_base.run_base_val_inference",
                "producer": "scripts.show_base.produce_base_val_measurement",
                "selector": "scripts.show_base.select_base_official_adapt_long",
                "legacy": "scripts.show_base.select_base_official_adapt",
            }
            fake_modules = {}
            for key, name in module_names.items():
                path = show_base / (name.rsplit(".", 1)[-1] + ".py")
                path.write_text("", encoding="utf-8")
                module = ModuleType(name)
                module.__file__ = str(path)
                fake_modules[key] = module
            fake_modules["legacy"].DIFFSHEG_PINNED_RECEIPT = {
                "autoencoders": {
                    "fgd": copy.deepcopy(
                        BRIDGE.EXPECTED_DIFFSHEG_FGD_PROVENANCE
                    )
                }
            }
            by_name = {
                name: fake_modules[key] for key, name in module_names.items()
            }
            prefixes = ("scripts", "models", "dataloaders", "utils")
            project_names = [
                name
                for name in tuple(sys.modules)
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in prefixes
                )
            ]
            saved = {name: sys.modules.pop(name) for name in project_names}
            old_path = list(sys.path)

            def import_runtime(name: str):
                module = by_name[name]
                sys.modules[name] = module
                return module

            def pinned_file(path: Path, _label: str):
                digest = (
                    BRIDGE.RUNTIME_VALIDATION_CONTRACT_SHA256
                    if path.name == "base_long_val_contract.py"
                    else BRIDGE.RUNTIME_VALIDATION_SELECTOR_SHA256
                )
                return path, b"", digest

            try:
                with (
                    mock.patch.object(
                        BRIDGE.importlib,
                        "import_module",
                        side_effect=import_runtime,
                    ),
                    mock.patch.object(BRIDGE, "_validate_runtime_source"),
                    mock.patch.object(
                        BRIDGE, "_safe_file", side_effect=pinned_file
                    ),
                ):
                    loaded = BRIDGE._load_validation_modules(runtime)
                self.assertEqual(set(loaded), set(module_names))
                self.assertEqual(sys.path[0], str(runtime))
                lazy = importlib.import_module("models.lazy_probe")
                self.assertEqual(lazy.SOURCE, "runtime")
                self.assertEqual(
                    Path(lazy.__file__).resolve(strict=True),
                    models / "lazy_probe.py",
                )
                BRIDGE._project_modules_are_from(runtime)
            finally:
                sys.path[:] = old_path
                for name in tuple(sys.modules):
                    if any(
                        name == prefix or name.startswith(prefix + ".")
                        for prefix in prefixes
                    ):
                        sys.modules.pop(name, None)
                sys.modules.update(saved)

    def test_full_runtime_projection_is_exact_and_keeps_evidence_immutable(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-runtime-projection-", dir="/private/tmp"
        ) as raw:
            fixture = self._runtime_projection_fixture(Path(raw).resolve())
            before = copy.deepcopy(fixture["evidence_pipeline"])
            with mock.patch.object(
                BRIDGE,
                "_validate_runtime_validation_roles",
                return_value=(
                    fixture["evidence_root"], fixture["runtime_root"]
                ),
            ):
                evidence, runtime, runtime_root = (
                    BRIDGE._runtime_execution_pipeline(
                        fixture["authority"],
                        fixture["modules_for"](fixture["runtime_receipt"]),
                    )
                )
            self.assertIs(evidence, fixture["evidence_pipeline"])
            self.assertEqual(evidence, before)
            self.assertEqual(fixture["evidence_pipeline"], before)
            self.assertEqual(runtime_root, fixture["runtime_root"])
            self.assertEqual(
                tuple(runtime["source_closure"]),
                BRIDGE.RUNTIME_EXECUTION_SOURCE_FILES,
            )
            self.assertEqual(len(runtime["source_closure"]), 29)
            self.assertEqual(
                runtime["source"], fixture["authority"]["runtime_validation_source"]
            )
            for relative in BRIDGE.RUNTIME_EXECUTION_SOURCE_FILES:
                self.assertEqual(
                    Path(runtime["source_closure"][relative]["path"]),
                    fixture["runtime_root"] / relative,
                )
            self.assertEqual(
                runtime["inference_entrypoint"],
                runtime["source_closure"][
                    "scripts/show_base/run_base_val_inference.py"
                ],
            )
            self.assertEqual(
                runtime["inference_helper"],
                runtime["source_closure"][
                    "scripts/show_base/semtalk_base_inference_core.py"
                ],
            )
            self.assertEqual(
                runtime["receipt_payload_sha256"], BRIDGE._payload_sha(runtime)
            )

    def test_inspect_exposes_exact_frozen_evidence_evaluator(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-evidence-evaluator-", dir="/private/tmp"
        ) as raw:
            fixture = self._runtime_projection_fixture(Path(raw).resolve())
            preflight_artifact = {
                "path": str(Path(raw).resolve() / "work-preflight.json"),
                "sha256": "1" * 64,
                "bytes": 123,
                "receipt_payload_sha256": "2" * 64,
            }
            authority_reference = {
                "path": str(Path(raw).resolve() / "work-authority.json"),
                "sha256": "3" * 64,
                "bytes": 456,
                "receipt_payload_sha256": "4" * 64,
            }
            preflight = {
                "candidate_epochs": [1],
                "work_authority": authority_reference,
            }
            authority_artifact = {
                key: authority_reference[key]
                for key in ("path", "sha256", "bytes")
            }
            modules = fixture["modules_for"](fixture["runtime_receipt"])
            with (
                mock.patch.object(
                    BRIDGE,
                    "_preflight_artifact",
                    return_value=(preflight_artifact, preflight),
                ),
                mock.patch.object(
                    BRIDGE,
                    "_validate_work_authority",
                    return_value=(
                        authority_artifact,
                        fixture["authority"],
                        modules,
                    ),
                ),
                mock.patch.object(
                    BRIDGE,
                    "_validate_runtime_validation_roles",
                    return_value=(
                        fixture["evidence_root"], fixture["runtime_root"]
                    ),
                ),
            ):
                result = BRIDGE.inspect_preflight(
                    SimpleNamespace(
                        preflight=Path(preflight_artifact["path"]),
                        expected_preflight_sha256=preflight_artifact["sha256"],
                    )
                )

            relative = BRIDGE.DIFFSHEG_EVALUATOR_RELATIVE
            expected_entry = fixture["evidence_pipeline"]["source_closure"][
                relative
            ]
            expected_source = fixture["evidence_pipeline"]["source"]
            self.assertEqual(
                set(result),
                {
                    "status", "split", "test_visible",
                    "selection_eligible", "epoch", "preflight",
                    "frozen_evidence_evaluator",
                },
            )
            self.assertEqual(result["preflight"], preflight_artifact)
            self.assertEqual(
                result["frozen_evidence_evaluator"],
                {
                    **expected_entry,
                    "repository_root": expected_source["source_root"],
                    "repository_git_head": expected_source["commit"],
                    "repository_git_tree": expected_source["tree"],
                    "repository_origin": expected_source["origin"],
                    "repository_clean": True,
                    "repository_detached": True,
                    "repository_local_branches_at_commit": [],
                },
            )
            self.assertNotEqual(
                result["frozen_evidence_evaluator"]["path"],
                fixture["runtime_receipt"]["files"][relative]["path"],
            )

    def test_frozen_evidence_evaluator_rejects_relocated_or_tampered_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-evidence-evaluator-tamper-", dir="/private/tmp"
        ) as raw:
            fixture = self._runtime_projection_fixture(Path(raw).resolve())
            original = fixture["evidence_pipeline"]
            before = copy.deepcopy(original)
            relative = BRIDGE.DIFFSHEG_EVALUATOR_RELATIVE
            mutations = []

            missing = copy.deepcopy(original)
            missing["source_closure"].pop(relative)
            mutations.append(missing)

            relocated = copy.deepcopy(original)
            relocated["source_closure"][relative]["path"] = str(
                fixture["evidence_root"] / "elsewhere.py"
            )
            mutations.append(relocated)

            bad_sha = copy.deepcopy(original)
            bad_sha["source_closure"][relative]["sha256"] = "not-a-sha"
            mutations.append(bad_sha)

            dirty = copy.deepcopy(original)
            dirty["source"]["clean"] = False
            mutations.append(dirty)

            headed = copy.deepcopy(original)
            headed["source"]["local_branches_at_commit"] = ["refs/heads/main"]
            mutations.append(headed)

            for index, pipeline in enumerate(mutations):
                with self.subTest(index=index), self.assertRaises(
                    BRIDGE.LiveConsumerError
                ):
                    BRIDGE._frozen_evidence_evaluator_binding(pipeline)
            self.assertEqual(fixture["evidence_pipeline"], before)

    def test_runtime_projection_rejects_closure_tamper_before_engine_use(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-runtime-tamper-", dir="/private/tmp"
        ) as raw:
            fixture = self._runtime_projection_fixture(Path(raw).resolve())
            unchanged = next(
                relative
                for relative in BRIDGE.RUNTIME_EXECUTION_SOURCE_FILES
                if relative not in BRIDGE.RUNTIME_EXECUTION_CHANGED_FILES
            )
            tampered_receipts = []

            missing = copy.deepcopy(fixture["runtime_receipt"])
            missing["files"].pop(unchanged)
            tampered_receipts.append(missing)

            extra = copy.deepcopy(fixture["runtime_receipt"])
            extra["files"]["models/extra.py"] = copy.deepcopy(
                extra["files"][unchanged]
            )
            tampered_receipts.append(extra)

            escaped = copy.deepcopy(fixture["runtime_receipt"])
            escaped["files"][unchanged]["path"] = str(
                fixture["runtime_root"].parent / "escaped.py"
            )
            tampered_receipts.append(escaped)

            third_delta = copy.deepcopy(fixture["runtime_receipt"])
            third_delta["files"][unchanged]["sha256"] = "f" * 64
            third_delta["files"][unchanged]["git_blob_sha1"] = "e" * 40
            third_delta["files"][unchanged]["bytes"] += 1
            tampered_receipts.append(third_delta)

            changed_pair = copy.deepcopy(fixture["runtime_receipt"])
            changed_relative = next(iter(BRIDGE.RUNTIME_EXECUTION_CHANGED_FILES))
            changed_pair["files"][changed_relative]["sha256"] = "f" * 64
            tampered_receipts.append(changed_pair)

            with mock.patch.object(
                BRIDGE,
                "_validate_runtime_validation_roles",
                return_value=(
                    fixture["evidence_root"], fixture["runtime_root"]
                ),
            ):
                for index, receipt in enumerate(tampered_receipts):
                    with self.subTest(index=index):
                        with self.assertRaises(BRIDGE.LiveConsumerError):
                            BRIDGE._runtime_execution_pipeline(
                                fixture["authority"],
                                fixture["modules_for"](receipt),
                            )

    def test_engine_projection_fixes_same_bytes_different_root_and_restores(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-runtime-adapter-", dir="/private/tmp"
        ) as raw:
            root = Path(raw).resolve()
            relative = "models/semtalk.py"
            evidence_file = root / "evidence" / relative
            runtime_file = root / "runtime" / relative
            evidence_file.parent.mkdir(parents=True)
            runtime_file.parent.mkdir(parents=True)
            payload = b"SOURCE = 'same tracked bytes'\n"
            evidence_file.write_bytes(payload)
            runtime_file.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            evidence_pipeline = {
                "source_closure": {
                    relative: {
                        "path": str(evidence_file),
                        "sha256": digest,
                        "bytes": len(payload),
                    }
                }
            }
            runtime_pipeline = copy.deepcopy(evidence_pipeline)
            runtime_pipeline["source_closure"][relative]["path"] = str(
                runtime_file
            )
            evidence_before = copy.deepcopy(evidence_pipeline)
            runtime_before = copy.deepcopy(runtime_pipeline)
            runtime_module = ModuleType("models.semtalk")
            runtime_module.__file__ = str(runtime_file)
            observed = []

            def original_helper(pipeline):
                entry = pipeline["source_closure"][relative]
                module_path = Path(runtime_module.__file__).resolve(strict=True)
                snapshot = module_path.read_bytes()
                if hashlib.sha256(snapshot).hexdigest() != entry["sha256"]:
                    raise RuntimeError("byte mismatch")
                if module_path != Path(entry["path"]):
                    raise RuntimeError("was imported from another checkout")
                observed.append(("helper", entry["path"]))
                pipeline["consumer_mutation"] = True
                return runtime_module

            def original_joint(helper, pipeline):
                del helper
                self.assertNotIn("consumer_mutation", pipeline)
                observed.append(
                    ("joint", pipeline["source_closure"][relative]["path"])
                )
                return {}, {}

            def original_models(
                helper, *, epoch, preflight, pipeline, device
            ):
                del helper, epoch, preflight, device
                self.assertNotIn("consumer_mutation", pipeline)
                observed.append(
                    ("models", pipeline["source_closure"][relative]["path"])
                )
                return {}, {}

            engine = ModuleType("runtime_engine")
            engine._preflight_artifact = lambda *_args: ("old", "preflight")
            engine._load_pinned_helper = original_helper
            engine._pinned_joint_mask_arrays = original_joint
            engine._load_models = original_models
            originals = {
                name: getattr(engine, name)
                for name in (
                    "_preflight_artifact",
                    "_load_pinned_helper",
                    "_pinned_joint_mask_arrays",
                    "_load_models",
                )
            }

            with self.assertRaisesRegex(RuntimeError, "another checkout"):
                original_helper(copy.deepcopy(evidence_pipeline))

            with BRIDGE._engine_adapter(
                engine,
                evidence_pipeline=evidence_pipeline,
                runtime_pipeline=runtime_pipeline,
            ):
                helper = engine._load_pinned_helper(
                    copy.deepcopy(evidence_pipeline)
                )
                engine._pinned_joint_mask_arrays(
                    helper, copy.deepcopy(evidence_pipeline)
                )
                engine._load_models(
                    helper,
                    epoch=1,
                    preflight={},
                    pipeline=copy.deepcopy(evidence_pipeline),
                    device="cpu",
                )
                tampered = copy.deepcopy(evidence_pipeline)
                tampered["source_closure"][relative]["path"] = str(runtime_file)
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError, "immutable evidence"
                ):
                    engine._load_pinned_helper(tampered)

            self.assertEqual(evidence_pipeline, evidence_before)
            self.assertEqual(runtime_pipeline, runtime_before)
            self.assertEqual(
                observed,
                [
                    ("helper", str(runtime_file)),
                    ("joint", str(runtime_file)),
                    ("models", str(runtime_file)),
                ],
            )
            for name, original in originals.items():
                self.assertIs(getattr(engine, name), original)

            with self.assertRaisesRegex(RuntimeError, "engine failure"):
                with BRIDGE._engine_adapter(
                    engine,
                    evidence_pipeline=evidence_pipeline,
                    runtime_pipeline=runtime_pipeline,
                ):
                    raise RuntimeError("engine failure")
            for name, original in originals.items():
                self.assertIs(getattr(engine, name), original)

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

    def test_failed_run_inventory_is_exact_and_has_zero_semantic_outputs(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-recovery-inventory-", dir="/private/tmp"
        ) as raw:
            root = Path(raw).resolve()
            for relative in BRIDGE.FAILED_RUN_DIRECTORIES:
                if relative != ".":
                    (root / relative).mkdir(parents=True, exist_ok=True)
            marker_payload = (
                "Traceback (most recent call last):\n"
                + BRIDGE.FAILED_SHARD_FAILURE_MARKER
                + "\n"
            ).encode()
            for relative in BRIDGE.FAILED_RUN_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = (
                    marker_payload
                    if relative in BRIDGE.FAILED_SHARD_LOGS
                    else ("fixture:" + relative + "\n").encode()
                )
                path.write_bytes(payload)
            fixture_pins = {
                relative: (
                    hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                    (root / relative).stat().st_size,
                )
                for relative in BRIDGE.FAILED_RUN_FILES
            }
            with mock.patch.object(
                BRIDGE, "FAILED_RUN_FILE_PINS", fixture_pins
            ):
                inventory = BRIDGE._relative_inventory(root)
                self.assertEqual(inventory["semantic_outputs"], [])
                self.assertEqual(
                    inventory["directories"], sorted(BRIDGE.FAILED_RUN_DIRECTORIES)
                )
                self.assertEqual(
                    [row["relative_path"] for row in inventory["files"]],
                    sorted(BRIDGE.FAILED_RUN_FILES),
                )
                self.assertEqual(
                    inventory["shard_log_sha256"],
                    hashlib.sha256(marker_payload).hexdigest(),
                )

                non_shard = root / "work-preflight.json"
                original = non_shard.read_bytes()
                non_shard.write_bytes(b"tampered\n")
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError, "SHA/byte pins"
                ):
                    BRIDGE._relative_inventory(root)
                non_shard.write_bytes(original)

                semantic = root / "candidates/e1/shards/receipt.json"
                semantic.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError, "semantic or unknown output"
                ):
                    BRIDGE._relative_inventory(root)

    def test_recovery_control_rejects_unrelated_and_predates_minimum(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-recovery-control-", dir="/private/tmp"
        ) as raw:
            core = {"path": str(Path(raw) / "unused"), "sha256": "a" * 64, "bytes": 1}
            source = {
                "root": str(Path(raw).resolve()),
                "origin": BRIDGE.EXPECTED_ORIGIN,
                "commit": BRIDGE.REJECTED_RECOVERY_CONTROL_COMMIT,
                "tree": BRIDGE.REJECTED_RECOVERY_CONTROL_TREE,
                "supervisor": core,
                "launcher": core,
                "bridge": core,
                "authority_adapter": core,
            }
            with mock.patch.object(BRIDGE, "_git", return_value=(0, "")):
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError, "clean detached branchless"
                ):
                    BRIDGE._validate_recovery_control_source(source)

            source["commit"] = "1" * 40
            source["tree"] = "2" * 40
            with mock.patch.object(BRIDGE, "_git", return_value=(1, "")):
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError, "clean detached branchless"
                ):
                    BRIDGE._validate_recovery_control_source(source)

    def test_failed_incident_artifacts_are_sha_and_byte_pinned(self) -> None:
        for role, (digest, size) in BRIDGE.FAILED_INCIDENT_ARTIFACTS.items():
            with self.subTest(role=role):
                BRIDGE._require_failed_incident_artifact(
                    {"path": "/pinned", "sha256": digest, "bytes": size}, role
                )
                with self.assertRaises(BRIDGE.LiveConsumerError):
                    BRIDGE._require_failed_incident_artifact(
                        {"path": "/pinned", "sha256": "f" * 64, "bytes": size},
                        role,
                    )
                with self.assertRaises(BRIDGE.LiveConsumerError):
                    BRIDGE._require_failed_incident_artifact(
                        {"path": "/pinned", "sha256": digest, "bytes": size + 1},
                        role,
                    )

    def test_supervisor_self_hash_abi_cross_module_with_non_ascii(self) -> None:
        unsigned = {
            "format": "supervisor-abi-probe",
            "candidate_epoch": 1,
            "note": "恢复验证",
            "nested": {"accent": "é", "exact": True},
        }
        with tempfile.TemporaryDirectory(
            prefix="live-supervisor-abi-", dir="/private/tmp"
        ) as raw:
            root = Path(raw).resolve()
            for payload_key in (
                "campaign_payload_sha256",
                "claim_payload_sha256",
                "receipt_payload_sha256",
            ):
                with self.subTest(payload_key=payload_key):
                    value = SUPERVISOR._add_self_hash(unsigned, payload_key)
                    self.assertEqual(
                        value[payload_key],
                        BRIDGE._supervisor_payload_sha(value, payload_key),
                    )
                    self.assertNotEqual(
                        value[payload_key],
                        BRIDGE._payload_sha(value, payload_key),
                    )
                    payload = SUPERVISOR.canonical_json_bytes(value)
                    path = root / (payload_key + ".json")
                    path.write_bytes(payload)
                    artifact = {
                        "path": str(path),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                    observed_artifact, observed_value = (
                        BRIDGE._self_hashed_document(
                            artifact,
                            keys=frozenset(value),
                            payload_key=payload_key,
                            label="supervisor ABI probe",
                            supervisor_payload_abi=True,
                        )
                    )
                    self.assertEqual(observed_artifact, artifact)
                    self.assertEqual(observed_value, value)
                    with self.assertRaisesRegex(
                        BRIDGE.LiveConsumerError, "payload SHA mismatch"
                    ):
                        BRIDGE._self_hashed_document(
                            artifact,
                            keys=frozenset(value),
                            payload_key=payload_key,
                            label="bridge-native ABI negative probe",
                        )

        native = BRIDGE._with_payload_sha(unsigned)
        self.assertEqual(
            native["receipt_payload_sha256"], BRIDGE._payload_sha(native)
        )
        self.assertNotEqual(
            native["receipt_payload_sha256"],
            BRIDGE._supervisor_payload_sha(
                native, "receipt_payload_sha256"
            ),
        )

    def test_recovery_supervisor_abi_is_bound_at_every_production_callsite(
        self,
    ) -> None:
        tree = ast.parse(Path(BRIDGE.__file__).read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        expected_counts = {
            "_campaign_document": 1,
            "_validate_recovery_request": 3,
        }
        observed = {}
        for function_name, expected_count in expected_counts.items():
            function = functions[function_name]
            calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_self_hashed_document"
            ]
            observed[function_name] = len(calls)
            self.assertEqual(len(calls), expected_count)
            for call in calls:
                keywords = {item.arg: item.value for item in call.keywords}
                self.assertIn("supervisor_payload_abi", keywords)
                value = keywords["supervisor_payload_abi"]
                self.assertIsInstance(value, ast.Constant)
                self.assertIs(value.value, True)
        self.assertEqual(observed, expected_counts)

        every_production_call = [
            node
            for function in functions.values()
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_self_hashed_document"
        ]
        self.assertEqual(len(every_production_call), 4)

    def test_direct_recovery_request_rejects_rehashed_fake_incident(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-recovery-forgery-", dir="/private/tmp"
        ) as raw:
            root = Path(raw).resolve()
            artifact = {"path": str(root / "x.json"), "sha256": "1" * 64, "bytes": 1}
            request = {
                "format": BRIDGE.RECOVERY_REQUEST_FORMAT,
                "status": "ready_for_single_recovery",
                "split": "val",
                "test_visible": False,
                "selection_eligible": False,
                "candidate_epoch": 1,
                "failed_campaign": artifact,
                "failed_job_claim": artifact,
                "failed_active_claim": artifact,
                "failed_authorization": artifact,
                "failed_work_authority": artifact,
                "failed_consumer_claim": artifact,
                "failed_runner_status": artifact,
                "failed_runner_log": artifact,
                "failed_run_root": str(root / "failed"),
                "failed_run_inventory": {},
                "failed_process_proof": {},
                "guard_proof": {},
                "new_campaign": artifact,
                "new_control_source": {},
                "new_work_authority": artifact,
                "new_run_root": str(root / "new"),
                "created_unix": 1.0,
            }
            request = BRIDGE._with_payload_sha(request)
            request_path = root / "request.json"
            payload = BRIDGE._canonical_json(request, newline=True)
            request_path.write_bytes(payload)
            forged_campaign = {
                "path": str(root / "forged-campaign.json"),
                "sha256": "f" * 64,
                "bytes": BRIDGE.FAILED_INCIDENT_ARTIFACTS["campaign"][1],
            }
            with mock.patch.object(
                BRIDGE,
                "_campaign_document",
                return_value=(forged_campaign, {}),
            ):
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError, "pinned artifact"
                ):
                    BRIDGE._validate_recovery_request(
                        request_path,
                        hashlib.sha256(payload).hexdigest(),
                        new_run_must_exist=False,
                    )

    def test_recovery_work_authority_replay_allows_only_publication_metadata(self) -> None:
        failed_value = BRIDGE._with_payload_sha({
            "format": BRIDGE.WORK_FORMAT,
            "candidate_epoch": 1,
            "execution_contract": {"expected_shards": 8},
            "published_unix": 1.0,
        })
        new_value = BRIDGE._with_payload_sha({
            **{
                key: copy.deepcopy(value)
                for key, value in failed_value.items()
                if key not in {"published_unix", "receipt_payload_sha256"}
            },
            "published_unix": 2.0,
        })
        failed_artifact = {
            "path": "/failed/work.json",
            "sha256": "a" * 64,
            "bytes": 101,
            "receipt_payload_sha256": failed_value["receipt_payload_sha256"],
        }
        new_artifact = {
            "path": "/new/work.json",
            "sha256": "b" * 64,
            "bytes": 99,
            "receipt_payload_sha256": new_value["receipt_payload_sha256"],
        }
        request = {
            "failed_work_authority": BRIDGE._artifact_core(failed_artifact),
            "new_work_authority": BRIDGE._artifact_core(new_artifact),
        }

        BRIDGE._validate_recovery_work_authority_replay(
            request,
            failed_artifact,
            failed_value,
            new_artifact,
            new_value,
        )

        tampered = copy.deepcopy(new_value)
        tampered["candidate_epoch"] = 2
        tampered["receipt_payload_sha256"] = BRIDGE._payload_sha(tampered)
        with self.assertRaisesRegex(
            BRIDGE.LiveConsumerError, "does not semantically replay"
        ):
            BRIDGE._validate_recovery_work_authority_replay(
                request,
                failed_artifact,
                failed_value,
                new_artifact,
                tampered,
            )

        mismatched_request = copy.deepcopy(request)
        mismatched_request["new_work_authority"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(
            BRIDGE.LiveConsumerError, "does not semantically replay"
        ):
            BRIDGE._validate_recovery_work_authority_replay(
                mismatched_request,
                failed_artifact,
                failed_value,
                new_artifact,
                new_value,
            )

    def test_failed_process_proof_is_exact_pid_only_and_read_only(self) -> None:
        status = {
            "wrapper_pid": 900_000_001,
            "child_pid": 900_000_002,
            "command": ["/bin/bash", "/formal/val/launcher.sh"],
        }
        proof = {
            "wrapper_pid": status["wrapper_pid"],
            "child_pid": status["child_pid"],
            "wrapper_proc_state": "absent",
            "child_proc_state": "absent",
            "runner_command": list(status["command"]),
            "checked_unix": 1.0,
        }
        self.assertEqual(
            BRIDGE._validate_failed_process_proof(proof, status), proof
        )
        tampered = dict(proof)
        tampered["runner_command"] = ["/bin/bash", "/other/launcher.sh"]
        with self.assertRaises(BRIDGE.LiveConsumerError):
            BRIDGE._validate_failed_process_proof(tampered, status)
        with mock.patch("os.path.lexists", return_value=True):
            with self.assertRaisesRegex(
                BRIDGE.LiveConsumerError, "no longer absent"
            ):
                BRIDGE._validate_failed_process_proof(proof, status)

    def test_inspect_is_read_only_and_reserve_publishes_claim_first(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-recovery-reserve-", dir="/private/tmp"
        ) as raw:
            root = Path(raw).resolve()
            state = root / "state"
            state.mkdir()
            claim_parent = root / "training/live_val_consumer_recovery_claims"
            claim_parent.mkdir(parents=True)
            claim_path = claim_parent / "epoch-0001.json"
            output = state / "recovery-authority.epoch-0001.json"
            request_artifact = {
                "path": str(root / "request.json"),
                "sha256": "1" * 64,
                "bytes": 100,
            }
            request = {"receipt_payload_sha256": "2" * 64}
            bindings = {
                "failed_consumer_claim": {
                    "path": str(root / "old-claim.json"),
                    "sha256": "3" * 64,
                    "bytes": 10,
                },
                "failed_runner_status": {
                    "path": str(root / "old-status.json"),
                    "sha256": "4" * 64,
                    "bytes": 11,
                },
                "new_campaign": {
                    "path": str(state / "campaign.json"),
                    "sha256": "5" * 64,
                    "bytes": 12,
                },
                "new_control_source": {"commit": "6" * 40},
                "new_work_authority": {
                    "path": str(state / "authorities/epoch-0001.json"),
                    "sha256": "7" * 64,
                    "bytes": 13,
                },
                "new_work_authority_value": {"candidate_epoch": 1},
                "new_run_root": str(root / "new-run"),
            }
            validated = (
                request_artifact,
                request,
                bindings,
                claim_path,
                output,
            )
            args = argparse.Namespace(
                request=Path(request_artifact["path"]),
                expected_request_sha256=request_artifact["sha256"],
                output_authority=output,
            )
            with (
                mock.patch.object(
                    BRIDGE, "_validate_recovery_request", return_value=validated
                ),
                mock.patch.object(
                    BRIDGE, "_recovery_claim_path", return_value=claim_path
                ),
            ):
                inspected = BRIDGE.inspect_recovery(args)
                self.assertEqual(
                    set(inspected),
                    {"status", "recovery_authority", "recovery_claim_path"},
                )
                self.assertFalse(claim_path.exists())
                self.assertFalse(output.exists())

                order = []
                original = BRIDGE._write_strict_new

                def write(path, value, label):
                    order.append(label)
                    return original(path, value, label)

                with mock.patch.object(
                    BRIDGE, "_write_strict_new", side_effect=write
                ):
                    reserved = BRIDGE.reserve_recovery(args)
                self.assertEqual(
                    order,
                    ["consumer recovery claim", "consumer recovery authority"],
                )
                self.assertEqual(
                    set(reserved),
                    {"status", "recovery_authority", "recovery_claim"},
                )
                claim = json.loads(claim_path.read_bytes())
                authority = json.loads(output.read_bytes())
                self.assertEqual(
                    claim["recovery_authority"], reserved["recovery_authority"]
                )
                self.assertEqual(authority["recovery_claim_path"], str(claim_path))
                with self.assertRaises(BRIDGE.DuplicateConsumptionError):
                    BRIDGE.reserve_recovery(args)

            failed_claim_path = (
                root / "training-failed/live_val_consumer_recovery_claims"
                / "epoch-0001.json"
            )
            failed_claim_path.parent.mkdir(parents=True)
            failed_output = state / "recovery-authority-failed.epoch-0001.json"
            failed_validated = (
                request_artifact,
                request,
                bindings,
                failed_claim_path,
                failed_output,
            )
            failed_args = argparse.Namespace(
                request=Path(request_artifact["path"]),
                expected_request_sha256=request_artifact["sha256"],
                output_authority=failed_output,
            )
            with (
                mock.patch.object(
                    BRIDGE, "_validate_recovery_request",
                    return_value=failed_validated,
                ),
                mock.patch.object(
                    BRIDGE, "_recovery_claim_path",
                    return_value=failed_claim_path,
                ),
            ):
                order = []
                original = BRIDGE._write_strict_new

                def fail_authority(path, value, label):
                    order.append(label)
                    if label == "consumer recovery authority":
                        raise BRIDGE.LiveConsumerError(
                            "injected recovery authority failure"
                        )
                    return original(path, value, label)

                with mock.patch.object(
                    BRIDGE, "_write_strict_new", side_effect=fail_authority
                ):
                    with self.assertRaisesRegex(
                        BRIDGE.LiveConsumerError, "authority failure"
                    ):
                        BRIDGE.reserve_recovery(failed_args)
                self.assertEqual(
                    order,
                    ["consumer recovery claim", "consumer recovery authority"],
                )
                self.assertTrue(failed_claim_path.exists())
                self.assertFalse(failed_output.exists())
                with self.assertRaises(BRIDGE.DuplicateConsumptionError):
                    BRIDGE.reserve_recovery(failed_args)

    def test_recovery_cli_and_prepare_arguments_are_exact(self) -> None:
        request = "/formal/val/recovery-request.json"
        output = "/formal/val/state/recovery-authority.epoch-0001.json"
        parsed = BRIDGE.build_parser().parse_args(
            [
                "inspect-recovery", "--request", request,
                "--expected-request-sha256", "a" * 64,
                "--output-authority", output,
            ]
        )
        self.assertEqual(parsed.command, "inspect-recovery")
        parsed = BRIDGE.build_parser().parse_args(
            [
                "reserve-recovery", "--request", request,
                "--expected-request-sha256", "a" * 64,
                "--output-authority", output,
            ]
        )
        self.assertEqual(parsed.command, "reserve-recovery")

        with tempfile.TemporaryDirectory(
            prefix="live-recovery-prepare-", dir="/private/tmp"
        ) as raw:
            run_root = Path(raw).resolve()
            args = argparse.Namespace(
                run_root=run_root,
                output=run_root / "work-preflight.json",
                work_authority=run_root / "work.json",
                expected_work_authority_sha256="b" * 64,
                recovery_authority=run_root / "recovery.json",
                expected_recovery_authority_sha256=None,
                recovery_claim=None,
                expected_recovery_claim_sha256=None,
            )
            with mock.patch.object(
                BRIDGE,
                "_validate_work_authority",
                return_value=(
                    {"path": str(args.work_authority), "sha256": "b" * 64},
                    {"candidate_epoch": 1},
                    None,
                ),
            ):
                with self.assertRaisesRegex(
                    BRIDGE.LiveConsumerError, "all-or-none"
                ):
                    BRIDGE.prepare(args)

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
        expected_commit: str,
        expected_tree: str,
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
                return 0, ("f" * 40 if wrong_commit else expected_commit)
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return 0, expected_tree
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

    def test_validation_sources_are_exact_clean_detached_and_branchless(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="live-source-", dir="/private/tmp"
        ) as raw:
            for validator, commit, tree in (
                (
                    BRIDGE._validate_evidence_source,
                    BRIDGE.VALIDATION_EVIDENCE_SOURCE_COMMIT,
                    BRIDGE.VALIDATION_EVIDENCE_SOURCE_TREE,
                ),
                (
                    BRIDGE._validate_runtime_source,
                    BRIDGE.RUNTIME_VALIDATION_SOURCE_COMMIT,
                    BRIDGE.RUNTIME_VALIDATION_SOURCE_TREE,
                ),
            ):
                source = {
                    "origin": BRIDGE.EXPECTED_ORIGIN,
                    "source_root": raw,
                    "commit": commit,
                    "tree": tree,
                    "clean": True,
                    "detached": True,
                    "local_branches_at_commit": [],
                }
                valid_git = self._source_git(
                    expected_commit=commit, expected_tree=tree
                )
                with mock.patch.object(BRIDGE, "_git", side_effect=valid_git):
                    self.assertEqual(validator(source), Path(raw))
                for variant in (
                    {"dirty": True},
                    {"attached": True},
                    {"heads": True},
                    {"wrong_origin": True},
                    {"wrong_commit": True},
                ):
                    with self.subTest(validator=validator.__name__, variant=variant):
                        side_effect = self._source_git(
                            expected_commit=commit,
                            expected_tree=tree,
                            **variant,
                        )
                        with mock.patch.object(
                            BRIDGE, "_git", side_effect=side_effect
                        ):
                            with self.assertRaises(BRIDGE.LiveConsumerError):
                                validator(source)
                malformed = dict(source)
                malformed["clean"] = 1
                valid_git = self._source_git(
                    expected_commit=commit, expected_tree=tree
                )
                with mock.patch.object(BRIDGE, "_git", side_effect=valid_git):
                    with self.assertRaises(BRIDGE.LiveConsumerError):
                        validator(malformed)

    def test_runtime_validation_roles_are_distinct_and_proof_exact(self) -> None:
        authority = {
            "validation_evidence_source": {"role": "evidence"},
            "runtime_validation_source": {"role": "runtime"},
            "runtime_validation_proof": (
                BRIDGE._expected_runtime_validation_proof()
            ),
        }
        with (
            mock.patch.object(
                BRIDGE, "_validate_evidence_source", return_value=Path("/evidence")
            ),
            mock.patch.object(
                BRIDGE, "_validate_runtime_source", return_value=Path("/runtime")
            ),
            mock.patch.object(BRIDGE, "_git", return_value=(0, "")),
        ):
            self.assertEqual(
                BRIDGE._validate_runtime_validation_roles(authority),
                (Path("/evidence"), Path("/runtime")),
            )

        with (
            mock.patch.object(
                BRIDGE, "_validate_evidence_source", return_value=Path("/same")
            ),
            mock.patch.object(
                BRIDGE, "_validate_runtime_source", return_value=Path("/same")
            ),
        ):
            with self.assertRaisesRegex(
                BRIDGE.LiveConsumerError, "must be distinct"
            ):
                BRIDGE._validate_runtime_validation_roles(authority)

        tampered = copy.deepcopy(authority)
        tampered["runtime_validation_proof"]["runtime_source"]["commit"] = (
            BRIDGE.VALIDATION_EVIDENCE_SOURCE_COMMIT
        )
        with (
            mock.patch.object(
                BRIDGE, "_validate_evidence_source", return_value=Path("/evidence")
            ),
            mock.patch.object(
                BRIDGE, "_validate_runtime_source", return_value=Path("/runtime")
            ),
        ):
            with self.assertRaisesRegex(
                BRIDGE.LiveConsumerError, "successor proof changed"
            ):
                BRIDGE._validate_runtime_validation_roles(tampered)

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
        evidence_source = {
            "origin": BRIDGE.EXPECTED_ORIGIN,
            "source_root": str(root / "evidence"),
            "commit": BRIDGE.VALIDATION_EVIDENCE_SOURCE_COMMIT,
            "tree": BRIDGE.VALIDATION_EVIDENCE_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        runtime_source = {
            "origin": BRIDGE.EXPECTED_ORIGIN,
            "source_root": str(root / "runtime"),
            "commit": BRIDGE.RUNTIME_VALIDATION_SOURCE_COMMIT,
            "tree": BRIDGE.RUNTIME_VALIDATION_SOURCE_TREE,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        reconciliation = {
            "validation_evidence_source": evidence_source,
            "runtime_validation_source": runtime_source,
            "runtime_validation_proof": (
                BRIDGE._expected_runtime_validation_proof()
            ),
            "pipeline_source": evidence_source,
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
                    BRIDGE,
                    "_validate_runtime_validation_roles",
                    return_value=(root, root),
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
