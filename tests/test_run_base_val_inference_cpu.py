from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np


from scripts.show_base import run_base_val_inference as PRODUCER


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class ValInferenceProducerCpuTest(unittest.TestCase):
    @staticmethod
    def _pinned_source_bytes(path: str) -> bytes:
        return (Path(__file__).resolve().parents[1] / path).read_bytes()

    def _write_pinned_joint_fixture(
        self,
        root: Path,
        *,
        include_context: bool = True,
        corrupt_context: bool = False,
        include_authority: bool = True,
        corrupt_authority: bool = False,
    ) -> tuple[Path, Path, dict[str, object]]:
        helper_relative = "scripts/show_base/semtalk_base_inference_core.py"
        helper_path = root / helper_relative
        context_path = (
            root / PRODUCER.PINNED_JOINT_CONTEXT_SOURCE["relative_path"]
        )
        helper_payload = self._pinned_source_bytes(helper_relative)
        helper_path.parent.mkdir(parents=True)
        helper_path.write_bytes(helper_payload)
        if include_context:
            context_payload = self._pinned_source_bytes(
                PRODUCER.PINNED_JOINT_CONTEXT_SOURCE["relative_path"]
            )
            if corrupt_context:
                context_payload += b"\n# corrupt\n"
            context_path.parent.mkdir(parents=True)
            context_path.write_bytes(context_payload)
        if include_authority:
            authority_path = (
                root
                / PRODUCER.PINNED_JOINT_AUTHORITY_SOURCE["relative_path"]
            )
            authority_payload = self._pinned_source_bytes(
                PRODUCER.PINNED_JOINT_AUTHORITY_SOURCE["relative_path"]
            )
            if corrupt_authority:
                authority_payload += b"\n# corrupt\n"
            authority_path.parent.mkdir(parents=True)
            authority_path.write_bytes(authority_payload)
        source = {
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "source_root": str(root.resolve()),
            "commit": "1" * 40,
            "tree": "2" * 40,
            "clean": True,
            "detached": True,
            "local_branches_at_commit": [],
        }
        helper_entry = {
            "path": str(helper_path.resolve()),
            "sha256": hashlib.sha256(helper_payload).hexdigest(),
            "bytes": len(helper_payload),
            "git_mode": "100644",
            "git_blob_sha1": PRODUCER._git_blob_sha1(helper_payload),
        }
        source_closure = {
            helper_relative: helper_entry,
        }
        if include_context:
            source_closure[
                PRODUCER.PINNED_JOINT_CONTEXT_SOURCE["relative_path"]
            ] = {
                "path": str(context_path.resolve()),
                "sha256": hashlib.sha256(context_payload).hexdigest(),
                "bytes": len(context_payload),
                "git_mode": "100644",
                "git_blob_sha1": PRODUCER._git_blob_sha1(context_payload),
            }
        if include_authority:
            source_closure[
                PRODUCER.PINNED_JOINT_AUTHORITY_SOURCE["relative_path"]
            ] = {
                "path": str(authority_path.resolve()),
                "sha256": hashlib.sha256(authority_payload).hexdigest(),
                "bytes": len(authority_payload),
                "git_mode": "100644",
                "git_blob_sha1": PRODUCER._git_blob_sha1(
                    authority_payload
                ),
            }
        pipeline = {
            "source": source,
            "source_closure": source_closure,
            "inference_helper": helper_entry,
        }
        return helper_path, context_path, pipeline

    @staticmethod
    def _literal_joints_list(payload: bytes) -> dict[str, object]:
        module = ast.parse(payload, filename="dataloaders/data_tools.py")
        assignments = [
            statement.value
            for statement in module.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "joints_list"
        ]
        if len(assignments) != 1:
            raise AssertionError("pinned oracle must define joints_list once")
        joints = ast.literal_eval(assignments[0])
        if not isinstance(joints, dict):
            raise AssertionError("pinned oracle joints_list must be a dict")
        return joints

    def test_cli_requires_explicit_val_and_rejects_e30(self) -> None:
        common = [
            "shard",
            "--preflight",
            "/tmp/val/preflight.json",
            "--expected-preflight-sha256",
            "a" * 64,
            "--epoch",
            "1",
            "--output-root",
            "/tmp/val/output",
            "--num-shards",
            "16",
            "--shard-id",
            "0",
            "--device",
            "cuda:0",
        ]
        with self.assertRaises(SystemExit):
            PRODUCER.parse_args(common)
        parsed = PRODUCER.parse_args(
            [common[0], "--split", "val", *common[1:]]
        )
        self.assertEqual(parsed.num_shards, 16)
        for forbidden in ("30", "3"):
            argv = [common[0], "--split", "val", *common[1:]]
            argv[argv.index("1")] = forbidden
            with self.assertRaises(SystemExit):
                PRODUCER.parse_args(argv)
        with self.assertRaises(SystemExit):
            PRODUCER.parse_args(
                [common[0], "--split", "test", *common[1:]]
            )

    def test_forbidden_labels_cover_epoch30_and_speaker2(self) -> None:
        for value in (
            "/safe/epoch_30/model.bin",
            "/safe/e30/model.bin",
            "/safe/Speaker2/model.bin",
        ):
            with self.assertRaises(PRODUCER.ValInferenceContractError):
                PRODUCER._reject_forbidden(value, "fixture")

    def test_path_filter_allows_latest_but_rejects_test_labels(self) -> None:
        real_validation_path = Path(
            "/local-ssd/xiangyuezhang/"
            "semtalk_show_base_canonical_cache_aa8e519_20260729_v1/"
            "clips/val/conan/"
            "Conan_On_Trump_s_Latest_Portrait_Faux_Pas_-_CONAN_on_TBS-"
            "WIVVOCz9r50.webm/115448-00_03_58-00_04_08.npz"
        )
        PRODUCER._reject_path(real_validation_path, "fixture")
        PRODUCER._reject_path(
            Path("/frozen/val/contest/Latest/results.npz"),
            "fixture",
        )
        for value in (
            "/frozen/val/testimonial/results.npz",
            "/frozen/val/testament/results.npz",
        ):
            with self.subTest(value=value):
                PRODUCER._reject_path(Path(value), "fixture")
        for value in (
            "/frozen/test/clip.npz",
            "/frozen/tests/clip.npz",
            "/frozen/test_predictions/clip.npz",
            "/frozen/testset/clip.npz",
            "/frozen/testsets/clip.npz",
            "/frozen/actual-test/clip.npz",
            "/frozen/beat2_semtalk_test.pkl",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    PRODUCER.selector.SelectionContractError,
                    "test-labeled",
                ):
                    PRODUCER._reject_path(Path(value), "fixture")

    def test_shard_model_load_does_not_revalidate_all_candidates(self) -> None:
        source = inspect.getsource(PRODUCER._load_models)
        self.assertNotIn("_validate_official_adapt_base", source)
        self.assertEqual(
            source.count("helper._read_verified_checkpoint_snapshot("),
            1,
        )
        self.assertNotIn("_prime_pinned_released_schema_cache", source)
        self.assertIn("_validate_base_model_state_schema", source)
        shard_source = inspect.getsource(PRODUCER.run_shard)
        self.assertNotIn("helper._joint_masks(", shard_source)
        self.assertLess(
            shard_source.index("_pinned_joint_mask_arrays("),
            shard_source.index("_load_models("),
        )
        self.assertIn(
            'runtime_contract["joint_masks"] = joint_mask_receipt',
            shard_source,
        )

    def test_model_loader_isolates_v3_short_and_formal_checkpoint_audits(
        self,
    ) -> None:
        class AuditAccepted(Exception):
            pass

        class FakeTensor:
            def __init__(self, values: list[float]) -> None:
                self._array = np.asarray(values, dtype=np.float32)
                self.shape = self._array.shape
                self.dtype = self._array.dtype

            def detach(self) -> "FakeTensor":
                return self

            def to(self, *args: object, **kwargs: object) -> "FakeTensor":
                del args, kwargs
                return self

            def contiguous(self) -> "FakeTensor":
                self._array = np.ascontiguousarray(self._array)
                return self

            def numpy(self) -> np.ndarray:
                return self._array

        epoch = 4
        updates_per_epoch = 124
        candidate_snapshot = b"short-quality-candidate"
        candidate_sha = hashlib.sha256(candidate_snapshot).hexdigest()
        frozen_receipt_sha = "f" * 64
        official_base_sha = "a" * 64
        state = {"weight": FakeTensor([1.0, 2.0])}
        semantic_sha = PRODUCER._model_state_semantic_sha256(state)

        def common_audit(checkpoint_format: str) -> dict[str, object]:
            return {
                "format": checkpoint_format,
                "completed_epochs": epoch,
                "optimizer_updates": epoch * updates_per_epoch,
                "frozen_receipt_sha256": frozen_receipt_sha,
                "official_base_checkpoint_sha256": official_base_sha,
                "speaker_scope": "SHOW_All",
                "speaker_rows": [0, 1, 2, 3],
                "vq_models_in_training_graph": False,
                "all_model_state_tensors_finite": True,
                "model_state_semantic_sha256": semantic_sha,
                "trajectory_anchor_match": None,
                "trajectory_probe_verified": True,
            }

        def short_contract(
            *,
            epochs: list[int],
            artifact_root: str,
            quality_role: str,
            reference_only: bool,
        ) -> dict[str, object]:
            return {
                "format": PRODUCER.SHORT_QUALITY_CHECKPOINT_FORMAT,
                "run_purpose": "topology_short_quality",
                "target_epochs": epochs,
                "quality_protocol_version": (
                    PRODUCER.SHORT_QUALITY_PROTOCOL_VERSION
                ),
                "artifact_root_namespace": (
                    PRODUCER.SHORT_QUALITY_ARTIFACT_ROOT_NAMESPACE
                ),
                "artifact_root": artifact_root,
                "quality_role": quality_role,
                "reference_only": reference_only,
                "late_w1_status": "not_measured",
                "w1_tail_equivalence_claimed": False,
            }

        def preflight(
            *,
            preflight_format: str,
            epochs: list[int],
            artifact_root: str,
            quality_role: str,
            reference_only: bool,
            contract: dict[str, object] | None,
        ) -> dict[str, object]:
            bundle: dict[str, object] = {
                "updates_per_epoch": updates_per_epoch,
                "manifest": {
                    "path": "/frozen/base/manifest.json",
                    "sha256": "1" * 64,
                },
                "status": {
                    "path": "/frozen/base/status.json",
                    "sha256": "2" * 64,
                },
                "frozen_inputs": {
                    "path": "/frozen/base/frozen-inputs.json",
                    "sha256": "3" * 64,
                    "receipt_sha256": frozen_receipt_sha,
                },
                "candidates": {
                    str(epoch): {
                        "path": "/frozen/base/epoch-4.pth",
                        "sha256": candidate_sha,
                        "bytes": len(candidate_snapshot),
                    }
                },
            }
            if contract is not None:
                bundle["checkpoint_audit_contract"] = contract
            return {
                "format": preflight_format,
                "candidate_epochs": epochs,
                "artifact_root": artifact_root,
                "quality_role": quality_role,
                "reference_only": reference_only,
                "candidate_bundle": bundle,
            }

        def invoke(
            payload: dict[str, object],
            audit: dict[str, object],
        ) -> None:
            helper = SimpleNamespace(
                OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT=(
                    "semtalk_show_base_official_adapt_checkpoint_v1"
                ),
                RELEASED_ALL_SPEAKERS_MODELS={
                    "base": {"sha256": official_base_sha}
                },
                _read_verified_checkpoint_snapshot=(
                    lambda *args, **kwargs: (
                        Path("/frozen/base/epoch-4.pth"),
                        candidate_snapshot,
                        candidate_sha,
                    )
                ),
                _torch_load_checkpoint=(
                    lambda *args, **kwargs: {
                        "model_state": state,
                        "audit": audit,
                    }
                ),
                _finite_state_dict=lambda *args, **kwargs: None,
                _validate_base_model_state_schema=(
                    lambda *args, **kwargs: None
                ),
                _model_args=lambda: {},
            )

            def accept_after_audit(_args: object) -> object:
                raise AuditAccepted

            fake_torch = ModuleType("torch")
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(
                    PRODUCER,
                    "_pinned_project_module",
                    side_effect=[
                        SimpleNamespace(),
                        SimpleNamespace(semtalk_base=accept_after_audit),
                    ],
                ),
                mock.patch.object(PRODUCER, "_reject_path"),
                mock.patch.object(
                    PRODUCER,
                    "_verified_json",
                    return_value={},
                ),
            ):
                PRODUCER._load_models(
                    helper,
                    epoch=epoch,
                    preflight=payload,
                    pipeline={},
                    device="cuda:0",
                )

        candidate_epochs = [1, 2, 4, 8, 16, 32]
        candidate_root = "/frozen/quality/candidate"
        candidate_contract = short_contract(
            epochs=candidate_epochs,
            artifact_root=candidate_root,
            quality_role="candidate_quality",
            reference_only=False,
        )
        candidate_preflight = preflight(
            preflight_format=PRODUCER.SHORT_QUALITY_PREFLIGHT_FORMAT,
            epochs=candidate_epochs,
            artifact_root=candidate_root,
            quality_role="candidate_quality",
            reference_only=False,
            contract=dict(candidate_contract),
        )
        candidate_audit = {
            **common_audit(PRODUCER.SHORT_QUALITY_CHECKPOINT_FORMAT),
            **candidate_contract,
        }
        with self.assertRaises(AuditAccepted):
            invoke(candidate_preflight, candidate_audit)

        reference_epochs = [1, 2, 4, 8]
        reference_root = "/frozen/quality/reference-w1"
        reference_contract = short_contract(
            epochs=reference_epochs,
            artifact_root=reference_root,
            quality_role="w1_reference",
            reference_only=True,
        )
        reference_preflight = preflight(
            preflight_format=PRODUCER.SHORT_QUALITY_PREFLIGHT_FORMAT,
            epochs=reference_epochs,
            artifact_root=reference_root,
            quality_role="w1_reference",
            reference_only=True,
            contract=dict(reference_contract),
        )
        reference_audit = {
            **common_audit(PRODUCER.SHORT_QUALITY_CHECKPOINT_FORMAT),
            **reference_contract,
        }
        with self.assertRaises(AuditAccepted):
            invoke(reference_preflight, reference_audit)

        formal_audit = common_audit(
            "semtalk_show_base_official_adapt_checkpoint_v1"
        )
        formal_preflight = preflight(
            preflight_format=PRODUCER.PREFLIGHT_FORMAT,
            epochs=[epoch],
            artifact_root="/frozen/formal",
            quality_role="formal_training",
            reference_only=False,
            contract=None,
        )
        with self.assertRaises(AuditAccepted):
            invoke(formal_preflight, formal_audit)

        invalid_cases: list[
            tuple[str, dict[str, object], dict[str, object]]
        ] = [
            (
                "long audit under short preflight",
                candidate_preflight,
                formal_audit,
            ),
            (
                "short audit under formal preflight",
                formal_preflight,
                candidate_audit,
            ),
        ]
        for field, invalid_value in (
            ("target_epochs", reference_epochs),
            ("artifact_root", "/frozen/quality/other"),
            ("quality_role", "w1_reference"),
            ("model_state_semantic_sha256", "0" * 64),
        ):
            invalid_audit = dict(candidate_audit)
            invalid_audit[field] = invalid_value
            invalid_cases.append(
                (f"short audit {field} drift", candidate_preflight, invalid_audit)
            )
        extra_key_audit = dict(candidate_audit)
        extra_key_audit["unbound_note"] = "forbidden"
        invalid_cases.append(
            ("short audit extra key", candidate_preflight, extra_key_audit)
        )
        mismatched_bundle_preflight = {
            **candidate_preflight,
            "candidate_bundle": {
                **candidate_preflight["candidate_bundle"],
                "checkpoint_audit_contract": {
                    **candidate_contract,
                    "target_epochs": reference_epochs,
                },
            },
        }
        invalid_cases.append(
            (
                "preflight checkpoint contract drift",
                mismatched_bundle_preflight,
                candidate_audit,
            )
        )
        for label, invalid_preflight, invalid_audit in invalid_cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "candidate audit",
            ):
                invoke(invalid_preflight, invalid_audit)

    def test_shard_expected_call_arithmetic_has_runtime_math_dependency(
        self,
    ) -> None:
        runtime_math = PRODUCER.run_shard.__globals__.get("math")
        self.assertIs(runtime_math, __import__("math"))
        for frames, expected in (
            (1, 1),
            (4, 1),
            (64, 1),
            (65, 2),
            (124, 2),
            (125, 3),
        ):
            self.assertEqual(
                max(
                    1,
                    runtime_math.ceil(
                        (frames - 4) / 60,
                    ),
                ),
                expected,
            )

    def test_fasttext_free_joint_masks_equal_pinned_helper(self) -> None:
        class FakeTensor:
            def __init__(self, array: np.ndarray) -> None:
                self.array = np.asarray(array)

            def to(
                self,
                *args: object,
                **kwargs: object,
            ) -> "FakeTensor":
                del kwargs
                if args and args[0] == "int64":
                    return FakeTensor(self.array.astype(np.int64))
                return self

            def __iadd__(self, other: "FakeTensor") -> "FakeTensor":
                self.array += other.array
                return self

            def __gt__(self, value: object) -> "FakeTensor":
                return FakeTensor(self.array > value)

            def any(self) -> "FakeTensor":
                return FakeTensor(np.asarray(self.array.any()))

            def item(self) -> object:
                return self.array.item()

        fake_torch = ModuleType("torch")
        fake_torch.int64 = "int64"  # type: ignore[attr-defined]
        fake_torch.zeros = (  # type: ignore[attr-defined]
            lambda size, **_kwargs: FakeTensor(
                np.zeros(size, dtype=np.int64)
            )
        )
        fake_torch.from_numpy = (  # type: ignore[attr-defined]
            lambda array: FakeTensor(array.copy())
        )
        original_import = builtins.__import__

        def no_fasttext_import(
            name: str,
            *args: object,
            **kwargs: object,
        ) -> object:
            if name == "fasttext":
                raise AssertionError("joint mask construction imported fasttext")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory(
            prefix="semtalk_joint_equivalence_"
        ) as raw:
            root = Path(raw)
            helper_path, context_path, pipeline = (
                self._write_pinned_joint_fixture(root)
            )
            helper_sha_before = hashlib.sha256(
                helper_path.read_bytes()
            ).hexdigest()
            context_sha_before = hashlib.sha256(
                context_path.read_bytes()
            ).hexdigest()
            self.assertEqual(
                context_sha_before,
                PRODUCER.PINNED_JOINT_CONTEXT_SOURCE["sha256"],
            )
            specification = importlib.util.spec_from_file_location(
                "_pinned_joint_reference",
                helper_path,
            )
            self.assertIsNotNone(specification)
            self.assertIsNotNone(specification.loader)
            helper = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(helper)
            data_tools_payload = self._pinned_source_bytes(
                "dataloaders/data_tools.py"
            )
            self.assertEqual(
                hashlib.sha256(data_tools_payload).hexdigest(),
                (
                    "6fd248c2a13ce164c80bab2d76012fa57e4d5fcb507e43e6a5309c1b8fac4db8"
                ),
            )
            joints_list = self._literal_joints_list(data_tools_payload)
            package = ModuleType("dataloaders")
            package.__path__ = []  # type: ignore[attr-defined]
            data_tools = ModuleType("dataloaders.data_tools")
            data_tools.joints_list = joints_list  # type: ignore[attr-defined]
            with (
                mock.patch.object(
                    builtins,
                    "__import__",
                    side_effect=no_fasttext_import,
                ),
                mock.patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "dataloaders": package,
                        "dataloaders.data_tools": data_tools,
                    },
                ),
            ):
                arrays, receipt = PRODUCER._pinned_joint_mask_arrays(
                    helper,
                    pipeline,
                )
                reference = helper._joint_masks("cpu")
                observed = PRODUCER._joint_masks_from_arrays(arrays, "cpu")
            self.assertEqual(set(reference), {"upper", "hands", "lower"})
            for name in reference:
                np.testing.assert_array_equal(
                    observed[name].array,
                    reference[name].array,
                )
            self.assertEqual(
                receipt["source"],
                pipeline["source"],
            )
            self.assertEqual(receipt["pose_dim"], 165)
            self.assertEqual(receipt["dtype"], "bool")
            self.assertEqual(
                receipt["receipt_payload_sha256"],
                PRODUCER._payload_sha(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_payload_sha256"
                    }
                ),
            )
            _arrays_again, receipt_again = (
                PRODUCER._pinned_joint_mask_arrays(helper, pipeline)
            )
            self.assertEqual(receipt_again, receipt)
            self.assertEqual(
                hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                helper_sha_before,
            )
            self.assertEqual(
                hashlib.sha256(context_path.read_bytes()).hexdigest(),
                context_sha_before,
            )

    def test_pinned_joint_source_missing_or_corrupt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk_joint_failure_"
        ) as raw:
            root = Path(raw)
            helper_path, _joints_path, pipeline = (
                self._write_pinned_joint_fixture(
                    root,
                    include_context=False,
                )
            )
            helper = SimpleNamespace(
                __file__=str(helper_path),
                POSE_DIM=PRODUCER.SMPLX_POSE_DIM,
            )
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "joint-mask sources are absent",
            ):
                PRODUCER._pinned_joint_mask_arrays(helper, pipeline)
            helper.POSE_DIM = PRODUCER.SMPLX_POSE_DIM - 3
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "unexpected SMPL-X pose dimension",
            ):
                PRODUCER._pinned_joint_mask_arrays(helper, pipeline)

        with tempfile.TemporaryDirectory(
            prefix="semtalk_joint_corrupt_"
        ) as raw:
            root = Path(raw)
            helper_path, _joints_path, pipeline = (
                self._write_pinned_joint_fixture(
                    root,
                    corrupt_context=True,
                )
            )
            helper = SimpleNamespace(
                __file__=str(helper_path),
                POSE_DIM=PRODUCER.SMPLX_POSE_DIM,
            )
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "SHA mismatch",
            ):
                PRODUCER._pinned_joint_mask_arrays(helper, pipeline)

        with tempfile.TemporaryDirectory(
            prefix="semtalk_joint_authority_missing_"
        ) as raw:
            root = Path(raw)
            helper_path, _context_path, pipeline = (
                self._write_pinned_joint_fixture(
                    root,
                    include_authority=False,
                )
            )
            helper = SimpleNamespace(
                __file__=str(helper_path),
                POSE_DIM=PRODUCER.SMPLX_POSE_DIM,
            )
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "joint-mask sources are absent",
            ):
                PRODUCER._pinned_joint_mask_arrays(helper, pipeline)

        with tempfile.TemporaryDirectory(
            prefix="semtalk_joint_authority_corrupt_"
        ) as raw:
            root = Path(raw)
            helper_path, _context_path, pipeline = (
                self._write_pinned_joint_fixture(
                    root,
                    corrupt_authority=True,
                )
            )
            helper = SimpleNamespace(
                __file__=str(helper_path),
                POSE_DIM=PRODUCER.SMPLX_POSE_DIM,
            )
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "SHA mismatch",
            ):
                PRODUCER._pinned_joint_mask_arrays(helper, pipeline)

    def test_fresh_helper_has_no_released_representation_schema_dependency(
        self,
    ) -> None:
        helper_source = inspect.getsource(PRODUCER._load_pinned_helper)
        self.assertNotIn("_prime_pinned_released_schema_cache", helper_source)
        self.assertNotIn("_load_released_model_state_only", helper_source)
        self.assertNotIn(
            "_validate_released_model_state_schema", helper_source
        )
        shard_source = inspect.getsource(PRODUCER.run_shard)
        self.assertLess(
            shard_source.index("_load_pinned_helper(pipeline)"),
            shard_source.index("_set_deterministic(args.seed)"),
        )
        self.assertNotIn(
            "_prime_pinned_released_schema_cache",
            inspect.getsource(PRODUCER._load_models),
        )

    def test_verified_bytes_rejects_preopen_rename_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "input.json"
            archived = root / "input.before-open.json"
            original = b"original-input"
            target.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            real_open = os.open
            replaced = False

            def replace_then_open(
                path: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal replaced
                if (
                    not replaced
                    and path == target.name
                    and dir_fd is not None
                ):
                    target.rename(archived)
                    target.write_bytes(b"replaced-input")
                    replaced = True
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(
                PRODUCER.os,
                "open",
                side_effect=replace_then_open,
            ), self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "changed before it was opened",
            ):
                PRODUCER._verified_bytes(target, expected, "raced input")
            self.assertTrue(replaced)

    def test_artifact_builders_reject_postopen_rename_replacement(
        self,
    ) -> None:
        for function in (PRODUCER._artifact, PRODUCER._output_artifact):
            with self.subTest(
                function=function.__name__
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                target = root / "artifact.json"
                archived = root / "artifact.opened.json"
                target.write_bytes(b"original-artifact")
                real_open = os.open
                replaced = False

                def open_then_replace(
                    path: object,
                    flags: int,
                    mode: int = 0o777,
                    *,
                    dir_fd: int | None = None,
                ) -> int:
                    nonlocal replaced
                    descriptor = real_open(
                        path,
                        flags,
                        mode,
                        dir_fd=dir_fd,
                    )
                    if (
                        not replaced
                        and path == target.name
                        and dir_fd is not None
                    ):
                        target.rename(archived)
                        target.write_bytes(b"replaced-artifact")
                        replaced = True
                    return descriptor

                with mock.patch.object(
                    PRODUCER.os,
                    "open",
                    side_effect=open_then_replace,
                ), self.assertRaisesRegex(
                    PRODUCER.ValInferenceContractError,
                    "changed",
                ):
                    function(target)
                self.assertTrue(replaced)

    def test_copy_rejects_postopen_source_rename_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.npz"
            archived = root / "source.opened.npz"
            destination = root / "destination.npz"
            original = b"original-shard-output"
            source.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            real_open = os.open
            replaced = False

            def open_then_replace(
                path: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal replaced
                descriptor = real_open(
                    path,
                    flags,
                    mode,
                    dir_fd=dir_fd,
                )
                if (
                    not replaced
                    and path == source.name
                    and dir_fd is not None
                ):
                    source.rename(archived)
                    source.write_bytes(b"replaced-shard-output")
                    replaced = True
                return descriptor

            with mock.patch.object(
                PRODUCER.os,
                "open",
                side_effect=open_then_replace,
            ), self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "changed",
            ):
                PRODUCER._copy_inside_generation(
                    source,
                    destination,
                    expected_sha=expected,
                    expected_bytes=len(original),
                )
            self.assertTrue(replaced)
            self.assertFalse(destination.exists())

    def test_pinned_helper_executes_the_verified_byte_snapshot(self) -> None:
        source = inspect.getsource(PRODUCER._load_pinned_helper)
        self.assertIn("semtalk_base_inference_core.py", source)
        self.assertNotIn("run_base_" + "inference.py", source)
        self.assertIn("compile(source_snapshot", source)
        self.assertIn("exec(code, module.__dict__)", source)
        self.assertNotIn("exec_module(module)", source)

    def test_inference_auxiliary_loss_bypass_is_pinned_and_scoped(self) -> None:
        required = set(PRODUCER.selector.INFERENCE_HELPERS)
        self.assertIn(
            "_inference_only_auxiliary_loss_bypass",
            required,
        )
        self.assertIn(
            "_inference_auxiliary_loss_bypass_receipt",
            required,
        )
        shard_source = inspect.getsource(PRODUCER.run_shard)
        bypass = shard_source.index(
            "helper._inference_only_auxiliary_loss_bypass("
        )
        inference_mode = shard_source.rindex(
            "torch.inference_mode()",
            0,
            bypass,
        )
        infer = shard_source.index("helper._infer_clip(", bypass)
        self.assertLess(inference_mode, bypass)
        self.assertLess(bypass, infer)
        self.assertIn(
            'runtime_contract["auxiliary_loss_bypass"]',
            shard_source,
        )
        self.assertIn(
            "(frames - helper.PRE_FRAMES) / helper.STRIDE",
            shard_source,
        )

    def test_directory_tolerates_create_race_but_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk_val_dir_") as raw:
            root = Path(raw)
            target = root / "race"

            def create_then_race(path: Path, *args: object, **kwargs: object) -> None:
                del args, kwargs
                os.mkdir(path)
                raise FileExistsError(path)

            with (
                mock.patch("os.path.lexists", return_value=False),
                mock.patch.object(Path, "mkdir", create_then_race),
            ):
                self.assertEqual(
                    PRODUCER._directory(target, "race directory", create=True),
                    target.resolve(),
                )

            real_directory = root / "real"
            real_directory.mkdir()
            link = root / "link"
            link.symlink_to(real_directory, target_is_directory=True)
            with self.assertRaises(PRODUCER.ValInferenceContractError):
                PRODUCER._directory(link, "linked directory", create=True)

    def test_prepare_is_new_only_and_normalizes_integer_candidate_keys(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk_val_") as raw:
            root = Path(raw)
            output = root / "preflight.json"
            candidate = root / "candidate.bin"
            candidate.write_bytes(b"candidate")
            bundle = {
                "manifest": {"path": str(root / "manifest.json"), "sha256": "1" * 64},
                "status": {"path": str(root / "status.json"), "sha256": "2" * 64},
                "frozen_inputs": {
                    "path": str(root / "frozen.json"),
                    "sha256": "3" * 64,
                    "receipt_sha256": "4" * 64,
                },
                "candidates": {
                    epoch: {
                        "path": str(candidate),
                        "sha256": hashlib.sha256(b"candidate").hexdigest(),
                        "bytes": len(b"candidate"),
                    }
                    for epoch in PRODUCER.selector.EXPECTED_CANDIDATE_EPOCHS
                },
            }
            val_artifact = {
                "path": str(root / "val.json"),
                "sha256": "5" * 64,
                "receipt_payload_sha256": "6" * 64,
            }
            pipeline_artifact = {
                "path": str(root / "pipeline.json"),
                "sha256": "7" * 64,
                "receipt_payload_sha256": "8" * 64,
            }
            coverage = {
                "split": "val",
                "clip_count": 1715,
                "frame_count": 200000,
                "window_count": 1715,
                "uncovered_tail_frames": 0,
                "clip_ids_sha256": "9" * 64,
                "diffsheg_clip_manifest_sha256": "a" * 64,
                "_ordered_clips": [],
            }
            pipeline = {
                "source": PRODUCER.selector.VAL_INFERENCE_SOURCE,
                "inference_entrypoint": {
                    "path": str(root / "semtalk_base_inference_core.py"),
                    "sha256": hashlib.sha256(
                        self._pinned_source_bytes(
                            "scripts/show_base/"
                            "semtalk_base_inference_core.py"
                        )
                    ).hexdigest(),
                },
                "fixed_checkpoints": {
                    stage: {"sha256": str(index) * 64}
                    for index, stage in enumerate(
                        ("face", "hands", "upper", "lower", "global"),
                        start=1,
                    )
                },
            }
            args = PRODUCER.parse_args(
                [
                    "prepare",
                    "--split",
                    "val",
                    "--candidate-manifest",
                    str(root / "manifest.json"),
                    "--expected-candidate-manifest-sha256",
                    "1" * 64,
                    "--candidate-status",
                    str(root / "status.json"),
                    "--expected-candidate-status-sha256",
                    "2" * 64,
                    "--frozen-inputs",
                    str(root / "frozen.json"),
                    "--expected-frozen-inputs-sha256",
                    "3" * 64,
                    "--val-inputs",
                    str(root / "val.json"),
                    "--expected-val-inputs-sha256",
                    "5" * 64,
                    "--pipeline",
                    str(root / "pipeline.json"),
                    "--expected-pipeline-sha256",
                    "7" * 64,
                    "--output",
                    str(output),
                ]
            )
            with (
                mock.patch.object(
                    PRODUCER.selector,
                    "validate_candidate_bundle",
                    return_value=bundle,
                ),
                mock.patch.object(
                    PRODUCER.selector,
                    "validate_val_inputs",
                    return_value=(val_artifact, coverage),
                ),
                mock.patch.object(
                    PRODUCER.selector,
                    "validate_pipeline",
                    return_value=(pipeline_artifact, pipeline),
                ),
            ):
                artifact = PRODUCER.prepare(args)
                payload = json.loads(output.read_text())
                self.assertEqual(
                    set(payload["candidate_bundle"]["candidates"]),
                    {
                        str(epoch)
                        for epoch in (
                            PRODUCER.selector.EXPECTED_CANDIDATE_EPOCHS
                        )
                    },
                )
                self.assertEqual(artifact["path"], str(output.resolve()))
                with self.assertRaises(FileExistsError):
                    PRODUCER.prepare(args)

    def test_single_finalizer_copies_shards_and_publishes_one_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk_val_") as raw:
            root = Path(raw)
            output_root = root / "output"
            output_root.mkdir()
            preflight_path = root / "preflight.json"
            candidate_path = root / "candidate.bin"
            candidate_path.write_bytes(b"candidate")
            candidate_sha = hashlib.sha256(b"candidate").hexdigest()
            preflight_artifact = {
                "path": str(preflight_path),
                "sha256": "b" * 64,
                "receipt_payload_sha256": "c" * 64,
            }
            canonical_rows = [
                {
                    "global_index": position,
                    "clip_id": f"oliver/video-{position}/sequence-{position}",
                    "frames": 88,
                }
                for position in range(4)
            ]
            candidate = {
                "path": str(candidate_path.resolve()),
                "sha256": candidate_sha,
                "bytes": candidate_path.stat().st_size,
            }
            coverage = {
                "split": "val",
                "clip_count": 4,
                "frame_count": 4 * 88,
                "window_count": 4,
                "uncovered_tail_frames": 0,
                "clip_ids_sha256": hashlib.sha256(
                    "".join(
                        f"oliver__sequence-{position}\n"
                        for position in range(4)
                    ).encode()
                ).hexdigest(),
                "diffsheg_clip_manifest_sha256": "d" * 64,
            }
            preflight = {
                "candidate_bundle": {
                    "candidates": {"1": candidate},
                    "updates_per_epoch": (
                        PRODUCER.selector.EXPECTED_UPDATES_PER_EPOCH
                    ),
                    "frozen_inputs": {
                        "path": str(root / "frozen.json"),
                        "sha256": "2" * 64,
                        "receipt_sha256": "3" * 64,
                    },
                },
                "val_inputs_receipt": {
                    "path": str(root / "val.json"),
                    "sha256": "e" * 64,
                    "receipt_payload_sha256": "f" * 64,
                },
                "pipeline_receipt": {
                    "path": str(root / "pipeline.json"),
                    "sha256": "0" * 64,
                    "receipt_payload_sha256": "1" * 64,
                },
                "coverage": coverage,
            }
            prerequisite_selection = {
                "path": str((root / "prerequisite-selection.json").resolve()),
                "sha256": "4" * 64,
                "bytes": 123,
                "receipt_payload_sha256": "5" * 64,
            }
            fixed = {}
            for stage_index, stage_name in enumerate(
                ("face", "hands", "upper", "lower", "global"),
                start=1,
            ):
                rate = 1_988 if stage_name == "global" else 497
                fixed[stage_name] = {
                    "stage": stage_name,
                    "path": str((root / f"{stage_name}.pth").resolve()),
                    "sha256": f"{stage_index}" * 64,
                    "bytes": 100 + stage_index,
                    "source": "show_val_selected_v1",
                    "selection_split": "val",
                    "test_visible": False,
                    "epoch": 20,
                    "optimizer_updates": 20 * rate,
                    "updates_per_epoch": rate,
                    "candidate_audit_sha256": "a" * 64,
                    "selection_metric": f"{stage_name}_metric",
                    "measurement_receipt": {
                        "path": str(
                            (root / f"{stage_name}-measurement.json").resolve()
                        ),
                        "sha256": "b" * 64,
                        "receipt_payload_sha256": "c" * 64,
                    },
                }
            pipeline = {
                "fixed_checkpoints": fixed,
                "prerequisite_selection": prerequisite_selection,
            }
            num_shards = 2
            runtime = {"software": "same"}
            models = {
                "base": {
                    "path": str(candidate_path.resolve()),
                    "sha256": candidate_sha,
                    "bytes": candidate_path.stat().st_size,
                    "stage": "base",
                    "candidate_epoch": 1,
                    "optimizer_updates": PRODUCER.selector.EXPECTED_UPDATES_PER_EPOCH,
                    "updates_per_epoch": PRODUCER.selector.EXPECTED_UPDATES_PER_EPOCH,
                    "frozen_receipt_sha256": "3" * 64,
                    "strict_state_dict_load": True,
                    "all_model_state_tensors_finite": True,
                    "frozen_eval": True,
                }
            }
            for stage_name, fixed_row in fixed.items():
                models[stage_name] = {
                    **fixed_row,
                    "prerequisite_selection": prerequisite_selection,
                    "model_state_tensors": 7,
                    "model_state_schema_sha256": "d" * 64,
                    "strict_state_dict_load": True,
                    "all_model_state_tensors_finite": True,
                    "frozen_eval": True,
                }
            for shard_id in range(num_shards):
                shard_root = (
                    output_root
                    / PRODUCER.SHARDS_DIRECTORY
                    / PRODUCER._shard_name(shard_id, num_shards)
                )
                prediction_dir = shard_root / "predictions" / "val"
                ground_truth_dir = shard_root / "ground-truth" / "val"
                prediction_dir.mkdir(parents=True)
                ground_truth_dir.mkdir(parents=True)
                rows = []
                for position, canonical in enumerate(canonical_rows):
                    if position % num_shards != shard_id:
                        continue
                    output_id = PRODUCER.selector.canonical_clip_id(
                        canonical["clip_id"]
                    )
                    prediction = prediction_dir / f"res_{output_id}.npz"
                    ground_truth = ground_truth_dir / f"gt_{output_id}.npz"
                    prediction_payload = f"prediction:{position}".encode()
                    ground_truth_payload = f"ground-truth:{position}".encode()
                    prediction.write_bytes(prediction_payload)
                    ground_truth.write_bytes(ground_truth_payload)
                    rows.append(
                        {
                            "canonical_position": position,
                            "global_index": position,
                            "split": "val",
                            "source_clip_id": canonical["clip_id"],
                            "canonical_clip_id": output_id,
                            "frames": 88,
                            "epoch": 1,
                            "candidate_checkpoint_sha256": candidate_sha,
                            "prediction": {
                                "path": str(prediction.resolve()),
                                "sha256": hashlib.sha256(
                                    prediction_payload
                                ).hexdigest(),
                                "bytes": len(prediction_payload),
                            },
                            "ground_truth": {
                                "path": str(ground_truth.resolve()),
                                "sha256": hashlib.sha256(
                                    ground_truth_payload
                                ).hexdigest(),
                                "bytes": len(ground_truth_payload),
                            },
                        }
                    )
                manifest = shard_root / PRODUCER.SHARD_MANIFEST_FILENAME
                manifest_sha = _write(
                    manifest,
                    b"".join(_json_bytes(row) for row in rows),
                )
                receipt_body = {
                    "format": PRODUCER.SHARD_FORMAT,
                    "status": "complete",
                    "split": "val",
                    "test_visible": False,
                    "epoch": 1,
                    "candidate_checkpoint": {
                        "path": candidate["path"],
                        "sha256": candidate_sha,
                    },
                    "preflight_receipt": preflight_artifact,
                    "assignment": PRODUCER.ASSIGNMENT,
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                    "clip_count": len(rows),
                    "frame_count": len(rows) * 88,
                    "prediction_files": len(rows),
                    "ground_truth_files": len(rows),
                    "manifest": {
                        "path": str(manifest.resolve()),
                        "sha256": manifest_sha,
                    },
                    "model_receipts": models,
                    "model_receipts_sha256": PRODUCER._payload_sha(models),
                    "runtime_contract": runtime,
                    "runtime_contract_sha256": PRODUCER._payload_sha(runtime),
                    "device": {"device": f"cuda:{shard_id}"},
                    "exact_once": True,
                    "finite": True,
                }
                receipt = PRODUCER._with_payload_sha(receipt_body)
                _write(
                    shard_root / PRODUCER.SHARD_RECEIPT_FILENAME,
                    _json_bytes(receipt),
                )

            args = PRODUCER.parse_args(
                [
                    "finalize",
                    "--split",
                    "val",
                    "--preflight",
                    str(preflight_path),
                    "--expected-preflight-sha256",
                    "b" * 64,
                    "--epoch",
                    "1",
                    "--output-root",
                    str(output_root),
                    "--num-shards",
                    str(num_shards),
                ]
            )
            with (
                mock.patch.object(
                    PRODUCER,
                    "_preflight_artifact",
                    return_value=(preflight_artifact, preflight),
                ),
                mock.patch.object(
                    PRODUCER,
                    "_load_preflight_children",
                    return_value=({}, pipeline, canonical_rows, {}),
                ),
                mock.patch.object(
                    PRODUCER.selector,
                    "EXPECTED_VAL_CLIPS",
                    4,
                ),
                mock.patch.object(
                    PRODUCER.selector,
                    "validate_val_inference_lineage",
                ) as consumer,
            ):
                artifact = PRODUCER.finalize(args)
                final_root = output_root / PRODUCER.FINAL_DIRECTORY
                self.assertTrue(final_root.is_dir())
                self.assertEqual(
                    len(list((final_root / "predictions" / "val").iterdir())),
                    4,
                )
                self.assertEqual(
                    len(
                        list(
                            (final_root / "ground-truth" / "val").iterdir()
                        )
                    ),
                    4,
                )
                self.assertEqual(
                    artifact["path"],
                    str((final_root / PRODUCER.LINEAGE_FILENAME).resolve()),
                )
                consumer.assert_called_once()
                with self.assertRaises(FileExistsError):
                    PRODUCER.finalize(args)


if __name__ == "__main__":
    unittest.main()
