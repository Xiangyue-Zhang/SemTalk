from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
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
        return subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "show",
                (
                    f"{PRODUCER.selector.VAL_INFERENCE_SOURCE['commit']}:"
                    f"{path}"
                ),
            ],
            check=True,
            capture_output=True,
        ).stdout

    def _write_pinned_joint_fixture(
        self,
        root: Path,
        *,
        include_context: bool = True,
        corrupt_context: bool = False,
        include_authority: bool = True,
        corrupt_authority: bool = False,
    ) -> tuple[Path, Path, dict[str, object]]:
        helper_path = root / "scripts" / "show_base" / "run_base_inference.py"
        context_path = (
            root / PRODUCER.PINNED_JOINT_CONTEXT_SOURCE["relative_path"]
        )
        helper_payload = self._pinned_source_bytes(
            "scripts/show_base/run_base_inference.py"
        )
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
        pipeline = {
            "inference_entrypoint": {
                "path": str(helper_path.resolve()),
                "sha256": hashlib.sha256(helper_payload).hexdigest(),
            }
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
                PRODUCER.selector.VAL_INFERENCE_SOURCE,
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
            with self.assertRaises(FileNotFoundError):
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
            with self.assertRaises(FileNotFoundError):
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

    def test_pinned_schema_prime_precedes_deterministic_seed(self) -> None:
        helper_source = inspect.getsource(PRODUCER._load_pinned_helper)
        prime = helper_source.index(
            "_prime_pinned_released_schema_cache(module)"
        )
        self.assertLess(helper_source.index("if missing:"), prime)
        self.assertLess(prime, helper_source.index("return module"))
        shard_source = inspect.getsource(PRODUCER.run_shard)
        self.assertLess(
            shard_source.index("_load_pinned_helper(pipeline)"),
            shard_source.index("_set_deterministic(args.seed)"),
        )
        self.assertNotIn(
            "_prime_pinned_released_schema_cache",
            inspect.getsource(PRODUCER._load_models),
        )

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

    def test_pinned_meta_schema_cuda_shim_is_meta_only_and_scoped(
        self,
    ) -> None:
        class FakeTensor:
            def __init__(self, device_type: str) -> None:
                self.device = type(
                    "FakeDevice",
                    (),
                    {"type": device_type},
                )()

            def cuda(self, *args: object, **kwargs: object) -> str:
                del args, kwargs
                return "original-cuda"

        fake_torch = type("FakeTorch", (), {"Tensor": FakeTensor})()
        original_cuda = FakeTensor.cuda
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            with PRODUCER._pinned_meta_schema_cuda_compat() as intercepted:
                meta_tensor = FakeTensor("meta")
                self.assertIs(meta_tensor.cuda(), meta_tensor)
                with self.assertRaisesRegex(
                    PRODUCER.ValInferenceContractError,
                    "non-meta",
                ):
                    FakeTensor("cpu").cuda()
                with self.assertRaisesRegex(
                    PRODUCER.ValInferenceContractError,
                    "parameterized",
                ):
                    meta_tensor.cuda(0)
                self.assertEqual(intercepted, {"calls": 1})
            self.assertIs(FakeTensor.cuda, original_cuda)
            self.assertEqual(FakeTensor("cpu").cuda(), "original-cuda")

    def test_pinned_meta_schema_cache_is_primed_without_relaxing_stages(
        self,
    ) -> None:
        class FakeTensor:
            def __init__(self) -> None:
                self.device = type(
                    "FakeDevice",
                    (),
                    {"type": "meta"},
                )()

            def cuda(self) -> None:
                raise RuntimeError("legacy pinned helper meta failure")

        class FakeDType:
            pass

        fake_torch = type(
            "FakeTorch",
            (),
            {"Tensor": FakeTensor, "dtype": FakeDType},
        )()
        original_cuda = FakeTensor.cuda
        expected_stages = {"face", "global", "hands", "upper", "lower"}

        class FakeHelper:
            def __init__(
                self,
                *,
                intercepts: int = 24,
                shape: tuple[int, ...] = (1,),
                stable: bool = True,
            ) -> None:
                self.intercepts = intercepts
                self.shape = shape
                self.stable = stable
                self.cache: dict[
                    str,
                    dict[str, tuple[FakeDType, tuple[int, ...]]],
                ] | None = None

            def _expected_released_representation_schemas(self) -> dict[
                str,
                dict[str, tuple[FakeDType, tuple[int, ...]]],
            ]:
                if self.cache is None or not self.stable:
                    for _index in range(self.intercepts):
                        tensor = FakeTensor()
                        if tensor.cuda() is not tensor:
                            raise AssertionError(
                                "meta shim did not preserve tensor"
                            )
                    self.cache = {
                        stage: {"weight": (FakeDType(), self.shape)}
                        for stage in expected_stages
                    }
                return self.cache

        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            PRODUCER._prime_pinned_released_schema_cache(FakeHelper())
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "invalid released schema cache",
            ):
                PRODUCER._prime_pinned_released_schema_cache(
                    FakeHelper(intercepts=23)
                )
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "malformed released schema entry",
            ):
                PRODUCER._prime_pinned_released_schema_cache(
                    FakeHelper(shape=(-1,))
                )
            with self.assertRaisesRegex(
                PRODUCER.ValInferenceContractError,
                "cache is not stable",
            ):
                PRODUCER._prime_pinned_released_schema_cache(
                    FakeHelper(stable=False)
                )
        self.assertIs(FakeTensor.cuda, original_cuda)

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
                    "path": str(root / "run_base_inference.py"),
                    "sha256": PRODUCER.selector.VAL_INFERENCE_SOURCE[
                        "entrypoint_sha256"
                    ],
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
                    {"1", "2", "4", "8", "16", "32", "40"},
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
                "candidate_bundle": {"candidates": {"1": candidate}},
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
            num_shards = 2
            runtime = {"software": "same"}
            models = {"base": {"path": str(candidate_path), "sha256": candidate_sha}}
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
                    return_value=({}, {}, canonical_rows, {}),
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
