from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - minimal local environments
    torch = None

from utils import lower_target_cache as contract


REPOSITORY = Path(__file__).resolve().parents[1]
BUILDER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "build_lower_target_joints_cache.py"
)
CHECKER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "check_lower_target_joints_cache.py"
)


def load_script(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def protocol() -> dict[str, object]:
    return {
        "dataset": "show_base",
        "formal_stage": "lower",
        "speaker_scope": "All",
        "speaker_ids": [0, 1, 2, 3],
        "speaker_map": dict(contract.EXPECTED_SPEAKER_MAP),
        "split": "train",
        "entries": contract.EXPECTED_ENTRIES,
        "window_length": contract.WINDOW_LENGTH,
        "joints": contract.JOINT_COUNT,
        "coordinates": contract.COORDINATE_COUNT,
        "dtype": "<f4",
        "byte_order": "little",
        "storage": "raw_c_order",
        "key_format": "%010d",
        "exact_once": True,
        "compute_batch_windows": contract.COMPUTE_BATCH_WINDOWS,
        "compute_rows": contract.COMPUTE_ROWS,
        "full_compute_batches": contract.FULL_COMPUTE_BATCHES,
        "total_compute_batches": contract.TOTAL_COMPUTE_BATCHES,
        "tail_real_windows": contract.TAIL_REAL_WINDOWS,
        "tail_padding_windows": contract.TAIL_PADDING_WINDOWS,
        "tail_padding_policy": contract.TAIL_PADDING_POLICY,
        "lower_joint_indices": list(contract.LOWER_JOINT_INDICES),
        "lower_pose_columns": list(contract.LOWER_POSE_COLUMNS),
        "target_pose_preprocess": contract.TARGET_POSE_PREPROCESS,
    }


def receipts() -> tuple[dict[str, object], dict[str, object]]:
    canonical_source = {
        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "commit": "aa8e51951270e13f3dc5efad934a661f2c1efc28",
        "tree": "1634c7f15811cf3cbd3f44524d672162a21a7767",
        "entrypoint": "/frozen/build_show_cache.py",
        "entrypoint_sha256": "6" * 64,
    }
    canonical_receipt = {
        "manifest": "/frozen/canonical/manifest.jsonl",
        "manifest_sha256": "a" * 64,
        "summary": "/frozen/canonical/summary.json",
        "summary_sha256": "b" * 64,
        "lineage": "/frozen/canonical/lineage.json",
        "lineage_sha256": "c" * 64,
        "lineage_contract_sha256": "d" * 64,
        "source_receipt": canonical_source,
    }
    representation_source = {
        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "commit": "8a826248e5e0000419266870295d55d29b48744f",
        "tree": "f65c0962b46e7f09d4e855ca97791b1ea62bfcf9",
        "entrypoint": "/frozen/build_representation_lmdb.py",
        "entrypoint_sha256": "7" * 64,
    }
    representation = {
        "entries": contract.EXPECTED_ENTRIES,
        "format": "semtalk_show_representation_lmdb_v2_global_foot",
        "window_length": contract.WINDOW_LENGTH,
        "speaker_scope": "All",
        "speaker_ids": [0, 1, 2, 3],
        "speaker_map": dict(contract.EXPECTED_SPEAKER_MAP),
        "data_mdb_sha256": "a" * 64,
        "lock_mdb_sha256": "9" * 64,
        "summary_sha256": "b" * 64,
        "lineage_sha256": "c" * 64,
        "lineage_payload_sha256": "7" * 64,
        "entry_aggregate_sha256": "0" * 64,
        "canonical_smplx_asset_sha256": "d" * 64,
        "lmdb_path": "/frozen/representation.lmdb",
        "summary_path": "/frozen/representation-summary.json",
        "lineage_path": "/frozen/representation-summary.json",
        "source_receipt": representation_source,
        "canonical_manifest_sha256": {
            "/frozen/canonical/manifest.jsonl": "a" * 64,
        },
        "canonical_receipt": canonical_receipt,
    }
    smplx = {
        "model_dir": "/frozen/smplx-models",
        "asset_path": (
            "/frozen/smplx-models/smplx/SMPLX_NEUTRAL_2020.npz"
        ),
        "asset_sha256": "d" * 64,
        "model_type": "smplx",
        "gender": "NEUTRAL_2020",
        "num_betas": 300,
        "num_expression_coeffs": 100,
        "use_face_contour": False,
        "use_pca": False,
        "ext": "npz",
        "output_joints": 127,
        "return_verts": False,
        "return_joints": True,
        "return_shaped": False,
        "target_forward": "torch.no_grad+return_shaped_false",
        "translation": "tar_trans_minus_itself",
        "expression": "zeros_float32",
        "lower_joint_indices": list(contract.LOWER_JOINT_INDICES),
        "lower_pose_columns": list(contract.LOWER_POSE_COLUMNS),
        "target_pose_preprocess": contract.TARGET_POSE_PREPROCESS,
    }
    source = {
        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "commit": "8a826248e5e0000419266870295d55d29b48744f",
        "tree": "f65c0962b46e7f09d4e855ca97791b1ea62bfcf9",
        "entrypoint": "/frozen/build_lower_target_joints_cache.py",
        "entrypoint_sha256": "1" * 64,
    }
    manifest = {
        "format": contract.CACHE_FORMAT,
        "cache_version": contract.CACHE_VERSION,
        "status": "complete",
        "protocol": protocol(),
        "source_receipt": source,
        "representation_receipt": representation,
        "smplx_receipt": smplx,
        "runtime": {
            "python": "3.10.0",
            "numpy": "1.26.0",
            "torch": "frozen",
            "torch_cuda": "12.4",
            "smplx": "0.1.28",
            "device": "cuda:0",
            "rotation_conversions_sha256": "8" * 64,
            "rotation_conversions": "/frozen/rotation_conversions.py",
            "device_name": "NVIDIA H200",
            "device_capability": [9, 0],
            "default_dtype": "torch.float32",
            "grad_enabled_during_forward": False,
        },
        "lmdb": {
            "path": "/frozen/lower-target.lmdb",
            "map_size_bytes": 16 * 1024**3,
            "entries": contract.EXPECTED_ENTRIES,
            "data_mdb_sha256": "2" * 64,
            "lock_mdb_sha256": "3" * 64,
        },
        "entries": contract.EXPECTED_ENTRIES,
        "entry_shape": list(contract.ENTRY_SHAPE),
        "entry_bytes": contract.ENTRY_BYTES,
        "entry_aggregate_sha256": "4" * 64,
        "finite": True,
        "exact_once": True,
        "observed_speaker_ids": [0, 1, 2, 3],
        "target_requires_grad": False,
        "target_optimizer_member": False,
    }
    checker = {
        "format": contract.CHECKER_FORMAT,
        "cache_version": contract.CACHE_VERSION,
        "status": "complete",
        "protocol": protocol(),
        "source_receipt": {
            **source,
            "entrypoint": "/frozen/check_lower_target_joints_cache.py",
            "entrypoint_sha256": "5" * 64,
        },
        "representation_receipt": representation,
        "smplx_receipt": smplx,
        "runtime": manifest["runtime"],
        "cache_manifest_sha256": "6" * 64,
        "cache_path": "/frozen/lower-target.lmdb",
        "cache_data_mdb_sha256": "2" * 64,
        "entry_aggregate_sha256": "4" * 64,
        "traversal": {
            "method": "numpy_default_rng_producer_batch_permutation",
            "seed": 20_260_729,
            "entries": contract.EXPECTED_ENTRIES,
            "covers_all_entries": True,
            "duplicates": 0,
            "missing": 0,
            "batches": contract.TOTAL_COMPUTE_BATCHES,
            "covers_all_batches": True,
            "batch_duplicates": 0,
            "batch_missing": 0,
            "producer_batch_layout_preserved": True,
            "within_batch_read_order": (
                "numpy_default_rng_permutation_then_canonical_slot"
            ),
            "within_batch_compute_layout": "canonical_contiguous",
            "slot_mapping": "index_mod_64",
            "batch_windows": contract.COMPUTE_BATCH_WINDOWS,
            "tail_real_windows": contract.TAIL_REAL_WINDOWS,
            "tail_padding_windows": contract.TAIL_PADDING_WINDOWS,
            "tail_padding_policy": contract.TAIL_PADDING_POLICY,
        },
        "checked_entries": contract.EXPECTED_ENTRIES,
        "mismatch_count": 0,
        "torch_equal_all": True,
        "raw_bytes_equal_all": True,
        "exact_once": True,
        "finite": True,
        "observed_speaker_ids": [0, 1, 2, 3],
    }
    return manifest, checker


def representation_summary(
    representation: dict[str, object],
) -> dict[str, object]:
    return {
        "format": representation["format"],
        "status": "complete",
        "entries": contract.EXPECTED_ENTRIES,
        "train_clips": 13_687,
        "data_mdb_sha256": representation["data_mdb_sha256"],
        "lock_mdb_sha256": representation["lock_mdb_sha256"],
        "entry_aggregate_sha256": representation[
            "entry_aggregate_sha256"
        ],
        "canonical_receipt": representation["canonical_receipt"],
        "canonical_manifest_sha256": representation[
            "canonical_manifest_sha256"
        ],
        "source_receipt": representation["source_receipt"],
        "speaker_clip_counts": {
            "oliver": 5_246,
            "chemistry": 1_949,
            "seth": 1_984,
            "conan": 4_508,
        },
        "speaker_window_counts": {
            "oliver": 50_285,
            "chemistry": 14_374,
            "seth": 19_310,
            "conan": 43_317,
        },
        "protocol": {
            "split": "train",
            "window_length": contract.WINDOW_LENGTH,
            "stride": 20,
            "tail_policy": "drop_incomplete",
            "clip_boundary_policy": "floor_to_whole_seconds_at_30fps",
            "filtering": "none",
            "speaker_map": dict(contract.EXPECTED_SPEAKER_MAP),
            "global_foot_fastpath": {
                "enabled": True,
                "contract": "semtalk_show_global_foot_fastpath_v1",
                "field": "lower_foot_local",
                "shape": [contract.WINDOW_LENGTH, 4, 3],
                "dtype": "float32",
                "activation_env": (
                    "SEMTALK_SHOW_GLOBAL_FOOT_FASTPATH=1"
                ),
            },
        },
    }


class RawValueContractTests(unittest.TestCase):
    def test_exact_shape_dtype_endian_and_roundtrip(self) -> None:
        entry = np.arange(
            np.prod(contract.ENTRY_SHAPE),
            dtype=np.float32,
        ).reshape(contract.ENTRY_SHAPE)
        payload = contract.encode_raw_entry(entry)
        self.assertEqual(len(payload), 97_536)
        self.assertEqual(len(payload), contract.ENTRY_BYTES)
        self.assertEqual(
            payload[:4],
            np.asarray([0], dtype="<f4").tobytes(),
        )
        decoded = contract.decode_raw_entry(payload)
        self.assertEqual(decoded.shape, (64, 127, 3))
        self.assertEqual(decoded.dtype, np.float32)
        self.assertTrue(np.array_equal(entry, decoded))

    def test_producer_consumer_entry_aggregate_compatibility(self) -> None:
        entry = np.full(contract.ENTRY_SHAPE, np.float32(1.25))
        payload = contract.encode_raw_entry(entry)
        producer = hashlib.sha256()
        contract.update_entry_aggregate(producer, 7, payload)
        consumer = hashlib.sha256()
        consumer.update(b"0000000007")
        consumer.update(hashlib.sha256(payload).digest())
        self.assertEqual(producer.hexdigest(), consumer.hexdigest())
        self.assertTrue(
            np.array_equal(contract.decode_raw_entry(payload), entry)
        )

    def test_bad_shape_dtype_and_nonfinite_are_rejected(self) -> None:
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.encode_raw_entry(np.zeros((64, 126, 3), np.float32))
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.encode_raw_entry(
                np.zeros(contract.ENTRY_SHAPE, np.float64)
            )
        value = np.zeros(contract.ENTRY_SHAPE, np.float32)
        value[0, 0, 0] = np.nan
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.encode_raw_entry(value)
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.decode_raw_entry(b"\0" * (contract.ENTRY_BYTES - 4))

    def test_verified_json_hashes_and_parses_one_byte_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(b'{"value":"different-on-disk"}\n')
            snapshot = b'{"value":"frozen-snapshot"}\n'
            digest = hashlib.sha256(snapshot).hexdigest()
            with mock.patch.object(
                contract,
                "read_regular_file_snapshot",
                return_value=(path.resolve(), snapshot),
            ):
                loaded_path, payload, actual = contract._load_verified_json(
                    path,
                    digest,
                    "test receipt",
                )
            self.assertEqual(loaded_path, path.resolve())
            self.assertEqual(payload, {"value": "frozen-snapshot"})
            self.assertEqual(actual, digest)

    def test_frozen_batch_and_tail_arithmetic(self) -> None:
        self.assertEqual(contract.FULL_COMPUTE_BATCHES, 1_988)
        self.assertEqual(contract.TOTAL_COMPUTE_BATCHES, 1_989)
        self.assertEqual(contract.TAIL_REAL_WINDOWS, 54)
        self.assertEqual(contract.TAIL_PADDING_WINDOWS, 10)
        self.assertEqual(contract.COMPUTE_ROWS, 4_096)
        self.assertEqual(
            contract.FULL_COMPUTE_BATCHES
            * contract.COMPUTE_BATCH_WINDOWS
            + contract.TAIL_REAL_WINDOWS,
            contract.EXPECTED_ENTRIES,
        )
        self.assertEqual(
            contract.LOWER_POSE_COLUMNS,
            (
                0, 1, 2,
                3, 4, 5,
                6, 7, 8,
                12, 13, 14,
                15, 16, 17,
                21, 22, 23,
                24, 25, 26,
                30, 31, 32,
                33, 34, 35,
            ),
        )


class ReceiptContractTests(unittest.TestCase):
    def test_runtime_receipt_normalizes_version_string_subclasses(self) -> None:
        class VersionString(str):
            pass

        class FakeDevice:
            type = "cuda"

            def __str__(self) -> str:
                return "cuda:0"

        fake_torch = SimpleNamespace(
            __version__=VersionString("2.5.1"),
            version=SimpleNamespace(cuda=VersionString("12.4")),
            device=lambda _device: FakeDevice(),
            cuda=SimpleNamespace(
                get_device_name=lambda _device: "NVIDIA H200",
                get_device_capability=lambda _device: (9, 0),
            ),
            get_default_dtype=lambda: "torch.float32",
        )
        fake_rotation_conversions = SimpleNamespace(__file__=__file__)
        with (
            mock.patch.object(
                contract,
                "_require_torch",
                return_value=fake_torch,
            ),
            mock.patch(
                "importlib.metadata.version",
                return_value=VersionString("0.1.28"),
            ),
            mock.patch.dict(
                sys.modules,
                {"utils.rotation_conversions": fake_rotation_conversions},
            ),
        ):
            runtime = contract.current_runtime_receipt("cuda:0")
        for key in ("numpy", "torch", "torch_cuda", "smplx"):
            with self.subTest(key=key):
                self.assertIs(type(runtime[key]), str)

    def test_canonical_and_current_producer_sources_are_independent(self) -> None:
        manifest, _ = receipts()
        representation = manifest["representation_receipt"]
        summary = representation_summary(representation)
        canonical_source = representation["canonical_receipt"][
            "source_receipt"
        ]
        producer_source = representation["source_receipt"]
        self.assertNotEqual(
            (canonical_source["commit"], canonical_source["tree"]),
            (producer_source["commit"], producer_source["tree"]),
        )
        contract.validate_representation_summary_payload(
            summary,
            representation=representation,
        )

        tampered_summary = copy.deepcopy(summary)
        tampered_summary["source_receipt"] = copy.deepcopy(canonical_source)
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_representation_summary_payload(
                tampered_summary,
                representation=representation,
            )

        tampered_representation = copy.deepcopy(representation)
        tampered_representation["source_receipt"] = copy.deepcopy(
            canonical_source
        )
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_representation_summary_payload(
                summary,
                representation=tampered_representation,
            )

    def test_manifest_and_independent_checker_receipts_validate(self) -> None:
        manifest, checker = receipts()
        contract.validate_manifest_payload(manifest)
        contract.validate_checker_payload(
            checker,
            manifest_sha256="6" * 64,
            manifest=manifest,
        )

    def test_manifest_and_checker_tampering_fail_closed(self) -> None:
        manifest, _ = receipts()
        manifest["smplx_receipt"]["translation"] = "absolute_translation"
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_manifest_payload(manifest)

        manifest, _ = receipts()
        manifest["representation_receipt"]["speaker_map"] = {"oliver": 0}
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_manifest_payload(manifest)

        manifest, checker = receipts()
        checker["traversal"]["seed"] = 0
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_checker_payload(
                checker,
                manifest_sha256="6" * 64,
                manifest=manifest,
            )
        for key, value in (
            ("method", "numpy_default_rng_permutation"),
            ("covers_all_batches", False),
            ("producer_batch_layout_preserved", False),
            ("within_batch_read_order", "canonical"),
            ("within_batch_compute_layout", "permuted"),
            ("slot_mapping", "permutation_offset"),
        ):
            with self.subTest(checker_traversal_key=key):
                manifest, checker = receipts()
                checker["traversal"][key] = value
                with self.assertRaises(contract.LowerTargetCacheError):
                    contract.validate_checker_payload(
                        checker,
                        manifest_sha256="6" * 64,
                        manifest=manifest,
                    )

    def test_every_contract_integer_rejects_bool_string_and_float(self) -> None:
        manifest_paths = (
            ("cache_version",),
            ("entries",),
            ("entry_bytes",),
            ("entry_shape", 0),
            ("observed_speaker_ids", 0),
            ("protocol", "entries"),
            ("protocol", "window_length"),
            ("protocol", "speaker_ids", 0),
            ("protocol", "speaker_map", "oliver"),
            ("protocol", "lower_joint_indices", 0),
            ("protocol", "lower_pose_columns", 0),
            ("representation_receipt", "entries"),
            ("representation_receipt", "window_length"),
            ("representation_receipt", "speaker_ids", 0),
            ("representation_receipt", "speaker_map", "oliver"),
            ("smplx_receipt", "num_betas"),
            ("smplx_receipt", "num_expression_coeffs"),
            ("smplx_receipt", "output_joints"),
            ("runtime", "device_capability", 0),
            ("lmdb", "entries"),
            ("lmdb", "map_size_bytes"),
        )
        checker_paths = (
            ("cache_version",),
            ("checked_entries",),
            ("mismatch_count",),
            ("observed_speaker_ids", 0),
            ("protocol", "entries"),
            ("protocol", "speaker_ids", 0),
            ("protocol", "speaker_map", "oliver"),
            ("traversal", "seed"),
            ("traversal", "entries"),
            ("traversal", "duplicates"),
            ("traversal", "missing"),
            ("traversal", "batches"),
            ("traversal", "batch_duplicates"),
            ("traversal", "batch_missing"),
            ("traversal", "batch_windows"),
            ("traversal", "tail_real_windows"),
            ("traversal", "tail_padding_windows"),
        )

        def replace(root: object, path: tuple[object, ...], value: object) -> None:
            cursor = root
            for component in path[:-1]:
                cursor = cursor[component]
            cursor[path[-1]] = value

        for path in manifest_paths:
            for bad_value in (True, "1", 1.0):
                with self.subTest(kind="manifest", path=path, bad=bad_value):
                    manifest, _ = receipts()
                    replace(manifest, path, bad_value)
                    with self.assertRaises(contract.LowerTargetCacheError):
                        contract.validate_manifest_payload(manifest)
        for path in checker_paths:
            for bad_value in (True, "1", 1.0):
                with self.subTest(kind="checker", path=path, bad=bad_value):
                    manifest, checker = receipts()
                    replace(checker, path, bad_value)
                    with self.assertRaises(contract.LowerTargetCacheError):
                        contract.validate_checker_payload(
                            checker,
                            manifest_sha256="6" * 64,
                            manifest=manifest,
                        )

    def test_protocol_rejects_extra_keys_and_boolean_integer_aliases(self) -> None:
        manifest, _ = receipts()
        manifest["protocol"]["unexpected_count"] = 1
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_manifest_payload(manifest)

        manifest, _ = receipts()
        manifest["protocol"]["exact_once"] = 1
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_manifest_payload(manifest)

    def test_checker_must_have_independent_entrypoint_and_exact_equality(self) -> None:
        manifest, checker = receipts()
        checker["source_receipt"] = manifest["source_receipt"]
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_checker_payload(
                checker,
                manifest_sha256="6" * 64,
                manifest=manifest,
            )
        manifest, checker = receipts()
        checker["torch_equal_all"] = False
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_checker_payload(
                checker,
                manifest_sha256="6" * 64,
                manifest=manifest,
            )
        manifest, checker = receipts()
        checker["raw_bytes_equal_all"] = False
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_checker_payload(
                checker,
                manifest_sha256="6" * 64,
                manifest=manifest,
            )

    def test_resume_and_final_receipt_interface_is_immutable(self) -> None:
        receipt = {
            "format": contract.CACHE_FORMAT,
            "receipt_sha256": "7" * 64,
        }
        payload: dict[str, object] = {}
        contract.attach_lower_target_cache_receipt(payload, receipt)
        contract.verify_lower_target_cache_resume_receipt(payload, receipt)
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.attach_lower_target_cache_receipt(
                payload,
                {"format": "different"},
            )
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.verify_lower_target_cache_resume_receipt(
                payload,
                {"format": "different"},
            )
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.verify_lower_target_cache_resume_receipt(payload, None)

    def test_null_receipt_key_never_bypasses_attach_or_resume(self) -> None:
        receipt = {
            "format": contract.CACHE_FORMAT,
            "receipt_sha256": "7" * 64,
        }
        for expected in (None, receipt):
            with self.subTest(operation="resume", expected=expected):
                with self.assertRaises(contract.LowerTargetCacheError):
                    contract.verify_lower_target_cache_resume_receipt(
                        {contract.RECEIPT_KEY: None},
                        expected,
                    )
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.attach_lower_target_cache_receipt(
                {contract.RECEIPT_KEY: None},
                receipt,
            )
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.attach_lower_target_cache_receipt(
                {contract.RECEIPT_KEY: None},
                None,
            )

    def test_current_inputs_bind_actual_representation_smplx_and_runtime(self) -> None:
        manifest, _ = receipts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            representation = root / "representation.lmdb"
            representation.mkdir()
            data = representation / "data.mdb"
            lock = representation / "lock.mdb"
            data.write_bytes(b"representation-data")
            lock.write_bytes(b"representation-lock")
            summary = root / "representation-summary.json"
            summary_payload = {"status": "complete", "token": "frozen"}
            summary.write_bytes(contract.canonical_json_bytes(summary_payload))
            model_dir = root / "models"
            asset = model_dir / "smplx" / "SMPLX_NEUTRAL_2020.npz"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"smplx-asset")

            representation_receipt = manifest["representation_receipt"]
            representation_receipt.update(
                {
                    "lmdb_path": str(representation.resolve()),
                    "data_mdb_sha256": contract.sha256_file(data),
                    "lock_mdb_sha256": contract.sha256_file(lock),
                    "summary_path": str(summary.resolve()),
                    "lineage_path": str(summary.resolve()),
                    "summary_sha256": contract.sha256_file(summary),
                    "lineage_sha256": contract.sha256_file(summary),
                    "lineage_payload_sha256": (
                        contract.canonical_json_sha256(summary_payload)
                    ),
                }
            )
            smplx_receipt = manifest["smplx_receipt"]
            smplx_receipt.update(
                {
                    "model_dir": str(model_dir.resolve()),
                    "asset_path": str(asset.resolve()),
                    "asset_sha256": contract.sha256_file(asset),
                }
            )
            runtime = {"runtime": "exact"}
            manifest["runtime"] = runtime
            args = SimpleNamespace(
                train_path=str(representation),
                dataset_summary=str(summary),
                lineage_manifest=str(summary),
                expected_smplx_asset_sha256=contract.sha256_file(asset),
            )
            with (
                mock.patch.object(
                    contract,
                    "validate_representation_summary_payload",
                ),
                mock.patch.object(
                    contract,
                    "validate_canonical_files",
                    return_value={"status": "bound"},
                ),
                mock.patch.object(
                    contract,
                    "validate_representation_canonical_coverage",
                ),
                mock.patch.object(
                    contract,
                    "current_runtime_receipt",
                    return_value=runtime,
                ),
                mock.patch(
                    "utils.project_paths.smplx_model_dir",
                    return_value=model_dir,
                ),
            ):
                current = contract.validate_current_inputs_against_manifest(
                    args,
                    manifest,
                    device="cuda:0",
                )
                self.assertEqual(
                    current["representation_lmdb_path"],
                    str(representation.resolve()),
                )
                self.assertEqual(
                    current["smplx_asset_sha256"],
                    contract.sha256_file(asset),
                )
                args.expected_smplx_asset_sha256 = "0" * 64
                with self.assertRaises(contract.LowerTargetCacheError):
                    contract.validate_current_inputs_against_manifest(
                        args,
                        manifest,
                        device="cuda:0",
                    )

    def test_canonical_lineage_deep_validation_covers_rows_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(contract.EXPECTED_SPLIT_MISSING_COUNT, 55)
            source = {
                "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                "commit": "a" * 40,
                "tree": "b" * 40,
                "entrypoint": str(root / "build_show_cache.py"),
                "entrypoint_sha256": "c" * 64,
            }
            lineage_contract = {
                "schema_name": "semtalk-show-canonical-motion",
                "schema_version": 1,
                "source_receipt": source,
                "split_counts": {
                    "train": 13_687,
                    "val": 1_715,
                    "test": 1_708,
                },
                "split_missing_count": 55,
                "speaker_mapping": dict(contract.EXPECTED_SPEAKER_MAP),
                "pose_fps": 30,
                "source_audio_sample_rate": 22_000,
                "hubert_target_sample_rate": 16_000,
                "smplx_asset_path": "/frozen/SMPLX_NEUTRAL_2020.npz",
                "smplx_asset_sha256": "9" * 64,
            }
            lineage_contract_sha = contract.canonical_json_sha256(
                lineage_contract
            )
            manifest = root / "manifest.jsonl"
            rows = []
            speaker_names = tuple(contract.EXPECTED_SPEAKER_MAP)
            for index in range(17_110):
                if index < 13_687:
                    split = "train"
                elif index < 13_687 + 1_715:
                    split = "val"
                else:
                    split = "test"
                speaker = speaker_names[index % len(speaker_names)]
                rows.append(
                    {
                        "global_index": index,
                        "clip_id": f"clip-{index:05d}",
                        "split": split,
                        "speaker": speaker,
                        "speaker_id": contract.EXPECTED_SPEAKER_MAP[speaker],
                        "frames": 1,
                        "pose_fps": 30,
                        "lineage_contract_sha256": lineage_contract_sha,
                        "source_pkl_sha256": "1" * 64,
                        "source_wav_sha256": "2" * 64,
                        "canonical_npz_sha256": "3" * 64,
                        "lower_foot_local_sha256": "4" * 64,
                    }
                )
            manifest.write_bytes(
                b"".join(contract.canonical_json_bytes(row) for row in rows)
            )
            shard_manifest = root / "shard-manifest.jsonl"
            shard_summary = root / "shard-summary.json"
            shard_lineage = root / "shard-lineage.json"
            shard_manifest.write_bytes(b"shard-manifest\n")
            shard_summary.write_bytes(b"shard-summary\n")
            shard_lineage.write_bytes(b"shard-lineage\n")
            build_runtime = {"runtime": "frozen"}
            build_runtime_sha = contract.canonical_json_sha256(build_runtime)
            canonical_lineage = {
                "lineage_contract": lineage_contract,
                "lineage_contract_sha256": lineage_contract_sha,
                "build_runtime": build_runtime,
                "build_runtime_sha256": build_runtime_sha,
                "shards": [
                    {
                        "shard_id": 0,
                        "manifest_path": str(shard_manifest.resolve()),
                        "manifest_sha256": contract.sha256_file(shard_manifest),
                        "summary_path": str(shard_summary.resolve()),
                        "summary_sha256": contract.sha256_file(shard_summary),
                        "lineage_path": str(shard_lineage.resolve()),
                        "lineage_sha256": contract.sha256_file(shard_lineage),
                        "clip_count": 17_110,
                    }
                ],
                "final_manifest_sha256": contract.sha256_file(manifest),
            }
            lineage = root / "lineage.json"
            lineage.write_bytes(
                contract.canonical_json_bytes(canonical_lineage)
            )
            canonical_summary = {
                "status": "complete",
                "schema_name": "semtalk-show-canonical-motion",
                "schema_version": 1,
                "clip_count": 17_110,
                "frame_count": 17_110,
                "split_counts": {
                    "train": 13_687,
                    "val": 1_715,
                    "test": 1_708,
                },
                "num_shards": 1,
                "manifest_sha256": contract.sha256_file(manifest),
                "lineage_contract_sha256": lineage_contract_sha,
                "source_receipt_sha256": contract.canonical_json_sha256(source),
                "build_runtime_sha256": build_runtime_sha,
                "lineage_sha256": contract.canonical_json_sha256(
                    canonical_lineage
                ),
                "finite": True,
                "exact_once": True,
                "split_disjoint": True,
            }
            summary = root / "summary.json"
            summary.write_bytes(
                contract.canonical_json_bytes(canonical_summary)
            )
            canonical_receipt = {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": contract.sha256_file(manifest),
                "summary": str(summary.resolve()),
                "summary_sha256": contract.sha256_file(summary),
                "lineage": str(lineage.resolve()),
                "lineage_sha256": contract.sha256_file(lineage),
                "lineage_contract_sha256": lineage_contract_sha,
                "source_receipt": source,
            }
            representation = {
                "canonical_receipt": canonical_receipt,
                "canonical_manifest_sha256": {
                    str(manifest.resolve()): contract.sha256_file(manifest),
                },
            }
            validated = contract.validate_canonical_files(representation)
            self.assertEqual(
                validated["manifest_sha256"],
                contract.sha256_file(manifest),
            )
            tampered_representation = copy.deepcopy(representation)
            tampered_representation["canonical_receipt"][
                "source_receipt"
            ]["commit"] = "d" * 40
            with self.assertRaises(contract.LowerTargetCacheError):
                contract.validate_canonical_files(
                    tampered_representation
                )

    def test_representation_counts_are_bound_to_canonical_train_rows(self) -> None:
        clip_counts = {
            "oliver": 5_246,
            "chemistry": 1_949,
            "seth": 1_984,
            "conan": 4_508,
        }
        window_counts = {
            "oliver": 50_285,
            "chemistry": 14_374,
            "seth": 19_310,
            "conan": 43_317,
        }
        summary = {
            "speaker_clip_counts": dict(clip_counts),
            "speaker_window_counts": dict(window_counts),
        }
        canonical = {
            "train_clips": 13_687,
            "train_windows": 127_286,
            "train_speaker_clip_counts": dict(clip_counts),
            "train_speaker_window_counts": dict(window_counts),
        }
        contract.validate_representation_canonical_coverage(
            summary,
            canonical,
        )
        summary["speaker_window_counts"]["oliver"] = True
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_representation_canonical_coverage(
                summary,
                canonical,
            )

    def test_activation_is_explicit_all_only_and_never_partial(self) -> None:
        baseline = SimpleNamespace(
            use_lower_target_joints_cache=False,
            lower_target_joints_cache=None,
            lower_target_joints_cache_manifest=None,
            expected_lower_target_joints_cache_manifest_sha256=None,
            lower_target_joints_cache_checker_receipt=None,
            expected_lower_target_joints_cache_checker_sha256=None,
        )
        self.assertFalse(contract.validate_activation_args(baseline))
        baseline.lower_target_joints_cache = "/unexpected"
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_activation_args(baseline)

        formal = SimpleNamespace(
            use_lower_target_joints_cache=True,
            lower_target_joints_cache="/cache",
            lower_target_joints_cache_manifest="/manifest",
            expected_lower_target_joints_cache_manifest_sha256="a" * 64,
            lower_target_joints_cache_checker_receipt="/checker",
            expected_lower_target_joints_cache_checker_sha256="b" * 64,
            dataset="show_base",
            formal_stage="lower",
            tar_joints="beat_smplx_lower",
            train_only=True,
            pose_length=64,
            batch_size=64,
            training_speakers=[0, 1, 2, 3],
            rec_ver_weight=1.0,
            train_path="/representation.lmdb",
            dataset_summary="/representation-summary.json",
            lineage_manifest="/representation-summary.json",
            expected_smplx_asset_sha256="c" * 64,
        )
        self.assertTrue(contract.validate_activation_args(formal))
        formal.rec_ver_weight = float("nan")
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_activation_args(formal)
        formal.rec_ver_weight = 1.0
        formal.training_speakers = [1]
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.validate_activation_args(formal)


@unittest.skipIf(torch is None, "torch is unavailable")
class FrozenTensorTests(unittest.TestCase):
    def test_index_select_is_exact_no_grad_and_order_preserving(self) -> None:
        tensor = torch.arange(
            4 * 64 * 127 * 3,
            dtype=torch.float32,
        ).reshape(4, 64, 127, 3)
        indices = torch.tensor([3, 0, 3, 1], dtype=torch.int64)
        selected = contract.index_select_frozen_targets(
            tensor,
            indices,
            expected_entries=4,
        )
        self.assertTrue(
            torch.equal(selected, torch.stack([tensor[3], tensor[0], tensor[3], tensor[1]]))
        )
        self.assertFalse(selected.requires_grad)
        self.assertIsNone(selected.grad_fn)

    def test_index_select_rejects_float_mutable_or_out_of_range_index(self) -> None:
        tensor = torch.zeros((2, 64, 127, 3), dtype=torch.float32)
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.index_select_frozen_targets(
                tensor,
                torch.tensor([0.0]),
                expected_entries=2,
            )
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.index_select_frozen_targets(
                tensor,
                torch.tensor([2], dtype=torch.int64),
                expected_entries=2,
            )
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.index_select_frozen_targets(
                tensor.requires_grad_(),
                torch.tensor([0], dtype=torch.int64),
                expected_entries=2,
            )

    def test_optimizer_exclusion_is_enforced(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        frozen = torch.zeros((1, 64, 127, 3))
        contract.assert_optimizer_excludes_tensor(optimizer, frozen)
        with self.assertRaises(contract.LowerTargetCacheError):
            contract.assert_optimizer_excludes_tensor(optimizer, parameter)


class StaticIntegrationContractTests(unittest.TestCase):
    def test_builder_validates_manifest_before_publishing_lmdb(self) -> None:
        source = BUILDER.read_text(encoding="utf-8")
        validation = source.index("validate_manifest_payload(manifest)")
        publication = source.index("os.replace(temporary, output)")
        self.assertLess(validation, publication)
        self.assertIn("shutil.rmtree(output, ignore_errors=True)", source)

    def test_producer_checker_protocol_and_raw_payload_are_compatible(self) -> None:
        builder = load_script(BUILDER, "lower_target_builder_under_test")
        checker = load_script(CHECKER, "lower_target_checker_under_test")
        self.assertEqual(builder.protocol_receipt(), checker._protocol())
        entry = np.arange(
            np.prod(contract.ENTRY_SHAPE),
            dtype=np.float32,
        ).reshape(contract.ENTRY_SHAPE)
        payload = contract.encode_raw_entry(entry)
        self.assertTrue(np.array_equal(checker._decode_raw(payload), entry))

    def test_checker_randomizes_only_complete_producer_batches(self) -> None:
        checker = load_script(CHECKER, "lower_target_checker_batch_order")
        permutation = checker._producer_batch_permutation()
        self.assertEqual(
            permutation.shape,
            (contract.TOTAL_COMPUTE_BATCHES,),
        )
        self.assertEqual(
            set(int(value) for value in permutation),
            set(range(contract.TOTAL_COMPUTE_BATCHES)),
        )
        self.assertNotEqual(
            [int(value) for value in permutation],
            list(range(contract.TOTAL_COMPUTE_BATCHES)),
        )
        covered: list[int] = []
        tail_layouts: list[tuple[int, int]] = []
        for raw_batch_index in permutation:
            batch_index = int(raw_batch_index)
            start = batch_index * contract.COMPUTE_BATCH_WINDOWS
            stop = min(
                start + contract.COMPUTE_BATCH_WINDOWS,
                contract.EXPECTED_ENTRIES,
            )
            for offset, index in enumerate(range(start, stop)):
                self.assertEqual(index % contract.COMPUTE_BATCH_WINDOWS, offset)
            read_permutation = checker._producer_batch_read_permutation(
                batch_index,
                stop - start,
            )
            self.assertEqual(
                set(int(value) for value in read_permutation),
                set(range(stop - start)),
            )
            covered.extend(range(start, stop))
            if stop - start != contract.COMPUTE_BATCH_WINDOWS:
                tail_layouts.append((batch_index, stop - start))
        self.assertEqual(
            sorted(covered),
            list(range(contract.EXPECTED_ENTRIES)),
        )
        self.assertEqual(
            tail_layouts,
            [
                (
                    contract.TOTAL_COMPUTE_BATCHES - 1,
                    contract.TAIL_REAL_WINDOWS,
                )
            ],
        )

    def test_checker_restores_random_reads_to_canonical_slots(self) -> None:
        checker = load_script(CHECKER, "lower_target_checker_slot_restore")
        batch_index = 1_056
        start = batch_index * contract.COMPUTE_BATCH_WINDOWS
        read_permutation = checker._producer_batch_read_permutation(
            batch_index,
            contract.COMPUTE_BATCH_WINDOWS,
        )
        restored: list[str | None] = [None] * contract.COMPUTE_BATCH_WINDOWS
        for raw_offset in read_permutation:
            index = start + int(raw_offset)
            slot = checker._canonical_batch_offset(
                batch_start=start,
                index=index,
                real_windows=contract.COMPUTE_BATCH_WINDOWS,
            )
            restored[slot] = f"sentinel-{index}"
        self.assertEqual(
            restored,
            [
                f"sentinel-{start + offset}"
                for offset in range(contract.COMPUTE_BATCH_WINDOWS)
            ],
        )

    def test_checker_tail_layout_keeps_ten_zero_padding_slots(self) -> None:
        real_windows = contract.TAIL_REAL_WINDOWS
        pose = np.zeros(
            (contract.COMPUTE_BATCH_WINDOWS, contract.WINDOW_LENGTH, 165),
            dtype=np.float32,
        )
        beta = np.zeros(
            (contract.COMPUTE_BATCH_WINDOWS, contract.WINDOW_LENGTH, 300),
            dtype=np.float32,
        )
        trans = np.zeros(
            (contract.COMPUTE_BATCH_WINDOWS, contract.WINDOW_LENGTH, 3),
            dtype=np.float32,
        )
        pose[:real_windows] = 1.0
        beta[:real_windows] = 2.0
        trans[:real_windows] = 3.0
        self.assertTrue(np.all(pose[:real_windows] == 1.0))
        self.assertTrue(np.all(beta[:real_windows] == 2.0))
        self.assertTrue(np.all(trans[:real_windows] == 3.0))
        self.assertTrue(np.all(pose[real_windows:] == 0.0))
        self.assertTrue(np.all(beta[real_windows:] == 0.0))
        self.assertTrue(np.all(trans[real_windows:] == 0.0))
        self.assertEqual(
            contract.COMPUTE_BATCH_WINDOWS - real_windows,
            contract.TAIL_PADDING_WINDOWS,
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_raw_byte_check_distinguishes_signed_zero(self) -> None:
        positive = np.asarray([0.0], dtype=np.dtype("<f4"))
        negative = np.asarray([-0.0], dtype=np.dtype("<f4"))
        self.assertTrue(torch.equal(
            torch.from_numpy(positive),
            torch.from_numpy(negative),
        ))
        self.assertNotEqual(
            positive.tobytes(order="C"),
            negative.tobytes(order="C"),
        )

    def test_builder_and_checker_are_independent_and_gpu_imports_are_lazy(self) -> None:
        for path in (BUILDER, CHECKER):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            top_level_modules = {
                alias.name
                for node in tree.body
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertNotIn("lmdb", top_level_modules)
            self.assertNotIn("smplx", top_level_modules)
            self.assertNotIn("torch", top_level_modules)
        builder_source = BUILDER.read_text(encoding="utf-8")
        checker_source = CHECKER.read_text(encoding="utf-8")
        self.assertNotIn("check_lower_target_joints_cache", builder_source)
        self.assertNotIn("build_lower_target_joints_cache", checker_source)
        self.assertIn("np.random.default_rng", checker_source)
        self.assertIn("torch.equal", checker_source)
        self.assertIn("live_payload != cached_payload", checker_source)
        self.assertNotIn("torch.allclose", checker_source)
        for module, function_name in (
            (load_script(BUILDER, "builder_json_snapshot_test"), "verified_json"),
            (load_script(CHECKER, "checker_json_snapshot_test"), "_verified_json"),
        ):
            with self.subTest(function=function_name):
                snapshot = b'{"snapshot":true}\n'
                digest = hashlib.sha256(snapshot).hexdigest()
                path = Path("/frozen/receipt.json")
                with mock.patch.object(
                    module,
                    "read_regular_file_snapshot",
                    return_value=(path, snapshot),
                ):
                    _, payload, actual = getattr(module, function_name)(
                        str(path),
                        digest,
                        "snapshot",
                    )
                self.assertEqual(payload, {"snapshot": True})
                self.assertEqual(actual, digest)

    def test_both_paths_reconstruct_the_trainer_exact_pose_slices(self) -> None:
        builder = " ".join(BUILDER.read_text(encoding="utf-8").split())
        checker = " ".join(CHECKER.read_text(encoding="utf-8").split())
        expected_builder = (
            "jaw_pose=flat_pose[:, 66:69]",
            "global_orient=flat_pose[:, 0:3]",
            "body_pose=flat_pose[:, 3:66]",
            "left_hand_pose=flat_pose[:, 75:120]",
            "right_hand_pose=flat_pose[:, 120:165]",
            "leye_pose=flat_pose[:, 69:72]",
            "reye_pose=flat_pose[:, 72:75]",
            "transl=flat_trans - flat_trans",
        )
        expected_checker = (
            "jaw_pose=pose_tensor[:, 66:69]",
            "global_orient=pose_tensor[:, :3]",
            "body_pose=pose_tensor[:, 3:66]",
            "left_hand_pose=pose_tensor[:, 75:120]",
            "right_hand_pose=pose_tensor[:, 120:165]",
            "leye_pose=pose_tensor[:, 69:72]",
            "reye_pose=pose_tensor[:, 72:75]",
            "transl=trans_tensor - trans_tensor",
        )
        for token in expected_builder:
            self.assertIn(token, builder)
        for token in expected_checker:
            self.assertIn(token, checker)
        for source in (builder, checker):
            self.assertIn("axis_angle_to_matrix", source)
            self.assertIn("matrix_to_rotation_6d", source)
            self.assertIn("rotation_6d_to_matrix", source)
            self.assertIn("matrix_to_axis_angle", source)
            self.assertIn("LOWER_POSE_COLUMNS", source)
            self.assertIn("return_shaped=False", source)

    def test_consumer_recomputes_full_aggregate_and_rehashes_lmdb(self) -> None:
        source = (
            REPOSITORY / "utils" / "lower_target_cache.py"
        ).read_text(encoding="utf-8")
        preload = source[source.index("def from_formal_args("):]
        self.assertIn("loaded_entries != EXPECTED_ENTRIES", preload)
        self.assertIn("update_entry_aggregate(", preload)
        self.assertIn("aggregate.hexdigest()", preload)
        self.assertIn('sha256_file(cache_path / "data.mdb")', preload)
        self.assertIn(
            "final_data_sha != receipt[\"data_mdb_sha256\"]",
            preload,
        )

    def test_consumer_uses_immutable_dataset_index_and_explicit_branch(self) -> None:
        loader_source = (
            REPOSITORY / "dataloaders" / "show_base.py"
        ).read_text(encoding="utf-8")
        trainer_source = (
            REPOSITORY / "aelower_trainer.py"
        ).read_text(encoding="utf-8")
        config_source = (
            REPOSITORY / "utils" / "config.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"sample_index": np.int64(index)', loader_source)
        self.assertIn("validate_activation_args(args)", trainer_source)
        self.assertIn(
            'dict_data["sample_index"]',
            trainer_source,
        )
        self.assertIn(
            "LowerTargetJointsCache.from_formal_args",
            trainer_source,
        )
        self.assertIn("--use_lower_target_joints_cache", config_source)
        self.assertIn(
            "--expected_lower_target_joints_cache_checker_sha256",
            config_source,
        )
        self.assertIn("--expected_smplx_asset_sha256", config_source)

    def test_formal_launcher_uses_audited_live_lower_backend(self) -> None:
        launcher = (
            REPOSITORY
            / "scripts"
            / "show_base"
            / "run_five_prerequisites.sh"
        ).read_text(encoding="utf-8")
        formal = (
            REPOSITORY / "show_base_train.py"
        ).read_text(encoding="utf-8")
        config_source = (
            REPOSITORY / "utils" / "config.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("LOWER_TARGET_CACHE", launcher)
        self.assertNotIn("lower_target_manifest", launcher)
        self.assertNotIn("lower_target_checker", launcher)
        self.assertNotIn("lower_target_gate", launcher)
        self.assertNotIn("lower_target_builder_process", launcher)
        self.assertIn("--use_lower_target_joints_cache false", launcher)
        self.assertNotIn("--use_lower_target_joints_cache true", launcher)
        self.assertIn(
            'LOWER_TARGET_BACKEND_RECEIPT_KEY = "lower_target_backend"',
            formal,
        )
        self.assertIn(
            "_verify_lower_target_backend_resume_receipt(",
            formal,
        )
        self.assertGreaterEqual(
            formal.count("_attach_lower_target_backend_receipt("),
            3,
        )
        self.assertEqual(
            config_source.count(
                'parser.add("--expected_smplx_asset_sha256"'
            ),
            1,
        )
        for relative in (
            "scripts/show_base/build_base_features.py",
            "scripts/show_base/run_base_inference.py",
        ):
            consumer = (REPOSITORY / relative).read_text(encoding="utf-8")
            self.assertIn(
                "LOWER_TARGET_BACKEND_RECEIPT_KEY",
                consumer,
            )
            self.assertIn(
                "lower live backend receipt is missing or inconsistent",
                consumer,
            )

    def test_new_cache_scope_contains_no_sparse_semgate_or_speaker2_path(self) -> None:
        paths = (
            BUILDER,
            CHECKER,
            REPOSITORY / "utils" / "lower_target_cache.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8").lower()
            for forbidden in ("semgate", "sparse", "speaker2"):
                self.assertNotIn(forbidden, source, str(path))


if __name__ == "__main__":
    unittest.main()
