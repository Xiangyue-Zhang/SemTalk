from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


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
        self.assertIn("_prime_pinned_released_schema_cache", source)
        self.assertIn("_validate_base_model_state_schema", source)

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
            with PRODUCER._pinned_meta_schema_cuda_compat():
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

        fake_torch = type("FakeTorch", (), {"Tensor": FakeTensor})()
        original_cuda = FakeTensor.cuda
        expected_stages = {"face", "global", "hands", "upper", "lower"}

        class FakeHelper:
            @staticmethod
            def _expected_released_representation_schemas() -> dict[
                str,
                dict[str, tuple[str, tuple[int, ...]]],
            ]:
                tensor = FakeTensor()
                if tensor.cuda() is not tensor:
                    raise AssertionError("meta shim did not preserve tensor")
                return {
                    stage: {"weight": ("float32", (1,))}
                    for stage in expected_stages
                }

        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            PRODUCER._prime_pinned_released_schema_cache(FakeHelper())
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
