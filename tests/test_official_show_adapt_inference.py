from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/show_base/run_base_inference.py"
SPEC = importlib.util.spec_from_file_location(
    "official_show_adapt_inference_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
INFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFERENCE)
TRANSFER_MODULE_PATH = ROOT / "scripts/show_base/train_official_transfer.py"
TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "official_show_transfer_under_test",
    TRANSFER_MODULE_PATH,
)
assert TRANSFER_SPEC is not None and TRANSFER_SPEC.loader is not None
TRANSFER = importlib.util.module_from_spec(TRANSFER_SPEC)
TRANSFER_SPEC.loader.exec_module(TRANSFER)


class _FakeTensorBytes:
    dtype = "torch.float32"
    shape = (1,)

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def detach(self) -> "_FakeTensorBytes":
        return self

    def cpu(self) -> "_FakeTensorBytes":
        return self

    def contiguous(self) -> "_FakeTensorBytes":
        return self

    def numpy(self) -> "_FakeTensorBytes":
        return self

    def tobytes(self, order: str) -> bytes:
        if order != "C":
            raise AssertionError(order)
        return self.payload


FACE_TRANSFER_STATE = {
    "encoder.weight": _FakeTensorBytes(b"encoder"),
    "quantizer.weight": _FakeTensorBytes(b"quantizer"),
    "decoder.weight": _FakeTensorBytes(b"decoder"),
}


def _argv() -> list[str]:
    sha = "a" * 64
    argv = [
        "--canonical-manifest",
        "/frozen/canonical.jsonl",
        "--canonical-summary-json",
        "/frozen/canonical-summary.json",
        "--canonical-lineage-json",
        "/frozen/canonical-lineage.json",
        "--expected-canonical-manifest-sha256",
        sha,
        "--audio-manifest",
        *[f"/frozen/audio-{index}.jsonl" for index in range(8)],
        "--audio-summary-json",
        *[f"/frozen/audio-{index}.summary.json" for index in range(8)],
        "--audio-lineage-json",
        *[f"/frozen/audio-{index}.lineage.json" for index in range(8)],
        "--prerequisite-source",
        INFERENCE.OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
        "--base-checkpoint-source",
        INFERENCE.OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
        "--expected-inference-script-sha256",
        sha,
        "--expected-source-commit",
        "1" * 40,
        "--expected-source-tree",
        "2" * 40,
        "--expected-base-training-source-commit",
        "3" * 40,
        "--expected-base-training-source-tree",
        "4" * 40,
        "--expected-transfer-training-source-commit",
        "9" * 40,
        "--expected-transfer-training-source-tree",
        "a" * 40,
        "--expected-input-source-commit",
        "5" * 40,
        "--expected-input-source-tree",
        "6" * 40,
        "--expected-canonical-source-commit",
        "7" * 40,
        "--expected-canonical-source-tree",
        "8" * 40,
        "--expected-hubert-tree-sha256",
        sha,
        "--base-checkpoint",
        "/adapt/candidates/base_official_adapt_epoch_08.bin",
        "--expected-base-sha256",
        sha,
        "--base-status-json",
        "/adapt/base/status.json",
        "--base-candidate-manifest",
        "/adapt/base/candidate_manifest.json",
        "--expected-base-candidate-manifest-sha256",
        sha,
        "--expected-base-formal-status-sha256",
        sha,
        "--base-frozen-inputs-json",
        "/adapt/base/frozen_inputs.json",
        "--expected-base-frozen-inputs-sha256",
        sha,
        "--face-checkpoint",
        "/adapt/face/best_face_transfer.bin",
        "--expected-face-sha256",
        sha,
        "--face-status-json",
        "/adapt/face/summary.json",
        "--expected-face-status-sha256",
        sha,
        "--global-checkpoint",
        "/adapt/global/best_global_transfer.bin",
        "--expected-global-sha256",
        sha,
        "--global-status-json",
        "/adapt/global/summary.json",
        "--expected-global-status-sha256",
        sha,
        "--output-root",
        "/output/official-adapt",
    ]
    for stage in ("hands", "upper", "lower"):
        specification = INFERENCE.RELEASED_ALL_SPEAKERS_MODELS[stage]
        argv.extend(
            [
                f"--{stage}-checkpoint",
                f"/official/{specification['filename']}",
                f"--expected-{stage}-sha256",
                specification["sha256"],
            ]
        )
    return argv


def _remove_pair(argv: list[str], option: str) -> None:
    index = argv.index(option)
    del argv[index : index + 2]


class OfficialShowAdaptCliTests(unittest.TestCase):
    def test_exact_mode_is_accepted_without_legacy_or_zero_shot_receipts(
        self,
    ) -> None:
        args = INFERENCE.parse_args(_argv())
        self.assertEqual(
            args.base_checkpoint_source,
            INFERENCE.OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
        )
        self.assertEqual(
            args.prerequisite_source,
            INFERENCE.OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE,
        )
        self.assertIsNone(args.base_training_lineage_manifest)
        self.assertIsNone(args.released_cross_domain_gate_json)
        self.assertIsNone(args.expected_base_final_checkpoint_sha256)
        self.assertIsNone(args.upper_status_json)
        self.assertIsNone(args.hands_status_json)
        self.assertIsNone(args.lower_status_json)
        self.assertIsNone(args.expected_training_source_commit)
        self.assertEqual(
            args.expected_base_training_source_commit,
            "3" * 40,
        )
        self.assertEqual(
            args.expected_transfer_training_source_commit,
            "9" * 40,
        )

    def test_every_new_external_receipt_root_is_required(self) -> None:
        for option in (
            "--base-status-json",
            "--expected-base-formal-status-sha256",
            "--base-candidate-manifest",
            "--expected-base-candidate-manifest-sha256",
            "--base-frozen-inputs-json",
            "--expected-base-frozen-inputs-sha256",
            "--face-status-json",
            "--expected-face-status-sha256",
            "--global-status-json",
            "--expected-global-status-sha256",
            "--expected-base-training-source-commit",
            "--expected-base-training-source-tree",
            "--expected-transfer-training-source-commit",
            "--expected-transfer-training-source-tree",
        ):
            with self.subTest(option=option):
                argv = _argv()
                _remove_pair(argv, option)
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    INFERENCE.parse_args(argv)

    def test_mixed_mode_and_legacy_receipts_are_rejected(self) -> None:
        argv = _argv()
        source_index = argv.index("--base-checkpoint-source")
        argv[source_index + 1] = INFERENCE.SHOW_TRAINED_CHECKPOINT_SOURCE
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            INFERENCE.parse_args(argv)

        argv = _argv() + [
            "--base-training-summary-json",
            "/legacy/base-summary.json",
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            INFERENCE.parse_args(argv)

        argv = _argv() + [
            "--expected-training-source-commit",
            "b" * 40,
            "--expected-training-source-tree",
            "c" * 40,
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            INFERENCE.parse_args(argv)

    def test_only_exact_official_hands_upper_lower_are_accepted(self) -> None:
        for stage in ("hands", "upper", "lower"):
            with self.subTest(stage=stage):
                argv = _argv()
                index = argv.index(f"--expected-{stage}-sha256")
                argv[index + 1] = "b" * 64
                with redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    INFERENCE.parse_args(argv)

    def test_withdrawn_e30_and_speaker2_paths_are_rejected(self) -> None:
        for marker in ("e30", "Speaker2"):
            with self.subTest(marker=marker):
                argv = _argv()
                index = argv.index("--base-checkpoint")
                argv[index + 1] = (
                    f"/adapt/{marker}/base_official_adapt_epoch_08.bin"
                )
                with redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    INFERENCE.parse_args(argv)


class OfficialShowAdaptReceiptTests(unittest.TestCase):
    def test_base_and_transfer_producer_sources_are_independent(self) -> None:
        producer_sources = (
            INFERENCE._official_adapt_producer_source_receipts(
                SimpleNamespace(
                    expected_base_training_source_commit="3" * 40,
                    expected_base_training_source_tree="4" * 40,
                    expected_transfer_training_source_commit="9" * 40,
                    expected_transfer_training_source_tree="a" * 40,
                )
            )
        )
        self.assertEqual(
            producer_sources["base_training"],
            {
                "format": (
                    "semtalk_show_base_training_source_expectation_v1"
                ),
                "origin": INFERENCE.EXPECTED_ORIGIN,
                "commit": "3" * 40,
                "tree": "4" * 40,
            },
        )
        self.assertEqual(
            producer_sources["transfer_training"],
            {
                "format": (
                    "semtalk_show_transfer_training_source_expectation_v1"
                ),
                "origin": INFERENCE.EXPECTED_ORIGIN,
                "commit": "9" * 40,
                "tree": "a" * 40,
            },
        )
        roles = INFERENCE._official_adapt_source_roles(
            inference_source={"commit": "1" * 40},
            producer_sources=producer_sources,
            input_source={"commit": "5" * 40},
            canonical_source={"commit": "7" * 40},
        )
        self.assertEqual(
            list(roles),
            [
                "inference",
                "base_training",
                "transfer_training",
                "input_artifact",
                "canonical",
            ],
        )
        self.assertEqual(roles["inference"], {"commit": "1" * 40})
        self.assertNotEqual(
            roles["inference"],
            roles["base_training"],
        )
        self.assertNotEqual(
            roles["base_training"],
            roles["transfer_training"],
        )

        with self.assertRaises(INFERENCE.InferenceContractError):
            INFERENCE._official_adapt_producer_source_receipts(
                SimpleNamespace(
                    expected_base_training_source_commit="3" * 40,
                    expected_base_training_source_tree="4" * 40,
                    expected_transfer_training_source_commit="9" * 40,
                    expected_transfer_training_source_tree=None,
                )
            )
        with self.assertRaises(INFERENCE.InferenceContractError):
            INFERENCE._official_adapt_source_roles(
                inference_source={},
                producer_sources={
                    "base_training": producer_sources["base_training"]
                },
                input_source={},
                canonical_source={},
            )

    def test_transfer_trainer_records_and_requires_detached_source(
        self,
    ) -> None:
        def git_result(
            command: list[str],
            *,
            attached: bool,
            **_: object,
        ) -> subprocess.CompletedProcess[str]:
            if "symbolic-ref" in command:
                return subprocess.CompletedProcess(
                    command,
                    0 if attached else 1,
                    "main\n" if attached else "",
                    "",
                )
            tail = command[3:]
            outputs = {
                ("remote", "get-url", "origin"): TRANSFER.EXPECTED_ORIGIN,
                ("rev-parse", "HEAD"): "1" * 40,
                ("rev-parse", "HEAD^{tree}"): "2" * 40,
                (
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ): "",
            }
            return subprocess.CompletedProcess(
                command,
                0,
                outputs[tuple(tail)] + "\n",
                "",
            )

        with mock.patch.object(
            TRANSFER.subprocess,
            "run",
            side_effect=lambda command, **kwargs: git_result(
                command,
                attached=False,
                **kwargs,
            ),
        ):
            receipt = TRANSFER._git_source_receipt("1" * 40, "2" * 40)
        self.assertIsNone(receipt["branch"])
        self.assertTrue(receipt["clean"])

        with (
            mock.patch.object(
                TRANSFER.subprocess,
                "run",
                side_effect=lambda command, **kwargs: git_result(
                    command,
                    attached=True,
                    **kwargs,
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "detached"),
        ):
            TRANSFER._git_source_receipt("1" * 40, "2" * 40)

    def test_base_receipt_hash_matches_trainer_for_unicode_paths(self) -> None:
        payload = {"path": "/frozen/中文/official", "epoch": 8}
        expected = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            INFERENCE.official_base_adapt_json_sha256(payload),
            expected,
        )
        self.assertNotEqual(
            INFERENCE.canonical_json_sha256(payload),
            expected,
        )

    def _face_transfer_payload(
        self,
        *,
        protocol_test_visible: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        source = {
            "origin": INFERENCE.EXPECTED_ORIGIN,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "branch": None,
            "clean": True,
            "entrypoint": "/producer/train_official_transfer.py",
            "entrypoint_sha256": "a" * 64,
        }
        cache = {
            "format": "semtalk_show_official_transfer_cache_v1",
            "stage": "face",
            "roots": ["/cache/face"],
            "canonical_receipt": {"canonical": "frozen"},
        }
        protocol = {
            "stage": "face",
            "selection_split": "val",
            "test_visible": protocol_test_visible,
            "trainable": "decoder only",
        }
        policy = {
            "trainable_parameter_names": ["decoder.block.weight"],
            "trainable_parameter_count": 1,
            "frozen_encoder": True,
            "frozen_quantizer": True,
            "frozen_rvq_ema": True,
        }
        transfer: dict[str, object] = {
            "format": INFERENCE.OFFICIAL_SHOW_ADAPT_TRANSFER_FORMAT,
            "stage": "face",
            "epoch": 2,
            "official_initialization": {"face": {}},
            "source_receipt": source,
            "cache_receipt": cache,
            "trainable_policy": policy,
            "protocol": protocol,
            "validation": {"total": 0.1},
            "model_state_sha256": (
                INFERENCE.official_transfer_state_dict_sha256(
                    FACE_TRANSFER_STATE
                )
            ),
            "frozen_encoder_quantizer_sha256": (
                INFERENCE.official_transfer_state_dict_sha256(
                    {
                        key: value
                        for key, value in FACE_TRANSFER_STATE.items()
                        if (
                            key.startswith("encoder.")
                            or key.startswith("quantizer.")
                        )
                    }
                )
            ),
            "withdrawn_e30_allowed": False,
            "test_visible": False,
        }
        transfer["receipt_sha256"] = INFERENCE.canonical_json_sha256(
            transfer
        )
        checkpoint: dict[str, object] = {
            "model_state": {},
            "optimizer_state": {},
            "epoch": 2,
            "transfer_receipt": transfer,
        }
        summary: dict[str, object] = {
            "format": INFERENCE.OFFICIAL_SHOW_ADAPT_TRANSFER_FORMAT,
            "status": "complete",
            "stage": "face",
            "epochs": 4,
            "best_epoch": 2,
            "best_validation_total": 0.1,
            "official_initialization": transfer["official_initialization"],
            "source_receipt": source,
            "cache_receipt": cache,
            "trainable_policy": policy,
            "protocol": protocol,
            "frozen_encoder_quantizer_sha256": transfer[
                "frozen_encoder_quantizer_sha256"
            ],
            "withdrawn_e30_allowed": False,
            "test_visible": False,
            "metrics_jsonl": "/metrics/face.jsonl",
            "metrics_jsonl_sha256": "d" * 64,
            "elapsed_seconds": 1.0,
        }
        summary["receipt_sha256"] = INFERENCE.canonical_json_sha256(
            summary
        )
        return checkpoint, summary

    def _validate_mock_face_transfer(
        self,
        checkpoint: dict[str, object],
        summary: dict[str, object],
    ) -> None:
        with (
            mock.patch.object(
                INFERENCE,
                "_read_verified_checkpoint_snapshot",
                return_value=(
                    Path("/adapt/face/best_face_transfer.bin"),
                    b"checkpoint",
                    "e" * 64,
                ),
            ),
            mock.patch.object(
                INFERENCE,
                "_torch_load_checkpoint",
                return_value=checkpoint,
            ),
            mock.patch.object(
                INFERENCE,
                "_normalize_data_parallel_state",
                return_value=FACE_TRANSFER_STATE,
            ),
            mock.patch.object(INFERENCE, "_finite_state_dict"),
            mock.patch.object(
                INFERENCE,
                "_validate_released_model_state_schema",
            ),
            mock.patch.object(
                INFERENCE,
                "_validate_adapt_producer_source",
                return_value={},
            ) as producer_validator,
            mock.patch.object(
                INFERENCE,
                "_validate_exact_official_reference",
                return_value={},
            ),
            mock.patch.object(
                INFERENCE,
                "_verified_json_object",
                return_value=(
                    Path("/adapt/face/summary.json"),
                    summary,
                    "f" * 64,
                ),
            ),
            mock.patch.object(
                INFERENCE,
                "_resolved_regular_file",
                return_value=Path("/metrics/face.jsonl"),
            ),
            mock.patch.object(
                INFERENCE,
                "sha256_file",
                return_value="d" * 64,
            ),
            mock.patch.object(
                INFERENCE,
                "load_jsonl",
                return_value=[
                    {
                        "epoch": epoch,
                        "train": {"total": float(epoch)},
                        "val": {
                            "total": (
                                0.1 if epoch == 2 else float(epoch)
                            )
                        },
                        "elapsed_seconds": float(epoch),
                        "rank_count": 1,
                        "finite": True,
                    }
                    for epoch in range(1, 5)
                ],
            ),
        ):
            INFERENCE._validate_official_transfer_checkpoint(
                path=Path("/adapt/face/best_face_transfer.bin"),
                formal_stage="face",
                expected_sha256="e" * 64,
                status_path=Path("/adapt/face/summary.json"),
                expected_status_sha256="f" * 64,
                expected_source_receipt={},
                expected_canonical_receipt={"canonical": "frozen"},
            )
            self.assertTrue(
                producer_validator.call_args.kwargs[
                    "require_detached_receipt"
                ]
            )

    def test_face_best_checkpoint_is_val_only_and_decoder_frozen(self) -> None:
        checkpoint, summary = self._face_transfer_payload()
        self._validate_mock_face_transfer(checkpoint, summary)

        leaked_checkpoint, leaked_summary = self._face_transfer_payload(
            protocol_test_visible=True
        )
        with self.assertRaises(INFERENCE.InferenceContractError):
            self._validate_mock_face_transfer(
                leaked_checkpoint,
                leaked_summary,
            )

        tampered_checkpoint, tampered_summary = self._face_transfer_payload()
        tampered_transfer = tampered_checkpoint["transfer_receipt"]
        assert isinstance(tampered_transfer, dict)
        tampered_transfer["model_state_sha256"] = "0" * 64
        tampered_transfer.pop("receipt_sha256")
        tampered_transfer["receipt_sha256"] = (
            INFERENCE.canonical_json_sha256(tampered_transfer)
        )
        with self.assertRaises(INFERENCE.InferenceContractError):
            self._validate_mock_face_transfer(
                tampered_checkpoint,
                tampered_summary,
            )

    def test_transfer_selection_is_recomputed_from_validation_jsonl(
        self,
    ) -> None:
        rows = [
            {
                "epoch": epoch,
                "train": {"total": float(epoch)},
                "val": {"total": total},
                "elapsed_seconds": float(epoch),
                "rank_count": 1,
                "finite": True,
            }
            for epoch, total in ((1, 0.1), (2, 0.2))
        ]
        with mock.patch.object(INFERENCE, "load_jsonl", return_value=rows):
            INFERENCE._validate_official_transfer_metrics(
                path=Path("/metrics/face.jsonl"),
                summary={
                    "epochs": 2,
                    "best_epoch": 1,
                    "best_validation_total": 0.1,
                },
                transfer={"validation": {"total": 0.1}},
                stage="face",
            )
            with self.assertRaises(INFERENCE.InferenceContractError):
                INFERENCE._validate_official_transfer_metrics(
                    path=Path("/metrics/face.jsonl"),
                    summary={
                        "epochs": 2,
                        "best_epoch": 2,
                        "best_validation_total": 0.2,
                    },
                    transfer={"validation": {"total": 0.2}},
                    stage="face",
                )

    def test_current_inference_source_must_be_detached(self) -> None:
        attached = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="main\n",
            stderr="",
        )
        with (
            mock.patch.object(
                INFERENCE.subprocess,
                "run",
                return_value=attached,
            ),
            self.assertRaises(INFERENCE.InferenceContractError),
        ):
            INFERENCE._require_official_adapt_detached_source(ROOT)

        detached = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="",
        )
        with mock.patch.object(
            INFERENCE.subprocess,
            "run",
            return_value=detached,
        ):
            INFERENCE._require_official_adapt_detached_source(ROOT)

    def test_base_producer_receipt_requires_clean_detached_exact_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entrypoint = Path(temporary) / "trainer.py"
            entrypoint.write_text("print('frozen')\n", encoding="utf-8")
            expected = {
                "origin": INFERENCE.EXPECTED_ORIGIN,
                "commit": "1" * 40,
                "tree": "2" * 40,
            }
            receipt = {
                **expected,
                "branch": None,
                "clean": True,
                "entrypoint": str(entrypoint),
                "entrypoint_sha256": INFERENCE.sha256_file(entrypoint),
            }
            observed = INFERENCE._validate_adapt_producer_source(
                receipt,
                expected_source_receipt=expected,
                label="fixture",
                require_detached_receipt=True,
            )
            self.assertEqual(observed, receipt)
            transfer_expected = {
                "origin": INFERENCE.EXPECTED_ORIGIN,
                "commit": "9" * 40,
                "tree": "a" * 40,
            }
            with self.assertRaises(INFERENCE.InferenceContractError):
                INFERENCE._validate_adapt_producer_source(
                    receipt,
                    expected_source_receipt=transfer_expected,
                    label="Base receipt checked as transfer producer",
                    require_detached_receipt=True,
                )
            for key, value in (
                ("clean", False),
                ("branch", "main"),
                ("commit", "3" * 40),
            ):
                tampered = dict(receipt)
                tampered[key] = value
                with self.subTest(key=key), self.assertRaises(
                    INFERENCE.InferenceContractError
                ):
                    INFERENCE._validate_adapt_producer_source(
                        tampered,
                        expected_source_receipt=expected,
                        label="fixture",
                        require_detached_receipt=True,
                    )

    def test_locked_runtime_interface_uses_predicted_lower_codes(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        start = source.index("def _infer_clip(")
        end = source.index("\ndef _output_arrays(", start)
        implementation = source[start:end]
        self.assertIn("stitched_indices[name]", implementation)
        self.assertIn('indices["lower"]', implementation)
        self.assertIn('models["lower"]', implementation)
        self.assertIn('to_global = decoded["lower"].clone()', implementation)
        self.assertIn('models["global"](to_global)', implementation)
        self.assertNotIn("canonical[\"lower\"]", implementation)

    def test_contract_is_one_shot_full_show_diffsheg_seven_metrics(
        self,
    ) -> None:
        inputs = {
            "canonical_manifest": Path("/frozen/canonical.jsonl"),
            "canonical_manifest_sha256": "a" * 64,
            "canonical_summary_sha256": "b" * 64,
            "canonical_lineage_sha256": "c" * 64,
            "audio_manifest_sha256": ["d" * 64],
            "audio_summary_sha256": ["e" * 64],
            "audio_lineage_sha256": ["f" * 64],
            "training_lineage_manifest_sha256": {},
            "base_training_summary_sha256": None,
            "accepted_training_lineages": {},
            "source": {},
            "source_roles": {},
            "base_checkpoint_source": (
                INFERENCE.OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
            ),
            "prerequisite_source": (
                INFERENCE.OFFICIAL_SHOW_ADAPT_CHECKPOINT_SOURCE
            ),
            "released_prerequisite_import_receipt": None,
            "inference_mode": INFERENCE.OFFICIAL_SHOW_ADAPT_MODE,
        }
        receipt = INFERENCE._stable_contract_receipt(
            SimpleNamespace(seed=1001),
            inputs,
            {},
        )
        boundary = receipt["diffsheg_evaluation_boundary"]
        self.assertEqual(
            boundary["metrics"],
            [
                "FMD",
                "FED",
                "expression_diversity",
                "FGD",
                "BA",
                "PCM",
                "gesture_diversity",
            ],
        )
        self.assertEqual(
            boundary["input"],
            "full canonical SHOW NPZ; never a body-only projection",
        )
        self.assertEqual(
            receipt["official_show_adapt_interface"]["test_evaluations"],
            1,
        )
        self.assertFalse(
            receipt["official_show_adapt_interface"][
                "test_feedback_into_selection"
            ]
        )


if __name__ == "__main__":
    unittest.main()
