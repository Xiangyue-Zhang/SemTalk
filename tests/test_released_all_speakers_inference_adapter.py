from __future__ import annotations

import copy
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - minimal local environments
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/show_base/run_base_inference.py"
SPEC = importlib.util.spec_from_file_location(
    "released_all_speakers_inference_adapter_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
INFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFERENCE)


def _default_argv() -> list[str]:
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
        "--base-training-lineage-manifest",
        "/frozen/base-lineage.json",
        "--base-training-summary-json",
        "/frozen/base-summary.json",
        "--representation-training-lineage-manifest",
        "/frozen/representation-lineage.json",
        "--expected-inference-script-sha256",
        sha,
        "--expected-source-commit",
        "1" * 40,
        "--expected-source-tree",
        "2" * 40,
        "--expected-canonical-source-commit",
        "3" * 40,
        "--expected-canonical-source-tree",
        "4" * 40,
        "--expected-hubert-tree-sha256",
        sha,
    ]
    for stage in INFERENCE.CHECKPOINT_STAGES:
        argv.extend(
            [
                f"--{stage}-checkpoint",
                f"/frozen/{stage}.bin",
                f"--expected-{stage}-sha256",
                sha,
                f"--{stage}-status-json",
                f"/frozen/{stage}-status.json",
            ]
        )
    argv.extend(
        [
            "--base-candidate-manifest",
            "/frozen/base_candidate_manifest.json",
            "--expected-base-candidate-manifest-sha256",
            sha,
            "--expected-base-formal-status-sha256",
            sha,
            "--expected-base-final-checkpoint-sha256",
            sha,
            "--output-root",
            "/output",
        ]
    )
    return argv


def _remove_option(argv: list[str], option: str) -> None:
    index = argv.index(option)
    del argv[index : index + 2]


def _released_argv(*, released_base: bool = False) -> list[str]:
    argv = _default_argv()
    argv.extend(
        [
            "--prerequisite-source",
            INFERENCE.RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
            "--expected-training-source-commit",
            "5" * 40,
            "--expected-training-source-tree",
            "6" * 40,
            "--expected-input-source-commit",
            "7" * 40,
            "--expected-input-source-tree",
            "8" * 40,
        ]
    )
    for stage in set(INFERENCE.CHECKPOINT_STAGES) - {"base"}:
        _remove_option(argv, f"--{stage}-status-json")
        specification = INFERENCE.RELEASED_ALL_SPEAKERS_MODELS[stage]
        checkpoint_index = argv.index(f"--{stage}-checkpoint")
        argv[checkpoint_index + 1] = (
            f"/official/{specification['filename']}"
        )
        sha_index = argv.index(f"--expected-{stage}-sha256")
        argv[sha_index + 1] = specification["sha256"]
    if released_base:
        argv.extend(
            [
                "--base-checkpoint-source",
                INFERENCE.RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
            ]
        )
        for option in (
            "--base-status-json",
            "--base-candidate-manifest",
            "--expected-base-candidate-manifest-sha256",
            "--expected-base-formal-status-sha256",
            "--expected-base-final-checkpoint-sha256",
        ):
            _remove_option(argv, option)
        specification = INFERENCE.RELEASED_ALL_SPEAKERS_MODELS["base"]
        checkpoint_index = argv.index("--base-checkpoint")
        argv[checkpoint_index + 1] = (
            f"/official/{specification['filename']}"
        )
        sha_index = argv.index("--expected-base-sha256")
        argv[sha_index + 1] = specification["sha256"]
    return argv


class ReleasedCliContractTests(unittest.TestCase):
    def test_default_show_trained_contract_is_unchanged(self) -> None:
        args = INFERENCE.parse_args(_default_argv())
        self.assertEqual(
            args.prerequisite_source,
            INFERENCE.SHOW_TRAINED_CHECKPOINT_SOURCE,
        )
        self.assertEqual(
            args.base_checkpoint_source,
            INFERENCE.SHOW_TRAINED_CHECKPOINT_SOURCE,
        )
        self.assertEqual(
            args.expected_training_source_commit,
            args.expected_source_commit,
        )
        self.assertEqual(
            args.expected_input_source_commit,
            args.expected_source_commit,
        )

    def test_release_representation_and_release_base_are_explicit(self) -> None:
        representation_only = INFERENCE.parse_args(_released_argv())
        self.assertEqual(
            representation_only.prerequisite_source,
            INFERENCE.RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
        )
        self.assertEqual(
            representation_only.base_checkpoint_source,
            INFERENCE.SHOW_TRAINED_CHECKPOINT_SOURCE,
        )
        all_released = INFERENCE.parse_args(
            _released_argv(released_base=True)
        )
        self.assertEqual(
            all_released.base_checkpoint_source,
            INFERENCE.RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE,
        )
        self.assertIsNone(all_released.base_status_json)
        self.assertIsNone(all_released.base_candidate_manifest)

    def test_release_sources_are_fail_closed_and_independent(self) -> None:
        argv = _released_argv()
        self.assertNotEqual(
            INFERENCE.parse_args(argv).expected_source_commit,
            INFERENCE.parse_args(argv).expected_training_source_commit,
        )
        for option in (
            "--expected-training-source-commit",
            "--expected-training-source-tree",
            "--expected-input-source-commit",
            "--expected-input-source-tree",
        ):
            with self.subTest(option=option):
                incomplete = list(argv)
                _remove_option(incomplete, option)
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    INFERENCE.parse_args(incomplete)
                self.assertEqual(raised.exception.code, 2)

    def test_release_forbids_show_status_and_hash_substitution(self) -> None:
        with_status = _released_argv()
        with_status.extend(
            ["--face-status-json", "/attacker/status.json"]
        )
        with (
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            INFERENCE.parse_args(with_status)
        wrong_hash = _released_argv()
        index = wrong_hash.index("--expected-face-sha256")
        wrong_hash[index + 1] = "0" * 64
        with (
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            INFERENCE.parse_args(wrong_hash)

    def test_release_launcher_owns_all_checkpoint_trust_roots(self) -> None:
        launcher = (
            ROOT
            / "scripts/show_base/"
            "run_base_released_all_speakers_inference.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "--prerequisite-source released_all_speakers_v1",
            launcher,
        )
        self.assertIn(
            "--base-checkpoint-source released_all_speakers_v1",
            launcher,
        )
        for stage, specification in (
            INFERENCE.RELEASED_ALL_SPEAKERS_MODELS.items()
        ):
            with self.subTest(stage=stage):
                self.assertIn(specification["filename"], launcher)
                self.assertIn(specification["sha256"], launcher)
                self.assertIn(
                    f"--expected-{stage}-sha256",
                    launcher,
                )


@unittest.skipIf(torch is None, "torch is unavailable")
class ReleasedCheckpointPrimitiveTests(unittest.TestCase):
    def _write(
        self,
        root: Path,
        *,
        payload: dict | None = None,
        filename: str = "tiny.bin",
    ) -> tuple[Path, str]:
        path = root / filename
        torch.save(
            payload
            if payload is not None
            else {"model_state": {"module.weight": torch.ones(1)}},
            path,
        )
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_weights_only_tensor_finite_normalized_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write(Path(temporary))
            state, resolved, snapshot, observed = (
                INFERENCE._load_released_model_state_only(
                    path,
                    expected_filename=path.name,
                    expected_sha256=digest,
                )
            )
            self.assertEqual(set(state), {"weight"})
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(observed, digest)
            self.assertEqual(
                hashlib.sha256(snapshot).hexdigest(),
                digest,
            )

    def test_rejects_symlink_extra_envelope_non_tensor_and_nonfinite(self) -> None:
        cases = (
            (
                {
                    "model_state": {"weight": torch.ones(1)},
                    "audit": {},
                },
                "only model_state",
            ),
            ({"model_state": {"weight": 1}}, "must be a tensor"),
            (
                {"model_state": {"weight": torch.tensor([float("nan")])}},
                "non-finite",
            ),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    path, digest = self._write(
                        Path(temporary),
                        payload=payload,
                    )
                    with self.assertRaisesRegex(
                        INFERENCE.InferenceContractError,
                        message,
                    ):
                        INFERENCE._load_released_model_state_only(
                            path,
                            expected_filename=path.name,
                            expected_sha256=digest,
                        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, digest = self._write(root, filename="target.bin")
            link = root / "tiny.bin"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                INFERENCE.InferenceContractError,
                "non-symlink",
            ):
                INFERENCE._load_released_model_state_only(
                    link,
                    expected_filename=link.name,
                    expected_sha256=digest,
                )

    def test_no_unsafe_torch_load_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write(Path(temporary))
            with (
                mock.patch.object(
                    torch,
                    "load",
                    side_effect=TypeError("no weights_only"),
                ),
                self.assertRaisesRegex(
                    INFERENCE.InferenceContractError,
                    "weights_only=True",
                ),
            ):
                INFERENCE._load_released_model_state_only(
                    path,
                    expected_filename=path.name,
                    expected_sha256=digest,
                )

    def test_official_base_envelope_is_exact_and_finite(self) -> None:
        optimizer_state = {
            index: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
            for index in range(1_655)
        }
        group = {
            "lr": 0.00030000000000000003,
            "betas": (0.5, 0.999),
            "eps": 1e-08,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
            "initial_lr": 0.00030000000000000003,
            "params": list(range(1_783)),
        }
        payload = {
            "model_state": {"module.weight": torch.ones(1)},
            "epoch": 401,
            "opt_state": {
                "state": optimizer_state,
                "param_groups": [group],
            },
            "lrs": dict(INFERENCE.RELEASED_BASE_LRS),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write(
                Path(temporary),
                filename="best_semtalk_base.bin",
                payload=payload,
            )
            state, resolved, snapshot, observed, auxiliary = (
                INFERENCE._load_released_base_state(
                    path,
                    expected_filename=path.name,
                    expected_sha256=digest,
                )
            )
            self.assertEqual(set(state), {"weight"})
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(hashlib.sha256(snapshot).hexdigest(), observed)
            self.assertEqual(auxiliary["epoch_counter"], 401)
            self.assertEqual(
                auxiliary["optimizer_state_entries"],
                1_655,
            )
            payload["epoch"] = 400
            tampered, tampered_digest = self._write(
                Path(temporary),
                filename="tampered.bin",
                payload=payload,
            )
            with self.assertRaisesRegex(
                INFERENCE.InferenceContractError,
                "envelope",
            ):
                INFERENCE._load_released_base_state(
                    tampered,
                    expected_filename=tampered.name,
                    expected_sha256=tampered_digest,
                )


class ReleasedLineageReceiptTests(unittest.TestCase):
    def _fixture(self, root: Path):
        specifications = {}
        records = {}
        files = {}
        source = INFERENCE._released_weight_source_receipt()
        source_sha = INFERENCE.canonical_json_sha256(source)
        args = {}
        for stage in set(INFERENCE.CHECKPOINT_STAGES) - {"base"}:
            path = root / f"{stage}.bin"
            path.write_bytes(stage.encode("ascii"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            specification = {
                "filename": path.name,
                "sha256": digest,
                "model_class": f"Model{stage}",
            }
            specifications[stage] = specification
            schema_sha = hashlib.sha256(
                f"schema:{stage}".encode("ascii")
            ).hexdigest()
            record = {
                "path": str(path.resolve()),
                "filename": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "formal_stage": stage,
                "prerequisite_source": (
                    INFERENCE.RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
                ),
                "classification": (
                    INFERENCE.RELEASED_ALL_SPEAKERS_CLASSIFICATION
                ),
                "training_dataset": "BEAT2",
                "speaker_scope": "All-Speakers",
                "show_trained": False,
                "checkpoint_container_schema": ["model_state"],
                "model_class": specification["model_class"],
                "model_state_tensors": 1,
                "model_state_schema_sha256": schema_sha,
                "all_model_state_tensors_finite": True,
                "strict_state_dict_load": True,
                "frozen_eval": True,
                "source_receipt": source,
                "source_receipt_sha256": source_sha,
            }
            records[stage] = record
            files[stage] = {
                key: record[key]
                for key in (
                    "path",
                    "filename",
                    "sha256",
                    "bytes",
                    "model_class",
                    "model_state_schema_sha256",
                )
            }
            args[f"{stage}_checkpoint"] = path
        receipt = {
            "format": (
                "semtalk_released_all_speakers_import_receipt_v1"
            ),
            "status": "complete",
            "prerequisite_source": (
                INFERENCE.RELEASED_ALL_SPEAKERS_CHECKPOINT_SOURCE
            ),
            "classification": (
                INFERENCE.RELEASED_ALL_SPEAKERS_CLASSIFICATION
            ),
            "training_dataset": "BEAT2",
            "speaker_scope": "All-Speakers",
            "show_trained": False,
            "source_receipt": source,
            "source_receipt_sha256": source_sha,
            "files": files,
            "strict_state_dict_load": True,
            "all_model_state_tensors_finite": True,
            "frozen_eval": True,
        }
        receipt["receipt_sha256"] = INFERENCE.canonical_json_sha256(
            receipt
        )
        lineage = {
            "formal_checkpoints": records,
            "prerequisite_source_receipt": receipt,
        }
        return specifications, SimpleNamespace(**args), lineage

    def test_exact_builder_receipt_is_accepted_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            specifications, args, lineage = self._fixture(
                Path(temporary)
            )
            model_specifications = {
                "base": INFERENCE.RELEASED_ALL_SPEAKERS_MODELS["base"],
                **specifications,
            }
            with mock.patch.object(
                INFERENCE,
                "RELEASED_ALL_SPEAKERS_MODELS",
                model_specifications,
            ):
                receipt, records = (
                    INFERENCE._validate_released_prerequisite_lineage_binding(
                        lineage=lineage,
                        args=args,
                    )
                )
                self.assertEqual(
                    receipt["receipt_sha256"],
                    lineage["prerequisite_source_receipt"][
                        "receipt_sha256"
                    ],
                )
                self.assertEqual(
                    set(records),
                    set(INFERENCE.CHECKPOINT_STAGES) - {"base"},
                )
                self.assertEqual(
                    receipt["source_receipt"]["release_trust_root"],
                    INFERENCE.RELEASED_ALL_SPEAKERS_RELEASE_TRUST_ROOT,
                )

                coherently_tampered = copy.deepcopy(lineage)
                tampered_source = copy.deepcopy(
                    coherently_tampered[
                        "prerequisite_source_receipt"
                    ]["source_receipt"]
                )
                tampered_source["release_trust_root"]["commit"] = "0" * 40
                tampered_source_sha = INFERENCE.canonical_json_sha256(
                    tampered_source
                )
                for record in coherently_tampered[
                    "formal_checkpoints"
                ].values():
                    record["source_receipt"] = tampered_source
                    record["source_receipt_sha256"] = tampered_source_sha
                tampered_receipt = coherently_tampered[
                    "prerequisite_source_receipt"
                ]
                tampered_receipt["source_receipt"] = tampered_source
                tampered_receipt[
                    "source_receipt_sha256"
                ] = tampered_source_sha
                tampered_receipt.pop("receipt_sha256")
                tampered_receipt[
                    "receipt_sha256"
                ] = INFERENCE.canonical_json_sha256(tampered_receipt)
                with self.assertRaises(
                    INFERENCE.InferenceContractError
                ):
                    INFERENCE._validate_released_prerequisite_lineage_binding(
                        lineage=coherently_tampered,
                        args=args,
                    )

                lineage["formal_checkpoints"]["face"][
                    "classification"
                ] = "SHOW-trained"
                with self.assertRaises(
                    INFERENCE.InferenceContractError
                ):
                    INFERENCE._validate_released_prerequisite_lineage_binding(
                        lineage=lineage,
                        args=args,
                    )


if __name__ == "__main__":
    unittest.main()
