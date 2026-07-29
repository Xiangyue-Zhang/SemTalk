from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPOSITORY / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module(
    "build_base_features_lower_live_test",
    "scripts/show_base/build_base_features.py",
)
INFERENCE = _load_module(
    "run_base_inference_lower_live_test",
    "scripts/show_base/run_base_inference.py",
)


def _fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    source = {
        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
        "commit": "1" * 40,
        "tree": "2" * 40,
        "entrypoint": "/immutable/show_base_train.py",
        "entrypoint_sha256": "3" * 64,
    }
    smplx_sha = BUILDER.FORMAL_SMPLX_SHA256
    receipt: dict[str, object] = {
        "format": "semtalk_show_lower_target_backend_v1",
        "backend": "live_smplx",
        "formal_stage": "lower",
        "cache_enabled": False,
        "target_forward": "utils.smplx_training.smplx_target_forward",
        "target_forward_sha256": BUILDER.sha256(
            REPOSITORY / "utils" / "smplx_training.py"
        ),
        "target_batch_contract": "current_mixed_dataloader_batch",
        "torch_no_grad": True,
        "return_shaped": False,
        "source_binding": {
            key: source[key] for key in ("origin", "commit", "tree")
        },
        "smplx_asset_sha256": smplx_sha,
    }
    receipt["receipt_sha256"] = BUILDER.compact_payload_sha256(receipt)
    audit = {
        "source_receipt": source,
        "lower_target_backend": receipt,
    }
    dataset = {
        "smplx_asset": {
            "format": "semtalk_show_smplx_asset_v1",
            "filename": BUILDER.FORMAL_SMPLX_FILENAME,
            "sha256": smplx_sha,
        },
        "lower_target_backend": receipt,
    }
    status = {"lower_target_backend": receipt}
    return receipt, audit, dataset, status


class LowerLiveBackendReceiptTest(unittest.TestCase):
    def validate_both(
        self,
        *,
        formal_stage: str,
        audit: dict[str, object],
        dataset: dict[str, object],
        status: dict[str, object],
    ) -> None:
        path = Path("/frozen/rvq_lower_600.bin")
        BUILDER.validate_lower_target_backend_binding(
            formal_stage=formal_stage,
            audit=audit,
            dataset_receipt=dataset,
            status=status,
            path=path,
        )
        INFERENCE._validate_lower_target_backend_binding(
            formal_stage=formal_stage,
            audit=audit,
            dataset_receipt=dataset,
            status=status,
            path=path,
        )
        INFERENCE._validate_lower_target_backend_binding(
            formal_stage=formal_stage,
            audit=audit,
            dataset_receipt=dataset,
            status=None,
            path=path,
        )

    def assert_rejected_by_both(
        self,
        *,
        formal_stage: str,
        audit: dict[str, object],
        dataset: dict[str, object],
        status: dict[str, object],
    ) -> None:
        path = Path("/frozen/rvq_lower_600.bin")
        with self.assertRaises(RuntimeError):
            BUILDER.validate_lower_target_backend_binding(
                formal_stage=formal_stage,
                audit=audit,
                dataset_receipt=dataset,
                status=status,
                path=path,
            )
        with self.assertRaises(INFERENCE.InferenceContractError):
            INFERENCE._validate_lower_target_backend_binding(
                formal_stage=formal_stage,
                audit=audit,
                dataset_receipt=dataset,
                status=status,
                path=path,
            )

    def test_valid_live_backend_is_identical_and_strict_in_both_consumers(
        self,
    ) -> None:
        receipt, audit, dataset, status = _fixture()
        self.assertEqual(
            set(receipt),
            BUILDER.LOWER_TARGET_BACKEND_RECEIPT_KEYS,
        )
        self.assertEqual(
            BUILDER.LOWER_TARGET_BACKEND_RECEIPT_KEYS,
            INFERENCE.LOWER_TARGET_BACKEND_RECEIPT_KEYS,
        )
        self.validate_both(
            formal_stage="lower",
            audit=audit,
            dataset=dataset,
            status=status,
        )

    def test_missing_receipt_fails_closed(self) -> None:
        for location in ("audit", "dataset", "status"):
            with self.subTest(location=location):
                _, audit, dataset, status = _fixture()
                {
                    "audit": audit,
                    "dataset": dataset,
                    "status": status,
                }[location].pop("lower_target_backend")
                self.assert_rejected_by_both(
                    formal_stage="lower",
                    audit=audit,
                    dataset=dataset,
                    status=status,
                )

    def test_tampered_receipt_or_digest_fails_closed(self) -> None:
        for mutation in ("semantic", "digest"):
            with self.subTest(mutation=mutation):
                receipt, audit, dataset, status = _fixture()
                tampered = copy.deepcopy(receipt)
                if mutation == "semantic":
                    tampered["target_batch_contract"] = "static_dataset_index"
                else:
                    tampered["receipt_sha256"] = "0" * 64
                for payload in (audit, dataset, status):
                    payload["lower_target_backend"] = tampered
                self.assert_rejected_by_both(
                    formal_stage="lower",
                    audit=audit,
                    dataset=dataset,
                    status=status,
                )

    def test_coherently_tampered_smplx_sha_fails_closed(self) -> None:
        for invalid_sha in (None, "not-a-sha", "0" * 64):
            with self.subTest(invalid_sha=invalid_sha):
                receipt, audit, dataset, status = _fixture()
                tampered = copy.deepcopy(receipt)
                tampered["smplx_asset_sha256"] = invalid_sha
                tampered_without_sha = dict(tampered)
                tampered_without_sha.pop("receipt_sha256")
                tampered["receipt_sha256"] = (
                    BUILDER.compact_payload_sha256(tampered_without_sha)
                )
                for payload in (audit, dataset, status):
                    payload["lower_target_backend"] = tampered
                dataset["smplx_asset"]["sha256"] = invalid_sha
                self.assert_rejected_by_both(
                    formal_stage="lower",
                    audit=audit,
                    dataset=dataset,
                    status=status,
                )

    def test_cache_live_mix_is_rejected(self) -> None:
        _, audit, dataset, status = _fixture()
        for payload in (audit, dataset, status):
            payload["lower_target_joints_cache"] = {"forbidden": True}
        self.assert_rejected_by_both(
            formal_stage="lower",
            audit=audit,
            dataset=dataset,
            status=status,
        )

    def test_non_lower_carrying_live_receipt_is_rejected(self) -> None:
        _, audit, dataset, status = _fixture()
        self.assert_rejected_by_both(
            formal_stage="face",
            audit=audit,
            dataset=dataset,
            status=status,
        )

    def test_formal_launcher_has_no_cache_artifact_inputs(self) -> None:
        launcher = (
            REPOSITORY
            / "scripts"
            / "show_base"
            / "run_five_prerequisites.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("if [[ $# -ne 10 && $# -ne 11 ]]", launcher)
        self.assertNotIn("LOWER_TARGET_CACHE", launcher)
        self.assertNotIn("lower_target_manifest", launcher)
        self.assertNotIn("lower_target_checker", launcher)
        self.assertNotIn("lower_target_gate", launcher)
        self.assertNotIn("lower_target_builder_process", launcher)
        self.assertNotIn("--use_lower_target_joints_cache true", launcher)
        self.assertEqual(
            launcher.count("--use_lower_target_joints_cache false"),
            1,
        )

    def test_training_contract_threads_live_receipt_and_disables_cache(
        self,
    ) -> None:
        training = (REPOSITORY / "show_base_train.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"use_lower_target_joints_cache": True', training)
        self.assertIn(
            'LOWER_TARGET_BACKEND_RECEIPT_KEY = "lower_target_backend"',
            training,
        )
        self.assertIn("_verify_lower_target_backend_resume_receipt(", training)
        self.assertIn("_attach_lower_target_backend_receipt(", training)
        self.assertIn("_lower_target_backend_overlay(", training)
        self.assertIn(
            '"target_batch_contract": "current_mixed_dataloader_batch"',
            training,
        )


if __name__ == "__main__":
    unittest.main()
