from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/show_base/seal_short_quality_campaign.py"
SPEC = importlib.util.spec_from_file_location("short_quality_hardened", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sealmod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sealmod)


def write_receipt(path: Path, body: dict) -> tuple[dict, dict]:
    payload = dict(body)
    payload["receipt_payload_sha256"] = sealmod._payload_sha(
        payload, "receipt_payload_sha256"
    )
    raw = sealmod._pretty_json_bytes(payload)
    path.write_bytes(raw)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "receipt_payload_sha256": payload["receipt_payload_sha256"],
    }, payload


def fixture_contracts(root: Path):
    runtime = {
        "path": "/formal/python",
        "realpath": "/formal/python-real",
        "execution_path": "/formal/python-real",
        "real_sha256": "1" * 64,
        "version": [3, 11, 9],
        "isolated_mode": True,
    }
    source = {
        "origin": sealmod.EXPECTED_ORIGIN,
        "source_root": "/official/source",
        "commit": sealmod.EXPECTED_SOURCE_COMMIT,
        "tree": sealmod.EXPECTED_SOURCE_TREE,
        "clean": True,
        "detached": True,
        "local_branches_at_commit": [],
        "producer": {
            "path": "/official/source/" + sealmod.PRODUCER_RELATIVE,
            "sha256": sealmod.PRODUCER_SHA256,
            "bytes": 123,
        },
        "val_adapter": {
            "path": "/official/source/" + sealmod.ADAPTER_RELATIVE,
            "sha256": "2" * 64,
            "bytes": 456,
        },
    }
    preflight_artifact = {
        "path": str(root / "preflight.json"),
        "sha256": "3" * 64,
        "bytes": 789,
        "receipt_payload_sha256": "4" * 64,
    }
    preflight = {
        "adapter_source": dict(source),
        "pipeline_source": dict(source),
    }
    closure = {
        "topology_mode": "fixture-mode",
        "quality_gate_spec": {"sha256": "5" * 64},
        "topology_gate_spec": {"sha256": "6" * 64},
        "candidate_epochs": list(sealmod.EPOCHS),
        "split": "val",
        "test_visible": False,
    }
    arguments = argparse.Namespace(
        source_root=Path("/official/source"),
        expected_sealer_commit="a" * 40,
        expected_sealer_blob_oid="b" * 40,
        expected_sealer_file_sha256="c" * 64,
        preflight=root / "preflight.json",
        expected_preflight_sha256="3" * 64,
        expected_preflight_bytes=789,
        expected_preflight_payload_sha256="4" * 64,
        validation_root=root / "validation",
        campaign_root=root / "campaign",
    )
    return runtime, source, preflight_artifact, preflight, closure, arguments


def make_integrated_sealer_repository(root: Path) -> tuple[Path, str, str, str]:
    repository = root / "integrated-sealer"
    repository.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Xiangyue-Zhang")
    git(
        "config",
        "user.email",
        "85532891+Xiangyue-Zhang@users.noreply.github.com",
    )
    git("remote", "add", "origin", sealmod.EXPECTED_ORIGIN)
    sealer = repository / sealmod.SEALER_RELATIVE
    sealer.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, sealer)
    git("add", "--", sealmod.SEALER_RELATIVE)
    git("commit", "--quiet", "-m", "Integrate formal sealer")
    commit = git("rev-parse", "HEAD")
    blob = git("rev-parse", f"HEAD:{sealmod.SEALER_RELATIVE}")
    file_sha256 = hashlib.sha256(sealer.read_bytes()).hexdigest()
    return repository, commit, blob, file_sha256


def publish_fake_official_outputs(
    _runtime,
    _source,
    _common,
    quality_path: Path,
    short_path: Path,
):
    short_body = {
        "format": "fixture-short",
        "status": "complete",
        "split": "val",
        "test_visible": False,
    }
    short_body["receipt_payload_sha256"] = sealmod._payload_sha(
        short_body, "receipt_payload_sha256"
    )
    short_raw = sealmod._pretty_json_bytes(short_body)
    short_path.write_bytes(short_raw)
    short_artifact = {
        "path": str(short_path),
        "sha256": hashlib.sha256(short_raw).hexdigest(),
        "bytes": len(short_raw),
        "receipt_payload_sha256": short_body["receipt_payload_sha256"],
    }
    quality_body = {
        "format": "fixture-quality",
        "status": "pass",
        "split": "val",
        "test_visible": False,
        "short_trajectory_receipt": short_artifact,
    }
    quality_body["receipt_sha256"] = sealmod._payload_sha(
        quality_body, "receipt_sha256"
    )
    quality_path.write_bytes(sealmod._pretty_json_bytes(quality_body))
    return {
        "argv_sha256": "7" * 64,
        "return_code": 0,
        "stdout_sha256": "8" * 64,
        "stderr_sha256": "9" * 64,
    }


def real_schema_closure_fixture(root: Path) -> tuple[dict, Path, str]:
    mode = "validation_gated_w8_l256_g2048_empirical_acceleration"
    validation_root = root / "validation"
    validation_root.mkdir()
    ready = []
    for epoch in sealmod.EPOCHS:
        artifact, _payload = write_receipt(
            root / f"ready-{epoch}.json",
            {"epoch": epoch},
        )
        ready.append(artifact)
        candidate = validation_root / "candidates" / f"e{epoch}"
        (candidate / "final").mkdir(parents=True)
        write_receipt(
            candidate / "final" / "val-inference-lineage.json",
            {"epoch": epoch, "split": "val", "test_visible": False},
        )
        (candidate / "diffsheg-val-fgd.json").write_bytes(
            sealmod._pretty_json_bytes(
                {
                    "status": "ok",
                    "protocol": {
                        "selection_split": "val",
                        "test_visible": False,
                    },
                    "metrics": {"fgd": float(epoch)},
                }
            )
        )
    throughput = {"topology_mode": mode, "status": "pass"}
    short_status, _status_payload = write_receipt(
        root / "short-quality-status.json",
        {"status": "complete", "throughput_gate": throughput},
    )
    val_inputs_full, _val_payload = write_receipt(
        root / "val-inputs.json",
        {"status": "complete", "split": "val", "test_visible": False},
    )
    pipeline_full, _pipeline_payload = write_receipt(
        root / "pipeline.json",
        {"status": "complete", "split": "val", "test_visible": False},
    )
    val_inputs = {
        key: value for key, value in val_inputs_full.items() if key != "bytes"
    }
    pipeline = {
        key: value for key, value in pipeline_full.items() if key != "bytes"
    }
    topology_path = (
        REPOSITORY
        / "configs/show_base/semtalk_base_topology_gate_spec_20260731.json"
    ).resolve(strict=True)
    quality_path = (
        REPOSITORY
        / "configs/show_base/semtalk_base_topology_quality_gate_spec_v4_20260801.json"
    ).resolve(strict=True)
    topology_payload = json.loads(topology_path.read_text(encoding="utf-8"))
    preflight = {
        "status": "complete",
        "split": "val",
        "test_visible": False,
        "candidate_epochs": list(sealmod.EPOCHS),
        "coverage": {"clip_count": 1715},
        "candidate_bundle": {
            "updates_per_epoch": topology_payload["topology_matrix"][mode][
                "updates_per_epoch"
            ]
        },
        "val_inputs_receipt": val_inputs,
        "pipeline_receipt": pipeline,
        "short_quality_authority": {
            "topology_mode": mode,
            "topology_gate_spec": {
                "path": str(topology_path),
                "sha256": hashlib.sha256(topology_path.read_bytes()).hexdigest(),
            },
            "quality_gate_spec": {
                "path": str(quality_path),
                "sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
            },
            "candidate_ready_receipts": ready,
            "short_quality_status": short_status,
            "throughput_gate": throughput,
        },
    }
    return preflight, validation_root, mode


class HardenedSealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.official_temporary = tempfile.TemporaryDirectory()
        cls.official_repository = (
            Path(cls.official_temporary.name).resolve(strict=True)
            / "official-4066f20"
        )
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
                sealmod.EXPECTED_SOURCE_COMMIT,
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
                sealmod.EXPECTED_ORIGIN,
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.official_temporary.cleanup()

    def common_patches(self, root: Path):
        runtime, source, preflight_artifact, preflight, closure, args = (
            fixture_contracts(root)
        )
        patches = (
            mock.patch.object(sealmod, "verify_runtime", return_value=runtime),
            mock.patch.object(sealmod, "verify_source", return_value=source),
            mock.patch.object(
                sealmod,
                "_strict_pinned_json",
                return_value=(preflight_artifact, preflight),
            ),
            mock.patch.object(sealmod, "_require_source_binding"),
            mock.patch.object(sealmod, "replay_preflight", return_value={}),
            mock.patch.object(
                sealmod,
                "build_input_closure",
                return_value=(closure, ["--fixture"]),
            ),
            mock.patch.object(
                sealmod,
                "replay_outputs",
                side_effect=lambda _r, _s, mode, _p, digest, _q, _t: {
                    "mode": mode,
                    "candidate_fgd": {"1": 1.0},
                    "report_sha256": digest,
                    "test_visible": False,
                    "split": "val",
                },
            ),
            mock.patch.object(
                sealmod, "CLAIM_REGISTRY", root / "global-claims"
            ),
            mock.patch.object(
                sealmod,
                "verify_sealer_authority",
                return_value={
                    "origin": sealmod.EXPECTED_ORIGIN,
                    "repository_root": "/integrated/source",
                    "path": "/integrated/source/" + sealmod.SEALER_RELATIVE,
                    "relative_path": sealmod.SEALER_RELATIVE,
                    "commit": "a" * 40,
                    "tree": "d" * 40,
                    "git_blob_oid": "b" * 40,
                    "file_sha256": "c" * 64,
                    "bytes": 123,
                    "tracked": True,
                    "index_stage": 0,
                    "git_mode": "100644",
                    "clean_at_path": True,
                    "launcher_pinned": True,
                },
            ),
        )
        return runtime, source, closure, args, patches

    def test_exact_official_source_is_accepted(self):
        source = sealmod.verify_source(self.official_repository)
        self.assertEqual(source["commit"], sealmod.EXPECTED_SOURCE_COMMIT)
        self.assertEqual(source["tree"], sealmod.EXPECTED_SOURCE_TREE)
        self.assertEqual(
            source["producer"]["sha256"], sealmod.PRODUCER_SHA256
        )

    def test_integrated_hash_pinned_sealer_is_accepted_and_fully_recorded(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            repository, commit, blob, file_sha256 = (
                make_integrated_sealer_repository(root)
            )
            authority = sealmod.verify_sealer_authority(
                commit,
                blob,
                file_sha256,
                script_path=repository / sealmod.SEALER_RELATIVE,
            )
            self.assertEqual(authority["path"], str(repository / sealmod.SEALER_RELATIVE))
            self.assertEqual(authority["relative_path"], sealmod.SEALER_RELATIVE)
            self.assertEqual(authority["commit"], commit)
            self.assertEqual(authority["git_blob_oid"], blob)
            self.assertEqual(authority["file_sha256"], file_sha256)
            self.assertTrue(authority["tracked"])
            self.assertTrue(authority["launcher_pinned"])

    def test_tampered_integrated_sealer_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            repository, commit, blob, file_sha256 = (
                make_integrated_sealer_repository(root)
            )
            sealer = repository / sealmod.SEALER_RELATIVE
            with sealer.open("ab") as stream:
                stream.write(b"\n# tampered after integration\n")
            with self.assertRaisesRegex(sealmod.SealError, "dirty|bytes"):
                sealmod.verify_sealer_authority(
                    commit,
                    blob,
                    file_sha256,
                    script_path=sealer,
                )

    def test_untracked_sealer_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            repository, commit, blob, file_sha256 = (
                make_integrated_sealer_repository(root)
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "rm",
                    "--quiet",
                    "--cached",
                    "--",
                    sealmod.SEALER_RELATIVE,
                ],
                check=True,
            )
            with self.assertRaisesRegex(sealmod.SealError, "not tracked"):
                sealmod.verify_sealer_authority(
                    commit,
                    blob,
                    file_sha256,
                    script_path=repository / sealmod.SEALER_RELATIVE,
                )

    def test_sealer_from_wrong_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            repository, expected_commit, blob, file_sha256 = (
                make_integrated_sealer_repository(root)
            )
            marker = repository / "wrong-commit-marker.txt"
            marker.write_text("moves HEAD without changing sealer bytes\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(repository), "add", "--", marker.name],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "commit",
                    "--quiet",
                    "-m",
                    "Move integrated authority",
                ],
                check=True,
            )
            with self.assertRaisesRegex(sealmod.SealError, "pinned commit"):
                sealmod.verify_sealer_authority(
                    expected_commit,
                    blob,
                    file_sha256,
                    script_path=repository / sealmod.SEALER_RELATIVE,
                )

    def test_unrelated_local_branch_is_rejected(self):
        reference = "refs/heads/unrelated-negative"
        parent = subprocess.run(
            [
                "git",
                "-C",
                str(self.official_repository),
                "rev-parse",
                f"{sealmod.EXPECTED_SOURCE_COMMIT}^",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            [
                "git",
                "-C",
                str(self.official_repository),
                "update-ref",
                reference,
                parent,
            ],
            check=True,
        )
        try:
            with self.assertRaisesRegex(sealmod.SealError, "branchless"):
                sealmod.verify_source(self.official_repository)
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.official_repository),
                    "update-ref",
                    "-d",
                    reference,
                ],
                check=True,
            )

    def test_fake_producer_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            producer = root / sealmod.PRODUCER_RELATIVE
            producer.parent.mkdir(parents=True)
            producer.write_text("print('fake')\n", encoding="utf-8")
            with self.assertRaises(sealmod.SealError):
                sealmod.verify_source(root)

    def test_real_preflight_schema_binds_authority_mode_into_producer_argv(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            preflight, validation_root, mode = real_schema_closure_fixture(root)
            closure, command = sealmod.build_input_closure(
                preflight, validation_root
            )
            self.assertEqual(closure["topology_mode"], mode)
            mode_index = command.index("--mode")
            self.assertEqual(command[mode_index + 1], mode)
            self.assertNotIn("None", command)

            top_level_spoof = copy.deepcopy(preflight)
            top_level_spoof["topology_mode"] = mode
            top_level_spoof["short_quality_authority"].pop("topology_mode")
            with self.assertRaisesRegex(sealmod.SealError, "topology mode"):
                sealmod.build_input_closure(top_level_spoof, validation_root)

            empty = copy.deepcopy(preflight)
            empty["short_quality_authority"]["topology_mode"] = ""
            with self.assertRaisesRegex(sealmod.SealError, "topology mode"):
                sealmod.build_input_closure(empty, validation_root)

            mismatched = copy.deepcopy(preflight)
            mismatched["short_quality_authority"]["throughput_gate"] = {
                "topology_mode": "official_w1_b64_reference",
                "status": "pass",
            }
            with self.assertRaisesRegex(sealmod.SealError, "throughput"):
                sealmod.build_input_closure(mismatched, validation_root)

    def test_missing_or_wrong_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact, _payload = write_receipt(
                Path(raw).resolve() / "receipt.json", {"epoch": 1}
            )
            sealmod._strict_receipt(artifact, "receipt")
            missing = dict(artifact)
            missing.pop("bytes")
            with self.assertRaisesRegex(sealmod.SealError, "schema"):
                sealmod._strict_receipt(missing, "receipt")
            wrong_type = dict(artifact)
            wrong_type["bytes"] = str(artifact["bytes"])
            with self.assertRaisesRegex(sealmod.SealError, "positive integer"):
                sealmod._strict_receipt(wrong_type, "receipt")
            wrong_value = dict(artifact)
            wrong_value["bytes"] += 1
            with self.assertRaisesRegex(sealmod.SealError, "byte count"):
                sealmod._strict_receipt(wrong_value, "receipt")

    def test_sha_payload_relative_path_and_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact, _payload = write_receipt(
                Path(raw).resolve() / "receipt.json", {"epoch": 1}
            )
            wrong_sha = dict(artifact)
            wrong_sha["sha256"] = "f" * 64
            with self.assertRaisesRegex(sealmod.SealError, "SHA-256 changed"):
                sealmod._strict_receipt(wrong_sha, "receipt")
            wrong_payload = dict(artifact)
            wrong_payload["receipt_payload_sha256"] = "e" * 64
            with self.assertRaisesRegex(sealmod.SealError, "payload closure"):
                sealmod._strict_receipt(wrong_payload, "receipt")
            relative = dict(artifact)
            relative["path"] = "receipt.json"
            with self.assertRaisesRegex(sealmod.SealError, "absolute"):
                sealmod._strict_receipt(relative, "receipt")
            with self.assertRaisesRegex(sealmod.SealError, "duplicate JSON key"):
                sealmod._strict_json_bytes(b'{"a":1,"a":2}', "duplicate")

    def test_symlink_input_and_symlink_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            alias = root / "alias.json"
            alias.symlink_to(target)
            with self.assertRaisesRegex(sealmod.SealError, "symlink"):
                sealmod._safe_read(alias, "alias")
            real_parent = root / "real-parent"
            real_parent.mkdir()
            parent_alias = root / "parent-alias"
            parent_alias.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(sealmod.SealError, "symlink"):
                sealmod._create_campaign_root(parent_alias / "campaign")
            self.assertFalse((real_parent / "campaign").exists())

    def test_pathname_swap_during_fd_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            victim = root / "victim.json"
            replacement = root / "replacement.json"
            victim.write_text('{"a":1}', encoding="utf-8")
            replacement.write_text('{"a":2}', encoding="utf-8")
            original = sealmod._pathname_stat
            attacked = False

            def swap(path):
                nonlocal attacked
                if Path(path) == victim and not attacked:
                    attacked = True
                    os.replace(replacement, victim)
                return original(path)

            with mock.patch.object(sealmod, "_pathname_stat", side_effect=swap):
                with self.assertRaisesRegex(sealmod.SealError, "pathname changed"):
                    sealmod._safe_read(victim, "victim")

    def test_same_inode_output_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            quality = root / "quality.json"
            short = root / "short.json"
            quality.write_text("fixture", encoding="utf-8")
            os.link(quality, short)
            artifact_q = {
                "path": str(quality),
                "sha256": "a" * 64,
                "bytes": 7,
            }
            artifact_s = {
                "path": str(short),
                "sha256": "a" * 64,
                "bytes": 7,
            }
            qjson = {"receipt_sha256": "b" * 64}
            sjson = {"receipt_payload_sha256": "b" * 64}
            with mock.patch.object(
                sealmod,
                "_strict_artifact",
                side_effect=[(artifact_q, qjson), (artifact_s, sjson)],
            ), mock.patch.object(sealmod, "_payload_sha", return_value="b" * 64):
                with self.assertRaisesRegex(sealmod.SealError, "distinct inodes"):
                    sealmod._output_artifacts(quality, short)

    def test_success_completion_is_last_and_second_call_is_idempotent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            _runtime, _source, _closure, args, patches = self.common_patches(root)
            producer = mock.Mock(side_effect=publish_fake_official_outputs)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], mock.patch.object(
                sealmod, "run_producer", producer
            ):
                first = sealmod.seal(args)
                self.assertEqual(first["status"], "complete")
                self.assertTrue(first["sealer"]["tracked"])
                global_claim = json.loads(
                    Path(first["global_claim"]["path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(global_claim["sealer"], first["sealer"])
                campaign_claim = json.loads(
                    (args.campaign_root / "claim.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    campaign_claim["binding"]["sealer"], first["sealer"]
                )
                self.assertTrue(
                    (args.campaign_root / "sealed" / "completion.json").is_file()
                )
                self.assertFalse((args.campaign_root / "failed").exists())
                second = sealmod.seal(args)
                self.assertEqual(
                    second["receipt_payload_sha256"],
                    first["receipt_payload_sha256"],
                )
            self.assertEqual(producer.call_count, 1)

    def test_same_semantic_campaign_cannot_be_consumed_at_a_second_root(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            _runtime, _source, _closure, args, patches = self.common_patches(root)
            producer = mock.Mock(side_effect=publish_fake_official_outputs)
            second_args = argparse.Namespace(**vars(args))
            second_args.campaign_root = root / "second-campaign"
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], mock.patch.object(
                sealmod, "run_producer", producer
            ):
                sealmod.seal(args)
                with self.assertRaisesRegex(sealmod.SealError, "different root"):
                    sealmod.seal(second_args)
            self.assertEqual(producer.call_count, 1)
            self.assertFalse(second_args.campaign_root.exists())

    def test_mid_producer_failure_moves_all_partial_outputs_to_failed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            _runtime, _source, _closure, args, patches = self.common_patches(root)
            calls = 0

            def fail_mid(_r, _s, _c, _quality, short):
                nonlocal calls
                calls += 1
                short.write_text('{"status":"complete"}\n', encoding="utf-8")
                raise sealmod.SealError("injected mid-producer failure")

            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], mock.patch.object(
                sealmod, "run_producer", side_effect=fail_mid
            ):
                with self.assertRaisesRegex(sealmod.SealError, "injected"):
                    sealmod.seal(args)
                self.assertFalse((args.campaign_root / "sealed").exists())
                self.assertTrue((args.campaign_root / "failed").is_dir())
                self.assertTrue(
                    (args.campaign_root / "failed" / "failure.json").is_file()
                )
                self.assertFalse(
                    (args.campaign_root / "failed" / "completion.json").exists()
                )
                with self.assertRaisesRegex(sealmod.SealError, "already consumed"):
                    sealmod.seal(args)
            self.assertEqual(calls, 1)

    def test_completed_output_tamper_is_rejected_on_idempotent_replay(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            _runtime, _source, _closure, args, patches = self.common_patches(root)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], mock.patch.object(
                sealmod,
                "run_producer",
                side_effect=publish_fake_official_outputs,
            ):
                sealmod.seal(args)
                quality = args.campaign_root / "sealed" / "quality-report.json"
                quality.unlink()
                quality.write_text("{}", encoding="utf-8")
                with self.assertRaises(sealmod.SealError):
                    sealmod.seal(args)

    def test_python_39_runtime_is_rejected_even_if_binary_sha_is_pinned(self):
        if sys.version_info[:2] >= (3, 10):
            self.skipTest("host interpreter is already Python >=3.10")
        real = Path(sys.executable).resolve(strict=True)
        digest = hashlib.sha256(real.read_bytes()).hexdigest()
        with mock.patch.object(
            sealmod, "EXPECTED_PYTHON_REAL_SHA256", digest
        ):
            with self.assertRaisesRegex(sealmod.SealError, "runtime contract"):
                sealmod.verify_runtime(Path(sys.executable))


if __name__ == "__main__":
    unittest.main(verbosity=2)
