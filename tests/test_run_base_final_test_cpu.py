from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts.show_base import run_base_final_test as target


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def artifact(path: Path, payload: bytes = b"x") -> dict[str, object]:
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def authority_args(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        fresh_test_authority=path,
        expected_test_authority_sha256=SHA_A,
        expected_test_authority_bytes=7,
        expected_test_authority_receipt_payload_sha256=SHA_B,
        seed=20260801,
    )


def fake_authority(root: Path) -> dict[str, object]:
    checkpoints = {
        stage: {
            "path": str((root.parent / f"{stage}.bin").resolve()),
            "sha256": hashlib.sha256(stage.encode()).hexdigest(),
            "bytes": len(stage),
        }
        for stage in target.CHECKPOINT_STAGES
    }
    return {
        "format": target.final_authority.FORMAT,
        "status": "authorized_pre_inference",
        "receipt_payload_sha256": SHA_B,
        "contract": {
            "split": "test",
            "test_clips": target.TEST_CLIPS,
            "num_shards": target.NUM_SHARDS,
        },
        "expected_output_root": str(root.resolve()),
        "canonical": {
            "manifest": {"path": "/canonical.jsonl", "sha256": SHA_A, "bytes": 1},
            "summary": {"path": "/summary.json", "sha256": SHA_B, "bytes": 1},
            "lineage": {"path": "/lineage.json", "sha256": SHA_C, "bytes": 1},
        },
        "audio": [],
        "inference_source": {
            "source_root": str(target.PROJECT_ROOT.resolve()),
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "commit": "1" * 40,
            "tree": "2" * 40,
            "clean": True,
            "entrypoint": "scripts/show_base/semtalk_base_inference_core.py",
            "entrypoint_sha256": SHA_A,
        },
        "checkpoints": checkpoints,
        "winner_selection": {
            "path": "/winner.json",
            "sha256": SHA_A,
            "bytes": 1,
            "receipt_payload_sha256": SHA_B,
            "selected_epoch": 400,
            "selected_optimizer_updates": 99_200,
            "selected_checkpoint": checkpoints["base"],
        },
        "prerequisite_selection": {
            "path": "/five.json",
            "sha256": SHA_B,
            "bytes": 1,
            "receipt_payload_sha256": SHA_C,
            "stages": {},
        },
    }


class FinalTestCpuContractTests(unittest.TestCase):
    def test_cli_has_no_split_epoch_or_checkpoint_override(self) -> None:
        base = [
            "context",
            "--fresh-test-authority", "/authority.json",
            "--expected-test-authority-sha256", SHA_A,
            "--expected-test-authority-bytes", "7",
            "--expected-test-authority-receipt-payload-sha256", SHA_B,
        ]
        for forbidden in (
            ["--split", "test"],
            ["--epoch", "400"],
            ["--base-checkpoint", "/candidate.bin"],
            ["--selection-metric", "test.FGD"],
        ):
            with self.subTest(forbidden=forbidden), self.assertRaises(SystemExit):
                target.parse_args(base + forbidden)

    def test_external_authority_pins_are_forwarded_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            path.write_text("{}")
            args = authority_args(path)
            expected = fake_authority(Path(directory) / "final")
            with mock.patch.object(
                target.final_authority,
                "validate_test_authority",
                return_value=expected,
            ) as validator:
                observed = target._validated_authority(args)
            self.assertIs(observed, expected)
            validator.assert_called_once_with(
                str(path.resolve()),
                expected_file_sha256=SHA_A,
                expected_bytes=7,
                expected_receipt_payload_sha256=SHA_B,
            )

    def test_authority_identity_cannot_be_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            path.write_text("{}")
            value = fake_authority(Path(directory) / "final")
            value["status"] = "authorized"
            with mock.patch.object(
                target.final_authority,
                "validate_test_authority",
                return_value=value,
            ), self.assertRaisesRegex(
                target.FinalTestContractError, "identity changed"
            ):
                target._validated_authority(authority_args(path))

    def test_prepared_receipt_avoids_replaying_twenty_two_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "final"
            value = fake_authority(root)
            authority_path = base / "authority.json"
            authority_path.write_bytes(target._canonical_json_bytes(value))
            authority_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
            fresh_args = SimpleNamespace(
                fresh_test_authority=authority_path,
                expected_test_authority_sha256=authority_sha,
                expected_test_authority_bytes=authority_path.stat().st_size,
                expected_test_authority_receipt_payload_sha256=SHA_B,
                prepared_authority=None,
                expected_prepared_authority_sha256=None,
                expected_prepared_authority_bytes=None,
                expected_prepared_authority_receipt_payload_sha256=None,
                prepared_authority_output=base / "prepared.json",
                nul_context=False,
                seed=20260801,
            )
            with mock.patch.object(
                target.final_authority,
                "validate_test_authority",
                return_value=value,
            ), redirect_stdout(io.StringIO()):
                context = target.run_context(fresh_args)
            prepared = context["prepared_authority"]
            prepared_args = SimpleNamespace(
                fresh_test_authority=authority_path,
                expected_test_authority_sha256=authority_sha,
                expected_test_authority_bytes=authority_path.stat().st_size,
                expected_test_authority_receipt_payload_sha256=SHA_B,
                prepared_authority=Path(prepared["path"]),
                expected_prepared_authority_sha256=prepared["sha256"],
                expected_prepared_authority_bytes=prepared["bytes"],
                expected_prepared_authority_receipt_payload_sha256=prepared[
                    "receipt_payload_sha256"
                ],
                seed=20260801,
            )
            with mock.patch.object(
                target.final_authority,
                "validate_test_authority",
                side_effect=AssertionError("full replay must not repeat"),
            ), mock.patch.object(
                target, "_validate_live_source_boundary"
            ):
                observed = target._validated_authority(prepared_args)
            self.assertEqual(observed, value)

    def test_prepared_authority_rebinds_the_live_source_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            value = fake_authority(base / "final")
            authority_path = base / "authority.json"
            authority_path.write_bytes(target._canonical_json_bytes(value))
            authority_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
            fresh_args = SimpleNamespace(
                fresh_test_authority=authority_path,
                expected_test_authority_sha256=authority_sha,
                expected_test_authority_bytes=authority_path.stat().st_size,
                expected_test_authority_receipt_payload_sha256=SHA_B,
                prepared_authority=None,
                expected_prepared_authority_sha256=None,
                expected_prepared_authority_bytes=None,
                expected_prepared_authority_receipt_payload_sha256=None,
                prepared_authority_output=base / "prepared.json",
                nul_context=False,
                seed=20260801,
            )
            with mock.patch.object(
                target.final_authority,
                "validate_test_authority",
                return_value=value,
            ), redirect_stdout(io.StringIO()):
                context = target.run_context(fresh_args)
            prepared = context["prepared_authority"]
            prepared_args = SimpleNamespace(
                fresh_test_authority=authority_path,
                expected_test_authority_sha256=authority_sha,
                expected_test_authority_bytes=authority_path.stat().st_size,
                expected_test_authority_receipt_payload_sha256=SHA_B,
                prepared_authority=Path(prepared["path"]),
                expected_prepared_authority_sha256=prepared["sha256"],
                expected_prepared_authority_bytes=prepared["bytes"],
                expected_prepared_authority_receipt_payload_sha256=prepared[
                    "receipt_payload_sha256"
                ],
                seed=20260801,
            )
            with mock.patch.object(
                target,
                "_validate_live_source_boundary",
                side_effect=target.FinalTestContractError("live source changed"),
            ) as live_proof, self.assertRaisesRegex(
                target.FinalTestContractError, "live source changed"
            ):
                target._validated_authority(prepared_args)
            live_proof.assert_called_once_with(value)

    def test_live_source_boundary_rejects_a_dirty_detached_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def git(*arguments: str) -> str:
                return target.subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    check=True,
                    stdout=target.subprocess.PIPE,
                    stderr=target.subprocess.PIPE,
                    text=True,
                ).stdout.strip()

            git("init", "-q")
            git("config", "user.name", "Xiangyue-Zhang")
            git(
                "config",
                "user.email",
                "85532891+Xiangyue-Zhang@users.noreply.github.com",
            )
            (root / "tracked.txt").write_text("frozen\n")
            git("add", "tracked.txt")
            git("commit", "-qm", "fixture")
            branch = git("branch", "--show-current")
            commit = git("rev-parse", "HEAD^{commit}")
            tree = git("rev-parse", "HEAD^{tree}")
            git("remote", "add", "origin", "fixture://semtalk")
            git("checkout", "-q", "--detach")
            git("branch", "-D", branch)
            value = {
                "inference_source": {
                    "source_root": str(root),
                    "origin": "fixture://semtalk",
                    "commit": commit,
                    "tree": tree,
                }
            }
            with mock.patch.object(target, "PROJECT_ROOT", root):
                target._validate_live_source_boundary(value)
                (root / "tracked.txt").write_text("mutated\n")
                with self.assertRaisesRegex(
                    target.FinalTestContractError, "clean detached authority tree"
                ):
                    target._validate_live_source_boundary(value)

    def test_verified_tree_loader_executes_tracked_source_not_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def git(*arguments: str) -> str:
                return target.subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    check=True,
                    stdout=target.subprocess.PIPE,
                    stderr=target.subprocess.PIPE,
                    text=True,
                ).stdout.strip()

            git("init", "-q")
            git("config", "user.name", "Xiangyue-Zhang")
            git(
                "config",
                "user.email",
                "85532891+Xiangyue-Zhang@users.noreply.github.com",
            )
            source = root / "verified_fixture.py"
            source.write_text("VALUE = 'tracked'\n")
            git("add", "verified_fixture.py")
            git("commit", "-qm", "fixture")
            tree = git("rev-parse", "HEAD^{tree}")
            finder = target._VerifiedTreeFinder(
                {"inference_source": {"source_root": str(root), "tree": tree}}
            )
            target.sys.meta_path.insert(0, finder)
            target.sys.path.insert(0, str(root))
            target.sys.modules.pop("verified_fixture", None)
            try:
                module = target.importlib.import_module("verified_fixture")
                self.assertEqual(module.VALUE, "tracked")
                self.assertIsNone(module.__cached__)
                self.assertEqual(
                    module.__verified_source_receipt__["loader"],
                    "verified_git_source_snapshot_no_bytecode",
                )
                target.sys.modules.pop("verified_fixture", None)
                source.write_text("VALUE = 'transient-mutation'\n")
                with self.assertRaisesRegex(
                    target.FinalTestContractError, "differ from the authority tree"
                ):
                    target.importlib.import_module("verified_fixture")
            finally:
                target.sys.modules.pop("verified_fixture", None)
                target.sys.meta_path.remove(finder)
                target.sys.path.remove(str(root))

    def test_rehashed_prepared_authority_cannot_change_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "final"
            value = fake_authority(root)
            authority_path = base / "authority.json"
            authority_path.write_bytes(target._canonical_json_bytes(value))
            authority_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
            fresh_args = SimpleNamespace(
                fresh_test_authority=authority_path,
                expected_test_authority_sha256=authority_sha,
                expected_test_authority_bytes=authority_path.stat().st_size,
                expected_test_authority_receipt_payload_sha256=SHA_B,
                prepared_authority=None,
                expected_prepared_authority_sha256=None,
                expected_prepared_authority_bytes=None,
                expected_prepared_authority_receipt_payload_sha256=None,
                prepared_authority_output=base / "prepared.json",
                nul_context=False,
                seed=20260801,
            )
            with mock.patch.object(
                target.final_authority,
                "validate_test_authority",
                return_value=value,
            ), redirect_stdout(io.StringIO()):
                context = target.run_context(fresh_args)
            prepared_path = Path(context["prepared_authority"]["path"])
            forged = json.loads(prepared_path.read_text())
            forged["authority"]["winner_selection"]["selected_epoch"] = 1
            forged.pop("receipt_payload_sha256")
            forged["receipt_payload_sha256"] = target._canonical_json_sha256(
                forged
            )
            forged_path = base / "forged.json"
            forged_path.write_bytes(target._canonical_json_bytes(forged))
            forged_sha = hashlib.sha256(forged_path.read_bytes()).hexdigest()
            forged_args = SimpleNamespace(
                fresh_test_authority=authority_path,
                expected_test_authority_sha256=authority_sha,
                expected_test_authority_bytes=authority_path.stat().st_size,
                expected_test_authority_receipt_payload_sha256=SHA_B,
                prepared_authority=forged_path,
                expected_prepared_authority_sha256=forged_sha,
                expected_prepared_authority_bytes=forged_path.stat().st_size,
                expected_prepared_authority_receipt_payload_sha256=forged[
                    "receipt_payload_sha256"
                ],
                seed=20260801,
            )
            with self.assertRaisesRegex(
                target.FinalTestContractError,
                "differs from the externally pinned file",
            ):
                target._validated_authority(forged_args)

    def test_context_refuses_reuse_of_final_or_shard_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for occupied in ("final", "shards"):
                with self.subTest(occupied=occupied):
                    root = base / f"case-{occupied}" / "final"
                    root.parent.mkdir()
                    (root if occupied == "final" else root.parent / "shards").mkdir()
                    authority = fake_authority(root)
                    args = SimpleNamespace(
                        nul_context=False,
                        **vars(authority_args(base / "a")),
                    )
                    with mock.patch.object(
                        target, "_validated_authority", return_value=authority
                    ), self.assertRaisesRegex(
                        target.FinalTestContractError, "already consumed"
                    ):
                        target.run_context(args)

    def test_contract_binds_val_winner_and_five_without_test_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            value = fake_authority(root)
            contract = target._contract(value, authority_args(Path(directory) / "a"))
            self.assertEqual(contract["split"], "test")
            self.assertEqual(contract["test_clips"], 1_708)
            self.assertEqual(contract["validation_winner"]["selected_epoch"], 400)
            self.assertEqual(
                set(contract["validation_selected_five"]["checkpoint_sha256"]),
                set(target.REPRESENTATION_STAGES),
            )
            self.assertTrue(
                contract["selection_policy"]["validation_only_for_selection"]
            )
            self.assertEqual(contract["selection_policy"]["test_evaluations"], 1)
            self.assertIs(
                contract["selection_policy"]["test_feedback_into_selection"],
                False,
            )
            self.assertNotIn("candidate", contract)

    def test_input_loader_rejects_authority_row_hash_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            target,
            TEST_CLIPS=4,
            TEST_GLOBAL_START=10,
            TEST_GLOBAL_STOP=14,
            NUM_SHARDS=2,
        ):
            root = Path(directory)
            speakers = list(target.SHOW_SPEAKER_IDS)
            canonical_rows = []
            for offset, speaker in enumerate(speakers):
                npz = root / f"canonical-{offset}.npz"
                npz.write_bytes(f"canonical-{offset}".encode())
                canonical_rows.append(
                    {
                        "split": "test",
                        "global_index": 10 + offset,
                        "clip_id": f"{speaker}/video/sequence{offset}",
                        "speaker": speaker,
                        "speaker_id": target.SHOW_SPEAKER_IDS[speaker],
                        "frames": 61,
                        "canonical_npz": str(npz.resolve()),
                        "canonical_npz_sha256": hashlib.sha256(
                            npz.read_bytes()
                        ).hexdigest(),
                    }
                )
            canonical_payload = target._canonical_jsonl_bytes(canonical_rows)
            canonical = artifact(root / "canonical.jsonl", canonical_payload)
            audio_authority = []
            for shard_id in range(2):
                audio_rows = []
                for row in canonical_rows:
                    if row["global_index"] % 2 != shard_id:
                        continue
                    feature = root / f"feature-{row['global_index']}.npz"
                    feature.write_bytes(str(row["global_index"]).encode())
                    audio_rows.append(
                        {
                            "clip_id": row["clip_id"],
                            "split": "test",
                            "shard_id": shard_id,
                            "num_shards": 2,
                            "frames": row["frames"],
                            "canonical_npz_sha256": row["canonical_npz_sha256"],
                            "audio_feature_npz": str(feature.resolve()),
                            "audio_feature_npz_sha256": hashlib.sha256(
                                feature.read_bytes()
                            ).hexdigest(),
                        }
                    )
                manifest = artifact(
                    root / f"audio-{shard_id}.jsonl",
                    target._canonical_jsonl_bytes(audio_rows),
                )
                audio_authority.append(
                    {
                        "shard_id": shard_id,
                        "manifest": manifest,
                        "rows": len(audio_rows),
                        "ordered_row_sha256": [
                            target._canonical_json_sha256(row) for row in audio_rows
                        ],
                    }
                )
            value = {
                "canonical": {
                    "manifest": canonical,
                    "ordered_row_sha256": [
                        target._canonical_json_sha256(row) for row in canonical_rows
                    ],
                },
                "audio": audio_authority,
            }
            rows, by_id, audio = target._load_inputs(value)
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(by_id), set(audio))
            value["canonical"]["ordered_row_sha256"][2] = SHA_A
            with self.assertRaisesRegex(
                target.FinalTestContractError, "fresh authority"
            ):
                target._load_inputs(value)

    def test_legacy_distribution_gate_is_not_exposed(self) -> None:
        with self.assertRaises(SystemExit):
            target.parse_args(["distribution"])
        source = Path(target.__file__).resolve().read_text(encoding="utf-8")
        self.assertNotIn('add_parser("distribution"', source)
        self.assertNotIn("run_distribution", source)
        self.assertNotIn("validation_gate_json", source)

    def test_metric_seal_subcommand_is_not_exposed(self) -> None:
        with self.assertRaises(SystemExit):
            target.parse_args(["seal"])
        parser_source = (
            Path(target.__file__).resolve().read_text(encoding="utf-8")
        )
        self.assertNotIn('add_parser("seal"', parser_source)
        self.assertNotIn("run_seal", parser_source)

        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            target,
            TEST_CLIPS=1,
            TEST_GLOBAL_START=10,
            TEST_GLOBAL_STOP=11,
            NUM_SHARDS=1,
        ):
            root = Path(directory) / "final"
            npz_root = root / "npz" / "test"
            npz_root.mkdir(parents=True)
            prediction = artifact(npz_root / "res_oliver__sequence.npz", b"pred")
            ground_truth = artifact(npz_root / "gt_oliver__sequence.npz", b"gt")
            row = {
                "global_index": 10,
                "source_clip_id": "oliver/video/sequence",
                "canonical_clip_id": "oliver__sequence",
                "speaker": "oliver",
                "speaker_id": 0,
                "frames": 61,
                "canonical_npz": "/canonical.npz",
                "canonical_npz_sha256": SHA_A,
                "audio_feature_npz": "/audio.npz",
                "audio_feature_npz_sha256": SHA_B,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "evaluation_index": 0,
            }
            manifest_payload = target._canonical_jsonl_bytes([row])
            artifact(root / "final_manifest.jsonl", manifest_payload)
            contract = {"format": "fixture"}
            runtime = {"format": "fixture-runtime"}
            clip_payload = b"oliver__sequence\n"
            lineage = {
                "format": target.FINAL_LINEAGE_FORMAT,
                "status": "complete",
                "contract": contract,
                "contract_sha256": target._canonical_json_sha256(contract),
                "runtime": runtime,
                "runtime_sha256": target._canonical_json_sha256(runtime),
                "shards": [{}],
                "final_manifest_sha256": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
                "clip_manifest_sha256": hashlib.sha256(clip_payload).hexdigest(),
            }
            (root / "final_lineage.json").write_bytes(
                target._canonical_json_bytes(lineage)
            )
            value = {"expected_output_root": str(root.resolve())}
            _manifest, rows, _lineage = target._load_final_rows(value)
            self.assertEqual(len(rows), 1)
            Path(prediction["path"]).write_bytes(b"changed-after-metrics")
            with self.assertRaisesRegex(
                target.FinalTestContractError, "SHA-256 mismatch"
            ):
                target._load_final_rows(value)

    def test_shell_tracks_pending_children_and_signal_handlers_exit(self) -> None:
        shell = (
            target.PROJECT_ROOT
            / "scripts"
            / "show_base"
            / "run_base_final_test.sh"
        ).read_text()
        self.assertIn("pending_pid=$pid", shell)
        self.assertIn("pending_pid=$metric_pid", shell)
        self.assertIn("terminate_pending_child", shell)
        self.assertIn("trap 'signal_exit 130' INT", shell)
        self.assertIn("trap 'signal_exit 143' TERM", shell)
        self.assertIn("trap 'signal_exit 129' HUP", shell)
        signal_body = shell.split("\nsignal_exit() {", 1)[1].split("}", 1)[0]
        self.assertIn("trap - EXIT INT TERM HUP", signal_body)
        self.assertIn("cleanup_children", signal_body)
        self.assertIn('exit "$code"', signal_body)
        pending_body = shell.split("\nterminate_pending_child() {", 1)[1]
        pending_body = "terminate_pending_child() {" + pending_body.split(
            "\n}\n\ncleanup_children()", 1
        )[0] + "\n}"
        harness = f"""
set -euo pipefail
proc_snapshot() {{ printf '%s 77 deadbeef\\n' "$$"; }}
kill() {{ printf 'kill:%s\\n' "$3"; }}
wait() {{ printf 'wait:%s\\n' "$1"; }}
pending_pid=4242
pending_entry=/expected/python
pending_subcommand=shard
{pending_body}
terminate_pending_child
[[ -z $pending_pid ]]
"""
        completed = target.subprocess.run(
            ["bash", "-c", harness],
            check=True,
            stdout=target.subprocess.PIPE,
            stderr=target.subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.stdout, "kill:4242\nwait:4242\n")


if __name__ == "__main__":
    unittest.main()
