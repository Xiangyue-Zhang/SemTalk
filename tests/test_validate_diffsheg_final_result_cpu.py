from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest import mock
from typing import Optional


from scripts.show_base import evaluate_diffsheg_final_test as producer
from scripts.show_base import validate_diffsheg_final_result as validator


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(producer.canonical_json_bytes(value))


class FinalResultFixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.inference_root = self.root / "inference"
        self.output_root = self.inference_root / "diffsheg-final-metrics"
        self.inference_root.mkdir()
        self.output_root.mkdir()
        self.preflight_path = self.root / "preflight.json"
        self.claim_path = producer._claim_path(self.inference_root)
        self.paspa_path = self.output_root / "paspa_diffsheg_show_metrics.json"
        self.final_path = self.output_root / "final_metrics.json"
        self.authority = {
            "fresh_test_authority": {
                "path": str(self.root / "authority.json"),
                "sha256": "1" * 64,
                "bytes": 1,
                "receipt_payload_sha256": "2" * 64,
            },
            "source": {
                "source_root": str(validator.PROJECT_ROOT),
                "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                "commit": "3" * 40,
                "tree": "4" * 40,
                "clean": True,
                "detached": True,
                "local_branches_at_commit": [],
            },
        }
        self.preflight = {
            "format": producer.PREFLIGHT_FORMAT,
            "status": "formal_ready",
            "authority": self.authority,
            "inference": {
                "root": str(self.inference_root),
                "npz_root": str(self.inference_root / "npz"),
                "clip_manifest": {"path": str(self.root / "clips.txt")},
                "input_set_sha256": "5" * 64,
                "frame_count": 456,
                "window_count": 123,
                "uncovered_tail_frames": 7,
                "paspa_clip_manifest_sha256": "6" * 64,
            },
            "protocol": {
                "name": "diffsheg_show_reconstructed",
                "version": 1,
                "compatibility_reports": {
                    "talkshow_body_face": {
                        "status": "separate",
                        "primary": False,
                        "selection_feedback": False,
                        "metric_spaces_must_not_be_mixed": True,
                    }
                },
            },
            "assets": {
                "paspa": {
                    "evaluator": {
                        "path": "/pinned/diffsheg_show_eval.py",
                        "sha256": producer.PASPA_EVALUATOR_SHA256,
                    }
                },
                "diffsheg": {
                    "root": "/assets/diffsheg",
                    "autoencoders": {
                        metric: {
                            "path": f"/assets/{pin['filename']}",
                            "sha256": pin["sha256"],
                            "input_dim": pin["input_dim"],
                        }
                        for metric, pin in producer.DIFFSHEG_AE_PINS.items()
                    },
                },
                "talkshow": {
                    "root": "/assets/talkshow",
                    "commit": producer.TALKSHOW_COMMIT,
                },
                "smplx": {
                    "model_path": "/assets/SMPLX_NEUTRAL.npz",
                    "sha256": producer.SMPLX_NEUTRAL_SHA256,
                },
                "audio": {"root": "/assets/audio"},
            },
        }
        self.preflight["receipt_payload_sha256"] = (
            producer.canonical_json_sha256(self.preflight)
        )
        _write_json(self.preflight_path, self.preflight)
        self.preflight_sha = producer.sha256_file(self.preflight_path)
        self.metrics = {
            "fmd": 0.4,
            "fed": 0.5,
            "expression_diversity": 0.3,
            "fgd": 0.6,
            "ba": 0.7,
            "pcm": 0.1,
            "gesture_diversity": 0.2,
        }
        self.paspa_report = {
            "status": "ok",
            "protocol": {
                "name": "diffsheg_show_reconstructed",
                "version": 1,
                "status": "reconstructed_from_public_components",
                "diffsheg_reference_commit": producer.DIFFSHEG_COMMIT,
                "window_length": 88,
                "window_stride": 88,
                "overlap_length": 0,
                "precision": "float32 AE inference; no autocast",
            },
            "inputs": {
                "clip_count": 1_708,
                "window_count": 123,
                "frame_count": 456,
                "uncovered_tail_frames": 7,
                "clip_manifest_sha256": "6" * 64,
                "audio_protocol": "talkshow_original_source",
                "stats": {"sha256": producer.DIFFSHEG_STATS_SHA256},
                "checkpoint_paths": {
                    metric: f"/assets/{pin['filename']}"
                    for metric, pin in producer.DIFFSHEG_AE_PINS.items()
                },
            },
            "metrics": self.metrics,
            "provenance": {
                "evaluator": {
                    "sha256": producer.PASPA_EVALUATOR_SHA256,
                    "repository_git_head": producer.PASPA_COMMIT,
                },
                "autoencoders": {
                    metric: {
                        "path": f"/assets/{pin['filename']}",
                        "sha256": pin["sha256"],
                        "input_dim": pin["input_dim"],
                        "feature_count": 123,
                    }
                    for metric, pin in producer.DIFFSHEG_AE_PINS.items()
                },
                "ba": {
                    "talkshow_git_head": producer.TALKSHOW_COMMIT,
                    "smplx_neutral_asset_sha256": (
                        producer.SMPLX_NEUTRAL_SHA256
                    ),
                },
            },
        }
        _write_json(self.paspa_path, self.paspa_report)
        self.claim = {
            "format": producer.CLAIM_FORMAT,
            "status": "claimed",
            "authority": self.authority,
            "preflight": {
                "path": str(self.preflight_path),
                "sha256": self.preflight_sha,
                "receipt_payload_sha256": self.preflight[
                    "receipt_payload_sha256"
                ],
            },
            "output_root": str(self.output_root),
            "test_evaluations": 1,
            "test_feedback_into_selection": False,
            "input_set_sha256": "5" * 64,
        }
        producer._atomic_new(self.claim_path, self.claim, "fixture claim")
        self.final = {
            "format": producer.RESULT_FORMAT,
            "status": "complete",
            "authority": self.authority,
            "one_shot_claim": {
                "path": str(self.claim_path),
                "sha256": producer.sha256_file(self.claim_path),
            },
            "preflight": self.claim["preflight"],
            "paspa_report": {
                "path": str(self.paspa_path),
                "sha256": producer.sha256_file(self.paspa_path),
                "bytes": self.paspa_path.stat().st_size,
            },
            "evaluator_command_sha256": "7" * 64,
            "protocol": self.preflight["protocol"],
            "assets": self.preflight["assets"],
            "coverage": {
                "clip_count": 1_708,
                "exact_once": True,
                "num_generation_shards": 8,
                "window_length": 88,
                "window_stride": 88,
                "window_count": 123,
            },
            "metrics": self.metrics,
            "all_metrics_finite": True,
            "test_evaluations": 1,
            "validation_only_for_selection": True,
            "test_feedback_into_selection": False,
            "compatibility_report_policy": self.preflight["protocol"][
                "compatibility_reports"
            ],
        }
        self.final["receipt_payload_sha256"] = producer.canonical_json_sha256(
            self.final
        )
        _write_json(self.final_path, self.final)
        self.args = SimpleNamespace(
            preflight_json=self.preflight_path,
            expected_preflight_sha256=self.preflight_sha,
            final_metrics_json=self.final_path,
            expected_final_metrics_sha256=producer.sha256_file(self.final_path),
            expected_final_metrics_bytes=self.final_path.stat().st_size,
            expected_final_metrics_canonical_payload_sha256=self.final[
                "receipt_payload_sha256"
            ],
        )
        self.source_receipt = {
            "source_root": str(validator.PROJECT_ROOT),
            "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
            "commit": "3" * 40,
            "tree": "4" * 40,
            "path": str(validator.Path(validator.__file__).resolve()),
            "relative_path": "scripts/show_base/validate_diffsheg_final_result.py",
            "git_blob_oid": "8" * 40,
            "sha256": "9" * 64,
            "bytes": 1,
            "clean": True,
            "detached": True,
            "local_branches": [],
        }

    def validate(self, preflight: Optional[dict] = None) -> dict:
        with mock.patch.object(
            validator,
            "_validate_validator_source_closure",
            return_value=self.source_receipt,
        ):
            return validator.validate_final_result(
                self.args,
                self.preflight if preflight is None else preflight,
            )


class FinalResultValidationTests(unittest.TestCase):
    def test_terminal_authority_path_replays_full_authority_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            final_root = root / "final"
            final_root.mkdir()
            checkpoint = root / "base.bin"
            checkpoint.write_bytes(b"base")
            selected = {
                "path": str(checkpoint),
                "sha256": producer.sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
            authority = {
                "winner_selection": {
                    "path": str(root / "winner.json"),
                    "sha256": "1" * 64,
                    "bytes": 1,
                    "receipt_payload_sha256": "2" * 64,
                    "canonical_payload_sha256": "3" * 64,
                    "selected_epoch": 400,
                    "selected_optimizer_updates": 99_200,
                    "selected_checkpoint": selected,
                },
                "test_claim": {
                    "test_policy": {
                        "authorized_evaluations": 1,
                        "one_shot_claim_required": True,
                        "selection_feedback": False,
                        "num_shards": 8,
                        "canonical_test_clips": 1_708,
                    }
                },
                "checkpoints": {
                    stage: {
                        "path": str(root / f"{stage}.bin"),
                        "sha256": hashlib.sha256(stage.encode()).hexdigest(),
                        "bytes": 1,
                    }
                    for stage in producer.final_test.CHECKPOINT_STAGES
                },
                "inference_source": {
                    "source_root": str(producer.PROJECT_ROOT),
                    "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                    "commit": "4" * 40,
                    "tree": "5" * 40,
                    "entrypoint": (
                        "scripts/show_base/semtalk_base_inference_core.py"
                    ),
                    "entrypoint_sha256": "6" * 64,
                    "clean": True,
                    "detached": True,
                    "local_branches_at_commit": [],
                },
            }
            authority["checkpoints"]["base"] = selected
            args = SimpleNamespace(
                fresh_test_authority=root / "authority.json",
                expected_test_authority_sha256="7" * 64,
                expected_test_authority_bytes=1,
                expected_test_authority_receipt_payload_sha256="8" * 64,
                prepared_authority=None,
                expected_prepared_authority_sha256=None,
                expected_prepared_authority_bytes=None,
                expected_prepared_authority_receipt_payload_sha256=None,
                inference_final_root=final_root,
            )
            with (
                mock.patch.object(
                    producer.final_test.final_authority,
                    "validate_test_authority",
                    return_value=authority,
                ) as full_replay,
                mock.patch.object(
                    producer.final_test,
                    "_validate_authority_identity",
                    side_effect=lambda value: value,
                ),
                mock.patch.object(
                    producer.final_test,
                    "_output_roots",
                    return_value=(final_root, root / "shards"),
                ),
            ):
                observed, _receipt = producer._load_current_authority(args)
            self.assertIs(observed, authority)
            full_replay.assert_called_once_with(
                str(args.fresh_test_authority.resolve()),
                expected_file_sha256="7" * 64,
                expected_bytes=1,
                expected_receipt_payload_sha256="8" * 64,
            )

    def test_terminal_cli_rejects_every_prepared_authority_option(self) -> None:
        base = [
            "--fresh-test-authority", "/authority.json",
            "--expected-test-authority-sha256", "1" * 64,
            "--expected-test-authority-bytes", "1",
            "--expected-test-authority-receipt-payload-sha256", "2" * 64,
            "--inference-final-root", "/inference",
            "--paspa-root", "/paspa",
            "--diffsheg-root", "/diffsheg",
            "--talkshow-root", "/talkshow",
            "--source-audio-root", "/audio",
            "--smplx-path", "/smplx.npz",
            "--expected-audio-set-sha256", "3" * 64,
            "--preflight-json", "/preflight.json",
            "--expected-preflight-sha256", "4" * 64,
            "--final-metrics-json", "/final_metrics.json",
            "--expected-final-metrics-sha256", "5" * 64,
            "--expected-final-metrics-bytes", "1",
            "--expected-final-metrics-canonical-payload-sha256", "6" * 64,
            "--output-report", "/validation.json",
        ]
        prepared_options = (
            ("--prepared-authority", "/prepared.json"),
            ("--expected-prepared-authority-sha256", "7" * 64),
            ("--expected-prepared-authority-bytes", "1"),
            (
                "--expected-prepared-authority-receipt-payload-sha256",
                "8" * 64,
            ),
        )
        for option in prepared_options:
            with self.subTest(option=option[0]), mock.patch.object(
                producer, "build_preflight"
            ) as build:
                with self.assertRaises(SystemExit) as raised:
                    validator.main([*base, *option])
                self.assertEqual(raised.exception.code, 2)
                build.assert_not_called()

    def test_validator_source_is_bound_to_commit_tree_and_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path = (
                root
                / "scripts"
                / "show_base"
                / "validate_diffsheg_final_result.py"
            )
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(Path(validator.__file__).read_bytes())

            def git(*arguments: str) -> str:
                result = subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()

            subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
            git("config", "user.name", "Xiangyue-Zhang")
            git(
                "config",
                "user.email",
                "85532891+Xiangyue-Zhang@users.noreply.github.com",
            )
            git("remote", "add", "origin", "git@github.com:Xiangyue-Zhang/SemTalk.git")
            git("add", "scripts/show_base/validate_diffsheg_final_result.py")
            git("commit", "-m", "fixture")
            branch = git("symbolic-ref", "--short", "HEAD")
            commit = git("rev-parse", "HEAD^{commit}")
            tree = git("rev-parse", "HEAD^{tree}")
            git("checkout", "--detach", commit)
            git("branch", "-D", branch)
            preflight = {
                "authority": {
                    "source": {
                        "source_root": str(root),
                        "origin": "git@github.com:Xiangyue-Zhang/SemTalk.git",
                        "commit": commit,
                        "tree": tree,
                        "clean": True,
                        "detached": True,
                        "local_branches_at_commit": [],
                    }
                }
            }
            with (
                mock.patch.object(validator, "PROJECT_ROOT", root),
                mock.patch.object(validator, "__file__", str(source_path)),
            ):
                receipt = validator._validate_validator_source_closure(
                    preflight
                )
            self.assertEqual(receipt["commit"], commit)
            self.assertEqual(receipt["tree"], tree)
            self.assertEqual(receipt["relative_path"], (
                "scripts/show_base/validate_diffsheg_final_result.py"
            ))
            self.assertEqual(receipt["git_blob_oid"], git(
                "rev-parse",
                f"{tree}:scripts/show_base/validate_diffsheg_final_result.py",
            ))

    def test_accepts_fresh_exact_seven_metric_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            result = fixture.validate()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["metrics"], fixture.metrics)
            self.assertEqual(result["metric_count"], 7)
            self.assertEqual(
                result["final_metrics"]["sha256"],
                fixture.args.expected_final_metrics_sha256,
            )
            self.assertTrue(
                result["one_shot_claim"]["exclusive_0600_single_link"]
            )

    def test_rejects_nonempty_evaluator_output_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            fixture.final_path.write_bytes(b'{"status":"forged-but-nonempty"}\n')
            with self.assertRaisesRegex(
                validator.FinalResultValidationError, "SHA-256 mismatch"
            ):
                fixture.validate()

    def test_rejects_byte_and_external_pin_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            payload = fixture.final_path.read_bytes()
            fixture.final_path.write_bytes(payload[:-2] + b" \n")
            with self.assertRaisesRegex(
                validator.FinalResultValidationError, "SHA-256 mismatch"
            ):
                fixture.validate()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            fixture.args.expected_final_metrics_sha256 = "a" * 64
            with self.assertRaisesRegex(
                validator.FinalResultValidationError, "SHA-256 mismatch"
            ):
                fixture.validate()

    def test_rejects_payload_content_and_payload_pin_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            changed = copy.deepcopy(fixture.final)
            changed["metrics"]["fgd"] = 9.9
            _write_json(fixture.final_path, changed)
            fixture.args.expected_final_metrics_sha256 = producer.sha256_file(
                fixture.final_path
            )
            fixture.args.expected_final_metrics_bytes = fixture.final_path.stat().st_size
            with self.assertRaisesRegex(
                validator.FinalResultValidationError,
                "canonical payload SHA-256 mismatch",
            ):
                fixture.validate()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            fixture.args.expected_final_metrics_canonical_payload_sha256 = (
                "b" * 64
            )
            with self.assertRaisesRegex(
                validator.FinalResultValidationError,
                "canonical payload SHA-256 mismatch",
            ):
                fixture.validate()

    def test_rejects_stale_fresh_authority_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            stale = copy.deepcopy(fixture.preflight)
            stale["authority"]["source"]["commit"] = "c" * 40
            stale["receipt_payload_sha256"] = producer.canonical_json_sha256(
                {
                    key: value
                    for key, value in stale.items()
                    if key != "receipt_payload_sha256"
                }
            )
            with self.assertRaisesRegex(
                validator.FinalResultValidationError,
                "fresh authority/preflight replay differs",
            ):
                fixture.validate(stale)

    def test_rejects_stale_claim_and_paspa_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            fixture.claim_path.write_bytes(b'{"status":"stale"}\n')
            fixture.claim_path.chmod(0o600)
            with self.assertRaisesRegex(
                validator.FinalResultValidationError,
                "stale or malformed|claim receipt changed",
            ):
                fixture.validate()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FinalResultFixture(Path(temporary))
            changed = copy.deepcopy(fixture.paspa_report)
            changed["metrics"]["fgd"] = 9.9
            _write_json(fixture.paspa_path, changed)
            with self.assertRaisesRegex(
                validator.FinalResultValidationError,
                "PASPA DiffSHEG report SHA-256 mismatch",
            ):
                fixture.validate()

    def test_launcher_uses_pipe_pins_then_atomic_validator_receipt(self) -> None:
        shell = (
            validator.PROJECT_ROOT
            / "scripts"
            / "show_base"
            / "run_base_final_test.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("coproc METRIC", shell)
        self.assertIn("mkfifo -m 600", shell)
        self.assertIn('exec 9<"$metric_fifo"', shell)
        self.assertIn('rm -- "$metric_fifo"', shell)
        self.assertIn('while IFS= read -r metric_stdout_line <&9', shell)
        self.assertIn("metric stdout FIFO endpoint identity changed", shell)
        self.assertIn("unlinked metric stdout stream identity changed", shell)
        self.assertIn('<<<"$metric_stdout"', shell)
        self.assertNotIn("metric_log=$log_root/diffsheg-metrics.log", shell)
        self.assertNotIn("pathlib.Path(sys.argv[1]).read_text", shell)
        self.assertIn("validate_diffsheg_final_result.py", shell)
        validator_invocation = shell.split("validation_result=$(", 1)[1].split(
            "\n)", 1
        )[0]
        self.assertIn('"${authority_args[@]}"', validator_invocation)
        self.assertNotIn(
            '"${workload_authority_args[@]}"', validator_invocation
        )
        self.assertLess(
            shell.index("validate_diffsheg_final_result.py"),
            shell.index("SemTalk SHOW DiffSHEG final test complete"),
        )
        self.assertIn("observed_validation_sha=$(sha256sum", shell)
        self.assertIn("observed_validation_bytes=$(stat -c %s", shell)
        self.assertIn("terminal PASS receipt payload pin changed", shell)
        self.assertNotIn(
            "[[ -f $diffsheg_output/final_metrics.json", shell
        )

    def test_producer_structured_return_exposes_all_external_pins(self) -> None:
        source = (
            validator.PROJECT_ROOT
            / "scripts"
            / "show_base"
            / "evaluate_diffsheg_final_test.py"
        ).read_text(encoding="utf-8")
        for token in (
            '"final_metrics": completion_pin',
            '"sha256": completion_file_sha256',
            '"bytes": completion_bytes',
            '"canonical_payload_sha256": completion[',
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
