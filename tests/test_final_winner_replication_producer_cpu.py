from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
PRODUCER_PATH = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "produce_final_winner_replication_seed.py"
)
LAUNCHER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "run_final_winner_replication_gate_8shard.sh"
)
HELPER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "semtalk_base_inference_core.py"
)
SPEC = importlib.util.spec_from_file_location(
    "final_winner_replication_producer_under_test",
    PRODUCER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
PRODUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCER)


class FinalWinnerReplicationProducerCpuTests(unittest.TestCase):
    def test_neutral_inference_closure_has_no_random_call(self) -> None:
        callables, matches = PRODUCER._callable_closure(HELPER.read_bytes())
        self.assertEqual(matches, [])
        names = {item["qualified_name"] for item in callables}
        self.assertIn(
            "scripts/show_base/semtalk_base_inference_core.py._infer_clip",
            names,
        )
        self.assertEqual(
            callables,
            sorted(callables, key=lambda item: item["qualified_name"]),
        )

    def test_static_randomness_scan_fails_closed(self) -> None:
        source = b"def infer():\n    return np.random.random()\n"
        _callables, matches = PRODUCER._callable_closure(source)
        self.assertEqual(matches, ["np.random.random"])

    def test_test_visibility_and_test_labelled_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            PRODUCER.FinalWinnerReplicationError,
            "test_visible",
        ):
            PRODUCER._reject_tree(
                {"split": "val", "test_visible": True},
                "fixture",
            )
        with self.assertRaises(PRODUCER.FinalWinnerReplicationError):
            PRODUCER._reject_tree(
                {"path": "/frozen/SHOW/test/manifest.jsonl"},
                "fixture",
            )

    def test_process_identity_requires_linux_starttime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / "self").mkdir()
            # /proc/PID/stat fields after the command name start at field 3;
            # index 19 below is Linux field 22 (process starttime).
            tail = ["S", *[str(index) for index in range(1, 19)], "424242"]
            (proc / "self" / "stat").write_text(
                f"123 (python) {' '.join(tail)}\n",
                encoding="utf-8",
            )
            identity = PRODUCER._process_identity(proc)
            self.assertIn("start=424242", identity)
            (proc / "self" / "stat").write_text(
                "123 malformed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                PRODUCER.FinalWinnerReplicationError,
                "malformed",
            ):
                PRODUCER._process_identity(proc)

    def test_cli_has_no_split_or_test_data_switch(self) -> None:
        parser = PRODUCER.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "shard",
                        "--authority",
                        "/formal/authority.json",
                        "--expected-authority-sha256",
                        "1" * 64,
                        "--seed-root",
                        "/formal/seed-0",
                        "--seed",
                        "0",
                        "--num-shards",
                        "8",
                        "--shard-id",
                        "0",
                        "--device",
                        "cuda:0",
                        "--split",
                        "test",
                    ]
                )
        help_text = parser.format_help()
        self.assertNotIn("--split", help_text)
        self.assertNotIn("--test-data", help_text)

    def test_launcher_is_guarded_pinned_and_exactly_8_shards_per_seed(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            '. "$guard_contract"\nsemtalk_require_exact_guarded_runner_all_gpus',
            source,
        )
        self.assertIn(
            "expected_guarded_runner_sha256", source
        )
        self.assertIn("/tmp/globaldiff_guarded_runner.py", source)
        self.assertIn("for shard_id in {0..7}", source)
        self.assertIn('run_seed 0\nrun_seed 15', source)
        self.assertIn("--num-shards 8", source)
        self.assertIn('--device "cuda:$shard_id"', source)
        self.assertIn("--scope final_winner", source)
        self.assertNotIn("pgrep", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("killall", source)

    def test_instrumentation_loads_the_pinned_helper_once(self) -> None:
        source = PRODUCER_PATH.read_text(encoding="utf-8")
        start = source.index("    def instrumented_loader(")
        stop = source.index("\n    inference._load_pinned_helper =", start)
        loader_source = source[start:stop]
        self.assertEqual(loader_source.count("original_loader(pipeline)"), 1)

    def test_seed_finalizer_merges_8_shards_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_root = root / "seed-0"
            seed_root.mkdir()
            output = root / "seed-0-receipt.json"
            authority_artifact = {
                "path": str(root / "authority.json"),
                "sha256": "1" * 64,
                "bytes": 10,
                "receipt_payload_sha256": "2" * 64,
            }
            authority = {
                "preflight": {
                    "path": str(root / "preflight.json"),
                    "sha256": "3" * 64,
                },
                "selected_epoch": 400,
                "selected_checkpoint": {
                    "path": str(root / "winner.bin"),
                    "sha256": "4" * 64,
                    "bytes": 20,
                },
                "subset_manifest": {
                    "path": str(root / "subset.json"),
                    "sha256": "5" * 64,
                    "bytes": 30,
                },
                "source_closure": {"fixture": True},
                "model_bundle": {"fixture": True},
            }

            def shard_fixture(
                _path: Path,
                *,
                authority_artifact: object,
                authority: object,
                seed: int,
                shard_id: int,
            ) -> tuple[dict[str, object], dict[str, object]]:
                del authority_artifact, authority
                clips = [
                    {"gate_position": position, "fixture": position}
                    for position in range(
                        shard_id,
                        PRODUCER.gate.EXPECTED_VAL_CLIPS,
                        PRODUCER.NUM_SHARDS,
                    )
                ]
                return (
                    {
                        "path": str(root / f"shard-{shard_id}.json"),
                        "sha256": f"{shard_id + 10:064x}",
                        "bytes": 100,
                        "receipt_payload_sha256": f"{shard_id + 20:064x}",
                    },
                    {
                        "process_identity": f"host:pid={100 + shard_id}",
                        "clip_count": len(clips),
                        "clips": clips,
                        "inference_shard_receipt": {
                            "path": str(root / f"inference-{shard_id}.json"),
                            "sha256": f"{shard_id + 30:064x}",
                            "bytes": 101,
                            "receipt_payload_sha256": f"{shard_id + 40:064x}",
                        },
                    },
                )

            def inference_fixture(**kwargs: object) -> tuple[dict[str, str], list[int]]:
                shard_id = int(kwargs["shard_id"])
                count = len(
                    range(
                        shard_id,
                        PRODUCER.gate.EXPECTED_VAL_CLIPS,
                        PRODUCER.NUM_SHARDS,
                    )
                )
                return (
                    {
                        "receipt_payload_sha256": f"{shard_id + 40:064x}",
                        "model_receipts_sha256": "a" * 64,
                        "runtime_contract_sha256": "b" * 64,
                    },
                    list(range(count)),
                )

            def artifact_fixture(
                path: Path,
                _label: str,
                **_kwargs: object,
            ) -> dict[str, object]:
                if path == output:
                    payload = path.read_bytes()
                    value = json.loads(payload)
                    return {
                        "path": str(path),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "receipt_payload_sha256": value[
                            "receipt_payload_sha256"
                        ],
                    }
                shard_id = int(path.stem.rsplit("-", 1)[-1])
                return {
                    "path": str(path),
                    "sha256": f"{shard_id + 30:064x}",
                    "bytes": 101,
                    "receipt_payload_sha256": f"{shard_id + 40:064x}",
                }

            args = type(
                "Args",
                (),
                {
                    "seed": 0,
                    "authority": Path(authority_artifact["path"]),
                    "expected_authority_sha256": authority_artifact["sha256"],
                    "seed_root": seed_root,
                    "output_json": output,
                },
            )()
            with (
                mock.patch.object(
                    PRODUCER,
                    "_load_authority",
                    return_value=(authority_artifact, authority),
                ),
                mock.patch.object(
                    PRODUCER.inference,
                    "_preflight_artifact",
                    return_value=(
                        {"path": authority["preflight"]["path"]},
                        {
                            "candidate_bundle": {
                                "candidates": {
                                    "400": authority[
                                        "selected_checkpoint"
                                    ]
                                }
                            }
                        },
                    ),
                ),
                mock.patch.object(
                    PRODUCER.inference,
                    "_load_preflight_children",
                    return_value=({}, {}, [{}] * 1715, {}),
                ),
                mock.patch.object(
                    PRODUCER,
                    "_load_shard",
                    side_effect=shard_fixture,
                ),
                mock.patch.object(
                    PRODUCER.inference,
                    "_validate_shard",
                    side_effect=inference_fixture,
                ),
                mock.patch.object(
                    PRODUCER,
                    "_artifact",
                    side_effect=artifact_fixture,
                ),
                mock.patch.object(
                    PRODUCER,
                    "_process_identity",
                    return_value="host:pid=999:start=42",
                ),
                mock.patch.object(PRODUCER.gate, "_validate_seed_run"),
            ):
                artifact = PRODUCER.finalize_seed(args)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["path"], str(output))
            self.assertEqual(len(receipt["clips"]), 1715)
            self.assertEqual(
                [item["gate_position"] for item in receipt["clips"]],
                list(range(1715)),
            )
            self.assertEqual(receipt["seed"], 0)
            self.assertEqual(
                receipt["runtime"]["devices"],
                [f"cuda:{index}" for index in range(8)],
            )

    def test_launcher_and_producer_cpu_entrypoints_parse(self) -> None:
        shell = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(shell.returncode, 0, shell.stderr)
        producer = subprocess.run(
            ["python3", str(PRODUCER_PATH), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(producer.returncode, 0, producer.stderr)
        self.assertIn("finalize-seed", producer.stdout)
        launcher = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        self.assertIn("expected-guarded-runner-sha256", launcher.stdout)


if __name__ == "__main__":
    unittest.main()
