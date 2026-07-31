from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest

from scripts.show_base import base_final_authority as authority
from scripts.show_base import prepare_base_final_authority_inputs as producer


class PrepareBaseFinalAuthorityInputsCpuTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "source"
        repository = Path(__file__).resolve().parents[1]
        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "--shared",
                "--no-checkout",
                "--no-tags",
                str(repository),
                str(source),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "-q", "--detach", "HEAD"],
            cwd=source,
            check=True,
        )
        for reference in subprocess.check_output(
            [
                "git",
                "for-each-ref",
                "--format=%(refname)",
                "refs/heads",
            ],
            cwd=source,
            text=True,
        ).splitlines():
            subprocess.run(
                ["git", "update-ref", "-d", reference],
                cwd=source,
                check=True,
            )
        subprocess.run(
            ["git", "remote", "set-url", "origin", authority.ORIGIN],
            cwd=source,
            check=True,
        )
        return source.resolve()

    def _fixture(self, root: Path) -> SimpleNamespace:
        inputs = root / "inputs"
        inputs.mkdir()

        def create(name: str) -> Path:
            path = (inputs / name).resolve()
            path.write_bytes(f"fixture:{name}\n".encode())
            return path

        audio = []
        for shard_id in range(authority.NUM_SHARDS):
            audio.append(
                [
                    str(shard_id),
                    str(create(f"audio-{shard_id}.jsonl")),
                    str(create(f"audio-{shard_id}-summary.json")),
                    str(create(f"audio-{shard_id}-lineage.json")),
                ]
            )
        checkpoints = [
            [stage, str(create(f"{stage}.bin"))]
            for stage in authority.CHECKPOINT_STAGES
        ]
        return SimpleNamespace(
            expected_output_root=(root / "formal" / "final").resolve(),
            canonical_manifest=create("canonical.jsonl"),
            canonical_summary=create("canonical-summary.json"),
            canonical_lineage=create("canonical-lineage.json"),
            audio_shard=audio,
            base_long_manifest=create("base-long-manifest.json"),
            base_long_status=create("base-long-status.json"),
            base_long_frozen_inputs=create("base-long-frozen.json"),
            winner_selection=create("winner.json"),
            continuation_decision=create("stop.json"),
            winner_validation_metric_closure=create("diffsheg-val.json"),
            test_claim=create("test-claim.json"),
            inference_source_root=self._source(root),
            checkpoint=checkpoints,
            output_json=(root / "authority-inputs.json").resolve(),
        )

    def test_builds_exact_complete_input_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args = self._fixture(root)
            value = producer.build_inputs(args)
            self.assertEqual(value["format"], authority.INPUTS_FORMAT)
            self.assertEqual(value["continuation_waves"], [])
            self.assertEqual(
                [item["shard_id"] for item in value["audio_authorities"]],
                list(range(authority.NUM_SHARDS)),
            )
            self.assertEqual(
                set(value["checkpoints"]),
                set(authority.CHECKPOINT_STAGES),
            )
            self.assertEqual(
                value["receipt_payload_sha256"],
                authority.canonical_json_sha256(
                    {
                        key: item
                        for key, item in value.items()
                        if key != "receipt_payload_sha256"
                    }
                ),
            )
            source = value["inference_source"]
            self.assertEqual(source["origin"], authority.ORIGIN)
            self.assertIs(source["clean"], True)

    def test_rejects_missing_shard_and_symlinked_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args = self._fixture(root)
            args.audio_shard.pop()
            with self.assertRaisesRegex(producer.AuthorityInputsError, "coverage"):
                producer.build_inputs(args)
            second = root / "second"
            second.mkdir()
            args = self._fixture(second)
            original = args.winner_selection
            linked = original.parent / "winner-link.json"
            linked.symlink_to(original)
            args.winner_selection = linked
            with self.assertRaisesRegex(producer.AuthorityInputsError, "canonical"):
                producer.build_inputs(args)


if __name__ == "__main__":
    unittest.main()
