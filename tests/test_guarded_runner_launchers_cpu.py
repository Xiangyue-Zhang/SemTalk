from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY / "scripts" / "show_base" / "guarded_runner_contract.sh"
RUNNER = "/tmp/globaldiff_guarded_runner.py"
GPUS = "0,1,2,3,4,5,6,7"


class GuardedRunnerArgvContractTests(unittest.TestCase):
    def _accepted(self, *argv: str) -> bool:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; shift; _semtalk_guarded_runner_argv_is_exact "$@"',
                "guard-contract-test",
                str(HELPER),
                *argv,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def test_exact_runner_forms_are_accepted(self) -> None:
        self.assertTrue(
            self._accepted(
                "/usr/bin/python3",
                RUNNER,
                "--gpus",
                GPUS,
                "--",
                "/bin/bash",
                "/formal/launcher.sh",
            )
        )
        self.assertTrue(
            self._accepted(
                "/opt/conda/bin/python3.12",
                RUNNER,
                f"--gpus={GPUS}",
                "--status",
                "/formal/status.json",
                "--",
                "/bin/bash",
                "/formal/launcher.sh",
                "--gpus",
                "workload-value-is-inert",
            )
        )

    def test_false_parent_and_inert_runner_tokens_are_rejected(self) -> None:
        self.assertFalse(
            self._accepted(
                "/usr/bin/python3",
                "/tmp/not-the-runner.py",
                "--note",
                RUNNER,
                "--gpus",
                GPUS,
                "--",
                "/bin/bash",
            )
        )
        self.assertFalse(
            self._accepted(
                "/bin/bash",
                RUNNER,
                "--gpus",
                GPUS,
                "--",
                "/bin/bash",
            )
        )
        self.assertFalse(
            self._accepted(
                "/usr/bin/python3",
                RUNNER,
                "--note",
                RUNNER,
                "--gpus",
                GPUS,
                "--",
                "/bin/bash",
            )
        )

    def test_gpu_authority_must_be_unique_and_before_delimiter(self) -> None:
        self.assertFalse(
            self._accepted(
                "/usr/bin/python3",
                RUNNER,
                "--",
                "/bin/bash",
                "--gpus",
                GPUS,
            )
        )
        self.assertFalse(
            self._accepted(
                "/usr/bin/python3",
                RUNNER,
                "--gpus",
                GPUS,
                "--gpus=7",
                "--",
                "/bin/bash",
            )
        )
        self.assertFalse(
            self._accepted(
                "/usr/bin/python3",
                RUNNER,
                "--gpus=0,1,2,3,4,5,6",
                "--",
                "/bin/bash",
            )
        )
        self.assertFalse(
            self._accepted(
                "/usr/bin/python3",
                RUNNER,
                "--gpus",
                GPUS,
                "/bin/bash",
            )
        )

    def test_every_current_formal_gpu_launcher_calls_shared_guard(self) -> None:
        directory = REPOSITORY / "scripts" / "show_base"
        gpu_markers = (
            "CUDA_VISIBLE_DEVICES",
            "torch.distributed.run",
            "run_base_inference.py",
            "run_lower_target_cache_builder.py",
        )
        launchers = []
        for path in sorted(directory.glob("run_*.sh")):
            source = path.read_text(encoding="utf-8")
            if any(marker in source for marker in gpu_markers):
                launchers.append(path)
                self.assertIn(
                    '. "$launcher_dir/guarded_runner_contract.sh"',
                    source,
                    path.name,
                )
                self.assertIn(
                    "semtalk_require_exact_guarded_runner_all_gpus",
                    source,
                    path.name,
                )
        self.assertEqual(
            {path.name for path in launchers},
            {
                "run_audio_cache_8shard.sh",
                "run_base_formal_inference.sh",
                "run_base_official_adapt_long.sh",
                "run_base_released_all_speakers_inference.sh",
                "run_base_training.sh",
                "run_canonical_cache_8shard.sh",
                "run_five_prerequisites.sh",
                "run_lower_target_cache_builder_guarded.sh",
                "run_prerequisite_val_8shard.sh",
            },
        )


if __name__ == "__main__":
    unittest.main()
