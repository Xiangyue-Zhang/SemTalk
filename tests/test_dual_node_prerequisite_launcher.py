from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "run_five_prerequisites.sh"
)
TRAINING_ENTRYPOINT = REPOSITORY / "show_base_train.py"


class DualNodePrerequisiteLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.training_source = TRAINING_ENTRYPOINT.read_text(
            encoding="utf-8"
        )

    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(
            ["/usr/bin/env", "bash", "-n", str(LAUNCHER)],
            check=True,
        )

    def test_partition_is_explicit_and_fail_closed(self) -> None:
        self.assertIn(
            "SEMTALK_FORMAL_PARTITION must be exactly master or worker",
            self.source,
        )
        self.assertIn("active_stages=(face hands global)", self.source)
        self.assertIn("active_stages=(upper lower)", self.source)
        self.assertIn(
            'for stage in "${active_stages[@]}"',
            self.source,
        )

    def test_single_node_w4_partition_is_exact(self) -> None:
        expected_launch_fragments = (
            "face 0,1,2,3 29611",
            "hands 4,5,6,7 29612",
            "upper 0,1,2,3 29613",
            "lower 4,5,6,7 29614",
        )
        for fragment in expected_launch_fragments:
            self.assertIn(fragment, self.source)
        self.assertIn("show_ft_face_200.bin 200 disabled", self.source)
        self.assertIn("show_ft_hands_200.bin 200 disabled", self.source)
        self.assertIn("show_ft_upper_200.bin 200 disabled", self.source)
        self.assertIn("show_ft_lower_200.bin 200 disabled", self.source)
        self.assertIn("show_ft_global_200.bin 200 disabled", self.source)
        self.assertIn(
            'local expected_world_size=4',
            self.source,
        )
        self.assertIn(
            '--nproc_per_node="${#stage_gpus[@]}"',
            self.source,
        )
        self.assertIn(
            '--batch_size "$((256 / ${#stage_gpus[@]}))"',
            self.source,
        )
        self.assertIn("--global_batch_size 256", self.source)
        self.assertIn("updates = entries // 256", self.source)
        self.assertIn('global "$global_gpu" 29615', self.source)
        self.assertIn('global_gpu=0', self.source)
        self.assertIn('global_gpu=4', self.source)
        self.assertIn('"$name" == face || "$name" == hands', self.source)
        self.assertIn('global_launched=false', self.source)

    def test_ddp_jobs_forbid_detached_pool_helpers(self) -> None:
        self.assertNotIn("POOL_GATE_REPORT", self.source)
        self.assertNotIn("target_offload", self.source)
        self.assertIn(
            "--smplx_training_pool_mode disabled",
            self.source,
        )

    def test_first_round_and_offline_candidate_cadence_are_locked(self) -> None:
        self.assertIn(
            "REPRESENTATION_CANDIDATE_INTERVAL_EPOCHS = 20",
            self.training_source,
        )
        self.assertIn(
            '"selection_status": "offline_validation_pending"',
            self.training_source,
        )
        for final_name in (
            "show_ft_face_200.bin",
            "show_ft_hands_200.bin",
            "show_ft_upper_200.bin",
            "show_ft_lower_200.bin",
            "show_ft_global_200.bin",
        ):
            self.assertIn(
                f'"final_ckpt_name": "{final_name}"',
                self.training_source,
            )
        self.assertEqual(self.training_source.count('"epochs": 200,'), 5)

    def test_scope_is_show_all_base_without_sparse_generation(self) -> None:
        self.assertIn("--training_speakers 0 1 2 3", self.source)
        self.assertIn("--sparse 0", self.source)
        self.assertIn("load_official_model_state(", self.training_source)
        self.assertNotIn(
            "freeze_face_for_decoder_transfer",
            self.training_source,
        )
        self.assertNotIn("requires_grad_(False)", self.training_source)
        self.assertNotIn("speaker2", self.source)
        self.assertNotIn("SemGate", self.source)
        self.assertNotIn("--use_lower_target_joints_cache true", self.source)

    def test_frozen_representation_and_trainer_sources_are_separate(self) -> None:
        self.assertIn(
            "invalid representation producer source receipt",
            self.training_source,
        )
        self.assertIn('"input_source_binding"', self.training_source)
        self.assertNotIn(
            "representation/training source receipt mismatch",
            self.training_source,
        )


if __name__ == "__main__":
    unittest.main()
