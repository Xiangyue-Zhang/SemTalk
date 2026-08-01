from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BaseFinalLauncherProtocolTests(unittest.TestCase):
    def test_formal_entrypoints_expose_one_combined_test_evaluation(self) -> None:
        launcher = (
            ROOT / "scripts" / "show_base" / "run_base_final_test.sh"
        ).read_text(encoding="utf-8")
        consumer = (
            ROOT / "scripts" / "show_base" / "run_base_final_test.py"
        ).read_text(encoding="utf-8")
        authority = (
            ROOT / "scripts" / "show_base" / "base_final_authority.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn('add_parser("seal"', consumer)
        self.assertNotIn('args.command == "seal"', consumer)
        self.assertIn("evaluate_diffsheg_final_test.py", launcher)
        # TalkSHOW is called in-process by the sole combined producer; the
        # shell must never launch a second metric evaluator.
        self.assertNotIn("evaluate_talkshow_show_metrics.py", launcher)
        self.assertIn("sole formal test-metric process", launcher)
        self.assertIn("both required", launcher)
        self.assertIn("released2", authority.lower())
        self.assertIn("paper16", authority.lower())
        self.assertIn("paspa_diffsheg_show_seven", authority)
        self.assertIn("talkshow_show_body_face", authority)
        self.assertIn("final_metric_event", consumer)
        self.assertIn("prepare_diffsheg_audio_view.py", launcher)
        self.assertLess(
            launcher.index('"$adapter" finalize'),
            launcher.index('"$python_bin" "$audio_view_builder"'),
        )
        self.assertLess(
            launcher.index('"$python_bin" "$audio_view_builder"'),
            launcher.index("--preflight-only"),
        )
        self.assertIn(
            '--source-audio-root "$diffsheg_audio_view"',
            launcher,
        )
        self.assertIn("--talkshow-validation-gate-json", launcher)
        self.assertNotIn("validation_gate", consumer)
        self.assertNotIn('add_parser("distribution"', consumer)
        self.assertEqual(
            launcher.count(
                'CUDA_VISIBLE_DEVICES=0 "$python_bin" "$evaluator"'
            ),
            1,
        )
        self.assertIn("--preflight-only", launcher)
        self.assertIn("--expected-preflight-sha256", launcher)

    def test_authority_has_only_the_long_validation_selection_protocol(self) -> None:
        authority = (
            ROOT / "scripts" / "show_base" / "base_final_authority.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'BASE_SELECTION_PROTOCOL = "diffsheg_show_validation_fgd_v1"',
            authority,
        )
        self.assertIn(
            'BASE_SELECTION_METRIC = "validation.diffsheg.metrics.fgd"',
            authority,
        )
        self.assertIn(
            '"semtalk_show_base_official_adapt_long_selection_v1"',
            authority,
        )


if __name__ == "__main__":
    unittest.main()
