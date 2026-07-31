from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BaseFinalLauncherProtocolTests(unittest.TestCase):
    def test_formal_entrypoints_expose_only_diffsheg_test_evaluation(self) -> None:
        launcher = (
            ROOT / "scripts" / "show_base" / "run_base_final_test.sh"
        ).read_text(encoding="utf-8")
        consumer = (
            ROOT / "scripts" / "show_base" / "run_base_final_test.py"
        ).read_text(encoding="utf-8")
        authority = (
            ROOT / "scripts" / "show_base" / "base_final_authority.py"
        ).read_text(encoding="utf-8")

        forbidden = (
            "evaluate_talkshow_show_metrics",
            "talkshow_metrics",
            "released2",
            "paper16",
        )
        for token in forbidden:
            self.assertNotIn(token, launcher.lower())
            self.assertNotIn(token, consumer.lower())
            self.assertNotIn(token, authority.lower())

        self.assertNotIn('add_parser("seal"', consumer)
        self.assertNotIn('args.command == "seal"', consumer)
        self.assertIn("evaluate_diffsheg_final_test.py", launcher)
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
