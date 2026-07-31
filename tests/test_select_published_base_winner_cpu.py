from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from tests.test_published_test_winner_claim_cpu import (
    ClaimFixture,
    _write_json,
    _write_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "show_base" / "select_published_base_winner.py"
SPEC = importlib.util.spec_from_file_location(
    "select_published_base_winner_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)
AUTHORITY = SELECTOR.authority


class SelectorFixture:
    def __init__(self, root: Path) -> None:
        self.base = ClaimFixture(root)
        self.evidence_payload = {
            "format": SELECTOR.CANDIDATE_EVIDENCE_FORMAT,
            "payload_hash_algorithm": AUTHORITY.PAYLOAD_HASH_ALGORITHM,
            "status": "complete",
            "generator": "SemTalk Base Motion Generation",
            "dataset": "SHOW",
            "target_speaker_scope": AUTHORITY.EXPECTED_SCOPE,
            "split": "val",
            "test_visible": False,
            "candidate_epochs": list(AUTHORITY.BASE_CANDIDATE_EPOCHS),
            "updates_per_epoch": AUTHORITY.BASE_UPDATES_PER_EPOCH,
            "prerequisite_selection": self.base.prerequisite_artifact,
            "continuation_decision": self.base.continuation_artifact,
            "real_feature_cache": self.base.real_feature_cache,
            "candidates": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "body_released2_fgd"
                }
                for row in self.base.rows
            ],
        }
        self.evidence_artifact = _write_receipt(
            root / "candidate-evidence.json", self.evidence_payload
        )

    def rebuild_evidence(self) -> None:
        self.evidence_artifact = _write_receipt(
            Path(self.evidence_artifact["path"]), self.evidence_payload
        )

    def build(self) -> dict[str, object]:
        adapter = self.base.adapter()
        with mock.patch.object(
            AUTHORITY,
            "_fresh_local_module",
            side_effect=lambda name: self.base.source_module(name, adapter),
        ):
            return SELECTOR.build_published_base_winner_selection(
                self.evidence_artifact,
                prerequisite_selection=self.base.prerequisite_artifact,
                continuation_decision=self.base.continuation_artifact,
            )

    def rewrite_report_score(
        self,
        row_index: int,
        score: float,
        *,
        replay_score: float | None = None,
    ) -> None:
        row = self.evidence_payload["candidates"][row_index]
        report_path = Path(row["talkshow_metric_report"]["path"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["body"]["released2"]["metrics"]["FGD"] = score
        report.pop("report_payload_sha256")
        report["report_payload_sha256"] = AUTHORITY.canonical_json_sha256(
            report, newline=True
        )
        row["talkshow_metric_report"] = _write_json(report_path, report)
        replay_path = Path(row["primary_replay_receipt"]["path"])
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay["report_payload_sha256"] = report[
            "report_payload_sha256"
        ]
        if replay_score is not None:
            replay["primary_metric"] = replay_score
        row["primary_replay_receipt"] = _write_receipt(
            replay_path, replay
        )
        self.rebuild_evidence()


class PublishedBaseSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SelectorFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_fresh_replayed_released2_fgd_winner(self) -> None:
        result = self.fixture.build()
        self.assertEqual(result["format"], AUTHORITY.WINNER_SELECTION_FORMAT)
        self.assertEqual(result["primary_metric"], AUTHORITY.PRIMARY_METRIC)
        self.assertEqual(
            result["selection_policy"]["ordering"],
            [AUTHORITY.PRIMARY_METRIC, "epoch", "optimizer_updates"],
        )
        self.assertEqual(
            result["selected"]["epoch"],
            AUTHORITY.BASE_CANDIDATE_EPOCHS[5],
        )
        self.assertEqual(
            len(result["candidates"]),
            len(AUTHORITY.BASE_CANDIDATE_EPOCHS),
        )
        self.assertEqual(
            self.fixture.base.adapter_calls,
            len(AUTHORITY.BASE_CANDIDATE_EPOCHS),
        )

    def test_output_round_trips_through_claim_validator(self) -> None:
        selection = self.fixture.build()
        artifact = _write_json(
            Path(self.temporary.name) / "selection.json", selection
        )
        artifact["receipt_payload_sha256"] = selection[
            "receipt_payload_sha256"
        ]
        adapter = self.fixture.base.adapter()
        with mock.patch.object(
            AUTHORITY,
            "_fresh_local_module",
            side_effect=lambda name: self.fixture.base.source_module(
                name,
                adapter,
            ),
        ):
            normalized, replayed, winner = AUTHORITY._validate_winner_selection(
                artifact,
                prerequisite_artifact=self.fixture.base.prerequisite_artifact,
                continuation_artifact=self.fixture.base.continuation_artifact,
                expected_real_feature_cache=(
                    self.fixture.base.real_feature_cache
                ),
            )
        self.assertEqual(normalized, artifact)
        self.assertEqual(replayed, selection)
        self.assertEqual(winner, selection["selected"])

    def test_tie_breaks_by_epoch_then_optimizer_updates(self) -> None:
        self.fixture.rewrite_report_score(0, 0.1, replay_score=0.1)
        self.fixture.rewrite_report_score(1, 0.1, replay_score=0.1)
        result = self.fixture.build()
        self.assertEqual(result["selected"]["epoch"], 1)
        self.assertEqual(
            result["selected"]["optimizer_updates"],
            AUTHORITY.BASE_UPDATES_PER_EPOCH,
        )

    def test_report_self_declared_fgd_never_controls_ranking(self) -> None:
        self.fixture.rewrite_report_score(0, 0.01)
        result = self.fixture.build()
        self.assertEqual(
            result["selected"]["epoch"],
            AUTHORITY.BASE_CANDIDATE_EPOCHS[5],
        )
        self.assertEqual(
            result["candidates"][0]["body_released2_fgd"],
            10.0,
        )

    def test_candidate_schedule_change_fails(self) -> None:
        self.fixture.evidence_payload["candidate_epochs"][-1] = 401
        self.fixture.rebuild_evidence()
        with self.assertRaisesRegex(
            SELECTOR.PublishedBaseSelectionError,
            "identity changed",
        ):
            self.fixture.build()

    def test_candidate_evidence_file_tampering_fails(self) -> None:
        Path(self.fixture.evidence_artifact["path"]).write_bytes(b"{}\n")
        with self.assertRaisesRegex(
            SELECTOR.PublishedBaseSelectionError,
            "artifact changed",
        ):
            self.fixture.build()

    def test_foreign_generator_fails(self) -> None:
        self.fixture.evidence_payload["generator"] = "Foreign Base"
        self.fixture.rebuild_evidence()
        with self.assertRaisesRegex(
            SELECTOR.PublishedBaseSelectionError,
            "identity changed",
        ):
            self.fixture.build()

    def test_metric_adapter_failure_has_no_fallback(self) -> None:
        adapter = SimpleNamespace(
            validate_report=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("primitive replay rejected")
            )
        )
        with mock.patch.object(
            AUTHORITY,
            "_fresh_local_module",
            side_effect=lambda name: self.fixture.base.source_module(
                name,
                adapter,
            ),
        ), self.assertRaisesRegex(
            SELECTOR.PublishedBaseSelectionError,
            "neutral TalkSHOW report replay failed",
        ):
            SELECTOR.build_published_base_winner_selection(
                self.fixture.evidence_artifact,
                prerequisite_selection=self.fixture.base.prerequisite_artifact,
                continuation_decision=self.fixture.base.continuation_artifact,
            )

    def test_candidate_cannot_self_declare_real_feature_cache(self) -> None:
        replacement_cache = _write_receipt(
            Path(self.temporary.name) / "replacement-cache.json",
            {
                "format": "semtalk_show_released2_real_feature_cache_v1",
                "status": "complete",
                "feature_count": 1715,
            },
        )
        row = self.fixture.evidence_payload["candidates"][0]
        replay_path = Path(row["primary_replay_receipt"]["path"])
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay["real_feature_cache"] = replacement_cache
        row["primary_replay_receipt"] = _write_receipt(
            replay_path, replay
        )
        self.fixture.rebuild_evidence()
        with self.assertRaisesRegex(
            SELECTOR.PublishedBaseSelectionError,
            "fresh replay authority changed",
        ):
            self.fixture.build()

    def test_missing_or_reused_candidate_evidence_fails(self) -> None:
        for index, message in enumerate(
            ("exactly 22", "checkpoint path was reused")
        ):
            with self.subTest(case=index):
                fixture = SelectorFixture(
                    Path(self.temporary.name) / f"case-{index}"
                )
                if index == 0:
                    fixture.evidence_payload["candidates"].pop()
                else:
                    fixture.evidence_payload["candidates"][1][
                        "candidate_checkpoint"
                    ] = fixture.evidence_payload["candidates"][0][
                        "candidate_checkpoint"
                    ]
                fixture.rebuild_evidence()
                with self.assertRaisesRegex(
                    SELECTOR.PublishedBaseSelectionError, message
                ):
                    fixture.build()


if __name__ == "__main__":
    unittest.main()
