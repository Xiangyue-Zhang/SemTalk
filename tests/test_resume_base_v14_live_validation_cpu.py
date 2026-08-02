#!/usr/bin/env python3
"""CPU-only checks for the narrow pre-authorization V14 resume tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resume = load("resume_v14", ROOT / "scripts/show_base/resume_base_v14_live_validation.py")
fixtures = load("v14_live_test_fixtures", ROOT / "tests/test_supervise_base_v14_live_validation_cpu.py")
sup = fixtures.sup


class OrphanClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = fixtures.Fixture()
        self.campaign = self.fx.campaign()
        self.job = self.fx.jobs[0]
        fixtures.write_json(Path(self.job["candidate_receipt_path"]), {"epoch": 1})
        sup._campaign_claim_v2(self.campaign, lambda: 1.0)
        self.active = sup._active_v2(self.campaign, "run-next", lambda: 2.0)
        candidate = sup._candidate_receipt_artifact(self.job)
        authorize = sup._authorize_argv(self.campaign, self.job)
        self.claim = sup.write_new_json(
            sup._job_claim_path_v2(self.campaign, 1),
            sup._job_claim_v2(self.campaign, self.job, candidate, authorize, 3.0),
            "test job claim",
        )

    def tearDown(self) -> None:
        self.fx.close()

    def test_accepts_only_claimed_pre_authorization_head(self) -> None:
        head, candidate, active, claim = resume._validate_orphaned_claim(sup, self.campaign)
        self.assertEqual(head["epoch"], 1)
        self.assertEqual(candidate["path"], self.job["candidate_receipt_path"])
        self.assertEqual(active, self.active)
        self.assertEqual(claim, self.claim)

    def test_rejects_any_post_claim_output(self) -> None:
        fixtures.write_json(Path(self.job["authority_path"]), {"candidate_epoch": 1})
        with self.assertRaisesRegex(resume.RecoveryError, "work authority already exists"):
            resume._validate_orphaned_claim(sup, self.campaign)

    def test_rejects_changed_active_claim(self) -> None:
        path = self.fx.state / "active_invocation.claim.json"
        value = sup.strict_json(path.read_bytes(), "active")
        path.chmod(0o600)
        value["operation"] = "finalize"
        path.write_bytes(sup.canonical_json_bytes(value))
        path.chmod(0o400)
        with self.assertRaises(sup.SupervisorError):
            resume._validate_orphaned_claim(sup, self.campaign)


if __name__ == "__main__":
    unittest.main()
