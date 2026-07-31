from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest import mock

from scripts.show_base import base_final_authority as AUTH
from scripts.show_base import base_long_val_contract as LONG
import scripts.show_base as SHOW_BASE_PACKAGE


def artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def write_json(path: Path, value: object) -> None:
    path.write_bytes(AUTH.canonical_json_bytes(value))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(AUTH.canonical_json_bytes(row) for row in rows)
    )


def write_wave(
    path: Path,
    *,
    boundary: int,
    predecessor: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    unsigned: dict[str, object] = {
        "format": "semtalk_show_prerequisite_continuation_wave_v1",
        "status": "authorized",
        "test_visible": False,
        "decision": {
            "path": str((path.parent / f"continue-e{boundary}.json").resolve()),
            "sha256": hashlib.sha256(
                f"continue-file-{boundary}".encode()
            ).hexdigest(),
            "receipt_payload_sha256": hashlib.sha256(
                f"continue-payload-{boundary}".encode()
            ).hexdigest(),
        },
        "trigger_stages": ["face"],
        "boundary_epoch": boundary,
        "target_epoch": boundary + 20,
        "stages": [
            {
                "stage": stage,
                "old_segment": {
                    "predecessor_wave": copy.deepcopy(predecessor)
                },
            }
            for stage in AUTH.REPRESENTATION_STAGES
        ],
    }
    receipt = {
        **unsigned,
        "receipt_payload_sha256": AUTH.canonical_json_sha256(unsigned),
    }
    write_json(path, receipt)
    binding = {
        **artifact(path),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }
    return binding, receipt


class AuthorityFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source"
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
                str(self.source),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "-q", "--detach", "HEAD"],
            cwd=self.source,
            check=True,
        )
        local_refs = subprocess.check_output(
            [
                "git",
                "for-each-ref",
                "--format=%(refname)",
                "refs/heads",
            ],
            cwd=self.source,
            text=True,
        ).splitlines()
        for reference in local_refs:
            subprocess.run(
                ["git", "update-ref", "-d", reference],
                cwd=self.source,
                check=True,
            )
        subprocess.run(
            ["git", "remote", "set-url", "origin", AUTH.ORIGIN],
            cwd=self.source,
            check=True,
        )
        entrypoint = self.source / "scripts/show_base/run_base_inference.py"
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=self.source,
            text=True,
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=self.source,
            text=True,
        ).strip()
        self.inference_source = {
            "source_root": str(self.source.resolve()),
            "origin": AUTH.ORIGIN,
            "commit": commit,
            "tree": tree,
            "clean": True,
            "entrypoint": "scripts/show_base/run_base_inference.py",
            "entrypoint_sha256": hashlib.sha256(
                entrypoint.read_bytes()
            ).hexdigest(),
        }
        self.canonical_manifest = root / "canonical.jsonl"
        self.canonical_summary = root / "canonical-summary.json"
        self.canonical_lineage = root / "canonical-lineage.json"
        contract = {
            "source_receipt": {
                "origin": AUTH.ORIGIN,
                "commit": commit,
                "tree": tree,
            }
        }
        contract_sha = AUTH.canonical_json_sha256(contract)
        self.canonical_rows: list[dict[str, object]] = []
        speakers = tuple(AUTH.SHOW_SPEAKER_IDS)
        for offset, global_index in enumerate(
            range(AUTH.TEST_GLOBAL_START, AUTH.TEST_GLOBAL_STOP)
        ):
            speaker = speakers[offset % len(speakers)]
            self.canonical_rows.append(
                {
                    "global_index": global_index,
                    "clip_id": f"{speaker}/video/clip{offset:04d}",
                    "split": "test",
                    "speaker": speaker,
                    "speaker_id": AUTH.SHOW_SPEAKER_IDS[speaker],
                    "pose_fps": AUTH.POSE_FPS,
                    "frames": 64 + offset % 3,
                    "canonical_npz_sha256": (
                        hashlib.sha256(
                            f"npz-{offset}".encode()
                        ).hexdigest()
                    ),
                    "source_wav_sha256": (
                        hashlib.sha256(
                            f"wav-{offset}".encode()
                        ).hexdigest()
                    ),
                    "lineage_contract_sha256": contract_sha,
                }
            )
        write_jsonl(self.canonical_manifest, self.canonical_rows)
        lineage = {
            "final_manifest_sha256": artifact(
                self.canonical_manifest
            )["sha256"],
            "lineage_contract": contract,
            "lineage_contract_sha256": contract_sha,
        }
        write_json(self.canonical_lineage, lineage)
        summary = {
            "status": "complete",
            "manifest_sha256": artifact(
                self.canonical_manifest
            )["sha256"],
            "finite": True,
            "exact_once": True,
            "split_disjoint": True,
            "lineage_contract_sha256": contract_sha,
            "lineage_sha256": AUTH.canonical_json_sha256(lineage),
            "source_receipt_sha256": AUTH.canonical_json_sha256(
                contract["source_receipt"]
            ),
        }
        write_json(self.canonical_summary, summary)
        self.canonical_root_receipt = {
            "manifest": artifact(self.canonical_manifest),
            "summary": artifact(self.canonical_summary),
            "lineage": artifact(self.canonical_lineage),
            "source_commit": commit,
            "source_tree": tree,
        }
        self.audio: list[dict[str, object]] = []
        canonical_receipt = {
            "manifest": str(self.canonical_manifest.resolve()),
        }
        audio_features = root / "audio-features"
        audio_features.mkdir()
        self.audio_feature_paths: list[Path] = []
        for shard_id in range(AUTH.NUM_SHARDS):
            rows = []
            for offset, canonical in enumerate(self.canonical_rows):
                if offset % AUTH.NUM_SHARDS != shard_id:
                    continue
                feature_path = audio_features / f"{offset:04d}.npz"
                feature_path.write_bytes(
                    f"feature-{offset}".encode("utf-8")
                )
                self.audio_feature_paths.append(feature_path)
                rows.append(
                    {
                        "split": "test",
                        "clip_id": canonical["clip_id"],
                        "shard_id": shard_id,
                        "num_shards": AUTH.NUM_SHARDS,
                        "canonical_npz_sha256": canonical[
                            "canonical_npz_sha256"
                        ],
                        "source_wav_sha256": canonical[
                            "source_wav_sha256"
                        ],
                        "frames": canonical["frames"],
                        "lineage_contract_sha256": contract_sha,
                        "audio_feature_npz": str(feature_path.resolve()),
                        "audio_feature_npz_sha256": hashlib.sha256(
                            feature_path.read_bytes()
                        ).hexdigest(),
                    }
                )
            manifest_path = root / f"audio-{shard_id}.jsonl"
            lineage_path = root / f"audio-{shard_id}-lineage.json"
            summary_path = root / f"audio-{shard_id}-summary.json"
            write_jsonl(manifest_path, rows)
            audio_lineage = {
                "status": "complete",
                "output_manifest_sha256": artifact(
                    manifest_path
                )["sha256"],
                "canonical_manifest_sha256": {
                    str(self.canonical_manifest.resolve()): artifact(
                        self.canonical_manifest
                    )["sha256"],
                },
                "canonical_receipt": canonical_receipt,
                "lineage_contract_sha256": contract_sha,
                "shard_id": shard_id,
                "num_shards": AUTH.NUM_SHARDS,
                "source_receipt": {
                    "origin": AUTH.ORIGIN,
                    "commit": commit,
                    "tree": tree,
                },
                "runtime": {
                    "python": "fixture",
                    "librosa": "fixture",
                    "soundfile": "fixture",
                    "soxr": "fixture",
                },
                "hubert_model_tree_sha256": hashlib.sha256(
                    b"hubert"
                ).hexdigest(),
                "shard_clips": len(rows),
            }
            write_json(lineage_path, audio_lineage)
            audio_summary = {
                "status": "complete",
                "output_manifest_sha256": artifact(
                    manifest_path
                )["sha256"],
                "lineage_json_sha256": artifact(
                    lineage_path
                )["sha256"],
            }
            write_json(summary_path, audio_summary)
            self.audio.append(
                {
                    "shard_id": shard_id,
                    "manifest": artifact(manifest_path),
                    "summary": artifact(summary_path),
                    "lineage": artifact(lineage_path),
                }
            )
        self.checkpoints: dict[str, dict[str, object]] = {}
        for stage in AUTH.CHECKPOINT_STAGES:
            checkpoint = root / f"{stage}.bin"
            checkpoint.write_bytes(f"{stage}-checkpoint".encode())
            self.checkpoints[stage] = artifact(checkpoint)
        base_manifest = root / "base-long-manifest.json"
        base_status = root / "base-long-status.json"
        base_frozen = root / "base-long-frozen.json"
        write_json(base_manifest, {"kind": "base-long-manifest"})
        write_json(base_status, {"kind": "base-long-status"})
        write_json(base_frozen, {"kind": "base-long-frozen"})
        self.base_long_candidate_artifacts = {
            "manifest": artifact(base_manifest),
            "status": artifact(base_status),
            "frozen_inputs": artifact(base_frozen),
        }
        long_trainer = (
            self.source
            / "scripts"
            / "show_base"
            / "train_base_official_adapt_long.py"
        )
        candidate_rows: list[dict[str, object]] = []
        for epoch in LONG.EXPECTED_CANDIDATE_EPOCHS:
            if epoch == 400:
                checkpoint = self.checkpoints["base"]
            else:
                candidate_path = root / f"base-e{epoch:04d}.bin"
                candidate_path.write_bytes(
                    f"base-candidate-{epoch}".encode("ascii")
                )
                checkpoint = artifact(candidate_path)
            candidate_rows.append(
                {
                    "epoch": epoch,
                    "optimizer_updates": (
                        epoch * LONG.EXPECTED_UPDATES_PER_EPOCH
                    ),
                    "candidate_checkpoint": checkpoint,
                }
            )
        self.base_long_bundle = {
            "artifacts": self.base_long_candidate_artifacts,
            "manifest": {
                key: self.base_long_candidate_artifacts["manifest"][key]
                for key in ("path", "sha256")
            },
            "status": {
                key: self.base_long_candidate_artifacts["status"][key]
                for key in ("path", "sha256")
            },
            "frozen_inputs": {
                "path": self.base_long_candidate_artifacts[
                    "frozen_inputs"
                ]["path"],
                "sha256": self.base_long_candidate_artifacts[
                    "frozen_inputs"
                ]["sha256"],
                "receipt_sha256": hashlib.sha256(
                    b"base-long-frozen-receipt"
                ).hexdigest(),
            },
            "producer_source": {
                "origin": AUTH.ORIGIN,
                "commit": commit,
                "tree": tree,
                "branch": None,
                "clean": True,
                "entrypoint": str(long_trainer.resolve()),
                "entrypoint_sha256": hashlib.sha256(
                    long_trainer.read_bytes()
                ).hexdigest(),
                "source_root": str(self.source.resolve()),
                "official_remote_ref": "refs/heads/main",
                "official_remote_commit": commit,
                "official_main_ancestor": True,
                "official_baseline_ancestor": True,
                "detached": True,
                "local_branches_at_commit": [],
            },
            "candidate_epochs": list(LONG.EXPECTED_CANDIDATE_EPOCHS),
            "updates_per_epoch": LONG.EXPECTED_UPDATES_PER_EPOCH,
            "candidates": candidate_rows,
        }
        self.winner_selection = root / "winner.json"
        self.prerequisite_selection = root / "prerequisite-selection.json"
        self.continuation = root / "continuation.json"
        self.continuation_waves: list[dict[str, object]] = []
        self.test_claim = root / "test-claim.json"
        for stage in AUTH.REPRESENTATION_STAGES:
            (root / f"{stage}-measurement.json").write_bytes(
                f"{stage}-measurement".encode()
            )
        prerequisite_unsigned = {
            "format": "semtalk_show_prerequisite_val_selection_v1",
            "status": "selected",
            "split": "val",
            "test_visible": False,
            "protocol": {
                "candidate_epochs": list(range(20, 201, 20)),
                "candidates_per_stage": 10,
            },
            "stages": [
                {
                    "stage": stage,
                    "selection_metric": f"{stage}_metric",
                    "epoch": 200,
                    "optimizer_updates": 200 * 497,
                    "selection_score": 0.1,
                    "candidate_checkpoint": self.checkpoints[stage],
                    "measurement_receipt": {
                        "path": str(
                            (root / f"{stage}-measurement.json").resolve()
                        ),
                        "sha256": hashlib.sha256(
                            f"{stage}-measurement".encode()
                        ).hexdigest(),
                        "receipt_payload_sha256": hashlib.sha256(
                            f"{stage}-payload".encode()
                        ).hexdigest(),
                    },
                }
                for stage in AUTH.REPRESENTATION_STAGES
            ],
        }
        self.prerequisite_selection_payload = {
            **prerequisite_unsigned,
            "receipt_payload_sha256": AUTH.canonical_json_sha256(
                prerequisite_unsigned
            ),
        }
        write_json(
            self.prerequisite_selection,
            self.prerequisite_selection_payload,
        )
        prerequisite_binding = {
            "path": str(self.prerequisite_selection.resolve()),
            "sha256": artifact(self.prerequisite_selection)["sha256"],
            "receipt_payload_sha256": self.prerequisite_selection_payload[
                "receipt_payload_sha256"
            ],
        }
        self.prerequisite_bridge = {
            "format": "semtalk_show_selected_prerequisite_bridge_v1",
            "selection": prerequisite_binding,
            "selected": {
                stage: {
                    "candidate_checkpoint": self.checkpoints[stage]
                }
                for stage in AUTH.REPRESENTATION_STAGES
            },
            "global_verified_not_consumed": True,
            "test_visible": False,
        }
        winner_selection_unsigned = {
            "format": (
                "semtalk_show_base_talkshow_released2_fgd_selection_v1"
            ),
            "status": "selected",
            "candidates": copy.deepcopy(
                self.base_long_bundle["candidates"]
            ),
        }
        self.winner_selection_payload = {
            **winner_selection_unsigned,
            "receipt_payload_sha256": AUTH.canonical_json_sha256(
                winner_selection_unsigned
            ),
        }
        write_json(
            self.winner_selection,
            self.winner_selection_payload,
        )
        self.continuation_payload = {
            "format": "semtalk_show_prerequisite_continuation_decision_v1",
            "status": "complete",
            "decision": "stop",
            "test_visible": False,
            "inputs": {"selection": prerequisite_binding},
            "stages": [],
            "receipt_payload_sha256": hashlib.sha256(
                b"continuation-payload"
            ).hexdigest(),
        }
        write_json(
            self.continuation,
            self.continuation_payload,
        )
        self.base_validation = {
            "format": "semtalk_show_base_long_test_winner_authorization_v1",
            "status": "validated",
            "selection": {
                "path": str(self.winner_selection.resolve()),
                "sha256": artifact(self.winner_selection)["sha256"],
                "receipt_payload_sha256": hashlib.sha256(
                    b"winner-payload"
                ).hexdigest(),
            },
            "selected_epoch": 400,
            "selected_fgd": 1.25,
            "selected_checkpoint": self.checkpoints["base"],
        }
        self.test_claim_payload = {
            **self.base_validation,
            "status": "authorized",
            "authorized_test_evaluations": 1,
            "continuation_waves": self.continuation_waves,
        }
        self.test_claim_payload["receipt_payload_sha256"] = (
            AUTH.canonical_json_sha256(self.test_claim_payload)
        )
        write_json(
            self.test_claim,
            self.test_claim_payload,
        )
        self.output_root = root / "formal-output" / "final"
        self.published_validation = {
            "claim_artifact": artifact(self.test_claim),
            "receipt_payload_sha256": self.test_claim_payload[
                "receipt_payload_sha256"
            ],
            "winner_selection": {
                **artifact(self.winner_selection),
                "receipt_payload_sha256": self.winner_selection_payload[
                    "receipt_payload_sha256"
                ],
            },
            "selected_base_checkpoint": self.checkpoints["base"],
            "fixed_checkpoints": {
                stage: self.checkpoints[stage]
                for stage in AUTH.REPRESENTATION_STAGES
            },
            "continuation_waves": self.continuation_waves,
            "expected_output_root": str(self.output_root.resolve()),
            "test_policy": {
                "authorized_evaluations": 1,
                "selection_feedback": False,
            },
        }

    def kwargs(self) -> dict[str, object]:
        return {
            "expected_output_root": self.output_root,
            "canonical_manifest": artifact(self.canonical_manifest),
            "canonical_summary": artifact(self.canonical_summary),
            "canonical_lineage": artifact(self.canonical_lineage),
            "canonical_root_receipt": self.canonical_root_receipt,
            "audio_authorities": self.audio,
            "base_long_candidate_artifacts": (
                self.base_long_candidate_artifacts
            ),
            "winner_selection": artifact(self.winner_selection),
            "continuation_decision": artifact(self.continuation),
            "continuation_waves": self.continuation_waves,
            "test_claim": artifact(self.test_claim),
            "inference_source": self.inference_source,
            "checkpoints": self.checkpoints,
        }

    @contextmanager
    def fresh_control_validators(self):
        published = mock.Mock(
            return_value=dict(self.published_validation)
        )
        published_module = mock.Mock()
        published_module.validate_published_test_winner_claim = published
        with (
            mock.patch.object(
                AUTH,
                "_replay_continuation",
                return_value=dict(self.continuation_payload),
            ) as continuation,
            mock.patch.object(
                AUTH,
                "_replay_prerequisite_selection",
                return_value=dict(self.prerequisite_selection_payload),
            ) as prerequisite,
            mock.patch.object(
                AUTH,
                "_replay_prerequisite_bridge",
                return_value=dict(self.prerequisite_bridge),
            ) as prerequisite_bridge,
            mock.patch.object(
                AUTH,
                "_replay_base_long_candidate_bundle",
                return_value=copy.deepcopy(self.base_long_bundle),
            ) as base_long,
            mock.patch.object(
                AUTH,
                "_control_module",
                return_value=published_module,
            ),
            mock.patch.object(
                AUTH,
                "_remote_main_oid",
                return_value=self.inference_source["commit"],
            ),
        ):
            yield (
                continuation,
                prerequisite,
                prerequisite_bridge,
                base_long,
                published,
            )


class BaseFinalAuthorityTest(unittest.TestCase):
    def test_base_long_bundle_uses_fresh_twenty_two_candidate_validator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            raw_candidates = {
                row["epoch"]: row["candidate_checkpoint"]
                for row in fixture.base_long_bundle["candidates"]
            }
            raw_source = {
                key: fixture.base_long_bundle["producer_source"][key]
                for key in {
                    "origin",
                    "commit",
                    "tree",
                    "branch",
                    "clean",
                    "entrypoint",
                    "entrypoint_sha256",
                }
            }
            replayed = {
                "manifest": fixture.base_long_bundle["manifest"],
                "status": fixture.base_long_bundle["status"],
                "frozen_inputs": fixture.base_long_bundle["frozen_inputs"],
                "producer_source": raw_source,
                "candidates": raw_candidates,
            }
            validator = mock.Mock(return_value=replayed)
            module = mock.Mock(
                EXPECTED_CANDIDATE_EPOCHS=(
                    LONG.EXPECTED_CANDIDATE_EPOCHS
                ),
                EXPECTED_UPDATES_PER_EPOCH=(
                    LONG.EXPECTED_UPDATES_PER_EPOCH
                ),
                validate_candidate_bundle=validator,
            )
            with (
                mock.patch.object(
                    AUTH,
                    "_control_module",
                    return_value=module,
                ),
                mock.patch.object(
                    AUTH,
                    "_validate_official_training_source",
                    return_value=fixture.base_long_bundle[
                        "producer_source"
                    ],
                ) as source_validator,
            ):
                observed = AUTH._replay_base_long_candidate_bundle(
                    fixture.base_long_candidate_artifacts,
                    inference_source={
                        **fixture.inference_source,
                        "official_remote_commit": fixture.inference_source[
                            "commit"
                        ],
                    },
                )
            self.assertEqual(observed, fixture.base_long_bundle)
            validator.assert_called_once_with(
                manifest_path=Path(
                    fixture.base_long_candidate_artifacts["manifest"][
                        "path"
                    ]
                ),
                expected_manifest_sha256=(
                    fixture.base_long_candidate_artifacts["manifest"][
                        "sha256"
                    ]
                ),
                status_path=Path(
                    fixture.base_long_candidate_artifacts["status"]["path"]
                ),
                expected_status_sha256=(
                    fixture.base_long_candidate_artifacts["status"][
                        "sha256"
                    ]
                ),
                frozen_inputs_path=Path(
                    fixture.base_long_candidate_artifacts[
                        "frozen_inputs"
                    ]["path"]
                ),
                expected_frozen_inputs_sha256=(
                    fixture.base_long_candidate_artifacts[
                        "frozen_inputs"
                    ]["sha256"]
                ),
            )
            source_validator.assert_called_once_with(
                raw_source,
                inference_source={
                    **fixture.inference_source,
                    "official_remote_commit": fixture.inference_source[
                        "commit"
                    ],
                },
            )

    def test_published_winner_replay_has_neutral_checkpoint_free_signature(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim_path = root / "claim.json"
            claim_path.write_bytes(b"claim")
            claim = artifact(claim_path)
            payload_sha = hashlib.sha256(b"claim-payload").hexdigest()
            payload = {"receipt_payload_sha256": payload_sha}
            output_root = root / "formal-output"
            prerequisite = {
                **claim,
                "receipt_payload_sha256": payload_sha,
            }
            continuation = {
                **claim,
                "receipt_payload_sha256": payload_sha,
            }
            validated = {
                "claim_artifact": claim,
                "receipt_payload_sha256": payload_sha,
                "winner_selection": {
                    **artifact(claim_path),
                    "receipt_payload_sha256": payload_sha,
                },
                "selected_base_checkpoint": artifact(claim_path),
                "fixed_checkpoints": {
                    stage: artifact(claim_path)
                    for stage in AUTH.REPRESENTATION_STAGES
                },
                "continuation_waves": [],
                "expected_output_root": str(output_root),
                "test_policy": {"authorized_evaluations": 1},
            }
            validator = mock.Mock(return_value=validated)
            module = mock.Mock()
            module.validate_published_test_winner_claim = validator
            with mock.patch.object(
                AUTH,
                "_control_module",
                return_value=module,
            ):
                observed = AUTH._replay_published_test_winner_claim(
                    claim,
                    payload,
                    expected_output_root=output_root,
                    prerequisite_selection=prerequisite,
                    continuation_decision=continuation,
                    continuation_waves=[],
                )
            self.assertEqual(observed, validated)
            validator.assert_called_once_with(
                claim_path.resolve(),
                expected_claim_sha256=claim["sha256"],
                expected_claim_bytes=claim["bytes"],
                expected_claim_payload_sha256=payload_sha,
                expected_output_root=output_root,
                prerequisite_selection=prerequisite,
                continuation_decision=continuation,
                continuation_waves=[],
            )
            self.assertNotIn(
                "checkpoints",
                validator.call_args.kwargs,
            )

    def test_round_trip_and_external_three_way_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            with fixture.fresh_control_validators() as validators:
                authority = AUTH.build_test_authority(**fixture.kwargs())
                authority_path = fixture.root / "authority.json"
                AUTH.atomic_write_new(authority_path, authority)
                validated = AUTH.validate_test_authority(
                    authority_path,
                    expected_file_sha256=hashlib.sha256(
                        authority_path.read_bytes()
                    ).hexdigest(),
                    expected_bytes=authority_path.stat().st_size,
                    expected_receipt_payload_sha256=authority[
                        "receipt_payload_sha256"
                    ],
                )
            self.assertEqual(validated, authority)
            payload_artifact_keys = {
                "path",
                "sha256",
                "bytes",
                "receipt_payload_sha256",
            }
            for role in (
                "winner_selection",
                "continuation_decision",
                "prerequisite_selection",
            ):
                self.assertTrue(
                    payload_artifact_keys.issubset(authority[role])
                )
            self.assertEqual(
                authority["continuation_decision"][
                    "prerequisite_selection"
                ],
                {
                    key: authority["prerequisite_selection"][key]
                    for key in payload_artifact_keys
                },
            )
            for stage in authority["prerequisite_selection"][
                "stages"
            ].values():
                self.assertEqual(
                    set(stage["measurement_receipt"]),
                    payload_artifact_keys,
                )
            self.assertEqual(
                [validator.call_count for validator in validators],
                [2, 2, 2, 2, 2],
            )
            with self.assertRaises(AUTH.BaseFinalAuthorityError):
                AUTH.validate_test_authority(
                    authority_path,
                    expected_file_sha256="0" * 64,
                    expected_bytes=authority_path.stat().st_size,
                    expected_receipt_payload_sha256=authority[
                        "receipt_payload_sha256"
                    ],
                )

    def test_self_authored_inference_contract_and_receipts_are_not_api(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            kwargs = fixture.kwargs()
            kwargs["inference_contract"] = {
                "base_training_summary_sha256": "f" * 64,
                "source_roles": {
                    "training": {"commit": "f" * 40},
                },
            }
            kwargs["inference_checkpoint_receipts"] = {
                "face": {"audit": {"untrusted_extra": True}},
                "base": {
                    "candidate_checkpoints": [
                        {"evil": index} for index in range(22)
                    ]
                },
            }
            with fixture.fresh_control_validators():
                with self.assertRaisesRegex(
                    TypeError,
                    "unexpected keyword argument",
                ):
                    AUTH.build_test_authority(**kwargs)

    def test_rehashed_base_candidate_row_cannot_change_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            with fixture.fresh_control_validators():
                authority = AUTH.build_test_authority(**fixture.kwargs())
                tampered = copy.deepcopy(authority)
                tampered["base_long_candidate_bundle"]["candidates"][0][
                    "candidate_checkpoint"
                ] = dict(tampered["checkpoints"]["base"])
                tampered.pop("receipt_payload_sha256")
                tampered["receipt_payload_sha256"] = (
                    AUTH.canonical_json_sha256(tampered)
                )
                path = fixture.root / "tampered-base-row.json"
                AUTH.atomic_write_new(path, tampered)
                with self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "fresh replay",
                ):
                    AUTH.validate_test_authority(
                        path,
                        expected_file_sha256=artifact(path)["sha256"],
                        expected_bytes=artifact(path)["bytes"],
                        expected_receipt_payload_sha256=tampered[
                            "receipt_payload_sha256"
                        ],
                    )

    def test_fresh_e220_prerequisite_stop_is_accepted_but_rehash_is_not(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            fixture.prerequisite_selection_payload["protocol"] = {
                "candidate_epochs": [
                    *range(20, 201, 20),
                    220,
                ],
                "candidates_per_stage": 11,
            }
            face = fixture.prerequisite_selection_payload["stages"][0]
            face["epoch"] = 220
            face["optimizer_updates"] = 220 * 497
            with (
                fixture.fresh_control_validators(),
                self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "does not cover every appended",
                ),
            ):
                AUTH.build_test_authority(**fixture.kwargs())

            wave, receipt = write_wave(
                fixture.root / "wave-e200-e220.json",
                boundary=200,
                predecessor=None,
            )
            wave_module = mock.Mock()
            wave_module.replay_wave_file.return_value = receipt
            with mock.patch.object(
                AUTH,
                "_control_module",
                return_value=wave_module,
            ):
                normalized_waves = AUTH._replay_continuation_waves(
                    [wave],
                    candidate_epochs=[*range(20, 201, 20), 220],
                )
            fixture.continuation_waves.append(wave)
            fixture.published_validation["continuation_waves"] = [wave]
            with (
                fixture.fresh_control_validators() as validators,
                mock.patch.object(
                    AUTH,
                    "_replay_continuation_waves",
                    return_value=normalized_waves,
                ),
            ):
                authority = AUTH.build_test_authority(**fixture.kwargs())
                self.assertEqual(
                    authority["prerequisite_selection"]["stages"]["face"][
                        "epoch"
                    ],
                    220,
                )
                # Rehashing the stored authority cannot turn the freshly
                # replayed e220 selection into an unaudited e240 selection.
                tampered = copy.deepcopy(authority)
                tampered["prerequisite_selection"]["stages"]["face"][
                    "epoch"
                ] = 240
                tampered.pop("receipt_payload_sha256")
                tampered["receipt_payload_sha256"] = (
                    AUTH.canonical_json_sha256(tampered)
                )
                path = fixture.root / "self-signed-extension.json"
                AUTH.atomic_write_new(path, tampered)
                with self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "fresh replay",
                ):
                    AUTH.validate_test_authority(
                        path,
                        expected_file_sha256=artifact(path)["sha256"],
                        expected_bytes=artifact(path)["bytes"],
                        expected_receipt_payload_sha256=tampered[
                            "receipt_payload_sha256"
                        ],
                    )
            self.assertGreaterEqual(validators[1].call_count, 2)

    def test_continuation_wave_chain_is_exact_and_predecessor_linked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_receipt = write_wave(
                root / "wave-e200-e220.json",
                boundary=200,
                predecessor=None,
            )
            predecessor = {
                key: first[key]
                for key in ("path", "sha256", "receipt_payload_sha256")
            }
            second, second_receipt = write_wave(
                root / "wave-e220-e240.json",
                boundary=220,
                predecessor=predecessor,
            )
            receipts = {
                first["path"]: first_receipt,
                second["path"]: second_receipt,
            }
            module = mock.Mock()
            module.replay_wave_file.side_effect = (
                lambda path, _sha: copy.deepcopy(receipts[str(path)])
            )
            with mock.patch.object(
                AUTH,
                "_control_module",
                return_value=module,
            ):
                observed = AUTH._replay_continuation_waves(
                    [first, second],
                    candidate_epochs=[*range(20, 201, 20), 220, 240],
                )
                self.assertEqual(
                    [(row["boundary_epoch"], row["target_epoch"]) for row in observed],
                    [(200, 220), (220, 240)],
                )
                with self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "does not cover every appended",
                ):
                    AUTH._replay_continuation_waves(
                        [first],
                        candidate_epochs=[
                            *range(20, 201, 20),
                            220,
                            240,
                        ],
                    )
                receipts[second["path"]]["stages"][0]["old_segment"][
                    "predecessor_wave"
                ] = None
                with self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "predecessor chain mismatch",
                ):
                    AUTH._replay_continuation_waves(
                        [first, second],
                        candidate_epochs=[
                            *range(20, 201, 20),
                            220,
                            240,
                        ],
                    )

    def test_all_twenty_two_published_rows_bind_long_training_bundle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            selection = copy.deepcopy(fixture.winner_selection_payload)
            selection["candidates"][0]["candidate_checkpoint"] = dict(
                fixture.checkpoints["base"]
            )
            selection.pop("receipt_payload_sha256")
            selection["receipt_payload_sha256"] = (
                AUTH.canonical_json_sha256(selection)
            )
            write_json(fixture.winner_selection, selection)
            fixture.published_validation["winner_selection"] = {
                **artifact(fixture.winner_selection),
                "receipt_payload_sha256": selection[
                    "receipt_payload_sha256"
                ],
            }
            with (
                fixture.fresh_control_validators(),
                self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "Base e1 candidate",
                ),
            ):
                AUTH.build_test_authority(**fixture.kwargs())

    def test_unreachable_base_training_source_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            source = {
                key: fixture.base_long_bundle["producer_source"][key]
                for key in {
                    "origin",
                    "commit",
                    "tree",
                    "branch",
                    "clean",
                    "entrypoint",
                    "entrypoint_sha256",
                }
            }
            source["commit"] = "f" * 40
            with self.assertRaisesRegex(
                AUTH.BaseFinalAuthorityError,
                "reachable official-main",
            ):
                AUTH._validate_official_training_source(
                    source,
                    inference_source={
                        **fixture.inference_source,
                        "official_remote_commit": fixture.inference_source[
                            "commit"
                        ],
                    },
                )

    def test_base_training_source_receipt_cannot_hide_attached_head(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            subprocess.run(
                ["git", "switch", "-q", "-c", "forbidden-producer-branch"],
                cwd=fixture.source,
                check=True,
            )
            source = {
                key: fixture.base_long_bundle["producer_source"][key]
                for key in {
                    "origin",
                    "commit",
                    "tree",
                    "branch",
                    "clean",
                    "entrypoint",
                    "entrypoint_sha256",
                }
            }
            self.assertIsNone(source["branch"])
            with self.assertRaisesRegex(
                AUTH.BaseFinalAuthorityError,
                "reachable official-main",
            ):
                AUTH._validate_official_training_source(
                    source,
                    inference_source={
                        **fixture.inference_source,
                        "official_remote_commit": fixture.inference_source[
                            "commit"
                        ],
                    },
                )

    def test_same_path_cached_control_module_is_ignored(self) -> None:
        expected_path = (
            Path(AUTH.__file__).resolve().parent
            / "prerequisite_val_contract.py"
        )
        poisoned = types.SimpleNamespace(
            __file__=str(expected_path),
            REQUIRED_CANDIDATE_EPOCHS=(999,),
            EXPECTED_UPDATES_PER_EPOCH=1,
            validate_candidate_epochs=lambda _value: (999,),
        )
        full_name = "scripts.show_base.prerequisite_val_contract"
        with (
            mock.patch.dict(sys.modules, {full_name: poisoned}),
            mock.patch.object(
                SHOW_BASE_PACKAGE,
                "prerequisite_val_contract",
                poisoned,
                create=True,
            ),
        ):
            observed = AUTH._control_module(
                "prerequisite_val_contract"
            )
        self.assertIsNot(observed, poisoned)
        self.assertEqual(
            observed.REQUIRED_CANDIDATE_EPOCHS,
            tuple(range(20, 201, 20)),
        )
        self.assertEqual(observed.EXPECTED_UPDATES_PER_EPOCH, 497)

    def test_each_checkpoint_replacement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            for stage in AUTH.CHECKPOINT_STAGES:
                with self.subTest(stage=stage):
                    kwargs = fixture.kwargs()
                    checkpoints = {
                        key: dict(value)
                        for key, value in fixture.checkpoints.items()
                    }
                    replacement = fixture.root / f"{stage}-replacement.bin"
                    replacement.write_bytes(b"replacement")
                    checkpoints[stage] = artifact(replacement)
                    kwargs["checkpoints"] = checkpoints
                    with (
                        fixture.fresh_control_validators(),
                        self.assertRaisesRegex(
                            AUTH.BaseFinalAuthorityError,
                            "(?i)bind",
                        ),
                    ):
                        AUTH.build_test_authority(**kwargs)

    def test_byte_identical_checkpoint_copy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            kwargs = fixture.kwargs()
            checkpoints = {
                key: dict(value)
                for key, value in fixture.checkpoints.items()
            }
            copied = fixture.root / "copied-face.bin"
            copied.write_bytes(
                Path(checkpoints["hands"]["path"]).read_bytes()
            )
            checkpoints["face"] = artifact(copied)
            kwargs["checkpoints"] = checkpoints
            with (
                fixture.fresh_control_validators(),
                self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "share a path or content",
                ),
            ):
                AUTH.build_test_authority(**kwargs)

    def test_canonical_audio_source_and_control_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            with fixture.fresh_control_validators():
                authority = AUTH.build_test_authority(**fixture.kwargs())
            authority_path = fixture.root / "authority.json"
            AUTH.atomic_write_new(authority_path, authority)
            tamper_targets = [
                fixture.canonical_manifest,
                Path(fixture.audio[0]["manifest"]["path"]),
                fixture.audio_feature_paths[0],
                fixture.winner_selection,
                fixture.prerequisite_selection,
                fixture.continuation,
                fixture.test_claim,
            ]
            for target in tamper_targets:
                with self.subTest(target=target.name):
                    original = target.read_bytes()
                    target.write_bytes(original + b" ")
                    try:
                        with (
                            fixture.fresh_control_validators(),
                            self.assertRaises(
                                AUTH.BaseFinalAuthorityError
                            ),
                        ):
                            AUTH.validate_test_authority(
                                authority_path,
                                expected_file_sha256=hashlib.sha256(
                                    authority_path.read_bytes()
                                ).hexdigest(),
                                expected_bytes=authority_path.stat().st_size,
                                expected_receipt_payload_sha256=authority[
                                    "receipt_payload_sha256"
                                ],
                            )
                    finally:
                        target.write_bytes(original)

    def test_source_must_be_detached_official_descendant_without_branches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            subprocess.run(
                ["git", "branch", "forbidden-local-branch"],
                cwd=fixture.source,
                check=True,
            )
            with (
                fixture.fresh_control_validators(),
                self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "zero local branches",
                ),
            ):
                AUTH.build_test_authority(**fixture.kwargs())

    def test_source_must_equal_live_official_main(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            with (
                fixture.fresh_control_validators(),
                mock.patch.object(
                    AUTH,
                    "_remote_main_oid",
                    return_value="0" * 40,
                ),
                self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "official baseline descendant",
                ),
            ):
                AUTH.build_test_authority(**fixture.kwargs())

    def test_canonical_root_external_source_pin_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AuthorityFixture(Path(directory))
            kwargs = fixture.kwargs()
            wrong = dict(fixture.canonical_root_receipt)
            wrong["source_commit"] = "0" * 40
            kwargs["canonical_root_receipt"] = wrong
            with (
                fixture.fresh_control_validators(),
                self.assertRaisesRegex(
                    AUTH.BaseFinalAuthorityError,
                    "canonical root/source lineage",
                ),
            ):
                AUTH.build_test_authority(**kwargs)

    def test_globaldiff_named_evaluator_asset_is_not_a_generator(self) -> None:
        AUTH._reject_forbidden(
            {
                "evaluator_asset": (
                    "/frozen/globaldiff_show_metrics/TalkSHOW/body_ae.pth"
                ),
                "smplx_asset": "/frozen/globaldiff/smplx/SMPLX_NEUTRAL.npz",
            },
            "pinned evaluator assets",
        )
        with self.assertRaisesRegex(
            AUTH.BaseFinalAuthorityError,
            "forbidden generator identity",
        ):
            AUTH._reject_forbidden(
                {"generator": "GlobalDiff speaker2 sparse model"},
                "generator",
            )

    def test_atomic_write_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                AUTH.atomic_write_new(path, {"value": 1})
            self.assertEqual(path.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
