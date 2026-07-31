from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts.show_base import decide_prerequisite_continuation as decision
from scripts.show_base import merge_prerequisite_val_shards as merger
from scripts.show_base import prerequisite_continuation_wave as wave
from scripts.show_base import prerequisite_val_contract as contract
from scripts.show_base import select_prerequisite_candidates as selector
from tests import test_prerequisite_val_selection_cpu as fixture_helpers


HOSTS = tuple(sorted(wave.FORMAL_HOSTS))


class FakeTensor:
    _continuation_test_tensor = True

    def __init__(self, *values: float, shape: tuple[int, ...] | None = None):
        self.values = tuple(values)
        self.shape = shape or (len(values),)
        self.bytes = repr((self.shape, self.values)).encode("ascii")
        self.dtype = "float32"


class ImprovingFixture:
    def __init__(
        self,
        *,
        schedule: tuple[int, ...] | None = None,
        source_receipt: dict[str, object] | None = None,
        run_for_epoch=None,
    ) -> None:
        self.case = fixture_helpers.PrerequisiteValidationSelectionTest(
            methodName="test_full_merge_and_exact_selection_bridge"
        )
        self.schedule = schedule or tuple(range(20, 201, 20))
        self.source_receipt_value = source_receipt or {}
        self.run_for_epoch = run_for_epoch

    def __getattr__(self, name: str):
        return getattr(self.case, name)

    def source_receipt(self, source_root: Path, name: str):
        return fixture_helpers.PrerequisiteValidationSelectionTest.source_receipt(
            self, source_root, name
        )

    def canonical_fixture(self, root: Path):
        return fixture_helpers.PrerequisiteValidationSelectionTest.canonical_fixture(
            self, root
        )

    def shard_fixture(self, *args, **kwargs):
        return fixture_helpers.PrerequisiteValidationSelectionTest.shard_fixture(
            self, *args, **kwargs
        )

    def build_full_fixture(self, root: Path):
        return fixture_helpers.PrerequisiteValidationSelectionTest.build_full_fixture(
            self, root
        )

    def training_source_receipt(self, source_root: Path, name: str):
        return copy.deepcopy(self.source_receipt_value)

    @staticmethod
    def candidate_error(stage: str, epoch: int) -> float:
        return (240.0 - epoch) / 100.0

    def candidate_fixture(self, root: Path, source: dict[str, object]):
        stages: dict[str, list[dict[str, object]]] = {}
        source_receipts = {
            stage: copy.deepcopy(source) for stage in contract.STAGES
        }
        for stage in contract.STAGES:
            entries = []
            for epoch in self.schedule:
                run = self.run_for_epoch(stage, epoch)
                path = (
                    run
                    / "representation_candidates"
                    / f"{stage}_epoch_{epoch:04d}.bin"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{stage}:{epoch}\n".encode())
                payload = path.read_bytes()
                entries.append(
                    {
                        "epoch": epoch,
                        "optimizer_updates": (
                            epoch * contract.updates_per_epoch(stage)
                        ),
                        "checkpoint": str(path.resolve()),
                        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                        "checkpoint_bytes": len(payload),
                        "checkpoint_audit_sha256": hashlib.sha256(
                            f"audit:{stage}:{epoch}".encode()
                        ).hexdigest(),
                    }
                )
            stages[stage] = entries
        value = contract.receipt_payload(
            {
                "format": contract.CANDIDATE_INDEX_FORMAT,
                "status": "complete",
                "target_dataset": "SHOW",
                "target_speaker_scope": contract.TARGET_SPEAKER_SCOPE,
                "selection_split": "val",
                "test_visible": False,
                "candidate_epochs": list(self.schedule),
                "updates_per_epoch": contract.updates_per_epoch_map(
                    contract.STAGES
                ),
                "source_policy": contract.build_source_policy(
                    source_receipts,
                    stages=contract.STAGES,
                    reprove_ancestry=False,
                ),
                "source_receipts": source_receipts,
                "config_sha256": {
                    stage: hashlib.sha256(stage.encode()).hexdigest()
                    for stage in contract.STAGES
                },
                "dataset_receipt_sha256": {
                    stage: hashlib.sha256(
                        f"dataset:{stage}".encode()
                    ).hexdigest()
                    for stage in contract.STAGES
                },
                "formal_training_status": {
                    stage: {
                        "path": str(
                            (self.run_for_epoch(stage, self.schedule[-1])
                             / "formal_training_status.json").resolve()
                        ),
                        "sha256": "0" * 64,
                    }
                    for stage in contract.STAGES
                },
                "stages": stages,
            }
        )
        path = root / "candidate_index.json"
        digest = fixture_helpers.write_json(path, value)
        contract.validate_candidate_index(value)
        return path, digest, value


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_receipt(
    repository: Path,
    *,
    commit: str,
    tree: str,
) -> dict[str, object]:
    entrypoint = repository / "scripts/show_base/train_official_transfer.py"
    digest = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    audit = {
        "commit": commit,
        "tree": tree,
        "origin": contract.EXPECTED_ORIGIN,
        "entrypoint": str(entrypoint),
        "entrypoint_sha256": digest,
    }
    return contract.receipt_payload(
        {
            "format": contract.TRAINING_SOURCE_FREEZE_FORMAT,
            "training_audit": audit,
            "source_root": str(repository),
            "portable_identity": {
                "origin": contract.EXPECTED_ORIGIN,
                "commit": commit,
                "tree": tree,
                "script_relative": "scripts/show_base/train_official_transfer.py",
                "script_sha256": digest,
            },
            "clean": True,
            "detached": True,
            "local_branch_count": 0,
            "official_baseline_commit": contract.OFFICIAL_BASELINE_COMMIT,
            "official_baseline_is_ancestor": True,
        }
    )


def _smplx(stage: str, host: str) -> dict[str, object] | None:
    if stage == "global":
        return None
    return {
        "format": "semtalk_show_smplx_asset_v1",
        "filename": wave.FORMAL_SMPLX_FILENAME,
        "path": f"/formal/assets/{host}/{wave.FORMAL_SMPLX_FILENAME}",
        "sha256": wave.FORMAL_SMPLX_SHA256,
        "bytes": wave.FORMAL_SMPLX_BYTES,
        "regular_file": True,
        "symlink": False,
    }


def _resume(stage: str, boundary: int) -> dict[str, object]:
    world = 4 if stage in wave.RVQ_STAGES else 1
    stage_updates = contract.updates_per_epoch(stage)
    updates = boundary * stage_updates
    rvq = (
        {
            f"layer_{index}": {
                "init": True,
                "code_sum": FakeTensor(1.0, shape=(1, 1)),
                "code_count": FakeTensor(1.0, shape=(1,)),
            }
            for index in range(6)
        }
        if stage in wave.RVQ_STAGES
        else {}
    )
    return {
        "format": "semtalk_show_train_resume_v5",
        "completed_epochs": boundary,
        "optimizer_updates": updates,
        "updates_per_epoch": stage_updates,
        "world_size": world,
        "model_state": {"weight": FakeTensor(1.0)},
        "optimizer_state": {
            "state": {
                0: {
                    "step": updates,
                    "exp_avg": FakeTensor(0.1),
                    "exp_avg_sq": FakeTensor(0.01),
                }
            },
            "param_groups": [{"params": [0], "lr": 0.001}],
        },
        "scheduler_state": {
            "base_values": [0.001],
            "decay_t": 1_000,
            "decay_rate": 0.5,
            "warmup_t": 0,
            "warmup_lr_init": 0.0,
            "noise_range_t": None,
            "t_in_epochs": True,
        },
        "rvq_ema_state": rvq,
        "rng_states": [
            {
                "python": (rank, 1),
                "numpy": ("MT19937", [rank], 0, 0, 0.0),
                "torch_cpu": FakeTensor(float(rank + 1)),
                "torch_cuda": FakeTensor(float(rank + 2)),
            }
            for rank in range(world)
        ],
    }


def _write_runtime_statuses(
    fixture: dict[str, object],
    *,
    boundary: int,
    source_receipt: dict[str, object],
    hosts: dict[str, str],
    run_for_stage,
) -> None:
    index = fixture["candidate_index"]
    for stage in contract.STAGES:
        run = run_for_stage(stage)
        run.mkdir(parents=True, exist_ok=True)
        summary_path = run / "representation_summary.json"
        summary_sha = fixture_helpers.write_json(
            summary_path,
            {
                "status": "complete",
                "protocol": {
                    "split": "train",
                    "speaker_map": dict(
                        wave.selected_contract.EXPECTED_SPEAKER_MAP
                    ),
                },
                "train_clips": wave.selected_contract.EXPECTED_TRAIN_CLIPS,
                "entries": wave.selected_contract.EXPECTED_TRAIN_WINDOWS,
                "speaker_clip_counts": dict(
                    wave.selected_contract.EXPECTED_TRAIN_SPEAKER_CLIPS
                ),
                "speaker_window_counts": dict(
                    wave.selected_contract.EXPECTED_TRAIN_SPEAKER_WINDOWS
                ),
            },
        )
        dataset = {
            "summary": str(summary_path),
            "summary_sha256": summary_sha,
            "train_clips": wave.selected_contract.EXPECTED_TRAIN_CLIPS,
            "entries": wave.selected_contract.EXPECTED_TRAIN_WINDOWS,
            "split_label": "SHOW available frozen subset",
            "source_binding": {
                key: source_receipt["portable_identity"][key]
                for key in ("origin", "commit", "tree")
            },
            "smplx_asset": _smplx(stage, hosts[stage]),
        }
        dataset_sha = contract.canonical_payload_sha256(dataset)
        index["dataset_receipt_sha256"][stage] = dataset_sha

        resume_path = run / "latest_resume.pt"
        resume_path.write_bytes(pickle.dumps(_resume(stage, boundary)))
        final_path = run / f"{stage}_final.bin"
        final_path.write_bytes(f"final:{stage}:{boundary}\n".encode())
        latest = index["stages"][stage][-1]
        status = {
            "status": "complete",
            "formal_stage": stage,
            "completed_epochs": boundary,
            "updates_per_epoch": contract.updates_per_epoch(stage),
            "optimizer_updates": (
                boundary * contract.updates_per_epoch(stage)
            ),
            "world_size": 4 if stage in wave.RVQ_STAGES else 1,
            "hostname": hosts[stage],
            "config_sha256": index["config_sha256"][stage],
            "source_receipt": source_receipt["training_audit"],
            "source_receipt_sha256": contract.canonical_payload_sha256(
                source_receipt["training_audit"]
            ),
            "dataset_receipt": dataset,
            "smplx_asset_receipt": _smplx(stage, hosts[stage]),
            "latest_representation_candidate": {
                "path": latest["checkpoint"],
                "sha256": latest["checkpoint_sha256"],
                "completed_epochs": boundary,
                "optimizer_updates": (
                    boundary * contract.updates_per_epoch(stage)
                ),
                "selection_status": "offline_validation_pending",
            },
            "latest_resume_sha256": hashlib.sha256(
                resume_path.read_bytes()
            ).hexdigest(),
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": hashlib.sha256(
                final_path.read_bytes()
            ).hexdigest(),
        }
        status_path = run / "formal_training_status.json"
        status_sha = fixture_helpers.write_json(status_path, status)
        index["formal_training_status"][stage] = {
            "path": str(status_path),
            "sha256": status_sha,
        }
    unsigned = dict(index)
    unsigned.pop("receipt_payload_sha256")
    fixture["candidate_index"] = contract.receipt_payload(unsigned)
    fixture["candidate_sha"] = fixture_helpers.write_json(
        fixture["candidate_path"],
        fixture["candidate_index"],
    )


def _build_selection_and_decision(
    root: Path,
    *,
    schedule: tuple[int, ...],
    boundary: int,
    source_receipt: dict[str, object],
    run_for_epoch,
    hosts: dict[str, str],
) -> dict[str, object]:
    builder = ImprovingFixture(
        schedule=schedule,
        source_receipt=source_receipt,
        run_for_epoch=run_for_epoch,
    )
    with mock.patch.object(contract, "EXPECTED_CANDIDATE_EPOCHS", schedule):
        fixture = builder.build_full_fixture(root)
    _write_runtime_statuses(
        fixture,
        boundary=boundary,
        source_receipt=source_receipt,
        hosts=hosts,
        run_for_stage=lambda stage: run_for_epoch(stage, boundary),
    )
    measurement_root = root / "measurements"
    merger.merge(
        candidate_index_path=fixture["candidate_path"],
        candidate_index_sha256=fixture["candidate_sha"],
        canonical_manifest=fixture["manifest"],
        canonical_manifest_sha256=fixture["manifest_sha"],
        canonical_summary=fixture["summary"],
        canonical_summary_sha256=fixture["summary_sha"],
        canonical_lineage=fixture["lineage"],
        canonical_lineage_sha256=fixture["lineage_sha"],
        shard_roots=fixture["shard_roots"],
        output_root=measurement_root,
        merge_source=fixture["merge_source"],
    )
    measurement_index = measurement_root / "measurement_index.json"
    selection_path = root / "selection.json"
    selection = selector.select(
        candidate_index_path=fixture["candidate_path"],
        candidate_index_sha256=fixture["candidate_sha"],
        measurement_index_path=measurement_index,
        measurement_index_sha256=hashlib.sha256(
            measurement_index.read_bytes()
        ).hexdigest(),
        output_json=selection_path,
        selector_source=fixture["selector_source"],
    )
    assert all(stage["epoch"] == boundary for stage in selection["stages"])
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    decision_path = root / "decision.json"
    decision_value = decision.decide(
        selection_path=selection_path,
        expected_selection_sha256=selection_sha,
        output_json=decision_path,
    )
    assert decision_value["decision"] == "continue"
    return {
        "fixture": fixture,
        "decision_path": decision_path,
        "decision_sha": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
    }


def _source_binding(receipt: dict[str, object]) -> dict[str, str]:
    portable = receipt["portable_identity"]
    return {
        "commit": portable["commit"],
        "tree": portable["tree"],
        "source_receipt_sha256": receipt["receipt_payload_sha256"],
    }


def _plans(
    built: dict[str, object],
    *,
    old_receipt: dict[str, object],
    new_receipt: dict[str, object],
    repository: Path,
    old_runs: dict[str, Path],
    new_runs: dict[str, Path],
    old_hosts: dict[str, str],
    new_hosts: dict[str, str],
    chains: dict[str, list[dict[str, object]]],
    predecessor: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    index = built["fixture"]["candidate_index"]
    result = {}
    for stage in contract.STAGES:
        semantic = hashlib.sha256(f"semantic:{stage}".encode()).hexdigest()
        result[stage] = {
            "stage": stage,
            "old_run_path": str(old_runs[stage]),
            "new_run_path": str(new_runs[stage]),
            "old_source": _source_binding(old_receipt),
            "new_source": _source_binding(new_receipt),
            "new_source_repository": str(repository),
            "old_host": old_hosts[stage],
            "new_host": new_hosts[stage],
            "old_smplx_asset": _smplx(stage, old_hosts[stage]),
            "new_smplx_asset": _smplx(stage, new_hosts[stage]),
            "candidate_segment_chain": chains[stage],
            "predecessor_wave": copy.deepcopy(predecessor),
            "old_config_sha256": index["config_sha256"][stage],
            "new_config_sha256": index["config_sha256"][stage],
            "old_config_semantic_sha256": semantic,
            "new_config_semantic_sha256": semantic,
            "old_dataset_semantic_sha256": semantic,
            "new_dataset_semantic_sha256": semantic,
        }
    return result


class ContinuationWaveFileReplayE2ETest(unittest.TestCase):
    def test_real_replay_file_recurses_e200_e220_e240(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repository = root / "source-repository"
            subprocess.run(
                [
                    "git",
                    "-c",
                    "advice.detachedHead=false",
                    "clone",
                    "-q",
                    "--no-local",
                    str(Path(__file__).resolve().parents[1]),
                    str(repository),
                ],
                check=True,
            )
            _git(repository, "remote", "set-url", "origin", contract.EXPECTED_ORIGIN)
            head = _git(repository, "rev-parse", "HEAD")
            _git(repository, "checkout", "-q", "--detach", head)
            for reference in _git(
                repository,
                "for-each-ref",
                "--format=%(refname)",
                "refs/heads",
            ).splitlines():
                _git(repository, "update-ref", "-d", reference)
            old_commit = contract.OFFICIAL_BASELINE_COMMIT
            old_tree = _git(repository, "rev-parse", f"{old_commit}^{{tree}}")
            new_tree = _git(repository, "rev-parse", "HEAD^{tree}")
            old_source = _source_receipt(
                repository,
                commit=old_commit,
                tree=old_tree,
            )
            new_source = _source_receipt(
                repository,
                commit=head,
                tree=new_tree,
            )

            base_runs = {
                stage: root / "runs" / "base" / stage
                for stage in contract.STAGES
            }
            e220_runs = {
                stage: root / "runs" / "e220" / stage
                for stage in contract.STAGES
            }
            e240_runs = {
                stage: root / "runs" / "e240" / stage
                for stage in contract.STAGES
            }
            old_hosts = {
                stage: HOSTS[index % len(HOSTS)]
                for index, stage in enumerate(contract.STAGES)
            }
            e220_hosts = {
                stage: HOSTS[(index + 1) % len(HOSTS)]
                for index, stage in enumerate(contract.STAGES)
            }
            e240_hosts = dict(old_hosts)

            first = _build_selection_and_decision(
                root / "first",
                schedule=tuple(range(20, 201, 20)),
                boundary=200,
                source_receipt=old_source,
                run_for_epoch=lambda stage, epoch: base_runs[stage],
                hosts=old_hosts,
            )
            first_chains = {
                stage: [
                    wave._make_chain_segment(
                        run_path=str(base_runs[stage]),
                        start_epoch=20,
                        end_epoch=200,
                        predecessor_segment_id=None,
                    )
                ]
                for stage in contract.STAGES
            }
            first_plans = _plans(
                first,
                old_receipt=old_source,
                new_receipt=new_source,
                repository=repository,
                old_runs=base_runs,
                new_runs=e220_runs,
                old_hosts=old_hosts,
                new_hosts=e220_hosts,
                chains=first_chains,
                predecessor=None,
            )

            fake_torch = types.ModuleType("torch")
            fake_torch.load = lambda stream, **_kwargs: pickle.load(stream)
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                first_receipt = wave.authorize_wave_from_replayed_inputs(
                    decision_path=first["decision_path"],
                    expected_decision_sha256=first["decision_sha"],
                    stage_plans=first_plans,
                )
                first_path = root / "wave-e200-e220.json"
                first_sha = fixture_helpers.write_json(first_path, first_receipt)
                self.assertEqual(
                    wave.replay_wave_file(first_path, first_sha),
                    first_receipt,
                )

                second = _build_selection_and_decision(
                    root / "second",
                    schedule=tuple(range(20, 221, 20)),
                    boundary=220,
                    source_receipt=new_source,
                    run_for_epoch=lambda stage, epoch: (
                        base_runs[stage] if epoch <= 200 else e220_runs[stage]
                    ),
                    hosts=e220_hosts,
                )
                predecessor = {
                    "path": str(first_path),
                    "sha256": first_sha,
                    "receipt_payload_sha256": first_receipt[
                        "receipt_payload_sha256"
                    ],
                }
                first_by_stage = {
                    entry["stage"]: entry for entry in first_receipt["stages"]
                }
                second_chains = {}
                for stage in contract.STAGES:
                    previous = first_by_stage[stage]
                    appended = wave._make_chain_segment(
                        run_path=str(e220_runs[stage]),
                        start_epoch=220,
                        end_epoch=220,
                        predecessor_segment_id=previous["new_segment"][
                            "predecessor_segment_id"
                        ],
                    )
                    second_chains[stage] = copy.deepcopy(
                        previous["old_segment"]["candidate_segment_chain"]
                    ) + [appended]
                second_plans = _plans(
                    second,
                    old_receipt=new_source,
                    new_receipt=new_source,
                    repository=repository,
                    old_runs=e220_runs,
                    new_runs=e240_runs,
                    old_hosts=e220_hosts,
                    new_hosts=e240_hosts,
                    chains=second_chains,
                    predecessor=predecessor,
                )
                second_receipt = wave.authorize_wave_from_replayed_inputs(
                    decision_path=second["decision_path"],
                    expected_decision_sha256=second["decision_sha"],
                    stage_plans=second_plans,
                )
                second_path = root / "wave-e220-e240.json"
                second_sha = fixture_helpers.write_json(
                    second_path,
                    second_receipt,
                )
                replayed = wave.replay_wave_file(second_path, second_sha)

            self.assertEqual(
                (first_receipt["boundary_epoch"], first_receipt["target_epoch"]),
                (200, 220),
            )
            self.assertEqual(
                (replayed["boundary_epoch"], replayed["target_epoch"]),
                (220, 240),
            )
            self.assertTrue(
                all(
                    entry["old_segment"]["predecessor_wave"] == predecessor
                    for entry in replayed["stages"]
                )
            )


if __name__ == "__main__":
    unittest.main()
