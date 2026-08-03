from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/show_base/dispatch_base_v14_live_validation_dual_host.py"
SHOW_DIR = SCRIPT.parent
if str(SHOW_DIR) not in sys.path:
    sys.path.insert(0, str(SHOW_DIR))
SPEC = importlib.util.spec_from_file_location("dual_dispatch_cpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dual = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dual
SPEC.loader.exec_module(dual)


def artifact(path: str = "/evidence/item.json", token: str = "a", size: int = 7):
    return {"path": path, "sha256": token * 64, "bytes": size}


def minimal_campaign(root: Path) -> dict:
    state = root / "state"
    state.mkdir()
    for name in (
        "job_claims", "completions", "runner_status", "runner_logs",
        "authorities", "authorizations",
    ):
        (state / name).mkdir()
    control_root = "/local-ssd/control"
    return {
        "_artifact": artifact(str(state / "campaign.json"), "1", 101),
        "_state_root": state,
        "_claims_dir": state / "job_claims",
        "_completions_dir": state / "completions",
        "_runner_status_dir": state / "runner_status",
        "_runner_logs_dir": state / "runner_logs",
        "_authorities_dir": state / "authorities",
        "_authorizations_dir": state / "authorizations",
        "_control_source": {
            "root": control_root,
            "origin": dual.endpoint.OFFICIAL_ORIGIN,
            "commit": "2" * 40,
            "tree": "3" * 40,
            "launcher": artifact(
                control_root + "/scripts/show_base/run_base_live_val_8shard.sh", "4", 9
            ),
        },
        "_formal_python": {
            "format": "formal_python_binding_v1",
            "argv0": "/safe/python",
            "venv_root": "/safe/venv",
            "symlink_chain": [],
            "resolved_target": artifact("/safe/python-real", "5", 12),
            "pyvenv_cfg": artifact("/safe/pyvenv.cfg", "6", 13),
        },
        "_guarded_runner": artifact("/tmp/globaldiff_guarded_runner.py", "7", 14),
        "_guard_verifier": artifact("/tmp/verify_globaldiff_guards.py", "8", 15),
        "_paspa_root": "/safe/paspa",
        "_diffsheg_root": "/safe/diffsheg",
        "_seed": 20260802,
        "_diffsheg_batch_size": 64,
        "_adopted_e1": artifact("/evidence/adopted-e1.json", "9", 16),
        "_jobs": [],
        "campaign_claim_path": str(state / "campaign.claim.json"),
        "summary_path": str(state / "final_summary.json"),
    }


def dynamic_job(root: Path, epoch: int) -> dict:
    return {
        "epoch": epoch,
        "candidate_receipt_path": str(root / f"candidate-{epoch}.json"),
        "authority_path": str(root / f"authority-{epoch}.json"),
        "authorization_path": str(root / f"authorization-{epoch}.json"),
        "run_root": str(root / f"run-e{epoch}"),
        "measurement_path": str(root / f"run-e{epoch}/candidates/e{epoch}/live-measurement.json"),
        "completion_path": str(root / f"completion-{epoch}.json"),
        "runner_status_path": str(root / f"status-{epoch}.json"),
        "runner_log_path": str(root / f"log-{epoch}.log"),
        "candidate_receipt": artifact(str(root / f"candidate-{epoch}.json"), "a", 17),
        "work_authority": artifact(str(root / f"authority-{epoch}.json"), "b", 18),
        "authorization": artifact(str(root / f"authorization-{epoch}.json"), "c", 19),
    }


class WavePlanTests(unittest.TestCase):
    def test_exact_twenty_plus_one_plan(self):
        self.assertEqual(
            dual.fresh_waves(),
            [[2, 4], [8, 16], [32, 40], [50, 60], [70, 80],
             [100, 120], [140, 160], [180, 200], [240, 280],
             [320, 360], [400]],
        )
        self.assertEqual(sum(map(len, dual.fresh_waves())), 21)

    def test_first_wave_from_adopted_e1(self):
        campaign = {"_jobs": [{"epoch": e} for e in dual.supervisor.CANDIDATE_EPOCHS]}
        index, jobs = dual._expected_wave(campaign, [object()], {"epoch": 2})
        self.assertEqual(index, 1)
        self.assertEqual([job["epoch"] for job in jobs], [2, 4])

    def test_tenth_pair(self):
        campaign = {"_jobs": [{"epoch": e} for e in dual.supervisor.CANDIDATE_EPOCHS]}
        index, jobs = dual._expected_wave(campaign, [object()] * 19, {"epoch": 320})
        self.assertEqual(index, 10)
        self.assertEqual([job["epoch"] for job in jobs], [320, 360])

    def test_terminal_singleton(self):
        campaign = {"_jobs": [{"epoch": e} for e in dual.supervisor.CANDIDATE_EPOCHS]}
        index, jobs = dual._expected_wave(campaign, [object()] * 21, {"epoch": 400})
        self.assertEqual(index, 11)
        self.assertEqual([job["epoch"] for job in jobs], [400])

    def test_all_complete(self):
        campaign = {"_jobs": [{"epoch": e} for e in dual.supervisor.CANDIDATE_EPOCHS]}
        index, jobs = dual._expected_wave(campaign, [object()] * 22, None)
        self.assertEqual(index, 12)
        self.assertEqual(jobs, [])

    def test_partial_pair_is_terminal(self):
        campaign = {"_jobs": [{"epoch": e} for e in dual.supervisor.CANDIDATE_EPOCHS]}
        with self.assertRaisesRegex(dual.DualDispatchError, "partial prior wave"):
            dual._expected_wave(campaign, [object()] * 2, {"epoch": 4})

    def test_wrong_head_is_rejected(self):
        campaign = {"_jobs": [{"epoch": e} for e in dual.supervisor.CANDIDATE_EPOCHS]}
        with self.assertRaisesRegex(dual.DualDispatchError, "queue head"):
            dual._expected_wave(campaign, [object()], {"epoch": 4})


class HashAndClaimTests(unittest.TestCase):
    def test_self_hash_round_trip(self):
        value = dual._add_hash({"format": "x", "status": "y"}, "payload_sha256")
        dual._validate_hash(value, "payload_sha256", "fixture")

    def test_rehashed_tamper_is_not_original(self):
        value = dual._add_hash({"format": "x", "status": "y"}, "payload_sha256")
        value["status"] = "z"
        with self.assertRaisesRegex(dual.DualDispatchError, "self-hash changed"):
            dual._validate_hash(value, "payload_sha256", "fixture")

    def test_dispatcher_claim_binds_topology_and_policy(self):
        campaign = {"_artifact": artifact("/campaign", "1", 1)}
        topology = artifact("/topology", "2", 2)
        value = dual._dispatcher_claim_body(campaign, topology, 10.5)
        dual._validate_hash(value, "claim_payload_sha256", "claim")
        self.assertEqual(value["topology"], topology)
        self.assertEqual(value["fresh_candidate_epochs"], list(dual.supervisor.CANDIDATE_EPOCHS[1:]))
        self.assertFalse(value["policy"]["retry_authorized"])
        self.assertFalse(value["policy"]["rerun_authorized"])
        self.assertEqual(value["policy"]["paired_fresh_candidates"], 20)

    def test_claim_is_create_new_then_exact_reload(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            campaign = minimal_campaign(Path(temporary))
            dual_root = dual._dual_root(campaign)
            dual_root.mkdir()
            topology = artifact(str(dual_root / "topology.json"), "2", 2)
            first_artifact, first_value = dual.load_or_create_dispatcher_claim(
                campaign, topology, clock=lambda: 11.0, allow_create=True
            )
            second_artifact, second_value = dual.load_or_create_dispatcher_claim(
                campaign, topology, clock=lambda: 999.0
            )
            self.assertEqual(first_artifact, second_artifact)
            self.assertEqual(first_value, second_value)

    def test_claim_topology_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            campaign = minimal_campaign(Path(temporary))
            dual_root = dual._dual_root(campaign)
            dual_root.mkdir()
            topology = artifact(str(dual_root / "topology.json"), "2", 2)
            dual.load_or_create_dispatcher_claim(
                campaign, topology, clock=lambda: 11.0, allow_create=True
            )
            changed = dict(topology)
            changed["sha256"] = "3" * 64
            with self.assertRaisesRegex(dual.DualDispatchError, "claim changed"):
                dual.load_or_create_dispatcher_claim(campaign, changed)

    def test_missing_dispatcher_claim_cannot_be_recreated_by_dispatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            campaign = minimal_campaign(Path(temporary))
            dual_root = dual._dual_root(campaign)
            dual_root.mkdir()
            topology = artifact(str(dual_root / "topology.json"), "2", 2)
            with self.assertRaisesRegex(
                dual.DualDispatchError, "replacement authority is not authorized"
            ):
                dual.load_or_create_dispatcher_claim(campaign, topology)

    def test_master_and_worker_must_be_distinct_physical_hosts(self):
        master = {
            "hostname": "master-0",
            "machine_id": artifact("/etc/machine-id", "a", 33),
            "host_identity": artifact(dual.endpoint.HOST_IDENTITY_PATH, "c", 37),
        }
        worker = {
            "hostname": "worker-0",
            "machine_id": artifact("/etc/machine-id", "a", 33),
            "host_identity": artifact(dual.endpoint.HOST_IDENTITY_PATH, "d", 37),
        }
        dual._require_distinct_hosts(master, worker)
        same_hostname = dict(worker)
        same_hostname["hostname"] = master["hostname"]
        with self.assertRaisesRegex(dual.DualDispatchError, "hostnames"):
            dual._require_distinct_hosts(master, same_hostname)
        same_identity = dict(worker)
        same_identity["host_identity"] = dict(master["host_identity"])
        with self.assertRaisesRegex(dual.DualDispatchError, "host identity fingerprints"):
            dual._require_distinct_hosts(master, same_identity)

    def test_worker_machine_id_requires_exact_fingerprint(self):
        value = dual._worker_machine_artifact("/etc/machine-id", "a" * 64, 33)
        self.assertEqual(value["bytes"], 33)
        with self.assertRaises(dual.DualDispatchError):
            dual._worker_machine_artifact("/etc/machine-id", "not-a-sha", 33)

    def test_worker_host_identity_requires_exact_fingerprint(self):
        value = dual._worker_host_identity_artifact(
            dual.endpoint.HOST_IDENTITY_PATH, "a" * 64, 37
        )
        self.assertEqual(value["bytes"], 37)
        with self.assertRaises(dual.DualDispatchError):
            dual._worker_host_identity_artifact(
                dual.endpoint.HOST_IDENTITY_PATH, "not-a-sha", 37
            )

    def test_ticket_schema_has_direct_topology_and_endpoint_bindings(self):
        self.assertIn("topology", dual.CENTRAL_TICKET_KEYS)
        self.assertIn("endpoint", dual.CENTRAL_TICKET_KEYS)


class TicketAndArgvTests(unittest.TestCase):
    def test_worker_ticket_and_ssh_argv_are_exact(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            campaign = minimal_campaign(root)
            directory = root / "wave"
            directory.mkdir()
            dynamic = dynamic_job(root, 4)
            topology = artifact(str(root / "topology.json"), "d", 20)
            topology_value = {
                "worker": {
                    "hostname": "worker-0",
                    "machine_id": artifact("/etc/machine-id", "e", 33),
                    "host_identity": artifact(
                        dual.endpoint.HOST_IDENTITY_PATH, "1", 37
                    ),
                    "source_root": campaign["_control_source"]["root"],
                    "source_origin": dual.endpoint.OFFICIAL_ORIGIN,
                    "source_commit": campaign["_control_source"]["commit"],
                    "source_tree": campaign["_control_source"]["tree"],
                    "ssh_host": "worker-0",
                },
                "endpoint": artifact(
                    campaign["_control_source"]["root"]
                    + "/scripts/show_base/base_live_val_remote_endpoint.py", "f", 21
                ),
                "ssh_binary": artifact("/usr/bin/ssh", "0", 22),
                "ssh_options": list(dual.SSH_OPTIONS),
            }
            wave = artifact(str(directory / "wave.claim.json"), "1", 23)
            active = artifact(str(directory / "active.claim.json"), "2", 24)
            claim = artifact(str(root / "job.json"), "3", 25)
            authorization = dynamic["authorization"]
            ticket, argv = dual._worker_ticket(
                campaign, topology_value, topology, wave, active, dynamic,
                claim, authorization, directory,
            )
            _artifact, ticket_value = dual._read_json_artifact(ticket["path"], "ticket")
            self.assertEqual(set(ticket_value), dual.CENTRAL_TICKET_KEYS)
            self.assertEqual(ticket_value["topology"], topology)
            self.assertEqual(ticket_value["endpoint"], topology_value["endpoint"])
            self.assertEqual(
                ticket_value["host_identity"],
                topology_value["worker"]["host_identity"],
            )
            self.assertEqual(argv[:len(dual.SSH_OPTIONS) + 1], ["/usr/bin/ssh", *dual.SSH_OPTIONS])
            self.assertEqual(argv[len(dual.SSH_OPTIONS) + 1], "worker-0")
            position = argv.index("--ticket")
            self.assertEqual(argv[position:position + 2], ["--ticket", ticket["path"]])
            self.assertNotIn("sh", argv)

    def test_argv_sha_is_nul_delimited(self):
        expected = dual.supervisor._sha256(b"a\0b\0")
        self.assertEqual(dual._argv_sha(["a", "b"]), expected)

    def test_remote_token_policy_rejects_spaces(self):
        self.assertIsNone(dual.SAFE_REMOTE_TOKEN.fullmatch("/path/with space"))
        self.assertIsNotNone(dual.SAFE_REMOTE_TOKEN.fullmatch("/safe/path-1"))

    def test_launch_claim_binds_both_commands(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            campaign = minimal_campaign(root)
            directory = root / "wave"
            directory.mkdir()
            dynamic = dynamic_job(root, 2)
            launch, value = dual._launch_claim(
                campaign,
                artifact("/dispatcher", "1", 1),
                artifact("/topology", "2", 2),
                artifact("/wave", "3", 3),
                artifact("/active", "4", 4),
                [dynamic], [artifact("/claim", "5", 5)],
                [dynamic["authorization"]], None, [], directory,
                lambda: 12.0,
            )
            self.assertEqual(value["worker_ssh_argv"], [])
            self.assertEqual(value["master_runner_argv_sha256"], dual._argv_sha(value["master_runner_argv"]))
            self.assertTrue(Path(launch["path"]).exists())

    def test_dispatch_refuses_shallow_endpoint_before_any_topology_read(self):
        campaign = {"_artifact": artifact("/campaign", "1", 1)}
        shallow = frozenset(set(dual.CENTRAL_TICKET_KEYS) - {"topology", "endpoint"})
        with mock.patch.object(dual.endpoint, "TICKET_KEYS", shallow), mock.patch.object(
            dual, "load_topology"
        ) as load:
            with self.assertRaisesRegex(dual.DualDispatchError, "ticket schema"):
                dual.dispatch_next(campaign, artifact("/topology", "2", 2))
        load.assert_not_called()


class WorkerReceiptTests(unittest.TestCase):
    def _fixture(self, root: Path):
        campaign = minimal_campaign(root)
        directory = root / "wave"
        directory.mkdir()
        dynamic = dynamic_job(root, 4)
        topology = artifact(str(root / "topology.json"), "d", 20)
        topology_value = {
            "worker": {
                "hostname": "worker-0",
                "machine_id": artifact("/etc/machine-id", "e", 33),
                "host_identity": artifact(
                    dual.endpoint.HOST_IDENTITY_PATH, "1", 37
                ),
                "source_root": campaign["_control_source"]["root"],
                "source_origin": dual.endpoint.OFFICIAL_ORIGIN,
                "source_commit": campaign["_control_source"]["commit"],
                "source_tree": campaign["_control_source"]["tree"],
                "ssh_host": "worker-0",
            },
            "endpoint": artifact(
                campaign["_control_source"]["root"]
                + "/scripts/show_base/base_live_val_remote_endpoint.py", "f", 21
            ),
            "ssh_binary": artifact("/usr/bin/ssh", "0", 22),
            "ssh_options": list(dual.SSH_OPTIONS),
        }
        wave = artifact(str(directory / "wave.claim.json"), "1", 23)
        active = artifact(str(directory / "active.claim.json"), "2", 24)
        claim = artifact(str(root / "job.json"), "3", 25)
        authorization = dynamic["authorization"]
        ticket, _argv = dual._worker_ticket(
            campaign, topology_value, topology, wave, active, dynamic,
            claim, authorization, directory,
        )
        guards = {str(index): 100 + index for index in range(8)}
        status_artifact = artifact(str(root / "status.json"), "4", 26)
        log_artifact = artifact(str(root / "log.txt"), "5", 27)
        guard_stdout = "PASS " + " ".join(
            f"GPU{index}=PID{guards[str(index)]}" for index in range(8)
        ) + "\n"
        runner_argv = dual.supervisor._runner_argv_v2(dynamic, campaign)
        receipt_value = dual.endpoint._add_self_hash({
            "format": dual.endpoint.RECEIPT_FORMAT,
            "status": "complete",
            "role": "worker",
            "hostname": "worker-0",
            "machine_id": topology_value["worker"]["machine_id"],
            "host_identity": topology_value["worker"]["host_identity"],
            "candidate_epoch": 4,
            "ticket": ticket,
            "campaign": campaign["_artifact"],
            "wave": wave,
            "active_claim": active,
            "job_claim": claim,
            "authorization": authorization,
            "formal_python": campaign["_formal_python"],
            "source_root": topology_value["worker"]["source_root"],
            "source_origin": dual.endpoint.OFFICIAL_ORIGIN,
            "source_commit": topology_value["worker"]["source_commit"],
            "source_tree": topology_value["worker"]["source_tree"],
            "guarded_runner": campaign["_guarded_runner"],
            "guard_verifier": campaign["_guard_verifier"],
            "runner_argv": runner_argv,
            "runner_argv_sha256": dual.endpoint.canonical_json_sha256(runner_argv),
            "runner_status": status_artifact,
            "runner_log": log_artifact,
            "guard_verifier_argv": [
                campaign["_formal_python"]["argv0"],
                campaign["_guard_verifier"]["path"],
                *[str(guards[str(index)]) for index in range(8)],
            ],
            "guard_verifier_stdout": guard_stdout,
            "restored_guards": guards,
            "completed_unix": 30.0,
        }, "receipt_payload_sha256")
        _ticket_artifact, ticket_value = dual._read_json_artifact(ticket["path"], "ticket")
        receipt = dual.supervisor.write_new_json(
            ticket_value["receipt_path"], receipt_value, "worker receipt"
        )
        process = dual.ProcessResult(
            0, dual.endpoint.canonical_json_bytes(receipt), b""
        )
        status_value = {"restored_guards": guards}
        return (
            campaign, topology_value, ticket, wave, active, dynamic, claim,
            authorization, process, status_artifact, status_value, log_artifact,
            receipt_value,
        )

    def test_full_worker_receipt_cross_link_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            values = self._fixture(Path(temporary))
            receipt, receipt_value = dual._validate_worker_receipt(*values[:-1])
            self.assertEqual(receipt_value, values[-1])
            self.assertTrue(Path(receipt["path"]).exists())

    def test_rehashed_worker_receipt_cross_link_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            values = list(self._fixture(Path(temporary)))
            receipt_value = dict(values[-1])
            del receipt_value["receipt_payload_sha256"]
            receipt_value["job_claim"] = artifact("/different", "f", 31)
            receipt_value = dual.endpoint._add_self_hash(
                receipt_value, "receipt_payload_sha256"
            )
            _ticket_artifact, ticket_value = dual._read_json_artifact(
                values[2]["path"], "ticket"
            )
            os.chmod(ticket_value["receipt_path"], 0o600)
            Path(ticket_value["receipt_path"]).write_bytes(
                dual.endpoint.canonical_json_bytes(receipt_value)
            )
            receipt_artifact = dual._artifact_from_path(
                ticket_value["receipt_path"], "tampered receipt"
            )
            values[8] = dual.ProcessResult(
                0, dual.endpoint.canonical_json_bytes(receipt_artifact), b""
            )
            with self.assertRaisesRegex(dual.DualDispatchError, "authority binding"):
                dual._validate_worker_receipt(*values[:-1])


class LaunchTests(unittest.TestCase):
    def test_pair_starts_worker_before_master_and_waits_both(self):
        calls = []

        class FakeProcess:
            def __init__(self, argv, **kwargs):
                calls.append(list(argv))
                self.returncode = 0
                self._stdout = b"worker\n" if argv[0] == "ssh" else b""

            def communicate(self):
                return self._stdout, b""

        with mock.patch.object(dual.subprocess, "Popen", side_effect=FakeProcess):
            master, worker = dual._default_launch(["runner"], ["ssh"])
        self.assertEqual(calls, [["ssh"], ["runner"]])
        self.assertEqual(master, dual.ProcessResult(0, b"", b""))
        self.assertEqual(worker, dual.ProcessResult(0, b"worker\n", b""))

    def test_singleton_never_spawns_worker(self):
        completed = subprocess.CompletedProcess(["runner"], 0, b"", b"")
        with mock.patch.object(dual.subprocess, "run", return_value=completed) as run:
            master, worker = dual._default_launch(["runner"], [])
        run.assert_called_once()
        self.assertEqual(master.returncode, 0)
        self.assertIsNone(worker)

    def test_master_spawn_failure_waits_worker_without_kill(self):
        worker = mock.Mock()
        worker.communicate.return_value = (b"", b"")
        with mock.patch.object(
            dual.subprocess, "Popen", side_effect=[worker, OSError("spawn")]
        ):
            with self.assertRaisesRegex(OSError, "spawn"):
                dual._default_launch(["runner"], ["ssh"])
        worker.communicate.assert_called_once_with()
        self.assertFalse(worker.kill.called)
        self.assertFalse(worker.terminate.called)


class PublicationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        campaign = minimal_campaign(root)
        directory = root / "wave"
        directory.mkdir()
        dynamics = [dynamic_job(root, 2), dynamic_job(root, 4)]
        completions = [
            dual.supervisor._add_self_hash(
                {"format": "completion", "epoch": dynamic["epoch"]}, "receipt_payload_sha256"
            )
            for dynamic in dynamics
        ]
        evidences = [
            {
                "runner_status": artifact(f"/status-{i}", "1", 1),
                "runner_log": artifact(f"/log-{i}", "2", 2),
                "measurement": artifact(f"/measurement-{i}", "3", 3),
                "bridge_replay_argv": ["python", "bridge"],
                "guard_proof": {
                    "guard_verifier_argv": ["python", "verify"],
                    "guard_verifier_stdout": "PASS\n",
                    "restored_guards": {str(gpu): 100 + gpu for gpu in range(8)},
                },
                "worker_endpoint_receipt": None,
            }
            for i in range(2)
        ]
        return campaign, directory, dynamics, completions, evidences

    def test_both_completions_publish_only_after_prepared_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            campaign, directory, dynamics, completions, evidences = self._fixture(root)
            with mock.patch.object(dual, "_replay_publication_inputs"), mock.patch.object(
                dual.supervisor, "_load_completion_v2",
                side_effect=lambda _campaign, dynamic: (
                    dual._artifact_from_path(dynamic["completion_path"], "completion replay"),
                    {},
                ),
            ):
                commit, published = dual._publish_wave(
                    campaign, artifact("/dispatcher", "1", 1),
                    artifact("/topology", "2", 2), artifact("/wave", "3", 3),
                    artifact("/launch", "4", 4), dynamics,
                    [artifact("/claim-1", "5", 5), artifact("/claim-2", "6", 6)],
                    [artifact("/auth-1", "7", 7), artifact("/auth-2", "8", 8)],
                    artifact("/ticket", "9", 9), artifact("/receipt", "a", 10),
                    completions, evidences, directory, lambda: 20.0,
                )
            self.assertEqual(len(published), 2)
            self.assertTrue(all(Path(item["path"]).exists() for item in published))
            self.assertTrue(Path(commit["path"]).exists())
            self.assertTrue((directory / "validation.json").exists())
            for dynamic in dynamics:
                self.assertTrue(
                    (directory / f"prepared-completion-epoch-{dynamic['epoch']:04d}.json").exists()
                )

    def test_preexisting_second_completion_prevents_first_publication(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            campaign, directory, dynamics, completions, evidences = self._fixture(root)
            Path(dynamics[1]["completion_path"]).write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(dual.DualDispatchError, "appeared"):
                dual._publish_wave(
                    campaign, artifact("/dispatcher", "1", 1),
                    artifact("/topology", "2", 2), artifact("/wave", "3", 3),
                    artifact("/launch", "4", 4), dynamics, [], [], None, None,
                    completions, evidences, directory, lambda: 20.0,
                )
            self.assertFalse(Path(dynamics[0]["completion_path"]).exists())
            self.assertFalse((directory / "validation.json").exists())


class ParserAndSafetyTests(unittest.TestCase):
    def test_parser_disables_abbreviation(self):
        with self.assertRaises(SystemExit):
            dual._parser().parse_args(["dispatch-next", "--camp", "x"])

    def test_dispatch_command_requires_topology_triple(self):
        parser = dual._parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "dispatch-next", "--campaign", "/c",
                "--expected-campaign-sha256", "a" * 64,
                "--expected-campaign-bytes", "1",
            ])

    def test_no_shell_kill_or_process_search_in_dispatcher(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)
        self.assertNotIn("pgrep", source)
        self.assertNotIn("kill(", source)
        self.assertNotIn("terminate(", source)

    def test_only_two_explicit_unlinks_release_success_locks(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(source.count("path.unlink()"), 1)
        self.assertIn("_release_active_v2", source)


if __name__ == "__main__":
    unittest.main()
