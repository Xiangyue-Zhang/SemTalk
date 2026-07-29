from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "run_lower_target_cache_formal_gate.py"
)
BUILDER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "build_lower_target_joints_cache.py"
)
BUILDER_WRAPPER = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "run_lower_target_cache_builder.py"
)
BUILDER_BOOTSTRAP = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "run_lower_target_cache_builder_guarded.sh"
)


def load_gate():
    specification = importlib.util.spec_from_file_location(
        "lower_target_cache_formal_gate",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class StaticProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = load_gate()

    def test_frozen_protocol_is_exact(self) -> None:
        protocol = self.gate.validate_static_gate_contract()
        self.assertEqual(
            protocol,
            {
                "speaker_scope": "All",
                "speakers": {
                    "oliver": 0,
                    "chemistry": 1,
                    "seth": 2,
                    "conan": 3,
                },
                "equivalence_updates": 2,
                "equivalence_fresh_processes": 4,
                "equivalence_repeats_per_mode": 2,
                "abba_order": ["legacy", "cache", "cache", "legacy"],
                "benchmark_fresh_processes": 4,
                "warmup_updates_per_block": 5,
                "measured_updates_per_block": 25,
                "minimum_speedup": 1.05,
                "training_updates": 600 * 1_988,
            },
        )

    def test_script_has_no_out_of_scope_import_or_launcher(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = (
            "semgate",
            "sparse_trainer",
            "speaker2",
            "beat2",
            "globaldiff_guarded_runner.py",
        )
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append((node.module or "").lower())
        for token in forbidden:
            self.assertFalse(
                any(token in imported for imported in imports),
                (token, imports),
            )
        self.assertNotIn("kill(", source)
        self.assertNotIn("pgrep", source)

    def test_external_wrapper_times_exact_full_child_lifetime(self) -> None:
        source = BUILDER_WRAPPER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        run_builder = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_builder"
        )
        body = ast.unparse(run_builder)
        self.assertLess(
            body.index("started_unix_ns = time.time_ns()"),
            body.index("process = subprocess.Popen"),
        )
        self.assertLess(
            body.index("process = subprocess.Popen"),
            body.index("return_code = process.wait()"),
        )
        self.assertLess(
            body.index("return_code = process.wait()"),
            body.index("completed_monotonic_ns = time.perf_counter_ns()"),
        )
        self.assertIn(
            '"immediately_before_popen_through_complete_child_exit"',
            source,
        )
        self.assertIn('"independent_checker"', source)
        self.assertIn('"guard_restore"', source)
        self.assertIn("sys.executable,", source)
        self.assertNotIn("Path(sys.executable).resolve()", source)
        bootstrap = BUILDER_BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("runner_pid=$PPID", bootstrap)
        self.assertIn("exec \"$python_bin\" \"$wrapper\"", bootstrap)
        self.assertIn("0,1,2,3,4,5,6,7", bootstrap)
        gate_source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"source": "external_full_child_process_receipt"',
            gate_source,
        )
        self.assertNotIn("compute_started_unix", gate_source)

    def test_every_gate_phase_uses_a_fresh_child_contract(self) -> None:
        self.assertEqual(
            self.gate.EQUIVALENCE_CHILDREN,
            (
                ("equivalence-legacy-0", "legacy", 0),
                ("equivalence-legacy-1", "legacy", 1),
                ("equivalence-cache-0", "cache", 0),
                ("equivalence-cache-1", "cache", 1),
            ),
        )
        self.assertEqual(
            self.gate.BENCHMARK_CHILDREN,
            (
                ("benchmark-0-legacy", "legacy", 0),
                ("benchmark-1-cache", "cache", 1),
                ("benchmark-2-cache", "cache", 2),
                ("benchmark-3-legacy", "legacy", 3),
            ),
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_observe_completed_formal_updates(", source)
        self.assertIn("initial_whole_state_exact", source)
        self.assertNotIn(
            '("equivalence-legacy", "equivalence")',
            source,
        )

    def test_cache_gate_remains_optional_but_formal_lower_is_live(self) -> None:
        launcher = (
            REPOSITORY
            / "scripts"
            / "show_base"
            / "run_five_prerequisites.sh"
        ).read_text(encoding="utf-8")
        formal = (REPOSITORY / "show_base_train.py").read_text(
            encoding="utf-8"
        )
        config = (REPOSITORY / "utils" / "config.py").read_text(
            encoding="utf-8"
        )
        for option in (
            "--lower_target_cache_gate_report",
            "--expected_lower_target_cache_gate_sha256",
            "--lower_target_cache_builder_process_receipt",
            "--expected_lower_target_cache_builder_process_receipt_sha256",
        ):
            self.assertNotIn(option, launcher)
            self.assertIn(option, config)
        self.assertNotIn("LOWER_TARGET_GATE", launcher)
        self.assertIn("--use_lower_target_joints_cache false", launcher)
        self.assertNotIn("--use_lower_target_joints_cache true", launcher)
        self.assertIn(
            "_formal_lower_target_cache_gate_receipt(",
            formal,
        )
        self.assertIn(
            "_formal_lower_target_backend_receipt(",
            formal,
        )
        self.assertIn(
            'lower_target_cache_receipt["formal_gate"]',
            formal,
        )
        self.assertIn(
            "initial_and_two_complete_updates_and_final_byte_exact",
            formal,
        )
        self.assertIn("repeat_controls_exact", formal)
        self.assertIn("initial_whole_state_exact", formal)
        self.assertIn("builder-amortized wall", formal)


class SpeedupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = load_gate()

    def blocks(
        self,
        legacy_wall: float = 2.0,
        cache_wall: float = 1.0,
        legacy_cuda: float = 1.8,
        cache_cuda: float = 0.9,
    ):
        values = {
            "legacy": (legacy_wall, legacy_cuda),
            "cache": (cache_wall, cache_cuda),
        }
        result = []
        for mode in ("legacy", "cache", "cache", "legacy"):
            wall, cuda = values[mode]
            result.append(
                {
                    "mode": mode,
                    "warmup_updates": 5,
                    "measured_updates": 25,
                    "wall_seconds": [wall] * 25,
                    "cuda_event_seconds": [cuda] * 25,
                }
            )
        return result

    def test_pooled_block_and_builder_amortized_pass(self) -> None:
        report = self.gate.calculate_speedup_report(
            self.blocks(),
            full_builder_seconds=100.0,
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["block_pairs"]), 2)
        self.assertGreaterEqual(report["pooled"]["wall"]["speedup"], 1.05)
        self.assertGreaterEqual(
            report["pooled"]["cuda_event"]["speedup"],
            1.05,
        )
        self.assertEqual(
            report["amortization"]["training_updates"],
            600 * 1_988,
        )
        self.assertGreaterEqual(
            report["amortization"]["speedup"],
            1.05,
        )

    def test_every_threshold_fails_closed(self) -> None:
        cases = []
        pooled_wall = self.blocks()
        pooled_wall[0]["wall_seconds"] = [1.0] * 25
        pooled_wall[3]["wall_seconds"] = [1.0] * 25
        cases.append(pooled_wall)

        pooled_cuda = self.blocks()
        pooled_cuda[0]["cuda_event_seconds"] = [0.9] * 25
        pooled_cuda[3]["cuda_event_seconds"] = [0.9] * 25
        cases.append(pooled_cuda)

        paired_wall = self.blocks()
        paired_wall[0]["wall_seconds"] = [1.0] * 25
        cases.append(paired_wall)

        paired_cuda = self.blocks()
        paired_cuda[3]["cuda_event_seconds"] = [0.8] * 25
        cases.append(paired_cuda)

        for blocks in cases:
            with self.subTest(case=cases.index(blocks)):
                with self.assertRaises(self.gate.GateError):
                    self.gate.calculate_speedup_report(
                        blocks,
                        full_builder_seconds=100.0,
                    )

        with self.assertRaises(self.gate.GateError):
            self.gate.calculate_speedup_report(
                self.blocks(
                    legacy_wall=1.051,
                    cache_wall=1.0,
                ),
                full_builder_seconds=100_000.0,
            )

    def test_wrong_abba_counts_or_nonfinite_samples_fail(self) -> None:
        wrong_order = self.blocks()
        wrong_order[1]["mode"] = "legacy"
        with self.assertRaises(self.gate.GateError):
            self.gate.calculate_speedup_report(
                wrong_order,
                full_builder_seconds=1.0,
            )
        wrong_count = self.blocks()
        wrong_count[0]["wall_seconds"] = [2.0] * 24
        with self.assertRaises(self.gate.GateError):
            self.gate.calculate_speedup_report(
                wrong_count,
                full_builder_seconds=1.0,
            )
        nonfinite = self.blocks()
        nonfinite[0]["cuda_event_seconds"][0] = float("nan")
        with self.assertRaises(self.gate.GateError):
            self.gate.calculate_speedup_report(
                nonfinite,
                full_builder_seconds=1.0,
            )


class BuilderProcessReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = load_gate()

    def payload(self):
        runner_argv = [
            "/venv/bin/python",
            self.gate.EXPECTED_RUNNER,
            "--gpus",
            self.gate.EXPECTED_RUNNER_GPUS,
        ]
        wrapper_argv = [
            "/venv/bin/python",
            str(BUILDER_WRAPPER.resolve()),
            "--formal-builder",
        ]
        builder_argv = [
            "/venv/bin/python",
            str(BUILDER.resolve()),
            "--formal-builder",
        ]
        source = {
            "origin": self.gate.EXPECTED_ORIGIN,
            "commit": "a" * 40,
            "tree": "b" * 40,
        }
        manifest_path = Path("/formal/cache/manifest.json")
        manifest = {
            "argv": builder_argv[1:],
            "entry_aggregate_sha256": "c" * 64,
        }
        receipt_source = {
            **source,
            "entrypoint": str(BUILDER_WRAPPER.resolve()),
            "entrypoint_sha256": self.gate.sha256_file(
                BUILDER_WRAPPER.resolve()
            ),
            "builder_entrypoint": str(BUILDER.resolve()),
            "builder_entrypoint_sha256": self.gate.sha256_file(
                BUILDER.resolve()
            ),
            "bootstrap_entrypoint": str(BUILDER_BOOTSTRAP.resolve()),
            "bootstrap_entrypoint_sha256": self.gate.sha256_file(
                BUILDER_BOOTSTRAP.resolve()
            ),
        }
        payload = {
            "format": self.gate.BUILDER_PROCESS_FORMAT,
            "status": "complete",
            "scope": {
                "dataset": "show_base",
                "formal_stage": "lower",
                "speaker_scope": "All",
                "speaker_ids": [0, 1, 2, 3],
            },
            "source_receipt": receipt_source,
            "guarded_runner": {
                "pid": 10,
                "ppid": 1,
                "state": "S",
                "starttime": "100",
                "argv": runner_argv,
                "argv_sha256": self.gate.argv_sha256(runner_argv),
            },
            "wrapper_process": {
                "pid": 20,
                "ppid": 10,
                "state": "R",
                "starttime": "200",
                "argv": wrapper_argv,
                "argv_sha256": self.gate.argv_sha256(wrapper_argv),
            },
            "builder_process": {
                "pid": 30,
                "ppid": 20,
                "starttime": "300",
                "argv": builder_argv,
                "argv_sha256": self.gate.argv_sha256(builder_argv),
                "observed_argv_sha256": self.gate.argv_sha256(builder_argv),
                "return_code": 0,
                "cuda_visible_devices": "0",
                "started_unix_ns": 1_000,
                "completed_unix_ns": 3_000,
                "started_monotonic_ns": 10_000,
                "completed_monotonic_ns": 30_000,
                "elapsed_monotonic_ns": 20_000,
                "elapsed_seconds": 0.00002,
                "timing_scope": (
                    "immediately_before_popen_through_complete_child_exit"
                ),
                "excludes": [
                    "independent_checker",
                    "formal_gate",
                    "guard_restore",
                    "receipt_validation_and_write",
                ],
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": "d" * 64,
                "status": "complete",
                "entries": self.gate.EXPECTED_ENTRIES,
                "entry_aggregate_sha256": "c" * 64,
            },
            "completed_unix_ns": 4_000,
        }
        return payload, manifest_path, manifest, source

    def validate(self, payload):
        _, manifest_path, manifest, source = self.payload()
        return self.gate.validate_builder_process_payload(
            payload,
            receipt_sha256="e" * 64,
            manifest_path=manifest_path,
            manifest_sha256="d" * 64,
            manifest=manifest,
            source=source,
        )

    def test_external_full_process_receipt_passes(self) -> None:
        payload, _, _, _ = self.payload()
        result = self.validate(payload)
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["elapsed_monotonic_ns"], 20_000)
        self.assertEqual(result["elapsed_seconds"], 0.00002)

    def test_external_receipt_mutations_fail_closed(self) -> None:
        payload, _, _, _ = self.payload()

        def runner_not_all_gpus(value) -> None:
            runner = value["guarded_runner"]
            runner["argv"][3] = "0"
            runner["argv_sha256"] = self.gate.argv_sha256(runner["argv"])

        def different_interpreter(value) -> None:
            process = value["builder_process"]
            process["argv"][0] = "/usr/bin/python3"
            process["argv_sha256"] = self.gate.argv_sha256(process["argv"])
            process["observed_argv_sha256"] = process["argv_sha256"]

        mutations = {
            "runner_not_all_gpus": runner_not_all_gpus,
            "runner_argv_sha": lambda value: value[
                "guarded_runner"
            ].__setitem__("argv_sha256", "f" * 64),
            "builder_return_code": lambda value: value[
                "builder_process"
            ].__setitem__("return_code", 1),
            "elapsed_not_full_interval": lambda value: value[
                "builder_process"
            ].__setitem__("elapsed_monotonic_ns", 19_999),
            "different_interpreter": different_interpreter,
        }
        for label, mutate in mutations.items():
            changed = copy.deepcopy(payload)
            mutate(changed)
            with self.subTest(label=label):
                with self.assertRaises(self.gate.GateError):
                    self.validate(changed)


class AtomicFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = load_gate()

    def test_atomic_json_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            self.gate.atomic_json_new(path, {"status": "pass"})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.gate.atomic_json_new(path, {"status": "changed"})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
