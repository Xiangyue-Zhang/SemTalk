from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT = (
    REPOSITORY
    / "scripts"
    / "show_base"
    / "formal_python_runtime_contract.sh"
)
LAUNCHERS = (
    "run_base_diffsheg_val_8shard.sh",
    "finalize_base_diffsheg_val_partitions.sh",
    "run_base_final_test.sh",
    "run_dual_node_guarded_transaction.sh",
    "run_diffsheg_final_test_eval.sh",
)


class FormalPythonRuntimeContractTests(unittest.TestCase):
    def _new_venv(self, root: Path, name: str = "venv") -> Path:
        venv_root = (root / name).resolve()
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(venv_root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        python = venv_root / "bin" / "python"
        self.assertTrue(python.is_symlink(), python)
        return python

    def _contract(self, python: Path, profile: str = "minimal") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-eu",
                "-o",
                "pipefail",
                "-c",
                (
                    'source "$1"; python_bin=$2; '
                    'semtalk_require_formal_venv_python "$python_bin" "$3"; '
                    'printf "%s\\n" "$python_bin"; '
                    '"$python_bin" -I -c "import sys; print(sys.executable)"'
                ),
                "formal-python-contract-test",
                str(CONTRACT),
                str(python),
                profile,
            ],
            check=False,
            cwd=REPOSITORY,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_real_venv_python_symlink_is_accepted_and_argv_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            result = self._contract(python)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(python), str(python)])

    def test_two_hop_venv_python_chain_preserves_exact_leaf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            resolved = python.resolve(strict=True)
            intermediate = python.with_name("python-formal-target")
            python.unlink()
            intermediate.symlink_to(resolved)
            python.symlink_to(intermediate.name)
            result = self._contract(python)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(python), str(python)])

    def test_two_hop_intermediate_redirect_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            intermediate = python.with_name("python-formal-target")
            python.unlink()
            intermediate.symlink_to("/bin/sh")
            python.symlink_to(intermediate.name)
            result = self._contract(python)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr)

    def test_resolved_base_interpreter_is_rejected(self) -> None:
        base = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        result = self._contract(base)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            "venv/bin" in result.stderr or "pyvenv.cfg" in result.stderr,
            result.stderr,
        )

    def test_noncanonical_parent_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            alias = root / "venv-alias"
            os.symlink(python.parents[1], alias, target_is_directory=True)
            result = self._contract(alias / "bin" / "python")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("canonical", result.stderr)

    def test_symlinked_pyvenv_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            configuration = python.parents[1] / "pyvenv.cfg"
            copied = root / "copied-pyvenv.cfg"
            copied.write_bytes(configuration.read_bytes())
            configuration.unlink()
            configuration.symlink_to(copied)
            result = self._contract(python)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pyvenv.cfg", result.stderr)

    def test_formal_semtalk_profile_rejects_dependency_empty_venv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            result = self._contract(python, "semtalk")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required module", result.stderr)

    def test_legacy_pyvenv_without_executable_field_remains_strictly_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            configuration = python.parents[1] / "pyvenv.cfg"
            lines = configuration.read_text(encoding="utf-8").splitlines()
            configuration.write_text(
                "\n".join(
                    line for line in lines
                    if not line.casefold().startswith("executable =")
                )
                + "\n",
                encoding="utf-8",
            )
            result = self._contract(python)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(python), str(python)])

    def test_optional_pyvenv_executable_field_must_match_supplied_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="semtalk-python-contract-") as raw:
            root = Path(raw).resolve()
            python = self._new_venv(root)
            configuration = python.parents[1] / "pyvenv.cfg"
            lines = configuration.read_text(encoding="utf-8").splitlines()
            retained = [
                line
                for line in lines
                if not line.casefold().startswith("executable =")
            ]
            configuration.write_text(
                "\n".join((*retained, "executable = /bin/sh")) + "\n",
                encoding="utf-8",
            )
            result = self._contract(python)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match", result.stderr)


class FormalPythonLauncherStaticTests(unittest.TestCase):
    def test_affected_launchers_use_shared_contract(self) -> None:
        directory = REPOSITORY / "scripts" / "show_base"
        for name in LAUNCHERS:
            source = (directory / name).read_text(encoding="utf-8")
            self.assertIn("formal_python_runtime_contract.sh", source, name)
            self.assertIn(
                'semtalk_require_formal_venv_python "$python_bin" semtalk',
                source,
                name,
            )

    def test_all_formal_shell_launchers_preserve_python_leaf_argv(self) -> None:
        directory = REPOSITORY / "scripts" / "show_base"
        for path in sorted(directory.glob("*.sh")):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("python_bin=$(realpath", source, path.name)
            self.assertNotIn('-L "$python_bin"', source, path.name)
            self.assertNotIn("-L $python_bin", source, path.name)

    def test_worker_and_finalizer_bind_shared_contract_as_tracked_source(self) -> None:
        directory = REPOSITORY / "scripts" / "show_base"
        for name in LAUNCHERS[:2]:
            source = (directory / name).read_text(encoding="utf-8")
            self.assertIn(
                "scripts/show_base/formal_python_runtime_contract.sh",
                source,
                name,
            )

    def test_contract_checks_runtime_identity_and_required_modules(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        for token in (
            "sys.executable",
            "sys.prefix",
            "sys.exec_prefix",
            "sys.base_prefix",
            "Path(expected).resolve(strict=True)",
            "pyvenv.cfg",
            '"numpy", "torch", "scipy", "einops", "smplx", "transformers", "lmdb"',
        ):
            self.assertIn(token, source)
        self.assertNotIn("sys._base_executable", source)


if __name__ == "__main__":
    unittest.main()
