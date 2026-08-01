from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

from scripts.show_base import base_final_authority as authority
from scripts.show_base import run_base_val_inference as producer
from scripts.show_base import semtalk_base_inference_core as core
from scripts.show_base import talkshow_base_val_contract as contract


ROOT = Path(__file__).resolve().parents[1]
CORE_RELATIVE = "scripts/show_base/semtalk_base_inference_core.py"
LEGACY_RELATIVE = "scripts/show_base/run_base_" + "inference.py"
INTERNAL_RELEASED2_EVALUATORS = frozenset(
    {
        "scripts/show_base/evaluate_talkshow_show_metrics.py",
        "scripts/show_base/replay_released2_primary.py",
    }
)
FORMAL_DIFFSHEG_ROOTS = frozenset(
    {
        "scripts/show_base/evaluate_diffsheg_val_fgd.py",
        "scripts/show_base/evaluate_diffsheg_final_test.py",
    }
)


def local_show_base_imports(relative: str) -> set[str]:
    syntax = ast.parse(
        (ROOT / relative).read_text(encoding="utf-8"),
        relative,
    )
    modules: set[str] = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module == "scripts.show_base":
                names = [
                    f"scripts.show_base.{alias.name}"
                    for alias in node.names
                ]
            elif (node.module or "").startswith("scripts.show_base."):
                names = [node.module or ""]
            else:
                names = []
        else:
            continue
        for name in names:
            prefix = "scripts.show_base."
            if not name.startswith(prefix):
                continue
            module = name[len(prefix):].split(".", 1)[0]
            candidate = f"scripts/show_base/{module}.py"
            if (ROOT / candidate).is_file():
                modules.add(candidate)
    for node in ast.walk(syntax):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        loader = node.func
        loader_name = (
            loader.id
            if isinstance(loader, ast.Name)
            else loader.attr
            if isinstance(loader, ast.Attribute)
            else ""
        )
        argument = node.args[0]
        if (
            loader_name
            not in {"import_module", "_fresh_local_module", "_control_module"}
            or not isinstance(argument, ast.Constant)
            or not isinstance(argument.value, str)
        ):
            continue
        module = argument.value.removeprefix("scripts.show_base.")
        candidate = f"scripts/show_base/{module}.py"
        if (ROOT / candidate).is_file():
            modules.add(candidate)
    return modules


def local_show_base_import_closure(roots: set[str]) -> set[str]:
    pending = list(roots)
    closure: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        closure.add(relative)
        pending.extend(local_show_base_imports(relative) - closure)
    return closure


def source_entry(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "git_mode": "100644",
        "git_blob_sha1": producer._git_blob_sha1(payload),
    }


class NeutralBaseInferenceClosureTest(unittest.TestCase):
    def test_every_formal_source_file_excludes_legacy_tokens_and_path(
        self,
    ) -> None:
        self.assertIn(CORE_RELATIVE, contract.FRESH_PIPELINE_SOURCE_FILES)
        self.assertNotIn(
            LEGACY_RELATIVE,
            contract.FRESH_PIPELINE_SOURCE_FILES,
        )
        # Both TalkSHOW metric modules remain hash-pinned.  The metric adapter
        # is now the sole in-process TalkSHOW suite in the combined final
        # event; only the archived replay CLI remains compatibility-only.
        self.assertTrue(
            INTERNAL_RELEASED2_EVALUATORS.issubset(
                contract.FRESH_PIPELINE_SOURCE_FILES
            )
        )

        control_modules = set(authority._CONTROL_DEPENDENCIES)
        for dependencies in authority._CONTROL_DEPENDENCIES.values():
            control_modules.update(dependencies)
        formal_roots = {
            "scripts/show_base/base_final_authority.py",
            *(f"scripts/show_base/{name}.py" for name in control_modules),
            "scripts/show_base/base_long_val_contract.py",
            "scripts/show_base/produce_base_val_measurement.py",
            "scripts/show_base/run_base_val_inference.py",
            "scripts/show_base/run_base_final_test.py",
            CORE_RELATIVE,
            *FORMAL_DIFFSHEG_ROOTS,
        }
        formal_closure = local_show_base_import_closure(formal_roots)
        self.assertTrue(FORMAL_DIFFSHEG_ROOTS.issubset(formal_closure))
        self.assertEqual(
            INTERNAL_RELEASED2_EVALUATORS & formal_closure,
            {"scripts/show_base/evaluate_talkshow_show_metrics.py"},
        )
        self.assertNotIn(LEGACY_RELATIVE, formal_closure)
        self.assertEqual(
            {
                relative
                for relative in formal_closure
                if Path(relative).name.startswith("evaluate_")
            },
            set(FORMAL_DIFFSHEG_ROOTS)
            | {"scripts/show_base/evaluate_talkshow_show_metrics.py"},
        )

    def test_neutral_core_exports_exact_pinned_callable_closure(self) -> None:
        required = set(contract.INFERENCE_HELPERS) | {
            "_strict_load_freeze_eval",
            "_normalize_data_parallel_state",
            "_read_verified_checkpoint_snapshot",
            "_torch_load_checkpoint",
            "_finite_state_dict",
            "_validate_base_model_state_schema",
            "_model_args",
            "_joint_masks",
            "deterministic_npz_bytes",
            "RELEASED_ALL_SPEAKERS_MODELS",
            "OFFICIAL_SHOW_ADAPT_BASE_CHECKPOINT_FORMAT",
            "PRE_FRAMES",
            "STRIDE",
            "POSE_DIM",
        }
        self.assertFalse(
            sorted(name for name in required if not hasattr(core, name))
        )
        rvq_source = "".join(inspect.getsourcelines(core._rvq_indices)[0])
        infer_source = "".join(inspect.getsourcelines(core._infer_clip)[0])
        self.assertEqual(
            hashlib.sha256(rvq_source.encode("utf-8")).hexdigest(),
            core.PINNED_RVQ_INDICES_SOURCE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(infer_source.encode("utf-8")).hexdigest(),
            core.PINNED_INFER_CLIP["source_sha256"],
        )

        syntax = ast.parse(
            (ROOT / CORE_RELATIVE).read_text(encoding="utf-8"),
            CORE_RELATIVE,
        )
        imported = {
            alias.name
            for node in ast.walk(syntax)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(syntax)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn(
            "scripts.show_base.run_base_" + "inference",
            imported,
        )

    def test_verified_neutral_snapshot_is_the_only_loaded_helper(self) -> None:
        core_path = (ROOT / CORE_RELATIVE).resolve()
        entry = source_entry(core_path)
        pipeline = {
            "source_closure": {CORE_RELATIVE: entry},
            "inference_helper": entry,
        }
        helper = producer._load_pinned_helper(pipeline)
        self.assertEqual(Path(helper.__file__).resolve(), core_path)
        self.assertEqual(helper.POSE_DIM, 165)
        self.assertTrue(callable(helper._infer_clip))

    def test_legacy_snapshot_substitution_is_rejected_before_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="semtalk-neutral-core-attack-",
            dir="/private/tmp",
        ) as raw:
            root = Path(raw)
            helper = root / LEGACY_RELATIVE
            helper.parent.mkdir(parents=True)
            marker = root / "executed"
            helper.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n",
                encoding="utf-8",
            )
            entry = source_entry(helper)
            pipeline = {
                "source_closure": {LEGACY_RELATIVE: entry},
                "inference_helper": entry,
            }
            with self.assertRaisesRegex(
                producer.ValInferenceContractError,
                "differs from the fresh source closure",
            ):
                producer._load_pinned_helper(pipeline)
            self.assertFalse(marker.exists())

    def test_final_authority_rejects_legacy_source_schema(self) -> None:
        with self.assertRaisesRegex(
            authority.BaseFinalAuthorityError,
            "not SemTalk official Base inference",
        ):
            authority._validate_source(
                {
                    "source_root": "/private/tmp",
                    "origin": authority.ORIGIN,
                    "commit": "1" * 40,
                    "tree": "2" * 40,
                    "clean": True,
                    "entrypoint": LEGACY_RELATIVE,
                    "entrypoint_sha256": "3" * 64,
                }
            )


if __name__ == "__main__":
    unittest.main()
