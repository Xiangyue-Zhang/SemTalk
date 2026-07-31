from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from utils.smplx_training import (
    SmplxTrainingPool,
    build_smplx_training_pool,
    clip_aligned_spans,
    combine_local_loss_parts,
    parse_smplx_helper_devices,
    smplx_local_loss_numerators,
)


REPOSITORY = Path(__file__).resolve().parents[1]


class SmplxTrainingPoolCpuTest(unittest.TestCase):
    def test_clip_aligned_spans_are_balanced_complete_and_ordered(self) -> None:
        self.assertEqual(
            clip_aligned_spans(35, 5, 3),
            (
                (0, 15, 0, 3),
                (15, 25, 3, 5),
                (25, 35, 5, 7),
            ),
        )
        for invalid in (
            (0, 5, 2),
            (35, 0, 2),
            (35, 5, 0),
            (34, 5, 2),
            (10, 5, 3),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    clip_aligned_spans(*invalid)

    @staticmethod
    def _reference_components(
        stage: str,
        rec: torch.Tensor,
        target: torch.Tensor,
        *,
        clip_length: int,
        static_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        components = {
            "ver": torch.nn.functional.mse_loss(rec, target)
        }
        if stage in {"face", "hands", "upper"}:
            velocity_rec = rec[:, 1:] - rec[:, :-1]
            velocity_target = target[:, 1:] - target[:, :-1]
            acceleration_rec = (
                rec[:, 2:] + rec[:, :-2] - 2 * rec[:, 1:-1]
            )
            acceleration_target = (
                target[:, 2:] + target[:, :-2]
                - 2 * target[:, 1:-1]
            )
            loss = (
                torch.nn.functional.mse_loss
                if stage == "face"
                else torch.nn.functional.l1_loss
            )
            components["ver_vel"] = loss(
                velocity_rec,
                velocity_target,
            )
            components["ver_acc"] = loss(
                acceleration_rec,
                acceleration_target,
            )
        else:
            assert static_mask is not None
            clip_count = rec.shape[0] // clip_length
            model_feet = rec.reshape(
                clip_count,
                clip_length,
                -1,
                3,
            )[:, :, (7, 8, 10, 11)]
            model_foot_v = torch.zeros_like(model_feet)
            model_foot_v[:, :-1] = (
                model_feet[:, 1:] - model_feet[:, :-1]
            )
            model_foot_v[~static_mask] = 0
            components["foot"] = torch.nn.functional.l1_loss(
                model_foot_v,
                torch.zeros_like(model_foot_v),
            )
        return components

    def test_sharded_local_losses_match_original_means_and_gradients(self) -> None:
        torch.manual_seed(17)
        clip_length = 4
        clip_count = 7
        total_rows = clip_length * clip_count
        weights = {
            "ver": 1.3,
            "ver_vel": 0.7,
            "ver_acc": 0.2,
            "foot": 20.0,
        }
        for stage in ("face", "hands", "upper", "lower"):
            with self.subTest(stage=stage):
                item_count = 13 if stage == "lower" else 9
                rec = torch.randn(
                    total_rows,
                    item_count,
                    3,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                target = torch.randn_like(rec).detach()
                static_mask = None
                if stage == "lower":
                    static_mask = (
                        torch.arange(clip_count * clip_length * 4)
                        .reshape(clip_count, clip_length, 4)
                        .remainder(3)
                        .ne(0)
                    )
                parts = []
                for row_start, row_end, clip_start, clip_end in (
                    clip_aligned_spans(total_rows, clip_length, 3)
                ):
                    output_key = (
                        "joints" if stage == "lower" else "vertices"
                    )
                    parts.append(
                        smplx_local_loss_numerators(
                            stage,
                            {output_key: rec[row_start:row_end]},
                            {output_key: target[row_start:row_end]},
                            clip_length=clip_length,
                            static_mask=(
                                None
                                if static_mask is None
                                else static_mask[clip_start:clip_end]
                            ),
                        )
                    )
                combined = combine_local_loss_parts(parts)
                reference = self._reference_components(
                    stage,
                    rec,
                    target,
                    clip_length=clip_length,
                    static_mask=static_mask,
                )
                self.assertEqual(tuple(combined), tuple(reference))
                for name in reference:
                    torch.testing.assert_close(
                        combined[name],
                        reference[name],
                        rtol=1e-12,
                        atol=1e-12,
                    )

                sharded_objective = sum(
                    combined[name] * weights[name] for name in combined
                )
                reference_objective = sum(
                    reference[name] * weights[name] for name in reference
                )
                sharded_gradient = torch.autograd.grad(
                    sharded_objective,
                    rec,
                    retain_graph=True,
                )[0]
                reference_gradient = torch.autograd.grad(
                    reference_objective,
                    rec,
                )[0]
                torch.testing.assert_close(
                    sharded_gradient,
                    reference_gradient,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_helper_parser_is_explicit_and_fail_closed(self) -> None:
        self.assertEqual(parse_smplx_helper_devices("1,3,7"), (1, 3, 7))
        self.assertEqual(parse_smplx_helper_devices([2, 4]), (2, 4))
        self.assertEqual(parse_smplx_helper_devices(""), ())
        for invalid in ("0", "1,1", "-1", "1,,2", "cuda:1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    parse_smplx_helper_devices(invalid)
        with self.assertRaises(TypeError):
            parse_smplx_helper_devices([True])

    def test_disabled_mode_rejects_hidden_helpers_without_cuda(self) -> None:
        args = SimpleNamespace(
            smplx_training_pool_mode="disabled",
            smplx_training_helper_devices="",
        )
        self.assertIsNone(
            build_smplx_training_pool(
                args,
                object(),
                optimizer=object(),
                training_model=object(),
            )
        )
        args.smplx_training_helper_devices = "1"
        with self.assertRaises(ValueError):
            build_smplx_training_pool(
                args,
                object(),
                optimizer=object(),
                training_model=object(),
            )

    def test_full_batch_dual_has_exactly_two_nonduplicating_jobs(self) -> None:
        source = (
            REPOSITORY / "utils" / "smplx_training.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forward_pair = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward_pair"
        )
        segment = ast.get_source_segment(source, forward_pair)
        assert segment is not None
        self.assertIn(
            "rec_device, target_device = self.helper_devices",
            segment,
        )
        full_batch_dual_branch = segment.split(
            'if self.mode == "target_offload":',
            1,
        )[1].split("else:", 1)[1]
        self.assertEqual(
            full_batch_dual_branch.count("expected_rows=total_rows"),
            2,
        )
        self.assertNotIn("repeat(", segment)
        self.assertNotIn("expand(", segment)
        self.assertNotIn("chunk(", segment)
        self.assertNotIn("split(", segment)

        copy_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_copy_full_batch_kwargs"
        )
        copy_segment = ast.get_source_segment(source, copy_method)
        assert copy_segment is not None
        self.assertNotIn("value[", copy_segment)

        init_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and any(
                isinstance(child, ast.Attribute)
                and child.attr == "helper_devices"
                for child in ast.walk(node)
            )
        )
        init_segment = ast.get_source_segment(source, init_method)
        assert init_segment is not None
        self.assertIn(
            "if self.helper_devices != (1, 2):",
            init_segment,
        )
        self.assertIn(
            "if visible_devices != len(self.helper_devices) + 1:",
            init_segment,
        )
        self.assertIn(
            "torch.nn.parallel.DistributedDataParallel",
            init_segment,
        )
        self.assertIn(
            "tuple(training_model.device_ids) != (self.primary_device,)",
            init_segment,
        )

    def test_full_batch_dual_forward_pair_keeps_stage_optional(self) -> None:
        pool = object.__new__(SmplxTrainingPool)
        pool.mode = "full_batch_dual"
        pool.primary_device = 0
        pool.helper_devices = (1, 2)
        pool._forward_rng_audited = True
        pool.last_forward_receipt = None
        pool._validate_kwargs = mock.Mock(return_value=4)
        rec_output = {
            "vertices": torch.ones(
                4,
                2,
                3,
                dtype=torch.float32,
                requires_grad=True,
            )
        }
        target_output = {
            "vertices": torch.zeros(4, 2, 3, dtype=torch.float32)
        }
        pool._schedule = mock.Mock(
            side_effect=[
                (rec_output, mock.sentinel.rec_ready),
                (target_output, mock.sentinel.target_ready),
            ]
        )
        pool._collect_full_batch = mock.Mock(
            side_effect=lambda scheduled: scheduled[0]
        )
        event = mock.Mock()
        with (
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=mock.sentinel.primary_stream,
            ),
            mock.patch.object(torch.cuda, "Event", return_value=event),
        ):
            observed_rec, observed_target = pool.forward_pair(
                rec_kwargs={"pose": mock.sentinel.rec_pose},
                target_kwargs={"pose": mock.sentinel.target_pose},
                clip_length=2,
                output_keys=("vertices",),
            )
        self.assertIs(observed_rec, rec_output)
        self.assertIs(observed_target, target_output)
        self.assertEqual(
            [call.kwargs["device"] for call in pool._schedule.call_args_list],
            [1, 2],
        )
        self.assertIsNone(pool.last_forward_receipt)
        event.record.assert_called_once_with(mock.sentinel.primary_stream)

        pool._schedule.reset_mock(side_effect=True)
        pool._schedule.side_effect = [
            (rec_output, mock.sentinel.rec_ready),
            (target_output, mock.sentinel.target_ready),
        ]
        pool._collect_full_batch.reset_mock()
        event.reset_mock()
        with (
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=mock.sentinel.primary_stream,
            ),
            mock.patch.object(torch.cuda, "Event", return_value=event),
        ):
            pool.forward_pair(
                rec_kwargs={"pose": mock.sentinel.rec_pose},
                target_kwargs={"pose": mock.sentinel.target_pose},
                clip_length=2,
                output_keys=("vertices",),
                stage=None,
            )
        self.assertEqual(
            [call.kwargs["device"] for call in pool._schedule.call_args_list],
            [1, 2],
        )
        self.assertIsNone(pool.last_forward_receipt)

    def test_target_offload_forward_pair_requires_stage_and_receipts_it(
        self,
    ) -> None:
        pool = object.__new__(SmplxTrainingPool)
        pool.mode = "target_offload"
        pool.primary_device = 0
        pool.helper_devices = (1,)
        pool._forward_rng_audited = True
        pool.last_forward_receipt = None
        pool.completed_forward_pairs = 0
        pool._validate_kwargs = mock.Mock(return_value=4)
        sequence: list[str] = []
        rec_output = {
            "vertices": torch.ones(
                4,
                2,
                3,
                dtype=torch.float32,
                requires_grad=True,
            )
        }
        target_output = {
            "vertices": mock.Mock(
                shape=(4, 2, 3),
                requires_grad=False,
            )
        }
        primary_model = mock.Mock(
            side_effect=lambda **_: (
                sequence.append("primary_rec_enqueue") or rec_output
            )
        )
        pool.models = {0: primary_model}
        scheduled = SimpleNamespace(
            outputs=target_output,
            retained_primary_inputs={
                "pose": mock.sentinel.primary_target_pose
            },
            retained_helper_inputs={
                "pose": mock.sentinel.helper_target_pose
            },
            retained_helper_outputs={
                "vertices": mock.sentinel.helper_vertices
            },
            ready=mock.sentinel.return_ready,
        )
        pool._schedule_target_offload = mock.Mock(
            side_effect=lambda **_: (
                sequence.append("target_auxiliary_enqueue") or scheduled
            )
        )

        common = {
            "rec_kwargs": {"pose": mock.sentinel.rec_pose},
            "target_kwargs": {"pose": mock.sentinel.target_pose},
            "clip_length": 2,
            "output_keys": ("vertices",),
        }
        with self.assertRaisesRegex(ValueError, "explicit supported"):
            pool.forward_pair(**common)
        with self.assertRaisesRegex(ValueError, "explicit supported"):
            pool.forward_pair(stage="global", **common)

        primary_stream = mock.Mock()
        primary_stream.wait_event.side_effect = lambda *_: sequence.append(
            "primary_wait_return"
        )
        inputs_ready = mock.Mock()
        with (
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=primary_stream,
            ),
            mock.patch.object(
                torch.cuda,
                "Event",
                return_value=inputs_ready,
            ),
        ):
            observed_rec, observed_target = pool.forward_pair(
                stage="face",
                **common,
            )
        self.assertIs(observed_rec["vertices"], rec_output["vertices"])
        self.assertEqual(set(observed_target), {"vertices"})
        self.assertIs(
            observed_target["vertices"],
            target_output["vertices"],
        )
        self.assertEqual(
            sequence,
            [
                "target_auxiliary_enqueue",
                "primary_rec_enqueue",
                "primary_wait_return",
            ],
        )
        primary_model.assert_called_once_with(**common["rec_kwargs"])
        primary_stream.wait_event.assert_called_once_with(
            mock.sentinel.return_ready
        )
        target_output["vertices"].record_stream.assert_called_once_with(
            primary_stream
        )
        self.assertEqual(pool.last_forward_receipt["stage"], "face")
        self.assertEqual(
            pool.last_forward_receipt["reconstruction_execution"],
            "stock_full_batch_primary_current_stream",
        )
        self.assertEqual(
            pool.last_forward_receipt["target_execution"],
            "detached_full_batch_helper",
        )
        self.assertEqual(
            pool.last_forward_receipt["output_keys"],
            ["vertices"],
        )
        self.assertEqual(pool.completed_forward_pairs, 1)
        inputs_ready.record.assert_called_once_with(primary_stream)

    def test_target_offload_preserves_stock_reconstruction_path(self) -> None:
        source = (
            REPOSITORY / "utils" / "smplx_training.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forward_pair = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward_pair"
        )
        segment = ast.get_source_segment(source, forward_pair)
        assert segment is not None
        self.assertIn("stage: str | None = None", segment)
        self.assertIn(
            'self.mode == "target_offload"',
            segment,
        )
        self.assertIn("stage not in _LOCAL_LOSS_STAGES", segment)
        self.assertIn('if self.mode == "target_offload":', segment)
        self.assertIn(
            "primary_output = self.models[self.primary_device](**rec_kwargs)",
            segment,
        )
        self.assertIn(
            "target_scheduled = self._schedule_target_offload(",
            segment,
        )
        self.assertLess(
            segment.index(
                "target_scheduled = self._schedule_target_offload("
            ),
            segment.index(
                "primary_output = "
                "self.models[self.primary_device](**rec_kwargs)"
            ),
        )
        self.assertLess(
            segment.index(
                "primary_output = "
                "self.models[self.primary_device](**rec_kwargs)"
            ),
            segment.index(
                "target_output = self._collect_target_offload("
            ),
        )
        target_branch = segment.split(
            'if self.mode == "target_offload":',
            1,
        )[1].split("else:", 1)[0]
        self.assertNotIn("_copy_shard_kwargs", target_branch)
        self.assertNotIn("_schedule_local_loss", target_branch)
        self.assertNotIn("rec_scheduled", target_branch)
        self.assertIn(
            '"stock_full_batch_primary_current_stream"',
            target_branch,
        )
        self.assertIn(
            '"detached_target_full_outputs_only"',
            target_branch,
        )
        self.assertIn('"stage": stage', target_branch)

        offload_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_schedule_target_offload"
        )
        offload_segment = ast.get_source_segment(
            source,
            offload_method,
        )
        assert offload_segment is not None
        self.assertIn("value.to(", offload_segment)
        self.assertIn(
            'primary_destination = torch.device(',
            offload_segment,
        )
        self.assertIn(
            "torch.cuda.stream(transfer_stream)",
            offload_segment,
        )
        self.assertIn(
            "transfer_stream.wait_event(inputs_ready)",
            offload_segment,
        )
        self.assertIn(
            "helper_stream.wait_event(input_handoff)",
            offload_segment,
        )
        self.assertIn(
            "transfer_stream.wait_event(compute_ready)",
            offload_segment,
        )
        self.assertIn(
            "value.record_stream(transfer_stream)",
            offload_segment,
        )
        self.assertIn(
            "value.record_stream(helper_stream)",
            offload_segment,
        )
        self.assertNotIn(
            "torch.cuda.current_stream",
            offload_segment,
        )

        collect_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_collect_target_offload"
        )
        collect_segment = ast.get_source_segment(source, collect_method)
        assert collect_segment is not None
        self.assertIn(
            "primary_stream.wait_event(scheduled.ready)",
            collect_segment,
        )
        self.assertIn(
            "value.record_stream(primary_stream)",
            collect_segment,
        )

        init_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and any(
                isinstance(child, ast.Attribute)
                and child.attr == "helper_devices"
                for child in ast.walk(node)
            )
        )
        init_segment = ast.get_source_segment(source, init_method)
        assert init_segment is not None
        self.assertIn(
            'if mode == "target_offload":',
            init_segment,
        )
        self.assertIn(
            "if self.helper_devices != (1,):",
            init_segment,
        )

    def test_trainers_use_paired_pool_only_for_train_forwards(self) -> None:
        for trainer in (
            "aeface_trainer.py",
            "ae_trainer.py",
            "aelower_trainer.py",
        ):
            with self.subTest(trainer=trainer):
                source = (REPOSITORY / trainer).read_text(encoding="utf-8")
                tree = ast.parse(source)
                train_method = next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "train"
                )
                train_source = ast.get_source_segment(source, train_method)
                assert train_source is not None
                self.assertIn("self.smplx_pool.forward_pair(", train_source)
                self.assertIn(
                    "self.smplx_pool.forward_local_losses(",
                    train_source,
                )
                self.assertIn("clip_length=n", train_source)
        expected_paired_stage = {
            "aeface_trainer.py": 'forward_pair(\n'
            '                                stage="face",',
            "ae_trainer.py": 'forward_pair(\n'
            "                                stage=self.args.formal_stage,",
            "aelower_trainer.py": 'forward_pair(\n'
            '                                    stage="lower",',
        }
        for trainer, paired_stage_source in expected_paired_stage.items():
            source = (REPOSITORY / trainer).read_text(encoding="utf-8")
            self.assertIn(paired_stage_source, source)
        expected_stage = {
            "aeface_trainer.py": 'stage="face"',
            "aelower_trainer.py": 'stage="lower"',
        }
        for trainer, stage_source in expected_stage.items():
            source = (REPOSITORY / trainer).read_text(encoding="utf-8")
            self.assertIn(stage_source, source)
        lower_source = (
            REPOSITORY / "aelower_trainer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("static_mask=static_idx", lower_source)
        ae_source = (REPOSITORY / "ae_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("stage=self.args.formal_stage", ae_source)
        face_source = (
            REPOSITORY / "aeface_trainer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'raise RuntimeError("SMPL-X training pooling is training-only")',
            face_source,
        )

    def test_config_defaults_to_disabled_and_requires_explicit_helpers(self) -> None:
        source = (
            REPOSITORY / "utils" / "config.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--smplx_training_pool_mode"', source)
        self.assertIn('"--smplx_training_helper_devices"', source)
        self.assertIn('default="disabled"', source)
        self.assertIn('default=""', source)
        self.assertIn('"target_offload"', source)
        self.assertIn('"sharded_local_loss"', source)

    def test_local_scheduler_returns_only_scalar_numerators(self) -> None:
        source = (
            REPOSITORY / "utils" / "smplx_training.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_schedule_local_loss"
        )
        segment = ast.get_source_segment(source, method)
        assert segment is not None
        self.assertIn(
            "for name, (numerator, count) in local_parts.items()",
            segment,
        )
        self.assertIn("numerator.ndim != 0", segment)
        self.assertIn("return primary_parts, ready", segment)
        self.assertNotIn('rec_output["vertices"].to(', segment)
        self.assertNotIn('rec_output["joints"].to(', segment)
        self.assertNotIn('target_output["vertices"].to(', segment)
        self.assertNotIn('target_output["joints"].to(', segment)
        self.assertNotIn("repeat(", segment)
        self.assertNotIn("expand(", segment)

    def test_local_runtime_receipt_is_explicit_and_copy_safe(self) -> None:
        fake_pool = SimpleNamespace(
            mode="sharded_local_loss",
            primary_device=0,
            helper_devices=(1, 2, 3),
            replicas={0: object(), 1: object(), 2: object(), 3: object()},
            last_forward_receipt={
                "spans": [{"device": 1, "clip_start": 0, "clip_end": 2}]
            },
        )
        receipt = SmplxTrainingPool.runtime_receipt(fake_pool)
        self.assertEqual(receipt["format"], "semtalk_smplx_training_pool_v2")
        self.assertEqual(
            receipt["partition"],
            "balanced_contiguous_whole_clips",
        )
        self.assertEqual(
            receipt["transfer_to_primary"],
            "differentiable_scalar_numerators_only",
        )
        self.assertEqual(receipt["helper_devices"], [1, 2, 3])
        receipt["last_forward"]["spans"][0]["device"] = 99
        self.assertEqual(
            fake_pool.last_forward_receipt["spans"][0]["device"],
            1,
        )

    def test_target_offload_runtime_receipt_is_explicit(self) -> None:
        fake_pool = SimpleNamespace(
            mode="target_offload",
            primary_device=0,
            helper_devices=(1,),
            replicas={0: object(), 1: object()},
            completed_forward_pairs=7,
            last_forward_receipt={
                "stage": "face",
                "reconstruction_device": 0,
                "reconstruction_execution": (
                    "stock_full_batch_primary_current_stream"
                ),
                "target_device": 1,
                "target_execution": "detached_full_batch_helper",
            },
        )
        receipt = SmplxTrainingPool.runtime_receipt(fake_pool)
        self.assertEqual(
            receipt["partition"],
            (
                "primary_stock_full_batch_reconstruction_and_"
                "helper_full_batch_target"
            ),
        )
        self.assertEqual(
            receipt["transfer_to_primary"],
            "detached_target_full_outputs_only",
        )
        self.assertEqual(receipt["helper_devices"], [1])
        self.assertEqual(receipt["completed_forward_pairs"], 7)
        self.assertEqual(
            receipt["last_forward"]["reconstruction_device"],
            0,
        )
        self.assertEqual(receipt["last_forward"]["target_device"], 1)

    def test_target_offload_forward_counter_restore_is_fail_closed(
        self,
    ) -> None:
        pool = object.__new__(SmplxTrainingPool)
        pool.mode = "target_offload"
        pool.completed_forward_pairs = 0
        pool.last_forward_receipt = None
        pool.restore_completed_forward_pairs(17)
        self.assertEqual(pool.completed_forward_pairs, 17)
        for invalid in (-1, True, 0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RuntimeError):
                    pool.restore_completed_forward_pairs(invalid)

        sharded = object.__new__(SmplxTrainingPool)
        sharded.mode = "sharded_local_loss"
        sharded.completed_forward_pairs = 0
        sharded.last_forward_receipt = None
        with self.assertRaises(RuntimeError):
            sharded.restore_completed_forward_pairs(0)


if __name__ == "__main__":
    unittest.main()
