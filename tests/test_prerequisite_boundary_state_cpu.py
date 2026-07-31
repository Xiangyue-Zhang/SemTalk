from __future__ import annotations

import copy
import struct
import unittest

from scripts.show_base import prerequisite_boundary_state as state


class Tensor:
    _continuation_test_tensor = True

    def __init__(self, values, shape=None, dtype="float32"):
        self.values = tuple(values)
        self.shape = tuple(shape or (len(self.values),))
        self.dtype = dtype
        self.bytes = b"".join(
            struct.pack("<d", float(value)) for value in self.values
        )


def cosine_state(boundary: int, base_lr: float = 0.001) -> dict:
    return {
        "base_values": [base_lr],
        "metric": None,
        "noise_range_t": None,
        "noise_pct": 0.67,
        "noise_type": "normal",
        "noise_std": 1.0,
        "noise_seed": 42,
        "t_initial": boundary,
        "t_mul": 1.0,
        "lr_min": 1e-6,
        "decay_rate": 1.0,
        "cycle_limit": 1,
        "warmup_t": 0,
        "warmup_lr_init": 1e-6,
        "warmup_prefix": False,
        "t_in_epochs": True,
        "warmup_steps": [1],
        "param_group_field": "lr",
        "_initial_param_group_field": "initial_lr",
    }


def rng_state(seed: int) -> dict:
    return {
        "python": (3, (seed, seed + 1), None),
        "numpy": (
            "MT19937",
            (seed, seed + 1),
            624,
            0,
            0.0,
        ),
        "torch_cpu": Tensor([seed + 1], dtype="uint8"),
        "torch_cuda": Tensor([seed + 2], dtype="uint8"),
    }


def resume_fixture(stage_name: str = "face", boundary: int = 200) -> dict:
    updates = boundary * state.UPDATES_PER_EPOCH
    scheduler = cosine_state(boundary)
    current_lr = state._cosine_epoch_values(scheduler, boundary - 1)[0]
    optimizer = {
        "state": {
            0: {
                "step": updates,
                "exp_avg": Tensor([0.1, -0.2]),
                "exp_avg_sq": Tensor([0.01, 0.04]),
            },
            1: {
                "step": updates,
                "exp_avg": Tensor([0.3]),
                "exp_avg_sq": Tensor([0.09]),
            },
        },
        "param_groups": [
            {
                "params": [0, 1],
                "lr": current_lr,
                "initial_lr": 0.001,
            }
        ],
    }
    rvq = {}
    if stage_name != "global":
        rvq = {
            f"quantizer.{index}": {
                "init": True,
                "code_sum": Tensor(
                    [1.0, 2.0, 3.0, 4.0],
                    shape=(2, 2),
                ),
                "code_count": Tensor([3.0, 7.0]),
            }
            for index in range(6)
        }
    world_size = 1 if stage_name == "global" else 4
    return {
        "format": "semtalk_show_train_resume_v5",
        "completed_epochs": boundary,
        "optimizer_updates": updates,
        "updates_per_epoch": state.UPDATES_PER_EPOCH,
        "world_size": world_size,
        "model_state": {
            "weight": Tensor([0.5, -0.25]),
            "bias": Tensor([0.1]),
        },
        "optimizer_state": optimizer,
        "scheduler_state": scheduler,
        "rvq_ema_state": rvq,
        "rng_states": [
            rng_state(index + 1) for index in range(world_size)
        ],
    }


class BoundaryStateTests(unittest.TestCase):
    def test_rvq_boundary_state_is_noninitial_and_exact(self) -> None:
        proof = state.build_boundary_state_proof(
            resume_fixture("face"),
            stage="face",
            boundary_epoch=200,
            world_size=4,
        )
        self.assertEqual(proof["trained_parameter_count"], 2)
        self.assertEqual(proof["adam_state_count"], 2)
        self.assertEqual(proof["adam_step"], 99_400)
        self.assertEqual(proof["rvq_ema_layers"], 6)
        self.assertEqual(proof["rng_rank_count"], 4)

    def test_global_requires_no_rvq_and_one_rng_rank(self) -> None:
        proof = state.build_boundary_state_proof(
            resume_fixture("global"),
            stage="global",
            boundary_epoch=200,
            world_size=1,
        )
        self.assertEqual(proof["rvq_ema_layers"], 0)
        self.assertEqual(proof["rng_rank_count"], 1)

    def test_missing_adam_parameter_state_is_rejected(self) -> None:
        resume = resume_fixture()
        resume["optimizer_state"]["state"].pop(1)
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "exactly cover trained parameters",
        ):
            state.build_boundary_state_proof(
                resume,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )

    def test_zero_or_missing_adam_moments_are_rejected(self) -> None:
        missing = resume_fixture()
        missing["optimizer_state"]["state"][0].pop("exp_avg")
        zero = resume_fixture()
        for item in zero["optimizer_state"]["state"].values():
            item["exp_avg"] = Tensor([0.0])
            item["exp_avg_sq"] = Tensor([0.0])
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "missing Adam moments",
        ):
            state.build_boundary_state_proof(
                missing,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "moments are all zero",
        ):
            state.build_boundary_state_proof(
                zero,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )

    def test_adam_step_reset_is_rejected(self) -> None:
        resume = resume_fixture()
        resume["optimizer_state"]["state"][0]["step"] = 0
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "step mismatch",
        ):
            state.build_boundary_state_proof(
                resume,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )

    def test_scheduler_reset_or_wrong_epoch_is_rejected(self) -> None:
        resume = resume_fixture()
        resume["optimizer_state"]["param_groups"][0]["lr"] = 0.001
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "scheduler progress",
        ):
            state.build_boundary_state_proof(
                resume,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )

    def test_rvq_empty_zero_and_global_extra_are_rejected(self) -> None:
        empty = resume_fixture()
        empty["rvq_ema_state"]["quantizer.0"]["init"] = False
        zero = resume_fixture()
        zero["rvq_ema_state"]["quantizer.1"]["code_count"] = Tensor(
            [0.0, 0.0]
        )
        global_extra = resume_fixture("global")
        global_extra["rvq_ema_state"] = {
            "unexpected": {
                "init": True,
                "code_sum": Tensor([1.0]),
                "code_count": Tensor([1.0]),
            }
        }
        for resume, stage_name, pattern, world in (
            (empty, "face", "uninitialized", 4),
            (zero, "face", "accumulators", 4),
            (global_extra, "global", "must not contain", 1),
        ):
            with self.assertRaisesRegex(state.BoundaryStateError, pattern):
                state.build_boundary_state_proof(
                    resume,
                    stage=stage_name,
                    boundary_epoch=200,
                    world_size=world,
                )

    def test_rng_omission_and_world_size_drift_are_rejected(self) -> None:
        omitted = resume_fixture()
        omitted["rng_states"].pop()
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "exact world size",
        ):
            state.build_boundary_state_proof(
                omitted,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "boundary accounting",
        ):
            state.build_boundary_state_proof(
                resume_fixture(),
                stage="face",
                boundary_epoch=200,
                world_size=1,
            )

    def test_model_zero_and_nonfinite_state_are_rejected(self) -> None:
        zero = resume_fixture()
        zero["model_state"] = {"weight": Tensor([0.0])}
        nonfinite = resume_fixture()
        nonfinite["optimizer_state"]["state"][0]["exp_avg"] = Tensor(
            [float("nan")]
        )
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "all zero",
        ):
            state.build_boundary_state_proof(
                zero,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "non-finite",
        ):
            state.build_boundary_state_proof(
                nonfinite,
                stage="face",
                boundary_epoch=200,
                world_size=4,
            )

    def test_resigned_proof_accounting_attack_is_rejected(self) -> None:
        proof = state.build_boundary_state_proof(
            resume_fixture(),
            stage="face",
            boundary_epoch=200,
            world_size=4,
        )
        attacked = copy.deepcopy(proof)
        attacked["adam_step"] = 1
        unsigned = dict(attacked)
        unsigned.pop("receipt_payload_sha256")
        attacked["receipt_payload_sha256"] = state.canonical_sha256(
            unsigned
        )
        with self.assertRaisesRegex(
            state.BoundaryStateError,
            "accounting mismatch",
        ):
            state.validate_boundary_state_proof(attacked)


if __name__ == "__main__":
    unittest.main()
