"""Strict distributed-training contracts for the SHOW representation stages."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.distributed as dist


RVQ_STAGES = frozenset({"face", "hands", "upper", "lower"})
REPRESENTATION_STAGES = frozenset({*RVQ_STAGES, "global"})
RECEIPT_FORMAT = "semtalk_show_representation_ddp_v1"
STATE_RECEIPT_FORMAT = "semtalk_show_rvq_rank_state_v1"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def representation_ddp_receipt(
    *,
    formal_stage: str,
    world_size: int,
    local_batch_size: int,
    train_samples: int,
    updates_per_epoch: int,
    seed: int,
) -> dict[str, Any]:
    """Build and validate the locked representation/base batch protocol."""

    for name, value in (
        ("world_size", world_size),
        ("local_batch_size", local_batch_size),
        ("train_samples", train_samples),
        ("updates_per_epoch", updates_per_epoch),
        ("seed", seed),
    ):
        if type(value) is not int:
            raise RuntimeError(f"{name} must be an exact integer")
    if formal_stage not in {*REPRESENTATION_STAGES, "base"}:
        raise RuntimeError(f"unsupported formal stage {formal_stage!r}")

    if formal_stage in RVQ_STAGES:
        if world_size not in {2, 4}:
            raise RuntimeError("formal RVQ requires world_size 2 or 4")
        if local_batch_size * world_size != 256:
            raise RuntimeError("formal RVQ global batch must equal 256")
        sampler = {
            "class": "torch.utils.data.distributed.DistributedSampler",
            "shuffle": True,
            "seed": seed,
            "drop_last": True,
            "set_epoch": "before every epoch",
        }
        ema = {
            "enabled": True,
            "assignment": (
                "rank-local Gumbel samples from seed + global rank"
            ),
            "statistics": ["code_count", "code_sum"],
            "collective": "all_reduce SUM",
            "initialization": "global rank-ordered prefix; rank0 broadcast",
            "dead_code_reset": (
                "global rank-ordered prefix on demand; rank0 broadcast"
            ),
            "perplexity": "global code_count all_reduce SUM",
            "rank_state": "exact SHA-256 agreement at save/resume/finalize",
        }
    elif formal_stage == "global":
        if world_size != 1 or local_batch_size != 256:
            raise RuntimeError(
                "formal Global keeps world_size 1 and batch size 256"
            )
        sampler = {
            "class": "RandomSampler",
            "shuffle": True,
            "seed": seed,
            "drop_last": True,
            "set_epoch": None,
        }
        ema = {"enabled": False}
    else:
        if world_size != 1 or local_batch_size != 64:
            raise RuntimeError(
                "formal Base keeps world_size 1 and batch size 64"
            )
        sampler = {
            "class": "RandomSampler",
            "shuffle": True,
            "seed": seed,
            "drop_last": True,
            "set_epoch": None,
        }
        ema = {"enabled": False}

    expected_updates = (
        497 if formal_stage in REPRESENTATION_STAGES else 1_988
    )
    expected_global_batch = (
        256 if formal_stage in REPRESENTATION_STAGES else 64
    )
    consumed_samples = updates_per_epoch * expected_global_batch
    if (
        train_samples != 127_286
        or updates_per_epoch != expected_updates
        or consumed_samples != 127_232
    ):
        raise RuntimeError(
            "formal SHOW receipt has invalid available/consumed sample "
            "accounting"
        )
    receipt = {
        "format": RECEIPT_FORMAT,
        "formal_stage": formal_stage,
        "world_size": world_size,
        "local_batch_size": local_batch_size,
        "global_batch_size": local_batch_size * world_size,
        "train_samples": train_samples,
        "available_train_samples": train_samples,
        "consumed_samples_per_epoch": consumed_samples,
        "dropped_samples_per_epoch": train_samples - consumed_samples,
        "padding_or_duplicate_samples_per_epoch": 0,
        "updates_per_epoch": updates_per_epoch,
        "loader_drop_last": True,
        "sampler": sampler,
        "rvq_ema": ema,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def validate_representation_ddp_receipt(
    receipt: Any,
    **expected: Any,
) -> dict[str, Any]:
    rebuilt = representation_ddp_receipt(**expected)
    if type(receipt) is not dict or receipt != rebuilt:
        raise RuntimeError("distributed training receipt mismatch")
    return receipt


def _update_tensor_digest(
    digest: Any,
    name: str,
    tensor: torch.Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))


def rvq_state_sha256(model: torch.nn.Module) -> str:
    """Hash model buffers plus the legacy non-buffer RVQ EMA accumulators."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"model state {name!r} is not a tensor")
        _update_tensor_digest(digest, f"model:{name}", tensor)
    for name, module in sorted(model.named_modules()):
        if module.__class__.__name__ != "QuantizeEMAReset":
            continue
        digest.update(f"ema:{name}:init={bool(module.init)}".encode("utf-8"))
        if bool(module.init):
            if module.code_sum is None or module.code_count is None:
                raise RuntimeError(f"initialized RVQ layer {name} lacks EMA state")
            _update_tensor_digest(digest, f"ema:{name}:sum", module.code_sum)
            _update_tensor_digest(
                digest,
                f"ema:{name}:count",
                module.code_count,
            )
    return digest.hexdigest()


def assert_rvq_rank_state(model: torch.nn.Module) -> dict[str, Any]:
    """Collect exact hashes and fail closed if any rank has drifted."""

    local_sha256 = rvq_state_sha256(model)
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if world_size == 1:
        digests = [local_sha256]
    else:
        digests = [None] * world_size
        dist.all_gather_object(digests, local_sha256)
    if any(value != local_sha256 for value in digests):
        raise RuntimeError(f"RVQ state differs across ranks: {digests}")
    return {
        "format": STATE_RECEIPT_FORMAT,
        "world_size": world_size,
        "state_sha256": local_sha256,
        "all_ranks_exact": True,
    }


def initialize_loaded_rvq_ema(model: torch.nn.Module) -> dict[str, Any]:
    """Attach a decay-aware prior to every loaded official codebook vector.

    Released SemTalk checkpoints persist all six codebooks but the legacy EMA
    accumulators are plain attributes and therefore absent.  A one-count prior
    is unsafe: after one unassigned update it becomes ``mu < 1`` and the
    legacy dead-code rule immediately replaces the released vector.

    The stationary one-assignment prior ``1 / (1 - mu)`` keeps an unassigned
    released vector above the dead-code threshold for a documented grace
    period while preserving the exact initial centre ratio
    ``code_sum / code_count == codebook``.  Used codes begin adapting on the
    first SHOW update; persistently unused codes remain eligible for the
    original deterministic reset once the prior decays below one.
    """

    layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "QuantizeEMAReset":
            continue
        codebook = module.codebook.detach()
        if (
            codebook.ndim != 2
            or tuple(codebook.shape)
            != (int(module.nb_code), int(module.code_dim))
            or not bool(codebook.isfinite().all().item())
        ):
            raise RuntimeError(f"loaded RVQ codebook {name} is invalid")
        decay = float(module.mu)
        if not 0.0 < decay < 1.0:
            raise RuntimeError(
                f"loaded RVQ layer {name} has invalid EMA decay {decay}"
            )
        prior_count = 1.0 / (1.0 - decay)
        module.code_sum = codebook.clone().mul_(prior_count)
        module.code_count = torch.full(
            (int(module.nb_code),),
            prior_count,
            device=codebook.device,
            dtype=codebook.dtype,
        )
        module.init = True
        layers.append(
            {
                "name": name,
                "ema_decay": decay,
                "prior_count": prior_count,
            }
        )
    if len(layers) != 6:
        raise RuntimeError(
            f"official RVQ must expose exactly six EMA layers, got {layers}"
        )
    return {
        "format": "semtalk_show_official_rvq_ema_prior_v2",
        "layers": layers,
        "init": True,
        "code_sum": "loaded codebook multiplied by decay-aware prior_count",
        "code_count": "1 / (1 - ema_decay) per code",
        "first_forward_codebook_reset": False,
        "unused_code_grace": (
            "legacy reset only after the decay-aware prior falls below one"
        ),
    }
