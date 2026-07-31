from __future__ import annotations

import socket
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal hosts.
    torch = None
    dist = None
    mp = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rvq_worker(rank: int, world_size: int, port: int, output: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from models.quantizer import QuantizeEMAReset
        from utils.rvq_distributed import assert_rvq_rank_state

        torch.manual_seed(123 + rank)
        quantizer = QuantizeEMAReset(4, 2, None).train()
        local_x = (
            torch.tensor([[0.0, 0.0], [1.0, 0.0]])
            if rank == 0
            else torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        )
        quantizer.init_codebook(local_x)
        local_indices = (
            torch.tensor([0, 1], dtype=torch.long)
            if rank == 0
            else torch.tensor([1, 3], dtype=torch.long)
        )
        perplexity = quantizer.update_codebook(local_x, local_indices)
        state_receipt = assert_rvq_rank_state(quantizer)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            {
                "codebook": quantizer.codebook.cpu(),
                "code_sum": quantizer.code_sum.cpu(),
                "code_count": quantizer.code_count.cpu(),
                "perplexity": perplexity.cpu(),
                "receipt": state_receipt,
            },
        )
        if rank == 0:
            torch.save(gathered, output)
    finally:
        dist.destroy_process_group()


def _rvq_2d_worker(rank: int, world_size: int, port: int, output: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from models.quantizer import QuantizeEMAReset2D

        torch.manual_seed(91 + rank)
        quantizer = QuantizeEMAReset2D(4, 2).train()
        local = torch.tensor(
            [[[[rank + 0.0, rank + 1.0]], [[0.0, 1.0]]]],
            dtype=torch.float32,
        )
        quantizer(local)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            {
                "codebook": quantizer.codebook.cpu(),
                "code_sum": quantizer.code_sum.cpu(),
                "code_count": quantizer.code_count.cpu(),
            },
        )
        if rank == 0:
            torch.save(gathered, output)
    finally:
        dist.destroy_process_group()


def _rvq_w4_worker(rank: int, world_size: int, port: int, output: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from models.quantizer import QuantizeEMAReset
        from utils.rvq_distributed import assert_rvq_rank_state

        torch.manual_seed(700 + rank)
        quantizer = QuantizeEMAReset(4, 2, None).train()
        local_x = torch.tensor(
            [[float(rank), 0.0], [float(rank), 1.0]]
        )
        quantizer.init_codebook(local_x)
        quantizer.update_codebook(
            local_x,
            torch.tensor([rank % 4, (rank + 1) % 4]),
        )
        receipt = assert_rvq_rank_state(quantizer)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, receipt)
        if rank == 0:
            torch.save(gathered, output)
    finally:
        dist.destroy_process_group()


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class RVQDistributedCPUTest(unittest.TestCase):
    def _spawn(self, worker, world_size: int = 2) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "result.pt")
            mp.spawn(
                worker,
                args=(world_size, _free_port(), output),
                nprocs=world_size,
                join=True,
            )
            return torch.load(output, map_location="cpu", weights_only=False)

    def test_ema_matches_one_process_global_batch(self) -> None:
        from models.quantizer import QuantizeEMAReset

        distributed = self._spawn(_rvq_worker)
        for other in distributed[1:]:
            for field in ("codebook", "code_sum", "code_count", "perplexity"):
                self.assertTrue(
                    torch.equal(distributed[0][field], other[field]),
                    field,
                )
            self.assertEqual(distributed[0]["receipt"], other["receipt"])
            self.assertTrue(other["receipt"]["all_ranks_exact"])

        torch.manual_seed(123)
        reference = QuantizeEMAReset(4, 2, None).train()
        global_x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        global_indices = torch.tensor([0, 1, 1, 3], dtype=torch.long)
        reference.init_codebook(global_x)
        reference_perplexity = reference.update_codebook(
            global_x,
            global_indices,
        )
        self.assertTrue(
            torch.equal(distributed[0]["codebook"], reference.codebook)
        )
        self.assertTrue(
            torch.equal(distributed[0]["code_sum"], reference.code_sum)
        )
        self.assertTrue(
            torch.equal(distributed[0]["code_count"], reference.code_count)
        )
        self.assertTrue(
            torch.equal(distributed[0]["perplexity"], reference_perplexity)
        )

    def test_2d_ema_state_is_identical(self) -> None:
        distributed = self._spawn(_rvq_2d_worker)
        for field in ("codebook", "code_sum", "code_count"):
            self.assertTrue(
                torch.equal(distributed[0][field], distributed[1][field]),
                field,
            )

    def test_w4_ema_state_is_identical(self) -> None:
        receipts = self._spawn(_rvq_w4_worker, world_size=4)
        self.assertEqual(len(receipts), 4)
        self.assertEqual(len({item["state_sha256"] for item in receipts}), 1)
        self.assertTrue(all(item["all_ranks_exact"] for item in receipts))

    def test_distributed_sampler_is_exact_and_epoch_seeded(self) -> None:
        dataset = list(range(128))
        epoch_zero = []
        epoch_one = []
        for rank in range(4):
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=4,
                rank=rank,
                shuffle=True,
                seed=2021,
                drop_last=True,
            )
            sampler.set_epoch(0)
            epoch_zero.append(list(sampler))
            sampler.set_epoch(1)
            epoch_one.append(list(sampler))
        flattened = [item for shard in epoch_zero for item in shard]
        self.assertEqual(len(flattened), 128)
        self.assertEqual(set(flattened), set(dataset))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertNotEqual(epoch_zero, epoch_one)

    def test_formal_size_sampler_has_no_padding_or_duplicates(self) -> None:
        dataset = list(range(127_286))
        consumed_by_epoch = []
        for epoch in (0, 1):
            shards = []
            for rank in range(4):
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=4,
                    rank=rank,
                    shuffle=True,
                    seed=2021,
                    drop_last=True,
                )
                sampler.set_epoch(epoch)
                local_indices = list(sampler)
                self.assertEqual(len(local_indices), 31_821)
                shards.append(local_indices[: 497 * 64])
            flattened = [item for shard in shards for item in shard]
            self.assertEqual(len(flattened), 127_232)
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertEqual(len(dataset) - len(flattened), 54)
            consumed_by_epoch.append(flattened)
        self.assertNotEqual(consumed_by_epoch[0], consumed_by_epoch[1])

    def test_receipt_locks_w2_w4_and_global_batch_256(self) -> None:
        from utils.rvq_distributed import (
            representation_ddp_receipt,
            validate_representation_ddp_receipt,
        )

        for world_size in (2, 4):
            receipt = representation_ddp_receipt(
                formal_stage="face",
                world_size=world_size,
                local_batch_size=256 // world_size,
                train_samples=127_286,
                updates_per_epoch=497,
                seed=2021,
            )
            validate_representation_ddp_receipt(
                receipt,
                formal_stage="face",
                world_size=world_size,
                local_batch_size=256 // world_size,
                train_samples=127_286,
                updates_per_epoch=497,
                seed=2021,
            )
            self.assertEqual(receipt["global_batch_size"], 256)
            self.assertEqual(receipt["consumed_samples_per_epoch"], 127_232)
            self.assertEqual(receipt["dropped_samples_per_epoch"], 54)
            self.assertEqual(
                receipt["padding_or_duplicate_samples_per_epoch"],
                0,
            )
        with self.assertRaisesRegex(RuntimeError, "global batch"):
            representation_ddp_receipt(
                formal_stage="face",
                world_size=4,
                local_batch_size=16,
                train_samples=127_286,
                updates_per_epoch=497,
                seed=2021,
            )
    def test_optimizer_lr_uses_global_batch(self) -> None:
        from types import SimpleNamespace
        from optimizers.optim_factory import optimizer_kwargs

        args = SimpleNamespace(
            opt="adam",
            lr_base=3e-4,
            batch_size=64,
            global_batch_size=256,
            weight_decay=0.0,
            momentum=0.9,
            opt_eps=None,
            opt_betas=[0.5, 0.999],
            opt_args=None,
        )
        self.assertEqual(
            optimizer_kwargs(args, 1)["learning_rate"],
            3e-4 * 256 / 128,
        )

    def test_official_codebook_prior_prevents_first_update_reset(self) -> None:
        from models.quantizer import QuantizeEMAReset
        from utils.rvq_distributed import initialize_loaded_rvq_ema

        class ReleasedRVQ(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [QuantizeEMAReset(4, 2, None) for _ in range(6)]
                )

        model = ReleasedRVQ()
        loaded = []
        for index, layer in enumerate(model.layers):
            codebook = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            codebook = codebook + index * 100
            layer.codebook.copy_(codebook)
            loaded.append(codebook.clone())
        receipt = initialize_loaded_rvq_ema(model)
        self.assertEqual(
            receipt["format"],
            "semtalk_show_official_rvq_ema_prior_v2",
        )
        self.assertFalse(receipt["first_forward_codebook_reset"])
        for layer, expected in zip(model.layers, loaded):
            self.assertTrue(layer.init)
            self.assertTrue(torch.equal(layer.codebook, expected))
            self.assertTrue(
                torch.equal(layer.code_sum, expected * 100.0)
            )
            self.assertTrue(
                torch.equal(layer.code_count, torch.full((4,), 100.0))
            )

        # Exercise the actual EMA update path with only code zero assigned.
        # The one-count implementation reset codes 1..3 here because their
        # counts fell from 1.0 to 0.99.  The decay-aware prior must keep them
        # bound to the released centres instead of replacing them with the
        # current latent prefix.
        layer = model.layers[0]
        before = layer.codebook.detach().clone()
        latent = torch.tensor([[999.0, -999.0], [998.0, -998.0]])
        indices = torch.zeros(2, dtype=torch.long)
        layer.update_codebook(latent, indices)
        self.assertTrue(
            torch.allclose(
                layer.codebook[1:],
                before[1:],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertGreaterEqual(float(layer.code_count[1:].min()), 1.0)


if __name__ == "__main__":
    unittest.main()
