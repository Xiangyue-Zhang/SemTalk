import os
import sys
import time
import csv
import signal
import warnings
import random
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt

from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

# --- HuBERT mean/std ---
def _load_hubert_stats(args, device):
    mean_path = getattr(args, "hubert_mean_path", None)
    std_path  = getattr(args, "hubert_std_path", None)
    if not mean_path or not std_path or (not os.path.isfile(mean_path)) or (not os.path.isfile(std_path)):
        logger.warning("HuBERT mean/std 路径缺失或文件不存在，跳过注入（回退到模型内部归一化）")
        return None, None
    m = np.load(mean_path).astype(np.float32)
    s = np.load(std_path).astype(np.float32)
    s = np.clip(s, 1e-6, None)
    return torch.from_numpy(m).to(device), torch.from_numpy(s).to(device)

def _inject_hubert_stats(model_wrapped, mean_t, std_t):
    if mean_t is None or std_t is None:
        return
    mod = model_wrapped.module if hasattr(model_wrapped, "module") else model_wrapped
    if hasattr(mod, "set_hubert_dataset_stats"):
        mod.set_hubert_dataset_stats(mean_t, std_t)
        logger.info("已向模型注入 HuBERT mean/std")
    else:
        logger.warning("模型未实现 set_hubert_dataset_stats，跳过 HuBERT mean/std 注入")

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = torch.device(args.local_rank if hasattr(args, "local_rank") else 0)
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/"

        if self.rank == 0:
            if self.args.stat == "ts":
                self.writer = SummaryWriter(log_dir=self.checkpoint_path)
            else:
                # 如需离线： export WANDB_MODE=offline
                wandb.init(project=args.project, dir=args.out_path, name=args.name[12:] + args.notes)
                wandb.config.update(args)
                self.writer = None

        # ===== Data =====
        self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).LMDBNPZDataset(args, "train")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=args.batch_size,
            shuffle=False if args.ddp else True,
            num_workers=args.loader_workers,
            drop_last=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if args.ddp else None,
        )
        self.train_length = len(self.train_loader)
        logger.info("Init train dataloader success")

        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).PickleDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=1,
                shuffle=False,
                num_workers=args.loader_workers,
                drop_last=False,
            )
            logger.info("Init test dataloader success")

        # ===== Model =====
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        if args.ddp:
            model_raw = getattr(model_module, args.g_name)(args).to(self.device)
            model_raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_raw)
            self.model = DDP(
                model_raw,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            model_raw = getattr(model_module, args.g_name)(args)
            self.model = torch.nn.DataParallel(model_raw, args.gpus).cuda()

        # 注入 HuBERT 统计
        mean_1024, std_1024 = _load_hubert_stats(self.args, self.device)
        _inject_hubert_stats(self.model, mean_1024, std_1024)

        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
            if args.stat == "wandb":
                wandb.watch(self.model)

        # ===== Discriminator (可选) =====
        if args.d_name is not None:
            if args.ddp:
                d_raw = getattr(model_module, args.d_name)(args).to(self.device)
                d_raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(d_raw)
                self.d_model = DDP(
                    d_raw,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            else:
                self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()

            if self.rank == 0:
                logger.info(self.d_model)
                logger.info(f"init {args.d_name} success")
                if args.stat == "wandb":
                    wandb.watch(self.d_model)

            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)

        # ===== Eval model (可选) =====
        if args.e_name is not None:
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            if self.args.ddp:
                self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.device)
                self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model)
                self.eval_model = DDP(
                    self.eval_model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.device)  # 单卡评测副本
            else:
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.device)

            other_tools.load_checkpoints(self.eval_copy, args.data_path + args.e_path, args.e_name)
            other_tools.load_checkpoints(self.eval_model, args.data_path + args.e_path, args.e_name)

            self.eval_model.eval()
            self.eval_copy.eval()

            if self.rank == 0:
                logger.info(self.eval_model)
                logger.info(f"init {args.e_name} success")
                if args.stat == "wandb":
                    wandb.watch(self.eval_model)

        # ===== Optim & Sched =====
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)

        # ===== SMPL-X =====
        self.smplx = smplx.create(
            self.args.data_path_1 + "smplx_models/",
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
        ).to(self.device).eval()

        # ===== Metrics/Tools =====
        self.alignmenter = metric.alignment(
            0.3, 7, self.train_data.avg_vel,
            upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]
        ) if self.rank == 0 else None
        self.align_mask = 60
        self.l1_calculator = metric.L1div() if self.rank == 0 else None

    # ===== Helpers =====
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(self.device)
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 165), device=self.device)
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 165), device=self.device)
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def inverse_selection_tensor_6d(self, filtered_t, selection_array, n):
        new_selected_array = np.zeros((330))
        new_selected_array[::2] = selection_array
        new_selected_array[1::2] = selection_array
        selection_array = torch.from_numpy(new_selected_array).to(self.device)
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 330), device=self.device)
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 330), device=self.device)
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = "[%03d][%03d/%03d]  " % (epoch, its, self.train_length)
        for name, states in self.tracker.loss_meters.items():
            mtr = states['train']
            if mtr.count > 0:
                pstr += "{}: {:.3f}\t".format(name, mtr.avg)
                if self.args.stat == "ts":
                    self.writer.add_scalar(f"train/{name}", mtr.avg, epoch*self.train_length+its)
                else:
                    wandb.log({name: mtr.avg}, step=epoch*self.train_length+its)
        pstr += "glr: {:.1e}\t".format(lr_g)
        if self.args.stat == "ts":
            self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its)
        else:
            wandb.log({'glr': lr_g}, step=epoch*self.train_length+its)
        if lr_d is not None:
            pstr += "dlr: {:.1e}\t".format(lr_d)
            if self.args.stat == "ts":
                self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its)
            else:
                wandb.log({'dlr': lr_d}, step=epoch*self.train_length+its)
        pstr += "dtime: %04d\t" % (t_data*1000)
        pstr += "ntime: %04d\t" % (t_train*1000)
        pstr += "mem: {:.2f} ".format(mem_cost*len(self.args.gpus))
        logger.info(pstr)

    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)


@logger.catch
def main_worker(args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # 从 torchrun 注入的环境读取
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = local_rank

    torch.cuda.set_device(local_rank)

    # 若指定 ddp 且多进程，初始化分布式
    if args.ddp and world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=10),
        )
    else:
        # 未用 torchrun 或 WORLD_SIZE==1，降级单进程
        args.ddp = False

    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)

    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) \
        if args.trainer != "base" else BaseTrainer(args)

    logger.info("Training from scratch ...")
    start_time = time.time()

    # 可选：加载权重
    if getattr(args, "load_ckpt", None):
        load_ckpt = args.load_ckpt
        if os.path.exists(load_ckpt):
            logger.info(f"Loading checkpoint from {load_ckpt}")
            other_tools.load_checkpoints(trainer.model, load_ckpt, "semtalk_model")
            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning(f"Checkpoint {load_ckpt} 不存在，忽略加载")

    # 训练/评测循环
    for epoch in range(args.epochs + 1):
        elapsed = time.time() - start_time
        if trainer.rank == 0:
            remain = (args.epochs / max(epoch, 1) - 1.0) * (elapsed / 60.0) if epoch > 0 else 0.0
            logger.info(f"Time >>> elapsed: {elapsed/60:.2f} mins, remain: {remain:.2f} mins")

        if epoch != args.epochs:
            if args.ddp and hasattr(trainer.train_loader, "sampler") and trainer.train_loader.sampler is not None:
                trainer.train_loader.sampler.set_epoch(epoch)
            trainer.tracker.reset()
            trainer.train(epoch)

        # debug 跑一次保存 + 回载 + 测试
        if args.debug:
            other_tools.save_checkpoints(
                os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"),
                trainer.model, opt=None, epoch=None, lrs=None
            )
            other_tools.load_checkpoints(
                trainer.model,
                os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"),
                args.g_name
            )
            _ = trainer.test(epoch)

        # 按策略评测与保存
        if epoch == 0 or (epoch > 100) or (epoch <= 100 and epoch % args.test_period == 0):
            if rank == 0:
                fid = trainer.test(epoch)  # 需你的 test 返回 fid（float）或可比较的指标
                is_best = (fid is not None) and (fid < getattr(trainer, "best_fid", float("inf")))
                if is_best:
                    trainer.best_fid = fid
                    ckpt_name = f"best_{epoch}.bin"
                else:
                    ckpt_name = f"last_{epoch}.bin"

                other_tools.save_checkpoints(
                    os.path.join(trainer.checkpoint_path, ckpt_name),
                    trainer.model, opt=None, epoch=None, lrs=None
                )

    if args.ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = config.parse_args()
    # 用 torchrun 启动：不需要在代码里设置 MASTER_ADDR/MASTER_PORT
    # 单机两卡示例：
    # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/EMAGE_2024/train.py --config scripts/EMAGE_2024/configs/mmm.yaml --ddp true
    main_worker(args)
