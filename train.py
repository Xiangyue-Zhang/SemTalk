import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
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
# import os
os.environ["HF_HOME"] = "/mnt/disk2T/hfcache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/disk2T/hfcache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/disk2T/hfcache/transformers"
os.environ["XDG_CACHE_HOME"] = "/mnt/disk2T/hfcache"
os.environ["TMPDIR"] = "/mnt/disk2T/tmp"


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        if args.ddp:
            # DDP 模式下 rank 对应 GPU id
            self.device = torch.device(f"cuda:{self.rank}")
        else:
            # 单机多卡 / 单卡
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        if self.rank==0:
            if self.args.stat == "ts":
                self.writer = SummaryWriter(log_dir=args.out_path + "custom/" + args.name + args.notes + "/")
            else:
                wandb.init(project=args.project, dir=args.out_path, name=args.name[12:] + args.notes)
                wandb.config.update(args)
                self.writer = None 
        if args.train_rvq:
            self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        else:
            self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).LMDBNPZDataset(args, "train")
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=args.batch_size,  
            shuffle=True if args.ddp else True,  
            num_workers=args.loader_workers,
            drop_last=True,

            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if args.ddp else None, 
        )
        self.train_length = len(self.train_loader)
        logger.info(f"Init train dataloader success")
       
        if self.rank == 0:
            if args.train_rvq:
                self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            else:
                self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).PickleDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
            logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{args.model}", fromlist=["something"])

        if args.ddp:
            self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        # mean_1024, std_1024 = _load_hubert_stats(self.args, self.device)
        # _inject_hubert_stats(self.model, mean_1024, std_1024)
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
            if args.stat == "wandb":
                wandb.watch(self.model)
        
        if args.d_name is not None:
            if args.ddp:
                self.d_model = getattr(model_module, args.d_name)(args).to(self.rank)
                self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model, process_group)   
                self.d_model = DDP(self.d_model, device_ids=[self.rank], output_device=self.rank, 
                                   broadcast_buffers=False, find_unused_parameters=False)
            else:    
                self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.d_model)
                logger.info(f"init {args.d_name} success")
                if args.stat == "wandb":
                    wandb.watch(self.d_model)
            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)
           
        if args.e_name is not None:
            """
            bugs on DDP training using eval_model, using additional eval_copy for evaluation 
            """
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            # eval copy is for single card evaluation
            if self.args.ddp:
                self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank) 
            else:
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                
            other_tools.load_checkpoints(self.eval_copy, args.data_path+args.e_path, args.e_name)
            other_tools.load_checkpoints(self.eval_model, args.data_path+args.e_path, args.e_name)
            if self.args.ddp:
                self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model, process_group)   
                self.eval_model = DDP(self.eval_model, device_ids=[self.rank], output_device=self.rank,
                                      broadcast_buffers=False, find_unused_parameters=False)
            self.eval_model.eval()
            self.eval_copy.eval()
            if self.rank == 0:
                logger.info(self.eval_model)
                logger.info(f"init {args.e_name} success")  
                if args.stat == "wandb":
                    wandb.watch(self.eval_model) 
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()
        self.alignmenter = metric.alignment(0.3, 7, self.train_data.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]) if self.rank == 0 else None
        self.align_mask = 60
        self.l1_calculator = metric.L1div() if self.rank == 0 else None
       
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t


    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 165)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 165), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def inverse_selection_tensor_6d(self, filtered_t, selection_array, n):
        new_selected_array = np.zeros((330))
        new_selected_array[::2] = selection_array
        new_selected_array[1::2] = selection_array 
        selection_array = new_selected_array
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 330)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 330), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = "[%03d][%03d/%03d]  "%(epoch, its, self.train_length)
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                pstr += "{}: {:.3f}\t".format(name, metric.avg)
                self.writer.add_scalar(f"train/{name}", metric.avg, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({name: metric.avg}, step=epoch*self.train_length+its)
        pstr += "glr: {:.1e}\t".format(lr_g)
        self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'glr': lr_g}, step=epoch*self.train_length+its)
        if lr_d is not None:
            pstr += "dlr: {:.1e}\t".format(lr_d)
            self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'dlr': lr_d}, step=epoch*self.train_length+its)
        pstr += "dtime: %04d\t"%(t_data*1000)        
        pstr += "ntime: %04d\t"%(t_train*1000)
        pstr += "mem: {:.2f} ".format(mem_cost*len(self.args.gpus))
        logger.info(pstr)
   
    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)

@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 
    logger.info("Training from scratch ...")          
    start_time = time.time()
    if args.inference:
        if rank == 0:
            if load_ckpt := args.load_ckpt:
                if os.path.exists(load_ckpt):
                    logger.info(f"Loading checkpoint from {load_ckpt}")
                    other_tools.load_checkpoints(trainer.model, load_ckpt, "semtalk_model")
                    logger.info("Checkpoint loaded successfully")
                    trainer.model.eval()
                else:
                    logger.warning(f"Checkpoint {load_ckpt} does not exist. Starting training from scratch.")
                    raise FileNotFoundError(f"Checkpoint {load_ckpt} does not exist.")
            trainer.inference(args.audio_infer_path)
        return
    if args.test_state:
        if rank == 0:
            if load_ckpt := args.load_ckpt:
                if os.path.exists(load_ckpt):
                    logger.info(f"Loading checkpoint from {load_ckpt}")
                    other_tools.load_checkpoints(trainer.model, load_ckpt, "semtalk_model")
                    logger.info("Checkpoint loaded successfully")
                    trainer.model.eval()
                    fid = trainer.test(0)
                    exit(0)
                else:
                    logger.warning(f"Checkpoint {load_ckpt} does not exist. Starting training from scratch.")
                    raise FileNotFoundError(f"Checkpoint {load_ckpt} does not exist.")
    # other_tools.load_checkpoints(trainer.model, '/mnt/disk2T/mm_data/zxy/SemTalk/weights/best_semtalk_base.bin', 'sem_model')
    # other_tools.load_checkpoints(trainer.model, '/mnt/disk2T/mm_data/zxy/SemTalk/weights/best_semtalk_sparse.bin', 'sem_model')
    
    # trainer.model.eval()
    # fid = trainer.test(400)
    # exit(0)
    for epoch in range(args.epochs+1):
        epoch_time = time.time()-start_time
        if trainer.rank == 0: logger.info("Time info >>>>  elapsed: %.2f mins\t"%(epoch_time/60)+"remain: %.2f mins"%((args.epochs/(epoch+1e-7)-1)*epoch_time/60))
        if epoch != args.epochs:
            if args.ddp: trainer.train_loader.sampler.set_epoch(epoch)
            trainer.tracker.reset()
            trainer.train(epoch)
        if args.debug:
            other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
            other_tools.load_checkpoints(trainer.model, os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), args.g_name)
            fid = trainer.test(epoch)

        # if (epoch) % args.test_period == 0 or epoch == 0:
        if epoch == 0 or (epoch > 100) or (epoch <= 100 and epoch % args.test_period == 0):
        # 执行测试逻辑
        # if epoch >=0:
            if rank == 0:
                # 先评测拿到当前 fid（假设 test 返回 fid 浮点数；若返回 dict，请改成拿 dict['fid']）
                fid = trainer.test(epoch)

                # 判断是否最优（FID 越低越好）
                is_best = (fid is not None) and (fid < getattr(trainer, "best_fid", float("inf")))
                if is_best:
                    trainer.best_fid = fid
                    ckpt_name = f"best_{epoch}.bin"
                    other_tools.save_checkpoints(
                    os.path.join(trainer.checkpoint_path, ckpt_name),
                    trainer.model, opt=None, epoch=None, lrs=None
                )
                # else:
                #     ckpt_name = f"last_{epoch}.bin"

               

                # exit(0)
                
       
    # if rank == 0:
    #     for k, v in trainer.tracker.values.items():
    #         if trainer.tracker.loss_meters[k]['val'].count > 0:
    #             other_tools.load_checkpoints(trainer.model, os.path.join(trainer.checkpoint_path, f"{k}.bin"), args.g_name)
    #             logger.info(f"inference on ckpt {k}_val_{v['val']['best']['epoch']}:")
    #             trainer.test(v['val']['best']['epoch'])
    #     other_tools.record_trial(args, trainer.tracker)
    #     if args.stat == "ts":
    #         trainer.writer.close()
    #     else:
    #         wandb.finish()
    
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='8680'
    args = config.parse_args()
    if args.ddp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(len(args.gpus), args,),
            nprocs=len(args.gpus),
                )
    else:
        
        main_worker(0, 1, args)
