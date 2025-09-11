import os, sys, pathlib
if __package__ is None or __package__ == "":
    # executed as a script: add project root (SemTalk) into sys.path
    THIS_FILE = pathlib.Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[1]   # .../SemTalk
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
import torch
import lmdb
import io
import numpy as np
from typing import Dict, Any, List
import pickle
import pandas as pd
from loguru import logger
from .data_tools import joints_list

class LMDBNPZDataset(torch.utils.data.Dataset):
    """
    读取你LMDB, 每个样本返回 {key: np.ndarray or object}
    """
    def __init__(self, args, loader_type='train'):
        self.env = lmdb.open(args.train_path, readonly=True, lock=False, readahead=False, max_readers=1, subdir=True)
        with self.env.begin(buffers=True) as txn:
            stat = txn.stat()
            self.length = stat["entries"]
       
        self.args = args
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0] # for trinity
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))  
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        if self.args.beat_align:
            if not os.path.exists(self.args.data_path+f"weights/mean_vel_{self.args.pose_rep}.npy"):
                ValueError("Please run beat_align.py first")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        
        split_rule = pd.read_csv(args.data_path+"train_test_split.csv")
        self.selected_file = split_rule.loc[(split_rule['type'] == loader_type) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            #self.selected_file = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[(split_rule['type'] == 'train') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = self.selected_file.iloc[0:8]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        key = f"{idx:010d}".encode("utf-8")
        with self.env.begin(buffers=True) as txn:
            v = txn.get(key)
            if v is None:
                raise IndexError(f"Key {key!r} not found")
            with io.BytesIO(bytes(v)) as bio:
                arrs = np.load(bio, allow_pickle=True)
                sample = {k: arrs[k] for k in arrs.files}
                for k in ("sentence", "emo"):
                    if k in sample and isinstance(sample[k], np.ndarray) and sample[k].dtype == object:
                        sample[k] = sample[k].tolist()
        return sample
    
class PickleDataset(torch.utils.data.Dataset):
    """
    读取.pkl()内容为 list[dict]）。
    为避免频繁 IO, 这里一次性 load 到内存；如果数据巨大可改为多文件分块。
    """
    def __init__(self, args, loader_type='test'):
        if not os.path.isfile(args.test_path):
            raise FileNotFoundError(f"Pickle file not found: {args.test_path}")
        with open(args.test_path, "rb") as f:
            self.samples: List[Dict[str, Any]] = pickle.load(f)
        if not isinstance(self.samples, list):
            raise ValueError(f"Pickle content is not a list, got {type(self.samples)}")
        self.args = args
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0] # for trinity
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))  
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        if self.args.beat_align:
            if not os.path.exists(self.args.data_path+f"weights/mean_vel_{self.args.pose_rep}.npy"):
                ValueError("Please run beat_align.py first")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        
        split_rule = pd.read_csv(args.data_path+"train_test_split.csv")
        self.selected_file = split_rule.loc[(split_rule['type'] == loader_type) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            #self.selected_file = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[(split_rule['type'] == 'train') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = self.selected_file.iloc[0:8]
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range 0..{len(self.samples)-1}")
        sample = self.samples[idx]
        # 将 sentence/emo 的 object ndarray 转成 Python 列表，便于后续使用
        for k in ("sentence", "emo"):
            if k in sample and isinstance(sample[k], np.ndarray) and sample[k].dtype == object:
                sample[k] = sample[k].tolist()
        return sample