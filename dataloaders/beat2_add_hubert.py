import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pyarrow
import librosa
import smplx
import torch.nn.functional as F
from numpy.lib import stride_tricks
from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools
from funasr import AutoModel
import torch.distributed as dist

def _dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank_default0() -> int:
    return dist.get_rank() if _dist_initialized() else 0

def get_world_size_default1() -> int:
    return dist.get_world_size() if _dist_initialized() else 1

def is_main_process() -> bool:
    return get_rank_default0() == 0

def safe_barrier():
    if _dist_initialized():
        dist.barrier()

emo_model = AutoModel(model="iic/emotion2vec_plus_large")
# emo_model = emo_model.to("cuda:0")
from itertools import groupby

def process_annotated_sentence(annotated_sentence):
    """
    处理从 TextGrid 文件中提取的标注句子，去除空白标注和重复单词，并输出一个正常的句子。
    
    Args:
        annotated_sentence (list of str): 输入的标注句子列表，包含重复单词和空白标注。
        
    Returns:
        str: 处理后的完整句子，单词间有空格分隔。
    """
    # 1. 移除空白标注
    filtered_sentence = [word for word in annotated_sentence if word.strip()]
    
    # 2. 去除重复的单词，同时保持原有顺序
    deduplicated_sentence = [key for key, _ in groupby(filtered_sentence)]
    
    # 3. 合并成一个完整的句子，添加空格
    final_sentence = ' '.join(deduplicated_sentence)
    
    return final_sentence
from transformers import Wav2Vec2Processor, HubertModel
# print("Loading the Wav2Vec2 Processor...")
# wav2vec2_processor = Wav2Vec2Processor.from_pretrained("./facebook/hubert-large-ls960-ft")
# print("Loading the HuBERT Model...")
# hubert_model = HubertModel.from_pretrained("./facebook/hubert-large-ls960-ft")
# hubert_model.eval()
def get_hubert_from_16k_speech_long(hubert_model, wav2vec2_processor, speech=None, device="cuda:0"):
        hubert_model = hubert_model.to(device)
        # if speech.ndim ==2:
        #     speech = speech[:, 0] # [T, 2] ==> [T,]
        input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0)  # [1, T]
        input_values_all = input_values_all.to(device)
        # For long audio sequence, due to the memory limitation, we cannot process them in one run
        # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
        # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
        # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
        # We have the equation to calculate out time step: T = floor((t-k)/s)
        # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
        # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
        res_lst = []
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            input_values = input_values_all[:, start_idx: end_idx]
            hidden_states = hubert_model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
            res_lst.append(hidden_states[0])
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
        # if input_values.shape[1] != 0:
        if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it
            hidden_states = hubert_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
            res_lst.append(hidden_states[0])

        ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
        # assert ret.shape[0] == expected_T
        assert abs(ret.shape[0] - expected_T) <= 1
        if ret.shape[0] < expected_T:
            ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
        else:
            ret = ret[:expected_T]
        return ret

def extract_rhythm_pause_features(audio_file, target_sr=16000, frame_length=1024, hop_length=512, target_fps=30):
        # 加载音频文件并重采样
        audio_each_file, sr = librosa.load(audio_file, sr=None)
        audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=target_sr)
    
        # 计算音量包络（Amplitude Envelope）
        shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
        strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
        rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
        amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        amplitude_envelope = amplitude_envelope[:len(audio_each_file) // hop_length]

        # 计算短时能量（Short-Time Energy）
        energy = np.array([
            np.sum(np.abs(audio_each_file[i:i+frame_length]**2))
            for i in range(0, len(audio_each_file), hop_length)
        ])
        energy = energy[:len(amplitude_envelope)]

        # 计算onset特征
        audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=target_sr, hop_length=hop_length, units='frames')
    
        # 确保onset_array足够大
        onset_array = np.zeros(len(amplitude_envelope), dtype=float)
    
        # 只使用有效范围内的onset点
        valid_onsets = audio_onset_f[audio_onset_f < len(onset_array)]
        onset_array[valid_onsets] = 1.0

        # 合并节奏和停顿特征
        features = np.stack([amplitude_envelope, energy, onset_array], axis=1)

        # 计算目标帧率的帧数
        duration = len(audio_each_file) / target_sr
        num_frames = int(duration * target_fps)
    
        # 重新采样到目标帧率（T, D）
        resampled_features = np.zeros((num_frames, features.shape[1]))
        for i in range(features.shape[1]):
            resampled_features[:, i] = np.interp(
                np.linspace(0, len(features) - 1, num_frames),
                np.arange(len(features)),
                features[:, i]
            )

        # 归一化特征 (选择性步骤)
        # resampled_features = resampled_features / np.max(np.abs(resampled_features), axis=0)

        return resampled_features
class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        if self.args.addHubert:
            from transformers import Wav2Vec2Processor, HubertModel
            print("Loading the Wav2Vec2 Processor...")
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("./facebook/hubert-large-ls960-ft")
            print("Loading the HuBERT Model...")
            self.hubert_model = HubertModel.from_pretrained("./facebook/hubert-large-ls960-ft")
            self.hubert_model = self.hubert_model.to(self.args.device)
            if loader_type == "test":
                self.hubert_model.eval()
            
        # self.rank = dist.get_rank()
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
        # select trainable joints
        self.smplx = smplx.create(
            model_path=self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()

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
        self.data_dir = args.data_path 
        
        if loader_type == "test": 
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length*self.args.audio_sr: 
            self.max_audio_pre_len = self.args.test_length*self.args.audio_sr
        
        if args.word_rep is not None:
            with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                
        preloaded_dir = self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"      
        if self.args.beat_align:
            if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            
        # if build_cache and dist.get_rank()==0:
        #     self.build_cache(preloaded_dir)
        if build_cache and is_main_process():
            self.build_cache(preloaded_dir)
        safe_barrier()

        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 

    
    def calculate_mean_velocity(self, save_path):
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        dir_p = self.data_dir + self.args.pose_rep + "/"
        all_list = []
        from tqdm import tqdm
        for tar in tqdm(os.listdir(dir_p)):
            if tar.endswith(".npz"):
                m_data = np.load(dir_p+tar, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length], 
                            transl=trans[i*max_length:(i+1)*max_length], 
                            expression=exps[i*max_length:(i+1)*max_length], 
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                            global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, :55, :].reshape(max_length, 55*3)
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r], 
                            transl=trans[s*max_length:s*max_length+r], 
                            expression=exps[s*max_length:s*max_length+r], 
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                            global_orient=poses[s*max_length:s*max_length+r,:3], 
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, :55, :].reshape(r, 55*3)
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0)
                joints = joints.permute(1, 0)
                dt = 1/30
            # first steps is forward diff (t+1 - t) / dt
                init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                # middle steps are second order (t+1 - t-1) / 2dt
                middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                # last step is backward diff (t - t-1) / dt
                final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
                vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                #print(vel_seq.shape)
                #.permute(1, 0).reshape(n, 55, 3)
                vel_seq_np = vel_seq.cpu().numpy()
                vel_joints_np = np.linalg.norm(vel_seq_np, axis=2) # n * 55
                all_list.append(vel_joints_np)
        avg_vel = np.mean(np.concatenate(all_list, axis=0),axis=0) # 55
        np.save(save_path, avg_vel)
        
    
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering, 
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)
        
    def __len__(self):
        return self.n_samples
    
    def idmapping(self, id):
        # map 1,2,3,4,5, 6,7,9,10,11,  12,13,15,16,17,  18,20,21,22,23,  24,25,27,28,30 to 0-24
        if id == 30: id = 8
        if id == 28: id = 14
        if id == 27: id = 19
        return id - 1
    def extract_rhythm_pause_features(self,audio_file, target_sr=16000, frame_length=1024, hop_length=512, target_fps=30):
        # 加载音频文件并重采样
        audio_each_file, sr = librosa.load(audio_file, sr=None)
        audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=target_sr)
    
        # 计算音量包络（Amplitude Envelope）
        shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
        strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
        rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
        amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        amplitude_envelope = amplitude_envelope[:len(audio_each_file) // hop_length]

        # 计算短时能量（Short-Time Energy）
        energy = np.array([
            np.sum(np.abs(audio_each_file[i:i+frame_length]**2))
            for i in range(0, len(audio_each_file), hop_length)
        ])
        energy = energy[:len(amplitude_envelope)]

        # 计算onset特征
        audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=target_sr, hop_length=hop_length, units='frames')
    
        # 确保onset_array足够大
        onset_array = np.zeros(len(amplitude_envelope), dtype=float)
    
        # 只使用有效范围内的onset点
        valid_onsets = audio_onset_f[audio_onset_f < len(onset_array)]
        onset_array[valid_onsets] = 1.0

        # 合并节奏和停顿特征
        features = np.stack([amplitude_envelope, energy, onset_array], axis=1)

        # 计算目标帧率的帧数
        duration = len(audio_each_file) / target_sr
        num_frames = int(duration * target_fps)
    
        # 重新采样到目标帧率（T, D）
        resampled_features = np.zeros((num_frames, features.shape[1]))
        for i in range(features.shape[1]):
            resampled_features[:, i] = np.interp(
                np.linspace(0, len(features) - 1, num_frames),
                np.arange(len(features)),
                features[:, i]
            )

        # 归一化特征 (选择性步骤)
        # resampled_features = resampled_features / np.max(np.abs(resampled_features), axis=0)

        return resampled_features
    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):

        self.n_out_samples = 0
        # create db for samples
        if not os.path.exists(out_lmdb_dir): os.makedirs(out_lmdb_dir)
        if len(self.args.training_speakers) == 1:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 50))# 50G
        else:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 200))# 200G
        n_filtered_out = defaultdict(int)
    
        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ext
            pose_each_file = []
            trans_each_file = []
            shape_each_file = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            word_each_sentence = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            id_pose = f_name #1_wayne_0_1_1
            
            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            if "smplx" in self.args.pose_rep:
                pose_data = np.load(pose_file, allow_pickle=True)
                assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                stride = int(30/self.args.pose_fps)
                pose_each_file = pose_data["poses"][::stride] 
                trans_each_file = pose_data["trans"][::stride]
                shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0)
                
                assert self.args.pose_fps == 30, "should 30"
                m_data = np.load(pose_file, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length], 
                            transl=trans[i*max_length:(i+1)*max_length], 
                            expression=exps[i*max_length:(i+1)*max_length], 
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                            global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r], 
                            transl=trans[s*max_length:s*max_length+r], 
                            expression=exps[s*max_length:s*max_length+r], 
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                            global_orient=poses[s*max_length:s*max_length+r,:3], 
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0) # all, 4, 3
                # print(joints.shape)
                feetv = torch.zeros(joints.shape[1], joints.shape[0])
                joints = joints.permute(1, 0, 2)
                #print(joints.shape, feetv.shape)
                feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
                #print(feetv.shape)
                contacts = (feetv < 0.01).numpy().astype(float)
                # print(contacts.shape, contacts)
                contacts = contacts.transpose(1, 0)
                pose_each_file = pose_each_file * self.joint_mask
                pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
                pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)
                # print(pose_each_file.shape)
                
                
                if self.args.facial_rep is not None:
                    logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                    facial_each_file = pose_data["expressions"][::stride]
                    if self.args.facial_norm: 
                        facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                    
            else:
                assert 120%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
                stride = int(120/self.args.pose_fps)
                with open(pose_file, "r") as pose_data:
                    for j, line in enumerate(pose_data.readlines()):
                        if j < 431: continue     
                        if j%stride != 0:continue
                        data = np.fromstring(line, dtype=float, sep=" ")
                        rot_data = rc.euler_angles_to_matrix(torch.from_numpy(np.deg2rad(data)).reshape(-1, self.joints,3), "XYZ")
                        rot_data = rc.matrix_to_axis_angle(rot_data).reshape(-1, self.joints*3) 
                        rot_data = rot_data.numpy() * self.joint_mask
                        
                        pose_each_file.append(rot_data)
                        trans_each_file.append(data[:3])
                        
                pose_each_file = np.array(pose_each_file)
                # print(pose_each_file.shape)
                trans_each_file = np.array(trans_each_file)
                shape_each_file = np.repeat(np.array(-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
                if self.args.facial_rep is not None:
                    logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                    facial_file = pose_file.replace(self.args.pose_rep, self.args.facial_rep).replace("bvh", "json")
                    assert 60%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
                    stride = int(60/self.args.pose_fps)
                    if not os.path.exists(facial_file):
                        logger.warning(f"# ---- file not found for Facial {id_pose}, skip all files with the same id ---- #")
                        self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                        continue
                    with open(facial_file, 'r') as facial_data_file:
                        facial_data = json.load(facial_data_file)
                        for j, frame_data in enumerate(facial_data['frames']):
                            if j%stride != 0:continue
                            facial_each_file.append(frame_data['weights'])
                    facial_each_file = np.array(facial_each_file)
                    if self.args.facial_norm: 
                        facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                        
            if self.args.id_rep is not None:
                int_value = self.idmapping(int(f_name.split("_")[0]))
                vid_each_file = np.repeat(np.array(int_value).reshape(1, 1), pose_each_file.shape[0], axis=0)
      
            if self.args.audio_rep is not None:
                logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
                audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
                if not os.path.exists(audio_file):
                    logger.warning(f"# ---- file not found for Audio  {id_pose}, skip all files with the same id ---- #")
                    self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    continue
                aud_ori, sr = librosa.load(audio_file)
                audio_each_file = librosa.resample(aud_ori, orig_sr=sr, target_sr=self.args.audio_sr)
                if self.args.audio_rep == "onset+amplitude":
                    from numpy.lib import stride_tricks
                    frame_length = 1024
                    # hop_length = 512
                    shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
                    strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
                    rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
                    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                    # pad the last frame_length-1 samples
                    amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
                    audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=self.args.audio_sr, units='frames')
                    onset_array = np.zeros(len(audio_each_file), dtype=float)
                    onset_array[audio_onset_f] = 1.0
                    # print(amplitude_envelope.shape, audio_each_file.shape, onset_array.shape)
                    audio_each_file_onset = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
                    beat_audio = self.extract_rhythm_pause_features(audio_file, target_sr=self.args.audio_sr, frame_length=1024, hop_length=512, target_fps=30)
                    if self.args.addHubert:
                        mel = librosa.feature.melspectrogram(y=audio_each_file, sr=self.args.audio_sr, n_mels=128, hop_length=int(self.args.audio_sr/30))
                        mel = mel[..., :-1]
                        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
                        audio_emb = audio_emb.unsqueeze(0)
                        
                        
                        hubert_feat = None
                        hubert_feat = self.get_hubert_from_16k_speech_long(self.hubert_model, self.wav2vec2_processor,
                                                                                        torch.from_numpy(aud_ori).unsqueeze(0),
                                                                                        )
                        # print('hubert_feat', hubert_feat.shape)
                        hubert_feat = F.interpolate(hubert_feat.swapaxes(-1, -2).unsqueeze(0),
                                                                    size=audio_emb.shape[-2], mode='linear', align_corners=True).swapaxes(-1, -2)
                        hubert_feat = hubert_feat.squeeze(0).numpy()
                        # print('audio_emb', audio_emb.shape)
                        # print('audio_each_file_onset', audio_each_file_onset.shape)
                        # print('hubert_feat', hubert_feat.shape)
                        # exit()
                    else:
                        hubert_feat = None
                elif self.args.audio_rep == "mfcc":
                    audio_each_file = librosa.feature.melspectrogram(y=audio_each_file, sr=self.args.audio_sr, n_mels=128, hop_length=int(self.args.audio_sr/self.args.audio_fps))
                    audio_each_file = audio_each_file.transpose(1, 0)
                    # print(audio_each_file.shape, pose_each_file.shape)
                if self.args.audio_norm and self.args.audio_rep == "wave16k": 
                    audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                    
            time_offset = 0
            if self.args.word_rep is not None:
                logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
                word_file = f"{self.data_dir}{self.args.word_rep}/{id_pose}.TextGrid"
                if not os.path.exists(word_file):
                    logger.warning(f"# ---- file not found for Word   {id_pose}, skip all files with the same id ---- #")
                    self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    continue
                tgrid = tg.TextGrid.fromFile(word_file)
                if self.args.t_pre_encoder == "bert":
                    from transformers import AutoTokenizer, BertModel
                    tokenizer = AutoTokenizer.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True)
                    model = BertModel.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True).eval()
                    list_word = []
                    all_hidden = []
                    max_len = 400
                    last = 0
                    word_token_mapping = []
                    first = True
                    for i, word in enumerate(tgrid[0]):
                        last = i
                        if (i%max_len != 0) or (i==0):
                            if word.mark == "":
                                list_word.append(".")
                            else:
                                list_word.append(word.mark)
                        else:
                            max_counter = max_len
                            str_word = ' '.join(map(str, list_word))
                            if first:
                                global_len = 0
                            end = -1
                            offset_word = []
                            for k, wordvalue in enumerate(list_word):
                                start = end+1 
                                end = start+len(wordvalue)
                                offset_word.append((start, end))
                            #print(offset_word)
                            token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                            #print(token_scan)
                            for start, end in offset_word:
                                sub_mapping = []
                                for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                                    if int(start) <= int(start_t) and int(end_t) <= int(end):
                                        #print(i+global_len)
                                        sub_mapping.append(i+global_len)
                                word_token_mapping.append(sub_mapping)
                            #print(len(word_token_mapping))
                            global_len = word_token_mapping[-1][-1] + 1    
                            list_word = []
                            if word.mark == "":
                                list_word.append(".")
                            else:
                                list_word.append(word.mark)
                            
                            with torch.no_grad():
                                inputs = tokenizer(str_word, return_tensors="pt")
                                outputs = model(**inputs)
                                last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                            all_hidden.append(last_hidden_states)
                     
                    #list_word = list_word[:10]
                    if list_word == []:
                        pass
                    else:
                        if first: 
                            global_len = 0
                        str_word = ' '.join(map(str, list_word))
                        end = -1
                        offset_word = []
                        for k, wordvalue in enumerate(list_word):
                            start = end+1 
                            end = start+len(wordvalue)
                            offset_word.append((start, end))
                        #print(offset_word)
                        token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                        #print(token_scan)
                        for start, end in offset_word:
                            sub_mapping = []
                            for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                                if int(start) <= int(start_t) and int(end_t) <= int(end):
                                    sub_mapping.append(i+global_len)
                                    #print(sub_mapping)
                            word_token_mapping.append(sub_mapping)
                        #print(len(word_token_mapping))
                        with torch.no_grad():
                            inputs = tokenizer(str_word, return_tensors="pt")
                            outputs = model(**inputs)
                            last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                        all_hidden.append(last_hidden_states)
                    last_hidden_states = np.concatenate(all_hidden, axis=0)
            
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    current_time = i/self.args.pose_fps + time_offset
                    j_last = 0
                    for j, word in enumerate(tgrid[0]): 
                        word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                        if word_s<=current_time and current_time<=word_e:
                            if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                                mapping_index = word_token_mapping[j]
                                #print(mapping_index, word_s, word_e)
                                s_t = np.linspace(word_s, word_e, len(mapping_index)+1)
                                #print(s_t)
                                for tt, t_sep in enumerate(s_t[1:]):
                                    if current_time <= t_sep:
                                        #if len(mapping_index) > 1: print(mapping_index[tt])
                                        word_each_file.append(last_hidden_states[mapping_index[tt]])
                                        break
                            else:
                                if word_n == " ":
                                    word_each_file.append(self.lang_model.PAD_token)
                                    word_each_sentence.append(word_n)
                                else:
                                    word_each_file.append(self.lang_model.get_word_index(word_n))
                                    word_each_sentence.append(word_n)
                            found_flag = True
                            j_last = j
                            break
                        else: continue   
                    if not found_flag: 
                        if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                            word_each_file.append(last_hidden_states[j_last])
                        else:
                            word_each_file.append(self.lang_model.UNK_token)
                word_each_file = np.array(word_each_file)
                word_each_sentence = np.array(word_each_sentence)
                #print(word_each_file.shape)
                
            if self.args.emo_rep is not None:
                logger.info(f"# ---- Building cache for Emo    {id_pose} and Pose {id_pose} ---- #")
                rtype, start = int(id_pose.split('_')[3]), int(id_pose.split('_')[3])
                if rtype == 0 or rtype == 2 or rtype == 4 or rtype == 6:
                    if start >= 1 and start <= 64:
                        score = 0
                    elif start >= 65 and start <= 72:
                        score = 1
                    elif start >= 73 and start <= 80:
                        score = 2
                    elif start >= 81 and start <= 86:
                        score = 3
                    elif start >= 87 and start <= 94:
                        score = 4
                    elif start >= 95 and start <= 102:
                        score = 5
                    elif start >= 103 and start <= 110:
                        score = 6
                    elif start >= 111 and start <= 118:
                        score = 7
                    else: pass
                else:
                    # you may denote as unknown in the future
                    score = 0
                emo_each_file = np.repeat(np.array(score).reshape(1, 1), pose_each_file.shape[0], axis=0)    
                #print(emo_each_file)
                
            if self.args.sem_rep is not None:
                logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
                sem_file = f"{self.data_dir}{self.args.sem_rep}/{id_pose}.txt" 
                # 读取文件，并处理可能的缺失列情况
                # sem_all = pd.read_csv(sem_file,
                #       sep='\t',
                #       header=None,  # 不指定标题行
                #       engine='python')  # 使用 Python 引擎来兼容

                # # 指定列名，并确保所有需要的列都存在
                # columns = ["name", "start_time", "end_time", "duration", "score", "keywords"]

                # # 如果某些列不存在，则添加这些列并填充为缺省值
                # for idx, col in enumerate(columns):
                #     if idx >= sem_all.shape[1]:
                #         sem_all[col] = ""  # 对缺失列填充空字符串
                #     else:
                #         sem_all.rename(columns={idx: col}, inplace=True)
                sem_all = pd.read_csv(sem_file, 
                    sep='\t', 
                    names=["name", "start_time", "end_time", "duration", "score"],
                    usecols=[0,1,2,3,4],
                    header=None)
                # we adopt motion-level semantic score here. 
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                        current_time = i/self.args.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            if score <= 0.1:
                                score = 0.
                            sem_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: sem_each_file.append(0.)
                sem_each_file = np.array(sem_each_file)
                #print(sem_each_file)
            print('sem_each_file', sem_each_file.shape)
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                word_each_sentence,
                audio_file, audio_each_file_onset,beat_audio, hubert_feat, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
                vid_each_file, emo_each_file, sem_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                                
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    @torch.no_grad()
    def get_hubert_from_16k_speech_long(self, hubert_model, wav2vec2_processor, speech, device="cuda:0"):
        """
        输入:
            speech: torch.Tensor, shape [1, T] 或 [T]，采样率必须是 16000 Hz
        输出:
            ret: torch.Tensor, shape [T_out, 1024]
        """
        # 归一到 [1, T]，float32，放到设备
        if speech.ndim == 1:
            speech = speech.unsqueeze(0)
        speech = speech.to(torch.float32).to(device)

        # 注意：Wav2Vec2Processor 不会帮你重采样，必须保证采样率确为 16000
        input_values_all = wav2vec2_processor(
            speech, return_tensors="pt", sampling_rate=16000
        ).input_values.squeeze(0).to(device)  # [1, T] -> [T] 再还原为 [1, T] 下方切片时再加 batch

        # HuBERT CNN 等效核与步幅
        kernel = 400
        stride = 320
        clip_length = stride * 1000  # 每段时间步 N=1000，对应原波形长度 stride*N

        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride

        res_lst = []

        # 切片前向
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)

            input_values = input_values_all[:, start_idx:end_idx]  # [1, seg_T]
            hidden_states = hubert_model(input_values).last_hidden_state  # [1, T_seg, 1024]
            res_lst.append(hidden_states[0])

        # 末段
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all

        if input_values.shape[1] >= kernel:
            hidden_states = hubert_model(input_values).last_hidden_state  # [1, T_last, 1024]
            res_lst.append(hidden_states[0])

        # 拼接并补裁到 expected_T
        ret = torch.cat(res_lst, dim=0).detach().cpu()  # [T_cat, 1024]
        assert abs(ret.shape[0] - expected_T) <= 1, f"HuBERT steps mismatch: {ret.shape[0]} vs {expected_T}"

        if ret.shape[0] < expected_T:
            ret = F.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
        else:
            ret = ret[:expected_T]

        return ret  # [T_out, 1024]

    def _sample_from_clip(
        self, dst_lmdb_env, word_each_sentence, audio_file, audio_each_file,beat_audio,hubert_feat, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        # audio_start = int(self.alignment[0] * self.args.audio_fps)
        # pose_start = int(self.alignment[1] * self.args.pose_fps)
        #logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        # audio_each_file = audio_each_file[audio_start:]
        # pose_each_file = pose_each_file[pose_start:]
        # trans_each_file = 
        #logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        #print(pose_each_file.shape)
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s
        #print(round_seconds_skeleton)
        if audio_each_file != []:
            if self.args.audio_rep != "wave16k":
                round_seconds_audio = len(audio_each_file) // self.args.audio_fps # assume 16,000,00 / 16,000 = 100 s
                round_seconds_audio_beat = beat_audio.shape[0] // 30
                if self.args.addHubert:
                    round_seconds_audio_hubert = hubert_feat.shape[0] // 30
                    print('round_seconds_audio', round_seconds_audio)
                    print('round_seconds_audio_hubert', round_seconds_audio_hubert)
            elif self.args.audio_rep == "mfcc":
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_fps
            else:
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
                if self.args.addHubert:
                    round_seconds_audio_hubert = hubert_feat.shape[0] // 30
            if facial_each_file != []:
                round_seconds_facial = facial_each_file.shape[0] // self.args.pose_fps
                logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
                # print('round_seconds_skeleton', round_seconds_skeleton)
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial, round_seconds_audio_hubert, round_seconds_audio_beat)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial, round_seconds_audio_hubert,round_seconds_audio_beat)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
                # print('round_seconds_audio', round_seconds_audio)
                # print('round_seconds_skeleton_min', round_seconds_skeleton)
                # print('round_seconds_facial', round_seconds_facial)
                # print('max_round', max_round)
            else:
                logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        # print('clean_first_seconds', clean_first_seconds)
        # print('clean_final_seconds', clean_final_seconds)
        # print('clip_s_t', clip_s_t)
        # print('clip_e_t', clip_e_t)
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [150,90*15]
        # print('clip_s_f_audio', clip_s_f_audio)
        # print('clip_e_f_audio', clip_e_f_audio)
        # print('clip_s_f_pose', clip_s_f_pose)
        # print('clip_e_f_pose', clip_e_f_pose)
        clip_s_f_audio_hubert, clip_e_f_audio_hubert = 30 * clip_s_t, clip_e_t * 30
        clip_s_f_audio_beat, clip_e_f_audio_beat = 30 * clip_s_t, clip_e_t * 30
        # print('clip_s_f_audio_hubert', clip_s_f_audio_hubert)
        # print('clip_e_f_audio_hubert', clip_e_f_audio_hubert)
        # exit()
        for ratio in self.args.multi_length_training:
            if is_test:# stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
                cut_emo = int(self.ori_length*ratio)
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")
            
            if audio_each_file != []:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                """
                for audio sr = 16000, fps = 15, pose_length = 34, 
                audio short length = 36266.7 -> 36266
                this error is fine.
                """
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
                audio_short_length_beat = math.floor(cut_length / self.args.pose_fps * 30)
                logger.info(f"beat audio from frame {clip_s_f_audio_beat} to {clip_e_f_audio_beat}, length {audio_short_length_beat}")
                if self.args.addHubert:
                    audio_short_length_hubert = math.floor(cut_length / self.args.pose_fps * 30)
                    logger.info(f"hubert audio from frame {clip_s_f_audio_hubert} to {clip_e_f_audio_hubert}, length {audio_short_length_hubert}")

            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_facial_list = []
            sample_shape_list = []
            sample_word_list = []
            sample_word_sentence_list = []
            sample_word_sentence_pre_list = []
            sample_emo_list = []
            sample_emo_pre_list = []
            sample_sem_list = []
            sample_vid_list = []
            sample_trans_list = []
           
            sample_audio_hubert_list = []
            sample_audio_beat_list = []
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length 
                sample_pose = pose_each_file[start_idx:fin_idx]

                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                if self.args.audio_rep is not None:
                    audio_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)
                    audio_end = audio_start + audio_short_length
                    sample_audio = audio_each_file[audio_start:audio_end]
                    
                    audio_start_beat = clip_s_f_audio_beat + math.floor(i * self.args.stride * 30 / self.args.pose_fps)
                    audio_end_beat = audio_start_beat + audio_short_length_beat
                    sample_audio_beat = beat_audio[audio_start_beat:audio_end_beat]

                    if self.args.addHubert:
                        audio_start_hubert = clip_s_f_audio_hubert + math.floor(i * self.args.stride * 30 / self.args.pose_fps)
                        audio_end_hubert = audio_start_hubert + audio_short_length_hubert
                        sample_audio_hubert = hubert_feat[audio_start_hubert:audio_end_hubert]
                else:
                    sample_audio = np.array([-1])
                
                sample_facial = facial_each_file[start_idx:fin_idx] if self.args.facial_rep is not None else np.array([-1])
                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                
                # sample_word_sentence = word_each_sentence[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                # sample_word_sentence  = process_annotated_sentence(sample_word_sentence)
                if is_test:
                    num_emo = cut_length // cut_emo
                    res_emo_clip = cut_length % cut_emo
                    sample_emo = np.zeros((cut_length, 1))
                    n = len(word_each_sentence)
                    remain = n%8
                    word_each_sentence = word_each_sentence[:n-remain]

                    roundt = (n - 4) // (64 - 4)
                    round_l = 64 - 4
                    for i in range(0, roundt):
                        sample_word_sentence_pre = process_annotated_sentence(word_each_sentence[i*round_l:(i+1)*round_l+4])
                        sample_word_sentence_pre_list.append(sample_word_sentence_pre)
                        sample_emo_pre = self.get_max_score_label(audio_file, i*round_l, (i+1)*round_l+4)
                        sample_emo_pre_list.append(sample_emo_pre)


                    # for i in range(num_emo):
                    #     sample_emo[i*cut_emo:(i+1)*cut_emo] = self.get_max_score_index(audio_file, i*cut_emo, (i+1)*cut_emo) if self.args.emo_rep is not None else np.array([-1])
                    #     # sample_word_sentence_pre =  process_annotated_sentence(word_each_sentence[i*cut_emo:(i+1)*cut_emo]) 
                    #     # sample_word_sentence_pre_list.append(sample_word_sentence_pre)
                    # if res_emo_clip != 0:
                    #     sample_emo[num_emo*cut_emo:] = self.get_max_score_index(audio_file, num_emo*cut_emo, fin_idx) if self.args.emo_rep is not None else np.array([-1])
                    #     # sample_word_sentence_pre = process_annotated_sentence(word_each_sentence[num_emo*cut_emo:])
                    #     # sample_word_sentence_pre_list.append(sample_word_sentence_pre)
                    sample_word_sentence = sample_word_sentence_pre_list
                    sample_emo = sample_emo_pre_list
                else:
                    sample_emo = self.get_max_score_label(audio_file, start_idx, fin_idx) if self.args.emo_rep is not None else np.array([-1])
                    sample_word_sentence = word_each_sentence[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                    sample_word_sentence  = process_annotated_sentence(sample_word_sentence)
                    # sample_emo = 
                # sample_emo = emo_each_file[start_idx:fin_idx] if self.args.emo_rep is not None else np.array([-1])
                sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])
                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
                
                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = (sample_pose != [])
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_facial_list.append(sample_facial)
                        sample_shape_list.append(sample_shape)
                        sample_word_list.append(sample_word)
                        sample_word_sentence_list.append(sample_word_sentence)
                        sample_vid_list.append(sample_vid)
                        sample_emo_list.append(sample_emo)
                        sample_sem_list.append(sample_sem)
                        sample_trans_list.append(sample_trans)

                        sample_audio_beat_list.append(sample_audio_beat)
                        if self.args.addHubert:
                            sample_audio_hubert_list.append(sample_audio_hubert)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, audio,beat,hubert,facial, shape, word, sentence, vid, emo, sem, trans in zip(
                        sample_pose_list,
                        sample_audio_list,
                        sample_audio_beat_list,
                        sample_audio_hubert_list,
                        sample_facial_list,
                        sample_shape_list,
                        sample_word_list,
                        sample_word_sentence_list,
                        sample_vid_list,
                        sample_emo_list,
                        sample_sem_list,
                        sample_trans_list,):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [pose, audio, beat, hubert, facial, shape, word, sentence, emo, sem, vid, trans]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out
    
    @torch.no_grad()
    def get_max_score_label(self, wav_file, start_frame, end_frame, model=emo_model, fps=30, target_sr=16000):
        """
        提取音频片段并从模型中找到分数最高的标签，同时将采样率设置为16kHz。

        参数:
        - wav_file: str, 音频文件的路径
        - start_frame: int, 起始帧
        - end_frame: int, 结束帧
        - model: 语音模型，用于生成预测结果
        - fps: int, 音频的帧率，默认值为 30
        - target_sr: int, 目标采样率，默认为 16000

        返回:
        - max_score_label_repeated: np.ndarray, 形状为 (帧数长度, 1)，内容为分数最高的标签
        """

        # 读取音频文件，并将采样率转换为 16kHz
        audio, sr = librosa.load(wav_file, sr=target_sr)  # sr=16000 表示强制采样率为16kHz

        # 计算起始时间和结束时间 (秒)
        start_time = start_frame / fps
        end_time = end_frame / fps

        # 根据时间截取音频片段
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]

        # 将音频片段输入模型进行预测
        res = model.generate(audio_segment, granularity="utterance", extract_embedding=False)

        # 过滤掉不需要的标签
        labels = res[0]['labels']
        scores = res[0]['scores']
        exclude_labels = ['其他/other', '<unk>']

        # 保留需要的标签和对应的分数
        # filtered_labels_scores = [(label, score) for label, score in zip(labels, scores) if label not in exclude_labels]
        filtered_labels_scores = [(label, score) for label, score in zip(labels, scores)]


        # 找到分数最高的标签
        if filtered_labels_scores:
            max_score_label = max(filtered_labels_scores, key=lambda x: x[1])[0]
        else:
            max_score_label = "无效标签"  # 处理没有符合条件标签的情况，返回 "无效标签" 表示无效

        # 计算帧数长度
        frame_length = end_frame - start_frame

        # 将分数最高的标签重复至 (帧数长度, 1)
        # max_score_label_repeated = np.array([[max_score_label]] * frame_length)
        max_score_label_repeated = max_score_label

        return max_score_label_repeated
    
    @torch.no_grad()
    def get_max_score_index(self, wav_file, start_frame, end_frame, model=emo_model, fps=30, target_sr=16000):
        """
        提取音频片段并从模型中找到分数最高的标签对应的索引，同时将采样率设置为16kHz。
    
        参数:
        - wav_file: str, 音频文件的路径
        - start_frame: int, 起始帧
        - end_frame: int, 结束帧
        - model: 语音模型，用于生成预测结果
        - fps: int, 音频的帧率，默认值为 30
        - target_sr: int, 目标采样率，默认为 16000

        返回:
        - max_score_index_repeated: np.ndarray, 形状为 (帧数长度, 1)，内容为分数最高的索引
        """

        # 读取音频文件，并将采样率转换为 16kHz
        audio, sr = librosa.load(wav_file, sr=target_sr)  # sr=16000 表示强制采样率为16kHz

        # 计算起始时间和结束时间 (秒)
        start_time = start_frame / fps
        end_time = end_frame / fps

        # 根据时间截取音频片段
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]

        # 将音频片段输入模型进行预测
        res = model.generate(audio_segment, granularity="utterance", extract_embedding=False)

        # 过滤掉不需要的标签
        labels = res[0]['labels']
        scores = res[0]['scores']
        exclude_labels = ['其他/other', '<unk>']

        # 保留需要的标签和对应的分数
        # filtered_labels_scores = [(index, score) for index, (label, score) in enumerate(zip(labels, scores)) if label not in exclude_labels]
        filtered_labels_scores = [(index, score) for index, (label, score) in enumerate(zip(labels, scores))]
        # 找到分数最高的索引
        if filtered_labels_scores:
            filtered_indexes, filtered_scores = zip(*filtered_labels_scores)
            max_score_index = filtered_indexes[filtered_scores.index(max(filtered_scores))]
        else:
            max_score_index = -1  # 处理没有符合条件标签的情况，返回 -1 表示无效索引

        # 计算帧数长度
        frame_length = end_frame - start_frame

        # 将分数最高的索引重复至 (帧数长度, 1)
        max_score_index_repeated = np.array([[max_score_index]] * frame_length)

        return max_score_index_repeated
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            tar_pose, in_audio, in_beat, in_hubert, in_facial, in_shape, in_word, in_sentence, emo, sem, vid, trans = sample
            #print(in_shape)
            #vid = torch.from_numpy(vid).int()
            
            # emo = torch.from_numpy(emo.copy()).int()
            sem = torch.from_numpy(sem.copy()).float() 
            in_audio = torch.from_numpy(in_audio.copy()).float() 
            in_beat = torch.from_numpy(in_beat.copy()).float() 
            in_hubert = torch.from_numpy(in_hubert.copy()).float()
            in_word = torch.from_numpy(in_word.copy()).float() if self.args.word_cache else torch.from_numpy(in_word.copy()).int() 
            # in_sentence = in_sentence.copy()
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose.copy()).float()
                trans = torch.from_numpy(trans.copy()).float()
                in_facial = torch.from_numpy(in_facial.copy()).float()
                vid = torch.from_numpy(vid.copy()).float()
                in_shape = torch.from_numpy(in_shape.copy()).float()
            else:
                in_shape = torch.from_numpy(in_shape.copy()).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans.copy()).reshape((trans.shape[0], -1)).float()
                vid = torch.from_numpy(vid.copy()).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose.copy()).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial.copy()).reshape((in_facial.shape[0], -1)).float()
            return {"pose":tar_pose, "audio":in_audio,"beat":in_beat,"hubert":in_hubert, "facial":in_facial, "beta": in_shape, "word":in_word, "sentence": in_sentence,"id":vid, "emo":emo, "sem":sem, "trans":trans}

         
class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        #self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
#         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
#         diff = np.mean(diff)

#         # th = 0.017
#         th = 0.02 #0.02  # exclude 3594
#         if diff < th:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
# #         th = 3.5 #0.02  # exclude 3594
# #         if 3.5 < diff < 5:
# #             if verbose:
# #                 print("skip - check_pose_diff {:.5f}".format(diff))
# #             return True
#         else:
#             if verbose:
#                 print("pass - check_pose_diff {:.5f}".format(diff))
        return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False