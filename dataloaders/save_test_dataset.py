# --- make imports work both as "python -m dataloaders.save_train_dataset"
# --- and as "python dataloaders/save_train_dataset.py" without changing tree
import os, sys, pathlib
if __package__ is None or __package__ == "":
    # executed as a script: add project root (SemTalk) into sys.path
    THIS_FILE = pathlib.Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[1]   # .../SemTalk
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import io
import os
import pickle
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import rotation_conversions as rc
import clip
from utils import other_tools
from dataloaders.data_tools import joints_list
from dataloaders.build_vocab import Vocab

# -------------------------
# Processor: plug your code
# -------------------------

class Processor:
    """
    Wrap your _load_data-style logic here.
    Replace the 3 TODOs in __init__ to bind:
      - rc: rotation conversion utils (axis_angle_to_matrix, matrix_to_rotation_6d)
      - vq models: vq_model_face/upper/hands/lower, each having map2index/map2zq
      - encode_text: CLIP text encoder
      - joint masks & args
    """

    def __init__(self, device: torch.device, args):
        self.device = device
        import importlib
        
        self.args = args
        # === TODO(2): bind your VQ models & encode_text ===
        rvq_model_module = __import__(f"models.rvq", fromlist=["something"])
        self.args.vae_test_dim = 106 # face
        self.vq_model_face = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_face, "./weights/pretrained_vq/rvq_face_600.bin", "vq_face")

        self.args.vae_test_dim = 78 # upper body
        self.vq_model_upper = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_upper, "./weights/pretrained_vq/rvq_upper_500.bin", "vq_upper")

        self.args.vae_test_dim = 180 # hands
        self.vq_model_hands = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_hands, "./weights/pretrained_vq/rvq_hands_500.bin", "vq_hands")
        
        self.args.vae_test_dim = 61 # lower body
        self.vq_model_lower = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_lower, "./weights/pretrained_vq/rvq_lower_600.bin", "vq_lower")
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.clip_model = self.load_and_freeze_clip(device=device)
        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
    def encode_text(self, raw_text, device):
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
    
    @torch.no_grad()
    def load_and_freeze_clip(self, clip_version='ViT-B/32', device='cuda:0'):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    @torch.no_grad()
    def process(self, dict_data: Dict[str, Any], mode: str = "train") -> Dict[str, np.ndarray]:
        """
        对单个样本完成你给出的处理，返回可序列化为 pickle 的 numpy 字典。
        假设 dict_data 中至少包含：
            pose(轴角, [1,64,169])、trans、facial、audio、hubert、beat、word、sentence、emo、beta、id ...
        """
        # ---- 取字段，并搬到 device ----
        tar_pose_raw = dict_data["pose"].to(self.device)            # [B,N,169]
        tar_pose = tar_pose_raw[:, :, :165]                          # 55*3 轴角
        tar_contact = tar_pose_raw[:, :, 165:169]                    # [B,N,4]
        tar_trans = dict_data["trans"].to(self.device)               # [B,N,3]
        tar_exps = dict_data["facial"].to(self.device)               # [B,N,100]
        in_audio = dict_data["audio"].to(self.device)                # [B,T,2] or [T,2]
        in_hubert = dict_data["hubert"].to(self.device)              # [B,N,1024]
        in_beat = dict_data["beat"].to(self.device)                  # [B,N,3]
        in_word = dict_data["word"].to(self.device)                  # [B,N] / tokens
        in_sentence = dict_data["sentence"]                          # list-like
        in_emo = dict_data["emo"]                                    # list-like
        tar_beta = dict_data["beta"].to(self.device)                 # [B,N,300]
        tar_id = dict_data["id"].to(self.device).long()              # [B,N,1]

        B, N = tar_pose.shape[0], tar_pose.shape[1]
        j = self.joints
        print('tar_pose_raw:', tar_pose_raw.shape)
        print('tar_pose:', tar_pose.shape)
        print('tar_contact:', tar_contact.shape)
        print("tar_trans:", tar_trans.shape)
        print("tar_exps:", tar_exps.shape)
        print("in_audio:", in_audio.shape)
        print("in_hubert:", in_hubert.shape)
        print("in_beat:", in_beat.shape)
        print("in_word:", in_word.shape)
        print("in_sentence:", in_sentence)
        print("len(in_sentence):", len(in_sentence))
        print("in_emo:", in_emo)    
        print("tar_beta:", tar_beta.shape)
        print("tar_id:", tar_id.shape)
        print("mode:", mode)
        print("B,N:", B,N)
        print("joints:", j)
        # exit(0)
        # ---- 文本编码 ----
        if mode == "train":
            feat_clip_text = self.encode_text(in_sentence, self.device)
            emo_clip_text  = self.encode_text(in_emo, self.device)
        else:
            in_sentence = in_sentence
            in_emo_sep = in_emo
            feat_clip_text_list, emo_clip_text_list = [], []
            assert len(in_sentence) == len(in_emo_sep)
            for i in range(len(in_sentence)):
                feat_clip_text_i = self.encode_text(in_sentence[i], self.device)
                emo_clip_text_i  = self.encode_text(in_emo_sep[i], self.device)
                feat_clip_text_list.append(feat_clip_text_i)
                emo_clip_text_list.append(emo_clip_text_i)
            feat_clip_text = torch.stack(feat_clip_text_list, dim=1)
            emo_clip_text  = torch.stack(emo_clip_text_list, dim=1)

        # ---- 各部分拆分 + 轴角->旋转矩阵->rot6d ----
        # jaw
        tar_pose_jaw = tar_pose[:, :, 66:69]  # [B,N,3]
        m_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(B, N, 1, 3))
        tar_pose_jaw_6d = rc.matrix_to_rotation_6d(m_jaw).reshape(B, N, 6)
        tar_pose_face = torch.cat([tar_pose_jaw_6d, tar_exps], dim=2)  # [B,N,106]

        # hands: 25*3:55*3 -> 30 joints
        tar_pose_hands_aa = tar_pose[:, :, 25*3:55*3].reshape(B, N, 30, 3)
        m_hands = rc.axis_angle_to_matrix(tar_pose_hands_aa)
        tar_pose_hands = rc.matrix_to_rotation_6d(m_hands).reshape(B, N, 30 * 6)  # [B,N,180]

        # upper: 13 joints by mask
        pose_upper_aa = tar_pose[:, :, self.joint_mask_upper.astype(bool)].reshape(B, N, 13, 3)
        m_upper = rc.axis_angle_to_matrix(pose_upper_aa)
        tar_pose_upper = rc.matrix_to_rotation_6d(m_upper).reshape(B, N, 13 * 6)   # [B,N,78]

        # lower: 9 joints by mask
        pose_leg_aa = tar_pose[:, :, self.joint_mask_lower.astype(bool)].reshape(B, N, 9, 3)
        m_leg = rc.axis_angle_to_matrix(pose_leg_aa)
        tar_pose_leg = rc.matrix_to_rotation_6d(m_leg).reshape(B, N, 9 * 6)        # [B,N,54]

        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)  # [B,N,61]

        # VQ indices & zq
        tar_index_value_face_top  = self.vq_model_face.map2index(tar_pose_face)     # [B, N/4, 6]
        tar_index_value_upper_top = self.vq_model_upper.map2index(tar_pose_upper)   # [B, N/4, 6]
        tar_index_value_hands_top = self.vq_model_hands.map2index(tar_pose_hands)   # [B, N/4, 6]
        tar_index_value_lower_top = self.vq_model_lower.map2index(tar_pose_lower)   # [B, N/4, 6]

        zq_face  = self.vq_model_face.map2zq(tar_pose_face)
        zq_upper = self.vq_model_upper.map2zq(tar_pose_upper)
        zq_hands = self.vq_model_hands.map2zq(tar_pose_hands)
        zq_lower = self.vq_model_lower.map2zq(tar_pose_lower)

        # 全 55 轴角 -> rot6d
        m_all = rc.axis_angle_to_matrix(tar_pose.reshape(B, N, j, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(m_all).reshape(B, N, j * 6)        # [B,N,330]
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)     # [B,N,337]

        # 组织输出（去掉 batch 维）
        out = {
             # 原始/中间特征 —— 去掉 batch 维
            "beat":                _to_np_squeezed(in_beat),
            "hubert":              _to_np_squeezed(in_hubert),


            "zq_face":             _to_np_squeezed(zq_face),
            "zq_upper":            _to_np_squeezed(zq_upper),
            "zq_hands":            _to_np_squeezed(zq_hands),
            "zq_lower":            _to_np_squeezed(zq_lower),


            "in_audio":            _to_np_squeezed(in_audio),
            "in_word":             _to_np_squeezed(in_word),

            "tar_trans":           _to_np_squeezed(tar_trans),
            "tar_exps":            _to_np_squeezed(tar_exps),
            "tar_beta":            _to_np_squeezed(tar_beta),
            "tar_pose":            _to_np_squeezed(tar_pose),

            "tar_index_value_face_top":   _to_np_squeezed(tar_index_value_face_top),
            "tar_index_value_upper_top":  _to_np_squeezed(tar_index_value_upper_top),
            "tar_index_value_hands_top":  _to_np_squeezed(tar_index_value_hands_top),
            "tar_index_value_lower_top":  _to_np_squeezed(tar_index_value_lower_top),

            "tar_id":              _to_np_squeezed(tar_id),
            "latent_all":          _to_np_squeezed(latent_all),
            "tar_contact":         _to_np_squeezed(tar_contact),

            # 文本编码结果 —— 去掉 batch 维
            "feat_clip_text":      _to_np_squeezed(feat_clip_text),
            "emo_clip_text":       _to_np_squeezed(emo_clip_text),

        }

        return out


# --------------------------------
# Pickle writer (替代原 LMDB + NPZ)
# --------------------------------
def write_pickle(stream: Iterable[Dict[str, np.ndarray]], dst_pkl: str) -> int:
    """
    将样本流写入一个 .pkl 文件（内容是 [sample_dict, sample_dict, ...] 的列表）
    注意：这会把所有样本收集到内存后一次性写盘；如果样本极多且内存有限，可改成分块多文件。
    """
    os.makedirs(os.path.dirname(dst_pkl), exist_ok=True)
    buffer: List[Dict[str, np.ndarray]] = []
    count = 0
    for sample in tqdm(stream, desc="Collecting samples"):
        buffer.append(sample)
        count += 1
    with open(dst_pkl, "wb") as f:
        # 使用较新的协议以更好地支持大数组
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return count


def _to_np(x):
    """torch.Tensor / np.ndarray / list -> np.ndarray"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_np_squeezed(x):
    """
    保存前去掉最前面的 batch 维(=1)；其他维保持不变。
    只对数值类数组做 squeeze，object 字段（sentence/emo）不在这里处理。
    """
    a = _to_np(x)
    if a.ndim >= 1 and a.shape[0] == 1:
        return np.squeeze(a, axis=0)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beat2_add_hubert", help="Dataset name for import")
    ap.add_argument("--addHubert", default=True, help="use hubert or not")
    ap.add_argument("--stride", default=20, type=int, help="stride for dataset")
    ap.add_argument("--pose_length", default=64, type=int, help="pose length for dataset")
    ap.add_argument("--pose_rep", default="smplxflame_30", help="pose representation for dataset")
    ap.add_argument("--pose_fps", default=30, type=int, help="pose fps for dataset")
    ap.add_argument("--audio_rep", default="onset+amplitude", help="audio representation for dataset")
    ap.add_argument("--audio_sr", default=16000, type=int, help="audio sample rate for dataset")
    ap.add_argument("--audio_fps", default=16000, type=int, help= "audio fps for dataset")
    ap.add_argument("--audio_norm", default=False, type=bool, help="use audio norm or not")
    ap.add_argument("--emo_rep", default="emo", help="emotion representation for dataset")
    ap.add_argument("--sem_rep", default="sem", help="semantic representation for dataset")
    ap.add_argument("--t_pre_encoder", default="fasttext", help="tokenizer and text encoder for dataset")
    ap.add_argument("--word_cache", default=False, type=bool)
    ap.add_argument("--test_length", default=64, type=int, help="test length for dataset")
    ap.add_argument("--facial_rep", default="smplxflame_30", help="facial representation for dataset")
    ap.add_argument("--facial_norm", default=False,type=bool, help="use facial norm or not")
    ap.add_argument("--id_rep", default="onehot", help="id representation for dataset")
    ap.add_argument("--ori_joints", default="beat_smplx_joints", help="original joints for dataset")
    ap.add_argument("--tar_joints", default="beat_smplx_full", help="target joints for dataset")
    ap.add_argument("--data_path_1", default="./weights/", help="original data path for dataset")
    ap.add_argument("--data_path", default="./BEAT2/beat_english_v2.0.0/", help="cache data path for dataset")
    ap.add_argument("--root_path", default="./", help="root path for dataset")
    ap.add_argument("--cache_path", default="./datasets/beat2_cache_2/", help="cache path for dataset")
    ap.add_argument("--word_rep", default="textgrid", help="word representation for dataset")
    ap.add_argument("--beat_align", default=True, type=bool)
    ap.add_argument("--training_speakers", type=int, nargs="*", default=[2])
    ap.add_argument("--multi_length_training", default=[1.0], type=float, nargs="*")
    ap.add_argument("--new_cache", default=False, type=bool)
    ap.add_argument("--disable_filtering", default=False, type=bool)
    ap.add_argument("--clean_first_seconds", default=0, type=int)
    ap.add_argument("--clean_final_seconds", default=0, type=int)
    ap.add_argument("--additional_data", default=False, type=bool)

    # 改为 pkl 目标路径
    ap.add_argument("--dst_pkl", default="./datasets/beat2_semtalk_test.pkl" ,help="Destination pickle file")

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--vae_test_dim", type=int, default=128) 
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    # === 如果目标 pkl 已存在则直接退出 ===
    if os.path.isfile(args.dst_pkl):
        print(f"[Skip] 目标文件已存在：{args.dst_pkl}，不再处理。")
        return
    test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=getattr(test_data, "collate_fn", None)
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = Processor(device=device, args=args)

    def sample_stream() -> Iterable[Dict[str, np.ndarray]]:
        for batch in tqdm(loader, desc="Processing"):
            # batch 可能是 dict 或 list/tuple -> dict_data
            dict_data = batch if isinstance(batch, dict) else batch[0]
            out = processor.process(dict_data, mode=args.mode)
            yield out

    total = write_pickle(
        stream=sample_stream(),
        dst_pkl=args.dst_pkl,
    )
    print(f"Done. Wrote {total} samples to {args.dst_pkl}")

    # ===== 验证读取：构建最简 Dataset，取第 0 个样本打印 =====
    print("\nVerifying written PKL by reading one sample ...")
    test_ds = PickleDataset(args.dst_pkl)
    test_function_data = test_ds[0]  # <- 你要的变量名
    print(f"Pickle entries: {len(test_ds)}")
    # 按 key 排序打印 shape / dtype
    for k in sorted(test_function_data.keys()):
        v = test_function_data[k]
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", type(v))
        print(f"{k}: shape = {shape}, dtype = {dtype}")


class PickleDataset(torch.utils.data.Dataset):
    """
    读取 write_pickle 写出的单文件 .pkl（内容为 list[dict]）。
    为避免频繁 IO，这里一次性 load 到内存；如果数据巨大可改为多文件分块。
    """
    def __init__(self, pkl_path: str):
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.samples: List[Dict[str, Any]] = pickle.load(f)
        if not isinstance(self.samples, list):
            raise ValueError(f"Pickle content is not a list, got {type(self.samples)}")

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


if __name__ == "__main__":
    main()
