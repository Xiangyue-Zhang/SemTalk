import os
import io
import pickle
from typing import Dict, Any, Iterable, List

import clip
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from utils import other_tools
from utils import rotation_conversions as rc
from utils.project_paths import pretrained_vq_path
from dataloaders.data_tools import joints_list


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_np_squeezed(x):
    a = _to_np(x)
    if a.ndim >= 1 and a.shape[0] == 1:
        return np.squeeze(a, axis=0)
    return a


def pack_npz_bytes(arr_dict: Dict[str, np.ndarray]) -> bytes:
    with io.BytesIO() as bio:
        np.savez_compressed(bio, **arr_dict)
        return bio.getvalue()


def write_lmdb(stream: Iterable[Dict[str, np.ndarray]], dst_path: str, map_size_gb: int = 300, commit_interval: int = 2048) -> int:
    os.makedirs(dst_path, exist_ok=True)
    env = lmdb.open(dst_path, map_size=int(map_size_gb * (1024 ** 3)), subdir=True, lock=True)
    txn = env.begin(write=True)
    dp_id = 0
    try:
        for sample in tqdm(stream, desc="Writing LMDB"):
            value_bytes = pack_npz_bytes(sample)
            key = f"{dp_id:010d}".encode("utf-8")
            txn.put(key, value_bytes)
            dp_id += 1
            if dp_id % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    finally:
        env.close()
    return dp_id


def write_pickle(stream: Iterable[Dict[str, np.ndarray]], dst_pkl: str) -> int:
    os.makedirs(os.path.dirname(dst_pkl), exist_ok=True)
    buffer: List[Dict[str, np.ndarray]] = []
    count = 0
    for sample in tqdm(stream, desc="Collecting samples"):
        buffer.append(sample)
        count += 1
    with open(dst_pkl, "wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return count


def lmdb_entries(lmdb_dir: str) -> int:
    try:
        env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, max_readers=1, subdir=True)
        with env.begin(buffers=True) as txn:
            return txn.stat()["entries"]
    except Exception:
        return 0


class PickleDataset(torch.utils.data.Dataset):
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
        for k in ("sentence", "emo"):
            if k in sample and isinstance(sample[k], np.ndarray) and sample[k].dtype == object:
                sample[k] = sample[k].tolist()
        return sample


class LMDBNPZDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_dir: str):
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, max_readers=1, subdir=True)
        with self.env.begin(buffers=True) as txn:
            self.length = txn.stat()["entries"]

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


class PreprocessProcessor:
    def __init__(self, device: torch.device, args, include_sem_mean: bool):
        self.device = device
        self.args = args
        self.include_sem_mean = include_sem_mean

        rvq_model_module = __import__("models.rvq", fromlist=["something"])
        self.args.vae_test_dim = 106
        self.vq_model_face = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_face, str(pretrained_vq_path("face")), "vq_face")

        self.args.vae_test_dim = 78
        self.vq_model_upper = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_upper, str(pretrained_vq_path("upper")), "vq_upper")

        self.args.vae_test_dim = 180
        self.vq_model_hands = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_hands, str(pretrained_vq_path("hands")), "vq_hands")

        self.args.vae_test_dim = 61
        self.vq_model_lower = getattr(rvq_model_module, "RVQVAE")(self.args).to(args.device)
        other_tools.load_checkpoints(self.vq_model_lower, str(pretrained_vq_path("lower")), "vq_lower")

        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
        self.joints = 55

        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.clip_model = self.load_and_freeze_clip(device=device)
        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()

    def encode_text(self, raw_text, device):
        text = clip.tokenize(raw_text, truncate=True).to(device)
        return self.clip_model.encode_text(text).float()

    @torch.no_grad()
    def load_and_freeze_clip(self, clip_version="ViT-B/32", device="cuda:0"):
        clip_model, _ = clip.load(clip_version, device=device, jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    @torch.no_grad()
    def process(self, dict_data: Dict[str, Any], mode: str = "train") -> Dict[str, np.ndarray]:
        tar_pose_raw = dict_data["pose"].to(self.device)
        tar_pose = tar_pose_raw[:, :, :165]
        tar_contact = tar_pose_raw[:, :, 165:169]
        tar_trans = dict_data["trans"].to(self.device)
        tar_exps = dict_data["facial"].to(self.device)
        in_audio = dict_data["audio"].to(self.device)
        in_hubert = dict_data["hubert"].to(self.device)
        in_beat = dict_data["beat"].to(self.device)
        in_word = dict_data["word"].to(self.device)
        in_sentence = dict_data["sentence"]
        in_emo = dict_data["emo"]
        tar_beta = dict_data["beta"].to(self.device)
        tar_id = dict_data["id"].to(self.device).long()
        in_sem = dict_data.get("sem")
        if in_sem is not None:
            in_sem = in_sem.to(self.device)

        B, N = tar_pose.shape[0], tar_pose.shape[1]
        j = self.joints

        if mode == "train":
            feat_clip_text = self.encode_text(in_sentence, self.device)
            emo_clip_text = self.encode_text(in_emo, self.device)
        else:
            feat_clip_text_list, emo_clip_text_list = [], []
            assert len(in_sentence) == len(in_emo)
            for i in range(len(in_sentence)):
                feat_clip_text_list.append(self.encode_text(in_sentence[i], self.device))
                emo_clip_text_list.append(self.encode_text(in_emo[i], self.device))
            feat_clip_text = torch.stack(feat_clip_text_list, dim=1)
            emo_clip_text = torch.stack(emo_clip_text_list, dim=1)

        sem_mean = None
        if self.include_sem_mean and in_sem is not None:
            sem_mean = in_sem.reshape(B, N // 4, 4).mean(-1)

        tar_pose_jaw = tar_pose[:, :, 66:69]
        m_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(B, N, 1, 3))
        tar_pose_jaw_6d = rc.matrix_to_rotation_6d(m_jaw).reshape(B, N, 6)
        tar_pose_face = torch.cat([tar_pose_jaw_6d, tar_exps], dim=2)

        tar_pose_hands_aa = tar_pose[:, :, 25 * 3:55 * 3].reshape(B, N, 30, 3)
        m_hands = rc.axis_angle_to_matrix(tar_pose_hands_aa)
        tar_pose_hands = rc.matrix_to_rotation_6d(m_hands).reshape(B, N, 30 * 6)

        pose_upper_aa = tar_pose[:, :, self.joint_mask_upper.astype(bool)].reshape(B, N, 13, 3)
        m_upper = rc.axis_angle_to_matrix(pose_upper_aa)
        tar_pose_upper = rc.matrix_to_rotation_6d(m_upper).reshape(B, N, 13 * 6)

        pose_leg_aa = tar_pose[:, :, self.joint_mask_lower.astype(bool)].reshape(B, N, 9, 3)
        m_leg = rc.axis_angle_to_matrix(pose_leg_aa)
        tar_pose_leg = rc.matrix_to_rotation_6d(m_leg).reshape(B, N, 9 * 6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)

        tar_index_value_face_top = self.vq_model_face.map2index(tar_pose_face)
        tar_index_value_upper_top = self.vq_model_upper.map2index(tar_pose_upper)
        tar_index_value_hands_top = self.vq_model_hands.map2index(tar_pose_hands)
        tar_index_value_lower_top = self.vq_model_lower.map2index(tar_pose_lower)

        zq_face = self.vq_model_face.map2zq(tar_pose_face)
        zq_upper = self.vq_model_upper.map2zq(tar_pose_upper)
        zq_hands = self.vq_model_hands.map2zq(tar_pose_hands)
        zq_lower = self.vq_model_lower.map2zq(tar_pose_lower)

        m_all = rc.axis_angle_to_matrix(tar_pose.reshape(B, N, j, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(m_all).reshape(B, N, j * 6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)

        out = {
            "beat": to_np_squeezed(in_beat),
            "hubert": to_np_squeezed(in_hubert),
            "zq_face": to_np_squeezed(zq_face),
            "zq_upper": to_np_squeezed(zq_upper),
            "zq_hands": to_np_squeezed(zq_hands),
            "zq_lower": to_np_squeezed(zq_lower),
            "in_audio": to_np_squeezed(in_audio),
            "in_word": to_np_squeezed(in_word),
            "tar_trans": to_np_squeezed(tar_trans),
            "tar_exps": to_np_squeezed(tar_exps),
            "tar_beta": to_np_squeezed(tar_beta),
            "tar_pose": to_np_squeezed(tar_pose),
            "tar_index_value_face_top": to_np_squeezed(tar_index_value_face_top),
            "tar_index_value_upper_top": to_np_squeezed(tar_index_value_upper_top),
            "tar_index_value_hands_top": to_np_squeezed(tar_index_value_hands_top),
            "tar_index_value_lower_top": to_np_squeezed(tar_index_value_lower_top),
            "tar_id": to_np_squeezed(tar_id),
            "latent_all": to_np_squeezed(latent_all),
            "tar_contact": to_np_squeezed(tar_contact),
            "feat_clip_text": to_np_squeezed(feat_clip_text),
            "emo_clip_text": to_np_squeezed(emo_clip_text),
        }
        if sem_mean is not None:
            out["sem_mean"] = to_np_squeezed(sem_mean)
        return out
