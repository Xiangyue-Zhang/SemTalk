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
import os
from typing import Iterable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.semtalk_preprocess_common import PickleDataset, PreprocessProcessor, write_pickle


class Processor(PreprocessProcessor):
    def __init__(self, device: torch.device, args):
        super().__init__(device=device, args=args, include_sem_mean=False)


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


if __name__ == "__main__":
    main()
