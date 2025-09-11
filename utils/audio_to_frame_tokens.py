import re
from typing import List, Tuple, Optional
import torch
import whisperx
import librosa
import numpy as np
import os
import torch.nn.functional as F
from numpy.lib import stride_tricks
import pickle
from funasr import AutoModel

# 可根据需要调整这个常量到你的本地路径
LOCAL_WHISPER_MODEL_DIR = os.path.abspath("./Systran/faster-whisper-large-v3")

def _normalize_word(w: str, lower: bool = True) -> str:
    if w is None:
        return ""
    w = w.strip()
    if lower:
        w = w.lower()
    # 对中英文都尽量温和处理：去掉首尾常见标点，保留撇号
    w = w.strip(" \t\r\n\"“”‘’.,?!:;()[]{}")
    return w
@torch.no_grad()
def get_word_sentence(
    audio_path: str,
    args,
    window: int = 64,                 # 每 64 帧作为一个窗口
    fps: Optional[float] = 30,        # 默认 30 fps
    total_frames: Optional[int] = None,
    model_size: str = LOCAL_WHISPER_MODEL_DIR,
    language: Optional[str] = None,
    batch_size: int = 1,              # 默认 batch_size=1
    device: Optional[str] = None,
) -> Tuple[torch.LongTensor, List[List[str]]]:
    """
    返回:
      in_word: torch.LongTensor, 形状 [1, frame]
      in_sentence: List[List[str]], 长度 = (frame - 4) // (window - 4)
                   第 k 个句子覆盖帧区间 [k*(window-4), k*(window-4)+window)
    """
    with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
        lang_model = pickle.load(f)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"

    use_model_arg = model_size if (isinstance(model_size, str) and os.path.isdir(model_size)) else model_size
    model = whisperx.load_model(use_model_arg, device=device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    sr = 16000
    duration_sec = len(audio) / sr

    asr = model.transcribe(audio, batch_size=batch_size, language=language)
    lang = asr.get("language") or language or "en"
    align_model, align_meta = whisperx.load_align_model(language_code=lang, device=device)
    aligned = whisperx.align(
        asr["segments"], align_model, align_meta, audio, device, return_char_alignments=False
    )

    # 1) 拉平成词级列表
    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []) or []:
            text = _normalize_word(w.get("word") or w.get("text") or "")
            start = w.get("start")
            end = w.get("end")
            if text and (start is not None) and (end is not None) and end >= start:
                words.append({"text": text, "start": float(start), "end": float(end)})

    # 2) 计算总帧数
    if total_frames is not None:
        frame = int(total_frames)
        fps_eff = frame / max(duration_sec, 1e-8)
    else:
        fps_eff = float(fps)
        frame = int(round(duration_sec * fps_eff))

    # 3) 为每一帧选择词
    in_word_list: List[int] = []
    words_for_frames: List[str] = []
    j = 0
    n_words = len(words)

    for i in range(frame):
        t = i / fps_eff
        while j < n_words and words[j]["end"] < t:
            j += 1

        idx_token = None
        word_text = ""
        if 0 <= j < n_words:
            w = words[j]
            if w["start"] <= t <= w["end"]:
                word_text = w["text"]
                if word_text == " ":
                    idx_token = int(lang_model.PAD_token)
                else:
                    idx_token = int(lang_model.get_word_index(word_text))

        if idx_token is None:
            idx_token = int(lang_model.UNK_token)

        in_word_list.append(idx_token)
        words_for_frames.append(word_text)

    # 4) 构造 in_sentence —— 每个窗口 64 帧，滑动步长 60 帧（重叠 4 帧）
    stride = window - 4
    num_windows = (frame - 4) // stride if frame >= 4 else 0

    in_sentence: List[List[str]] = []
    for k in range(num_windows):
        a = k * stride
        b = a + window
        if b > len(words_for_frames):
            break
        tokens = []
        last = None
        for wtxt in words_for_frames[a:b]:
            if not wtxt:
                continue
            if wtxt != last:
                tokens.append(wtxt)
                last = wtxt
        sent = " ".join(tokens) if tokens else ""
        in_sentence.append([sent])

    in_word = torch.tensor(in_word_list, dtype=torch.long).unsqueeze(0)  # [1, frame]
    return in_word, in_sentence

@torch.no_grad()
def get_emo(audio_path, frame):
    # 根据文件名提取情感标签
    emo_model = AutoModel(model="iic/emotion2vec_plus_large")
    roundt = (frame - 4) // (64 - 4)
    round_l = 64 - 4
    sample_emo_pre_list = []
    for i in range(0, roundt):
        sample_emo_pre = get_max_score_label(audio_path, i * round_l, (i + 1) * round_l + 4, emo_model)
        sample_emo_pre_list.append(sample_emo_pre)
    return sample_emo_pre_list
@torch.no_grad()
def get_max_score_label(wav_file, start_frame, end_frame, model, fps=30, target_sr=16000):
    """
    提取音频片段并从模型中找到分数最高的标签，同时将采样率设置为16kHz。
    返回:
    - max_score_label_repeated: 这里直接返回单个最高分标签字符串
    """
    audio, sr = librosa.load(wav_file, sr=target_sr)

    start_time = start_frame / fps
    end_time = end_frame / fps

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    audio_segment = audio[start_sample:end_sample]

    res = model.generate(audio_segment, granularity="utterance", extract_embedding=False)

    labels = res[0]['labels']
    scores = res[0]['scores']
    # exclude_labels = ['其他/other', '<unk>']
    # filtered_labels_scores = [(label, score) for label, score in zip(labels, scores) if label not in exclude_labels]
    filtered_labels_scores = [(label, score) for label, score in zip(labels, scores)]

    if filtered_labels_scores:
        max_score_label = max(filtered_labels_scores, key=lambda x: x[1])[0]
    else:
        max_score_label = "无效标签"

    return max_score_label

@torch.no_grad()
def get_hubert(audio_file, args):
    aud_ori, sr = librosa.load(audio_file)
    audio_each_file = librosa.resample(aud_ori, orig_sr=sr, target_sr=args.audio_sr)
    from transformers import Wav2Vec2Processor, HubertModel
    print("Loading the Wav2Vec2 Processor...")
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("../facebook/hubert-large-ls960-ft")
    print("Loading the HuBERT Model...")
    hubert_model = HubertModel.from_pretrained("../facebook/hubert-large-ls960-ft")
    hubert_model.eval()
    if args.audio_rep == "onset+amplitude":
        frame_length = 1024
        shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
        strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
        rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
        amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
        audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=args.audio_sr, units='frames')
        onset_array = np.zeros(len(audio_each_file), dtype=float)
        onset_array[audio_onset_f] = 1.0
        beat_audio = extract_rhythm_pause_features(audio_file, target_sr=args.audio_sr, frame_length=1024, hop_length=512, target_fps=30)
        mel = librosa.feature.melspectrogram(y=audio_each_file, sr=args.audio_sr, n_mels=128, hop_length=int(args.audio_sr/30))
        mel = mel[..., :-1]
        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
        audio_emb = audio_emb.unsqueeze(0)

        hubert_feat = get_hubert_from_16k_speech_long(
            hubert_model,
            wav2vec2_processor,
            torch.from_numpy(aud_ori).unsqueeze(0),
        )
        hubert_feat = F.interpolate(
            hubert_feat.swapaxes(-1, -2).unsqueeze(0),
            size=audio_emb.shape[-2],
            mode='linear',
            align_corners=True
        ).swapaxes(-1, -2)
        # hubert_feat = hubert_feat.squeeze(0)
        beat_audio = torch.from_numpy(beat_audio).float()
        return beat_audio.unsqueeze(0), hubert_feat

def extract_rhythm_pause_features(audio_file, target_sr=16000, frame_length=1024, hop_length=512, target_fps=30):
    audio_each_file, sr = librosa.load(audio_file, sr=None)
    audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=target_sr)

    shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
    strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
    rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
    amplitude_envelope = amplitude_envelope[:len(audio_each_file) // hop_length]

    energy = np.array([
        np.sum(np.abs(audio_each_file[i:i+frame_length]**2))
        for i in range(0, len(audio_each_file), hop_length)
    ])
    energy = energy[:len(amplitude_envelope)]

    audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=target_sr, hop_length=hop_length, units='frames')
    onset_array = np.zeros(len(amplitude_envelope), dtype=float)
    valid_onsets = audio_onset_f[audio_onset_f < len(onset_array)]
    onset_array[valid_onsets] = 1.0

    features = np.stack([amplitude_envelope, energy, onset_array], axis=1)

    duration = len(audio_each_file) / target_sr
    num_frames = int(duration * target_fps)

    resampled_features = np.zeros((num_frames, features.shape[1]))
    for i in range(features.shape[1]):
        resampled_features[:, i] = np.interp(
            np.linspace(0, len(features) - 1, num_frames),
            np.arange(len(features)),
            features[:, i]
        )

    return resampled_features
@torch.no_grad()
def get_hubert_from_16k_speech_long(hubert_model, wav2vec2_processor, speech=None, device="cuda:0"):
    hubert_model = hubert_model.to(device)
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0)  # [1, T]
    input_values_all = input_values_all.to(device)

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
    if input_values.shape[1] >= kernel:
        hidden_states = hubert_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])

    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret
