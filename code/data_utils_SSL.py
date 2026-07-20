import os
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from RawBoost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
)


def label_to_int(label_str: str):
    s = str(label_str).strip().lower()

    if s in ["bonafide", "bona-fide", "genuine", "real", "human", "1"]:
        return 1
    if s in ["spoof", "fake", "0"]:
        return 0

    raise ValueError(f"Unknown label string: {label_str}")


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_label = {}
    d_gen = {}
    file_list = []

    with open(dir_meta, "r", encoding="utf-8") as f:
        l_meta = f.readlines()

    if is_eval:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, key, _, _, _ = parts[:5]
            file_list.append(key)
        return file_list

    for line in l_meta:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        _, key, _, gen, label = parts[:5]
        file_list.append(key)
        d_label[key] = label_to_int(label)
        d_gen[key] = gen

    return d_label, d_gen, file_list


def parse_protocol_labels(dir_meta):
    labels = {}
    with open(dir_meta, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, utt_id, _, _, label = parts[:5]
            labels[utt_id] = label_to_int(label)
    return labels


def pad(x, max_len=64600):
    x_len = x.shape[0]

    if x_len >= max_len:
        return x[:max_len]

    if x_len == 0:
        return np.zeros(max_len, dtype=np.float32)

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x.astype(np.float32)


def _split_suffix_name(filename: str, proc_suffix: str):
    p = Path(filename)
    name = p.stem
    ext = p.suffix
    if name.endswith(proc_suffix):
        clean_name = name[:-len(proc_suffix)]
        return clean_name + ext
    return filename


def get_clean_utt_id_from_processed(proc_utt_id: str, proc_suffix: str):
    return _split_suffix_name(proc_utt_id, proc_suffix)


def get_processed_utt_id_from_clean(clean_utt_id: str, proc_suffix: str):
    p = Path(clean_utt_id)
    return f"{p.stem}{proc_suffix}{p.suffix}"


class Dataset_ASVspoof2019_train_pair(Dataset):
    """
    train:
    protocol 中的 utt_id 使用 processed 名称
    例如:
        LA_T_1000137_echoaec_noisyns_agc_vad.flac

    clean 对应文件通过去掉 proc_suffix 得到:
        LA_T_1000137.flac

    return:
        x_clean, x_proc, y_det, proc_utt_id
    """
    def __init__(self, args, list_IDs, labels, clean_base_dir, proc_base_dir, algo, proc_suffix):
        self.list_IDs = list_IDs
        self.labels = labels
        self.clean_base_dir = Path(clean_base_dir)
        self.proc_base_dir = Path(proc_base_dir)
        self.algo = algo
        self.args = args
        self.cut = 64600
        self.proc_suffix = proc_suffix

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        proc_utt_id = self.list_IDs[index]
        clean_utt_id = get_clean_utt_id_from_processed(proc_utt_id, self.proc_suffix)

        clean_wav_path = str(self.clean_base_dir / clean_utt_id)
        proc_wav_path = str(self.proc_base_dir / proc_utt_id)

        if not os.path.isfile(clean_wav_path):
            raise FileNotFoundError(
                f"Clean wav not found for processed utt_id={proc_utt_id}: {clean_wav_path}"
            )
        if not os.path.isfile(proc_wav_path):
            raise FileNotFoundError(f"Processed wav not found: {proc_wav_path}")

        x_clean, fs = librosa.load(clean_wav_path, sr=16000)
        x_proc, _ = librosa.load(proc_wav_path, sr=16000)

        x_clean = process_Rawboost_feature(x_clean, fs, self.args, self.algo)

        x_clean_pad = pad(x_clean, self.cut)
        x_proc_pad = pad(x_proc, self.cut)

        x_clean_inp = Tensor(x_clean_pad)
        x_proc_inp = Tensor(x_proc_pad)
        y_det = self.labels[proc_utt_id]
        return x_clean_inp, x_proc_inp, y_det, proc_utt_id


class Dataset_ASVspoof2019_devMixed(Dataset):
    """
    dev:
    使用 processed + clean 的混合集合，顺序固定为:
        [proc_1, clean_1, proc_2, clean_2, ...]

    输入 list_IDs 为 processed protocol 中的 utt_id 列表。
    labels 也使用 processed utt_id -> label 的映射。

    return:
        x_inp, y_det, mixed_utt_id
    """
    def __init__(self, list_IDs, labels, clean_base_dir, proc_base_dir, proc_suffix):
        self.labels = labels
        self.clean_base_dir = Path(clean_base_dir)
        self.proc_base_dir = Path(proc_base_dir)
        self.proc_suffix = proc_suffix
        self.cut = 64600
        self.items = []

        for proc_utt_id in list_IDs:
            clean_utt_id = get_clean_utt_id_from_processed(proc_utt_id, self.proc_suffix)
            self.items.append(("proc", proc_utt_id, proc_utt_id))
            self.items.append(("clean", clean_utt_id, proc_utt_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        view_type, wav_utt_id, proc_ref_id = self.items[index]

        if view_type == "proc":
            wav_path = str(self.proc_base_dir / wav_utt_id)
            out_utt_id = wav_utt_id
        else:
            wav_path = str(self.clean_base_dir / wav_utt_id)
            out_utt_id = wav_utt_id

        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"{view_type} wav not found: {wav_path}")

        x, _ = librosa.load(wav_path, sr=16000)
        x_pad = pad(x, self.cut)
        x_inp = Tensor(x_pad)

        y_det = self.labels[proc_ref_id]
        return x_inp, y_det, out_utt_id


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = Path(base_dir)
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = str(self.base_dir / utt_id)

        X, _ = librosa.load(wav_path, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)

        y_det = self.labels[utt_id]
        return x_inp, y_det, utt_id


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, protocol_path, base_dir, cut=64600):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = cut
        self.labels = parse_protocol_labels(protocol_path)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = str(self.base_dir / utt_id)

        X, _ = librosa.load(wav_path, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)

        y_det = self.labels[utt_id]
        return x_inp, y_det, utt_id


def save_curve(title, save_dir, loss, type_name):
    if torch.is_tensor(loss):
        loss = [x.item() for x in loss.detach().cpu().flatten()]
    elif isinstance(loss, list) and len(loss) > 0 and torch.is_tensor(loss[0]):
        loss = [x.item() for x in loss]

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label='Total Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel(type_name)
    plt.title(f'{type_name} ({title})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{type_name}_{title}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve to {save_path}")


def process_Rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
            args.maxF, args.minBW, args.maxBW, args.minCoeff,
            args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
            args.maxF, args.minBW, args.maxBW, args.minCoeff,
            args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
            args.maxF, args.minBW, args.maxBW, args.minCoeff,
            args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
            args.maxF, args.minBW, args.maxBW, args.minCoeff,
            args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    else:
        feature = feature

    return feature
