import argparse
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core_scripts.startup_config import set_random_seed
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2021_eval
from evaluation2019 import calculate_cm_eer
from model import Model

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

try:
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def label_to_int(label_str: str):
    """
    Convert protocol label to int.
    Positive class = bonafide = 1
    Negative class = spoof = 0
    """
    s = str(label_str).strip().lower()

    if s in ["bonafide", "bona-fide", "genuine", "real", "human", "1"]:
        return 1
    if s in ["spoof", "fake", "0"]:
        return 0

    raise ValueError(f"Unknown label string for metric computation: {label_str}")


def calculate_auc_from_score_file(score_file: str) -> float:
    """
    Read saved score file:
        utt_id src key score
    and compute AUC.
    Returns [0, 1].
    """
    labels = []
    scores = []

    with open(score_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            key = parts[2]
            score = float(parts[3])

            y = label_to_int(key)
            labels.append(y)
            scores.append(score)

    if len(labels) == 0:
        raise RuntimeError(f"No valid entries found in score file: {score_file}")

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    uniq = np.unique(labels)
    if len(uniq) < 2:
        raise RuntimeError("AUC cannot be computed because only one class is present.")

    if SKLEARN_AVAILABLE:
        return float(roc_auc_score(labels, scores))

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("AUC cannot be computed because one class is empty.")

    greater = 0.0
    ties = 0.0
    for ps in pos_scores:
        greater += np.sum(ps > neg_scores)
        ties += np.sum(ps == neg_scores)

    auc = (greater + 0.5 * ties) / (n_pos * n_neg)
    return float(auc)


def compute_f1_acc_from_arrays(labels, preds):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)

    if len(labels) == 0:
        raise RuntimeError("No labels found for F1/ACC computation.")

    if SKLEARN_AVAILABLE:
        f1 = f1_score(labels, preds, pos_label=1) * 100.0
        acc = accuracy_score(labels, preds) * 100.0
        return float(f1), float(acc)

    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    acc = np.mean(labels == preds) * 100.0

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return float(f1 * 100.0), float(acc)


def compute_eer_and_threshold(labels, scores):
    """
    labels: 0/1, bonafide=1, spoof=0
    scores: larger -> more bonafide
    Returns:
        eer_percent, eer_threshold
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    uniq = np.unique(labels)
    if len(uniq) < 2:
        raise RuntimeError("EER cannot be computed because only one class is present.")

    if SKLEARN_AVAILABLE:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        eer_threshold = thresholds[idx]
        return float(eer * 100.0), float(eer_threshold)

    thresholds = np.unique(scores)
    best_idx = 0
    best_gap = 1e18
    best_fpr = None
    best_fnr = None

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    for i, thr in enumerate(thresholds):
        preds = (scores >= thr).astype(np.int64)
        fp = np.sum((preds == 1) & neg_mask)
        fn = np.sum((preds == 0) & pos_mask)

        fpr = fp / (n_neg + 1e-12)
        fnr = fn / (n_pos + 1e-12)
        gap = abs(fnr - fpr)

        if gap < best_gap:
            best_gap = gap
            best_idx = i
            best_fpr = fpr
            best_fnr = fnr

    eer = (best_fpr + best_fnr) / 2.0
    eer_threshold = thresholds[best_idx]
    return float(eer * 100.0), float(eer_threshold)


def unwrap_state_dict(sd):
    """
    Support common checkpoint layouts from training:
    1) raw state_dict
    2) {'state_dict': ...}
    3) {'model': ...}
    4) {'model_state_dict': ...}
    5) {'det_model': ...}
    """
    if not isinstance(sd, dict):
        raise RuntimeError(f"Checkpoint is not a dict, got {type(sd)}")

    for k in ["det_model", "state_dict", "model", "model_state_dict"]:
        if k in sd and isinstance(sd[k], dict):
            return sd[k]

    return sd


def strip_module_prefix_if_needed(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd


def load_model_checkpoint(model, ckpt_path: str, device: str = "cpu"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw_ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = unwrap_state_dict(raw_ckpt)
    state_dict = strip_module_prefix_if_needed(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[Loader] Loaded checkpoint: {ckpt_path}")
    print(f"[Loader] missing keys: {len(missing)}")
    if len(missing) > 0:
        print(f"[Loader] missing example: {missing[:20]}")
    print(f"[Loader] unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print(f"[Loader] unexpected example: {unexpected[:20]}")

    if len(unexpected) > 0:
        raise RuntimeError(
            "Checkpoint contains unexpected keys. This usually means eval model code and training model code are inconsistent."
        )


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str
):
    """
    Inference path is adapted to the paired-training model.
    During eval, model(batch_x) returns single-branch logits [B, 2].

    Save format:
        utt_id src key score
    where score = logits[:, 1]
    """
    model.eval()

    with open(trial_path, "r", encoding="utf-8") as f_trl:
        trial_lines = f_trl.readlines()

    fname_list = []
    score_list = []

    loop = tqdm(data_loader, desc="Evaluating", unit="batch")
    for batch in loop:
        if len(batch) != 3:
            raise ValueError(f"Unexpected batch format in eval loader, got len={len(batch)}")

        batch_x, _, utt_id = batch
        batch_x = batch_x.to(device)

        with torch.no_grad():
            batch_out = model(batch_x)  # [B, 2]
            if batch_out.ndim != 2 or batch_out.size(-1) != 2:
                raise RuntimeError(f"Unexpected model output shape during eval: {tuple(batch_out.shape)}")
            batch_score = batch_out[:, 1].detach().cpu().numpy().ravel()

        fname_list.extend(list(utt_id))
        score_list.extend(batch_score.tolist())
        loop.set_postfix(current_batch=batch_x.size(0), total_scores=len(score_list))

    assert len(trial_lines) == len(fname_list) == len(score_list), \
        f"trial={len(trial_lines)} fname={len(fname_list)} score={len(score_list)}"

    labels = []
    with open(save_path, "w", encoding="utf-8") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            parts = trl.strip().split()
            if len(parts) < 5:
                raise ValueError(f"Protocol format error: {trl}")

            _, utt_id2, _, src, key = parts[:5]
            assert fn == utt_id2, f"utt mismatch: {fn} vs {utt_id2}"
            fh.write(f"{utt_id2} {src} {key} {sco}\n")
            labels.append(label_to_int(key))

    print(f"Scores saved to {save_path}")
    return np.asarray(labels, dtype=np.int64), np.asarray(score_list, dtype=np.float64)


def append_summary(summary_path, dataset_name, eer, auc, f1, acc, eer_threshold,
                   model_path, protocol_path, wav_path):
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Protocol: {protocol_path}\n")
        f.write(f"Wav root: {wav_path}\n")
        f.write(f"EER: {eer:.4f}%\n")
        f.write(f"AUC: {auc:.4f}%\n")
        f.write(f"F1@EER-threshold: {f1:.4f}%\n")
        f.write(f"ACC@EER-threshold: {acc:.4f}%\n")
        f.write(f"EER threshold: {eer_threshold:.8f}\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pure evaluation for paired-training anti-spoof model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth/.ckpt)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g., ASVspoof2019LA / ITW / MLAAD-EN / SpoofCeleb / FoR_original')
    parser.add_argument('--protocol_path', type=str, required=True,
                        help='Full path to protocol file')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='Root directory to evaluation wav/flac data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')

    parser.add_argument('--track', type=str, default='ASVspoof2019_vad')
    parser.add_argument('--train_tag', type=str, default='TFCL')
    parser.add_argument('--proc_suffix', type=str, default='_echoaec_noisyns_agc_vad')
    parser.add_argument('--align_weight', type=float, default=1.0)
    parser.add_argument('--layer', type=int, default=24,
                        help='Kept only for argparse compatibility if external scripts pass this arg')

    parser.add_argument('--cudnn-deterministic-toggle', action='store_false',
                        default=True,
                        help='use cudnn-deterministic? (default true)')
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true',
                        default=False,
                        help='use cudnn-benchmark? (default false)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cudnn.deterministic = args.cudnn_deterministic_toggle
    torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle

    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")
    if not os.path.isdir(args.wav_path):
        raise FileNotFoundError(f"wav_path not found: {args.wav_path}")
    if not os.path.isfile(args.protocol_path):
        raise FileNotFoundError(f"protocol_path not found: {args.protocol_path}")

    print(f"Dataset: {args.dataset}")
    print(f"Loading evaluation protocol: {args.protocol_path}")

    file_eval = genSpoof_list(dir_meta=args.protocol_path, is_train=False, is_eval=True)
    print(f'Number of evaluation trials: {len(file_eval)}')

    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_eval,
        protocol_path=args.protocol_path,
        base_dir=args.wav_path
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    model = Model(args, device).to(device)
    load_model_checkpoint(model, args.model_path, device=device)
    model.eval()
    print(f'Model loaded: {args.model_path}')

    eval_score_path = os.path.join(args.output_dir, f"{args.dataset}.txt")
    summary_path = os.path.join(args.output_dir, "summary.txt")

    labels_np, scores_np = produce_evaluation_file(
        eval_loader,
        model,
        device,
        eval_score_path,
        args.protocol_path
    )

    eval_eer_official = calculate_cm_eer(eval_score_path)
    eval_eer_from_scores, eer_threshold = compute_eer_and_threshold(labels_np, scores_np)
    preds_at_eer = (scores_np >= eer_threshold).astype(np.int64)
    eval_f1, eval_acc = compute_f1_acc_from_arrays(labels_np, preds_at_eer)
    eval_auc = calculate_auc_from_score_file(eval_score_path) * 100.0

    print("\n========== Evaluation Result ==========")
    print(f"Dataset              : {args.dataset}")
    print(f"EER (official)       : {eval_eer_official:.4f}%")
    print(f"EER (from scores)    : {eval_eer_from_scores:.4f}%")
    print(f"EER threshold        : {eer_threshold:.8f}")
    print(f"AUC                  : {eval_auc:.4f}%")
    print(f"F1 @ EER-threshold   : {eval_f1:.4f}%")
    print(f"ACC @ EER-threshold  : {eval_acc:.4f}%")
    print(f"Scores               : {eval_score_path}")
    print(f"Summary              : {summary_path}")
    print("=======================================\n")

    append_summary(
        summary_path=summary_path,
        dataset_name=args.dataset,
        eer=eval_eer_official,
        auc=eval_auc,
        f1=eval_f1,
        acc=eval_acc,
        eer_threshold=eer_threshold,
        model_path=args.model_path,
        protocol_path=args.protocol_path,
        wav_path=args.wav_path
    )

    print(f"Results saved to: {args.output_dir}")
