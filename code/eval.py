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
    from sklearn.metrics import roc_auc_score
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


# Parameters used only by paired TFCL training. None of these modules is
# traversed by Model.forward(x) during single-branch inference.
AUXILIARY_PREFIXES = (
    "bidirectional_attn_T.",
    "bidirectional_attn_D.",   # legacy TFCL checkpoint
    "feature_proj_d.",         # current feature-domain projection
)


def _torch_load_compat(path: str, map_location="cpu"):
    """
    Load checkpoints across PyTorch versions.

    PyTorch >= 2.6 exposes ``weights_only`` explicitly, while older versions
    do not accept the argument.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(checkpoint):
    """
    Support common training/inference checkpoint layouts:
      1) raw state_dict
      2) {"state_dict": ...}
      3) {"model": ...}
      4) {"model_state_dict": ...}
      5) {"det_model": ...}
    """
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Checkpoint must be a dict, but got {type(checkpoint).__name__}"
        )

    for key in ("det_model", "state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def normalize_state_dict_keys(state_dict):
    """
    Remove wrappers introduced by DDP/torch.compile.

    The loop is intentional because keys may be nested as
    ``module._orig_mod.<parameter>``.
    """
    normalized = {}

    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if not torch.is_tensor(value):
            continue

        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        if new_key in normalized:
            raise RuntimeError(
                f"Duplicate parameter after prefix normalization: {new_key}"
            )
        normalized[new_key] = value

    if not normalized:
        raise RuntimeError("No tensor parameters were found in the checkpoint.")

    return normalized


def is_auxiliary_key(key: str) -> bool:
    return key.startswith(AUXILIARY_PREFIXES)


def load_model_checkpoint(
    model,
    ckpt_path: str,
    device: str = "cpu",
    allow_partial: bool = False,
):
    """
    Load only the parameters required by single-branch inference.

    This loader supports both:
      * original paired-training checkpoints containing TFCL-only modules;
      * converted inference checkpoints containing only SSL + AASIST weights.

    Auxiliary TFCL parameters are ignored deliberately. Missing or
    shape-mismatched inference parameters remain fatal by default.
    """
    del device  # checkpoint is loaded on CPU to avoid temporary GPU duplication

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw_ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    checkpoint_state = normalize_state_dict_keys(unwrap_state_dict(raw_ckpt))
    model_state = model.state_dict()

    required_inference_keys = {
        key for key in model_state.keys() if not is_auxiliary_key(key)
    }

    loadable_state = {}
    ignored_auxiliary = []
    unexpected_inference = []
    shape_mismatches = []

    for key, value in checkpoint_state.items():
        if is_auxiliary_key(key):
            ignored_auxiliary.append(key)
            continue

        if key not in model_state:
            unexpected_inference.append(key)
            continue

        expected_shape = tuple(model_state[key].shape)
        checkpoint_shape = tuple(value.shape)
        if checkpoint_shape != expected_shape:
            shape_mismatches.append(
                (key, checkpoint_shape, expected_shape)
            )
            continue

        loadable_state[key] = value

    missing_inference = sorted(required_inference_keys - set(loadable_state))

    print(f"[Loader] Checkpoint: {ckpt_path}")
    print(f"[Loader] checkpoint tensor keys : {len(checkpoint_state)}")
    print(f"[Loader] inference keys loaded  : {len(loadable_state)}")
    print(f"[Loader] auxiliary keys ignored : {len(ignored_auxiliary)}")
    if ignored_auxiliary:
        print(f"[Loader] ignored example        : {ignored_auxiliary[:20]}")
    print(f"[Loader] unknown inference keys : {len(unexpected_inference)}")
    if unexpected_inference:
        print(f"[Loader] unknown example        : {unexpected_inference[:20]}")
    print(f"[Loader] shape mismatches       : {len(shape_mismatches)}")
    if shape_mismatches:
        print(f"[Loader] mismatch example       : {shape_mismatches[:10]}")
    print(f"[Loader] missing inference keys : {len(missing_inference)}")
    if missing_inference:
        print(f"[Loader] missing example        : {missing_inference[:20]}")

    errors = []
    if unexpected_inference:
        errors.append(
            f"{len(unexpected_inference)} unknown non-auxiliary checkpoint keys"
        )
    if shape_mismatches:
        errors.append(
            f"{len(shape_mismatches)} inference parameter shape mismatches"
        )
    if missing_inference:
        errors.append(
            f"{len(missing_inference)} required inference parameters are missing"
        )

    if errors and not allow_partial:
        raise RuntimeError(
            "Inference checkpoint validation failed: "
            + "; ".join(errors)
            + ". Use the checkpoint conversion script first. "
              "Use --allow_partial_checkpoint only for debugging, not for "
              "reported evaluation."
        )

    incompatible = model.load_state_dict(loadable_state, strict=False)

    # Missing auxiliary parameters are expected because the current Model
    # class still instantiates training-only modules.
    residual_missing = [
        key for key in incompatible.missing_keys if not is_auxiliary_key(key)
    ]
    residual_unexpected = list(incompatible.unexpected_keys)

    if (residual_missing or residual_unexpected) and not allow_partial:
        raise RuntimeError(
            "Model loading still has non-auxiliary incompatibilities: "
            f"missing={residual_missing[:20]}, "
            f"unexpected={residual_unexpected[:20]}"
        )

    print("[Loader] Inference checkpoint loaded successfully.")


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

    with open(save_path, "w", encoding="utf-8") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            parts = trl.strip().split()
            if len(parts) < 5:
                raise ValueError(f"Protocol format error: {trl}")

            _, utt_id2, _, src, key = parts[:5]
            assert fn == utt_id2, f"utt mismatch: {fn} vs {utt_id2}"
            fh.write(f"{utt_id2} {src} {key} {sco}\n")

    print(f"Scores saved to {save_path}")


def append_summary(
    summary_path,
    dataset_name,
    eer,
    auc,
    model_path,
    protocol_path,
    wav_path,
):
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Protocol: {protocol_path}\n")
        f.write(f"Wav root: {wav_path}\n")
        f.write(f"EER: {eer:.4f}%\n")
        f.write(f"AUC: {auc:.4f}%\n\n")

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

    parser.add_argument(
        '--allow_partial_checkpoint',
        action='store_true',
        help=(
            'Allow missing/unknown inference parameters. Intended only for '
            'checkpoint debugging; do not use for reported evaluation.'
        )
    )

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
    load_model_checkpoint(
        model,
        args.model_path,
        device=device,
        allow_partial=args.allow_partial_checkpoint,
    )
    model.eval()

    print(f'Model loaded: {args.model_path}')

    eval_score_path = os.path.join(args.output_dir, f"{args.dataset}.txt")
    summary_path = os.path.join(args.output_dir, "summary.txt")

    produce_evaluation_file(
        eval_loader,
        model,
        device,
        eval_score_path,
        args.protocol_path
    )

    eval_eer_official = calculate_cm_eer(eval_score_path)
    eval_auc = calculate_auc_from_score_file(eval_score_path) * 100.0

    print("\n========== Evaluation Result ==========")
    print(f"Dataset              : {args.dataset}")
    print(f"EER (official)       : {eval_eer_official:.4f}%")
    print(f"AUC                  : {eval_auc:.4f}%")
    print(f"Scores               : {eval_score_path}")
    print(f"Summary              : {summary_path}")
    print("=======================================\n")

    append_summary(
        summary_path=summary_path,
        dataset_name=args.dataset,
        eer=eval_eer_official,
        auc=eval_auc,
        model_path=args.model_path,
        protocol_path=args.protocol_path,
        wav_path=args.wav_path
    )

    print(f"Results saved to: {args.output_dir}")
