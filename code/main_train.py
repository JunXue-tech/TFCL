import argparse
import os
import warnings
from collections import OrderedDict
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from core_scripts.startup_config import set_random_seed
from evaluation2019 import calculate_cm_eer
from data_utils_SSL import (
    genSpoof_list,
    Dataset_ASVspoof2019_train_pair,
    Dataset_ASVspoof2019_devMixed,
    get_clean_utt_id_from_processed,
)
from model import Model

warnings.filterwarnings("ignore", category=FutureWarning)

__author__ = "Jun Xue"
__email__ = "junxue@whu.edu.cn"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# Parameters used only by paired TFCL consistency training. They are not
# traversed by Model.forward(x) during single-branch inference.
AUXILIARY_PREFIXES = (
    "bidirectional_attn_T.",
    "bidirectional_attn_D.",  # legacy TFCL implementation
    "feature_proj_d.",
)


def safe_remove(path: str):
    if path is not None and os.path.isfile(path):
        os.remove(path)


def _torch_load_compat(path: str, map_location="cpu"):
    """Load checkpoints across old and new PyTorch versions."""
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def unwrap_state_dict(checkpoint):
    """
    Extract a state_dict from common checkpoint layouts.

    Supported layouts:
      1) raw state_dict
      2) {"state_dict": ...}
      3) {"model": ...}
      4) {"model_state_dict": ...}
      5) {"det_model": ...}
    """
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Checkpoint must be a dict, got {type(checkpoint).__name__}"
        )

    for key in ("det_model", "state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def normalize_state_dict_keys(state_dict):
    """
    Remove leading wrappers introduced by DDP and torch.compile.

    Internal names such as ``ssl_model.model.*`` remain unchanged.
    """
    normalized = OrderedDict()

    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue

        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                    break

        if new_key in normalized:
            raise RuntimeError(
                f"Duplicate parameter after key normalization: {new_key}"
            )

        normalized[new_key] = value

    if not normalized:
        raise RuntimeError("No tensor parameters were found in the checkpoint.")

    return normalized


def is_auxiliary_key(key: str) -> bool:
    return key.startswith(AUXILIARY_PREFIXES)


def load_model_checkpoint(model, ckpt_path, device):
    """
    Load either a complete training checkpoint or an inference-only checkpoint.

    Missing TFCL auxiliary parameters are allowed because inference-only
    checkpoints intentionally omit them. Every SSL+AASIST inference parameter
    must still exist and have the correct shape.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw_checkpoint = _torch_load_compat(ckpt_path, map_location="cpu")
    checkpoint_state = normalize_state_dict_keys(
        unwrap_state_dict(raw_checkpoint)
    )
    model_state = model.state_dict()

    required_inference_keys = {
        key
        for key in model_state.keys()
        if not is_auxiliary_key(key)
    }

    loadable_state = OrderedDict()
    ignored_auxiliary = []
    unexpected_inference = []
    inference_shape_mismatches = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            if is_auxiliary_key(key):
                ignored_auxiliary.append(key)
            else:
                unexpected_inference.append(key)
            continue

        expected_shape = tuple(model_state[key].shape)
        checkpoint_shape = tuple(value.shape)

        if checkpoint_shape != expected_shape:
            if is_auxiliary_key(key):
                ignored_auxiliary.append(key)
            else:
                inference_shape_mismatches.append(
                    (key, checkpoint_shape, expected_shape)
                )
            continue

        loadable_state[key] = value

    missing_inference = sorted(
        required_inference_keys - set(loadable_state.keys())
    )
    missing_auxiliary = sorted(
        key
        for key in model_state.keys()
        if is_auxiliary_key(key) and key not in loadable_state
    )

    print(f"[Checkpoint] path                    : {ckpt_path}")
    print(f"[Checkpoint] checkpoint tensor keys  : {len(checkpoint_state)}")
    print(f"[Checkpoint] compatible keys loaded  : {len(loadable_state)}")
    print(f"[Checkpoint] missing auxiliary keys  : {len(missing_auxiliary)}")
    if missing_auxiliary:
        print(
            f"[Checkpoint] auxiliary init example : "
            f"{missing_auxiliary[:20]}"
        )
    print(f"[Checkpoint] ignored auxiliary keys  : {len(ignored_auxiliary)}")
    if ignored_auxiliary:
        print(
            f"[Checkpoint] ignored auxiliary example: "
            f"{ignored_auxiliary[:20]}"
        )
    print(f"[Checkpoint] unknown inference keys  : {len(unexpected_inference)}")
    if unexpected_inference:
        print(
            f"[Checkpoint] unknown example         : "
            f"{unexpected_inference[:20]}"
        )
    print(
        f"[Checkpoint] inference shape mismatch: "
        f"{len(inference_shape_mismatches)}"
    )
    if inference_shape_mismatches:
        print(
            f"[Checkpoint] mismatch example        : "
            f"{inference_shape_mismatches[:10]}"
        )
    print(f"[Checkpoint] missing inference keys  : {len(missing_inference)}")
    if missing_inference:
        print(
            f"[Checkpoint] missing example         : "
            f"{missing_inference[:20]}"
        )

    errors = []
    if unexpected_inference:
        errors.append(
            f"{len(unexpected_inference)} unknown non-auxiliary keys"
        )
    if inference_shape_mismatches:
        errors.append(
            f"{len(inference_shape_mismatches)} inference shape mismatches"
        )
    if missing_inference:
        errors.append(
            f"{len(missing_inference)} required inference keys are missing"
        )

    if errors:
        raise RuntimeError(
            "Checkpoint validation failed: " + "; ".join(errors)
        )

    incompatible = model.load_state_dict(loadable_state, strict=False)

    residual_missing = [
        key
        for key in incompatible.missing_keys
        if not is_auxiliary_key(key)
    ]
    residual_unexpected = list(incompatible.unexpected_keys)

    if residual_missing or residual_unexpected:
        raise RuntimeError(
            "Checkpoint loading still has non-auxiliary incompatibilities: "
            f"missing={residual_missing[:20]}, "
            f"unexpected={residual_unexpected[:20]}"
        )

    model.to(device)
    print("[Checkpoint] Model checkpoint loaded successfully.")


def build_inference_state_dict(model):
    """
    Build the exact checkpoint subset consumed by eval.py:

        waveform -> SSL encoder -> AASIST backend -> classifier

    """
    full_state = normalize_state_dict_keys(model.state_dict())

    inference_state = OrderedDict()
    removed_auxiliary = []

    for key, value in full_state.items():
        if is_auxiliary_key(key):
            removed_auxiliary.append(key)
            continue

        inference_state[key] = value.detach().cpu()

    if not inference_state:
        raise RuntimeError("The generated inference state_dict is empty.")

    remaining_auxiliary = [
        key for key in inference_state if is_auxiliary_key(key)
    ]
    if remaining_auxiliary:
        raise RuntimeError(
            "Auxiliary TFCL parameters remain in the inference checkpoint: "
            f"{remaining_auxiliary[:20]}"
        )

    expected_inference_keys = {
        key
        for key in full_state.keys()
        if not is_auxiliary_key(key)
    }
    saved_keys = set(inference_state.keys())

    if saved_keys != expected_inference_keys:
        missing = sorted(expected_inference_keys - saved_keys)
        extra = sorted(saved_keys - expected_inference_keys)
        raise RuntimeError(
            "Inference state_dict validation failed: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    return inference_state, removed_auxiliary


def save_inference_checkpoint(
    model,
    save_path: str,
    epoch: int,
    dev_eer: float,
    dev_loss: float,
):
    """
    Save only SSL+AASIST inference weights in a format directly accepted by
    the unchanged eval.py loader.
    """
    inference_state, removed_auxiliary = build_inference_state_dict(model)

    payload = {
        "state_dict": inference_state,
        "checkpoint_type": "tfcl_ssl_aasist_inference",
        "format_version": 1,
        "epoch": int(epoch),
        "dev_eer": float(dev_eer),
        "dev_loss": float(dev_loss),
        "removed_auxiliary_prefixes": list(AUXILIARY_PREFIXES),
        "num_inference_keys": len(inference_state),
    }

    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)

    temporary_path = save_path + ".tmp"
    safe_remove(temporary_path)

    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, save_path)
    finally:
        safe_remove(temporary_path)

    # Re-open the saved file and verify that no TFCL auxiliary tensors remain.
    saved_checkpoint = _torch_load_compat(save_path, map_location="cpu")
    saved_state = normalize_state_dict_keys(
        unwrap_state_dict(saved_checkpoint)
    )
    remaining_auxiliary = [
        key for key in saved_state if is_auxiliary_key(key)
    ]

    if remaining_auxiliary:
        safe_remove(save_path)
        raise RuntimeError(
            "Saved checkpoint still contains TFCL auxiliary parameters: "
            f"{remaining_auxiliary[:20]}"
        )

    if set(saved_state.keys()) != set(inference_state.keys()):
        safe_remove(save_path)
        raise RuntimeError(
            "Saved checkpoint key set differs from the intended "
            "inference key set."
        )

    print(f"[SAVE] inference keys saved   : {len(saved_state)}")
    print(f"[SAVE] auxiliary keys removed : {len(removed_auxiliary)}")
    if removed_auxiliary:
        print(f"[SAVE] removed example        : {removed_auxiliary[:20]}")


def build_mixed_dev_protocol(
    src_protocol_path: str,
    dst_protocol_path: str,
    proc_suffix: str,
):
    mixed_lines = []

    with open(src_protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            spk, proc_utt_id, third_col, src, label = parts[:5]
            clean_utt_id = get_clean_utt_id_from_processed(
                proc_utt_id,
                proc_suffix,
            )

            mixed_lines.append(
                f"{spk} {proc_utt_id} {third_col} {src} {label}\n"
            )
            mixed_lines.append(
                f"{spk} {clean_utt_id} {third_col} {src} {label}\n"
            )

    with open(dst_protocol_path, "w", encoding="utf-8") as f:
        f.writelines(mixed_lines)

    print(f"[DevMix] Mixed protocol saved to: {dst_protocol_path}")
    print(f"[DevMix] Number of mixed trials: {len(mixed_lines)}")


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
) -> float:
    model.eval()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    det_criterion = nn.CrossEntropyLoss(weight=weight)

    with open(trial_path, "r", encoding="utf-8") as f_trl:
        trial_lines = f_trl.readlines()

    fname_list = []
    score_list = []
    running_det = 0.0
    num_total = 0

    loop = tqdm(data_loader, desc="Evaluating", unit="batch")

    for batch in loop:
        batch_x, y_det, utt_id = batch

        bs = batch_x.size(0)
        batch_x = batch_x.to(device)
        y_det = y_det.view(-1).long().to(device)

        with torch.no_grad():
            det_logits = model(batch_x)
            loss_det = det_criterion(det_logits, y_det)

            running_det += float(loss_det.item()) * bs
            num_total += bs
            batch_score = (
                det_logits[:, 1]
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )

        fname_list.extend(list(utt_id))
        score_list.extend(batch_score.tolist())
        loop.set_postfix(
            current_batch=bs,
            total_scores=len(score_list),
        )

    assert len(trial_lines) == len(fname_list) == len(score_list), (
        f"trial={len(trial_lines)} "
        f"fname={len(fname_list)} "
        f"score={len(score_list)}"
    )

    avg_loss_det = running_det / max(num_total, 1)

    with open(save_path, "w", encoding="utf-8") as fh:
        for fn, sco, trl in zip(
            fname_list,
            score_list,
            trial_lines,
        ):
            parts = trl.strip().split()
            if len(parts) < 5:
                raise ValueError(
                    f"Bad protocol line: {trl.strip()}"
                )

            _, utt_id2, _, src, key = parts[:5]
            assert fn == utt_id2, (
                f"utt mismatch: {fn} vs {utt_id2}"
            )
            fh.write(
                f"{utt_id2} {src} {key} {sco}\n"
            )

    print(f"Scores saved to {save_path}")
    return avg_loss_det


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    device,
    align_weight,
):
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    det_criterion = nn.CrossEntropyLoss(weight=weight)

    running_loss = 0.0
    running_proc_ce = 0.0
    running_clean_ce = 0.0
    running_align_loss = 0.0
    correct_proc = 0
    correct_clean = 0
    total = 0

    for batch in tqdm(
        train_loader,
        desc="Training",
        unit="batch",
    ):
        batch_x_clean, batch_x_proc, y_det, _ = batch

        batch_x_clean = batch_x_clean.to(device)
        batch_x_proc = batch_x_proc.to(device)
        y_det = y_det.view(-1).long().to(device)

        optimizer.zero_grad(set_to_none=True)

        proc_logits, clean_logits, align_loss = model(
            batch_x_proc,
            x_clean=batch_x_clean,
            return_pair_loss=True,
        )

        proc_ce = det_criterion(proc_logits, y_det)
        clean_ce = det_criterion(clean_logits, y_det)
        loss = proc_ce + clean_ce + align_weight * align_loss

        loss.backward()
        optimizer.step()

        bs = y_det.size(0)
        running_loss += float(loss.item()) * bs
        running_proc_ce += float(proc_ce.item()) * bs
        running_clean_ce += float(clean_ce.item()) * bs
        running_align_loss += float(align_loss.item()) * bs

        with torch.no_grad():
            pred_proc = torch.argmax(proc_logits, dim=-1)
            pred_clean = torch.argmax(clean_logits, dim=-1)

            correct_proc += int(
                (pred_proc == y_det).sum().item()
            )
            correct_clean += int(
                (pred_clean == y_det).sum().item()
            )
            total += bs

    avg_loss = running_loss / max(total, 1)
    avg_proc_ce = running_proc_ce / max(total, 1)
    avg_clean_ce = running_clean_ce / max(total, 1)
    avg_align_loss = running_align_loss / max(total, 1)
    acc_proc = 100.0 * correct_proc / max(total, 1)
    acc_clean = 100.0 * correct_clean / max(total, 1)

    return (
        avg_loss,
        avg_proc_ce,
        avg_clean_ce,
        avg_align_loss,
        acc_proc,
        acc_clean,
        total,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Paired anti-spoofing training with "
            "time-frequency consistency learning"
        )
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="processed train wav dir",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        required=True,
        help="processed dev wav dir",
    )
    parser.add_argument(
        "--protocols_path",
        type=str,
        required=True,
        help="processed protocol root",
    )

    parser.add_argument(
        "--clean_train_data_path",
        type=str,
        required=True,
        help="clean train wav dir",
    )
    parser.add_argument(
        "--clean_dev_data_path",
        type=str,
        required=True,
        help="clean dev wav dir",
    )
    parser.add_argument(
        "--clean_protocols_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--proc_suffix",
        type=str,
        default="_echoaec_noisyns_agc_vad",
    )

    parser.add_argument(
        "--track",
        type=str,
        default="ASVspoof2019",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--train_tag",
        type=str,
        default="TFCL",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="your path",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr_det",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--align_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--cudnn-deterministic-toggle",
        dest="cudnn_deterministic_toggle",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cudnn-benchmark-toggle",
        dest="cudnn_benchmark_toggle",
        action="store_true",
        default=True,
    )

    parser.add_argument("--algo", type=int, default=5)
    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument(
        "--minBiasLinNonLin",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--maxBiasLinNonLin",
        type=int,
        default=20,
    )
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)

    args = parser.parse_args()

    set_random_seed(args.seed, args)

    if args.cudnn_deterministic_toggle:
        torch.backends.cudnn.deterministic = True
    if args.cudnn_benchmark_toggle:
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    train_dir = args.train_data_path
    dev_dir = args.dev_data_path
    clean_train_dir = args.clean_train_data_path
    clean_dev_dir = args.clean_dev_data_path

    train_protocol = os.path.join(
        args.protocols_path,
        f"{args.track}_train.txt",
    )
    dev_protocol = os.path.join(
        args.protocols_path,
        f"{args.track}_dev.txt",
    )

    required_paths = [
        train_dir,
        dev_dir,
        clean_train_dir,
        clean_dev_dir,
        train_protocol,
        dev_protocol,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required input path does not exist: {path}"
            )

    exp_tag = (
        f"{args.track}_bz{args.batch_size}_{args.train_tag}"
    )
    exp_root = os.path.join(args.out_path, exp_tag)
    ckpt_path = os.path.join(exp_root, "ckpt")
    log_path = os.path.join(exp_root, "logs")
    score_path = os.path.join(log_path, "scores")
    curve_path = os.path.join(log_path, "curve")
    result_log = os.path.join(
        log_path,
        f"train_{exp_tag}.log",
    )
    dev_score_path = os.path.join(
        score_path,
        "dev_score.txt",
    )
    mixed_dev_protocol = os.path.join(
        score_path,
        "mixed_dev_protocol.txt",
    )

    for directory in (
        ckpt_path,
        score_path,
        curve_path,
    ):
        os.makedirs(directory, exist_ok=True)

    build_mixed_dev_protocol(
        dev_protocol,
        mixed_dev_protocol,
        args.proc_suffix,
    )

    writer = SummaryWriter(log_dir=curve_path)

    d_label_trn, _, file_train = genSpoof_list(
        train_protocol,
        is_train=True,
        is_eval=False,
    )
    d_label_dev, _, file_dev = genSpoof_list(
        dev_protocol,
        is_train=False,
        is_eval=False,
    )

    print(
        f"[Data] train processed samples: "
        f"{len(file_train)}"
    )
    print(
        f"[Data] dev processed samples  : "
        f"{len(file_dev)}"
    )
    print(
        f"[Data] dev mixed samples      : "
        f"{len(file_dev) * 2}"
    )

    train_set = Dataset_ASVspoof2019_train_pair(
        args=args,
        list_IDs=file_train,
        labels=d_label_trn,
        clean_base_dir=clean_train_dir,
        proc_base_dir=train_dir,
        algo=args.algo,
        proc_suffix=args.proc_suffix,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    dev_set = Dataset_ASVspoof2019_devMixed(
        list_IDs=file_dev,
        labels=d_label_dev,
        clean_base_dir=clean_dev_dir,
        proc_base_dir=dev_dir,
        proc_suffix=args.proc_suffix,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model = Model(args, device).to(device)

    if (
        args.resume_ckpt is not None
        and str(args.resume_ckpt).lower() != "none"
    ):
        load_model_checkpoint(
            model,
            args.resume_ckpt,
            device,
        )

    nb_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print("nb_params(model):", nb_params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_det,
        weight_decay=args.weight_decay,
    )

    # The only model-selection criterion is development-set EER.
    best_dev_eer = float("inf")
    best_eer_ckpt = None
    no_improve = 0

    print(
        "\n========== Start Paired Training ==========\n"
    )

    for ep in range(args.max_epochs):
        (
            tr_loss,
            tr_proc_ce,
            tr_clean_ce,
            tr_align_loss,
            tr_acc_proc,
            tr_acc_clean,
            tr_total,
        ) = train_one_epoch(
            train_loader,
            model,
            optimizer,
            device,
            args.align_weight,
        )

        dev_loss = produce_evaluation_file(
            dev_loader,
            model,
            device,
            dev_score_path,
            mixed_dev_protocol,
        )
        dev_eer = calculate_cm_eer(
            cm_scores_file=dev_score_path
        )

        eer_improved = dev_eer < best_dev_eer

        if eer_improved:
            old_best_eer_ckpt = best_eer_ckpt
            best_dev_eer = dev_eer
            best_eer_ckpt = os.path.join(
                ckpt_path,
                (
                    f"best_eer_ep{ep + 1}_"
                    f"eer{dev_eer:.4f}.pth"
                ),
            )

            save_inference_checkpoint(
                model=model,
                save_path=best_eer_ckpt,
                epoch=ep + 1,
                dev_eer=dev_eer,
                dev_loss=dev_loss,
            )

            safe_remove(old_best_eer_ckpt)
            print(
                f"[SAVE] Best-EER inference model updated: "
                f"{best_eer_ckpt}"
            )
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[Epoch {ep + 1}/{args.max_epochs}] "
            f"train_loss={tr_loss:.6f} "
            f"train_proc_ce={tr_proc_ce:.6f} "
            f"train_clean_ce={tr_clean_ce:.6f} "
            f"train_align_loss={tr_align_loss:.6f} "
            f"train_proc_acc={tr_acc_proc:.2f}% "
            f"train_clean_acc={tr_acc_clean:.2f}% "
            f"train_total={tr_total} "
            f"dev_loss={dev_loss:.6f} "
            f"dev_eer={dev_eer:.4f}% "
            f"best_dev_eer={best_dev_eer:.4f}% "
            f"no_improve={no_improve}/{args.patience}"
        )

        writer.add_scalar(
            "train/loss",
            tr_loss,
            ep + 1,
        )
        writer.add_scalar(
            "train/proc_ce",
            tr_proc_ce,
            ep + 1,
        )
        writer.add_scalar(
            "train/clean_ce",
            tr_clean_ce,
            ep + 1,
        )
        writer.add_scalar(
            "train/align_loss",
            tr_align_loss,
            ep + 1,
        )
        writer.add_scalar(
            "train/proc_acc",
            tr_acc_proc,
            ep + 1,
        )
        writer.add_scalar(
            "train/clean_acc",
            tr_acc_clean,
            ep + 1,
        )
        writer.add_scalar(
            "dev/loss",
            dev_loss,
            ep + 1,
        )
        writer.add_scalar(
            "dev/eer",
            dev_eer,
            ep + 1,
        )
        writer.add_scalar(
            "dev/best_eer",
            best_dev_eer,
            ep + 1,
        )

        with open(
            result_log,
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                f"[{datetime.now().replace(microsecond=0)}] "
                f"ep{ep + 1}: "
                f"train_loss={tr_loss:.6f}, "
                f"train_proc_ce={tr_proc_ce:.6f}, "
                f"train_clean_ce={tr_clean_ce:.6f}, "
                f"train_align_loss={tr_align_loss:.6f}, "
                f"train_proc_acc={tr_acc_proc:.2f}, "
                f"train_clean_acc={tr_acc_clean:.2f}, "
                f"dev_loss={dev_loss:.6f}, "
                f"dev_eer={dev_eer:.4f}, "
                f"best_dev_eer={best_dev_eer:.4f}, "
                f"eer_improved={int(eer_improved)}, "
                f"no_improve={no_improve}/{args.patience}\n"
            )

        if no_improve >= args.patience:
            print(
                f"[Early Stop] No dev-EER improvement for "
                f"{args.patience} epochs; stopped at epoch "
                f"{ep + 1}."
            )
            break

    if best_eer_ckpt is not None:
        load_model_checkpoint(
            model,
            best_eer_ckpt,
            device,
        )
        print(
            f"[Final] Reloaded best-EER inference checkpoint: "
            f"{best_eer_ckpt}"
        )

    writer.close()

    print("\nTraining finished.")

    if best_eer_ckpt is not None:
        print(
            f"Best-EER checkpoint: {best_eer_ckpt}"
        )
        print(
            f"Best development EER: "
            f"{best_dev_eer:.4f}%"
        )
