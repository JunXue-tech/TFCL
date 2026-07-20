import argparse
import os
import warnings
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


def safe_remove(path: str):
    if path is not None and os.path.isfile(path):
        os.remove(path)


def load_model_checkpoint(model, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "det_model" in ckpt:
        model.load_state_dict(ckpt["det_model"], strict=False)
        print(f"[Checkpoint] Loaded det_model from dict ckpt: {ckpt_path}")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"[Checkpoint] Loaded raw state_dict from: {ckpt_path}")


def build_mixed_dev_protocol(src_protocol_path: str, dst_protocol_path: str, proc_suffix: str):
    mixed_lines = []
    with open(src_protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            spk, proc_utt_id, third_col, src, label = parts[:5]
            clean_utt_id = get_clean_utt_id_from_processed(proc_utt_id, proc_suffix)

            mixed_lines.append(f"{spk} {proc_utt_id} {third_col} {src} {label}\n")
            mixed_lines.append(f"{spk} {clean_utt_id} {third_col} {src} {label}\n")

    with open(dst_protocol_path, "w", encoding="utf-8") as f:
        f.writelines(mixed_lines)

    print(f"[DevMix] Mixed protocol saved to: {dst_protocol_path}")
    print(f"[DevMix] Number of mixed trials: {len(mixed_lines)}")


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str
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
            batch_score = det_logits[:, 1].detach().cpu().numpy().ravel()

        fname_list.extend(list(utt_id))
        score_list.extend(batch_score.tolist())
        loop.set_postfix(current_batch=bs, total_scores=len(score_list))

    assert len(trial_lines) == len(fname_list) == len(score_list), \
        f"trial={len(trial_lines)} fname={len(fname_list)} score={len(score_list)}"

    avg_loss_det = running_det / max(num_total, 1)

    with open(save_path, "w", encoding="utf-8") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            parts = trl.strip().split()
            if len(parts) < 5:
                raise ValueError(f"Bad protocol line: {trl.strip()}")

            _, utt_id2, _, src, key = parts[:5]
            assert fn == utt_id2, f"utt mismatch: {fn} vs {utt_id2}"
            fh.write(f"{utt_id2} {src} {key} {sco}\n")

    print(f"Scores saved to {save_path}")
    return avg_loss_det


def train_one_epoch(train_loader, model, optimizer, device, align_weight):
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

    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        batch_x_clean, batch_x_proc, y_det, _ = batch

        batch_x_clean = batch_x_clean.to(device)
        batch_x_proc = batch_x_proc.to(device)
        y_det = y_det.view(-1).long().to(device)

        optimizer.zero_grad(set_to_none=True)

        proc_logits, clean_logits, align_loss = model(
            batch_x_proc, x_clean=batch_x_clean, return_pair_loss=True
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
            correct_proc += int((pred_proc == y_det).sum().item())
            correct_clean += int((pred_clean == y_det).sum().item())
            total += bs

    avg_loss = running_loss / max(total, 1)
    avg_proc_ce = running_proc_ce / max(total, 1)
    avg_clean_ce = running_clean_ce / max(total, 1)
    avg_align_loss = running_align_loss / max(total, 1)
    acc_proc = 100.0 * correct_proc / max(total, 1)
    acc_clean = 100.0 * correct_clean / max(total, 1)
    return avg_loss, avg_proc_ce, avg_clean_ce, avg_align_loss, acc_proc, acc_clean, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paired anti-spoofing training with time-frequency consistency learning')

    parser.add_argument('--train_data_path', type=str, required=True, help='processed train wav dir')
    parser.add_argument('--dev_data_path', type=str, required=True, help='processed dev wav dir')
    parser.add_argument('--protocols_path', type=str, required=True, help='processed protocol root')

    parser.add_argument('--clean_train_data_path', type=str, required=True, help='clean train wav dir')
    parser.add_argument('--clean_dev_data_path', type=str, required=True, help='clean dev wav dir')
    parser.add_argument('--clean_protocols_path', type=str, default=None)
    parser.add_argument('--proc_suffix', type=str, default='_echoaec_noisyns_agc_vad')

    parser.add_argument('--track', type=str, default="ASVspoof2019")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train_tag', type=str, default='TFCL')
    parser.add_argument('--out_path', type=str, default='your path')
    parser.add_argument('--resume_ckpt', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_det', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--align_weight', type=float, default=1.0)

    parser.add_argument(
        '--cudnn-deterministic-toggle',
        dest='cudnn_deterministic_toggle',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--cudnn-benchmark-toggle',
        dest='cudnn_benchmark_toggle',
        action='store_true',
        default=True
    )

    parser.add_argument('--algo', type=int, default=5)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    args = parser.parse_args()

    set_random_seed(args.seed, args)

    if args.cudnn_deterministic_toggle:
        torch.backends.cudnn.deterministic = True
    if args.cudnn_benchmark_toggle:
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dir = args.train_data_path
    dev_dir = args.dev_data_path
    clean_train_dir = args.clean_train_data_path
    clean_dev_dir = args.clean_dev_data_path

    train_protocol = os.path.join(args.protocols_path, f"{args.track}_train.txt")
    dev_protocol = os.path.join(args.protocols_path, f"{args.track}_dev.txt")

    for path in [train_dir, dev_dir, clean_train_dir, clean_dev_dir, train_protocol, dev_protocol]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input path does not exist: {path}")

    exp_tag = f"{args.track}_bz{args.batch_size}_{args.train_tag}"
    exp_root = os.path.join(args.out_path, exp_tag)
    ckpt_path = os.path.join(exp_root, "ckpt")
    log_path = os.path.join(exp_root, "logs")
    score_path = os.path.join(log_path, "scores")
    curve_path = os.path.join(log_path, "curve")
    result_log = os.path.join(log_path, f"train_{exp_tag}.log")
    dev_score_path = os.path.join(score_path, "dev_score.txt")
    mixed_dev_protocol = os.path.join(score_path, "mixed_dev_protocol.txt")

    for d in [ckpt_path, score_path, curve_path]:
        os.makedirs(d, exist_ok=True)

    build_mixed_dev_protocol(dev_protocol, mixed_dev_protocol, args.proc_suffix)

    writer = SummaryWriter(log_dir=curve_path)

    d_label_trn, _, file_train = genSpoof_list(train_protocol, is_train=True, is_eval=False)
    d_label_dev, _, file_dev = genSpoof_list(dev_protocol, is_train=False, is_eval=False)

    print(f"[Data] train processed samples: {len(file_train)}")
    print(f"[Data] dev processed samples  : {len(file_dev)}")
    print(f"[Data] dev mixed samples      : {len(file_dev) * 2}")

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
        pin_memory=True
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
        pin_memory=True
    )

    model = Model(args, device).to(device)

    if args.resume_ckpt is not None and str(args.resume_ckpt).lower() != "none":
        load_model_checkpoint(model, args.resume_ckpt, device)

    nb_params = sum(p.numel() for p in model.parameters())
    print("nb_params(model):", nb_params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_det,
        weight_decay=args.weight_decay
    )

    best_dev_loss = float("inf")
    best_dev_eer = float("inf")
    best_loss_ckpt = None
    best_eer_ckpt = None
    no_improve = 0

    print("\n========== Start Paired Training ==========\n")

    for ep in range(args.max_epochs):
        tr_loss, tr_proc_ce, tr_clean_ce, tr_align_loss, tr_acc_proc, tr_acc_clean, tr_total = train_one_epoch(
            train_loader, model, optimizer, device, args.align_weight
        )

        dev_loss = produce_evaluation_file(
            dev_loader, model, device, dev_score_path, mixed_dev_protocol
        )
        dev_eer = calculate_cm_eer(cm_scores_file=dev_score_path)

        print(
            f"[Epoch {ep+1}/{args.max_epochs}] "
            f"train_loss={tr_loss:.6f} "
            f"train_proc_ce={tr_proc_ce:.6f} "
            f"train_clean_ce={tr_clean_ce:.6f} "
            f"train_align_loss={tr_align_loss:.6f} "
            f"train_proc_acc={tr_acc_proc:.2f}% "
            f"train_clean_acc={tr_acc_clean:.2f}% "
            f"train_total={tr_total} "
            f"dev_loss={dev_loss:.6f} "
            f"dev_eer={dev_eer:.4f}% "
            f"no_improve={no_improve}/{args.patience}"
        )

        writer.add_scalar("train/loss", tr_loss, ep + 1)
        writer.add_scalar("train/proc_ce", tr_proc_ce, ep + 1)
        writer.add_scalar("train/clean_ce", tr_clean_ce, ep + 1)
        writer.add_scalar("train/align_loss", tr_align_loss, ep + 1)
        writer.add_scalar("train/proc_acc", tr_acc_proc, ep + 1)
        writer.add_scalar("train/clean_acc", tr_acc_clean, ep + 1)
        writer.add_scalar("dev/loss", dev_loss, ep + 1)
        writer.add_scalar("dev/eer", dev_eer, ep + 1)

        with open(result_log, "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.now().replace(microsecond=0)}] "
                f"ep{ep+1}: "
                f"train_loss={tr_loss:.6f}, train_proc_ce={tr_proc_ce:.6f}, "
                f"train_clean_ce={tr_clean_ce:.6f}, train_align_loss={tr_align_loss:.6f}, "
                f"train_proc_acc={tr_acc_proc:.2f}, train_clean_acc={tr_acc_clean:.2f}, "
                f"dev_loss={dev_loss:.6f}, dev_eer={dev_eer:.4f}, "
                f"best_dev_loss={best_dev_loss:.6f}, best_dev_eer={best_dev_eer:.4f}, "
                f"no_improve={no_improve}/{args.patience}\n"
            )

        loss_improved = dev_loss < best_dev_loss
        eer_improved = dev_eer < best_dev_eer

        if loss_improved:
            old_best_loss_ckpt = best_loss_ckpt
            best_dev_loss = dev_loss
            best_loss_ckpt = os.path.join(
                ckpt_path,
                f"best_loss_ep{ep+1}_loss{dev_loss:.6f}_eer{dev_eer:.4f}.pth"
            )
            torch.save(model.state_dict(), best_loss_ckpt)
            safe_remove(old_best_loss_ckpt)
            print(f"[SAVE] Best-loss model updated: {best_loss_ckpt}")
            no_improve = 0
        else:
            no_improve += 1

        if eer_improved:
            old_best_eer_ckpt = best_eer_ckpt
            best_dev_eer = dev_eer
            best_eer_ckpt = os.path.join(
                ckpt_path,
                f"best_eer_ep{ep+1}_loss{dev_loss:.6f}_eer{dev_eer:.4f}.pth"
            )
            torch.save(model.state_dict(), best_eer_ckpt)
            safe_remove(old_best_eer_ckpt)
            print(f"[SAVE] Best-eer model updated: {best_eer_ckpt}")

        if no_improve >= args.patience:
            print(f"[Early Stop] triggered at epoch {ep+1}.")
            break

    if best_loss_ckpt is not None:
        load_model_checkpoint(model, best_loss_ckpt, device)
        print(f"[Final] Reloaded best-loss checkpoint: {best_loss_ckpt}")

    writer.close()

    print("\nTraining finished.")
    if best_loss_ckpt is not None:
        print(f"Best-loss checkpoint: {best_loss_ckpt}")
    if best_eer_ckpt is not None:
        print(f"Best-eer checkpoint : {best_eer_ckpt}")