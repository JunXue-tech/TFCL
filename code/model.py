import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq


___author__ = "Jun Xue"
__email__ = "junxue@whu.edu.cn"


class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()

        cp_path = '/data2/yzl/04_online_data_fsd/03_SSL_Anti_spoofing/pretrained_model/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        return

    def _ensure_device_dtype(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

    def extract_feat(self, input_data):
        self._ensure_device_dtype(input_data)
        self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        if emb.dim() == 3 and emb.shape[0] != input_tmp.shape[0]:
            emb = emb.transpose(0, 1).contiguous()
        emb = torch.nan_to_num(emb)
        return emb


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = torch.nan_to_num(out)
        return out


def hsic_biased(K: Tensor, L: Tensor) -> Tensor:
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - (1.0 / n)
    return torch.trace(K @ H @ L @ H)


def linear_cka(feats_A: Tensor, feats_B: Tensor, eps: float = 1e-8) -> Tensor:
    if feats_A.size(0) < 2:
        return feats_A.new_tensor(0.0)

    feats_A = feats_A - feats_A.mean(dim=0, keepdim=True)
    feats_B = feats_B - feats_B.mean(dim=0, keepdim=True)

    K = feats_A @ feats_A.transpose(0, 1)
    L = feats_B @ feats_B.transpose(0, 1)

    K = torch.nan_to_num(K)
    L = torch.nan_to_num(L)

    hsic_kk = hsic_biased(K, K).clamp_min(eps)
    hsic_ll = hsic_biased(L, L).clamp_min(eps)
    hsic_kl = hsic_biased(K, L)

    cka_val = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + eps)
    cka_val = torch.nan_to_num(cka_val, nan=0.0, posinf=0.0, neginf=0.0)
    cka_val = torch.clamp(cka_val, min=-1.0, max=1.0)
    return cka_val


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.ssl_model = SSLModel(self.device)

        # ===== SSL T/F consistency config =====
        self.ssl_feat_dim = self.ssl_model.out_dim
        self.ssl_time_steps = getattr(args, "ssl_time_steps", 201)
        self.lambda_d = getattr(args, "lambda_d", 0.3)

        # T-domain attention: 输入 [B, T, D]
        self.bidirectional_attn_T = nn.MultiheadAttention(
            embed_dim=self.ssl_feat_dim,
            num_heads=8,
            batch_first=True,
        )

        # F-domain: 不再使用 D-domain attention，只保留 shared linear projection
        # 作用在 [B, D, T] 的最后一维 T 上
        self.feature_proj_d = nn.Linear(self.ssl_time_steps, self.ssl_time_steps)

        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def _check_ssl_shape(self, clean_ssl_feat: Tensor, proc_ssl_feat: Tensor):
        if clean_ssl_feat.ndim != 3 or proc_ssl_feat.ndim != 3:
            raise ValueError(
                f"Expected SSL features [B, T, D], "
                f"got clean={clean_ssl_feat.shape}, proc={proc_ssl_feat.shape}"
            )

        if clean_ssl_feat.shape != proc_ssl_feat.shape:
            raise ValueError(
                f"clean/proc SSL feature shape mismatch: "
                f"clean={clean_ssl_feat.shape}, proc={proc_ssl_feat.shape}"
            )

        t_clean = clean_ssl_feat.size(1)
        t_proc = proc_ssl_feat.size(1)

        if t_clean != self.ssl_time_steps or t_proc != self.ssl_time_steps:
            raise ValueError(
                f"Expected fixed SSL time length T={self.ssl_time_steps}, "
                f"but got clean T={t_clean}, proc T={t_proc}."
            )

        if clean_ssl_feat.size(2) != self.ssl_feat_dim or proc_ssl_feat.size(2) != self.ssl_feat_dim:
            raise ValueError(
                f"Expected SSL feature dim D={self.ssl_feat_dim}, "
                f"but got clean D={clean_ssl_feat.size(2)}, proc D={proc_ssl_feat.size(2)}"
            )

    def _compute_feature_cka_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y: [B, D, T]

        F-domain consistency 按代码一逻辑：
        以 feature 维 D 作为 CKA 的统计单元，而不是 batch 维 B。
        """
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

        x = x.permute(1, 0, 2).contiguous()  # [D, B, T]
        y = y.permute(1, 0, 2).contiguous()  # [D, B, T]

        n_feat = x.size(0)
        if n_feat < 2:
            return x.new_tensor(0.0)

        x_flat = x.reshape(n_feat, -1)       # [D, B*T]
        y_flat = y.reshape(n_feat, -1)       # [D, B*T]

        cka_val = linear_cka(x_flat, y_flat)
        loss = 1.0 - cka_val
        return torch.nan_to_num(loss)

    def compute_ssl_td_consistency(self, clean_ssl_feat: Tensor, proc_ssl_feat: Tensor) -> Tensor:
        """
        clean_ssl_feat / proc_ssl_feat: [B, T, D]

        T-domain:
            token = time step, 使用 bidirectional cross-attention + cosine consistency

        F-domain:
            [B, T, D] -> [B, D, T]
            不使用 attention，只做 shared linear projection + feature-level CKA
        """
        self._check_ssl_shape(clean_ssl_feat, proc_ssl_feat)

        clean_ssl_feat = torch.nan_to_num(clean_ssl_feat)
        proc_ssl_feat = torch.nan_to_num(proc_ssl_feat)

        # =========================
        # 1) T-domain consistency
        # =========================
        clean_t2p, _ = self.bidirectional_attn_T(
            query=clean_ssl_feat,
            key=proc_ssl_feat,
            value=proc_ssl_feat,
            need_weights=False,
        )
        proc_t2c, _ = self.bidirectional_attn_T(
            query=proc_ssl_feat,
            key=clean_ssl_feat,
            value=clean_ssl_feat,
            need_weights=False,
        )

        clean_t2p = F.normalize(torch.nan_to_num(clean_t2p), dim=-1)
        clean_tgt = F.normalize(clean_ssl_feat, dim=-1)

        proc_t2c = F.normalize(torch.nan_to_num(proc_t2c), dim=-1)
        proc_tgt = F.normalize(proc_ssl_feat, dim=-1)

        loss_t_c2p = 1.0 - F.cosine_similarity(clean_t2p, clean_tgt, dim=-1).mean()
        loss_t_p2c = 1.0 - F.cosine_similarity(proc_t2c, proc_tgt, dim=-1).mean()
        loss_t = 0.5 * (loss_t_c2p + loss_t_p2c)

        # =========================
        # 2) F-domain consistency
        # [B, T, D] -> [B, D, T]
        # transpose -> shared linear projection -> feature-level CKA
        # =========================
        clean_d = clean_ssl_feat.transpose(1, 2).contiguous()   # [B, D, T]
        proc_d = proc_ssl_feat.transpose(1, 2).contiguous()     # [B, D, T]

        clean_d_proj = self.feature_proj_d(clean_d)             # [B, D, T]
        proc_d_proj = self.feature_proj_d(proc_d)               # [B, D, T]

        clean_d_proj = torch.nan_to_num(clean_d_proj)
        proc_d_proj = torch.nan_to_num(proc_d_proj)

        loss_d = self._compute_feature_cka_loss(clean_d_proj, proc_d_proj)

        align_loss = loss_t + self.lambda_d * loss_d
        align_loss = torch.nan_to_num(align_loss)
        return align_loss

    def backend_forward_from_ssl(self, x_ssl_feat):
        x_ssl_feat = torch.nan_to_num(x_ssl_feat)

        x = self.LL(x_ssl_feat)
        x = torch.nan_to_num(x)

        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        x = torch.nan_to_num(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        x = torch.nan_to_num(x)

        w = self.attention(x)
        w = torch.nan_to_num(w)

        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        e_S = torch.nan_to_num(e_S)

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        e_T = torch.nan_to_num(e_T)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        last_hidden = torch.nan_to_num(last_hidden)

        output = self.out_layer(last_hidden)
        output = torch.nan_to_num(output)
        return output

    def forward(self, x, x_clean=None, return_pair_loss=False):
        proc_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))

        if return_pair_loss:
            if x_clean is None:
                raise ValueError("x_clean must be provided when return_pair_loss=True")

            clean_ssl_feat = self.ssl_model.extract_feat(x_clean.squeeze(-1))

            proc_logits = self.backend_forward_from_ssl(proc_ssl_feat)
            clean_logits = self.backend_forward_from_ssl(clean_ssl_feat)

            align_loss = self.compute_ssl_td_consistency(
                clean_ssl_feat=clean_ssl_feat,
                proc_ssl_feat=proc_ssl_feat,
            )
            return proc_logits, clean_logits, align_loss

        proc_logits = self.backend_forward_from_ssl(proc_ssl_feat)
        return proc_logits


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_d', type=float, default=0.3)
    parser.add_argument('--ssl_time_steps', type=int, default=201)
    args = parser.parse_args()

    device = 'cpu'
    model = Model(args=args, device=device).to(device)

    x = torch.rand((4, 32000, 1)).to(device)
    x_clean = torch.rand((4, 32000, 1)).to(device)

    y = model(x)
    print("single forward output shape:", y.shape)

    proc_logits, clean_logits, align_loss = model(
        x, x_clean=x_clean, return_pair_loss=True
    )
    print("pair proc logits shape :", proc_logits.shape)
    print("pair clean logits shape:", clean_logits.shape)
    print("align loss:", align_loss.item())
