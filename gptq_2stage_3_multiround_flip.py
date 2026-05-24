import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    """
    Hessian-Weighted Alternating Quantization (Advanced 2-Stage GPTQ).
    
    Phase I: Bootstrap
        Stage 1: Independent local initialization (Hessian-weighted grid search).
        Stage 2: Core GPTQ (E-step) with global H^-1 error diffusion.
    
    Phase II: Alternating Optimization (Strictly Monotonic)
        Stage 3: Closed-form global reconstruction (M-step) via batched linear solve.
        Stage 4: EMA momentum update to prevent grid jump/oscillation.
        Stage 5: Group-wise parallel local flip (Lightweight E-step) strictly 
                 guaranteed to decrease local Hessian loss without parallel collisions.
    """

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False,
        static_groups=False, stage1_hessian=False, 
        max_iters=3, ema_alpha=0.2, num_local_flips=2
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        assert groupsize > 0, \
            "GPTQ_2Stage requires groupsize > 0 (group-wise quantization)"
        assert not actorder, \
            "GPTQ_2Stage baseline does not support actorder"

        d_out = self.rows
        d_in = self.columns
        g = groupsize
        n_g = d_in // g
        assert d_in % g == 0, "d_in must be divisible by groupsize"

        # ------------------------------------------------------------------
        # Preserve original Hessian and weights for Phase II.
        # ------------------------------------------------------------------
        H_orig = self.H.clone()
        W_orig = W.clone()

        H = self.H
        del self.H

        # Dead-neuron handling
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        maxq = self.quantizer.maxq.to(self.dev)

        # ==================================================================
        # PHASE I: Bootstrap (Stage 1 & 2)
        # ==================================================================
        all_scales = torch.zeros((d_out, n_g), device=self.dev)
        all_zeros = torch.zeros((d_out, n_g), device=self.dev)

        # STAGE 1: Local Grid Initialization
        if static_groups:
            for gi in range(n_g):
                s = gi * g
                e = s + g
                W_grp = W[:, s:e]

                if stage1_hessian:
                    H_blk = H[s:e, s:e]
                    all_scales[:, gi], all_zeros[:, gi] = self.quantizer.find_params_hessian_weighted(
                        W_grp, H_blk
                    )
                else:
                    self.quantizer.find_params(W_grp, weight=True)
                    all_scales[:, gi] = self.quantizer.scale.flatten()
                    all_zeros[:, gi] = self.quantizer.zero.flatten()

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        W_int = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(d_in, device=self.dev)
        H[diag_idx, diag_idx] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # STAGE 2: Core GPTQ
        for i1 in range(0, d_in, blocksize):
            i2 = min(i1 + blocksize, d_in)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W_int1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                gi = (i1 + i) // g

                if not static_groups and (i1 + i) % g == 0:
                    if stage1_hessian:
                        s = gi * g
                        H_blk = H_orig[s:s + g, s:s + g]
                        all_scales[:, gi], all_zeros[:, gi] = self.quantizer.find_params_hessian_weighted(
                            W[:, (i1 + i):(i1 + i + g)], H_blk
                        )
                    else:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + g)], weight=True)
                        all_scales[:, gi] = self.quantizer.scale.flatten()
                        all_zeros[:, gi] = self.quantizer.zero.flatten()

                scale = all_scales[:, gi:gi + 1]
                w_int_col = quantize_int(w.unsqueeze(1), scale, all_zeros[:, gi:gi + 1], maxq).flatten()
                q = scale.flatten() * w_int_col
                
                Q1[:, i] = q
                W_int1[:, i] = w_int_col
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_int[:, i1:i2] = W_int1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        w_int_3d = W_int.reshape(d_out, n_g, g)
        s_current = all_scales.clone()

        # Pre-compute fixed components for Phase II
        H_4d = H_orig.float().reshape(n_g, g, n_g, g)
        H_diag = torch.diag(H_orig.float()).view(1, n_g, g)
        v = W_orig.float() @ H_orig.float()
        v_3d = v.reshape(d_out, n_g, g)

        # ==================================================================
        # PHASE II: Alternating Optimization (EMA + Local Flip)
        # ==================================================================
        for it in range(max_iters):
            # --------------------------------------------------------------
            # STAGE 3: Closed-Form Global Reconstruction (M-Step)
            # --------------------------------------------------------------
            chunk_size = 256
            H_s = torch.empty(d_out, n_g, n_g, device=self.dev, dtype=torch.float32)
            for start in range(0, d_out, chunk_size):
                end = min(start + chunk_size, d_out)
                w_chunk = w_int_3d[start:end]
                tmp = torch.einsum('cig, igjh -> cijh', w_chunk, H_4d)
                H_s[start:end] = torch.einsum('cijh, cjh -> cij', tmp, w_chunk)
                del tmp

            H_s.diagonal(dim1=-2, dim2=-1).add_(1e-6)
            b = torch.einsum('oig, oig -> oi', w_int_3d, v_3d)
            s_opt = torch.linalg.solve(H_s, b)

            s_current = s_opt
            s_3d = s_current.unsqueeze(2) # [d_out, n_g, 1]

            # --------------------------------------------------------------
            # STAGE 5: Row-wise Local Flip (Strictly Monotonic E-Step)
            # --------------------------------------------------------------
            # 我们将翻转限制为：每个输出通道 (d_out) 每次绝对只允许翻转 1 个权重。
            for flip_iter in range(num_local_flips): 
                Q_3d = s_3d * w_int_3d
                Q_flat = Q_3d.contiguous().view(d_out, d_in)
                E = W_orig.float() - Q_flat
                G = E @ H_orig.float()
                G_3d = G.view(d_out, n_g, g)

                score_up = s_3d * G_3d - 0.5 * (s_3d ** 2) * H_diag
                score_down = -s_3d * G_3d - 0.5 * (s_3d ** 2) * H_diag

                zero_3d = all_zeros.unsqueeze(2)
                
                # 【修复2】：引入 0.1 浮点容差，防止边界被误判
                valid_up = w_int_3d < (maxq - zero_3d - 0.1)
                valid_down = w_int_3d > (-zero_3d + 0.1)

                score_up = torch.where(valid_up, score_up, torch.tensor(-float('inf'), device=self.dev))
                score_down = torch.where(valid_down, score_down, torch.tensor(-float('inf'), device=self.dev))

                # 【修复3】：将视野摊平到整行！从 d_out 的全通道视角寻找唯一最优点
                stacked_scores = torch.stack([score_down, score_up], dim=-1) # [d_out, n_g, g, 2]
                flat_scores = stacked_scores.view(d_out, n_g * g * 2)

                # 提取整行收益最大的唯一动作
                best_scores, best_idx = torch.max(flat_scores, dim=-1) # [d_out]
                flip_mask = best_scores > 1e-7

                if not flip_mask.any():
                    break

                # 精确解码动作到 (组索引, 组内元素索引, 翻转方向)
                best_dir = best_idx % 2
                best_elem = (best_idx // 2) % g
                best_group = (best_idx // 2) // g

                delta = torch.where(best_dir == 1, 1, -1)
                o_idx = torch.arange(d_out, device=self.dev)
                
                # Batched 更新
                w_int_3d[o_idx, best_group, best_elem] += torch.where(flip_mask, delta, 0)

        # ==================================================================
        # Finalization
        # ==================================================================
        Q = (s_current.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)
        
        hinv_diag2 = torch.diag(Hinv) ** 2
        Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2

        self.layer_scales = s_current.clone()
        self.layer_zeros = all_zeros.clone()

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        print('loss (Alternating final)', torch.sum((self.layer(self.inp1).float() - self.out1.float()) ** 2).item())
        print('error (Alternating final)', torch.sum(Finalerror).item())

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        torch.cuda.empty_cache()