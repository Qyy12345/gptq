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
    Two-stage GPTQ with Closed-form Global Scale Reconstruction & Error Compensation.

    Stage 1: Independent local initialization -- Hessian-weighted grid search
             finds optimal per-group scale factors.
    Stage 2: Core GPTQ -- standard iterative quantization with Hessian-based
             error compensation, using the initial scales from Stage 1.
    Stage 3: Closed-form global reconstruction -- freeze integer weights and
             solve for globally optimal scale factors via a batched linear
             system. Incorporates cross-layer error compensation via the 
             cross-covariance matrix M = E[X X̂^T].
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
        
        # H = E[X̂ X̂^T]: 用于构建缩放因子海森矩阵 H_s 的基石
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        # M = E[X X̂^T]: 交叉协方差矩阵，用于直接计算包含误差补偿的目标投影
        self.M = torch.zeros((self.columns, self.columns), device=self.dev)
        
        self.nsamples = 0
        self.fp_inp = []

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
            
        decay = self.nsamples / (self.nsamples + tmp)
        self.H *= decay
        self.M *= decay
        self.nsamples += tmp
        
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        
        # --- 跨层误差补偿数据收集 ---
        if len(self.fp_inp) > 0:
            # 严格对齐全精度输入的形状管道
            fp_raw = self.fp_inp[0]
            if len(fp_raw.shape) == 2:
                fp_raw = fp_raw.unsqueeze(0)
            if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
                if len(fp_raw.shape) == 3:
                    fp_raw = fp_raw.reshape((-1, fp_raw.shape[-1]))
                fp_raw = fp_raw.t()
            if isinstance(self.layer, nn.Conv2d):
                unfold = nn.Unfold(
                    self.layer.kernel_size,
                    dilation=self.layer.dilation,
                    padding=self.layer.padding,
                    stride=self.layer.stride
                )
                fp_raw = unfold(fp_raw)
                fp_raw = fp_raw.permute([1, 0, 2])
                fp_raw = fp_raw.flatten(1)
                
            inp_fp = fp_raw.float() * math.sqrt(2 / self.nsamples)
            
            # [极简代数]: 直接计算 M = X X̂^T，自动包容基础曲率与误差偏差
            self.M += inp_fp.matmul(inp.t())
            del self.fp_inp[0]
        else:
            # 防御性回退: 如果外部未传入 fp_inp (即单路前向传播)
            # 假设 X ≈ X̂，此时 M 自动退化为标准的海森矩阵 H
            self.M += inp.matmul(inp.t())
            
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False,
        static_groups=True, stage1_hessian=False
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
        assert self.quantizer.sym, \
            "GPTQ_2Stage requires symmetric quantization (sym=True)"
        assert not actorder, \
            "GPTQ_2Stage baseline does not support actorder yet"

        d_out = self.rows
        d_in = self.columns
        g = groupsize
        n_g = d_in // g
        assert d_in % g == 0, "d_in must be divisible by groupsize"

        # ------------------------------------------------------------------
        # Preserve original statistics for Stage 3.
        # Cloned here, BEFORE any in-place modifications to H.
        # ------------------------------------------------------------------
        H_orig = self.H.clone()
        W_orig = W.clone()
        M_orig = self.M.clone()

        H = self.H
        del self.H

        # Dead-neuron handling (same as original GPTQ)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        maxq = self.quantizer.maxq.to(self.dev)

        # ==================================================================
        # STAGE 1: Independent Local Initialization
        # ==================================================================
        all_scales = torch.zeros((d_out, n_g), device=self.dev)

        if static_groups:
            for gi in range(n_g):
                s = gi * g
                e = s + g
                W_grp = W[:, s:e]              # [d_out, g]

                if stage1_hessian:
                    H_blk = H[s:e, s:e]        # [g, g]
                    all_scales[:, gi] = self.quantizer.find_params_hessian_weighted(
                        W_grp, H_blk
                    )
                else:
                    self.quantizer.find_params(W_grp, weight=True)
                    all_scales[:, gi] = self.quantizer.scale.flatten()

        # Fixed zero-point for symmetric quantization
        zero_val = torch.full((d_out, 1), (maxq.item() + 1) / 2, device=self.dev)

        # Standard GPTQ Hessian inversion: damp -> Cholesky -> inverse
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

        # ==================================================================
        # STAGE 2: Core GPTQ Process
        # ==================================================================
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

                # Select the scale for this column's group
                gi = (i1 + i) // g

                if not static_groups and (i1 + i) % g == 0:
                    # Dynamic: recompute scale at group boundary
                    if stage1_hessian:
                        s = gi * g
                        H_blk = H_orig[s:s + g, s:s + g]
                        all_scales[:, gi] = self.quantizer.find_params_hessian_weighted(
                            W[:, (i1 + i):(i1 + i + g)], H_blk
                        )
                    else:
                        self.quantizer.find_params(
                            W[:, (i1 + i):(i1 + i + g)], weight=True
                        )
                        all_scales[:, gi] = self.quantizer.scale.flatten()

                scale = all_scales[:, gi:gi + 1]  # [d_out, 1]

                # 确保 quantize_int 在 quant.py 中返回纯整数网格
                w_int_col = quantize_int(
                    w.unsqueeze(1), scale, zero_val, maxq
                ).flatten()
                
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

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        # ==================================================================
        # STAGE 3: Closed-Form Global Reconstruction
        # ==================================================================

        # ------------------------------------------------------------------
        # 3a. Integer weights reshape
        # ------------------------------------------------------------------
        w_int_3d = W_int.reshape(d_out, n_g, g)                     # [d_out, n_g, g]

        # ------------------------------------------------------------------
        # 3b. Feature projection with native cross-layer error compensation
        #     Using the cross-covariance matrix M_orig = E[X X̂^T] directly!
        # ------------------------------------------------------------------
        v_compensated = W_orig.float() @ M_orig.float()             # [d_out, d_in]
        v_3d = v_compensated.reshape(d_out, n_g, g)                 # [d_out, n_g, g]

        # ------------------------------------------------------------------
        # 3c. Reshape H_orig (E[X̂ X̂^T]) into 4-D block view
        # ------------------------------------------------------------------
        H_4d = H_orig.float().reshape(n_g, g, n_g, g)               # [n_g, g, n_g, g]

        # ------------------------------------------------------------------
        # 3d. Build scale-factor Hessian H_s  [d_out, n_g, n_g]
        # ------------------------------------------------------------------
        H_s = torch.einsum('oig, igjh, ojh -> oij', w_int_3d, H_4d, w_int_3d)

        # ------------------------------------------------------------------
        # 3e. Damping for numerical stability
        # ------------------------------------------------------------------
        H_s.diagonal(dim1=-2, dim2=-1).add_(1e-6)

        # ------------------------------------------------------------------
        # 3f. Build constant vector b  [d_out, n_g]
        # ------------------------------------------------------------------
        b = torch.einsum('oig, oig -> oi', w_int_3d, v_3d)          # [d_out, n_g]

        # ------------------------------------------------------------------
        # 3g. Solve for globally optimal scales: H_s @ s_opt = b
        # ------------------------------------------------------------------
        s_opt = torch.linalg.solve(H_s, b)                          # [d_out, n_g]

        # ------------------------------------------------------------------
        # 3h. Reconstruct final quantized weights
        # ------------------------------------------------------------------
        Q = (s_opt.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)    # [d_out, d_in]

        # Recompute final loss with Stage-3 reconstructed Q (original Losses
        # reflects only Stage-2 quantization, not the global reconstruction)
        FinalLosses = (W_orig - Q) ** 2

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error (stage2)', torch.sum(Losses).item())
        print('error (stage3)', torch.sum(FinalLosses).item())

        # Save globally optimal scales for downstream pack() / INT export
        self.layer_scales = s_opt.clone()

        # Sync quantizer internal state so downstream code (pack, export)
        # sees the Stage-3 optimal scales, not a stale Stage-1 value.
        self.quantizer.scale = s_opt.unsqueeze(1).to(self.quantizer.scale.dtype)
        self.quantizer.zero = zero_val.to(self.quantizer.zero.dtype)

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.M = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()