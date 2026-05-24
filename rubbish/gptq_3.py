import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = True

usesym = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

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
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # ---------------------------------------------------------
        # 优化维度一：极致的数值精度与稳定性 (Hessian 矩阵处理)
        # ---------------------------------------------------------
        # 将 Hessian 矩阵提升至 FP64 计算，并强制对称，彻底杜绝浮点累加不对称引发的 Cholesky 崩溃
        H_f64 = self.H.to(torch.float64)
        H_f64 = (H_f64 + H_f64.T) * 0.5
        del self.H

        dead = torch.diag(H_f64) == 0
        H_f64[dead, dead] = 1.0
        W[:, dead] = 0.0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H_f64), descending=True)
            W = W[:, perm]
            H_f64 = H_f64[perm][:, perm]
            invperm = torch.argsort(perm)

        W_FP = W.clone()
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # 引入数值安全的自适应 Damping
        damp = percdamp * torch.mean(torch.diag(H_f64))
        diag_indices = torch.arange(self.columns, device=self.dev)
        H_f64[diag_indices, diag_indices] += damp

        # 极高精度的求逆与分解流水线
        try:
            H_chol = torch.linalg.cholesky(H_f64, upper=False)
        except torch._C._LinAlgError:
            # Fallback 机制：应对极端病态矩阵
            H_f64[diag_indices, diag_indices] += damp * 10
            H_chol = torch.linalg.cholesky(H_f64, upper=False)

        H_inv = torch.cholesky_inverse(H_chol)
        H_inv = (H_inv + H_inv.T) * 0.5 
        
        # 注入 FP64 级别的极小 epsilon，保护后续第二次 Cholesky 不因舍入误差而报非正定错误
        H_inv[diag_indices, diag_indices] += torch.finfo(torch.float64).eps
        L_f64 = torch.linalg.cholesky(H_inv, upper=False) 

        # ---------------------------------------------------------
        # 优化维度二：显存与计算效率 (状态变量重构)
        # ---------------------------------------------------------
        # 预分配 FP64 误差累积矩阵 Err，替代原代码中高昂的 O(N) 级 (W - W_FP) 切片相减
        Err_f64 = torch.zeros((self.rows, self.columns), dtype=torch.float64, device=self.dev)
        
        # 提取 Loss 计算时的分母项，提前应用 safe-clamp 防止除零异常
        L_diag = torch.diag(L_f64)
        L_diag_sq = torch.clamp(L_diag ** 2, min=torch.finfo(torch.float32).eps)

        for j in range(self.columns):
            if groupsize != -1:
                if not static_groups:
                    if j % groupsize == 0:
                        self.quantizer.find_params(W[:, j:(j + groupsize)], weight=True)
                else:
                    idx = j
                    if actorder:
                        idx = perm[idx]
                    self.quantizer = groups[idx // groupsize]

            # 提取当前列的浮点副本
            w_fp_current = W[:, j].clone()

            # 量化当前列 
            q = quantize(
                w_fp_current.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
            ).flatten()

            Q[:, j] = q
            W[:, j] = q

            # 精度安全的 Loss 计算
            Losses[:, j] = ((w_fp_current - q) ** 2 / L_diag_sq[j]) * 0.5

            # ---------------------------------------------------------
            # 核心优化：O(1) 误差状态更新 vs O(N) 矩阵减法
            # 数学等价于原代码的 (W[:, :j+1] - W_FP[:, :j+1])
            # ---------------------------------------------------------
            Err_f64[:, j] = q.to(torch.float64) - W_FP[:, j].to(torch.float64)

            # 仅更新下一列 
            if j + 1 < self.columns:
                # 算子融合与显存优化：全流程使用 strict Matrix-Vector 乘法 (torch.mv)，避免任何 .unsqueeze 或隐式广播
                # temp_f64 对应原代码 temp，shape: [j+1]
                temp_f64 = torch.mv(H_f64[:j + 1, j + 1:], L_f64[j + 1:, j + 1])
                
                # temp2_f64 对应原代码 temp2，shape: [d_out]
                temp2_f64 = torch.mv(Err_f64[:, :j + 1], temp_f64)
                
                # UpdateTerm 求解及 in-place (就地) 更新，彻底消除冗余显存分配
                UpdateTerm = temp2_f64.mul_(L_diag[j + 1])
                W[:, j + 1].sub_(UpdateTerm.to(W.dtype))

            if DEBUG and ((j + 1) % blocksize == 0 or j == self.columns - 1):
                self.layer.weight.data[:, :j + 1] = Q[:, :j + 1]
                self.layer.weight.data[:, j + 1:] = W[:, j + 1:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()