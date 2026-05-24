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
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False,
        g_global=None, alpha_vec=None
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

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        # ========== TS-PTQ: Intercept the True Inverse ==========
        H_inv_true = H.clone()
        # ===========================================================
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # ========== TS-PTQ: Newton Pre-shift with Trust Region ==========
        if g_global is not None and alpha_vec is not None:
            g_global = g_global.to(self.dev)
            alpha_vec = alpha_vec.to(self.dev)

            # TS-PTQ alpha_vec statistics
            print(
                'TS-PTQ alpha_vec  '
                f'mean={alpha_vec.mean().item():.6e}  '
                f'std={alpha_vec.std().item():.6e}  '
                f'min={alpha_vec.min().item():.6e}  '
                f'max={alpha_vec.max().item():.6e}  '
                f'shape={list(alpha_vec.shape)}'
            )

            if actorder:
                g_global = g_global[:, perm]

            # 1. Compute the raw Newton direction
            newton_dir = torch.matmul(g_global, H_inv_true)  # shape: [out_features, in_features]

            # 2. Compute the proposed optimal shift
            v_proposed = -alpha_vec * newton_dir  # shape: [out_features, in_features]

            # ----------------- Trust Region Clipping -----------------
            # 3. Quantization scale from original weights (computed above at find_params)
            if groupsize == -1:
                # Per-channel: [out_features, 1] — broadcasts directly
                S = self.quantizer.scale
            else:
                # Per-group: compute each group's scale on current W,
                # expand to [out_features, in_features] for element-wise clipping
                import copy
                S_list = []
                for g_start in range(0, self.columns, groupsize):
                    g_end = min(g_start + groupsize, self.columns)
                    tmp_q = copy.deepcopy(self.quantizer)
                    tmp_q.find_params(W[:, g_start:g_end], weight=True)
                    S_list.append(tmp_q.scale.expand(-1, g_end - g_start))
                S = torch.cat(S_list, dim=1)  # [out_features, in_features]

            # 4. Sub-grid safety margin
            tau = 0.5
            safe_limit = tau * S  # [out_features, 1] or [out_features, in_features]

            # 5. Physical truncation: preserve direction, clamp magnitude
            v_clamped = torch.max(torch.min(v_proposed, safe_limit), -safe_limit)

            # Trust region diagnostics
            clip_ratio = (v_proposed.abs() > safe_limit.abs()).float().mean().item()
            print(
                f'Trust Region: tau={tau}  '
                f'clip_ratio={clip_ratio:.4f}  '
                f'|v|_max={v_proposed.abs().max().item():.6e}  '
                f'|v|_mean={v_proposed.abs().mean().item():.6e}  '
                f'safe_limit_max={safe_limit.abs().max().item():.6e}'
            )
            # --------------------------------------------------------

            # 6. Execute final target shift
            W = W + v_clamped  # W is now W_target

            # Re-compute quantization parameters based on shifted W_target
            self.quantizer.find_params(W, weight=True)

            # Re-compute static groups on W_target if applicable
            if static_groups:
                import copy
                groups = []

                # Fix: temporarily undo actorder permutation so groups are
                # computed on the correct physical channel boundaries
                W_unpermuted = W.clone()
                if actorder:
                    W_unpermuted[:, perm] = W.clone()

                for i in range(0, self.columns, groupsize):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.find_params(W_unpermuted[:, i:(i + groupsize)], weight=True)
                    groups.append(quantizer)
                del W_unpermuted

            # Memory cleanup
            del g_global, alpha_vec, H_inv_true, newton_dir, v_proposed, v_clamped
        else:
            del H_inv_true
        # ==================================================================

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
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
