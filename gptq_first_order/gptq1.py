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
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, g_global=None
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        # ==============================================================
        # Phase 1: Pre-processing & Newton Pre-shift (TS-PTQ)
        # ==============================================================
        # Native Damping & full inverse computation
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        # Compute Cholesky and full inverse (needed for Newton shift)
        H_chol = torch.linalg.cholesky(H)
        H_inv_full = torch.cholesky_inverse(H_chol)
        del H_chol

        if g_global is not None:
            # --- Phase 1a: Load & Device Transfer (FP32, detached from graph) ---
            g_dev = g_global.to(self.dev).detach().float()

            # Dimensionality alignment: apply same transforms as W
            if isinstance(self.layer, nn.Conv2d):
                g_dev = g_dev.flatten(1)
            if isinstance(self.layer, transformers.Conv1D):
                g_dev = g_dev.t()

            # Align with actorder permutation
            if actorder:
                g_dev = g_dev[:, perm]

            # Zero out dead columns
            g_dev[:, dead] = 0

            # --- Phase 1b: Robust Shift Step Size ---
            alpha = 5*1e-2

            # --- Phase 1c: Newton offset via g_global @ H_inv (all FP32) ---
            v = -alpha * g_dev.matmul(H_inv_full)

            # Diagnostics: log gradient, H_inv, and offset magnitudes
            print('  [DIAG] alpha=%.4e  ||g||=%.4e  mean(|g|)=%.4e  '
                  '||H_inv||=%.4e  mean(|H_inv|)=%.4e  max(|H_inv|)=%.4e  '
                  '||v||=%.4e  mean(|v|)=%.4e  max(|v|)=%.4e  '
                  '||W||=%.4e  mean(|W|)=%.4e' % (
                alpha,
                g_dev.norm().item(), g_dev.abs().mean().item(),
                H_inv_full.norm().item(), H_inv_full.abs().mean().item(), H_inv_full.abs().max().item(),
                v.norm().item(), v.abs().mean().item(), v.abs().max().item(),
                W.norm().item(), W.abs().mean().item()))

            # --- Phase 1d: Clamp offset to prevent extreme weight destruction ---
            # Limit the per-element shift to +/- 2 std-dev of the original weights.
            # This prevents any single outlier from corrupting the entire matrix.
            max_shift = W.std().clamp(min=1e-6) * 2.0
            v = torch.clamp(v, min=-max_shift, max=max_shift)

            # --- Phase 1e: Derive new target continuous weights ---
            W_new = W + v

            # Safety: if NaN/Inf leaked through, fall back to original W
            if torch.any(torch.isnan(W_new)) or torch.any(torch.isinf(W_new)):
                print('TS-PTQ WARNING: NaN/Inf detected in W_target, reverting to original W.')
                del W_new
            else:
                W = W_new
                del W_new

            # --- Phase 1f: Log & Memory Cleanup ---
            _alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            _shift_val = max_shift.item() if isinstance(max_shift, torch.Tensor) else max_shift
            del g_dev, v, alpha, max_shift
            torch.cuda.empty_cache()
            print('TS-PTQ: Newton pre-shift applied (alpha=%.4e, max_shift=%.4e).' % (
                _alpha_val, _shift_val))
        else:
            print('TS-PTQ: No global gradient provided, skipping Newton pre-shift.')

        # Compute upper triangular Cholesky factor for native lazy update loop
        # This is mathematically: Hinv = cholesky(H_damped^{-1}, upper=True)
        Hinv = torch.linalg.cholesky(H_inv_full, upper=True)
        del H_inv_full

        # ==============================================================
        # Phase 2: Dynamic Grid Allocation
        # ==============================================================
        if g_global is not None:
            # Force recalculate quantization parameters for W_target
            self.quantizer.find_params(W, weight=True)
        else:
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        # ==============================================================
        # Phase 3: Native faster() inner loop — COMPLETELY UNCHANGED
        # W is now W_target (shifted), all operations target the new weights
        # ==============================================================
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

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
