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

        # ========== TS-PTQ: Newton Pre-shift ==========
        if g_global is not None and alpha_vec is not None:
            g_global = g_global.to(self.dev)
            alpha_vec = alpha_vec.to(self.dev)

            if actorder:
                g_global = g_global[:, perm]

            # --- Distribution dumps before Newton shift ---
            # |g_global|
            gg = g_global.abs()
            print(
                'TS-PTQ |g_global|   '
                f'mean={gg.mean().item():.6e}  std={gg.std().item():.6e}  '
                f'min={gg.min().item():.6e}  max={gg.max().item():.6e}  '
                f'median={gg.median().item():.6e}  shape={list(gg.shape)}'
            )
            del gg

            # |W| (before shift)
            Wabs = W.abs()
            print(
                'TS-PTQ |W|          '
                f'mean={Wabs.mean().item():.6e}  std={Wabs.std().item():.6e}  '
                f'min={Wabs.min().item():.6e}  max={Wabs.max().item():.6e}  '
                f'median={Wabs.median().item():.6e}  shape={list(Wabs.shape)}'
            )
            del Wabs

            # |diag(H^{-1})|
            diag_Hinv = H_inv_true.diag().abs()
            print(
                'TS-PTQ |diag(H^-1)| '
                f'mean={diag_Hinv.mean().item():.6e}  std={diag_Hinv.std().item():.6e}  '
                f'min={diag_Hinv.min().item():.6e}  max={diag_Hinv.max().item():.6e}  '
                f'median={diag_Hinv.median().item():.6e}  len={diag_Hinv.numel()}'
            )
            del diag_Hinv

            # 1. Matrix multiplication using the TRUE inverse
            newton_direction = torch.matmul(g_global, H_inv_true)  # shape: [out_features, in_features]

            # |newton_direction|
            nd = newton_direction.abs()
            print(
                'TS-PTQ |newton_dir| '
                f'mean={nd.mean().item():.6e}  std={nd.std().item():.6e}  '
                f'min={nd.min().item():.6e}  max={nd.max().item():.6e}  '
                f'median={nd.median().item():.6e}  shape={list(nd.shape)}'
            )
            del nd

            # alpha_vec
            print(
                'TS-PTQ alpha_vec    '
                f'mean={alpha_vec.mean().item():.6e}  std={alpha_vec.std().item():.6e}  '
                f'min={alpha_vec.min().item():.6e}  max={alpha_vec.max().item():.6e}  '
                f'median={alpha_vec.median().item():.6e}  shape={list(alpha_vec.shape)}'
            )

            # 2. Channel-wise scaling to get the final offset v
            scale = - 2.0  # manually adjustable scale factor
            v = scale * alpha_vec * newton_direction  # shape: [out_features, in_features]

            # |v| (shift magnitude)
            vabs = v.abs()
            print(
                'TS-PTQ |v|          '
                f'mean={vabs.mean().item():.6e}  std={vabs.std().item():.6e}  '
                f'min={vabs.min().item():.6e}  max={vabs.max().item():.6e}  '
                f'median={vabs.median().item():.6e}  shape={list(vabs.shape)}'
            )
            del vabs

            # Create target weights
            W = W + v  # W is now W_target

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
            del g_global, alpha_vec, H_inv_true, newton_direction, v
        else:
            del H_inv_true
        # ===========================================================

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
