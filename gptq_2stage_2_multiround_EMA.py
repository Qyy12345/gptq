import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

DEBUG = True
EMA_ALPHA = 0.5       # EMA update rate: s^{(k)} = alpha * s_opt + (1-alpha) * s^{(k-1)}
NUM_EMA_ROUNDS = 1   # Number of EMA refinement rounds after initial round (total = 1 + NUM_EMA_ROUNDS)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    """
    Two-stage GPTQ with EMA-smoothed alternating minimization.

    Round 1: Stage 1 -> Stage 2 -> Stage 3 (s_opt) -> write Q=s_opt*w_int
             -> EMA: s_ema = alpha*s_opt + (1-alpha)*s_prev
    Round 2..N: Stage 2 (static s_ema) -> Stage 3 (s_opt) -> write Q=s_opt*w_int
             -> EMA: s_ema = alpha*s_opt + (1-alpha)*s_ema
    Final: Last round's Q = s_opt * w_int is the output.
           EMA only smooths the scale fed into the next round's Stage 2.
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
        static_groups=False, stage1_hessian=False
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
        # Preserve original Hessian and weights for Stage 3.
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
        # STAGE 1: Independent Local Initialization
        # ==================================================================
        all_scales = torch.zeros((d_out, n_g), device=self.dev)
        all_zeros = torch.zeros((d_out, n_g), device=self.dev)

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
        # STAGE 2 (Round 1): Core GPTQ Process
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

                gi = (i1 + i) // g

                if not static_groups and (i1 + i) % g == 0:
                    if stage1_hessian:
                        s = gi * g
                        H_blk = H_orig[s:s + g, s:s + g]
                        all_scales[:, gi], all_zeros[:, gi] = self.quantizer.find_params_hessian_weighted(
                            W[:, (i1 + i):(i1 + i + g)], H_blk
                        )
                    else:
                        self.quantizer.find_params(
                            W[:, (i1 + i):(i1 + i + g)], weight=True
                        )
                        all_scales[:, gi] = self.quantizer.scale.flatten()
                        all_zeros[:, gi] = self.quantizer.zero.flatten()

                scale = all_scales[:, gi:gi + 1]

                w_int_col = quantize_int(
                    w.unsqueeze(1), scale, all_zeros[:, gi:gi + 1], maxq
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
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2).item())
                print(torch.sum(Losses).item())

        # ==================================================================
        # STAGE 3 (Round 1): Closed-Form Global Reconstruction
        # ==================================================================
        w_int_3d = W_int.reshape(d_out, n_g, g)

        v = W_orig.float() @ H_orig.float()
        v_3d = v.reshape(d_out, n_g, g)

        H_4d = H_orig.float().reshape(n_g, g, n_g, g)

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

        hinv_diag2 = torch.diag(Hinv) ** 2

        # Reconstruct with s_opt (this round's output)
        Q = (s_opt.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)
        Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        loss_r1 = torch.sum((self.layer(self.inp1).float() - self.out1.float()) ** 2).item()
        print(f'loss (round1 stage3) {loss_r1}')
        print(f'error (round1 stage3) {torch.sum(Finalerror).item()}')

        # EMA update AFTER this round's output: s_ema = alpha * s_opt + (1-alpha) * s_prev
        # s_prev for Round 1 = all_scales (from Stage 1 / dynamic Stage 2)
        s_ema = EMA_ALPHA * s_opt + (1 - EMA_ALPHA) * all_scales

        # Save W_int for flip rate computation in next round
        W_int_prev = W_int.clone()

        # ==================================================================
        # EMA Refinement Rounds
        #
        # Each round: Stage 2 (static s_ema) -> Stage 3 (solve s_opt)
        #             -> EMA: s_ema = alpha * s_opt + (1-alpha) * s_ema
        # ==================================================================
        for rnd in range(NUM_EMA_ROUNDS):
            # Reinitialize W from original weights
            W = W_orig.clone()
            W[:, dead] = 0

            # Use EMA'd scales as static input to Stage 2
            all_scales = s_ema

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)
            W_int = torch.zeros_like(W)

            # ==============================================================
            # STAGE 2: Core GPTQ with static s_ema scales
            # ==============================================================
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
                    scale = all_scales[:, gi:gi + 1]

                    w_int_col = quantize_int(
                        w.unsqueeze(1), scale, all_zeros[:, gi:gi + 1], maxq
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
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2).item())
                    print(torch.sum(Losses).item())

            # ==============================================================
            # Flip rate: L0 distance between W_int^(k) and W_int^(k-1)
            # ==============================================================
            rnd_label = rnd + 2
            flip_count = (W_int != W_int_prev).sum().item()
            total_count = W_int.numel()
            flip_rate = flip_count / total_count
            print(f'flip_rate (round{rnd_label}) {flip_count}/{total_count} = {flip_rate:.6f}')

            W_int_prev = W_int.clone()

            # ==============================================================
            # STAGE 3: Solve for s_opt given new w_int
            # ==============================================================
            w_int_3d = W_int.reshape(d_out, n_g, g)

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

            # Reconstruct with s_opt (this round's output)
            Q = (s_opt.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)
            Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2
            if isinstance(self.layer, transformers.Conv1D):
                Q = Q.t()
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype
            )
            loss_rnd = torch.sum((self.layer(self.inp1).float() - self.out1.float()) ** 2).item()
            print(f'loss (round{rnd_label} stage3) {loss_rnd}')
            print(f'error (round{rnd_label} stage3) {torch.sum(Finalerror).item()}')

            # EMA update AFTER this round's output
            s_ema = EMA_ALPHA * s_opt + (1 - EMA_ALPHA) * s_ema

        # ==================================================================
        # Final output: last round's s_opt * w_int already written to layer
        # ==================================================================
        self.layer_scales = s_opt.clone()
        self.layer_zeros = all_zeros.clone()
        print(f'[EMA] alpha={EMA_ALPHA}, rounds={1 + NUM_EMA_ROUNDS}')

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
