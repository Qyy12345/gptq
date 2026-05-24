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
    Two-stage GPTQ with closed-form global scale reconstruction,
    plus an iterative refinement round.

    Round 1: Stage 1 -> Stage 2 -> Stage 3  (same as gptq_2stage)
    Round 2: Feed Stage 3's s_opt and all_zeros into Stage 2 as
             static scales (no dynamic find_params), then re-run Stage 3.
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
        # Cloned here, BEFORE any in-place modifications to H.
        # ------------------------------------------------------------------
        H_orig = self.H.clone()
        W_orig = W.clone()

        H = self.H
        del self.H

        # Dead-neuron handling (same as original GPTQ)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        maxq = self.quantizer.maxq.to(self.dev)

        # ==================================================================
        # STAGE 1: Independent Local Initialization
        #
        # For each group, find the optimal initial scale factor s_i by
        # minimizing the Hessian-weighted reconstruction error:
        #   min_{s_i}  (s_i * w_int_i - w_i) H_{i,i} (s_i * w_int_i - w_i)^T
        #
        # Uses a grid search identical in structure to the mse=True path
        # in Quantizer.find_params, but with Hessian-weighted error metric.
        # The loop is over groups only; output channels are vectorized.
        # ==================================================================
        all_scales = torch.zeros((d_out, n_g), device=self.dev)
        all_zeros = torch.zeros((d_out, n_g), device=self.dev)

        if static_groups:
            for gi in range(n_g):
                s = gi * g
                e = s + g
                W_grp = W[:, s:e]              # [d_out, g]

                if stage1_hessian:
                    H_blk = H[s:e, s:e]        # [g, g]
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
        #
        # Standard iterative quantization with Hessian-based error
        # compensation. Scales may be dynamically recomputed at group
        # boundaries (when static_groups=False).
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
                    # Dynamic: recompute scale and zero at group boundary
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

                scale = all_scales[:, gi:gi + 1]  # [d_out, 1]

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
        #
        # Freeze w_int from Stage 2, then solve for the globally optimal
        # per-group scale factors s_opt via the normal equations:
        #
        #   H_s @ s_opt = b
        #
        # where:
        #   (H_s)_{i,j} = w_int_i  H_{i,j}  w_int_j^T     (n_g x n_g)
        #   b_i          = w_int_i  (W_orig H)_i^T           (n_g x 1)
        #
        # All tensor ops are batched over d_out output channels.
        # NO Python loops over channels or groups.
        # ==================================================================

        # ------------------------------------------------------------------
        # 3a. Integer weights captured directly during Stage 2 via
        #     quantize_int(). Reshape to group view for batched ops.
        # ------------------------------------------------------------------
        w_int_3d = W_int.reshape(d_out, n_g, g)                   # [d_out, n_g, g]

        # ------------------------------------------------------------------
        # 3b. Feature projection: v = W_orig @ H_orig
        # ------------------------------------------------------------------
        v = W_orig.float() @ H_orig.float()                         # [d_out, d_in]
        v_3d = v.reshape(d_out, n_g, g)                             # [d_out, n_g, g]

        # ------------------------------------------------------------------
        # 3c. Reshape H_orig into 4-D block view
        # ------------------------------------------------------------------
        H_4d = H_orig.float().reshape(n_g, g, n_g, g)              # [n_g, g, n_g, g]

        # ------------------------------------------------------------------
        # 3d. Build scale-factor Hessian H_s  [d_out, n_g, n_g]
        # ------------------------------------------------------------------
        chunk_size = 256
        H_s = torch.empty(d_out, n_g, n_g, device=self.dev, dtype=torch.float32)
        for start in range(0, d_out, chunk_size):
            end = min(start + chunk_size, d_out)
            w_chunk = w_int_3d[start:end]
            tmp = torch.einsum('cig, igjh -> cijh', w_chunk, H_4d)
            H_s[start:end] = torch.einsum('cijh, cjh -> cij', tmp, w_chunk)
            del tmp

        # ------------------------------------------------------------------
        # 3e.  Damping for numerical stability
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

        # Hessian-weighted error, same metric as Stage 2 Losses: (w-q)^2 / d^2
        hinv_diag2 = torch.diag(Hinv) ** 2
        Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2

        # Write Round 1 Stage 3 weights to layer before computing loss
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # Save Round 1 Stage 3 results for later comparison
        Q_best = self.layer.weight.data.clone()
        s_opt_best = s_opt.clone()
        zeros_best = all_zeros.clone()
        loss_r1 = torch.sum((self.layer(self.inp1).float() - self.out1.float()) ** 2).item()
        print('loss (round1 stage3)', loss_r1)
        print('error (round1 stage3)', torch.sum(Finalerror).item())



        # ==================================================================
        # ==================================================================
        # ROUND 2: Re-run Stage 2 + Stage 3
        #
        # Feed Round 1 Stage 3's s_opt and all_zeros into Stage 2 as
        # STATIC scales (equivalent to static_groups=True -- no dynamic
        # find_params at group boundaries). Then re-solve Stage 3 for
        # new globally optimal scales.
        # ==================================================================
        # ==================================================================

        # Reinitialize W from original weights + dead-neuron handling
        W = W_orig.clone()
        W[:, dead] = 0

        # Use Round 1 Stage 3's globally optimal scales as static input
        all_scales = s_opt.clone()
        # all_zeros carried over from Round 1 (unchanged)

        # Fresh tensors for Round 2 Stage 2
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        W_int = torch.zeros_like(W)

        # ==================================================================
        # STAGE 2 (Round 2): Core GPTQ with static scales
        #
        # Same as Round 1 Stage 2 but with NO dynamic find_params.
        # Scales and zeros are fully fixed from Round 1 Stage 3 output.
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

                # Select the static scale for this column's group
                gi = (i1 + i) // g
                scale = all_scales[:, gi:gi + 1]  # [d_out, 1]

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
        # STAGE 3 (Round 2): Closed-Form Global Reconstruction
        #
        # Same computation as Round 1 Stage 3, but with the new W_int
        # from Round 2 Stage 2. v, v_3d, H_4d are unchanged since they
        # only depend on W_orig and H_orig.
        # ==================================================================

        # 3a. Reshape new integer weights
        w_int_3d = W_int.reshape(d_out, n_g, g)                   # [d_out, n_g, g]

        # 3d. Rebuild scale-factor Hessian H_s with new w_int
        H_s = torch.empty(d_out, n_g, n_g, device=self.dev, dtype=torch.float32)
        for start in range(0, d_out, chunk_size):
            end = min(start + chunk_size, d_out)
            w_chunk = w_int_3d[start:end]
            tmp = torch.einsum('cig, igjh -> cijh', w_chunk, H_4d)
            H_s[start:end] = torch.einsum('cijh, cjh -> cij', tmp, w_chunk)
            del tmp

        # Damping
        H_s.diagonal(dim1=-2, dim2=-1).add_(1e-6)

        # 3f. Rebuild constant vector b with new w_int (v_3d unchanged)
        b = torch.einsum('oig, oig -> oi', w_int_3d, v_3d)          # [d_out, n_g]

        # 3g. Solve for new globally optimal scales
        s_opt = torch.linalg.solve(H_s, b)                          # [d_out, n_g]

        # 3h. Reconstruct final quantized weights
        Q = (s_opt.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)    # [d_out, d_in]

        Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2

        # Write Round 2 weights to layer and compute loss in fp32
        if isinstance(self.layer, transformers.Conv1D):
            Q_r2 = Q.t()
        else:
            Q_r2 = Q
        self.layer.weight.data = Q_r2.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        loss_r2 = torch.sum((self.layer(self.inp1).float() - self.out1.float()) ** 2).item()
        print('loss (round2 stage3)', loss_r2)
        print('error (round2 stage3)', torch.sum(Finalerror).item())

        # ==================================================================
        # Select the better result between Round 1 and Round 2
        # Default to round2 when losses are incomparable (both inf) or equal.
        # ==================================================================
        if loss_r2 <= loss_r1 or math.isinf(loss_r1):
            # Round 2 is better or tied -- already written to layer
            self.layer_scales = s_opt.clone()
            self.layer_zeros = all_zeros.clone()
            print(f'[selected] round2 (loss {loss_r2:.4f} <= {loss_r1:.4f})')
        else:
            # Round 1 is strictly better -- restore its weights
            self.layer.weight.data = Q_best.reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype
            )
            self.layer_scales = s_opt_best
            self.layer_zeros = zeros_best
            print(f'[selected] round1 (loss {loss_r1:.4f} < {loss_r2:.4f})')

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
