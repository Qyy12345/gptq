import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    """
    Two-stage GPTQ with closed-form global scale reconstruction.

    Stage 1: Independent local initialization -- Hessian-weighted grid search
             finds optimal per-group scale factors.
    Stage 2: Core GPTQ -- standard iterative quantization with Hessian-based
             error compensation, using the initial scales from Stage 1.
    Stage 3: Closed-form global reconstruction -- freeze integer weights and
             solve for globally optimal scale factors via a batched linear
             system (no Python loops over channels or groups).
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
        self.R = torch.zeros((self.columns, self.columns), device=self.dev)
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
        self.R *= decay
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        if len(self.fp_inp) > 0:
            # Process fp input through the same shape pipeline as inp
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
            #  R = (X - X̂) X̂^T
            dX = inp_fp - inp
            self.R += dX.matmul(inp.t())
            del self.fp_inp[0]
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
        R_orig = self.R.clone()

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
        #
        # Standard iterative quantization with Hessian-based error
        # compensation. Scales are FIXED from Stage 1 -- no dynamic
        # find_params calls inside the loop.
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
        # 3b. Feature projection with cross-layer error compensation
        #
        #     Base projection:  v = W_orig @ H_orig
        #     Compensation term: u = W_orig @ R_orig, where R = E[ΔX · X^T]
        #     Combined: v_compensated = v + u
        #
        #     This corrects for the input distribution shift caused by
        #     quantization of previous layers (ΔX = X̂ - X).
        # ------------------------------------------------------------------
        v = W_orig.float() @ H_orig.float()                         # [d_out, d_in]
        u = W_orig.float() @ R_orig.float()                         # [d_out, d_in]
        v_compensated = v + u                                        # [d_out, d_in]
        v_3d = v_compensated.reshape(d_out, n_g, g)                 # [d_out, n_g, g]

        # ------------------------------------------------------------------
        # 3c. Reshape H_orig into 4-D block view
        #
        #     H_orig is [d_in, d_in] = [n_g*g, n_g*g].
        #     Reshape to [n_g, g, n_g, g] so that H_4d[i, :, j, :] is the
        #     g x g block of H between group i and group j.
        # ------------------------------------------------------------------
        H_4d = H_orig.float().reshape(n_g, g, n_g, g)              # [n_g, g, n_g, g]

        # ------------------------------------------------------------------
        # 3d. Build scale-factor Hessian H_s  [d_out, n_g, n_g]
        #
        #     H_s[o, i, j] = sum_{g1} sum_{g2}
        #         w_int_3d[o, i, g1] * H_4d[i, g1, j, g2] * w_int_3d[o, j, g2]
        #
        #     Physical meaning:
        #       This is the second derivative of the reconstruction loss
        #       with respect to scale factors s_i and s_j.  Diagonal blocks
        #       (i == j) capture within-group curvature; off-diagonal blocks
        #       capture cross-group coupling through the feature Hessian.
        #
        #     Einsum contraction:
        #       'oig, igjh, ojh -> oij'
        #
        #       Operand 1:  w_int_3d  [o, i, g]   = [d_out, n_g, g]
        #       Operand 2:  H_4d      [i, g, j, h] = [n_g, g, n_g, g]
        #       Operand 3:  w_int_3d  [o, j, h]   = [d_out, n_g, g]
        #       Output:     H_s       [o, i, j]   = [d_out, n_g, n_g]
        #
        #     The contraction sums over the element dimensions g and h,
        #     leaving the batch (o), row-group (i), and col-group (j) axes.
        # ------------------------------------------------------------------
        H_s = torch.einsum('oig, igjh, ojh -> oij', w_int_3d, H_4d, w_int_3d)

        # ------------------------------------------------------------------
        # 3e. Damping for numerical stability
        # ------------------------------------------------------------------
        H_s.diagonal(dim1=-2, dim2=-1).add_(1e-6)

        # ------------------------------------------------------------------
        # 3f. Build constant vector b  [d_out, n_g]
        #
        #     b[o, i] = sum_g w_int_3d[o, i, g] * v_3d[o, i, g]
        #
        #     Physical meaning:
        #       This is the first derivative of the reconstruction loss
        #       w.r.t. s_i, projected onto the integer weights.  It
        #       represents how much the loss would decrease by scaling
        #       group i's quantized values to better match the projection
        #       of the original weights onto the feature covariance.
        #
        #     Einsum contraction:
        #       'oig, oig -> oi'
        #       Sums over the element dimension g, leaving batch (o) and
        #       group (i) axes.
        # ------------------------------------------------------------------
        b = torch.einsum('oig, oig -> oi', w_int_3d, v_3d)          # [d_out, n_g]

        # ------------------------------------------------------------------
        # 3g. Solve for globally optimal scales: H_s @ s_opt = b
        #
        #     torch.linalg.solve batches over the leading d_out dimension,
        #     solving n_g x n_g systems for each output channel in parallel
        #     on the GPU via batched LAPACK (gesv).
        # ------------------------------------------------------------------
        s_opt = torch.linalg.solve(H_s, b)                          # [d_out, n_g]

        # ------------------------------------------------------------------
        # 3h. Reconstruct final quantized weights
        #
        #     w_q[o, i, g] = s_opt[o, i] * w_int[o, i, g]
        #
        #     Broadcasting: s_opt.unsqueeze(2) is [d_out, n_g, 1],
        #     w_int_3d is [d_out, n_g, g].  The broadcast multiplies each
        #     group's integer values by its globally optimal scale.
        # ------------------------------------------------------------------
        Q = (s_opt.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)    # [d_out, d_in]

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        # Save globally optimal scales for downstream pack() / INT export
        self.layer_scales = s_opt.clone()

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
        self.R = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
