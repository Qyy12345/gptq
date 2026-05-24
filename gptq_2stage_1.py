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
    Two-stage GPTQ with Joint Closed-Form Global Scale & Zero-Point Reconstruction.

    Stage 1: Independent local initialization -- Hessian-weighted grid search
             finds optimal per-group scale factors and zero-points.
    Stage 2: Core GPTQ -- standard iterative quantization with Hessian-based
             error compensation, using the initial scales from Stage 1.
    Stage 3: Joint Closed-Form Global Reconstruction -- freeze absolute integer 
             weights and solve for globally optimal scale (s) and offset (c=s*z) 
             simultaneously via a 2*n_g dimensional batched linear system.
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
                zero = all_zeros[:, gi:gi + 1]    # [d_out, 1]

                # w_int_col here is (q_int - zero)
                w_int_col = quantize_int(
                    w.unsqueeze(1), scale, zero, maxq
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
                # print(torch.sum((self.layer(self.inp1).float() - self.out1.float()) ** 2).item())
                # print(torch.sum(Losses).item())

        # ==================================================================
        # STAGE 3: Joint Closed-Form Global Reconstruction (Scale + Zero)
        #
        # Freeze absolute integer weights w_int \in [0, maxq], and solve for 
        # both s and c (where c = s * z) in a 2*n_g dimensional linear system:
        #
        #   [ H_ss  H_sc ] [ s ] = [ b_s ]
        #   [ H_cs  H_cc ] [ c ]   [ b_c ]
        # ==================================================================

        # ------------------------------------------------------------------
        # 3a. Recover the Absolute Raw Integer Weights
        # W_int currently stores (q_int - zero). We must add zero back to 
        # freeze the absolute discrete state q_int in [0, maxq].
        # ------------------------------------------------------------------
        w_int_3d = W_int.reshape(d_out, n_g, g) + all_zeros.unsqueeze(2)    # [d_out, n_g, g]

        # ------------------------------------------------------------------
        # 3b. Feature projection
        # ------------------------------------------------------------------
        v = W_orig.float() @ H_orig.float()                                 # [d_out, d_in]
        v_3d = v.reshape(d_out, n_g, g)                                     # [d_out, n_g, g]
        H_4d = H_orig.float().reshape(n_g, g, n_g, g)                       # [n_g, g, n_g, g]

        # ------------------------------------------------------------------
        # 3c. Build Block Matrices for the Super Hessian
        # ------------------------------------------------------------------
        # Block 1: H_ss [d_out, n_g, n_g]
        chunk_size = 256
        H_ss = torch.empty(d_out, n_g, n_g, device=self.dev, dtype=torch.float32)
        for start in range(0, d_out, chunk_size):
            end = min(start + chunk_size, d_out)
            w_chunk = w_int_3d[start:end]
            tmp = torch.einsum('cig, igjh -> cijh', w_chunk, H_4d)
            H_ss[start:end] = torch.einsum('cijh, cjh -> cij', tmp, w_chunk)
            del tmp
            
        # Block 2: H_cc [d_out, n_g, n_g]
        # H_cc[i, j] = 1 * H[i, j] * 1^T = sum_{g, h} H_4d[i, g, j, h]
        H_cc_2d = H_4d.sum(dim=(1, 3))                                      # [n_g, n_g]
        H_cc = H_cc_2d.unsqueeze(0).expand(d_out, -1, -1)                   # [d_out, n_g, n_g]

        # Block 3: H_sc [d_out, n_g, n_g]
        # H_sc[i, j] = - w_int_i * H_ij * 1^T
        H_sum_cols = H_4d.sum(dim=3)                                        # [n_g, g, n_g]
        H_sc = -torch.einsum('oig, igj -> oij', w_int_3d, H_sum_cols)       # [d_out, n_g, n_g]

        # Block 4: H_cs [d_out, n_g, n_g]
        # H_cs is the transpose of H_sc
        H_cs = H_sc.transpose(-1, -2)                                       # [d_out, n_g, n_g]

        # ------------------------------------------------------------------
        # 3d. Assemble the 2*n_g x 2*n_g Super System
        # ------------------------------------------------------------------
        H_full = torch.empty(d_out, 2 * n_g, 2 * n_g, device=self.dev, dtype=torch.float32)
        H_full[:, :n_g, :n_g] = H_ss
        H_full[:, :n_g, n_g:] = H_sc
        H_full[:, n_g:, :n_g] = H_cs
        H_full[:, n_g:, n_g:] = H_cc

        # Damping for numerical stability
        H_full.diagonal(dim1=-2, dim2=-1).add_(1e-6)

        # ------------------------------------------------------------------
        # 3e. Build the Constant Vectors
        # ------------------------------------------------------------------
        b_s = torch.einsum('oig, oig -> oi', w_int_3d, v_3d)                # [d_out, n_g]
        b_c = -v_3d.sum(dim=-1)                                             # [d_out, n_g]
        b_full = torch.cat([b_s, b_c], dim=-1)                              # [d_out, 2*n_g]

        # ------------------------------------------------------------------
        # 3f. Solve for (s, c)
        # ------------------------------------------------------------------
        x_opt = torch.linalg.solve(H_full, b_full)                          # [d_out, 2*n_g]

        s_opt = x_opt[:, :n_g]                                              # [d_out, n_g]
        c_opt = x_opt[:, n_g:]                                              # [d_out, n_g]
        
        # Recover optimal zeros: z_opt = c_opt / s_opt
        z_opt = c_opt / (s_opt + 1e-8)                                      # [d_out, n_g]

        # ------------------------------------------------------------------
        # 3g. Reconstruct final quantized weights
        # q = s * w_raw_int - c
        # ------------------------------------------------------------------
        Q = (s_opt.unsqueeze(2) * w_int_3d - c_opt.unsqueeze(2)).reshape(d_out, d_in)

        # Hessian-weighted error
        hinv_diag2 = torch.diag(Hinv) ** 2
        Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2

        # Save globally optimal scales and per-group zeros
        self.layer_scales = s_opt.clone()
        self.layer_zeros = z_opt.clone()

        # Write Stage 3 weights before computing output loss
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        print('loss (stage3)', torch.sum((self.layer(self.inp1).float() - self.out1.float()) ** 2).item())
        print('error (stage3)', torch.sum(Finalerror).item())

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