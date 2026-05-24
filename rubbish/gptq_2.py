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
        """
        Novel GPTQ algorithm with sequential column-by-column quantization.

        Algorithm flow:
        1. For each column j: QUANTIZE W[:,j] first
        2. Update next column using cumulative error from quantized columns W[:,:j+1]
        3. UpdateTerm = L[j+1,j+1] * ((W[:,:j+1] - W_FP[:,:j+1]) @ H[:j+1,j+1:] @ L[j+1:,j+1])

        Key differences from standard GPTQ:
        1. Quantize-then-update: quantize column j BEFORE computing updates
        2. Single column update: only updates column j+1 (not all remaining columns)
        3. No lazy updates: processes columns sequentially without block batching
        4. Cumulative error: uses error from ALL previously quantized columns (W[:,:j+1] - W_FP[:,:j+1])
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # IMPORTANT: Make a copy of H before Cholesky decomposition
        # The novel algorithm requires the original H matrix for UpdateTerm computation
        H = self.H.clone()
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

        # Store FP weights for (W - W^FP) term computation
        W_FP = W.clone()

        # Prepare quantization output matrix
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # Add damping for numerical stability
        # λ = 1% of average diagonal value prevents numerical issues
        damp = percdamp * torch.mean(torch.diag(H))
        diag_indices = torch.arange(self.columns, device=self.dev)
        H_damped = H.clone()
        H_damped[diag_indices, diag_indices] += damp

        # Compute L = Cholesky((H + λI)^{-1}) using numerically stable approach
        # This is the CRITICAL fix for large matrices (e.g., LLaMA's 4096×4096)
        # Using cholesky_inverse is more stable than torch.inverse for large matrices
        # Reference: Original GPTQ (gptq.py lines 101-103)
        H_chol = torch.linalg.cholesky(H_damped, upper=False)  # H_damped = L_chol @ L_chol^T
        H_inv = torch.cholesky_inverse(H_chol)  # Computes H_damped^{-1} efficiently
        H_inv = (H_inv + H_inv.T) / 2  # Symmetrize to prevent floating-point asymmetry
        L = torch.linalg.cholesky(H_inv, upper=False)  # L where L @ L^T = (H + λI)^{-1}
        # L is lower triangular [d_in, d_in]

        # Shape annotations:
        # H:   [d_in, d_in] - original Hessian (preserved)
        # L:   [d_in, d_in] - lower triangular Cholesky of (H + λI)^{-1}
        # W:   [d_out, d_in] - weight matrix (modified in-place)
        # W_FP: [d_out, d_in] - original FP weights (constant reference)

        # Sequential column-by-column quantization
        for j in range(self.columns):
            # Handle groupsize quantization if enabled
            if groupsize != -1:
                if not static_groups:
                    if j % groupsize == 0:
                        self.quantizer.find_params(W[:, j:(j + groupsize)], weight=True)
                else:
                    idx = j
                    if actorder:
                        idx = perm[idx]
                    self.quantizer = groups[idx // groupsize]

            # Step 1: QUANTIZE current column FIRST (new algorithm)
            # Save original FP value for loss computation
            w_fp = W[:, j].clone()  # Shape: [d_out]

            # Quantize current column and apply immediately
            q = quantize(
                w_fp.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
            ).flatten()  # Shape: [d_out]

            # Store quantized result in output matrix
            Q[:, j] = q

            # Apply quantization to weight matrix (immediately, not at end)
            W[:, j] = q

            # Compute loss for current column (matching gptq.py line 134)
            # Loss = (w_fp - q)^2 / L[j,j]^2 / 2
            # Note: gptq.py divides by 2 in line 141: Losses1 / 2
            d = L[j, j]
            Losses[:, j] = ((w_fp - q) ** 2 / (d ** 2)) / 2

            # Step 2: Update NEXT column only (if exists)
            if j + 1 < self.columns:
                # UpdateTerm: L[j+1, j+1] * ((W[:, :j+1] - W_FP[:, :j+1]) @ H[:j+1, j+1:] @ L[j+1:, j+1])
                # Breakdown:
                # W[:, :j+1]: [d_out, (j+1)] - first j+1 columns (NOW QUANTIZED, including current column)
                # W_FP[:, :j+1]: [d_out, (j+1)] - original FP weights for first j+1 columns
                # (W - W_FP)[:, :j+1]: [d_out, (j+1)] - cumulative quantization error from quantized columns
                # H[:j+1, j+1:]: [(j+1), d_in-(j+1)] - rows 0 to j, columns j+1 to end of Hessian
                # L[j+1:, j+1]: [d_in-(j+1)] - column j+1 of L, from row j+1 onwards
                # temp = H[:j+1, j+1:] @ L[j+1:, j+1]: [(j+1), d_in-(j+1)] @ [d_in-(j+1)] = [(j+1)]
                # (W - W_FP)[:, :j+1] @ temp: [d_out, (j+1)] @ [(j+1)] = [d_out]
                # L[j+1, j+1]: scalar - diagonal element
                # UpdateTerm shape: [d_out]

                # Compute H[:j+1, j+1:] @ L[j+1:, j+1]
                # This is: [(j+1), d_in-(j+1)] @ [d_in-(j+1)] = [(j+1)]
                temp = H[:j + 1, j + 1:] @ L[j + 1:, j + 1]

                # Compute (W[:, :j+1] - W_FP[:, :j+1]) @ temp
                # Note: W[:, :j+1] is now QUANTIZED (including the column we just quantized)
                # This is: [d_out, (j+1)] @ [(j+1)] = [d_out]
                temp2 = (W[:, :j + 1] - W_FP[:, :j + 1]) @ temp

                # Scale by L[j+1, j+1]
                UpdateTerm = L[j + 1, j + 1] * temp2

                # Update the NEXT column only
                # W[:, j+1] = W[:, j+1] - UpdateTerm
                W[:, j + 1] = W[:, j + 1] - UpdateTerm

            # Print progress at block intervals (matching gptq.py's output frequency)
            # gptq.py outputs once per block (blocksize=128), so we do the same
            if DEBUG and ((j + 1) % blocksize == 0 or j == self.columns - 1):
                self.layer.weight.data[:, :j + 1] = Q[:, :j + 1]
                self.layer.weight.data[:, j + 1:] = W[:, j + 1:]
                # Print output reconstruction error and total weight error (matching gptq.py)
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
