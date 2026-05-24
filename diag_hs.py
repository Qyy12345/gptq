"""
Diagnostic: monkey-patches gptq_2stage_2.GPTQ.fasterquant to print
full-channel min-eigenvalue stats for H_s (both rounds) BEFORE damping.
No modifications to original files.
"""
import torch
import numpy as np

# Must patch BEFORE llama.py is imported
import gptq_2stage_2 as gptq_mod
_orig_fasterquant = gptq_mod.GPTQ.fasterquant


def _diag_fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1,
                      actorder=False, static_groups=False, stage1_hessian=False):
    # Call original — it will do everything including printing losses.
    # We intercept by wrapping the build+solve of H_s.
    # Instead, we just run the original and add a post-hoc diagnostic
    # by re-building H_s with saved data.

    # Actually, the cleanest approach: run original fasterquant,
    # then re-build H_s for diagnostics.
    # But the original deletes self.H, so we need to save H_orig before.

    # Save H for post-hoc diagnostics
    H_for_diag = self.H.clone()

    # Run original fasterquant
    _orig_fasterquant(self, blocksize=blocksize, percdamp=percdamp,
                      groupsize=groupsize, actorder=actorder,
                      static_groups=static_groups, stage1_hessian=stage1_hessian)

    # Post-hoc: re-build H_s from saved data to check eigenvalues
    # We need W_int which was computed inside fasterquant but not saved.
    # So this approach won't work easily. Let me use a different strategy.


def _make_diag_fn(round_label):
    """Return a function that diagnoses H_s for a given round."""
    def diag(H_s, d_out, n_g, dev):
        min_eigs = torch.linalg.eigvalsh(H_s)[:, 0]
        neg_count = (min_eigs < 0).sum().item()
        zero_count = (min_eigs == 0).sum().item()

        print(f"[DIAG {round_label}] min_eig  "
              f"min={min_eigs.min().item():.6e}  "
              f"p1={torch.quantile(min_eigs.float(), 0.01).item():.6e}  "
              f"p5={torch.quantile(min_eigs.float(), 0.05).item():.6e}  "
              f"p50={torch.quantile(min_eigs.float(), 0.50).item():.6e}  "
              f"max={min_eigs.max().item():.6e}  "
              f"neg={neg_count}  zero={zero_count}")

        # Log10 histogram
        log_eigs = torch.log10(min_eigs.clamp(min=1e-30))
        bins = [-30, -15, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 5]
        parts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            cnt = ((log_eigs >= lo) & (log_eigs < hi)).sum().item()
            if cnt > 0:
                parts.append(f"[{lo:+3d},{hi:+3d}):{cnt}")
        print(f"[DIAG {round_label}] hist: {'  '.join(parts)}")
    return diag


# Strategy: patch the build-H_s-and-solve block.
# We'll replace fasterquant entirely, adding diag calls at the two
# H_s construction points (round1 line 252 and round2 line 382).

def _patched_fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1,
                         actorder=False, static_groups=False, stage1_hessian=False):
    import math, time
    import torch.nn as nn
    import transformers
    from quant import quantize_int

    W = self.layer.weight.data.clone()
    if isinstance(self.layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()
    tick = time.time()

    assert groupsize > 0
    assert not actorder

    d_out = self.rows
    d_in = self.columns
    g = groupsize
    n_g = d_in // g

    H_orig = self.H.clone()
    W_orig = W.clone()
    H = self.H
    del self.H

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    maxq = self.quantizer.maxq.to(self.dev)

    # Stage 1
    all_scales = torch.zeros((d_out, n_g), device=self.dev)
    all_zeros = torch.zeros((d_out, n_g), device=self.dev)
    if static_groups:
        for gi in range(n_g):
            s = gi * g; e = s + g
            W_grp = W[:, s:e]
            if stage1_hessian:
                H_blk = H[s:e, s:e]
                all_scales[:, gi], all_zeros[:, gi] = self.quantizer.find_params_hessian_weighted(W_grp, H_blk)
            else:
                self.quantizer.find_params(W_grp, weight=True)
                all_scales[:, gi] = self.quantizer.scale.flatten()
                all_zeros[:, gi] = self.quantizer.zero.flatten()

    Losses = torch.zeros_like(W); Q = torch.zeros_like(W); W_int = torch.zeros_like(W)
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(d_in, device=self.dev)
    H[diag_idx, diag_idx] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    # Stage 2 (Round 1)
    for i1 in range(0, d_in, blocksize):
        i2 = min(i1 + blocksize, d_in); count = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1)
        W_int1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]; d = Hinv1[i, i]; gi = (i1 + i) // g
            if not static_groups and (i1 + i) % g == 0:
                if stage1_hessian:
                    s = gi * g; H_blk = H_orig[s:s + g, s:s + g]
                    all_scales[:, gi], all_zeros[:, gi] = self.quantizer.find_params_hessian_weighted(
                        W[:, (i1 + i):(i1 + i + g)], H_blk)
                else:
                    self.quantizer.find_params(W[:, (i1 + i):(i1 + i + g)], weight=True)
                    all_scales[:, gi] = self.quantizer.scale.flatten()
                    all_zeros[:, gi] = self.quantizer.zero.flatten()
            scale = all_scales[:, gi:gi + 1]
            w_int_col = quantize_int(w.unsqueeze(1), scale, all_zeros[:, gi:gi + 1], maxq).flatten()
            q = scale.flatten() * w_int_col
            Q1[:, i] = q; W_int1[:, i] = w_int_col
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1; W_int[:, i1:i2] = W_int1
        Losses[:, i1:i2] = Losses1 / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        if gptq_mod.DEBUG:
            self.layer.weight.data[:, :i2] = Q[:, :i2]
            self.layer.weight.data[:, i2:] = W[:, i2:]
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2).item())
            print(torch.sum(Losses).item())

    # Stage 3 (Round 1) — build H_s
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

    # ---- DIAG Round 1 ----
    _make_diag_fn("R1")(H_s, d_out, n_g, self.dev)

    H_s.diagonal(dim1=-2, dim2=-1).add_(1e-6)
    b = torch.einsum('oig, oig -> oi', w_int_3d, v_3d)
    s_opt = torch.linalg.solve(H_s, b)
    Q = (s_opt.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)
    hinv_diag2 = torch.diag(Hinv) ** 2
    Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2
    if isinstance(self.layer, transformers.Conv1D):
        Q_best = Q.t().clone()
    else:
        Q_best = Q.clone()
    s_opt_best = s_opt.clone(); zeros_best = all_zeros.clone()
    loss_r1 = torch.sum((self.layer(self.inp1) - self.out1) ** 2).item()
    print('loss (round1 stage3)', loss_r1)
    print('error (round1 stage3)', torch.sum(Finalerror).item())

    # Round 2
    W = W_orig.clone(); W[:, dead] = 0
    all_scales = s_opt.clone()
    Losses = torch.zeros_like(W); Q = torch.zeros_like(W); W_int = torch.zeros_like(W)
    for i1 in range(0, d_in, blocksize):
        i2 = min(i1 + blocksize, d_in); count = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1)
        W_int1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]; d = Hinv1[i, i]; gi = (i1 + i) // g
            scale = all_scales[:, gi:gi + 1]
            w_int_col = quantize_int(w.unsqueeze(1), scale, all_zeros[:, gi:gi + 1], maxq).flatten()
            q = scale.flatten() * w_int_col
            Q1[:, i] = q; W_int1[:, i] = w_int_col
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1; W_int[:, i1:i2] = W_int1
        Losses[:, i1:i2] = Losses1 / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        if gptq_mod.DEBUG:
            self.layer.weight.data[:, :i2] = Q[:, :i2]
            self.layer.weight.data[:, i2:] = W[:, i2:]
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2).item())
            print(torch.sum(Losses).item())

    # Stage 3 (Round 2) — rebuild H_s
    w_int_3d = W_int.reshape(d_out, n_g, g)
    H_s = torch.empty(d_out, n_g, n_g, device=self.dev, dtype=torch.float32)
    for start in range(0, d_out, chunk_size):
        end = min(start + chunk_size, d_out)
        w_chunk = w_int_3d[start:end]
        tmp = torch.einsum('cig, igjh -> cijh', w_chunk, H_4d)
        H_s[start:end] = torch.einsum('cijh, cjh -> cij', tmp, w_chunk)
        del tmp

    # ---- DIAG Round 2 ----
    _make_diag_fn("R2")(H_s, d_out, n_g, self.dev)

    H_s.diagonal(dim1=-2, dim2=-1).add_(1e-6)
    b = torch.einsum('oig, oig -> oi', w_int_3d, v_3d)
    s_opt = torch.linalg.solve(H_s, b)
    Q = (s_opt.unsqueeze(2) * w_int_3d).reshape(d_out, d_in)
    Finalerror = ((W_orig - Q) ** 2) / hinv_diag2 / 2
    if isinstance(self.layer, transformers.Conv1D):
        Q_r2 = Q.t()
    else:
        Q_r2 = Q
    self.layer.weight.data = Q_r2.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
    loss_r2 = torch.sum((self.layer(self.inp1) - self.out1) ** 2).item()
    print('loss (round2 stage3)', loss_r2)
    print('error (round2 stage3)', torch.sum(Finalerror).item())

    if loss_r2 < loss_r1:
        self.layer_scales = s_opt.clone()
        self.layer_zeros = all_zeros.clone()
        print(f'[selected] round2 (loss {loss_r2:.4f} < {loss_r1:.4f})')
    else:
        self.layer.weight.data = Q_best.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer_scales = s_opt_best
        self.layer_zeros = zeros_best
        print(f'[selected] round1 (loss {loss_r1:.4f} <= {loss_r2:.4f})')

    torch.cuda.synchronize()
    print('time %.2f' % (time.time() - tick))


# Monkey-patch
gptq_mod.GPTQ.fasterquant = _patched_fasterquant

if __name__ == '__main__':
    import argparse
    import llama as llama_mod

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--nearest', action='store_true')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16])
    parser.add_argument('--groupsize', type=int, default=-1)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--stage1_hessian', action='store_true')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--new-eval', action='store_true')
    parser.add_argument('--act-order', action='store_true')
    parser.add_argument('--true-sequential', action='store_true')
    parser.add_argument('--static-groups', action='store_true')
    args = parser.parse_args()

    from datautils import get_loaders
    from modelutils import find_layers, DEV

    model = llama_mod.get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.model, seqlen=model.seqlen)

    if args.wbits < 16 and not args.nearest:
        llama_mod.args = args
        quantizers = llama_mod.llama_sequential(model, dataloader, DEV)
