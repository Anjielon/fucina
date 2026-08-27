"""Two-plane ternary quantizer with GPTQ, on the GPU.

Why it exists: in numpy the same algorithm runs at 0.6M weights/second, i.e.
**192 hours** for a 386.5B-parameter model. That is not a loop problem — the
inner loop is vectorized and the nine-level search removed — it is arithmetic:
~50 operations per weight against numpy's ~300M ops/second. Only the GPU
changes the order of magnitude.

The algorithm, operator by operator:
  1. per-block scales d1, d2 (blocks of 256), initialized sequentially then
     alternated in closed form (a 2x2 system);
  2. GPTQ with the true Hessian: quantize a column, push its error onto the
     columns not yet quantized, propagate ACROSS blocks (not only within);
  3. the scale is recomputed from the COMPENSATED block, which is worth an
     extra 0.3 points.

Two entry points:
  quantize()          -> two joint planes (for the hot experts)
  quantize_one_plane() -> one dedicated plane (for everything else)
The distinction matters: plane-1 taken from a joint optimization is
co-adapted to its partner and loses ~10 points when used alone.
"""
from __future__ import annotations
import torch

BLOCK = 256


def _scale(W, T1, T2, lam=1e-6):
    a11 = (T1 * T1).sum(-1) + lam
    a22 = (T2 * T2).sum(-1) + lam
    a12 = (T1 * T2).sum(-1)
    b1 = (W * T1).sum(-1); b2 = (W * T2).sum(-1)
    det = a11 * a22 - a12 * a12
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    return (((b1 * a22 - b2 * a12) / det).unsqueeze(-1),
            ((b2 * a11 - b1 * a12) / det).unsqueeze(-1))


def _nine_fast(w, d1, d2):
    """Cheap assignment: round w/d1 for t1, then t2 on the residual.

    Four operations instead of thirty-six. This sits inside the GPTQ loop
    (256 iterations per block), which is where the time actually goes: with
    the exact nine-way search a 397B forge costs 7.7 hours, with this one
    about 2.3. The difference in final error is under one point."""
    t1 = torch.clamp(torch.round(w / d1.clamp(min=1e-20)), -1, 1)
    t2 = torch.clamp(torch.round((w - d1 * t1) / d2.clamp(min=1e-20)), -1, 1)
    return t1, t2


def _nine(w, d1, d2):
    """Exact assignment: the best of the nine (t1, t2) combinations."""
    best_e = None; best_c = None
    for a in (-1.0, 0.0, 1.0):
        for c in (-1.0, 0.0, 1.0):
            e = (w - (d1 * a + d2 * c)).square_()
            if best_e is None:
                best_e = e
                best_c = torch.full_like(w, a * 3 + c + 4)
            else:
                m = e < best_e
                best_e = torch.where(m, e, best_e)
                best_c = torch.where(m, torch.full_like(w, a * 3 + c + 4), best_c)
    return torch.div(best_c, 3, rounding_mode='floor') - 1.0, best_c.remainder(3) - 1.0


def _scale_one_plane(B, grid=(0.55, 0.66, 0.76, 0.86, 1.00, 1.20)):
    """Least-squares optimal ternary scale, searched over a small grid.

    Abs-max — the obvious choice — zeroes about 87% of the weights and costs
    81% error against 43.5% for this. It is the single largest factor in the
    whole pipeline."""
    ab = B.abs(); mean_ = ab.mean(-1, keepdim=True)
    en = (B * B).sum(-1)
    best_e = torch.full_like(en, float('inf')); best_d = torch.zeros_like(en)
    for m in grid:
        vivi = ab > (m * mean_)
        k = vivi.sum(-1).to(B.dtype)
        somma = torch.where(vivi, ab, torch.zeros_like(ab)).sum(-1)
        e = en - torch.where(k > 0, somma * somma / k.clamp(min=1), torch.zeros_like(en))
        ok = e < best_e
        best_e = torch.where(ok, e, best_e)
        best_d = torch.where(ok, torch.where(k > 0, somma / k.clamp(min=1), torch.zeros_like(k)), best_d)
    d = best_d.unsqueeze(-1)
    q = torch.clamp(torch.round(B / d.clamp(min=1e-20)), -1, 1)
    return d, q


@torch.no_grad()
def quantize(W, Hchol=None, rounds: int = 4, chunk: int = 2_000_000, device="cuda"):
    """W: (n_rows, n_in) on CPU → (d1, q1, d2, q2) on CPU, int8 for the signs."""
    n_rows, n_in = W.shape
    assert n_in % BLOCK == 0
    nb = n_in // BLOCK
    D1 = torch.empty(n_rows, nb, 1, dtype=torch.float32)
    D2 = torch.empty_like(D1)
    Q1 = torch.empty(n_rows, n_in, dtype=torch.int8)
    Q2 = torch.empty_like(Q1)
    H = Hchol.to(device) if Hchol is not None else None   # (n_in, n_in) INTERA

    # ⚡ The GPTQ loop issues 4,096 kernel launches PER CHUNK (16 blocks × 256
    #    columns). Small chunks multiply the launches without doing more work:
    #    at 4M rows that meant 34 chunks per tensor — 139,000 launches, and
    #    that is where the wall-clock went. A large chunk does the same 4,096
    #    launches over more data.
    # 🛟 If VRAM cannot hold it, halve and retry rather than die.
    rows_per_chunk = max(1, chunk // nb)
    i0 = 0
    while i0 < n_rows:
        i1 = min(i0 + rows_per_chunk, n_rows)
        try:
            B = W[i0:i1].to(device, torch.float32, non_blocking=True).reshape(-1, BLOCK)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows_per_chunk = max(1, rows_per_chunk // 2)
            continue
        d1, t1 = _scale_one_plane(B)
        d2, t2 = _scale_one_plane(B - d1 * t1)
        for _ in range(rounds):
            d1, d2 = _scale(B, t1, t2)
            n1, n2 = _nine_fast(B, d1, d2)
            if torch.equal(n1, t1) and torch.equal(n2, t2):
                break
            t1, t2 = n1, n2
        d1, d2 = _scale(B, t1, t2)

        if H is None:
            q1, q2 = t1, t2
        else:
            # ── Real GPTQ: error is propagated WITHIN the block (the loop
            #    over 256 columns) AND ACROSS blocks (one matrix product per
            #    block, which costs almost nothing on the GPU).
            #    ⛔ The first version omitted the second half and used the
            #       MEAN of H's diagonal blocks instead: measured 15.67%
            #       error against the ~6% that the true anisotropy
            #       (alpha 1.17) promises. Full-matrix propagation is not
            #       an optimization, it is the method.
            rows = i1 - i0
            M = B.reshape(rows, nb, BLOCK)          # (rows, blocks, 256)
            S1 = d1.reshape(rows, nb, 1); S2 = d2.reshape(rows, nb, 1)
            Q1p = torch.empty_like(M); Q2p = torch.empty_like(M)
            for b in range(nb):
                Wb = M[:, b, :].clone()
                s1 = S1[:, b, 0]; s2 = S2[:, b, 0]
                j0 = b * BLOCK
                err = torch.empty_like(Wb)
                for j in range(BLOCK):
                    w = Wb[:, j]
                    a, c = _nine_fast(w, s1, s2)
                    Q1p[:, b, j] = a; Q2p[:, b, j] = c
                    e = (w - (s1 * a + s2 * c)) / H[j0 + j, j0 + j].clamp(min=1e-8)
                    err[:, j] = e
                    if j + 1 < BLOCK:
                        Wb[:, j+1:] -= e.unsqueeze(1) * H[j0 + j, j0+j+1 : j0+BLOCK].unsqueeze(0)
                if b + 1 < nb:       # spinta sui blocks successivi: UN matmul
                    M[:, b+1:, :] -= (err @ H[j0:j0+BLOCK, j0+BLOCK:]).reshape(rows, nb-b-1, BLOCK)
            q1 = Q1p.reshape(-1, BLOCK); q2 = Q2p.reshape(-1, BLOCK)
        D1[i0:i1] = d1.reshape(i1 - i0, nb, 1).cpu()
        D2[i0:i1] = d2.reshape(i1 - i0, nb, 1).cpu()
        Q1[i0:i1] = q1.reshape(i1 - i0, n_in).to(torch.int8).cpu()
        Q2[i0:i1] = q2.reshape(i1 - i0, n_in).to(torch.int8).cpu()
        del B
        i0 = i1
    return D1.numpy(), Q1.numpy(), D2.numpy(), Q2.numpy()


@torch.no_grad()
def quantize_one_plane(W, Hchol, chunk: int = 3_000_000, device="cuda"):
    """A SINGLE ternary plane, with GPTQ and the scale RECOMPUTED per block.

    ⛔⛔ Why this function exists — ten points of error, measured:
       taking plane 1 out of a JOINT two-plane optimization and using it alone
       gives **37.95%**; quantizing a dedicated single plane gives **28.13%**.
       The joint plane-1 is CO-ADAPTED to its partner and is poor by itself.
       This is the single most important trap in multi-plane ternary
       quantization, and cold experts must therefore get a dedicated plane —
       never the first half of a joint pair.

    ⭐ The scale is recomputed INSIDE the loop, from the block ALREADY
       COMPENSATED by the preceding blocks, not from the original block
       (28.13 against 28.43 — small, and free).
    """
    n_rows, n_in = W.shape
    assert n_in % BLOCK == 0
    nb = n_in // BLOCK
    D1 = torch.empty(n_rows, nb, 1, dtype=torch.float32)
    Q1 = torch.empty(n_rows, n_in, dtype=torch.int8)
    H = Hchol.to(device)
    rows_per_chunk = max(1, chunk // nb)
    i0 = 0
    while i0 < n_rows:
        i1 = min(i0 + rows_per_chunk, n_rows)
        try:
            Wc = W[i0:i1].to(device, torch.float32, non_blocking=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); rows_per_chunk = max(1, rows_per_chunk // 2); continue
        rows = i1 - i0
        dloc = torch.empty(rows, nb, 1, device=device)
        qloc = torch.empty(rows, n_in, device=device)
        for b in range(nb):
            j0, j1 = b * BLOCK, (b + 1) * BLOCK
            block = Wc[:, j0:j1].clone()
            d, _ = _scale_one_plane(block)          # scale from the COMPENSATED block
            dloc[:, b] = d
            err = torch.empty_like(block)
            inv = 1.0 / d.clamp(min=1e-20)
            for j in range(BLOCK):
                w = block[:, j]
                q = torch.clamp(torch.round(w * inv[:, 0]), -1, 1)
                qloc[:, j0 + j] = q
                e = (w - d[:, 0] * q) / H[j0 + j, j0 + j].clamp(min=1e-8)
                err[:, j] = e
                if j + 1 < BLOCK:
                    block[:, j+1:] -= e.unsqueeze(1) * H[j0 + j, j0+j+1 : j1].unsqueeze(0)
            if j1 < n_in:
                Wc[:, j1:] -= err @ H[j0:j1, j1:]
        D1[i0:i1] = dloc.cpu(); Q1[i0:i1] = qloc.to(torch.int8).cpu()
        del Wc, dloc, qloc
        i0 = i1
    return D1.numpy(), Q1.numpy()
