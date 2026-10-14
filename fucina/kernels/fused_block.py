"""2026-09-23 — TRITON KERNEL: one WHOLE block of the coordinate descent
in ONE SINGLE GPU launch.

Why (diagnosis redone after the blocked fix; the previous night's counts were
outdated): with the blocked update the memory traffic dropped from 4.4 TB
to ~310 GB per sweep — i.e. ~6 GB/s effective against a peak of 210-260.
We are NO longer bandwidth-bound. What dominates now is the NUMBER OF
LAUNCHES: 2048 positions x ~20 elementwise operations on tensors of just
131k elements (256x512) = ~41,000 GPU launches, each with a fixed overhead
of tens of microseconds that no data "fills".

This kernel removes exactly that: each program keeps the tile (rows x
block columns) in REGISTERS and walks the COUNT positions of the block
INTERNALLY, never returning to the host. Launches: 1 per block instead
of ~320 -> from ~41,000 to ~128 per sweep.

⛔ The math is COPIED LINE BY LINE from the blocked batch sweep reference
(already verified 0/268,435,456 different against the sequential reference),
including the dj==0 guard (the real bug found that night). It must be verified
with the same rigor BEFORE use: if it diverges by a single trit, it is not used.

⚠️ The update BETWEEN blocks (matmul Err @ H[i1:i2, i2:]) stays in PyTorch:
there the matmul is already efficient and is not the bottleneck.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _block_kernel(
    e_ptr, g_ptr, q_ptr, d_ptr, err_ptr,
    hloc_ptr, hdiag_ptr,
    ROWS, N_IN, I1,
    se_b, se_r, se_n,          # strides of e/g/q/d (B, rows, n_in)
    serr_b, serr_r, serr_c,    # strides of err (B, rows, COUNT)
    shl_b, shl_i, shl_j,       # strides of hloc (B, COUNT, COUNT)
    shd_b, shd_n,              # strides of hdiag (B, N_IN)
    COUNT: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)

    r_off = pid_r * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    r_mask = r_off < ROWS
    c_idx = tl.arange(0, COUNT)                 # 0..COUNT-1 (index INSIDE the block)
    c_off = I1 + c_idx                          # absolute column

    m2 = r_mask[:, None]

    base = pid_b * se_b + r_off[:, None] * se_r + c_off[None, :] * se_n
    e_t = tl.load(e_ptr + base, mask=m2, other=0.0)
    g_t = tl.load(g_ptr + base, mask=m2, other=0.0)
    q_t = tl.load(q_ptr + base, mask=m2, other=0.0)
    d_t = tl.load(d_ptr + base, mask=m2, other=0.0)

    hdiag = tl.load(hdiag_ptr + pid_b * shd_b + c_off * shd_n)          # (COUNT,)
    hloc = tl.load(hloc_ptr + pid_b * shl_b
                   + c_idx[:, None] * shl_i + c_idx[None, :] * shl_j)   # (COUNT, COUNT)

    err_t = tl.zeros((BLOCK_ROWS, COUNT), dtype=e_t.dtype)

    for jj in tl.static_range(COUNT):
        sel = (c_idx == jj)                     # (COUNT,) mask of column jj
        selc = sel.to(e_t.dtype)
        # EXACT extraction of column jj (a single non-zero element: no
        # rounding introduced by the sum)
        dj = tl.sum(d_t * selc[None, :], axis=1)
        q_cur = tl.sum(q_t * selc[None, :], axis=1)
        e_old = tl.sum(e_t * selc[None, :], axis=1)
        g_old = tl.sum(g_t * selc[None, :], axis=1)
        hd = tl.sum(hdiag * selc)               # scalar
        hrow = tl.sum(hloc * selc[:, None], axis=0)   # row jj of hloc, (COUNT,)

        w_j = e_old + dj * q_cur
        lin = g_old - hd * e_old

        # candidate -1.0 (set unconditionally, like the reference)
        e_m1 = w_j - dj * (-1.0)
        v_m1 = e_m1 * e_m1 * hd + 2.0 * e_m1 * lin
        best_val = v_m1
        best_e = e_m1
        best_q = tl.full(best_val.shape, -1.0, dtype=e_t.dtype)
        # candidate 0.0
        e_0 = w_j
        v_0 = e_0 * e_0 * hd + 2.0 * e_0 * lin
        improve = v_0 < best_val
        best_val = tl.where(improve, v_0, best_val)
        best_e = tl.where(improve, e_0, best_e)
        best_q = tl.where(improve, tl.zeros(best_q.shape, dtype=e_t.dtype), best_q)
        # candidate 1.0
        e_p1 = w_j - dj
        v_p1 = e_p1 * e_p1 * hd + 2.0 * e_p1 * lin
        improve = v_p1 < best_val
        best_val = tl.where(improve, v_p1, best_val)
        best_e = tl.where(improve, e_p1, best_e)
        best_q = tl.where(improve, tl.full(best_q.shape, 1.0, dtype=e_t.dtype), best_q)

        # ⛔ dj==0 guard (the real bug found that night): mathematical tie among
        # the three candidates -> the current trit is left untouched
        dj_zero = dj == 0.0
        best_q = tl.where(dj_zero, q_cur, best_q)
        best_e = tl.where(dj_zero, e_old, best_e)

        delta = best_e - e_old

        # LOCAL update inside the block: only columns [jj, COUNT)
        forward_ = (c_idx >= jj).to(e_t.dtype)
        g_t = g_t + delta[:, None] * (hrow * forward_)[None, :]

        e_t = tl.where(sel[None, :], best_e[:, None], e_t)
        q_t = tl.where(sel[None, :], best_q[:, None], q_t)
        err_t = tl.where(sel[None, :], delta[:, None], err_t)

    tl.store(e_ptr + base, e_t, mask=m2)
    tl.store(g_ptr + base, g_t, mask=m2)
    tl.store(q_ptr + base, q_t, mask=m2)
    err_base = pid_b * serr_b + r_off[:, None] * serr_r + c_idx[None, :] * serr_c
    tl.store(err_ptr + err_base, err_t, mask=m2)


@torch.no_grad()
def refine_trit_sweep_batch_fused(Wrot, Q, D, Hrot, device="cuda",
                                   blocksize=16, block_rows=64):
    """SAME contract and SAME math as the blocked batch sweep reference,
    but the loop inside the block runs inside ONE Triton kernel instead of
    in Python. Returns Q_new int8 on CPU."""
    from fucina.ternary.gptq_1024 import GROUP

    B, rows, n_in = Wrot.shape
    W = Wrot.to(device=device, dtype=torch.float64)
    Qc = Q.to(device=device, dtype=torch.float64).clone()
    Dg = D.to(device=device, dtype=torch.float64)
    H = torch.as_tensor(Hrot, dtype=torch.float64, device=device)

    d_per_pos = Dg.repeat_interleave(GROUP, dim=2).contiguous()
    e = (W - Qc * d_per_pos).contiguous()
    g = torch.stack([e[b] @ H[b] for b in range(B)]).contiguous()
    Hdiag = torch.diagonal(H, dim1=-2, dim2=-1).clamp(min=1e-12).contiguous()
    Qc = Qc.contiguous()

    err = torch.empty(B, rows, blocksize, dtype=torch.float64, device=device)

    for i1 in range(0, n_in, blocksize):
        count = min(blocksize, n_in - i1)
        i2 = i1 + count
        err_b = err if count == blocksize else err[:, :, :count].contiguous()
        fused_block(e, g, Qc, d_per_pos, H, Hdiag, i1, count, err_b,
                    block_rows=block_rows)
        if i2 < n_in:
            g[:, :, i2:] += torch.matmul(err_b, H[:, i1:i2, i2:])

    return Qc.to(torch.int8).cpu()


def fused_block(e, g, Qc, d_per_pos, H, Hdiag, i1, count, err_out, block_rows=64):
    """Runs the `count` positions of the block [i1, i1+count) in ONE launch.
    Modifies e/g/Qc in place and fills err_out (B, rows, count).
    e,g,Qc,d_per_pos: (B,rows,n_in) · H: (B,n_in,n_in) · Hdiag: (B,n_in)."""
    B, rows, n_in = e.shape
    hloc = H[:, i1:i1 + count, i1:i1 + count].contiguous()
    grid = (B, triton.cdiv(rows, block_rows))
    _block_kernel[grid](
        e, g, Qc, d_per_pos, err_out,
        hloc, Hdiag,
        rows, n_in, i1,
        e.stride(0), e.stride(1), e.stride(2),
        err_out.stride(0), err_out.stride(1), err_out.stride(2),
        hloc.stride(0), hloc.stride(1), hloc.stride(2),
        Hdiag.stride(0), Hdiag.stride(1),
        COUNT=count, BLOCK_ROWS=block_rows,
    )
