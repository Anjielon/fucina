#!/usr/bin/env python3
"""flash_qat — causal GQA flash attention for the full-attention layers of the Thor-T student (2026-10-09, new file;
part of the "QAT kernels under 3 hours" work). Replaces transformers' SDPA call, which on gfx1151 with fp32
inputs and the explicit additive mask runs the MATH backend: fp32 GEMMs without WMMA (~3 TFLOPS) and 2 x 16 x 4096 x
4096 fp32 score matrices in memory (~1.2 ms/token over the 10 attention layers in the TT6 profile).

Switch: E2E_KQAT contains `flash3` (fp32-accurate) or `flash1` (fp16 operands); otherwise nothing changes.

## Arithmetic
FlashAttention-2 (Dao, arXiv 2307.08691): online softmax over key blocks in the forward, log-sum-exp saved; backward
in two kernels (dK/dV per key block looping over the query blocks of the 8 query heads that share the kv head, and dQ
per query block), Delta = rowsum(dO * O) precomputed. No score matrix in memory, no atomics (deterministic).
Every product goes through `tl.dot` with fp16 operands (WMMA on gfx11) and fp32 accumulation:
  * flash3: each fp32 operand a = a_hi + a_lo, a_hi = fp16(a), a_lo = fp16(a - a_hi), and a.b ~ a_hi b_hi + a_hi b_lo
    + a_lo b_hi (Ootomo & Yokota, arXiv 2203.03341; the dropped a_lo b_lo is <= 2^-22 relative). Operands that can be
    small (dO, dS) are first scaled by ONE power of two per batch row so that max |.| is in [1/2, 1)
    (exact, undone exactly at the end; one scale per batch row so that the dK/dV accumulators sum the 8 heads of a kv
    head directly); q, k, v (RMS-normed / projected activations) and p in [0, 1] need no scale.
    Representation error of a split operand: <= 2^-24 absolute after scaling (fp16 subnormals reach 2^-24), i.e. the
    fp32 level relative to the maximum, like the fp32 GEMMs of the reference.
  * flash1: a single fp16 product per GEMM (q, k, v, p, dO, dS rounded to fp16 once), the precision of the engine's
    f16 flash attention with an f16 KV cache; to be judged against the reference's own noise (report).
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False


if _TRITON:
    @triton.jit
    def _split(a, SPLIT: tl.constexpr):
        """a = hi + lo 2^-11: the low part stored scaled by 2^11 (Ootomo-Yokota), a normal fp16 for |a| >= 2^-14."""
        h = a.to(tl.float16)
        if SPLIT == 3:
            lo = ((a - h.to(tl.float32)) * 2048.0).to(tl.float16)
        else:
            lo = h
        return h, lo

    @triton.jit
    def _mm(ah, al, bh, bl, SPLIT: tl.constexpr):
        """a @ b with fp16 operands, fp32 accumulation; SPLIT 3: hi*hi + (hi*lo + lo*hi) 2^-11."""
        if SPLIT == 3:
            c = tl.dot(ah, bl)
            c = tl.dot(al, bh, c)
            r = tl.dot(ah, bh) + c * (1.0 / 2048.0)
        else:
            r = tl.dot(ah, bh)
        return r

    # Head dim 256 does not fit in registers as one tile (AOT gfx1151 of the first version: 500-3300 VGPR spilled per
    # kernel): every product over the head dim is a sum of DC = 64-wide chunks loaded when needed (from L1/L2), and
    # the [., 256] accumulators are kept as four [., 64] tensors. Same products, same order of the sums.
    @triton.jit
    def _qk(pq, sq_t, om, mm_, pk, sk_t, on, mn, oc, SPLIT: tl.constexpr, NC: tl.constexpr, DC: tl.constexpr):
        """s = q k^T over the head dim in NC chunks of DC columns (q: BM rows at pq, k: BN rows at pk)."""
        s = tl.zeros((om.shape[0], on.shape[0]), tl.float32)
        for c in tl.static_range(NC):
            q = tl.load(pq + om[:, None].to(tl.int64) * sq_t + c * DC + oc[None, :], mask=mm_[:, None],
                        other=0.0).to(tl.float32)
            k = tl.load(pk + on[:, None].to(tl.int64) * sk_t + c * DC + oc[None, :], mask=mn[:, None],
                        other=0.0).to(tl.float32)
            qh, ql = _split(q, SPLIT)
            kh, kl = _split(k, SPLIT)
            s += _mm(qh, ql, tl.trans(kh), tl.trans(kl), SPLIT)
        return s

    @triton.jit
    def _load(p_, st, rows, mrows, c: tl.constexpr, oc, DC: tl.constexpr):
        return tl.load(p_ + rows[:, None].to(tl.int64) * st + c * DC + oc[None, :], mask=mrows[:, None],
                       other=0.0).to(tl.float32)

    @triton.jit
    def _fa_fwd(Q, K, V, O, LSE, LIM, sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, so_b, so_h, so_t,
                H, T, scale,
                GRP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
        DC: tl.constexpr = 64
        NC: tl.constexpr = D // 64
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        hk = h // GRP
        om = pid_m * BM + tl.arange(0, BM)
        oc = tl.arange(0, DC)
        mm_ = om < T
        pq = Q + b.to(tl.int64) * sq_b + h * sq_h
        pk = K + b.to(tl.int64) * sk_b + hk * sk_h
        pv = V + b.to(tl.int64) * sv_b + hk * sv_h
        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        a0 = tl.zeros((BM, DC), tl.float32)
        a1 = tl.zeros((BM, DC), tl.float32)
        a2 = tl.zeros((BM, DC), tl.float32)
        a3 = tl.zeros((BM, DC), tl.float32)
        end = tl.load(LIM + pid_m)                      # (pid_m + 1) * BM: last key of the causal block
        for n0 in range(0, end, BN):
            on = n0 + tl.arange(0, BN)
            mn = on < T
            s = _qk(pq, sq_t, om, mm_, pk, sk_t, on, mn, oc, SPLIT, NC, DC) * scale
            s = tl.where((on[None, :] <= om[:, None]) & mn[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            ph, pl = _split(p, SPLIT)
            vh, vl = _split(_load(pv, sv_t, on, mn, 0, oc, DC), SPLIT)
            a0 = a0 * alpha[:, None] + _mm(ph, pl, vh, vl, SPLIT)
            vh, vl = _split(_load(pv, sv_t, on, mn, 1, oc, DC), SPLIT)
            a1 = a1 * alpha[:, None] + _mm(ph, pl, vh, vl, SPLIT)
            if NC == 4:
                vh, vl = _split(_load(pv, sv_t, on, mn, 2, oc, DC), SPLIT)
                a2 = a2 * alpha[:, None] + _mm(ph, pl, vh, vl, SPLIT)
                vh, vl = _split(_load(pv, sv_t, on, mn, 3, oc, DC), SPLIT)
                a3 = a3 * alpha[:, None] + _mm(ph, pl, vh, vl, SPLIT)
            m_i = m_new
        po = O + b.to(tl.int64) * so_b + h * so_h + om[:, None].to(tl.int64) * so_t + oc[None, :]
        inv = 1.0 / l_i[:, None]
        tl.store(po, a0 * inv, mask=mm_[:, None])
        tl.store(po + DC, a1 * inv, mask=mm_[:, None])
        if NC == 4:
            tl.store(po + 2 * DC, a2 * inv, mask=mm_[:, None])
            tl.store(po + 3 * DC, a3 * inv, mask=mm_[:, None])
        tl.store(LSE + bh.to(tl.int64) * T + om, m_i + tl.log(l_i), mask=mm_)

    @triton.jit
    def _fa_bwd_dkdv(Q, K, V, DO, LSE, DEL, SC, DK, DV, LIMN,
                     sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sd_b, sd_h, sd_t,
                     sdk_b, sdk_h, sdk_t, H, HK, T, scale,
                     GRP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
        """dV[n] = sum_m P[m, n] dO[m];  dK[n] = scale sum_m dS[m, n] q[m],  dS = P (dP - Delta),  dP = dO V^T.
        dO arrives scaled by SC[b] (power of two, one per batch row), undone at the store."""
        DC: tl.constexpr = 64
        NC: tl.constexpr = D // 64
        pid_n = tl.program_id(0)
        bhk = tl.program_id(1)
        b = bhk // HK
        hk = bhk % HK
        on = pid_n * BN + tl.arange(0, BN)
        mn = on < T
        oc = tl.arange(0, DC)
        pk = K + b.to(tl.int64) * sk_b + hk * sk_h
        pv = V + b.to(tl.int64) * sv_b + hk * sv_h
        k0 = tl.zeros((BN, DC), tl.float32)
        k1 = tl.zeros((BN, DC), tl.float32)
        k2 = tl.zeros((BN, DC), tl.float32)
        k3 = tl.zeros((BN, DC), tl.float32)
        v0 = tl.zeros((BN, DC), tl.float32)
        v1 = tl.zeros((BN, DC), tl.float32)
        v2 = tl.zeros((BN, DC), tl.float32)
        v3 = tl.zeros((BN, DC), tl.float32)
        m_ini = tl.load(LIMN + pid_n)                    # (pid_n * BN // BM) * BM, loaded
        inv = 1.0 / tl.load(SC + b)
        for g in range(0, GRP):
            h = hk * GRP + g
            bh = b * H + h
            pq = Q + b.to(tl.int64) * sq_b + h * sq_h
            pdo = DO + b.to(tl.int64) * sd_b + h * sd_h
            for m0 in range(m_ini, T, BM):
                om = m0 + tl.arange(0, BM)
                mm_ = om < T
                lse = tl.load(LSE + bh.to(tl.int64) * T + om, mask=mm_, other=0.0)
                dl = tl.load(DEL + bh.to(tl.int64) * T + om, mask=mm_, other=0.0)
                s = _qk(pq, sq_t, om, mm_, pk, sk_t, on, mn, oc, SPLIT, NC, DC) * scale
                p = tl.exp(s - lse[:, None])
                p = tl.where((on[None, :] <= om[:, None]) & mm_[:, None] & mn[None, :], p, 0.0)
                dp = _qk(pdo, sd_t, om, mm_, pv, sv_t, on, mn, oc, SPLIT, NC, DC)     # dO V^T, same chunked product
                ds = p * (dp - dl[:, None])
                pth, ptl = _split(tl.trans(p), SPLIT)
                dsh, dsl = _split(tl.trans(ds), SPLIT)
                dh_, dl_ = _split(_load(pdo, sd_t, om, mm_, 0, oc, DC), SPLIT)
                v0 += _mm(pth, ptl, dh_, dl_, SPLIT)
                qh, ql = _split(_load(pq, sq_t, om, mm_, 0, oc, DC), SPLIT)
                k0 += _mm(dsh, dsl, qh, ql, SPLIT)
                dh_, dl_ = _split(_load(pdo, sd_t, om, mm_, 1, oc, DC), SPLIT)
                v1 += _mm(pth, ptl, dh_, dl_, SPLIT)
                qh, ql = _split(_load(pq, sq_t, om, mm_, 1, oc, DC), SPLIT)
                k1 += _mm(dsh, dsl, qh, ql, SPLIT)
                if NC == 4:
                    dh_, dl_ = _split(_load(pdo, sd_t, om, mm_, 2, oc, DC), SPLIT)
                    v2 += _mm(pth, ptl, dh_, dl_, SPLIT)
                    qh, ql = _split(_load(pq, sq_t, om, mm_, 2, oc, DC), SPLIT)
                    k2 += _mm(dsh, dsl, qh, ql, SPLIT)
                    dh_, dl_ = _split(_load(pdo, sd_t, om, mm_, 3, oc, DC), SPLIT)
                    v3 += _mm(pth, ptl, dh_, dl_, SPLIT)
                    qh, ql = _split(_load(pq, sq_t, om, mm_, 3, oc, DC), SPLIT)
                    k3 += _mm(dsh, dsl, qh, ql, SPLIT)
        pdk = DK + b.to(tl.int64) * sdk_b + hk * sdk_h + on[:, None].to(tl.int64) * sdk_t + oc[None, :]
        pdv = DV + b.to(tl.int64) * sdk_b + hk * sdk_h + on[:, None].to(tl.int64) * sdk_t + oc[None, :]
        fk = scale * inv
        tl.store(pdk, k0 * fk, mask=mn[:, None])
        tl.store(pdk + DC, k1 * fk, mask=mn[:, None])
        tl.store(pdv, v0 * inv, mask=mn[:, None])
        tl.store(pdv + DC, v1 * inv, mask=mn[:, None])
        if NC == 4:
            tl.store(pdk + 2 * DC, k2 * fk, mask=mn[:, None])
            tl.store(pdk + 3 * DC, k3 * fk, mask=mn[:, None])
            tl.store(pdv + 2 * DC, v2 * inv, mask=mn[:, None])
            tl.store(pdv + 3 * DC, v3 * inv, mask=mn[:, None])

    @triton.jit
    def _fa_bwd_dq(Q, K, V, DO, LSE, DEL, SC, DQ, LIM,
                   sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sd_b, sd_h, sd_t, sdq_b, sdq_h, sdq_t,
                   H, T, scale,
                   GRP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
        """dQ[m] = scale sum_n dS[m, n] k[n]."""
        DC: tl.constexpr = 64
        NC: tl.constexpr = D // 64
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        hk = h // GRP
        om = pid_m * BM + tl.arange(0, BM)
        mm_ = om < T
        oc = tl.arange(0, DC)
        pq = Q + b.to(tl.int64) * sq_b + h * sq_h
        pdo = DO + b.to(tl.int64) * sd_b + h * sd_h
        pk = K + b.to(tl.int64) * sk_b + hk * sk_h
        pv = V + b.to(tl.int64) * sv_b + hk * sv_h
        lse = tl.load(LSE + bh.to(tl.int64) * T + om, mask=mm_, other=0.0)
        dl = tl.load(DEL + bh.to(tl.int64) * T + om, mask=mm_, other=0.0)
        inv = 1.0 / tl.load(SC + b)
        q0 = tl.zeros((BM, DC), tl.float32)
        q1 = tl.zeros((BM, DC), tl.float32)
        q2 = tl.zeros((BM, DC), tl.float32)
        q3 = tl.zeros((BM, DC), tl.float32)
        end = tl.load(LIM + pid_m)
        for n0 in range(0, end, BN):
            on = n0 + tl.arange(0, BN)
            mn = on < T
            s = _qk(pq, sq_t, om, mm_, pk, sk_t, on, mn, oc, SPLIT, NC, DC) * scale
            p = tl.exp(s - lse[:, None])
            p = tl.where((on[None, :] <= om[:, None]) & mm_[:, None] & mn[None, :], p, 0.0)
            dp = _qk(pdo, sd_t, om, mm_, pv, sv_t, on, mn, oc, SPLIT, NC, DC)
            ds = p * (dp - dl[:, None])
            dsh, dsl = _split(ds, SPLIT)
            kh, kl = _split(_load(pk, sk_t, on, mn, 0, oc, DC), SPLIT)
            q0 += _mm(dsh, dsl, kh, kl, SPLIT)
            kh, kl = _split(_load(pk, sk_t, on, mn, 1, oc, DC), SPLIT)
            q1 += _mm(dsh, dsl, kh, kl, SPLIT)
            if NC == 4:
                kh, kl = _split(_load(pk, sk_t, on, mn, 2, oc, DC), SPLIT)
                q2 += _mm(dsh, dsl, kh, kl, SPLIT)
                kh, kl = _split(_load(pk, sk_t, on, mn, 3, oc, DC), SPLIT)
                q3 += _mm(dsh, dsl, kh, kl, SPLIT)
        pdq = DQ + b.to(tl.int64) * sdq_b + h * sdq_h + om[:, None].to(tl.int64) * sdq_t + oc[None, :]
        f = scale * inv
        tl.store(pdq, q0 * f, mask=mm_[:, None])
        tl.store(pdq + DC, q1 * f, mask=mm_[:, None])
        if NC == 4:
            tl.store(pdq + 2 * DC, q2 * f, mask=mm_[:, None])
            tl.store(pdq + 3 * DC, q3 * f, mask=mm_[:, None])


CFG = {"BM": int(os.getenv("E2E_FLASH_BM", "32")), "BN": int(os.getenv("E2E_FLASH_BN", "32")),
       "warps": int(os.getenv("E2E_FLASH_WARPS", "4")), "stages": int(os.getenv("E2E_FLASH_STAGES", "1"))}


def _scale_pot2(m: torch.Tensor) -> torch.Tensor:
    """m >= 0 -> 2^-e with m 2^-e in [1/2, 1) (1 where m is 0 or not finite), exact."""
    _, e = torch.frexp(m)
    ok = (m > 0) & torch.isfinite(m)
    e = torch.where(ok, e, torch.zeros_like(e)).clamp_(-125, 125)
    return torch.ldexp(torch.ones_like(m, dtype=torch.float32), -e)


class FlashQAT(torch.autograd.Function):
    """o (B, T, H, D) fp32 = causal softmax(q k^T scale) v with GQA (H = GRP * HK); q (B, H, T, D), k, v (B, HK, T, D)."""

    @staticmethod
    def forward(ctx, q, k, v, scale: float, split: int):
        B, H, T, D = q.shape
        HK = k.shape[1]
        assert H % HK == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1 and D in (128, 256)
        o = torch.empty(B, T, H, D, dtype=torch.float32, device=q.device)
        lse = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
        BM, BN = CFG["BM"], CFG["BN"]
        lim = ((torch.arange(triton.cdiv(T, BM), device=q.device, dtype=torch.int32) + 1) * BM).contiguous()
        _fa_fwd[(triton.cdiv(T, BM), B * H)](q, k, v, o, lse, lim, q.stride(0), q.stride(1), q.stride(2),
                                            k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
                                            o.stride(0), o.stride(2), o.stride(1), H, T, scale,
                                            H // HK, D, BM, BN, split, num_warps=CFG["warps"], num_stages=CFG["stages"])
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale, ctx.split = scale, split
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        HK = k.shape[1]
        do = do.to(torch.float32)
        sc = _scale_pot2(__import__("fucina.kernels.qat", fromlist=("amax_abs",)).amax_abs(do, dim=(1, 2, 3)))                                    # (B,) one per batch row
        do_s = (do * sc.view(B, 1, 1, 1)).contiguous()                                    # (B, T, H, D), exact
        delta = (do_s * o).sum(-1).transpose(1, 2).contiguous()                            # (B, H, T) fp32
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.empty(B, HK, T, D, dtype=torch.float32, device=q.device)
        dv = torch.empty_like(dk)
        BM, BN = CFG["BM"], CFG["BN"]
        lim = ((torch.arange(triton.cdiv(T, BM), device=q.device, dtype=torch.int32) + 1) * BM).contiguous()
        limn = ((torch.arange(triton.cdiv(T, BN), device=q.device, dtype=torch.int32) * BN // BM) * BM).contiguous()
        _fa_bwd_dkdv[(triton.cdiv(T, BN), B * HK)](q, k, v, do_s, lse, delta, sc, dk, dv, limn,
                                                  q.stride(0), q.stride(1), q.stride(2),
                                                  k.stride(0), k.stride(1), k.stride(2),
                                                  v.stride(0), v.stride(1), v.stride(2),
                                                  do_s.stride(0), do_s.stride(2), do_s.stride(1),
                                                  dk.stride(0), dk.stride(1), dk.stride(2), H, HK, T, ctx.scale,
                                                  H // HK, D, BM, BN, ctx.split, num_warps=CFG["warps"],
                                                  num_stages=CFG["stages"])
        _fa_bwd_dq[(triton.cdiv(T, BM), B * H)](q, k, v, do_s, lse, delta, sc, dq, lim,
                                               q.stride(0), q.stride(1), q.stride(2),
                                               k.stride(0), k.stride(1), k.stride(2),
                                               v.stride(0), v.stride(1), v.stride(2),
                                               do_s.stride(0), do_s.stride(2), do_s.stride(1),
                                               dq.stride(0), dq.stride(1), dq.stride(2), H, T, ctx.scale,
                                               H // HK, D, BM, BN, ctx.split, num_warps=CFG["warps"],
                                               num_stages=CFG["stages"])
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def attention(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None, **kw):
    """transformers attention interface. flash1/flash3 on: causal flash (the student's mask is the plain causal one,
    both in row mode and per segment); off: transformers' own SDPA, exactly the usual call."""
    from fucina.kernels import qat as KQ
    if (KQ.active("attn3") or KQ.active("attn1")) and not dropout:          # query blocks + split GEMMs
        sc = float(scaling if scaling is not None else query.shape[-1] ** -0.5)
        if os.getenv("E2E_ATTN_V", "3") == "3":
            o = AttnBlocks3.apply(query, key, value, sc, 3 if KQ.active("attn3") else 1)
        elif KQ.active("attn3") and os.getenv("E2E_ATTN_V") == "2":
            o = AttnBlocks2.apply(query, key, value, sc)
        else:
            o = AttnBlocks.apply(query, key, value, sc, 3 if KQ.active("attn3") else 1)
        return o.to(query.dtype), None
    if not (KQ.active("flash3") or KQ.active("flash1")) or dropout:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    split = 3 if KQ.active("flash3") else 1
    sc = float(scaling if scaling is not None else query.shape[-1] ** -0.5)
    o = FlashQAT.apply(query, key, value, sc, split)
    return o.to(query.dtype), None


def install(cfg=None) -> None:
    """Registers `kqat_flash` in transformers (falls back to SDPA when the switch is off) and points cfg at it."""
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("kqat_flash", attention)
    if cfg is not None:
        cfg._attn_implementation = "kqat_flash"


# ══════════════ causal attention by query blocks with split GEMMs (hipBLASLt), E2E_KQAT=attn3 / attn1 ══════════════
# The Triton flash kernels above spill hundreds of VGPRs on gfx1151 at head dim 256 (AOT: even a minimal FA at d=64
# spills; RDNA3 WMMA operands are replicated across the half-waves and Triton keeps them in registers). This variant
# keeps FlashAttention-2's arithmetic structure (log-sum-exp saved, scores recomputed in the backward, Delta) but at
# the granularity of query BLOCKS of QB rows, with every product a batched GEMM over the 16 heads (GQA: the 8 heads
# of a kv head share K/V) through gdn_qat.mm_split: fp16 split operands (hi + lo 2^-11) with fp32 accumulation
# (attn3) or one fp16 product (attn1). Causality: query block i only multiplies keys 0 .. (i+1) QB - 1 (the strictly
# upper blocks are never computed: 1 - (n+1)/(2n) of the full product is saved, 44 % for 8 blocks).
QB = int(os.getenv("E2E_ATTN_QB", "256"))   # measured 2026-10-09: v3 QB 256 296 ms, 512 305, 1024 386


def _mms(a, b, split):
    import importlib
    GQ = importlib.import_module("fucina.kernels.gdn_qat")
    return GQ.mm_split(a, b, split)


class AttnBlocks(torch.autograd.Function):
    """o (B, T, H, D) fp32 = causal softmax(q k^T scale) v; q (B, H, T, D), k, v (B, HK, T, D)."""

    @staticmethod
    def forward(ctx, q, k, v, scale: float, split: int):
        B, H, T, D = q.shape
        HK = k.shape[1]
        G = H // HK
        q32, k32, v32 = q.float(), k.float(), v.float()
        qg = q32.reshape(B, HK, G, T, D)                                      # heads of one kv head together
        o = torch.empty(B, HK, G, T, D, dtype=torch.float32, device=q.device)
        lse = torch.empty(B, HK, G, T, dtype=torch.float32, device=q.device)
        for i0 in range(0, T, QB):
            i1 = min(T, i0 + QB)
            qb = qg[:, :, :, i0:i1].reshape(B * HK, G * (i1 - i0), D)
            kb = k32[:, :, :i1].reshape(B * HK, i1, D)
            vb = v32[:, :, :i1].reshape(B * HK, i1, D)
            s = _mms(qb, kb.transpose(1, 2), split).mul_(scale).view(B * HK, G, i1 - i0, i1)
            mask = torch.ones(i1 - i0, i1, dtype=torch.bool, device=q.device).tril_(i0)
            s.masked_fill_(~mask, float("-inf"))
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            l_ = p.sum(-1, keepdim=True)
            p.div_(l_)
            ob = _mms(p.view(B * HK, G * (i1 - i0), i1), vb, split)
            o[:, :, :, i0:i1] = ob.view(B, HK, G, i1 - i0, D)
            lse[:, :, :, i0:i1] = (m + torch.log(l_)).view(B, HK, G, i1 - i0)
            del s, p
        ctx.save_for_backward(q32, k32, v32, o, lse)
        ctx.scale, ctx.split = scale, split
        return o.reshape(B, H, T, D).transpose(1, 2)

    @staticmethod
    def backward(ctx, do):
        q32, k32, v32, o, lse = ctx.saved_tensors
        scale, split = ctx.scale, ctx.split
        B, H, T, D = q32.shape
        HK = k32.shape[1]
        G = H // HK
        dog = do.float().transpose(1, 2).reshape(B, HK, G, T, D)
        sc = _scale_pot2(__import__("fucina.kernels.qat", fromlist=("amax_abs",)).amax_abs(dog, dim=(1, 2, 3, 4)))                          # one power of two per batch row
        dog = dog * sc.view(B, 1, 1, 1, 1)
        delta = (dog * o).sum(-1)                                                    # (B, HK, G, T)
        qg = q32.reshape(B, HK, G, T, D)
        dq = torch.empty_like(qg)
        dk = torch.zeros_like(k32)
        dv = torch.zeros_like(v32)
        for i0 in range(0, T, QB):
            i1 = min(T, i0 + QB)
            n = i1 - i0
            qb = qg[:, :, :, i0:i1].reshape(B * HK, G * n, D)
            kb = k32[:, :, :i1].reshape(B * HK, i1, D)
            vb = v32[:, :, :i1].reshape(B * HK, i1, D)
            dob = dog[:, :, :, i0:i1].reshape(B * HK, G * n, D)
            s = _mms(qb, kb.transpose(1, 2), split).mul_(scale).view(B * HK, G, n, i1)
            mask = torch.ones(n, i1, dtype=torch.bool, device=q32.device).tril_(i0)
            p = torch.exp(s - lse[:, :, :, i0:i1].reshape(B * HK, G, n, 1)).masked_fill_(~mask, 0.0)
            del s
            p2 = p.view(B * HK, G * n, i1)
            dv[:, :, :i1] += _mms(p2.transpose(1, 2), dob, split).view(B, HK, i1, D)
            dp = _mms(dob, vb.transpose(1, 2), split).view(B * HK, G, n, i1)
            ds = p.mul_(dp.sub_(delta[:, :, :, i0:i1].reshape(B * HK, G, n, 1))).view(B * HK, G * n, i1)
            del dp
            dq[:, :, :, i0:i1] = _mms(ds, kb, split).mul_(scale).view(B, HK, G, n, D)
            dk[:, :, :i1] += _mms(ds.transpose(1, 2), qb, split).mul_(scale).view(B, HK, i1, D)
            del ds, p
        inv = (1.0 / sc)
        dq = dq.mul_(inv.view(B, 1, 1, 1, 1)).reshape(B, H, T, D)
        dk.mul_(inv.view(B, 1, 1, 1))
        dv.mul_(inv.view(B, 1, 1, 1))
        return dq.to(q32.dtype), dk, dv, None, None


# ═══════════ attn3 v2: batched products with the fp32 operand split IN REGISTERS (no hi/lo copies of P, dS) ═══════════
# Measured 2026-10-09: AttnBlocks (mm_split on materialised hi/lo copies) 979 ms against 745 of the fp32 SDPA: the scaled /
# split copies of the 4096 x 4096 probability and dS blocks cost more than the products. Here the big operand A
# (P, P^T, dS, dS^T: fp32, any strides) is split in registers, hi + lo 2^-11 (Ootomo-Yokota), the small one (Q, K, V,
# dO: (B*HK, T, 256)) is split ONCE per call; three fp16 products per K step, each BK = 32 chunk added to the fp32
# accumulator with a rounded add (runtime-zero accumulator: no truncating WMMA chain along K).
if _TRITON:
    @triton.jit
    def _k_bmmx(a_ptr, sa_b, sa_m, sa_k, bh_ptr, bl_ptr, sb_b, sb_k, sb_n, c_ptr, sc_b, sc_m, sc_n, M, N, K, zero,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        bb = tl.program_id(1).to(tl.int64)
        npn = tl.cdiv(N, BN)
        pm = pid // npn
        pn = pid % npn
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm_ = (rm < M)[:, None]
        mn = (rn < N)[None, :]
        z = tl.zeros((BM, BN), dtype=tl.float32) * zero
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        accx = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            mk = kk < K
            a = tl.load(a_ptr + bb * sa_b + rm[:, None].to(tl.int64) * sa_m + kk[None, :].to(tl.int64) * sa_k,
                        mask=mm_ & mk[None, :], other=0.0).to(tl.float32)
            ah = a.to(tl.float16)
            al = ((a - ah.to(tl.float32)) * 2048.0).to(tl.float16)
            boff = bb * sb_b + kk[:, None].to(tl.int64) * sb_k + rn[None, :].to(tl.int64) * sb_n
            bh = tl.load(bh_ptr + boff, mask=mk[:, None] & mn, other=0.0)
            bl = tl.load(bl_ptr + boff, mask=mk[:, None] & mn, other=0.0)
            acc += tl.dot(ah, bh, z)
            accx += tl.dot(ah, bl, tl.dot(al, bh, z))
        c = acc + accx * (1.0 / 2048.0)
        tl.store(c_ptr + bb * sc_b + rm[:, None].to(tl.int64) * sc_m + rn[None, :].to(tl.int64) * sc_n, c,
                 mask=mm_ & mn)


def _split_hi_lo(b: torch.Tensor):
    """(hi, lo 2^11) fp16 of a fp32 tensor whose values fit fp16 (|.| < 65504); exact representation to ~2^-22."""
    h = b.to(torch.float16)
    return h, ((b - h.float()) * 2048.0).to(torch.float16)


def bmmx(a: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor) -> torch.Tensor:
    """a (Bt, M, K) fp32 (any strides) @ (bh + bl 2^-11) (Bt, K, N) fp16 -> (Bt, M, N) fp32."""
    Bt, M, K = a.shape
    N = bh.shape[2]
    assert bh.stride() == bl.stride()
    c = torch.empty(Bt, M, N, dtype=torch.float32, device=a.device)
    BM, BN, BK, w = CFG_BMMX
    _k_bmmx[(triton.cdiv(M, BM) * triton.cdiv(N, BN), Bt)](
        a, a.stride(0), a.stride(1), a.stride(2), bh, bl, bh.stride(0), bh.stride(1), bh.stride(2),
        c, c.stride(0), c.stride(1), c.stride(2), M, N, K, 0.0, BM, BN, BK, num_warps=w)
    return c


CFG_BMMX = tuple(int(x) for x in os.getenv("E2E_ATTN_BMMX", "64,64,32,8").split(","))


class AttnBlocks2(torch.autograd.Function):
    """Same mathematics as AttnBlocks (attn3 / attn1 tiers), products through bmmx (split 3 in registers); split 1
    (attn1) keeps AttnBlocks. o (B, T, H, D); q (B, H, T, D), k, v (B, HK, T, D)."""

    @staticmethod
    def forward(ctx, q, k, v, scale: float):
        B, H, T, D = q.shape
        HK = k.shape[1]
        G = H // HK
        q32, k32, v32 = q.float(), k.float(), v.float()
        qg = q32.reshape(B * HK, G, T, D)
        kh, kl = _split_hi_lo(k32.reshape(B * HK, T, D).transpose(1, 2).contiguous())     # (BH, D, T)
        vh, vl = _split_hi_lo(v32.reshape(B * HK, T, D).contiguous())                       # (BH, T, D)
        o = torch.empty(B * HK, G, T, D, dtype=torch.float32, device=q.device)
        lse = torch.empty(B * HK, G, T, dtype=torch.float32, device=q.device)
        for i0 in range(0, T, QB):
            i1 = min(T, i0 + QB)
            n = i1 - i0
            qb = qg[:, :, i0:i1].reshape(B * HK, G * n, D)
            s = bmmx(qb, kh[:, :, :i1], kl[:, :, :i1]).mul_(scale).view(B * HK, G, n, i1)
            mask = torch.ones(n, i1, dtype=torch.bool, device=q.device).tril_(i0)
            s.masked_fill_(~mask, float("-inf"))
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s.sub_(m))
            l_ = p.sum(-1, keepdim=True)
            p.div_(l_)
            o[:, :, i0:i1] = bmmx(p.view(B * HK, G * n, i1), vh[:, :i1], vl[:, :i1]).view(B * HK, G, n, D)
            lse[:, :, i0:i1] = (m + torch.log(l_)).view(B * HK, G, n)
            del s, p
        ctx.save_for_backward(q32, k32, v32, o, lse)
        ctx.scale = scale
        return o.view(B, H, T, D).transpose(1, 2)

    @staticmethod
    def backward(ctx, do):
        q32, k32, v32, o, lse = ctx.saved_tensors
        scale = ctx.scale
        B, H, T, D = q32.shape
        HK = k32.shape[1]
        G = H // HK
        BH = B * HK
        dog = do.float().transpose(1, 2).reshape(BH, G, T, D)
        sc = _scale_pot2(dog.abs().reshape(B, -1).amax(1))                           # one power of two per row b
        dog = dog * sc.repeat_interleave(HK).view(BH, 1, 1, 1)
        delta = (dog * o).sum(-1)                                                    # (BH, G, T)
        qg = q32.reshape(BH, G, T, D)
        kt = k32.reshape(BH, T, D)
        kh, kl = _split_hi_lo(kt.transpose(1, 2).contiguous())                              # K^T (BH, D, T)
        kh2, kl2 = _split_hi_lo(kt.contiguous())                                            # K   (BH, T, D)
        vh, vl = _split_hi_lo(v32.reshape(BH, T, D).transpose(1, 2).contiguous())           # V^T (BH, D, T)
        dq = torch.empty(BH, G, T, D, dtype=torch.float32, device=q32.device)
        dk = torch.zeros(BH, T, D, dtype=torch.float32, device=q32.device)
        dv = torch.zeros(BH, T, D, dtype=torch.float32, device=q32.device)
        for i0 in range(0, T, QB):
            i1 = min(T, i0 + QB)
            n = i1 - i0
            qb = qg[:, :, i0:i1].reshape(BH, G * n, D)
            dob = dog[:, :, i0:i1].reshape(BH, G * n, D)
            qh, ql = _split_hi_lo(qb.contiguous())
            doh, dol = _split_hi_lo(dob.contiguous())
            s = bmmx(qb, kh[:, :, :i1], kl[:, :, :i1]).mul_(scale).view(BH, G, n, i1)
            mask = torch.ones(n, i1, dtype=torch.bool, device=q32.device).tril_(i0)
            p = torch.exp(s.sub_(lse[:, :, i0:i1].unsqueeze(-1))).masked_fill_(~mask, 0.0)
            del s
            p2 = p.view(BH, G * n, i1)
            dv[:, :i1] += bmmx(p2.transpose(1, 2), doh, dol)
            dp = bmmx(dob, vh[:, :, :i1], vl[:, :, :i1]).view(BH, G, n, i1)
            ds = p.mul_(dp.sub_(delta[:, :, i0:i1].unsqueeze(-1))).view(BH, G * n, i1)
            del dp
            dq[:, :, i0:i1] = bmmx(ds, kh2[:, :i1], kl2[:, :i1]).mul_(scale).view(BH, G, n, D)
            dk[:, :i1] += bmmx(ds.transpose(1, 2), qh, ql).mul_(scale)
            del ds, p
        inv = (1.0 / sc).repeat_interleave(HK)
        dq = dq.mul_(inv.view(BH, 1, 1, 1)).reshape(B, H, T, D)
        dk = dk.mul_(inv.view(BH, 1, 1)).view(B, HK, T, D)
        dv = dv.mul_(inv.view(BH, 1, 1)).view(B, HK, T, D)
        return dq, dk, dv, None


# ═══════════ fused causal softmax (forward) and dS (backward) for the blocked attention, E2E_ATTN_FUSO=1 ═══════════
# The blocked attention spent most of its time in 8-10 elementwise passes over the (G n) x i1 score blocks (2.4 GB per
# pass at T 4096). One program per score row: the row (<= 4096 fp32) is read ONCE, the probabilities written once.
# Same operations in the same order per element as the torch sequence (scale, mask, max, exp(s - m), sum, divide /
# exp(s - lse), mask, p (dp - delta)); the row sum is a tree reduction instead of torch's (rounding level, inside the
# attn3 tolerance: test F).
if _TRITON:
    @triton.jit
    def _k_smax(s_ptr, p_ptr, lse_ptr, NR, n, i0, i1, scale, BL: tl.constexpr):
        """rows r of s (NR = G n rows of length i1, row r <-> query i0 + r % n); p = softmax(scale s) causal."""
        r = tl.program_id(0)
        q = i0 + r % n
        j = tl.arange(0, BL)
        m_ok = (j < i1) & (j <= q)
        base = r.to(tl.int64) * i1
        s = tl.load(s_ptr + base + j, mask=j < i1, other=0.0) * scale
        s = tl.where(m_ok, s, float("-inf"))
        m = tl.max(s, 0)
        e = tl.exp(s - m)
        l = tl.sum(e, 0)
        tl.store(p_ptr + base + j, e / l, mask=j < i1)
        tl.store(lse_ptr + r, m + tl.log(l))

    @triton.jit
    def _k_ds(s_ptr, dp_ptr, lse_ptr, of_ptr, p_ptr, ds_ptr, n, i0, i1, scale, BL: tl.constexpr):
        """p = exp(scale s - lse) (causal), ds = p (dp - delta); writes p and ds (both read by the next products)."""
        r = tl.program_id(0)
        q = i0 + r % n
        j = tl.arange(0, BL)
        mj = j < i1
        base = r.to(tl.int64) * i1
        s = tl.load(s_ptr + base + j, mask=mj, other=0.0) * scale
        p = tl.exp(s - tl.load(lse_ptr + r))
        p = tl.where(mj & (j <= q), p, 0.0)
        dp = tl.load(dp_ptr + base + j, mask=mj, other=0.0)
        ds = p * (dp - tl.load(of_ptr + r))
        tl.store(p_ptr + base + j, p, mask=mj)
        tl.store(ds_ptr + base + j, ds, mask=mj)


class AttnBlocks3(torch.autograd.Function):
    """AttnBlocks2 with the fused row kernels (E2E_ATTN_V=3): products bmmx (split 3 in registers) or one fp16
    product (split 1, attn1); softmax and dS in one pass each."""

    @staticmethod
    def forward(ctx, q, k, v, scale: float, split: int):
        B, H, T, D = q.shape
        HK = k.shape[1]
        G = H // HK
        BH = B * HK
        q32, k32, v32 = q.float(), k.float(), v.float()
        qg = q32.reshape(BH, G, T, D)
        kh, kl = _split_hi_lo(k32.reshape(BH, T, D).transpose(1, 2).contiguous())
        vh, vl = _split_hi_lo(v32.reshape(BH, T, D).contiguous())
        o = torch.empty(BH, G, T, D, dtype=torch.float32, device=q.device)
        lse = torch.empty(BH, G, T, dtype=torch.float32, device=q.device)
        for i0 in range(0, T, QB):
            i1 = min(T, i0 + QB)
            n = i1 - i0
            qb = qg[:, :, i0:i1].reshape(BH, G * n, D)
            s = _prod(qb, kh[:, :, :i1], kl[:, :, :i1], split)                    # (BH, G n, i1)
            p = torch.empty_like(s)
            lb = torch.empty(BH, G * n, dtype=torch.float32, device=q.device)
            _k_smax[(BH * G * n,)](s, p, lb, BH * G * n, n, i0, i1, scale, triton.next_power_of_2(i1), num_warps=8)
            del s
            o[:, :, i0:i1] = _prod(p, vh[:, :i1], vl[:, :i1], split).view(BH, G, n, D)
            lse[:, :, i0:i1] = lb.view(BH, G, n)
            del p
        ctx.save_for_backward(q32, k32, v32, o, lse)
        ctx.scale, ctx.split = scale, split
        return o.view(B, H, T, D).transpose(1, 2)

    @staticmethod
    def backward(ctx, do):
        q32, k32, v32, o, lse = ctx.saved_tensors
        scale, split = ctx.scale, ctx.split
        B, H, T, D = q32.shape
        HK = k32.shape[1]
        G = H // HK
        BH = B * HK
        dog = do.float().transpose(1, 2).reshape(BH, G, T, D)
        sc = _scale_pot2(torch.linalg.vector_norm(dog.reshape(B, -1), float("inf"), 1))
        dog = dog * sc.repeat_interleave(HK).view(BH, 1, 1, 1)
        delta = (dog * o).sum(-1)
        qg = q32.reshape(BH, G, T, D)
        kt = k32.reshape(BH, T, D)
        kh, kl = _split_hi_lo(kt.transpose(1, 2).contiguous())
        kh2, kl2 = _split_hi_lo(kt.contiguous())
        vh, vl = _split_hi_lo(v32.reshape(BH, T, D).transpose(1, 2).contiguous())
        dq = torch.empty(BH, G, T, D, dtype=torch.float32, device=q32.device)
        dk = torch.zeros(BH, T, D, dtype=torch.float32, device=q32.device)
        dv = torch.zeros(BH, T, D, dtype=torch.float32, device=q32.device)
        for i0 in range(0, T, QB):
            i1 = min(T, i0 + QB)
            n = i1 - i0
            qb = qg[:, :, i0:i1].reshape(BH, G * n, D)
            dob = dog[:, :, i0:i1].reshape(BH, G * n, D)
            qh, ql = _split_hi_lo(qb.contiguous())
            doh, dol = _split_hi_lo(dob.contiguous())
            s = _prod(qb, kh[:, :, :i1], kl[:, :, :i1], split)
            dp = _prod(dob, vh[:, :, :i1], vl[:, :, :i1], split)
            p = torch.empty_like(s)
            ds = torch.empty_like(s)
            _k_ds[(BH * G * n,)](s, dp, lse[:, :, i0:i1].contiguous(), delta[:, :, i0:i1].contiguous(), p, ds, n, i0,
                                 i1, scale, triton.next_power_of_2(i1), num_warps=8)
            del s, dp
            dv[:, :i1] += _prod(p.transpose(1, 2), doh, dol, split)
            del p
            dq[:, :, i0:i1] = _prod(ds, kh2[:, :i1], kl2[:, :i1], split).mul_(scale).view(BH, G, n, D)
            dk[:, :i1] += _prod(ds.transpose(1, 2), qh, ql, split).mul_(scale)
            del ds
        inv = (1.0 / sc).repeat_interleave(HK)
        dq = dq.mul_(inv.view(BH, 1, 1, 1)).reshape(B, H, T, D)
        dk = dk.mul_(inv.view(BH, 1, 1)).view(B, HK, T, D)
        dv = dv.mul_(inv.view(BH, 1, 1)).view(B, HK, T, D)
        return dq, dk, dv, None, None


def _prod(a: torch.Tensor, bh: torch.Tensor, bl: torch.Tensor, split: int) -> torch.Tensor:
    """a fp32 (Bt, M, K) @ (bh [+ bl 2^-11]) -> fp32: split 3 = bmmx (hi/lo of a in registers), split 1 = one fp16
    product of fp16(a) and bh (Triton _k_bmm16, chunk-wise rounded accumulation)."""
    if split == 3:
        return bmmx(a, bh, bl)
    import importlib
    GQ = importlib.import_module("fucina.kernels.gdn_qat")
    sa = _scale_pot2(torch.linalg.vector_norm(a.reshape(a.shape[0], -1), float("inf"), 1)).view(-1, 1, 1)
    return GQ._bmm16((a * sa).to(torch.float16), bh).div_(sa)        # power-of-two scale per batch: exact
