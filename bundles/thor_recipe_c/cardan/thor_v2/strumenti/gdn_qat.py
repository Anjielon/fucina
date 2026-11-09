#!/usr/bin/env python3
"""gdn_qat — the chunked gated delta rule of the 30 Gated DeltaNet layers with every product on the WMMA units of
gfx1151 (9/10/2026, new file; report ~/tony-forgia/ricerche/KERNEL_QAT_SOTTO_3H.md). Select it with
E2E_GDN=veloce E2E_GDN_MODULO=gdn_qat and E2E_KQAT containing `gdn3` (fp32-accurate) or `gdn1` (fp16 operands);
without those flags every call goes to gdn_veloce_corsa_v5 (the version in service), unchanged.

Why: v5 does the same algebra in fp32 (`torch.matmul` / `baddbmm` / Triton `tl.dot` fp32): on gfx1151 fp32 products
have no matrix units (FMA, ~3 TFLOPS) — ~14 us/token per layer, ~0.43 ms/token over 30 layers (measure of 27/9).

How (same algebra as v5, step by step; the triangular inverse stays fp32 `solve_triangular`):
  * every batched product a @ b goes through `MM3`: each operand scaled by a power of two per matrix (max |.| in
    [1/2, 1), exact), split a = a_hi + a_lo in fp16 (gdn3) and a b ~ a_hi b_lo + a_lo b_hi + a_hi b_hi with fp16
    products exact in fp32 and fp32 accumulation (hipBLASLt batched GEMM, out_dtype fp32); gdn1 keeps a_hi b_hi.
    Backward: d a = g b^T, d b = a^T g through the same split product.
  * the recurrence across chunks (vn_i = u_i - w_i S_i ; S_{i+1} = S_i egl_i + kw_i^T vn_i, and its adjoint) runs
    in two Triton kernels that keep a (Dk x BV) slice of the fp32 state in registers and split it at every step.
    Gradients enter the adjoint scaled by one power of two per head (the adjoint is linear in them).
Low parts are stored scaled by 2^11 (Ootomo & Yokota): a = hi + lo 2^-11 keeps ~22 significant bits for every element
with |a| >= 2^-14 of its matrix maximum (without the 2^11 the low part of small elements falls in the fp16 subnormals
and loses bits: measured 4e-6 instead of 6e-7 on dv); the dropped lo x lo term is <= 2^-22 relative.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

_QUI = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("gdn_veloce_corsa_v5_qat", os.path.join(_QUI, "gdn_veloce_corsa_v5.py"))
V5 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(V5)

try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False


def _split_n() -> int:
    import e2e_kernel_qat as KQ
    return 3 if KQ.attivo("gdn3") else (1 if KQ.attivo("gdn1") else 0)


def _pot2(m: torch.Tensor) -> torch.Tensor:
    """m >= 0 -> 2^-e with m 2^-e in [1/2, 1) (1 where m is 0 / not finite), exact."""
    _, e = torch.frexp(m)
    ok = (m > 0) & torch.isfinite(m)
    e = torch.where(ok, e, torch.zeros_like(e)).clamp_(-125, 125)
    return torch.ldexp(torch.ones_like(m, dtype=torch.float32), -e)


_BMM = {"via": os.getenv("E2E_GDN_QAT_BMM", "triton")}   # 9/10 measured: torch.bmm(out_dtype=fp32) 6-7 TFLOPS


def _bmm16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """fp16 x fp16 batched product with fp32 accumulation and output (exact products). blas: torch.bmm with
    out_dtype fp32 (hipBLASLt); triton: _k_bmm16 (also the fallback if the BLAS route is refused)."""
    if not x.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        return torch.bmm(x.float(), y.float())
    if _BMM["via"] == "blas" and x.is_cuda:
        try:
            return torch.bmm(x, y, out_dtype=torch.float32)
        except Exception as e:  # noqa: BLE001
            import warnings
            warnings.warn(f"gdn_qat: torch.bmm(out_dtype=fp32) refused ({type(e).__name__}: {str(e)[:120]}); Triton")
            _BMM["via"] = "triton"
    return bmm16_triton(x, y)


def mm_split(a: torch.Tensor, b: torch.Tensor, split: int) -> torch.Tensor:
    """a (..., M, K) @ b (..., K, N) in fp32 via fp16 products (see the module header)."""
    sh = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(*sh, *a.shape[-2:]).reshape(-1, *a.shape[-2:]).to(torch.float32)
    b = b.expand(*sh, *b.shape[-2:]).reshape(-1, *b.shape[-2:]).to(torch.float32)
    sa = _pot2(__import__("e2e_kernel_qat").amax_abs(a, (-1, -2), keepdim=True))
    sb = _pot2(__import__("e2e_kernel_qat").amax_abs(b, (-1, -2), keepdim=True))
    a = a * sa
    b = b * sb
    ah = a.to(torch.float16)
    bh = b.to(torch.float16)
    if split == 3:                                       # low parts scaled by 2^11 (Ootomo-Yokota), see _sp
        al = ((a - ah.float()) * 2048.0).to(torch.float16)
        bl = ((b - bh.float()) * 2048.0).to(torch.float16)
        c = _bmm16(ah, bl)
        c += _bmm16(al, bh)
        r = _bmm16(ah, bh)
        r += c * (1.0 / 2048.0)
    else:
        r = _bmm16(ah, bh)
    r = r / (sa * sb)                                   # powers of two: exact
    return r.reshape(*sh, r.shape[-2], r.shape[-1])


class MM3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, split: int):
        ctx.save_for_backward(a, b)
        ctx.split = split
        return mm_split(a, b, split)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        da = db = None
        if ctx.needs_input_grad[0]:
            da = mm_split(g, b.transpose(-1, -2), ctx.split)
            if da.shape != a.shape:
                da = da.sum_to_size(a.shape)
        if ctx.needs_input_grad[1]:
            db = mm_split(a.transpose(-1, -2), g, ctx.split)
            if db.shape != b.shape:
                db = db.sum_to_size(b.shape)
        return da, db, None


def mm(a, b, split):
    return MM3.apply(a, b, split)


class _Inversa(torch.autograd.Function):
    """X = (I + Lm)^-1 (Lm strictly lower, C x C batched): fp32 forward substitution (as v5); backward
    dLm = -(X^T dX X^T) restricted to the strictly lower part, the two products split."""

    @staticmethod
    def forward(ctx, Lm, split: int):
        C = Lm.shape[-1]
        eye = torch.eye(C, dtype=Lm.dtype, device=Lm.device).expand(Lm.shape)
        X = torch.linalg.solve_triangular(Lm, eye, upper=False, unitriangular=True)
        ctx.save_for_backward(X)
        ctx.split = split
        return X

    @staticmethod
    def backward(ctx, dX):
        (X,) = ctx.saved_tensors
        Xt = X.transpose(-1, -2)
        return mm_split(mm_split(Xt, dX, ctx.split), Xt, ctx.split).neg_().tril_(-1), None


if _TRITON:
    @triton.jit
    def _sp(a, SPLIT: tl.constexpr):
        """a = hi + lo 2^-11 (Ootomo-Yokota: the low part is stored scaled by 2^11, so it stays a NORMAL fp16 for
        every |a| >= 2^-14: 22 significant bits instead of losing them in the subnormal range)."""
        h = a.to(tl.float16)
        if SPLIT == 3:
            lo = ((a - h.to(tl.float32)) * 2048.0).to(tl.float16)
        else:
            lo = h
        return h, lo

    @triton.jit
    def _mm3(a, b, SPLIT: tl.constexpr):
        ah, al = _sp(a, SPLIT)
        bh, bl = _sp(b, SPLIT)
        if SPLIT == 3:
            c = tl.dot(ah, bl)
            c = tl.dot(al, bh, c)
            r = tl.dot(ah, bh) + c * (1.0 / 2048.0)
        else:
            r = tl.dot(ah, bh)
        return r

    @triton.jit
    def _mm3s(a, bh, bl, SPLIT: tl.constexpr):
        """a @ b with b already split (bh, bl)."""
        ah, al = _sp(a, SPLIT)
        if SPLIT == 3:
            c = tl.dot(ah, bl)
            c = tl.dot(al, bh, c)
            r = tl.dot(ah, bh) + c * (1.0 / 2048.0)
        else:
            r = tl.dot(ah, bh)
        return r

    @triton.jit
    def _k_ric_av(u_ptr, w_ptr, kw_ptr, egl_ptr, s0_ptr, vn_ptr, sa_ptr, BH, N: tl.constexpr,
                  C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, SPLIT: tl.constexpr):
        """v4 forward (gdn_veloce_corsa_v5._k_ric_avanti) with split fp16 products, the chunk processed in two halves
        of C/2 rows (vn rows are independent; kw^T vn is a sum over the rows): fewer live registers (AOT gfx1151)."""
        CH: tl.constexpr = C // 2
        bh = tl.program_id(0).to(tl.int64)
        iv = tl.program_id(1)
        rc = tl.arange(0, CH)
        rk = tl.arange(0, DK)
        rv = iv * BV + tl.arange(0, BV)
        off_s = rk[:, None] * DV + rv[None, :]
        off_ck = rc[:, None] * DK + rk[None, :]
        off_cv = rc[:, None] * DV + rv[None, :]
        S = tl.load(s0_ptr + bh * DK * DV + off_s)
        for i in range(0, N):                            # N constexpr: constant trip count (triton#11378)
            base = i * BH + bh
            tl.store(sa_ptr + base * DK * DV + off_s, S)
            Sh, Sl = _sp(S, SPLIT)
            dlt = tl.zeros((DK, BV), tl.float32)
            for hh in tl.static_range(2):
                o_k = base * C * DK + hh * CH * DK + off_ck
                o_v = base * C * DV + hh * CH * DV + off_cv
                vni = tl.load(u_ptr + o_v) - _mm3s(tl.load(w_ptr + o_k), Sh, Sl, SPLIT)
                tl.store(vn_ptr + o_v, vni)
                vh, vl = _sp(vni, SPLIT)
                dlt += _mm3s(tl.trans(tl.load(kw_ptr + o_k)), vh, vl, SPLIT)
            S = S * tl.load(egl_ptr + base) + dlt
        tl.store(sa_ptr + (N * BH + bh) * DK * DV + off_s, S)

    @triton.jit
    def _k_ric_ind(dvn_ptr, dsa_ptr, w_ptr, kw_ptr, egl_ptr, BH, N: tl.constexpr,
                   C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, SPLIT: tl.constexpr):
        """v4 adjoint, in place, split products, two halves of C/2 rows (inputs scaled by a power of two per head)."""
        CH: tl.constexpr = C // 2
        bh = tl.program_id(0).to(tl.int64)
        iv = tl.program_id(1)
        rc = tl.arange(0, CH)
        rk = tl.arange(0, DK)
        rv = iv * BV + tl.arange(0, BV)
        off_s = rk[:, None] * DV + rv[None, :]
        off_ck = rc[:, None] * DK + rk[None, :]
        off_cv = rc[:, None] * DV + rv[None, :]
        dS = tl.load(dsa_ptr + (N * BH + bh) * DK * DV + off_s)
        for j in range(0, N):
            base = (N - 1 - j) * BH + bh
            Dh, Dl = _sp(dS, SPLIT)
            acc = tl.zeros((DK, BV), tl.float32)
            for hh in tl.static_range(2):
                o_k = base * C * DK + hh * CH * DK + off_ck
                o_v = base * C * DV + hh * CH * DV + off_cv
                dvni = tl.load(dvn_ptr + o_v) + _mm3s(tl.load(kw_ptr + o_k), Dh, Dl, SPLIT)
                tl.store(dvn_ptr + o_v, dvni)
                vh, vl = _sp(dvni, SPLIT)
                acc += _mm3s(tl.trans(tl.load(w_ptr + o_k)), vh, vl, SPLIT)
            p_ds = dsa_ptr + base * DK * DV + off_s
            dS = dS * tl.load(egl_ptr + base) + tl.load(p_ds) - acc
            tl.store(p_ds, dS)


if _TRITON:
    @triton.jit
    def _k_bmm16(a_ptr, sa_b, sa_m, sa_k, b_ptr, sb_b, sb_k, sb_n, c_ptr, sc_b, sc_m, sc_n, M, N, K, zero,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """c[b] = a[b] @ b[b], fp16 operands, fp32 accumulation/output; grid (tiles of M x N, batch).
        The WMMA accumulation of gfx1151 TRUNCATES: a K-long chain of `acc += dot` drifts linearly in K (measured on
        the attention: dk/dv 1.3-1.6e-5 at K = 4096, ten times the fp32 SDPA). Here every BK-step product starts from
        a RUNTIME zero (`zero` = 0.0 passed by the caller: Triton cannot fold the add into the WMMA accumulator) and
        is added to the fp32 accumulator by an ordinary rounded fp32 add: the truncation is confined to the 2 WMMA
        steps of one BK = 32 chunk."""
        pid = tl.program_id(0)
        bb = tl.program_id(1).to(tl.int64)
        npn = tl.cdiv(N, BN)
        pm = pid // npn
        pn = pid % npn
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        z = tl.zeros((BM, BN), dtype=tl.float32) * zero
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(a_ptr + bb * sa_b + rm[:, None] * sa_m + kk[None, :] * sa_k,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(b_ptr + bb * sb_b + kk[:, None] * sb_k + rn[None, :] * sb_n,
                        mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b, z)
        tl.store(c_ptr + bb * sc_b + rm[:, None] * sc_m + rn[None, :] * sc_n, acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def bmm16_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Bt, M, K = x.shape
    N = y.shape[2]
    c = torch.empty(Bt, M, N, dtype=torch.float32, device=x.device)
    if M >= 128 and N >= 128:                   # bench 9/10: 128 x 128 x 32, 8 warps ~23-27 TFLOPS on gfx1151
        BM, BN, w = 128, 128, 8
    else:
        BM, BN, w = (64 if M >= 64 else 32), (64 if N >= 64 else 32), 4
    _k_bmm16[(triton.cdiv(M, BM) * triton.cdiv(N, BN), Bt)](
        x, x.stride(0), x.stride(1), x.stride(2), y, y.stride(0), y.stride(1), y.stride(2),
        c, c.stride(0), c.stride(1), c.stride(2), M, N, K, 0.0, BM, BN, 32, num_warps=w)
    return c


_BV = int(os.getenv("E2E_GDN_QAT_BV", "32"))
_NW = int(os.getenv("E2E_GDN_QAT_WARPS", "4"))


class _Ricorrenza(torch.autograd.Function):
    """Same mathematics as gdn_veloce_corsa_v5._RicorrenzaTriton (u, w, kw, egl, qg, attn, S0 -> o, S_N), products
    split (SPLIT 3 / 1)."""

    @staticmethod
    def forward(ctx, u, w, kw, egl, qg, attn, S0, split: int):
        N, BH, C, Dv = u.shape
        Dk = w.shape[-1]
        assert Dv % _BV == 0 and u.dtype == torch.float32
        u, w, kw, egl = (t.contiguous() for t in (u, w, kw, egl))
        Sa = torch.empty(N + 1, BH, Dk, Dv, dtype=u.dtype, device=u.device)
        vn = torch.empty_like(u)
        _k_ric_av[(BH, Dv // _BV)](u, w, kw, egl, S0.contiguous(), vn, Sa, BH, N=N, C=C, DK=Dk, DV=Dv, BV=_BV,
                                   SPLIT=split, num_warps=_NW)
        Sp = Sa[:N]
        o = mm_split(qg, Sp, split)
        o += mm_split(attn, vn, split)
        ctx.save_for_backward(w, kw, egl, qg, attn, vn, Sa)
        ctx.split = split
        return o, Sa[N].clone()

    @staticmethod
    def backward(ctx, d_o, d_SN):
        w, kw, egl, qg, attn, vn, Sa = ctx.saved_tensors
        sp = ctx.split
        N, BH, C, Dv = vn.shape
        Dk = w.shape[-1]
        Sp = Sa[:N]
        d_o = d_o.contiguous().to(torch.float32)
        m = __import__("e2e_kernel_qat").amax_abs(d_o, dim=(0, 2, 3))                                          # per head (BH,)
        if d_SN is not None:
            m = torch.maximum(m, __import__("e2e_kernel_qat").amax_abs(d_SN, dim=(1, 2)))
        sc = _pot2(m)                                                              # exact scale per head
        d_os = d_o * sc.view(1, BH, 1, 1)
        d_attn = mm_split(d_os, vn.transpose(-1, -2), sp)
        d_qg = mm_split(d_os, Sp.transpose(-1, -2), sp)
        dvn = mm_split(attn.transpose(-1, -2), d_os, sp).contiguous()
        dSa = torch.empty_like(Sa)
        dSa[:N] = mm_split(qg.transpose(-1, -2), d_os, sp)
        if d_SN is None:
            dSa[N].zero_()
        else:
            dSa[N].copy_(d_SN * sc.view(BH, 1, 1))
        _k_ric_ind[(BH, Dv // _BV)](dvn, dSa, w.contiguous(), kw.contiguous(), egl.contiguous(), BH, N=N, C=C, DK=Dk,
                                    DV=Dv, BV=_BV, SPLIT=sp, num_warps=_NW)
        dS_dopo = dSa[1:]
        dw = mm_split(dvn, Sp.transpose(-1, -2), sp).neg_()
        dkw = mm_split(vn, dS_dopo.transpose(-1, -2), sp)
        degl = (Sp * dS_dopo).sum((-1, -2))
        inv = (1.0 / sc)                                                           # exact
        iv4 = inv.view(1, BH, 1, 1)
        return (dvn * iv4, dw * iv4, dkw * iv4, degl * inv.view(1, BH), d_qg * iv4, d_attn * iv4,
                dSa[0] * inv.view(BH, 1, 1), None)


class _RicorrenzaIbrida(torch.autograd.Function):
    """Same mathematics as gdn_veloce_corsa_v5._RicorrenzaUscita3: the sequential loop over the chunks stays the exact
    fp32 in-place baddbmm of v3/v5 (64 chunks x 2 small batched products: latency-bound, and the Triton recurrence
    spills on gfx1151 — AOT: 500-1600 VGPR), while every batched product OUTSIDE the loop (o = qg S + attn vn and the
    four of the backward) goes through mm_split (fp16 split, hipBLASLt). E2E_GDN_QAT_RIC=ibrida (default)."""

    @staticmethod
    def forward(ctx, u, w, kw, egl, qg, attn, S0, split: int):
        N, BH, C, Dv = u.shape
        Dk = w.shape[-1]
        Sa = torch.empty(N + 1, BH, Dk, Dv, dtype=u.dtype, device=u.device)
        Sa[0].copy_(S0)
        vn = u.clone()
        kwT = kw.transpose(-1, -2)
        eglv = egl.unsqueeze(-1).unsqueeze(-1)
        for i in range(N):
            vn[i].baddbmm_(w[i], Sa[i], alpha=-1)
            torch.mul(Sa[i], eglv[i], out=Sa[i + 1])
            Sa[i + 1].baddbmm_(kwT[i], vn[i])
        Sp = Sa[:N]
        o = mm_split(qg, Sp, split)
        o += mm_split(attn, vn, split)
        ctx.save_for_backward(w, kw, egl, qg, attn, vn, Sa)
        ctx.split = split
        return o, Sa[N].clone()

    @staticmethod
    def backward(ctx, d_o, d_SN):
        w, kw, egl, qg, attn, vn, Sa = ctx.saved_tensors
        sp = ctx.split
        N, BH, C, Dv = vn.shape
        Sp = Sa[:N]
        d_o = d_o.contiguous().to(torch.float32)
        d_attn = mm_split(d_o, vn.transpose(-1, -2), sp)
        d_qg = mm_split(d_o, Sp.transpose(-1, -2), sp)
        dvn = mm_split(attn.transpose(-1, -2), d_o, sp).contiguous()
        dSa = torch.empty_like(Sa)
        dSa[:N] = mm_split(qg.transpose(-1, -2), d_o, sp)
        if d_SN is None:
            dSa[N].zero_()
        else:
            dSa[N].copy_(d_SN)
        wT = w.transpose(-1, -2)
        eglv = egl.unsqueeze(-1).unsqueeze(-1)
        for i in range(N - 1, -1, -1):                                   # exact fp32 adjoint loop (v3)
            dvn[i].baddbmm_(kw[i], dSa[i + 1])
            dSa[i].addcmul_(dSa[i + 1], eglv[i])
            dSa[i].baddbmm_(wT[i], dvn[i], alpha=-1)
        dS_dopo = dSa[1:]
        dw = mm_split(dvn, Sp.transpose(-1, -2), sp).neg_()
        dkw = mm_split(vn, dS_dopo.transpose(-1, -2), sp)
        degl = (Sp * dS_dopo).sum((-1, -2))
        return dvn, dw, dkw, degl, d_qg, d_attn, dSa[0], None


_RIC = {"via": os.getenv("E2E_GDN_QAT_RIC", "ibrida")}


def _prep_el(q, k, v, g, beta, C: int, N: int, l2: bool):
    """The elementwise part of v5's _prep (no products): per-chunk layout (N, BH, C, .)."""
    if l2:
        q = V5.l2norm(q, dim=-1, eps=1e-6)
        k = V5.l2norm(k, dim=-1, eps=1e-6)
    q = q * (1 / (q.shape[-1] ** 0.5))
    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    q, k, k_beta, v_beta, g = [V5._in_chunk(x, N, C) for x in (q, k, k_beta, v_beta, g)]
    g = g.cumsum(-1)
    triu1 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(triu1, float("-inf")).exp()
    eg = g.exp()
    kbeg = k_beta * eg.unsqueeze(-1)
    qg = q * eg.unsqueeze(-1)
    gl = g[..., -1]
    kw = k * (gl.unsqueeze(-1) - g).exp().unsqueeze(-1)
    egl = gl.exp()
    return q, k, k_beta, v_beta, decay, kbeg, qg, kw, egl


_PREP_C = {"f": None, "fallita": False}
COMPILA = os.getenv("E2E_GDN_QAT_COMPILA", "1") == "1"


def _prep_scelta():
    """The elementwise preparation compiled with torch.compile (fused kernels, forward and backward), as v5 does for
    its own; on a compilation error, eager (same mathematics) and a warning."""
    if not COMPILA or _PREP_C["fallita"] or os.environ.get("TRITON_INTERPRET") == "1":
        return _prep_el
    if _PREP_C["f"] is None:
        _PREP_C["f"] = torch.compile(_prep_el, dynamic=False)

    def f(*a):
        try:
            return _PREP_C["f"](*a)
        except Exception as e:          # compilation failure: eager, loudly
            _PREP_C["fallita"] = True
            import warnings
            warnings.warn(f"gdn_qat: torch.compile of the preparation failed ({type(e).__name__}: {e}); eager")
            return _prep_el(*a)
    return f


def _nucleo(q, k, v, g, beta, chunk_size, initial_state, split: int, l2: bool):
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - T % C) % C
    dt = torch.float32
    q, k, v, beta, g = [x.to(dt) for x in (q, k, v, beta, g)]
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
    L = T + pad
    N = L // C
    qc, kc, k_beta, v_beta, decay, kbeg, qg, kw, egl = _prep_scelta()(q, k, v, g, beta, C, N, l2)
    triu0 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)
    kT = kc.transpose(-1, -2)
    Lm = (mm(k_beta, kT, split) * decay).masked_fill(triu0, 0)
    attn = mm(qc, kT, split) * decay
    X = _Inversa.apply(Lm, split)
    u = mm(X, v_beta, split)
    w = mm(X, kbeg, split)
    S0 = (torch.zeros(B * H, Dk, Dv, dtype=dt, device=q.device) if initial_state is None
          else initial_state.to(device=q.device, dtype=dt).reshape(B * H, Dk, Dv))
    Ric = _Ricorrenza if _RIC["via"] == "triton" else _RicorrenzaIbrida
    o, SN = Ric.apply(u, w, kw, egl, qg, attn, S0, split)
    o = o.reshape(N, B, H, C, Dv).permute(1, 0, 3, 2, 4).reshape(B, L, H, Dv)[:, :T]
    return o, SN.reshape(B, H, Dk, Dv)


def chunk_gated_delta_rule_qat(query, key, value, g, beta, chunk_size=64, initial_state=None,
                               output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
    """Same signature/result as transformers torch_chunk_gated_delta_rule. Without gdn1/gdn3: v5, unchanged."""
    split = _split_n()
    if split == 0 or cu_seqlens is not None or not (query.is_cuda or os.environ.get("TRITON_INTERPRET") == "1") \
            or value.shape[-1] % _BV:
        return V5.chunk_gated_delta_rule_veloce(query, key, value, g, beta, chunk_size, initial_state,
                                                output_final_state, use_qk_l2norm_in_kernel, cu_seqlens, **kwargs)
    initial_dtype = query.dtype
    o, SN = _nucleo(query, key, value, g, beta, chunk_size, initial_state, split, use_qk_l2norm_in_kernel)
    return o.contiguous().to(initial_dtype), (SN if output_final_state else None)


def installa(modulo=None) -> None:
    if modulo is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modulo
    modulo.torch_chunk_gated_delta_rule = chunk_gated_delta_rule_qat
