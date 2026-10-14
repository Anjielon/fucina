#!/usr/bin/env python3
"""folded_dense — multi-plane ternary dense layers «folded» into a single effective weight (2026-10-01, folded-dense
branch from optimized TT5 c333d5e). NEW file. Switch: E2E_DENSE_FOLDED = "" (default: OFF, nothing changes),
"x2" (effective weight as an fp16 hi+lo pair, fp16 products with fp32 accumulation) or "f32" (fp32 effective weight,
fp32 products). Design and counts: the project's folded-dense design note (fix #3 of the audit, idea 2 of the
literature review).

## Why
Every TT5 dense layer (`TernaryLinear` with name2/name3) is W = T1*d1 + T2*d2 + T3*d3 (trits T_k in {-1,0,1}, per-group
scales d_k fp16). Today it is computed as THREE `ExpertFamily` on the same input: three products in the forward, three
dX, three dL/dd (and in B3_ind32 the six backward products are fp32 = FMA, no WMMA on gfx1151). But
    Y   = X W^T                      with W = sum_k W_k             -> ONE product
    dX  = dY W                                                      -> ONE product
    G   = dY^T X       (equal for all planes: same X, same dY)      -> ONE product
    dL/dd_k[r, g] = sum_{c in g} T_k[r, c] G[r, c]                  -> elementwise epilogue, no product
and the statistics of the V step (proposal, verification: functions of G = dY^T X; curvature: of (dY^2)^T (X^2), also
equal for all planes) are derived from the SAME G with the reference functions of plane_codes.

## «x2» representation (fp32-accurate with fp16 products)
A SINGLE fp16 weight is not enough: W has up to ~11 + log2(d1/d3) bits and fp16(W) is off by 2^-11 relative (≈ 5e-4),
a hundred times the error of an fp32 product. The pair is used (Henry et al. 1904.06376 / Ootomo-Yokota 2203.03341, as
in `hilo_backward`):
    Wf = fl32(fl32(W1 + W2) + W3)                      (|Wf - W| <= 2^-23 * sum_k |W_k|, two fp32 roundings)
    row r scaled by a power of two: a_r = 2^-e_r, max_c |Wf[r,c]| a_r in [1/2, 1)   (exact)
    hi = fp16(Wf a_r), lo = fp16(Wf a_r - hi)           (|Wf a_r - hi - lo| <= 2^-24 absolute)
    => |hi + lo - Wf a_r| <= 2^-23 * max_c |W a_r|      (i.e. 2^-23 of the row maximum)
X (rotated activation, fp16) is EXACT; dY (fp32) is split as in `hilo_backward` (power of two per token row in dX,
per column r in G). Products (fp16 x fp16 exact in fp32, fp32 accumulation):
    forward: Y   = 2^e_r (X lo^T + X hi^T)                                    2 products (today 3)
    dX    : dX  = 2^f_s (h_lo hi + h_hi lo + h_hi hi),  h = dY 2^e_r 2^-f_s   3 products (today 3 fp32 / 6 fp16)
    G     : G   = 2^g_r (dY_lo^T X + dY_hi^T X),  dY scaled 2^-g_r            2 products (today 3 fp32 / 6 fp16)
Only h_lo lo is dropped (<= 2^-24 * 2^-24 relative). «f32»: Wf, dY and X in fp32, ONE fp32 product each.

## Error bounds (proof in §3 of the design note; u = 2^-24, gamma_n = n u / (1 - n u))
For each element, with B the sum of the absolute values of the terms (B_Y[s,r] = sum_c |x_sc| sum_k |W_k[r,c]|;
B_dd[r,g] = sum_{s, c in g} |dY_sr| |x_sc|; Bbar_dd[r,g] = max_s |dY_sr| sum_{s, c in g} |x_sc|):
    |Y_folded  - Y_today |  <= (2 gamma_{C+2}   + 2^-22) B_Y
    |dd_folded - dd_today|  <= (2 gamma_{S+gs}) B_dd + 2^-23 Bbar_dd    (x2; in f32 without the second term)
    dX: today it is fl16(fl16(fl16(dx1) + fl16(dx2)) + fl16(dx3)) (autograd accumulation in fp16), folded fl16(dX32):
        |dX_folded - dX_today| <= 5 * 2^-11 * sum_k |dx_k| + 2 gamma_{R} B_dX  (TODAY's fp16 rounding dominates)
These are deterministic worst-case bounds (Higham, ch. 3); the typical error is ~sqrt(n) u and is measured by the tests
(`e2e_test_densi_piegati.py`).

## When folding happens (otherwise: today's path, bit-exact)
Switch on, >= 2 planes, all of the same type_/shape with E = 1, recipe with dtype_gemm fp16 and (rotate or a
non-rotated family), fp16 input, torch or Triton engine (also +V). The V step with scale curvature (newton-theta,
`BufferV.HT`) uses today's per-plane function for HT (ONE plane at a time): TT5 does not use it (adamw).
"""
from __future__ import annotations

import os
import weakref
from typing import Optional

import torch

MODES = ("", "x2", "f32", "reorder")   # "reorder": ONLY the gate's noise arm (today's path, planes summed in reverse)
# gemm: "triton" (default since 2026-10-02) = Triton kernel `_k_mm16` (fp16 -> fp32); "auto" = torch.mm(out_dtype=fp32),
# i.e. hipBLASLt: on the 2026-10-01 GPU gate 23 863 ms versus 221 for attn_qkv, UNUSABLE; "f32" = always fp32 matmul.
# On the CPU (except with E2E_FOLD_KERNEL_CPU) the fp16 products are fp32 matmuls on the exact inputs (same products).
FOLD = {"mode": os.getenv("E2E_DENSE_FOLDED", "").strip(), "gemm": os.getenv("E2E_FOLD_GEMM", "triton").strip(),
         "cache_trit": os.getenv("E2E_FOLD_CACHE_TRIT", "0") == "1",
         # CPU tests ONLY (TRITON_INTERPRET=1): walks the GPU branches (Triton trit decode, gemm `_k_mm16`)
         "kernel_cpu": os.getenv("E2E_FOLD_KERNEL_CPU", "0") == "1"}
if FOLD["mode"] not in MODES:
    raise ValueError(f"E2E_DENSE_FOLDED={FOLD['mode']!r}: {MODES}")
COUNTS = {"folded": 0, "fallback": 0}          # forward calls (diagnostics, no effect)
_TRIT_CACHE: dict = {}


def set_(mode: Optional[str] = None, gemm: Optional[str] = None, cache_trit: Optional[bool] = None) -> None:
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: {MODES}")
        FOLD["mode"] = mode
    if gemm is not None:
        if gemm not in ("auto", "triton", "f32"):
            raise ValueError(f"gemm {gemm!r}")
        FOLD["gemm"] = gemm
    if cache_trit is not None:
        FOLD["cache_trit"] = bool(cache_trit)
        _TRIT_CACHE.clear()


def active() -> bool:
    return FOLD["mode"] in ("x2", "f32", "reorder")


def apply_reorder(xg: torch.Tensor, ds: list, fams: list, recipe, mot, oss) -> torch.Tensor:
    """NOISE arm of the gate (2026-10-02): TODAY's path (n ExpertFamily, same kernels) with the plane outputs summed in
    REVERSE order (today (y1 + y2) + y3, here (y3 + y2) + y1), hence also the dX accumulated by autograd in a different
    order. It changes ONLY fp32/fp16 roundings: it measures how much the 40 layers amplify a rounding perturbation of
    the dense layers (the Bk2_ind32 arm with ternary dense layers perturbs nothing: KL ~0, h bit-exact)."""
    from fucina.student import student as S
    COUNTS["folded"] += 1
    plane = S.plane_single(xg.shape[0], xg.device)
    y = None
    for f, d in reversed(list(zip(fams, ds))):
        yk = S.ExpertFamily.apply(xg.contiguous(), d, f, plane, recipe, mot, oss)
        y = yk if y is None else y + yk
    return y


def allowed(fams: list, recipe, mot, xg: torch.Tensor) -> bool:
    from fucina.student import student as S
    if len(fams) < 2 or xg.dtype != torch.float16 or recipe.dtype_gemm != torch.float16:
        return False
    if not isinstance(mot, (S.TorchEngine, S.TritonEngine)) or getattr(mot, "dense", False):
        return False
    f = fams[0]
    if f.rotated and not recipe.rotate:
        return False
    return all((g.type_, g.E, g.R, g.C, g.NBR) == (f.type_, 1, f.R, f.C, f.NBR) for g in fams)


# ═══════════════════════════════════ effective weight ═══════════════════════════════════
def trit(fam, mot) -> torch.Tensor:
    """Trits (R, C) of the plane (byte domain: rotated for TQ1_0R_1024, like the recipe B product). On GPU with the
    fused decode kernel (d = 1: trit x 1 exact in fp16); on CPU with the torch decoder. int8."""
    from fucina.student import student as S
    B = fam.blocks
    if FOLD["cache_trit"]:
        # key = IDENTITY of the byte tensor (weak reference: a freed tensor is not confused with a new one at the same
        # address) + version counter (the V-step skips write the bytes in place: _version increases)
        v = _TRIT_CACHE.get(id(B))
        if v is not None and v[0]() is B and v[1] == B._version:
            return v[2]
    if (isinstance(mot, S.TritonEngine) and B.is_cuda) or FOLD["kernel_cpu"]:
        t = S.decode_weights(fam, torch.ones_like(fam.d0), torch.float16, getattr(mot, "blk", None) or S.Blocks())[0] \
            .to(torch.int8)
    else:
        t = S.trit_expert(fam, 0)
    if FOLD["cache_trit"]:
        _TRIT_CACHE[id(B)] = (weakref.ref(B), B._version, t)
    return t


def weight_effective_f32(trits: list, ds: list, gs: int) -> torch.Tensor:
    """Wf (R, C) fp32 = fl(fl(T1 d1 + T2 d2) + T3 d3): each T_k d_k is exact (d_k fp16), the sums in plane order
    (the same order as today's outputs)."""
    R, C = trits[0].shape
    W = None
    for t, d in zip(trits, ds):
        w = (t.view(R, C // gs, gs).to(torch.float32) * d.reshape(R, C // gs, 1).to(torch.float32)).view(R, C)
        W = w if W is None else W.add_(w)
    return W


def scale_pot2(m: torch.Tensor) -> tuple:
    from fucina.kernels import hilo_backward as HL
    return HL.scale_pot2(m)


def split_fp16(v: torch.Tensor) -> tuple:
    from fucina.kernels import hilo_backward as HL
    return HL.split_fp16(v)


def weight_x2(Wf: torch.Tensor) -> tuple:
    """(hi, lo, inv_r): Wf[r] = inv_r[r] (hi[r] + lo[r]) within 2^-23 of the row maximum; hi, lo fp16 (R, C)."""
    a, inv = scale_pot2(__import__("fucina.kernels.qat", fromlist=("amax_abs",)).amax_abs(Wf, 1))
    hi, lo = split_fp16(Wf * a[:, None])
    return hi, lo, inv


# ═══════════════════════════════════ products ═══════════════════════════════════
try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

if _TRITON:
    @triton.jit
    def _k_mm16(a_ptr, sa_m, sa_k, b_ptr, sb_k, sb_n, c_ptr, sc_m, sc_n, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        """C (M, N) fp32 = A (M, K) @ B (K, N), A and B fp16 (any strides: transposes are views), fp32 accumulation."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        per_group = GROUP_M * npn
        gid = pid // per_group
        before = gid * GROUP_M
        gm = tl.minimum(npm - before, GROUP_M)
        pm = before + (pid % per_group) % gm
        pn = (pid % per_group) // gm
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(a_ptr + rm[:, None].to(tl.int64) * sa_m + kk[None, :].to(tl.int64) * sa_k,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(b_ptr + kk[:, None].to(tl.int64) * sb_k + rn[None, :].to(tl.int64) * sb_n,
                        mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_ptr + rm[:, None].to(tl.int64) * sc_m + rn[None, :].to(tl.int64) * sc_n, acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _k_mm16b(a_ptr, sa_m, sa_k, b_ptr, sb_k, sb_n, c_ptr, sc_m, sc_n, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, EVEN: tl.constexpr):
        """Same product as _k_mm16 (fp16 operands, fp32 accumulation chained along K in the same order), written with
        pointer increments and 32-bit offsets inside the tile (K multiple of BK): fewer registers and VALU per step,
        so that 128 x 128 tiles fit (2026-10-09 bench: plain fp16 Triton GEMM 25-29 TFLOPS at these tiles on gfx1151)."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        per_group = GROUP_M * npn
        gid = pid // per_group
        before = gid * GROUP_M
        gm = tl.minimum(npm - before, GROUP_M)
        pm = before + (pid % per_group) % gm
        pn = (pid % per_group) // gm
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm_ = (rm < M)[:, None]
        mn = (rn < N)[None, :]
        pa = a_ptr + (pm * BM).to(tl.int64) * sa_m + (tl.arange(0, BM) * sa_m)[:, None] + (rk * sa_k)[None, :]
        pb = b_ptr + (pn * BN).to(tl.int64) * sb_n + (rk * sb_k)[:, None] + (tl.arange(0, BN) * sb_n)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            if EVEN:                                      # M, N multiples of the tile: no masks (fewer registers)
                a = tl.load(pa)
                b = tl.load(pb)
            else:
                a = tl.load(pa, mask=mm_, other=0.0)
                b = tl.load(pb, mask=mn, other=0.0)
            acc += tl.dot(a, b)
            pa += BK * sa_k
            pb += BK * sb_k
        pc = c_ptr + (pm * BM).to(tl.int64) * sc_m + (pn * BN).to(tl.int64) * sc_n \
            + (tl.arange(0, BM) * sc_m)[:, None] + (tl.arange(0, BN) * sc_n)[None, :]
        if EVEN:
            tl.store(pc, acc)
        else:
            tl.store(pc, acc, mask=mm_ & mn)

    def mm16_triton(a: torch.Tensor, b: torch.Tensor, BM: int = 64, BN: int = 64, BK: int = 32, num_warps: int = 4,
                    GROUP_M: int = 8) -> torch.Tensor:
        assert a.dtype == b.dtype == torch.float16 and a.shape[1] == b.shape[0]
        from fucina.kernels import qat as KQ                 # 2026-10-09 E2E_KQAT=mm16_large: _k_mm16b with large tiles
        if KQ.active("mm16_large") and a.shape[1] % KQ.MM16_LARGE[2] == 0:
            BM2, BN2, BK2, w2 = KQ.MM16_LARGE       # same WMMA chain along K per element -> bit-exact (trial K7)
            M, K = a.shape
            N = b.shape[1]
            c = torch.empty(M, N, dtype=torch.float32, device=a.device)
            _k_mm16b[(triton.cdiv(M, BM2) * triton.cdiv(N, BN2),)](
                a, a.stride(0), a.stride(1), b, b.stride(0), b.stride(1), c, c.stride(0), c.stride(1), M, N, K,
                BM2, BN2, BK2, GROUP_M, int(M % BM2 == 0 and N % BN2 == 0), num_warps=w2, num_stages=2)
            return c
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty(M, N, dtype=torch.float32, device=a.device)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _k_mm16[grid](a, a.stride(0), a.stride(1), b, b.stride(0), b.stride(1), c, c.stride(0), c.stride(1),
                         M, N, K, BM, BN, BK, GROUP_M, num_warps=num_warps)
        return c


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b in fp32. fp16 inputs (exact values) -> exact products, fp32 accumulation; fp32 inputs -> fp32 GEMM."""
    if a.dtype == torch.float32 or FOLD["gemm"] == "f32" or not (a.is_cuda or FOLD["kernel_cpu"]):
        return torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if FOLD["gemm"] == "triton":
        return mm16_triton(a, b)
    return torch.mm(a, b, out_dtype=torch.float32)


# ═══════════════════════════════════ autograd node ═══════════════════════════════════
def _mode_v(mot) -> int:
    st = getattr(mot, "state", None)
    return int(getattr(st, "mode", 0) or 0) if st is not None else 0


def forward_(xg: torch.Tensor, ds: list, fams: list, mot, mode: str) -> tuple:
    """-> (Y fp32 (S, R), weights for the backward). weights = (trits, hi, lo, inv) in x2, (trits, Wf) in f32."""
    gs = fams[0].gs
    from fucina.kernels import qat as KQ                 # 2026-10-09 E2E_KQAT=cache_dense: folded weight cached per step
    weights = KQ.dense_cache_take(fams, mode) if mode == "x2" else None
    if weights is not None:
        trits, hi, lo, inv = weights
    else:
        trits = [trit(f, mot) for f in fams]
        Wf = weight_effective_f32(trits, ds, gs)
        if mode == "f32":
            return torch.matmul(xg.to(torch.float32), Wf.T), (trits, Wf)
        hi, lo, inv = weight_x2(Wf)
        del Wf
        if mode == "x2":                          # trits NOT kept in the cache (4.4 GB on 40 layers): re-decoded
            KQ.dense_cache_put(fams, mode, (None, hi, lo, inv))      # by the backward when it needs them
    if KQ.active("dense_blas") and KQ.active("fold1") and xg.is_cuda:   # 2026-10-09 fp16 level on hipBLASLt
        return torch.mm(xg, hi.T, out_dtype=torch.float32).mul_(inv[None, :]), (trits, hi, lo, inv)
    if KQ.active("dense_k") and not KQ.active("mm16_large") and xg.is_cuda:       # 2026-10-09: the products in ONE launch (_k_mmx), same values bit-exact
        return KQ.mmx(xg, hi.T, lo.T, 1 if KQ.active("fold1") else 3, ocscale=inv), (trits, hi, lo, inv)
    if KQ.active("fold1"):                       # 2026-10-09 fp16 level: SINGLE fp16 folded weight (hi), see kernels.qat
        return mm(xg, hi.T).mul_(inv[None, :]), (trits, hi, lo, inv)
    y = mm(xg, lo.T)
    y += mm(xg, hi.T)
    return y.mul_(inv[None, :]), (trits, hi, lo, inv)


def backward_x(g32: torch.Tensor, weights: tuple, mode: str) -> torch.Tensor:
    if mode == "f32":
        return torch.matmul(g32, weights[1])
    _, hi, lo, inv = weights
    from fucina.kernels import qat as KQ
    if KQ.active("dense_blas") and KQ.active("fold1") and KQ.active("x1") and g32.is_cuda:   # fp16 on hipBLASLt
        s, inv_s = scale_pot2((g32.abs() * inv[None, :]).amax(1, keepdim=True))
        return torch.mm((g32 * inv[None, :] * s).to(torch.float16), hi, out_dtype=torch.float32).mul_(inv_s)
    if KQ.active("dense_k") and not KQ.active("mm16_large") and g32.is_cuda:     # 2026-10-09: scale and split in registers, the products in ONE launch
        s, inv_s = scale_pot2((g32.abs() * inv[None, :]).amax(1, keepdim=True))
        t = (0 if KQ.active("x1") else 4) | (0 if KQ.active("fold1") else 2) | 1
        return KQ.mmx(g32, hi, lo, t, rscale=s, cscale=inv, oscale=inv_s)
    from fucina.kernels import qat as KQ                 # 2026-10-09 fp16 level: x1 = single fp16 dY, fold1 = single fp16 weight
    x1, p1 = KQ.active("x1"), KQ.active("fold1")
    if KQ.active("prep") and g32.is_cuda and g32.stride(1) == 1:      # 2026-10-09: fused preparation, same bytes
        hh, hl, s, inv_s = KQ.prep_rows(g32, inv.contiguous(), lo=not x1)
    else:
        h = g32 * inv[None, :]                                   # powers of two: exact
        s, inv_s = scale_pot2(__import__("fucina.kernels.qat", fromlist=("amax_abs",)).amax_abs(h, 1, keepdim=True))
        hh, hl = split_fp16(h * s)
        del h
    dx = None
    for a_, b_, need in ((hl, hi, not x1), (hh, lo, not p1), (hh, hi, True)):
        if need:
            dx = mm(a_, b_) if dx is None else dx.add_(mm(a_, b_))
    return dx.mul_(inv_s)


def product_g(g32: torch.Tensor, xg: torch.Tensor, mode: str) -> torch.Tensor:
    """G (R, C) fp32 = dY^T X."""
    if mode == "f32":
        return torch.matmul(g32.T, xg.to(torch.float32))
    s, inv = scale_pot2(__import__("fucina.kernels.qat", fromlist=("amax_abs",)).amax_abs(g32, 0))                   # per column r of dY (row r of G)
    from fucina.kernels import qat as KQ
    if KQ.active("dense_blas") and KQ.active("x1") and g32.is_cuda:     # fp16 on hipBLASLt
        return torch.mm((g32 * s[None, :]).to(torch.float16).T, xg, out_dtype=torch.float32).mul_(inv[:, None])
    if KQ.active("dense_k") and not KQ.active("mm16_large") and g32.is_cuda:     # 2026-10-09: split in registers, the two products in ONE launch
        return KQ.mmx(g32.T, xg, None, 1 if KQ.active("x1") else 5, rscale=s, oscale=inv)
    if KQ.active("prep") and g32.is_cuda and g32.stride(1) == 1:      # 2026-10-09: fused preparation, same bytes
        gh, gl, s, inv = KQ.prep_columns(g32, lo=not KQ.active("x1"))
    else:
        gh, gl = split_fp16(g32 * s[None, :])
    if KQ.active("x1"):                           # 2026-10-09 fp16 level: single fp16 dY (scaled by a power of two)
        return mm(gh.T, xg).mul_(inv[:, None])
    G = mm(gl.T, xg)
    G += mm(gh.T, xg)
    return G.mul_(inv[:, None])


def dd_from_g(G: torch.Tensor, t: torch.Tensor, gs: int) -> torch.Tensor:
    R, C = G.shape
    return (t.view(R, C // gs, gs).to(torch.float32) * G.view(R, C // gs, gs)).sum(-1)


class FoldedDense(torch.autograd.Function):
    """y = sum_k X W_k(d_k)^T with ONE effective weight (see the module header). Inputs: xg (S, C) fp16, then the plane
    scales d_1..d_n (1, R, NG). Same outputs as n summed `ExpertFamily` (dtype, shapes), within the stated bounds."""

    @staticmethod
    def forward(ctx, xg, fams, recipe, mot, oss, mode, *ds):
        y, weights = forward_(xg, list(ds), list(fams), mot, mode)
        ctx.save_for_backward(xg, *ds)
        ctx.weights, ctx.fams, ctx.recipe, ctx.mot, ctx.oss, ctx.mode = weights, list(fams), recipe, mot, oss, mode
        return y

    @staticmethod
    def backward(ctx, gy):
        from fucina.student import plane_codes as PV
        xg, *ds = ctx.saved_tensors
        fams, mot, oss, mode, weights = ctx.fams, ctx.mot, ctx.oss, ctx.mode, ctx.weights
        need_x = ctx.needs_input_grad[0]
        need_d = [ctx.needs_input_grad[6 + i] for i in range(len(ds))]
        g32 = gy.to(torch.float32).contiguous()
        if oss is not None:
            for f in fams:                                    # same diagnostics rows as today (one per plane)
                oss.gradient(f, gy, g32)
        if any(need_d):                                      # 2026-10-02 --pv-dd-only-skips: planes without prepared skips
            need_d = [sd and not PV.dd_redundant(mot, f) for sd, f in zip(need_d, fams)]
        dx = None
        if need_x:
            dx = backward_x(g32, weights, mode).to(xg.dtype)
            if oss is not None:
                oss.watch(dx)
        dds = [None] * len(ds)
        if any(need_d):
            trits = weights[0] if weights[0] is not None else [trit(f, mot) for f in fams]   # cache_dense: on demand
            gs = fams[0].gs
            mv = _mode_v(mot)
            bufs = [mot.state.buffer(f) if mv else None for f in fams]
            if mv == 3:                                       # curvature: dL/dd = 0, H from (dY^2)^T (X^2) (ONE for all)
                G2 = torch.matmul((g32 * g32).T, xg.to(torch.float32).pow(2))
                for i, f in enumerate(fams):
                    if not need_d[i]:
                        continue
                    dds[i] = torch.zeros(1, f.R, f.NG, dtype=ds[i].dtype, device=g32.device)
                    PV.curvature_from_dw2(bufs[i], 0, G2, gs)
                    if bufs[i].HT is not None:                # newton-theta: per plane (today's path)
                        plane = _plane(g32.shape[0], g32.device)
                        if type(mot).__name__ == "TritonEngineV":
                            PV.ht_triton(g32, xg.to(torch.float32).contiguous(), f, plane, bufs[i].HT)
                        else:
                            PV.ht_torch(g32, xg.to(torch.float32), f, plane, bufs[i].HT)
            else:
                G = product_g(g32, xg, mode)
                for i, f in enumerate(fams):
                    if not need_d[i]:
                        continue
                    dds[i] = dd_from_g(G, trits[i], gs).unsqueeze(0).to(ds[i].dtype)
                    if mv == 1:
                        PV.proposal_from_dw(bufs[i], 0, G, trits[i], f.d0[0], gs)
                    elif mv == 2:
                        PV.verify_from_dw(bufs[i], 0, G, gs)
        ctx.weights = None
        return (dx, None, None, None, None, None, *dds)


def _plane(T: int, device):
    from fucina.student import student as S
    return S.plane_single(T, device)


def apply(xg: torch.Tensor, ds: list, fams: list, recipe, mot, oss) -> torch.Tensor:
    if FOLD["mode"] == "reorder":
        return apply_reorder(xg, ds, fams, recipe, mot, oss)
    COUNTS["folded"] += 1
    return FoldedDense.apply(xg.contiguous(), tuple(fams), recipe, mot, oss, FOLD["mode"], *ds)


def byte_transients(R: int, C: int, S_: int, mode: str = "x2") -> dict:
    """Extra TRANSIENT memory compared with today for ONE dense layer (today: no materialized weight, decode in the
    kernel). trits int8 n=3 planes, Wf fp32 (during construction), hi+lo fp16 (kept until the layer's backward), G fp32,
    splits of dY (S, R) and h (S, R)."""
    RC = R * C
    t = 3 * RC
    if mode == "f32":
        return {"trit": t, "W": 4 * RC, "G": 4 * RC, "dY_split": 0, "peak": t + 8 * RC}
    return {"trit": t, "W_construction_f32": 4 * RC, "hi_lo": 4 * RC, "G": 4 * RC, "dY_split": 4 * S_ * R + 4 * S_ * R,
            "peak": t + 4 * RC + 4 * RC + 8 * S_ * R}
