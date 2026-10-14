#!/usr/bin/env python3
"""Backward pass of the EXPERTS with hi/lo split products (2026-09-27, bf16x3 backward branch).

## Why
The runs (EB2, and likewise EB3) use recipe `B_ind32`: the experts' forward pass is exact fp16 (rotated fp16
activation, fp16 trit x d weight), but the BACKWARD pass (dX = dY W and dL/dd = group_sum trit (.) dY^T X) uses fp32
inputs. On gfx1151 an fp32 `tl.dot` has no WMMA path (FMA, 1.3-3 TFLOPS versus 20-34 for fp16), and the fp32
`_k_dd2` spills 438 VGPRs. `B_ind32` was introduced because with an fp16 backward pass (recipe B) NO global scale K
avoids both overflow and underflow of the gradients (G1, gate family): it is a problem of dynamic RANGE, not of bits.

Here the same products are computed with the EXISTING fp16/bf16 kernels (those of recipe B), splitting dY into
two or three parts (Henry-Tang-Heinecke arXiv:1904.06376 for bf16x3; Ootomo-Yokota arXiv:2203.03341 for the fp16
split with scaling against underflow), and summing the fp32 outputs:

* **"fp16x2"** (recipes `B_ind2`, `B3_ind2`): the weight W = trit x d is EXACT in fp16 (d is fp16, trit in
  {-1,0,1}); X (rotated activation, silu(g)u) is EXACT in fp16. Only dY (fp32) needs splitting. To remove the range
  problem it is scaled by a power of two (exact) before splitting:
    dX:   per ROW (token) s:              v_s = dY_s * 2^-e_s,  max|v_s| in [0.5, 1)   (dX_s is linear in dY_s)
    dL/dd per (expert e, row r of W):    v_{s,r} = dY_{s,r} * 2^-f_{e,r} for s in the segment of e   (dL/dd[e,r,:]
          is linear in column r of dY restricted to the tokens of e)
    hi = fp16(v), lo = fp16(v - hi)                     (v - hi exact in fp32)
    dX   = 2^e_s  (lo W + hi W)                          two fp16 GEMMs -> fp32 (small term first)
    dL/dd = 2^f_er (D(lo, X) + D(hi, X))                 likewise with the dL/dd kernel
  Representation error of dY: |v - hi - lo| <= 2^-24 in absolute terms with max|v| >= 1/2, i.e. <= 2^-23 of the
  row (segment) maximum: the level of an fp32 product (2^-24), BETTER than bf16x3 (2^-16 per term).
  The lo x W term loses nothing: W has no low part. TWO fp16 products per GEMM, not three.
* **"bf16x3"** (recipes `B_ind3`, `B3_ind3`): dY = hi + lo in bf16 (fp32 range: no scaling), d = d_hi + d_lo
  in bf16 (fp16 d has 11 bits: the low part has <= 3 bits, EXACT in bf16), fp16 X = x_hi + x_lo in bf16 (exact);
  three products (lo hi + hi lo + hi hi; lo lo ~2^-16 is dropped). On the CPU it is tested with the torch engine
  (exact emulation); with Triton it must be tested ONLY on the GPU (the Triton interpreter has no bf16: it produces
  meaningless numbers, already seen in test K1) -> micro-test in the backward gate.

Accumulation: the kernels fuse `acc += tl.dot(...)` in WMMA, which on gfx1151 TRUNCATES (drift ~2e-8 per step of
16): dX has contraction R = 512 (gate/up) or 2048 (down) -> <= 3e-6 of drift; dL/dd contracts over the expert's
tokens (mean ~256). It is the same arithmetic as recipe B (G1 passed); the real measurement is the gate's GPU
micro-test (error per product against fp64).

dX is rounded to fp16 anyway (`.to(x.dtype)`, as in B_ind32: x_ord and silu(g)u are fp16): 2^-11 per element,
100 times the error of the split products. On dX, fp16x2/bf16x3 and fp32 therefore give the SAME tensor except
for ~1% of elements off by 1 fp16 ulp; precision matters for dL/dd (dtheta), which stays fp32.

## Cost (estimate [S]; the gate is the measurement)
Recipe B kernels (triton2, fp16, kernel gate of 2026-09-27, alongside EB2): experts backward ~0.037-0.043
ms/token per layer. fp32 backward (B_ind32 versus B, M0 2026-09-26, triton engine): +0.197 ms/token per layer.
fp16x2 = 2 x fp16 + splits (a few elementwise passes over dY) ~0.08-0.09; B_ind32 ~0.23-0.24 -> ~-0.15 ms/token
per layer, ~-6 ms/token over 40 layers. bf16x3 = 3 x ~0.04 = ~0.12.

## Usage
Recipes in the student's `RECIPES` (`B_ind2`, `B3_ind2`, `B_ind3`, `B3_ind3`); the hook is in the student's
`_backward_family` and fires ONLY with a non-empty `Recipe.backward_split` and `dtype_backward` None
(the fp32 reference recipe of G1 sets dtype_backward = fp32 and returns to the exact fp32 backward pass).
Default: no existing recipe changes (test H0 of the hi/lo backward tests).
"""
from __future__ import annotations

from typing import Optional

import torch

VIE = ("fp16x2", "bf16x3")
_E_MAX = 125          # |exponent| of the scale: 2^+-125 is finite and exact in fp32


# ═══════════════════════════════════ splits ═══════════════════════════════════
def scale_pot2(m: torch.Tensor) -> tuple:
    """m >= 0 (maxima). -> (s, inv) fp32 with s = 2^-e EXACT, m*s in [0.5, 1) (frexp), s = 1 where m == 0 or not
    finite; inv = 2^e (the inverse, exact). |e| <= 125."""
    _, e = torch.frexp(m)
    ok = (m > 0) & torch.isfinite(m)
    e = torch.where(ok, e, torch.zeros_like(e)).clamp_(-_E_MAX, _E_MAX)
    uno = torch.ones_like(m, dtype=torch.float32)
    return torch.ldexp(uno, -e), torch.ldexp(uno, e)


def split_fp16(v: torch.Tensor) -> tuple:
    """v fp32 (already scaled: |v| < 1) -> (hi, lo) fp16 with |v - hi - lo| <= 2^-24."""
    if v.dtype != torch.float32:
        raise TypeError(f"split_fp16: {v.dtype}, expected float32")
    hi = v.to(torch.float16)
    lo = (v - hi.float()).to(torch.float16)
    return hi, lo


def split_bf16(v: torch.Tensor) -> tuple:
    """v fp32 -> (hi, lo) bf16 (same as the bf16x3 module's decompose, duplicated here to avoid the dependency)."""
    if v.dtype != torch.float32:
        raise TypeError(f"split_bf16: {v.dtype}, expected float32")
    hi = v.to(torch.bfloat16)
    lo = (v - hi.float()).to(torch.bfloat16)
    return hi, lo


def expert_of_row(plane, S_: int) -> torch.Tensor:
    """(S,) expert index of each sorted row (without host synchronizations)."""
    rows = torch.arange(S_, device=plane.offs.device, dtype=plane.offs.dtype)
    return torch.searchsorted(plane.offs[1:], rows, right=True).clamp_(max=plane.E - 1)


def maxima_segment(a: torch.Tensor, eid: torch.Tensor, E: int) -> torch.Tensor:
    """a (S, R) >= 0, eid (S,) -> (E, R): maximum per (expert, column) over the expert's tokens (0 if empty)."""
    M = torch.zeros(E, a.shape[1], dtype=a.dtype, device=a.device)
    return M.scatter_reduce_(0, eid.unsqueeze(1).expand(-1, a.shape[1]), a, reduce="amax", include_self=True)


# ═══════════════════════════════════ products ═══════════════════════════════════
def _W_expert(fam, e: int, d: torch.Tensor) -> torch.Tensor:
    """Weight (R, C) fp32 = trit x d (exact: d has <= 11 bits), for the torch engine."""
    from fucina.student import student as S
    q = S.trit_expert(fam, e).to(torch.float32)
    return (q.view(fam.R, fam.NG, fam.gs) * d[e].to(torch.float32).unsqueeze(-1)).view(fam.R, fam.C)


def product_dx(mot, g: torch.Tensor, d: torch.Tensor, fam, plane) -> torch.Tensor:
    """g (S, R) fp16|bf16 @ W_e (trit x d) per expert -> (S, C) FP32 (fp32 accumulation, output NOT rounded).
    d: scales (E, R, NG) whose values are exact in the dtype of g (the kernel converts them to that dtype)."""
    from fucina.student import student as S
    if isinstance(mot, S.TorchEngine):
        out = torch.zeros(g.shape[0], fam.C, dtype=torch.float32, device=g.device)
        o = plane.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                out[o[e]:o[e + 1]] = torch.matmul(g[o[e]:o[e + 1]].float(), _W_expert(fam, e, d))
        return out
    if getattr(mot, "dense", False):
        raise NotImplementedError("split backward: engine triton_dense not supported (dense weight in the recipe "
                                  "dtype, the low part of d in bf16 does not fit)")
    if getattr(mot, "v2", False):
        return S._dx2_triton(g, fam, d.contiguous(), plane, torch.float32, mot.blk)
    return S._gemm_triton(g, fam, d.contiguous(), None, plane, True, torch.float32, mot.blk)


def product_dd(mot, g: torch.Tensor, x: torch.Tensor, fam, plane) -> torch.Tensor:
    """dL/dd (E, R, NG) fp32 = group_sum trit (.) (g^T x) per expert; g, x same dtype (fp16|bf16)."""
    return mot.dd(g, x, fam, plane, torch.float32)


# ═══════════════════════════════════ backward pass of ONE family ═══════════════════════════════════
def _mode_v(mot) -> int:
    """Mode of the engine's V step (plane codes): 0 = none. In modes 1-3 the V engine's dL/dd also collects
    statistics that are NOT linear in dY (proposal: argmax |acc|; verification: Q += v^2; curvature: dY^2): two calls
    with split dY would get them wrong. Mode 1/2 with the Triton engine goes to the split V kernel (full acc before
    the statistics); mode 3 and the torch V engine stay on the fp32 path of B_ind32 (same statistics, bit-exact)."""
    st = getattr(mot, "state", None)
    return int(getattr(st, "mode", 0) or 0) if st is not None else 0


def dx_split(mot, g32: torch.Tensor, d: torch.Tensor, fam, plane, via: str) -> torch.Tensor:
    """dX = dY W in FP32 (before rounding to the dtype of x) with the split products of `via`."""
    if via == "fp16x2":
        s, inv = scale_pot2(__import__("fucina.kernels.qat", fromlist=("amax_abs",)).amax_abs(g32, 1, keepdim=True))
        from fucina.kernels import qat as KQ               # 2026-10-09 E2E_KQAT=dx: split in registers, hi and lo in ONE kernel
        if KQ.active("dx") and getattr(mot, "v2", False) and not getattr(mot, "dense", False) \
                and type(mot).__name__ == "TritonEngine":
            return KQ.dx_s_triton(g32, s, inv, fam, d.contiguous(), plane, mot.blk, 1 if KQ.active("x1") else 3)
        hi, lo = split_fp16(g32 * s)
        if KQ.active("x1"):                       # 2026-10-09 fp16 level: single fp16 dY (per-row power-of-two scale)
            return product_dx(mot, hi, d, fam, plane).mul_(inv)
        y = product_dx(mot, lo, d, fam, plane)
        y += product_dx(mot, hi, d, fam, plane)
        del hi, lo
        return y.mul_(inv)
    if via == "bf16x3":
        gh, gl = split_bf16(g32)
        dh, dl = split_bf16(d.to(torch.float32))
        y = product_dx(mot, gl, dh, fam, plane)
        y += product_dx(mot, gh, dl, fam, plane)
        y += product_dx(mot, gh, dh, fam, plane)
        return y
    raise ValueError(f"via {via!r}: {VIE}")


def _dd_fp32(mot, g32, x, fam, plane, d):
    """dL/dd as on the fp32 path of `_backward_family` with dtype_backward fp32 (B_ind32): x (exact fp16) in fp32."""
    return mot.dd(g32, x.to(torch.float32).contiguous(), fam, plane, d.dtype)


def backward_family(gy, x, d, fam, plane, recipe, mot, oss, need_x: bool, need_d: bool):
    """Body of the student's `_backward_family` for `recipe.backward_split` in VIE. Same outputs as the fp32 path
    (dx in the dtype of x, dL/dd in the dtype of d), with split products in place of the fp32 products."""
    via = recipe.backward_split
    if via not in VIE:
        raise ValueError(f"backward_split {via!r}: {VIE}")
    if not (recipe.rotate or not fam.rotated):
        raise ValueError("split backward: only recipes with the rotated activation (B...), X must be the one of the "
                         "product")
    if x.dtype != torch.float16:
        raise TypeError(f"split backward: experts input {x.dtype}, expected float16 (recipes B)")
    g_in = gy
    g32 = gy.to(torch.float32).contiguous()
    if oss is not None:
        oss.gradient(fam, g_in, g32)             # as on the fp32 path: no fp16 underflow to count
    if need_d:                                   # 2026-10-02 --pv-dd-only-skips: dL/dd that nobody reads (plane codes)
        from fucina.student import plane_codes as PV
        if PV.dd_redundant(mot, fam):
            need_d = False
    dx = dd = None
    if need_x:
        dx = dx_split(mot, g32, d, fam, plane, via).to(x.dtype)
    if via == "fp16x2":
        if need_d:
            mode_v = _mode_v(mot)
            from fucina.kernels import qat as KQ           # 2026-10-09 E2E_KQAT=dd: dL/dd from partial sums, exact fp32 dY
            if mode_v == 0 and KQ.active("dd") and type(mot).__name__ == "TritonEngine" \
                    and getattr(mot, "v2", False) and fam.C % fam.gs == 0 and fam.gs in (128, 256):
                dd = KQ.dd_p_triton(g32, x.contiguous(), fam, plane).to(d.dtype)
            elif mode_v == 0 or (mode_v in (1, 2) and type(mot).__name__ == "TritonEngineV"):
                eid = expert_of_row(plane, g32.shape[0])
                s_er, inv_er = scale_pot2(maxima_segment(g32.abs(), eid, fam.E))
                hi, lo = split_fp16(g32 * s_er[eid])
                if mode_v == 0:
                    dd = product_dd(mot, lo, x, fam, plane)
                    dd += product_dd(mot, hi, x, fam, plane)
                    dd = dd * inv_er.unsqueeze(-1)
                else:                              # V step: full dY^T X in the kernel, then the statistics
                    from fucina.student import plane_codes as PV
                    dd = PV.dd_v_triton(hi, x, fam, plane, mot.blk_v, mot.state.buffer(fam), mode_v, dy_lo=lo,
                                        inv=inv_er)
                del hi, lo
                dd = dd.to(d.dtype)
            else:
                dd = _dd_fp32(mot, g32, x, fam, plane, d)
    else:                                         # bf16x3
        if need_d and _mode_v(mot) != 0:
            dd = _dd_fp32(mot, g32, x, fam, plane, d)
        elif need_d:
            gh, gl = split_bf16(g32)
            xh, xl = split_bf16(x.to(torch.float32))
            dd = product_dd(mot, gl, xh, fam, plane)
            dd += product_dd(mot, gh, xl, fam, plane)
            dd += product_dd(mot, gh, xh, fam, plane)
            dd = dd.to(d.dtype)
            del xh, xl, gh, gl
    if dx is not None and oss is not None:
        oss.watch(dx)
    return dx, dd


def error_representation(g: torch.Tensor, via: str, plane=None, E: Optional[int] = None) -> dict:
    """Tests/gate only: how much the split of g (fp32) loses, relative to the row maximum (dX) and the segment
    maximum (dL/dd), in fp64."""
    g32 = g.to(torch.float32)
    out = {}
    if via == "fp16x2":
        s, inv = scale_pot2(__import__("fucina.kernels.qat", fromlist=("amax_abs",)).amax_abs(g32, 1, keepdim=True))
        hi, lo = split_fp16(g32 * s)
        r = (hi.double() + lo.double()) * inv.double()
        den = g32.abs().amax(1, keepdim=True).double().clamp_min(1e-300)
        out["rows_max_rel"] = float(((r - g32.double()).abs() / den).max())
        if plane is not None:
            eid = expert_of_row(plane, g32.shape[0])
            s_er, inv_er = scale_pot2(maxima_segment(g32.abs(), eid, E))
            hi, lo = split_fp16(g32 * s_er[eid])
            r = (hi.double() + lo.double()) * inv_er[eid].double()
            den = maxima_segment(g32.abs(), eid, E)[eid].double().clamp_min(1e-300)
            out["segments_max_rel"] = float(((r - g32.double()).abs() / den).max())
    else:
        hi, lo = split_bf16(g32)
        r = hi.double() + lo.double()
        a = g32.double().abs()
        v = a >= 2.0 ** -100                  # below this, lo is subnormal in bf16: only the row-relative error counts
        out["elements_max_rel"] = float(((r - g32.double()).abs()[v] / a[v]).max()) if bool(v.any()) else 0.0
        den = g32.abs().amax(1, keepdim=True).double().clamp_min(1e-300)
        out["rows_max_rel"] = float(((r - g32.double()).abs() / den).max())
    return out
