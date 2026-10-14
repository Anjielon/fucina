"""⛔ REJECTED (2026-10-02, GPU measurement): NOT wired into the path (`hilo_backward` remains the standard path).
On the GPU, at 8192 tokens and 256 experts, the fused launch is 3-4x SLOWER than the two launches (gate 434 -> 1659
ms dX+dd, down 279 -> 935), and dL/dd is NOT bit-exact (relative 1e-7: with a row tile of 32 instead of 64 the
compiler accumulates differently); dX, on the other hand, is bit-exact. Kept as a negative result (fused hi/lo test).

2026-10-02 TT6-fast — SPLIT fp16x2 backward pass with the two halves (hi, lo) IN THE SAME kernel
(E2E_HILO_FUSED=1; off = the standard `hilo_backward`, two launches per product).

Current path (`hilo_backward.backward_family`, fp16x2 path, triton2 engine, outside the V step) for EACH family:
  dX:  s = 2^-e(max|g| per row); hi, lo = fp16(g s), fp16(g s - hi);  y = dx2(lo); y += dx2(hi); y *= 2^e; y -> fp16
  dd:  s = 2^-e(max|g| per expert x row); hi, lo as above;            d = dd2(lo); d += dd2(hi); d *= 2^e
i.e. 4 launches (2 dX + 2 dd), 2 trit decodes for dX, X read twice for dd, and in memory g s, hi, lo,
the gather s[eid] (S x R fp32), the sums y += / d += and the multiplications by 2^e.
Here: ONE launch for dX and ONE for dd. The tile loads g (fp32) and its scale, splits hi/lo in registers
(same operations: fp32 * 2^-e exact, RNE to fp16, fp32 subtraction, RNE to fp16), and carries TWO sets of
accumulators (lo, hi) that see the SAME products in the SAME order along K (same BK = 32 for dX, same
BS = 32 tokens for dd) as the two current launches; the epilogue computes (lo + hi) * 2^e as now (same fp32 sum,
same exact product) and, for dd, the per-group sums of trit (.) acc separately for lo and hi, like the two current
launches. To keep the current register budget, the dX token tile drops from 32 to 16 and the dd row tile from 64
to 32: this does not change the summation order of any element (rows/tokens are independent).
=> BIT-EXACT with the current path (tests: fused hi/lo test, CPU interpreter + GPU on a real layer)."""
from __future__ import annotations

import os

import torch

from fucina.student import student as S

try:
    import triton
    import triton.language as tl
    from fucina.student.student import _w_piece, _t_piece, _schema_subblock, _order_groups
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

COUNTS = {"fused": 0}
BS_DX = 16
BR_DD = 32


def active() -> bool:
    return os.environ.get("E2E_HILO_FUSED", "").strip() == "1"


if _TRITON:
    @triton.jit
    def _k_dx2_hl(g_ptr, sg_s, sc_ptr, iv_ptr, pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
                  te_ptr, ts_ptr, offs_ptr, R, C, NT,
                  GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                  BS: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        """`_k_dx2` with A = fp32 g split in registers into fp16 (hi, lo) and two sets of accumulators."""
        pid_t, pid_n = _order_groups(tl.program_id(0), NT, C // 256, GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_end = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_end
        lk = tl.arange(0, BK)
        c0 = pid_n * 256
        qs, qh = _schema_subblock(c0, QK, NB, ROT)
        i = tl.arange(0, 32)
        offr = (lk * NBR)[:, None]
        off0 = offr + (qs + i)[None, :]
        off5 = offr + (qs + 32 + (i % 16))[None, :]
        off7 = offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :]
        offd = lk * NG
        offa = (ls * sg_s)[:, None] + lk[None, :]
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        g = c0 // GS
        pe = pb + e64 * sb_e
        pde = d_ptr + e64 * sd_e + g
        pa = g_ptr + s0.to(tl.int64) * sg_s
        sc = tl.load(sc_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        iv = tl.load(iv_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        h0 = tl.zeros((BS, 32), dtype=tl.float32)
        l0 = tl.zeros((BS, 32), dtype=tl.float32)
        h1 = tl.zeros((BS, 32), dtype=tl.float32)
        l1 = tl.zeros((BS, 32), dtype=tl.float32)
        h2 = tl.zeros((BS, 32), dtype=tl.float32)
        l2 = tl.zeros((BS, 32), dtype=tl.float32)
        h3 = tl.zeros((BS, 32), dtype=tl.float32)
        l3 = tl.zeros((BS, 32), dtype=tl.float32)
        h4 = tl.zeros((BS, 32), dtype=tl.float32)
        l4 = tl.zeros((BS, 32), dtype=tl.float32)
        h5 = tl.zeros((BS, 32), dtype=tl.float32)
        l5 = tl.zeros((BS, 32), dtype=tl.float32)
        h6 = tl.zeros((BS, 32), dtype=tl.float32)
        l6 = tl.zeros((BS, 32), dtype=tl.float32)
        h7 = tl.zeros((BS, 32), dtype=tl.float32)
        l7 = tl.zeros((BS, 32), dtype=tl.float32)
        for r0 in range(0, R, BK):
            mr = (r0 + lk) < R
            prow = pe + r0 * NBR
            m2 = mr[:, None]
            v0 = tl.load(prow + off0, mask=m2, other=0).to(tl.int32)
            v5 = tl.load(prow + off5, mask=m2, other=0).to(tl.int32)
            v7 = tl.load(prow + off7, mask=m2, other=0).to(tl.int32)
            pdr = pde + r0 * NG
            dA = tl.load(pdr + offd, mask=mr, other=0.0).to(tl.float16)
            if GS == 128:
                dB = tl.load(pdr + 1 + offd, mask=mr, other=0.0).to(tl.float16)
            else:
                dB = dA
            v = tl.load(pa + r0 + offa, mask=ms[:, None] & mr[None, :], other=0.0) * sc
            hi = v.to(tl.float16)
            lo = (v - hi.to(tl.float32)).to(tl.float16)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 0, GS, 0, tl.float16)
            l0 += tl.dot(lo, w)
            h0 += tl.dot(hi, w)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 1, GS, 0, tl.float16)
            l1 += tl.dot(lo, w)
            h1 += tl.dot(hi, w)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 2, GS, 0, tl.float16)
            l2 += tl.dot(lo, w)
            h2 += tl.dot(hi, w)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 3, GS, 0, tl.float16)
            l3 += tl.dot(lo, w)
            h3 += tl.dot(hi, w)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 4, GS, 0, tl.float16)
            l4 += tl.dot(lo, w)
            h4 += tl.dot(hi, w)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 5, GS, 0, tl.float16)
            l5 += tl.dot(lo, w)
            h5 += tl.dot(hi, w)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 6, GS, 0, tl.float16)
            l6 += tl.dot(lo, w)
            h6 += tl.dot(hi, w)
            w = _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 7, GS, 0, tl.float16)
            l7 += tl.dot(lo, w)
            h7 += tl.dot(hi, w)
        OT = o_ptr.dtype.element_ty
        po = o_ptr + s0.to(tl.int64) * so_s + c0
        offo = (ls * so_s)[:, None] + i[None, :]
        mo = ms[:, None]
        tl.store(po + 0 + offo, ((l0 + h0) * iv).to(OT), mask=mo)
        tl.store(po + 32 + offo, ((l1 + h1) * iv).to(OT), mask=mo)
        tl.store(po + 64 + offo, ((l2 + h2) * iv).to(OT), mask=mo)
        tl.store(po + 96 + offo, ((l3 + h3) * iv).to(OT), mask=mo)
        tl.store(po + 128 + offo, ((l4 + h4) * iv).to(OT), mask=mo)
        tl.store(po + 160 + offo, ((l5 + h5) * iv).to(OT), mask=mo)
        tl.store(po + 192 + offo, ((l6 + h6) * iv).to(OT), mask=mo)
        tl.store(po + 224 + offo, ((l7 + h7) * iv).to(OT), mask=mo)

    @triton.jit
    def _k_dd2_hl(g_ptr, sg_s, sc_ptr, iv_ptr, x_ptr, sx_s, pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
                  GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                  BR: tl.constexpr, BS: tl.constexpr):
        """`_k_dd2` with dY = fp32 g split in registers (scale per expert x row) and two sets of accumulators."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        c0 = tl.program_id(2) * 256
        s_ini = tl.load(offs_ptr + e)
        s_end = tl.load(offs_ptr + e + 1)
        r0 = pid_r * BR
        lr = tl.arange(0, BR)
        mr = (r0 + lr) < R
        ls = tl.arange(0, BS)
        i = tl.arange(0, 32)
        offy = (ls * sg_s)[:, None] + lr[None, :]
        offx = (ls * sx_s)[:, None] + i[None, :]
        pdy = g_ptr + r0
        px = x_ptr + c0
        e64 = e.to(tl.int64)
        sc = tl.load(sc_ptr + e64 * R + r0 + lr, mask=mr, other=1.0)[None, :]
        h0 = tl.zeros((BR, 32), dtype=tl.float32)
        l0 = tl.zeros((BR, 32), dtype=tl.float32)
        h1 = tl.zeros((BR, 32), dtype=tl.float32)
        l1 = tl.zeros((BR, 32), dtype=tl.float32)
        h2 = tl.zeros((BR, 32), dtype=tl.float32)
        l2 = tl.zeros((BR, 32), dtype=tl.float32)
        h3 = tl.zeros((BR, 32), dtype=tl.float32)
        l3 = tl.zeros((BR, 32), dtype=tl.float32)
        h4 = tl.zeros((BR, 32), dtype=tl.float32)
        l4 = tl.zeros((BR, 32), dtype=tl.float32)
        h5 = tl.zeros((BR, 32), dtype=tl.float32)
        l5 = tl.zeros((BR, 32), dtype=tl.float32)
        h6 = tl.zeros((BR, 32), dtype=tl.float32)
        l6 = tl.zeros((BR, 32), dtype=tl.float32)
        h7 = tl.zeros((BR, 32), dtype=tl.float32)
        l7 = tl.zeros((BR, 32), dtype=tl.float32)
        for s0 in range(s_ini, s_end, BS):
            ms = (s0 + ls) < s_end
            s64 = s_ini.to(tl.int64) + (s0 - s_ini)
            v = tl.load(pdy + s64 * sg_s + offy, mask=ms[:, None] & mr[None, :], other=0.0) * sc
            hi = v.to(tl.float16)
            lo = (v - hi.to(tl.float32)).to(tl.float16)
            hit = tl.trans(hi)
            lot = tl.trans(lo)
            pxs = px + s64 * sx_s + offx
            mx = ms[:, None]
            xx = tl.load(pxs + 0, mask=mx, other=0.0)
            l0 += tl.dot(lot, xx)
            h0 += tl.dot(hit, xx)
            xx = tl.load(pxs + 32, mask=mx, other=0.0)
            l1 += tl.dot(lot, xx)
            h1 += tl.dot(hit, xx)
            xx = tl.load(pxs + 64, mask=mx, other=0.0)
            l2 += tl.dot(lot, xx)
            h2 += tl.dot(hit, xx)
            xx = tl.load(pxs + 96, mask=mx, other=0.0)
            l3 += tl.dot(lot, xx)
            h3 += tl.dot(hit, xx)
            xx = tl.load(pxs + 128, mask=mx, other=0.0)
            l4 += tl.dot(lot, xx)
            h4 += tl.dot(hit, xx)
            xx = tl.load(pxs + 160, mask=mx, other=0.0)
            l5 += tl.dot(lot, xx)
            h5 += tl.dot(hit, xx)
            xx = tl.load(pxs + 192, mask=mx, other=0.0)
            l6 += tl.dot(lot, xx)
            h6 += tl.dot(hit, xx)
            xx = tl.load(pxs + 224, mask=mx, other=0.0)
            l7 += tl.dot(lot, xx)
            h7 += tl.dot(hit, xx)
        qs, qh = _schema_subblock(c0, QK, NB, ROT)
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        prow = pb + e64 * sb_e + r0.to(tl.int64) * NBR
        offr = (lr * NBR)[:, None]
        m2 = mr[:, None]
        v0 = tl.load(prow + offr + (qs + i)[None, :], mask=m2, other=0).to(tl.int32)
        v5 = tl.load(prow + offr + (qs + 32 + (i % 16))[None, :], mask=m2, other=0).to(tl.int32)
        v7 = tl.load(prow + offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :], mask=m2,
                     other=0).to(tl.int32)
        L0 = l0 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 0, 0).to(tl.float32)
        L0 += l1 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 1, 0).to(tl.float32)
        L0 += l2 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 2, 0).to(tl.float32)
        L0 += l3 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 3, 0).to(tl.float32)
        L1 = l4 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 4, 0).to(tl.float32)
        L1 += l5 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 5, 0).to(tl.float32)
        L1 += l6 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 6, 0).to(tl.float32)
        L1 += l7 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 7, 0).to(tl.float32)
        H0 = h0 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 0, 0).to(tl.float32)
        H0 += h1 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 1, 0).to(tl.float32)
        H0 += h2 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 2, 0).to(tl.float32)
        H0 += h3 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 3, 0).to(tl.float32)
        H1 = h4 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 4, 0).to(tl.float32)
        H1 += h5 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 5, 0).to(tl.float32)
        H1 += h6 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 6, 0).to(tl.float32)
        H1 += h7 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 7, 0).to(tl.float32)
        g = c0 // GS
        iv = tl.load(iv_ptr + e64 * R + r0 + lr, mask=mr, other=1.0)
        pdd = dd_ptr + e64 * sdd_e + r0.to(tl.int64) * NG + g + lr * NG
        if GS == 128:
            tl.store(pdd, (tl.sum(L0, axis=1) + tl.sum(H0, axis=1)) * iv, mask=mr)
            tl.store(pdd + 1, (tl.sum(L1, axis=1) + tl.sum(H1, axis=1)) * iv, mask=mr)
        else:
            tl.store(pdd, (tl.sum(L0 + L1, axis=1) + tl.sum(H0 + H1, axis=1)) * iv, mask=mr)


def dx_fused(g32: torch.Tensor, d: torch.Tensor, fam, plane, blk, out_dtype) -> torch.Tensor:
    """dX = fp((dx2(lo) + dx2(hi)) * 2^e) with per-row scale, in ONE launch. g32 (S, R) contiguous fp32."""
    from fucina.kernels import hilo_backward as HL
    s, inv = HL.scale_pot2(g32.abs().amax(1))
    S._check_v2(fam)
    d = d.contiguous()
    out = torch.empty(g32.shape[0], fam.C, dtype=out_dtype, device=g32.device)
    te, ts, n_t = plane.table(BS_DX)
    _k_dx2_hl[(n_t * (fam.C // 256),)](g32, g32.stride(0), s.contiguous(), inv.contiguous(), fam.blocks,
                                       fam.blocks.stride(0), fam.NBR, d, d.stride(0), fam.NG, out, out.stride(0),
                                       te, ts, plane.offs, fam.R, fam.C, n_t, fam.gs, fam.qk, fam.nb,
                                       int(fam.rotated), BS_DX, blk.BK_dx2, blk.GROUP_M, num_warps=blk.num_warps2,
                                       **S._stages(blk.stages_dx2))
    COUNTS["fused"] += 1
    return out


def dd_fused(g32: torch.Tensor, x: torch.Tensor, fam, plane, blk) -> torch.Tensor:
    """dL/dd = (dd2(lo) + dd2(hi)) * 2^e with scale per expert x row, in ONE launch. -> (E, R, NG) fp32."""
    from fucina.kernels import hilo_backward as HL
    eid = HL.expert_of_row(plane, g32.shape[0])
    s_er, inv_er = HL.scale_pot2(HL.maxima_segment(g32.abs(), eid, fam.E))
    S._check_v2(fam)
    x = x.contiguous()
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=g32.device)
    grid = (fam.E, triton.cdiv(fam.R, BR_DD), fam.C // 256)
    _k_dd2_hl[grid](g32, g32.stride(0), s_er.contiguous(), inv_er.contiguous(), x, x.stride(0), fam.blocks,
                       fam.blocks.stride(0), fam.NBR, dd, dd.stride(0), fam.NG, plane.offs, fam.R, fam.C,
                       fam.gs, fam.qk, fam.nb, int(fam.rotated), BR_DD, blk.BS_dd2, num_warps=blk.num_warps2,
                       **S._stages(blk.stages_dd2))
    COUNTS["fused"] += 1
    return dd


def allowed(mot, fam, x) -> bool:
    """Only where the current path is the two triton2 launches outside the V step, with the same tiles along K."""
    from fucina.kernels import hilo_backward as HL
    blk = getattr(mot, "blk", None)
    return (_TRITON and active() and getattr(mot, "v2", False) and not getattr(mot, "dense", False)
            and HL._mode_v(mot) == 0 and blk is not None and blk.BK_dx2 == 32 and blk.BS_dd2 == 32
            and x.dtype == torch.float16 and (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"))
