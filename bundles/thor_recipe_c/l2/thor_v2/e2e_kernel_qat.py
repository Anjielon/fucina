#!/usr/bin/env python3
"""e2e_kernel_qat — faster kernels for training ONLY the group scales of a frozen ternary network (DI1 pari / RILQ /
TT6 share this student). New file (9/10/2026); report: ~/tony-forgia/ricerche/KERNEL_QAT_SOTTO_3H.md.

Switch: E2E_KQAT (comma list, default empty = nothing changes, old kernels stay as the reference):
  dd   scale gradient of the expert families from FORWARD PARTIAL SUMS (see below) instead of the split dY^T X;
  dx   expert dX with the hi/lo halves of dY in ONE kernel (one weight decode, one fp32 accumulator).

## dd from forward partial sums (the gradient of a group scale needs no dW)
For one expert e with weight W[r, c] = d[r, g(c)] T[r, c] (T in {-1, 0, 1}, d fp16, g(c) = c // GS):
    Y[t, r]          = sum_c X[t, c] W[r, c] = sum_g d[r, g] P[t, r, g],   P[t, r, g] = sum_{c in g} X[t, c] T[r, c]
    dL/dd[r, g]      = sum_t dY[t, r] P[t, r, g]
P is the forward product restricted to one group (K = GS columns: one WMMA product with fp16 operands that are EXACT:
X is the fp16 activation of the forward, T is an integer). The contraction over tokens is an fp32 x fp32 epilogue
(dY stays fp32: no hi/lo split at all) costing 1/GS of the product. Work: ONE fp16 GEMM the size of the forward,
instead of the two split GEMMs of today (hi and lo of dY), and the scale gradient is computed from the exact dY.

Error bound (Higham ch. 3, u = 2^-24, gamma_n = n u / (1 - n u)); the kernel splits a group in sub-groups of 128
columns (1 for GS 128, 2 for GS 256), accumulates each P_sub in fp32 inside the product and the token sum in fp32:
    |P_sub_hat - P_sub|    <= gamma_128 sum_{c in sub} |X[t, c]|                  (products X T exact)
    |dd_hat - dd|          <= gamma_{k N + 1} sum_t |dY| sum_sub |P_sub| + gamma_128 (1 + gamma_{k N+1}) sum_t |dY| sum_c |X|
with N = tokens of the expert, k = GS / 128. Today's split route: |dd_old - dd| <= 2^-22 sum_t |dY| |X T| (representation of dY in
two fp16, relative to the segment maximum) + gamma terms of the same order. Both are fp32-accurate; the new one does
not round dY at all. Tests: e2e_test_kernel_qat.py (fp64 reference on real GGUF tensors).

Program layout: one program per (row tile of BR rows, expert); the grid puts the row tiles of the SAME expert next
to each other, so the expert's X rows (tokens x C fp16) are read from L2 by all its row tiles. Inside: groups outer,
tokens inner; the trits of the (GS x BR) group tile are decoded ONCE per group (same integer arithmetic as
dequantize_row_tq1_0, per-column byte offsets and multipliers computed on a GS-long vector), the fp32 scale gradient
of the group is accumulated over the expert's tokens and written once: deterministic, no atomics.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

NOTI = frozenset({"dd", "dx", "caldi", "cache_densi", "flash1", "flash3", "x1", "piega1", "testa1", "gdn1", "gdn3", "densi_k", "esperti1", "testa3", "attn1", "attn3", "densi_blas", "silu", "mm16_grande", "conv32", "tile", "guardia", "amax", "prep"})
ATTIVI: set = set()


def imposta(voci) -> None:
    """Switch the new kernels on/off at run time (tests and gates compare both in one process)."""
    v = {x.strip() for x in (voci.split(",") if isinstance(voci, str) else voci) if x and x.strip()}
    if v - NOTI:
        raise ValueError(f"E2E_KQAT={sorted(v)}: known {sorted(NOTI)}")
    ATTIVI.clear()
    ATTIVI.update(v)


imposta(os.getenv("E2E_KQAT", ""))


def attivo(nome: str) -> bool:
    return nome in ATTIVI


GRAFO = {"n": int(os.getenv("E2E_KQAT_GRAFO", "0") or 0)}
MM16_GRANDE = tuple(int(x) for x in os.getenv("E2E_KQAT_MM16", "128,128,32,8").split(","))   # BM, BN, BK, warps
# E2E_KQAT=tile: tiles measured on gfx1151 (kernel_qat/misure/bench_tile.json, layer 5, 65536 assignments), all checked
# bit for bit against the defaults (the K order of every element does not depend on these tiles):
#   forward _k_gemm2: BS 64, BN 128, 8 warps, 1 stage  -> 15.8 / 15.7 TFLOPS (gate+up / down) instead of 11.4 / 12.5
#   dX split 3: BS 32, 4 warps (8.2 / 7.2 TF instead of 4.2 / 3.7); split 1: BS 64, 4 warps (13.7 TF instead of 5.9)
#   dd (_k_ddp): BR 64, BS 32, 4 warps (11.8 / 13.1 TF)
TILE = {"gemm2": (64, 128, 8, 1), "dx3": (32, 4), "dx1": (64, 4), "dd": (64, 32, 4)}


def blk_avanti(blk):
    """Blocchi with the measured forward tiles (E2E_KQAT=tile), otherwise unchanged."""
    if not attivo("tile"):
        return blk
    import dataclasses
    BS, BN, w, st = TILE["gemm2"]
    return dataclasses.replace(blk, BS2=BS, BN2=BN, num_warps2=w, stadi_gemm2=st)


def strati_con_grafo() -> int:
    """How many final layers keep the forward graph (no recompute in the backward); E2E_KQAT_GRAFO, default 0."""
    return GRAFO["n"]


# ── cache_densi: the folded x2 weight (trits, hi, lo, inv) of a multi-plane dense depends only on the bytes and on the
# scales fp16(d0 + theta); theta changes only at the optimizer step, so within a step the 16 micro-batches x (forward,
# recompute) rebuilt the SAME tensors 32 times. Memoised per dense module, keyed by the identity and version counter of
# bytes, d0 and theta of every plane (an in-place update of any of them bumps `_version`): same inputs -> same
# tensors, bit for bit. LineareTernaria sets CHIAVE_THETA around the call.
CACHE_DENSI: dict = {}
CHIAVE_THETA = None


def _chiave_densi(fams):
    if not attivo("cache_densi") or CHIAVE_THETA is None:
        return None
    return tuple((id(f.blocchi), f.blocchi._version, id(f.d0), f.d0._version) for f in fams) + (CHIAVE_THETA,)


def cache_densi_prendi(fams, modo):
    k = _chiave_densi(fams)
    if k is None:
        return None
    v = CACHE_DENSI.get(k[0][0])
    return v[1] if v is not None and v[0] == k else None


def cache_densi_metti(fams, modo, pesi) -> None:
    k = _chiave_densi(fams)
    if k is not None:
        CACHE_DENSI[k[0][0]] = (k, pesi)


def tabella(piano, BS: int, fams) -> tuple:
    """Tile table of the routing plan; with E2E_KQAT=caldi and a frozen 2nd/3rd expert plane (`fam.caldi`), the tiles
    of experts WITHOUT that plane are dropped (expert -1: the program returns at once) and the caller zero-fills the
    output. Today those tiles multiply byte 0 by scale 0: every product is +-0, so the result differs at most in the
    sign of a zero (y + 0 versus y + (-0)), which no later add/multiply can see. -> (te, ts, n, skipped?)."""
    te, ts, n = piano.tabella(BS)
    caldi = getattr(fams[0], "caldi", None) if attivo("caldi") else None
    if caldi is None or any(getattr(f, "caldi", None) is not caldi for f in fams):
        return te, ts, n, False
    chiave = ("caldi", BS, id(caldi))
    v = piano._tabelle.get(chiave)
    if v is None:
        tel = te.long()
        ok = (tel >= 0) & caldi[tel.clamp(min=0)]
        v = piano._tabelle[chiave] = (torch.where(ok, te, torch.full_like(te, -1)).contiguous(), ts, n)
    return v[0], v[1], v[2], True


if _TRITON:
    @triton.jit
    def _dec(v, pw):
        """((((b * 3^n) & 255) * 3) >> 8) - 1, the integer trit decode of dequantize_row_tq1_0."""
        return ((((v * pw) & 255) * 3) >> 8) - 1

    @triton.jit
    def _colonne_gruppo(c0, GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr):
        """Byte offset (inside a weight row) and base-3 multiplier of the GS columns c0..c0+GS-1 (c0 multiple of GS,
        GS in {128, 256}: the group never crosses a 256-column sub-block). ggml-common.h layout: TQ1_0 = qs[48] qh[4]
        d; TQ1_0R_1024 = 4 quarters (qs[4][48], qh[4][4]) + d[8]. In a sub-block: column j < 160 -> qs[j % 32] x
        3^(j // 32); 160 <= j < 240 -> qs[32 + (j - 160) % 16] x 3^((j - 160) // 16); j >= 240 -> qh[(j - 240) % 4] x
        3^((j - 240) // 4)."""
        b0 = (c0 // QK) * NB
        if ROT:
            q = (c0 % QK) // 256
            qs = b0 + q * 48
            qh = b0 + 192 + q * 4
        else:
            qs = b0
            qh = b0 + 48
        j = (c0 % 256) + tl.arange(0, GS)
        in1 = j < 160
        in2 = j < 240
        n = tl.where(in1, j // 32, tl.where(in2, (j - 160) // 16, (j - 240) // 4))
        pw = tl.where(n == 0, 1, tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))))
        ob = tl.where(in1, qs + (j % 32), tl.where(in2, qs + 32 + ((j - 160) % 16), qh + ((j - 240) % 4)))
        return ob, pw

    @triton.jit
    def _pezzo_gruppo(prow, offr, mr, c0, P: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr):
        """Trits (32 x BR, fp16, exact) of the 32 columns c0 + 32 P .. + 31 for the BR rows at prow (byte offsets of
        the rows in offr): per-column byte offset and base-3 multiplier on a 32-long vector (dequantize_row_tq1_0)."""
        c = c0 + P * 32
        b0 = (c // QK) * NB
        if ROT:
            q = (c % QK) // 256
            qs = b0 + q * 48
            qh = b0 + 192 + q * 4
        else:
            qs = b0
            qh = b0 + 48
        j = (c % 256) + tl.arange(0, 32)
        in1 = j < 160
        in2 = j < 240
        n = tl.where(in1, j // 32, tl.where(in2, (j - 160) // 16, (j - 240) // 4))
        pw = tl.where(n == 0, 1, tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))))
        ob = tl.where(in1, qs + (j % 32), tl.where(in2, qs + 32 + ((j - 160) % 16), qh + ((j - 240) % 4)))
        v = tl.load(prow + ob[:, None] + offr, mask=mr[None, :], other=0).to(tl.int32)
        return _dec(v, pw[:, None]).to(tl.float16)

    @triton.jit
    def _k_ddp(dy_ptr, sdy, x_ptr, sx, pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R,
               GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
               BR: tl.constexpr, BS: tl.constexpr):
        """dL/dd[e, r, g] = sum_{t in e} dY[t, r] * (sum_{c in g} X[t, c] T[e, r, c]); dY fp32, X fp16, T from bytes.
        The trits of a sub-group of 128 columns are decoded ONCE per program as 4 fp16 pieces of 32 x BR (the int32
        decode temporaries live only during the decode); P = sum over pieces of X_piece T_piece^T. A group of 256
        (TQ1_0) is two sub-groups: dd[r, g] = sum_t dY (P_1 + P_2) = sum_t dY P_1 + sum_t dY P_2 (linear in P),
        so only 4 pieces are ever live (AOT gfx1151: 0 spills at BR 64, BS 32, 8 warps, 1 stage)."""
        pid_r = tl.program_id(0)
        e = tl.program_id(1)
        s_ini = tl.load(offs_ptr + e)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        r0 = pid_r * BR
        lr = tl.arange(0, BR)
        mr = (r0 + lr) < R
        ls = tl.arange(0, BS)
        l32 = tl.arange(0, 32)
        prow = pb + e64 * sb_e + r0.to(tl.int64) * NBR
        offr = (lr * NBR)[None, :]
        pdd = dd_ptr + e64 * sdd_e + (r0 + lr).to(tl.int64) * NG
        for g in range(0, NG):
            acc = tl.zeros((BR,), dtype=tl.float32)
            for h in tl.static_range(GS // 128):          # GS 256 = two sub-groups of 128: dd is linear in P
                c0 = g * GS + h * 128
                w0 = _pezzo_gruppo(prow, offr, mr, c0, 0, QK, NB, ROT)
                w1 = _pezzo_gruppo(prow, offr, mr, c0, 1, QK, NB, ROT)
                w2 = _pezzo_gruppo(prow, offr, mr, c0, 2, QK, NB, ROT)
                w3 = _pezzo_gruppo(prow, offr, mr, c0, 3, QK, NB, ROT)
                for s0 in range(s_ini, s_fine, BS):
                    ms = (s0 + ls) < s_fine
                    s64 = (s0 + ls).to(tl.int64)
                    px = x_ptr + s64[:, None] * sx + c0 + l32[None, :]
                    mx = ms[:, None]
                    p = tl.dot(tl.load(px, mask=mx, other=0.0), w0)
                    p = tl.dot(tl.load(px + 32, mask=mx, other=0.0), w1, p)
                    p = tl.dot(tl.load(px + 64, mask=mx, other=0.0), w2, p)
                    p = tl.dot(tl.load(px + 96, mask=mx, other=0.0), w3, p)
                    dy = tl.load(dy_ptr + s64[:, None] * sdy + r0 + lr[None, :], mask=mx & mr[None, :], other=0.0)
                    acc += tl.sum(dy * p, axis=0)
            tl.store(pdd + g, acc, mask=mr)


def dd_p_triton(dY: torch.Tensor, X: torch.Tensor, fam, piano, BR: int = 64, BS: int = 32,
                num_warps: int = 4, num_stages: int = 1) -> torch.Tensor:
    """dL/dd (E, R, NG) fp32 of one family; dY (S, R) fp32 (exact, any scale), X (S, C) fp16, rows in expert order."""
    assert dY.dtype == torch.float32 and X.dtype == torch.float16, (dY.dtype, X.dtype)
    assert dY.shape[0] == X.shape[0] and dY.shape[1] == fam.R and X.shape[1] == fam.C
    assert fam.C % fam.gs == 0 and fam.gs in (128, 256), (fam.C, fam.gs)
    dY, X = dY.contiguous(), X.contiguous()
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=dY.device)
    griglia = (triton.cdiv(fam.R, BR), fam.E)
    _k_ddp[griglia](dY, dY.stride(0), X, X.stride(0), fam.blocchi, fam.blocchi.stride(0), fam.NBR,
                    dd, dd.stride(0), fam.NG, piano.offs, fam.R,
                    fam.gs, fam.qk, fam.nb, int(fam.ruotata), BR, BS, num_warps=num_warps, num_stages=num_stages)
    return dd


def dd_riferimento64(dY: torch.Tensor, X: torch.Tensor, fam, piano, esperti=None) -> torch.Tensor:
    """fp64 reference: sum_t dY[t, r] sum_{c in g} X[t, c] T[r, c], expert by expert (CPU or GPU); `esperti`: only
    those (the others stay 0)."""
    import e2e_studente as S
    off = piano.offs.tolist()
    out = torch.zeros(fam.E, fam.R, fam.NG, dtype=torch.float64, device=dY.device)
    for e in (range(fam.E) if esperti is None else esperti):
        a, b = off[e], off[e + 1]
        if a == b:
            continue
        T = S.trit_esperto(fam, e).to(torch.float64)                       # (R, C)
        G = dY[a:b].to(torch.float64).T @ X[a:b].to(torch.float64)          # (R, C)
        out[e] = (G * T).view(fam.R, fam.NG, fam.gs).sum(-1)
    return out


def limite_dd(dY: torch.Tensor, X: torch.Tensor, fam, piano, esperti=None) -> torch.Tensor:
    """Deterministic worst-case bound (E, R, NG) of |dd_p_triton - dd| (header of the module), fp64."""
    import e2e_studente as S
    # unit roundoff: 2^-24 (round to nearest); on the GPU 2^-23, because the WMMA accumulation of gfx1151 truncates
    # (TECNICHE_E2E_PV §17: ~2e-8 drift per 16-step) and a truncation error is up to one full ulp
    u = 2.0 ** -23 if dY.is_cuda else 2.0 ** -24
    off = piano.offs.tolist()
    out = torch.zeros(fam.E, fam.R, fam.NG, dtype=torch.float64, device=dY.device)
    sub = 128                                                               # the kernel's sub-group (see _k_ddp)
    nsub = fam.gs // sub
    gGS = sub * u / (1 - sub * u)
    for e in (range(fam.E) if esperti is None else esperti):
        a, b = off[e], off[e + 1]
        if a == b:
            continue
        n = b - a
        m = nsub * n + 1
        gN = m * u / (1 - m * u)
        T = S.trit_esperto(fam, e).to(torch.float64)
        x = X[a:b].to(torch.float64)
        ady = dY[a:b].to(torch.float64).abs()                               # (n, R)
        # sum over the sub-groups of |P_sub[t, r]| per group (n, R, NG)
        absP = torch.stack([sum((x[:, c:c + sub] @ T[:, c:c + sub].T).abs()
                                for c in range(g * fam.gs, (g + 1) * fam.gs, sub)) for g in range(fam.NG)], -1)
        sx = torch.stack([x[:, g * fam.gs:(g + 1) * fam.gs].abs().sum(1) for g in range(fam.NG)], -1)   # (n, NG)
        out[e] = gN * torch.einsum("tr,trg->rg", ady, absP) + gGS * (1 + gN) * (ady.T @ sx)
    return out


# ═══════════════════════════════════ dX with the hi/lo halves in ONE kernel ═══════════════════════════════════
# Today (e2e_indietro_hilo.dx_diviso, fp16x2): y = dx(lo); y += dx(hi); y *= inv, two launches of _k_dx2 that decode
# the same trits twice. Here one program keeps TWO sets of accumulators (lo and hi), each updated with exactly the
# products and the K order of _k_dx2 (BK = 32 steps along R, 8 pieces of 32 columns), and stores
# (acc_lo + acc_hi) * inv[row] in fp32: the same fp32 operations in the same order -> bit-identical to today
# (checked by the tests), with one decode and one read of the weights.
if _TRITON:
    import e2e_studente as _S

    @triton.jit
    def _k_dx2hl(a_ptr, a2_ptr, sa_s, inv_ptr, pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
                 te_ptr, ts_ptr, offs_ptr, R, C, NT,
                 GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                 BS: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, IN32: tl.constexpr,
                 SPLIT: tl.constexpr):
        """IN32 = 0: a_ptr = hi, a2_ptr = lo (fp16, already split). IN32 = 1: a_ptr = dY fp32, a2_ptr = the per-row
        power-of-two scale s; the split v = dY s, hi = fp16(v), lo = fp16(v - hi) happens in registers (the same
        operations as e2e_indietro_hilo.dividi_fp16: same hi, lo, same products). SPLIT 1: hi only (fp16 tier)."""
        DT = tl.float16
        pid_t, pid_n = _S._ordine_gruppi(tl.program_id(0), NT, C // 256, GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_fine
        lk = tl.arange(0, BK)
        c0 = pid_n * 256
        qs, qh = _S._schema_sottoblocco(c0, QK, NB, ROT)
        i = tl.arange(0, 32)
        offr = (lk * NBR)[:, None]
        off0 = offr + (qs + i)[None, :]
        off5 = offr + (qs + 32 + (i % 16))[None, :]
        off7 = offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :]
        offd = lk * NG
        offa = (ls * sa_s)[:, None] + lk[None, :]
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        g = c0 // GS
        pe = pb + e64 * sb_e
        pde = d_ptr + e64 * sd_e + g
        pa = a_ptr + s0.to(tl.int64) * sa_s
        if IN32:
            pa2 = a_ptr
            sv = tl.load(a2_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        else:
            pa2 = a2_ptr + s0.to(tl.int64) * sa_s
        l0 = tl.zeros((BS, 32), dtype=tl.float32)
        l1 = tl.zeros((BS, 32), dtype=tl.float32)
        l2 = tl.zeros((BS, 32), dtype=tl.float32)
        l3 = tl.zeros((BS, 32), dtype=tl.float32)
        l4 = tl.zeros((BS, 32), dtype=tl.float32)
        l5 = tl.zeros((BS, 32), dtype=tl.float32)
        l6 = tl.zeros((BS, 32), dtype=tl.float32)
        l7 = tl.zeros((BS, 32), dtype=tl.float32)
        h0 = tl.zeros((BS, 32), dtype=tl.float32)
        h1 = tl.zeros((BS, 32), dtype=tl.float32)
        h2 = tl.zeros((BS, 32), dtype=tl.float32)
        h3 = tl.zeros((BS, 32), dtype=tl.float32)
        h4 = tl.zeros((BS, 32), dtype=tl.float32)
        h5 = tl.zeros((BS, 32), dtype=tl.float32)
        h6 = tl.zeros((BS, 32), dtype=tl.float32)
        h7 = tl.zeros((BS, 32), dtype=tl.float32)
        for r0 in range(0, R, BK):
            mr = (r0 + lk) < R
            prow = pe + r0 * NBR
            m2 = mr[:, None]
            v0 = tl.load(prow + off0, mask=m2, other=0).to(tl.int32)
            v5 = tl.load(prow + off5, mask=m2, other=0).to(tl.int32)
            v7 = tl.load(prow + off7, mask=m2, other=0).to(tl.int32)
            pdr = pde + r0 * NG
            dA = tl.load(pdr + offd, mask=mr, other=0.0).to(DT)
            if GS == 128:
                dB = tl.load(pdr + 1 + offd, mask=mr, other=0.0).to(DT)
            else:
                dB = dA
            ma = ms[:, None] & mr[None, :]
            if IN32:
                v = tl.load(pa + r0 + offa, mask=ma, other=0.0) * sv
                b = v.to(DT)                                         # hi
                a = (v - b.to(tl.float32)).to(DT)                    # lo
            else:
                a = tl.load(pa2 + r0 + offa, mask=ma, other=0.0)   # lo
                b = tl.load(pa + r0 + offa, mask=ma, other=0.0)    # hi
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 0, GS, 0, DT)
            if SPLIT == 3:
                l0 += tl.dot(a, w)
            h0 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 1, GS, 0, DT)
            if SPLIT == 3:
                l1 += tl.dot(a, w)
            h1 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 2, GS, 0, DT)
            if SPLIT == 3:
                l2 += tl.dot(a, w)
            h2 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 3, GS, 0, DT)
            if SPLIT == 3:
                l3 += tl.dot(a, w)
            h3 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 4, GS, 0, DT)
            if SPLIT == 3:
                l4 += tl.dot(a, w)
            h4 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 5, GS, 0, DT)
            if SPLIT == 3:
                l5 += tl.dot(a, w)
            h5 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 6, GS, 0, DT)
            if SPLIT == 3:
                l6 += tl.dot(a, w)
            h6 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 7, GS, 0, DT)
            if SPLIT == 3:
                l7 += tl.dot(a, w)
            h7 += tl.dot(b, w)
        inv = tl.load(inv_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        po = o_ptr + s0.to(tl.int64) * so_s + c0
        offo = (ls * so_s)[:, None] + i[None, :]
        mo = ms[:, None]
        OT = o_ptr.dtype.element_ty
        tl.store(po + offo, ((l0 + h0) * inv).to(OT), mask=mo)
        tl.store(po + 32 + offo, ((l1 + h1) * inv).to(OT), mask=mo)
        tl.store(po + 64 + offo, ((l2 + h2) * inv).to(OT), mask=mo)
        tl.store(po + 96 + offo, ((l3 + h3) * inv).to(OT), mask=mo)
        tl.store(po + 128 + offo, ((l4 + h4) * inv).to(OT), mask=mo)
        tl.store(po + 160 + offo, ((l5 + h5) * inv).to(OT), mask=mo)
        tl.store(po + 192 + offo, ((l6 + h6) * inv).to(OT), mask=mo)
        tl.store(po + 224 + offo, ((l7 + h7) * inv).to(OT), mask=mo)


if _TRITON:
    @triton.jit
    def _k_dx2hl4(a_ptr, a2_ptr, sa_s, inv_ptr, pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
                  te_ptr, ts_ptr, offs_ptr, R, C, NT,
                  GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                  BS: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, IN32: tl.constexpr, P0: tl.constexpr):
        """_k_dx2hl (split 3) on HALF of each 256-column sub-block: pieces P0..P0+3 (P0 = 0 or 4), so 8 accumulators
        instead of 16 (AOT gfx1151: the 16 spilled 66-108 VGPR). Two launches (P0 = 0, 4) give the same products in the
        same order per output element as _k_dx2hl and as today's two _k_dx2 launches: bit for bit (tests K2)."""
        DT = tl.float16
        pid_t, pid_n = _S._ordine_gruppi(tl.program_id(0), NT, C // 256, GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_fine
        lk = tl.arange(0, BK)
        c0 = pid_n * 256
        qs, qh = _S._schema_sottoblocco(c0, QK, NB, ROT)
        i = tl.arange(0, 32)
        offr = (lk * NBR)[:, None]
        off0 = offr + (qs + i)[None, :]
        off5 = offr + (qs + 32 + (i % 16))[None, :]
        off7 = offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :]
        offd = lk * NG
        offa = (ls * sa_s)[:, None] + lk[None, :]
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        g = c0 // GS
        pe = pb + e64 * sb_e
        pde = d_ptr + e64 * sd_e + g
        pa = a_ptr + s0.to(tl.int64) * sa_s
        if IN32:
            pa2 = a_ptr
            sv = tl.load(a2_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        else:
            pa2 = a2_ptr + s0.to(tl.int64) * sa_s
        l0 = tl.zeros((BS, 32), dtype=tl.float32)
        l1 = tl.zeros((BS, 32), dtype=tl.float32)
        l2 = tl.zeros((BS, 32), dtype=tl.float32)
        l3 = tl.zeros((BS, 32), dtype=tl.float32)
        h0 = tl.zeros((BS, 32), dtype=tl.float32)
        h1 = tl.zeros((BS, 32), dtype=tl.float32)
        h2 = tl.zeros((BS, 32), dtype=tl.float32)
        h3 = tl.zeros((BS, 32), dtype=tl.float32)
        for r0 in range(0, R, BK):
            mr = (r0 + lk) < R
            prow = pe + r0 * NBR
            m2 = mr[:, None]
            v0 = tl.load(prow + off0, mask=m2, other=0).to(tl.int32)
            v5 = tl.load(prow + off5, mask=m2, other=0).to(tl.int32)
            v7 = tl.load(prow + off7, mask=m2, other=0).to(tl.int32)
            pdr = pde + r0 * NG
            dA = tl.load(pdr + offd, mask=mr, other=0.0).to(DT)
            if GS == 128:
                dB = tl.load(pdr + 1 + offd, mask=mr, other=0.0).to(DT)
            else:
                dB = dA
            ma = ms[:, None] & mr[None, :]
            if IN32:
                v = tl.load(pa + r0 + offa, mask=ma, other=0.0) * sv
                b = v.to(DT)
                a = (v - b.to(tl.float32)).to(DT)
            else:
                a = tl.load(pa2 + r0 + offa, mask=ma, other=0.0)
                b = tl.load(pa + r0 + offa, mask=ma, other=0.0)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, P0 + 0, GS, 0, DT)
            l0 += tl.dot(a, w)
            h0 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, P0 + 1, GS, 0, DT)
            l1 += tl.dot(a, w)
            h1 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, P0 + 2, GS, 0, DT)
            l2 += tl.dot(a, w)
            h2 += tl.dot(b, w)
            w = _S._w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, P0 + 3, GS, 0, DT)
            l3 += tl.dot(a, w)
            h3 += tl.dot(b, w)
        inv = tl.load(inv_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        po = o_ptr + s0.to(tl.int64) * so_s + c0 + P0 * 32
        offo = (ls * so_s)[:, None] + i[None, :]
        mo = ms[:, None]
        OT = o_ptr.dtype.element_ty
        tl.store(po + offo, ((l0 + h0) * inv).to(OT), mask=mo)
        tl.store(po + 32 + offo, ((l1 + h1) * inv).to(OT), mask=mo)
        tl.store(po + 64 + offo, ((l2 + h2) * inv).to(OT), mask=mo)
        tl.store(po + 96 + offo, ((l3 + h3) * inv).to(OT), mask=mo)


def _lancia_dx(a, a2, sa, inv, fam, d, out, te, ts, n_t, piano, blk, in32: int, split: int) -> None:
    import e2e_studente as S
    comuni = (fam.blocchi, fam.blocchi.stride(0), fam.NBR, d, d.stride(0), fam.NG, out, out.stride(0), te, ts,
              piano.offs, fam.R, fam.C, n_t, fam.gs, fam.qk, fam.nb, int(fam.ruotata), blk.BS_dx2, blk.BK_dx2,
              blk.GROUP_M, in32)
    w = blk.num_warps2
    if attivo("tile"):                                # measured tiles: the tile table was built with blk.BS_dx2
        w = TILE["dx3" if split == 3 else "dx1"][1]
    if split == 3 and DX_META["attiva"]:              # two half launches: 8 accumulators, no spills
        for p0 in (0, 4):
            _k_dx2hl4[(n_t * (fam.C // 256),)](a, a2, sa, inv, *comuni, p0, num_warps=w,
                                               **S._stadi(blk.stadi_dx2))
    else:
        _k_dx2hl[(n_t * (fam.C // 256),)](a, a2, sa, inv, *comuni, split, num_warps=w,
                                          **S._stadi(blk.stadi_dx2))


DX_META = {"attiva": os.getenv("E2E_KQAT_DX_META", "1") == "1"}


def dx_hl_triton(hi: torch.Tensor, lo: torch.Tensor, inv: torch.Tensor, fam, d: torch.Tensor, piano, blk,
                 out_dtype=torch.float32) -> torch.Tensor:
    """(dx(lo) + dx(hi)) * inv[row] in one launch; hi, lo (S, R) fp16, inv (S,) or (S, 1) fp32 (powers of two)."""
    import e2e_studente as S
    hi, lo, d = hi.contiguous(), lo.contiguous(), d.contiguous()
    inv = inv.reshape(-1).contiguous()
    S._controlla_v2(fam)
    assert hi.shape == lo.shape and hi.shape[1] == fam.R and hi.stride() == lo.stride()
    te, ts, n_t, salta = tabella(piano, blk.BS_dx2, [fam])
    out = (torch.zeros if salta else torch.empty)(hi.shape[0], fam.C, dtype=out_dtype, device=hi.device)
    _lancia_dx(hi, lo, hi.stride(0), inv, fam, d, out, te, ts, n_t, piano, blk, 0, 3)
    return out


def dx_s_triton(g32: torch.Tensor, s: torch.Tensor, inv: torch.Tensor, fam, d: torch.Tensor, piano, blk,
                split: int = 3, out_dtype=torch.float32) -> torch.Tensor:
    """Same result as dx_hl_triton on (hi, lo) = dividi_fp16(g32 * s), with the split done in registers (no hi/lo
    arrays in memory); split 1 = hi only (fp16 tier). g32 (S, R) fp32, s and inv (S,) powers of two."""
    import e2e_studente as S
    g32, d = g32.contiguous(), d.contiguous()
    s = s.reshape(-1).to(torch.float32).contiguous()
    inv = inv.reshape(-1).to(torch.float32).contiguous()
    S._controlla_v2(fam)
    assert g32.dtype == torch.float32 and g32.shape[1] == fam.R
    if attivo("tile"):
        import dataclasses
        blk = dataclasses.replace(blk, BS_dx2=TILE["dx3" if split == 3 else "dx1"][0])
    te, ts, n_t, salta = tabella(piano, blk.BS_dx2, [fam])
    out = (torch.zeros if salta else torch.empty)(g32.shape[0], fam.C, dtype=out_dtype, device=g32.device)
    _lancia_dx(g32, s, g32.stride(0), inv, fam, d, out, te, ts, n_t, piano, blk, 1, split)
    return out


# ═══════════════════════════════════ KL head with one fp16 product per direction ═══════════════════════════════════
class TestaX1:
    """Student head (V, H) as ONE fp16 matrix (fp16 tier, E2E_KQAT=testa1): logits = fp16(h) fp16(W)^T and
    dL/dh = (p 2^-e) fp16(W) 2^e with fp32 accumulation; e per row (power of two: max |p| 2^-e in [1/2, 1), exact),
    so differences p_S - p_T keep 11 significant bits down to 2^-24 of the row maximum. Same interface as
    e2e_bf16x3.TestaX3 (shape / device / dtype / logit / per)."""

    def __init__(self, W: torch.Tensor):
        self.shape = W.shape
        self.device = W.device
        self.dtype = torch.float32
        self.w16 = W.to(torch.float16).contiguous()          # (V, H)
        self.w16t = self.w16.t()

    @staticmethod
    def _mm(a, b):
        import e2e_densi_piegati as DP
        return DP.mm(a, b)

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        return self._mm(h.to(torch.float16), self.w16t)

    def per(self, p: torch.Tensor) -> torch.Tensor:
        _, e = torch.frexp(amax_abs(p, 1, keepdim=True))
        e = torch.where(torch.isfinite(amax_abs(p, 1, keepdim=True)), e, torch.zeros_like(e)).clamp_(-125, 125)
        uno = torch.ones_like(e, dtype=torch.float32)
        return self._mm((p * torch.ldexp(uno, -e)).to(torch.float16), self.w16).mul_(torch.ldexp(uno, e))


# ═══════════════════════════════════ dense (folded multi-plane) products with the split in registers ═══════════════
# e2e_densi_piegati (x2) computes, per dense call, up to 7 fp16 GEMMs with _k_mm16 plus elementwise passes that
# materialise scaled/split copies of dY (S x R fp32) twice. `_k_mmx` does one of those groups of GEMMs in ONE launch:
#   C[m, n] = sum_terms (A_t @ B_t)[m, n] * oscale[m]
# with A fp16 given (forward: X) or fp32 split in registers after the exact power-of-two scalings (row of A, column of
# A), B = hi and/or lo fp16. Every term keeps its own fp32 accumulator, summed at the end in the order of today's
# code (lo first, then hi): the same products in the same K order as separate _k_mm16 launches (BK = 32), so the
# result is the same bit for bit (checked by the tests). TERMS bit mask: 1 = A_hi B_hi, 2 = A_hi B_lo, 4 = A_lo B_hi.
if _TRITON:
    @triton.jit
    def _k_mmx(a_ptr, sa_m, sa_k, ra_ptr, ca_ptr, bh_ptr, bl_ptr, sb_k, sb_n, c_ptr, sc_m, sc_n, os_ptr, oc_ptr,
               M, N, K, zero,
               A32: tl.constexpr, RS: tl.constexpr, CS: tl.constexpr, OS: tl.constexpr, OC: tl.constexpr,
               TERMS: tl.constexpr, LOSC: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        per_gruppo = GROUP_M * npn
        gid = pid // per_gruppo
        prima = gid * GROUP_M
        gm = tl.minimum(npm - prima, GROUP_M)
        pm = prima + (pid % per_gruppo) % gm
        pn = (pid % per_gruppo) // gm
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm_ = rm < M
        mn = rn < N
        if RS:
            rsc = tl.load(ra_ptr + rm, mask=mm_, other=1.0)[:, None]
        zr = tl.zeros((BM, BN), dtype=tl.float32) * zero
        acc1 = tl.zeros((BM, BN), dtype=tl.float32)      # A_hi B_hi
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)      # A_hi B_lo
        acc4 = tl.zeros((BM, BN), dtype=tl.float32)      # A_lo B_hi
        for k0 in range(0, K, BK):
            kk = k0 + rk
            mk = kk < K
            a = tl.load(a_ptr + rm[:, None].to(tl.int64) * sa_m + kk[None, :].to(tl.int64) * sa_k,
                        mask=mm_[:, None] & mk[None, :], other=0.0)
            if A32:
                if CS:
                    a = a * tl.load(ca_ptr + kk, mask=mk, other=1.0)[None, :]
                if RS:
                    a = a * rsc
                ah = a.to(tl.float16)
                if LOSC:                                     # low part scaled by 2^11 (Ootomo-Yokota)
                    al = ((a - ah.to(tl.float32)) * 2048.0).to(tl.float16)
                else:
                    al = (a - ah.to(tl.float32)).to(tl.float16)
            else:
                ah = a
                al = a
            boff = kk[:, None].to(tl.int64) * sb_k + rn[None, :].to(tl.int64) * sb_n
            mb = mk[:, None] & mn[None, :]
            bh = tl.load(bh_ptr + boff, mask=mb, other=0.0)
            if LOSC:                                     # rounded fp32 adds between BK chunks (see gdn_qat._k_bmm16)
                acc1 += tl.dot(ah, bh, zr)
                acc4 += tl.dot(al, bh, zr)
                acc2 += tl.dot(ah, tl.load(bl_ptr + boff, mask=mb, other=0.0), zr)
            else:
                if (TERMS & 1) != 0:
                    acc1 += tl.dot(ah, bh)
                if (TERMS & 4) != 0:
                    acc4 += tl.dot(al, bh)
                if (TERMS & 2) != 0:
                    bl = tl.load(bl_ptr + boff, mask=mb, other=0.0)
                    acc2 += tl.dot(ah, bl)
        # today's order of the sums (e2e_densi_piegati): dX = (A_lo B_hi + A_hi B_lo) + A_hi B_hi;
        # forward = X lo + X hi (= term 2 + term 1 with A = X); G = A_lo X + A_hi X (= term 4 + term 1)
        if LOSC:                                             # lo terms carry 2^11: (A_lo B_hi + A_hi B_lo) 2^-11
            c = acc1 + (acc4 + acc2) * (1.0 / 2048.0)
        elif (TERMS & 4) != 0:
            c = acc4
            if (TERMS & 2) != 0:
                c = c + acc2
            if (TERMS & 1) != 0:
                c = c + acc1
        elif (TERMS & 2) != 0:
            c = acc2
            if (TERMS & 1) != 0:
                c = c + acc1
        else:
            c = acc1
        if OS:
            c = c * tl.load(os_ptr + rm, mask=mm_, other=1.0)[:, None]
        if OC:
            c = c * tl.load(oc_ptr + rn, mask=mn, other=1.0)[None, :]
        tl.store(c_ptr + rm[:, None].to(tl.int64) * sc_m + rn[None, :].to(tl.int64) * sc_n, c,
                 mask=mm_[:, None] & mn[None, :])


CFG_MMX = {"BM": int(os.getenv("E2E_MMX_BM", "64")), "BN": int(os.getenv("E2E_MMX_BN", "64")),
           "BK": 32, "warps": int(os.getenv("E2E_MMX_WARPS", "8")), "GROUP_M": 8}


def mmx(a: torch.Tensor, bh: torch.Tensor, bl: Optional[torch.Tensor], terms: int,
        rscale: Optional[torch.Tensor] = None, cscale: Optional[torch.Tensor] = None,
        oscale: Optional[torch.Tensor] = None, ocscale: Optional[torch.Tensor] = None,
        losc: bool = False) -> torch.Tensor:
    """sum of the TERMS products of a (M, K) [fp16, or fp32 split after scaling rows by rscale and columns by
    cscale] with bh / bl (K, N) fp16, rows of the result times oscale. Views with any strides."""
    M, K = a.shape
    N = bh.shape[1]
    assert bh.shape[0] == K and bh.dtype == torch.float16 and (bl is None or bl.stride() == bh.stride())
    a32 = a.dtype == torch.float32
    assert a32 or a.dtype == torch.float16
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)
    nul = c
    BM, BN, BK, w, G = CFG_MMX["BM"], CFG_MMX["BN"], CFG_MMX["BK"], CFG_MMX["warps"], CFG_MMX["GROUP_M"]
    _k_mmx[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
        a, a.stride(0), a.stride(1),
        rscale.reshape(-1).contiguous() if rscale is not None else nul,
        cscale.reshape(-1).contiguous() if cscale is not None else nul,
        bh, bl if bl is not None else bh, bh.stride(0), bh.stride(1), c, c.stride(0), c.stride(1),
        oscale.reshape(-1).contiguous() if oscale is not None else nul,
        ocscale.reshape(-1).contiguous() if ocscale is not None else nul, M, N, K, 0.0,
        int(a32), int(rscale is not None), int(cscale is not None), int(oscale is not None),
        int(ocscale is not None), terms, int(losc),
        BM, BN, BK, G, num_warps=w)
    return c


# ═══════════════════ fp16 tier for the experts: the frozen 2nd plane FOLDED into one fp16 weight ═══════════════════
# E2E_KQAT=esperti1. Today (and with `caldi`) an expert with a 2nd plane costs two products in the forward and two in
# dX: y = x W1(d1)^T + x W2(d2)^T. Here W_eff[r, c] = fp16(t1 d1 + t2 d2) (t in {-1, 0, 1}, d fp16: both products and
# their fp32 sum are exact, ONE rounding to fp16) is formed in registers from the two byte planes and used in ONE
# product (cold experts: the expanded 2nd plane has byte 0 and scale 0, the term is -0 and W_eff = t1 d1 exactly, the
# weight of today). dX = fp16(dY s) W1 + fp16(dY s) W2 (one product per plane, the 2nd only on its experts; the folded
# dX kernel _k_dx2p measured slower, 3.1 TFLOPS). The scale gradient of plane 1 is
# _k_ddp on plane 1 alone (exact fp32 dY): the 2nd plane is frozen. Error of the folded weight: <= 2^-11 relative,
# half an ulp of the fp16 scale grid itself; judged in the end-to-end gate against the reference's own noise.
if _TRITON:
    @triton.jit
    def _w_eff(v, u, pw, dA, eA):
        """fp16(t1 d1 + t2 d2) for one 32-column piece along K: v, u bytes of the two planes, pw multipliers (column
        vector), dA / eA the two scales (row vector)."""
        t1 = _S._dec_trit(v, pw).to(tl.float32)
        t2 = _S._dec_trit(u, pw).to(tl.float32)
        return (t1 * dA.to(tl.float32) + t2 * eA.to(tl.float32)).to(tl.float16)

    @triton.jit
    def _k_gemm2p(a_ptr, sa_s,
                  pb, pb_2, sb_e, NBR, d_ptr, d_2, sd_e, NG, o_ptr, so_s,
                  qb, qb_2, e_ptr, e_2, o2_ptr,
                  te_ptr, ts_ptr, offs_ptr, R, C, NT,
                  GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr, DUE: tl.constexpr,
                  BS: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
        """_k_gemm2 with the 2nd plane folded (see above). DUE: gate (pb, pb_2, d, d_2) and up (qb, qb_2, e, e_2)."""
        DT = tl.float16
        pid_t, pid_n = _S._ordine_gruppi(tl.program_id(0), NT, tl.cdiv(R, BN), GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_fine
        ln = tl.arange(0, BN)
        r0 = pid_n * BN
        mn = (r0 + ln) < R
        i = tl.arange(0, 32)
        pa = a_ptr + s0.to(tl.int64) * sa_s
        offa = (ls * sa_s)[:, None] + i[None, :]
        rb = e64 * sb_e + r0.to(tl.int64) * NBR
        rd = e64 * sd_e + r0.to(tl.int64) * NG
        offr = (ln * NBR)[None, :]
        offd = ln * NG
        m2 = mn[None, :]
        acc = tl.zeros((BS, BN), dtype=tl.float32)
        acc2 = tl.zeros((BS, BN), dtype=tl.float32)
        for k in range(0, C, 32):
            qs, qh = _S._schema_sottoblocco(k, QK, NB, ROT)
            j = (k % 256) + i
            n1 = j // 32
            n2 = (j - 160) // 16
            n3 = (j - 240) // 4
            n = tl.where(j < 160, n1, tl.where(j < 240, n2, n3))
            pw = tl.where(n == 0, 1, tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))))
            ob = tl.where(j < 160, qs + i, tl.where(j < 240, qs + 32 + (i % 16), qh + (i % 4)))
            offb = offr + ob[:, None]
            a = tl.load(pa + k + offa, mask=ms[:, None], other=0.0)
            dA = tl.load(d_ptr + rd + k // GS + offd, mask=mn, other=0.0)
            eA = tl.load(d_2 + rd + k // GS + offd, mask=mn, other=0.0)
            v = tl.load(pb + rb + offb, mask=m2, other=0).to(tl.int32)
            u = tl.load(pb_2 + rb + offb, mask=m2, other=0).to(tl.int32)
            acc += tl.dot(a, _w_eff(v, u, pw[:, None], dA[None, :], eA[None, :]))
            if DUE:
                dB = tl.load(e_ptr + rd + k // GS + offd, mask=mn, other=0.0)
                eB = tl.load(e_2 + rd + k // GS + offd, mask=mn, other=0.0)
                v2 = tl.load(qb + rb + offb, mask=m2, other=0).to(tl.int32)
                u2 = tl.load(qb_2 + rb + offb, mask=m2, other=0).to(tl.int32)
                acc2 += tl.dot(a, _w_eff(v2, u2, pw[:, None], dB[None, :], eB[None, :]))
        offo = (ls * so_s)[:, None] + ln[None, :]
        mo = ms[:, None] & mn[None, :]
        po = s0.to(tl.int64) * so_s + r0
        tl.store(o_ptr + po + offo, acc, mask=mo)
        if DUE:
            tl.store(o2_ptr + po + offo, acc2, mask=mo)

    @triton.jit
    def _k_dx2p(g_ptr, s_ptr, sg_s, inv_ptr, pb, pb_2, sb_e, NBR, d_ptr, d_2, sd_e, NG, o_ptr, so_s,
                te_ptr, ts_ptr, offs_ptr, R, C, NT,
                GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                BS: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, P0: tl.constexpr):
        """dX = fp16(dY s) W_eff * inv (fp16 tier): one product per piece; dY fp32 scaled by the per-row power of two s
        and rounded ONCE to fp16 in registers; W_eff as in _k_gemm2p, decoded per piece of 32 columns. Pieces P0..P0+3
        of each 256-column sub-block (two launches, P0 = 0 and 4: the two-plane decode of 8 pieces spilled ~230 VGPR)."""
        DT = tl.float16
        pid_t, pid_n = _S._ordine_gruppi(tl.program_id(0), NT, C // 256, GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_fine
        lk = tl.arange(0, BK)
        c0 = pid_n * 256
        qs, qh = _S._schema_sottoblocco(c0, QK, NB, ROT)
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
        pe2 = pb_2 + e64 * sb_e
        pde = d_ptr + e64 * sd_e + g
        pde2 = d_2 + e64 * sd_e + g
        pa = g_ptr + s0.to(tl.int64) * sg_s
        sv = tl.load(s_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        h0 = tl.zeros((BS, 32), dtype=tl.float32)
        h1 = tl.zeros((BS, 32), dtype=tl.float32)
        h2 = tl.zeros((BS, 32), dtype=tl.float32)
        h3 = tl.zeros((BS, 32), dtype=tl.float32)
        for r0 in range(0, R, BK):
            mr = (r0 + lk) < R
            m2 = mr[:, None]
            prow = pe + r0 * NBR
            prow2 = pe2 + r0 * NBR
            v0 = tl.load(prow + off0, mask=m2, other=0).to(tl.int32)
            v5 = tl.load(prow + off5, mask=m2, other=0).to(tl.int32)
            v7 = tl.load(prow + off7, mask=m2, other=0).to(tl.int32)
            u0 = tl.load(prow2 + off0, mask=m2, other=0).to(tl.int32)
            u5 = tl.load(prow2 + off5, mask=m2, other=0).to(tl.int32)
            u7 = tl.load(prow2 + off7, mask=m2, other=0).to(tl.int32)
            pdr = pde + r0 * NG
            pdr2 = pde2 + r0 * NG
            dA = tl.load(pdr + offd, mask=mr, other=0.0).to(tl.float32)
            eA = tl.load(pdr2 + offd, mask=mr, other=0.0).to(tl.float32)
            if GS == 128:
                dB = tl.load(pdr + 1 + offd, mask=mr, other=0.0).to(tl.float32)
                eB = tl.load(pdr2 + 1 + offd, mask=mr, other=0.0).to(tl.float32)
            else:
                dB = dA
                eB = eA
            b = (tl.load(pa + r0 + offa, mask=ms[:, None] & mr[None, :], other=0.0) * sv).to(DT)
            h0 += tl.dot(b, _wp2(v0, v5, v7, u0, u5, u7, dA, dB, eA, eB, pw5, pw6, pw7, P0 + 0, GS))
            h1 += tl.dot(b, _wp2(v0, v5, v7, u0, u5, u7, dA, dB, eA, eB, pw5, pw6, pw7, P0 + 1, GS))
            h2 += tl.dot(b, _wp2(v0, v5, v7, u0, u5, u7, dA, dB, eA, eB, pw5, pw6, pw7, P0 + 2, GS))
            h3 += tl.dot(b, _wp2(v0, v5, v7, u0, u5, u7, dA, dB, eA, eB, pw5, pw6, pw7, P0 + 3, GS))
        inv = tl.load(inv_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        po = o_ptr + s0.to(tl.int64) * so_s + c0 + P0 * 32
        offo = (ls * so_s)[:, None] + i[None, :]
        mo = ms[:, None]
        tl.store(po + offo, h0 * inv, mask=mo)
        tl.store(po + 32 + offo, h1 * inv, mask=mo)
        tl.store(po + 64 + offo, h2 * inv, mask=mo)
        tl.store(po + 96 + offo, h3 * inv, mask=mo)

    @triton.jit
    def _wp2(v0, v5, v7, u0, u5, u7, dA, dB, eA, eB, pw5, pw6, pw7, P: tl.constexpr, GS: tl.constexpr):
        """Piece P (32 columns along C, rows along K) of fp16(t1 d1 + t2 d2); scales are row vectors (BK,)."""
        t1 = _S._t_pezzo(v0, v5, v7, pw5, pw6, pw7, P, 0).to(tl.float32)
        t2 = _S._t_pezzo(u0, u5, u7, pw5, pw6, pw7, P, 0).to(tl.float32)
        if GS == 128:
            if P >= 4:
                d1 = dB
                d2 = eB
            else:
                d1 = dA
                d2 = eA
        else:
            d1 = dA
            d2 = eA
        return (t1 * d1[:, None] + t2 * d2[:, None]).to(tl.float16)


def avanti_piani(x: torch.Tensor, fams: list, fams2: list, ds: list, ds2: list, piano, blk) -> list:
    """Forward of one family (or gate+up) with its frozen 2nd plane folded (E2E_KQAT=esperti1); fp32 outputs."""
    import e2e_studente as S
    x = x.contiguous()
    f = fams[0]
    S._controlla_v2(f)
    for g in list(fams[1:]) + list(fams2):
        assert (g.tipo, g.E, g.R, g.NBR) == (f.tipo, f.E, f.R, f.NBR)
    ds = [d.contiguous() for d in ds]
    ds2 = [d.contiguous() for d in ds2]
    outs = [torch.empty(x.shape[0], f.R, dtype=torch.float32, device=x.device) for _ in fams]
    te, ts, n_t = piano.tabella(blk.BS2)
    due = len(fams) == 2
    fB, fB2 = (fams[1], fams2[1]) if due else (fams[0], fams2[0])
    dB, dB2 = (ds[1], ds2[1]) if due else (ds[0], ds2[0])
    _k_gemm2p[(n_t * triton.cdiv(f.R, blk.BN2),)](
        x, x.stride(0), f.blocchi, fams2[0].blocchi, f.blocchi.stride(0), f.NBR, ds[0], ds2[0], ds[0].stride(0), f.NG,
        outs[0], outs[0].stride(0), fB.blocchi, fB2.blocchi, dB, dB2, outs[-1],
        te, ts, piano.offs, f.R, f.C, n_t, f.gs, f.qk, f.nb, int(f.ruotata), int(due),
        blk.BS2, blk.BN2, blk.GROUP_M, num_warps=blk.num_warps2, **S._stadi(blk.stadi_gemm2))
    return outs


def dx_piani(g32: torch.Tensor, fam, fam2, d: torch.Tensor, d2: torch.Tensor, piano, blk) -> torch.Tensor:
    """fp32 dX = fp16(dY s) fp16(t1 d1 + t2 d2) / s (E2E_KQAT=esperti1)."""
    import e2e_studente as S
    import e2e_indietro_hilo as HL
    g32 = g32.contiguous()
    s, inv = HL.scala_pot2(amax_abs(g32, 1, keepdim=True))
    s, inv = s.reshape(-1).contiguous(), inv.reshape(-1).contiguous()
    out = torch.empty(g32.shape[0], fam.C, dtype=torch.float32, device=g32.device)
    te, ts, n_t = piano.tabella(blk.BS_dx2)
    d, d2 = d.contiguous(), d2.contiguous()
    for p0 in (0, 4):
        _k_dx2p[(n_t * (fam.C // 256),)](g32, s, g32.stride(0), inv, fam.blocchi, fam2.blocchi, fam.blocchi.stride(0),
                                         fam.NBR, d, d2, d.stride(0), fam.NG, out, out.stride(0), te, ts, piano.offs,
                                         fam.R, fam.C, n_t, fam.gs, fam.qk, fam.nb, int(fam.ruotata),
                                         blk.BS_dx2, blk.BK_dx2, blk.GROUP_M, p0, num_warps=blk.num_warps2,
                                         **S._stadi(blk.stadi_dx2))
    return out


class EspertiPiani(torch.autograd.Function):
    """y_k = x W_eff,k^T for one family (k = 1) or gate+up (k = 2), W_eff = fp16(t1 d1 + t2 d2) (fp16 tier).
    Differentiable in x and in the plane-1 scales d1 (the 2nd planes are frozen)."""

    @staticmethod
    def forward(ctx, x, d1a, d1b, fams, fams2, ds2, piano, mot):
        ds = [d1a] if d1b is None else [d1a, d1b]
        ys = avanti_piani(x, list(fams), list(fams2), ds, list(ds2), piano, mot.blk)
        ctx.save_for_backward(x, d1a, *(() if d1b is None else (d1b,)))
        ctx.fams, ctx.fams2, ctx.ds2, ctx.piano, ctx.mot, ctx.due = fams, fams2, ds2, piano, mot, d1b is not None
        return tuple(ys) if d1b is not None else ys[0]

    @staticmethod
    def backward(ctx, *gys):
        x, *ds = ctx.saved_tensors
        dx = None
        dds = []
        for k, (f, f2, d, d2, gy) in enumerate(zip(ctx.fams, ctx.fams2, ds, ctx.ds2, gys)):
            if gy is None:
                dds.append(None)
                continue
            g32 = gy.to(torch.float32).contiguous()
            if ctx.needs_input_grad[0]:
                # 9/10 bench (layer 5): the folded two-plane decode in dX ran at 3.1 TFLOPS (44 ms, gate); one fp16
                # product per plane (2nd plane only on its experts, `caldi`) is faster AND exact in the weights:
                # dX = fp16(dY s) W1 + fp16(dY s) W2 with fp32 accumulation, one rounding of dY (x1)
                import e2e_indietro_hilo as HL
                s_, inv_ = HL.scala_pot2(amax_abs(g32, 1, keepdim=True))
                p = dx_s_triton(g32, s_, inv_, f, d.to(torch.float32), ctx.piano, ctx.mot.blk, 1)
                p += dx_s_triton(g32, s_, inv_, f2, d2.to(torch.float32), ctx.piano, ctx.mot.blk, 1)
                dx = p if dx is None else dx.add_(p)
            dds.append(dd_p_triton(g32, x.contiguous(), f, ctx.piano).to(d.dtype) if ctx.needs_input_grad[1 + k]
                       else None)
        if dx is not None:
            dx = dx.to(x.dtype)
        return (dx, dds[0], dds[1] if len(dds) > 1 else None, None, None, None, None, None)


class TestaX3K:
    """Student head (V, H) for the EXACT tier (E2E_KQAT=testa3): W = (hi + lo 2^-11) / a_v, a_v a power of two per
    vocabulary row (max |W[v]| a_v in [1/2, 1)), lo scaled by 2^11 (Ootomo-Yokota); logits = h W^T and dL/dh = p W
    with h / p split in registers after an exact power-of-two scaling per row: three fp16 products, fp32 accumulation,
    in ONE launch each (_k_mmx). Same interface as e2e_bf16x3.TestaX3."""

    def __init__(self, W: torch.Tensor):
        self.shape = W.shape
        self.device = W.device
        self.dtype = torch.float32
        W = W.to(torch.float32)
        a = _pot2_vec(amax_abs(W, 1))
        Ws = W * a[:, None]
        self.hi = Ws.to(torch.float16).contiguous()
        self.lo = ((Ws - self.hi.float()) * 2048.0).to(torch.float16).contiguous()
        self.inv_a = (1.0 / a).contiguous()                  # exact

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(torch.float32)
        s = _pot2_vec(amax_abs(h, 1))
        return mmx(h, self.hi.t(), self.lo.t(), 7, rscale=s, oscale=1.0 / s, ocscale=self.inv_a, losc=True)

    def per(self, p: torch.Tensor) -> torch.Tensor:
        p = p.to(torch.float32)
        s = _pot2_vec((p.abs() * self.inv_a[None, :]).amax(1))
        return mmx(p, self.hi, self.lo, 7, rscale=s, cscale=self.inv_a, oscale=1.0 / s, losc=True)


def _pot2_vec(m: torch.Tensor) -> torch.Tensor:
    _, e = torch.frexp(m)
    ok = (m > 0) & torch.isfinite(m)
    e = torch.where(ok, e, torch.zeros_like(e)).clamp_(-125, 125)
    return torch.ldexp(torch.ones_like(m, dtype=torch.float32), -e)


# ═══════════════════ overflow guard of the fp16 tensors in ONE pass (Osservatore.guarda, E2E_KQAT=guardia) ═══════════════════
# Today: `trabocco |= ~torch.isfinite(t).all()` = isfinite (abs + compare) + reduction + not + or: 3-4 passes over every
# fp16 product input (x_ord is 65536 x 2048), ~0.2 ms/token over 40 layers in the 9/10 profile. Here one read per
# element and one atomic OR per program into the same bool flag (no synchronisation; the flag is read once per step
# by Studente.controlla as before). The flag is set iff some element is inf or NaN: the same predicate.
if _TRITON:
    @triton.jit
    def _k_guardia(t_ptr, n, flag_ptr, B: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
        v = tl.load(t_ptr + i, mask=i < n, other=0.0).to(tl.float32)
        bad = tl.max(((v != v) | (tl.abs(v) == float("inf"))).to(tl.int32), axis=0)
        if bad > 0:
            tl.atomic_or(flag_ptr, 1)


def guarda_fp16(trabocco: torch.Tensor, t: torch.Tensor, flag_i32: dict) -> None:
    """trabocco (bool, device) |= any(not finite(t)); t fp16 contiguous on the GPU, otherwise torch."""
    if not (_TRITON and t.is_cuda and t.is_contiguous()):
        trabocco |= ~torch.isfinite(t).all()
        return
    f = flag_i32.get(trabocco.device)
    if f is None:
        f = flag_i32[trabocco.device] = torch.zeros((), dtype=torch.int32, device=t.device)
    n = t.numel()
    B = 4096
    _k_guardia[(triton.cdiv(n, B),)](t, n, f, B, num_warps=4)
    trabocco |= f.bool()
    f.zero_()                                    # the bool flag is the sticky one (Studente.controlla resets it)


def amax_abs(t: torch.Tensor, *dim, **kw) -> torch.Tensor:
    """max |t| over `dim`: with E2E_KQAT=amax ONE reduction kernel (torch.linalg.vector_norm, ord inf: |.| inside the
    reduction) instead of abs (a full fp32 copy) + amax. max of |x| is exact in any order: same value bit for bit,
    NaN propagated by both. 9/10 profile, fp16 tier: 732 abs kernels = 309 ms on 4 layers (~0.4 ms/token on 40)."""
    if attivo("amax"):
        return torch.linalg.vector_norm(t, float("inf"), *dim, **kw)
    return t.abs().amax(*dim, **kw)


# ═══════════ fused preparation of the split dY operands of the dense layers (E2E_KQAT=prep) ═══════════
# Today (e2e_densi_piegati.indietro_x / prodotto_g): h = g * inv; m = |h|.amax; s = 2^-e; v = h * s; hi = fp16(v);
# lo = fp16(v - hi): 5-7 full passes over a 65536 x 8192 fp32 tensor per dense backward. Here: one Triton kernel
# per row block (max, then the scaled split written directly) for the row case, and one reduction + one writing
# kernel for the column case. Every step is the same IEEE operation on the same values (products by powers of two
# are exact, fp16 rounding is round-to-nearest-even in both): same tensors bit for bit (prova K11).
if _TRITON:
    @triton.jit
    def _k_prep_righe(g_ptr, sg, cs_ptr, R, C, hh_ptr, hl_ptr, sh, s_ptr, inv_ptr,
                      BR: tl.constexpr, BC: tl.constexpr, LO: tl.constexpr, CS: tl.constexpr):
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        mr = r < R
        r64 = r.to(tl.int64)
        m = tl.zeros((BR,), tl.float32)
        for c0 in range(0, C, BC):
            c = c0 + tl.arange(0, BC)
            mc = c < C
            v = tl.load(g_ptr + r64[:, None] * sg + c[None, :], mask=mr[:, None] & mc[None, :], other=0.0)
            if CS:
                v = v * tl.load(cs_ptr + c, mask=mc, other=1.0)[None, :]
            m = tl.maximum(m, tl.max(tl.abs(v), 1))
        # s = 2^-e with m * s in [1/2, 1) (frexp convention), s = 1 where m == 0 or not finite, |e| <= 125.
        # normal m (biased exponent E >= 1): frexp exponent E - 126; subnormal m: frexp exponent <= -126, clamped
        # to -125 as in scala_pot2. (A NaN in the row: Triton's max may drop it while torch.amax propagates it; a NaN
        # dY is an error caught by the guards before it is used.)
        bits = m.to(tl.int32, bitcast=True)
        ex = ((bits >> 23) & 255) - 126
        ex = tl.where(((bits >> 23) & 255) == 0, -125, ex)
        ok = (m > 0) & (m == m) & (m < float("inf"))
        ex = tl.where(ok, ex, 0)
        ex = tl.minimum(tl.maximum(ex, -125), 125)
        s = (( -ex + 127) << 23).to(tl.float32, bitcast=True)
        inv = ((ex + 127) << 23).to(tl.float32, bitcast=True)
        tl.store(s_ptr + r, s, mask=mr)
        tl.store(inv_ptr + r, inv, mask=mr)
        for c0 in range(0, C, BC):
            c = c0 + tl.arange(0, BC)
            mc = c < C
            msk = mr[:, None] & mc[None, :]
            v = tl.load(g_ptr + r64[:, None] * sg + c[None, :], mask=msk, other=0.0)
            if CS:
                v = v * tl.load(cs_ptr + c, mask=mc, other=1.0)[None, :]
            v = v * s[:, None]
            h = v.to(tl.float16)
            tl.store(hh_ptr + r64[:, None] * sh + c[None, :], h, mask=msk)
            if LO:
                tl.store(hl_ptr + r64[:, None] * sh + c[None, :], (v - h.to(tl.float32)).to(tl.float16), mask=msk)

    @triton.jit
    def _k_prep_colonne(g_ptr, sg, sc_ptr, R, C, hh_ptr, hl_ptr, sh, BR: tl.constexpr, BC: tl.constexpr,
                        LO: tl.constexpr):
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        c = tl.program_id(1) * BC + tl.arange(0, BC)
        msk = (r < R)[:, None] & (c < C)[None, :]
        r64 = r.to(tl.int64)
        v = tl.load(g_ptr + r64[:, None] * sg + c[None, :], mask=msk, other=0.0)
        v = v * tl.load(sc_ptr + c, mask=c < C, other=1.0)[None, :]
        h = v.to(tl.float16)
        tl.store(hh_ptr + r64[:, None] * sh + c[None, :], h, mask=msk)
        if LO:
            tl.store(hl_ptr + r64[:, None] * sh + c[None, :], (v - h.to(tl.float32)).to(tl.float16), mask=msk)


def prep_righe(g32: torch.Tensor, cscale: Optional[torch.Tensor], lo: bool):
    """(hh, hl or None, s (R, 1), inv_s (R, 1)) of v = g32 * cscale[None, :] * s, s per row as scala_pot2."""
    R, C = g32.shape
    assert g32.stride(1) == 1
    hh = torch.empty(R, C, dtype=torch.float16, device=g32.device)
    hl = torch.empty_like(hh) if lo else hh
    s = torch.empty(R, 1, dtype=torch.float32, device=g32.device)
    inv = torch.empty_like(s)
    BR, BC = 8, 512
    _k_prep_righe[(triton.cdiv(R, BR),)](g32, g32.stride(0), cscale if cscale is not None else s, R, C, hh, hl,
                                         hh.stride(0), s, inv, BR, BC, int(lo), int(cscale is not None),
                                         num_warps=4)
    return hh, (hl if lo else None), s, inv


def prep_colonne(g32: torch.Tensor, lo: bool):
    """(hh, hl or None, s (C,), inv (C,)) of v = g32 * s[None, :], s per column as scala_pot2."""
    import e2e_indietro_hilo as HL
    R, C = g32.shape
    s, inv = HL.scala_pot2(torch.linalg.vector_norm(g32, float("inf"), 0))
    hh = torch.empty(R, C, dtype=torch.float16, device=g32.device)
    hl = torch.empty_like(hh) if lo else hh
    BR, BC = 64, 128
    _k_prep_colonne[(triton.cdiv(R, BR), triton.cdiv(C, BC))](g32, g32.stride(0), s, R, C, hh, hl, hh.stride(0),
                                                              BR, BC, int(lo), num_warps=4)
    return hh, (hl if lo else None), s, inv
