#!/usr/bin/env python3
"""cardan_kd_moduli.py — CARDAN (arXiv 2609.21137) shared-basis low-rank correction of the routed experts, as a
TRAINABLE branch of the e2e student, for the end-to-end distillation of the paper (§3.3, Eq. 3) run in the TERNARY
flow (the student is the whole ternary model: every block sees the input produced by the ternary blocks before it,
and the target is the teacher's output distribution, never W_bf16 — lesson of 2/10).

Per layer L and family f in {gate, up} (the two projections that read the SAME input h):
    g_e(x) = W_e x  +  C_e (D^T x)          D [2048, r] shared by the 256 experts, C_e [512, r] private
All tensors live in the domain of the expert GEMM of the student: with the B recipes (ric.ruota) x is the ROTATED
activation x_rot = R x and W_e = trit x scale is the STORED (rotated) weight, so the correction in the stored basis is
exactly dW'_e = C_e D^T — the quantity the fold (cardan_piega.py) writes back into the existing planes and scales.

Initialisation: D = Sigma-whitened basis of CARDAN App. A.1 (cardan_base.py, rotated), C = 0 => the student is the
host GGUF AT THE BIT at step 0 (identity gate, as RILQ). The gradient reaches D as soon as C != 0.

Hook: MoEStudente.forward (e2e_studente.py, this copy) adds `self.cardan(x_ord, piano)` to g and u after the second
plane and before silu; the module is registered as `mlp.cardan`. Absent = bit-identical student.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

FAMS = ("gate", "up")


class _GroupedLowRank(torch.autograd.Function):
    """y_f[s] = C_f[e(s)] z_f[s] for tokens sorted by expert (offsets `offs`, python list E+1), f in {gate, up}.
    Forward and backward loop over the experts that received tokens; nothing of size [S, R, r] is materialised."""

    @staticmethod
    def forward(ctx, zg, zu, Cg, Cu, offs):
        S = zg.shape[0]
        yg = zg.new_zeros(S, Cg.shape[1])
        yu = zu.new_zeros(S, Cu.shape[1])
        for e in range(len(offs) - 1):
            a, b = offs[e], offs[e + 1]
            if b > a:
                yg[a:b] = zg[a:b] @ Cg[e].T
                yu[a:b] = zu[a:b] @ Cu[e].T
        ctx.save_for_backward(zg, zu, Cg, Cu)
        ctx.offs = offs
        return yg, yu

    @staticmethod
    def backward(ctx, gg, gu):
        zg, zu, Cg, Cu = ctx.saved_tensors
        offs = ctx.offs
        dzg = torch.zeros_like(zg) if ctx.needs_input_grad[0] else None
        dzu = torch.zeros_like(zu) if ctx.needs_input_grad[1] else None
        dCg = torch.zeros_like(Cg) if ctx.needs_input_grad[2] else None
        dCu = torch.zeros_like(Cu) if ctx.needs_input_grad[3] else None
        gg = gg.to(zg.dtype)
        gu = gu.to(zu.dtype)
        for e in range(len(offs) - 1):
            a, b = offs[e], offs[e + 1]
            if b <= a:
                continue
            if dzg is not None:
                dzg[a:b] = gg[a:b] @ Cg[e]
                dzu[a:b] = gu[a:b] @ Cu[e]
            if dCg is not None:
                dCg[e] = gg[a:b].T @ zg[a:b]
                dCu[e] = gu[a:b].T @ zu[a:b]
        return dzg, dzu, dCg, dCu, None


# ---------------------------------------------------------------------------------------------------------------------
# 21-22/10 CARDAN_GRUPPI=1: the same _GroupedLowRank as GROUPED GEMMs over a TILE TABLE (no Python loop over the
# experts, no offs.tolist() host sync, no per-expert slice copies / zeros_like). Off (default) = the loop above, at the bit.
# Sources (adapted, not copied verbatim):
#   - ScatterMoE (Tan et al., arXiv 2403.08245, github.com/shawntan/scattermoe, Apache-2.0, commit 47b5e15,
#     scattermoe/kernels/ops.py): grouped->grouped `_scatter2scatter` for y and dz, `_groupXtY` for dC (one weight
#     tensor read with two different strides instead of a transposed copy; fp32 accumulation; empty experts -> 0);
#   - TritonMoE (arXiv 2605.23911, triton-kernels, MIT) / our Piano.tabella: a block -> (expert, row offset) table, so
#     every tile belongs to ONE expert (v1 used ScatterMoE's per-row expert id and looped over all experts present in a
#     row block: with many small experts each block redid the whole K loop up to ~60 times; measured 22/10, Zipf a=2:
#     dz 25 ms vs 5 ms uniform). Tile table built on device from offs, no sync (grid = S//BM + E upper bound).
# Backward = ONE fused kernel per tile: g is read ONCE and gives dz (g C_e) and the tile's partial dC (g^T z) for both
# families; then a deterministic per-expert reduction of the partials in tile order (no atomics, empty experts = 0).
# Two launches for the whole backward, one for the forward; a big expert is split over many tiles (no load imbalance).
# fp32 dots in IEEE precision (no TF32). Active only for CUDA/HIP fp32 tensors; CPU fp64 reference recipes keep the loop.
# Multi-process on one GPU: no global state besides compiled kernels, outputs allocated per call. Multi-GPU: launches on
# the tensors' own device.
# ---------------------------------------------------------------------------------------------------------------------
import contextlib as _contextlib
import os as _os

_GRUPPI_K = None


def gruppi_attivo() -> bool:
    return _os.environ.get("CARDAN_GRUPPI", "0") not in ("", "0")


def _kernel_gruppi():
    """Triton kernels built lazily (importing this module stays triton-free when the switch is off)."""
    global _GRUPPI_K
    if _GRUPPI_K is not None:
        return _GRUPPI_K
    import triton
    import triton.language as tl

    @triton.jit
    def _k_gr_avanti(Z1, Z2, C1, C2, Y1, Y2, TE, TS, OFFS,
                     szm, sce, scn, sck, sym,
                     R: tl.constexpr, r: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, IEEE: tl.constexpr, ACC: tl.constexpr):
        # tile t = (expert e, rows s0..s0+BM of e's segment); Y_f[s, n] = sum_j Z_f[s, j] C_f[e, n, j]
        t = tl.program_id(0)
        pn = tl.program_id(1)
        e = tl.load(TE + t)
        if e >= 0:
            s0 = tl.load(TS + t)
            fine = tl.load(OFFS + e + 1).to(tl.int32)
            rm = s0 + tl.arange(0, BM)
            mm = rm < fine
            rn = pn * BN + tl.arange(0, BN)
            mn = rn < R
            rj = tl.arange(0, BR)
            mj = rj < r
            zo = rm[:, None] * szm + rj[None, :]
            co = e * sce + rj[:, None] * sck + rn[None, :] * scn
            z1 = tl.load(Z1 + zo, mask=mm[:, None] & mj[None, :], other=0.0)
            z2 = tl.load(Z2 + zo, mask=mm[:, None] & mj[None, :], other=0.0)
            c1 = tl.load(C1 + co, mask=mj[:, None] & mn[None, :], other=0.0)
            c2 = tl.load(C2 + co, mask=mj[:, None] & mn[None, :], other=0.0)
            if IEEE:
                y1 = tl.dot(z1, c1, input_precision="ieee", out_dtype=ACC)
                y2 = tl.dot(z2, c2, input_precision="ieee", out_dtype=ACC)
            else:
                y1 = tl.dot(z1, c1, out_dtype=ACC)
                y2 = tl.dot(z2, c2, out_dtype=ACC)
            yo = rm[:, None] * sym + rn[None, :]
            ms = mm[:, None] & mn[None, :]
            tl.store(Y1 + yo, y1.to(Y1.dtype.element_ty), mask=ms)
            tl.store(Y2 + yo, y2.to(Y2.dtype.element_ty), mask=ms)

    @triton.jit
    def _k_gr_indietro(G1, G2, Z1, Z2, C1, C2, DZ1, DZ2, P1, P2, TE, TS, OFFS,
                       sgm, szm, sce, scn, sck, spt, spn,
                       R: tl.constexpr, r: tl.constexpr,
                       BM: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr, IEEE: tl.constexpr, ACC: tl.constexpr,
                       FAI_DZ: tl.constexpr, FAI_DC: tl.constexpr):
        # one tile: read G_f[rows, :] ONCE in K-blocks of BK and produce
        #   dZ_f[rows, :] = G_f[rows, :] C_f[e]           (accumulated over the K-blocks, in order)
        #   P_f[t, kblock, :] = G_f[rows, kblock]^T Z_f[rows, :]   (partial dC of this tile, reduced per expert later)
        t = tl.program_id(0)
        e = tl.load(TE + t)
        if e >= 0:
            s0 = tl.load(TS + t)
            fine = tl.load(OFFS + e + 1).to(tl.int32)
            rm = s0 + tl.arange(0, BM)
            mm = rm < fine
            rj = tl.arange(0, BR)
            mj = rj < r
            zo = rm[:, None] * szm + rj[None, :]
            if FAI_DC:
                z1 = tl.load(Z1 + zo, mask=mm[:, None] & mj[None, :], other=0.0)
                z2 = tl.load(Z2 + zo, mask=mm[:, None] & mj[None, :], other=0.0)
            a1 = tl.zeros((BM, BR), dtype=ACC)
            a2 = tl.zeros((BM, BR), dtype=ACC)
            rk0 = tl.arange(0, BK)
            for k0 in range(0, R, BK):
                rk = k0 + rk0
                mk = rk < R
                go = rm[:, None] * sgm + rk[None, :]
                g1 = tl.load(G1 + go, mask=mm[:, None] & mk[None, :], other=0.0)
                g2 = tl.load(G2 + go, mask=mm[:, None] & mk[None, :], other=0.0)
                if FAI_DZ:
                    co = e * sce + rk[:, None] * scn + rj[None, :] * sck
                    c1 = tl.load(C1 + co, mask=mk[:, None] & mj[None, :], other=0.0)
                    c2 = tl.load(C2 + co, mask=mk[:, None] & mj[None, :], other=0.0)
                    if IEEE:
                        a1 = tl.dot(g1, c1, a1, input_precision="ieee", out_dtype=ACC)
                        a2 = tl.dot(g2, c2, a2, input_precision="ieee", out_dtype=ACC)
                    else:
                        a1 = tl.dot(g1, c1, a1, out_dtype=ACC)
                        a2 = tl.dot(g2, c2, a2, out_dtype=ACC)
                if FAI_DC:
                    if IEEE:
                        p1 = tl.dot(tl.trans(g1), z1, input_precision="ieee", out_dtype=ACC)
                        p2 = tl.dot(tl.trans(g2), z2, input_precision="ieee", out_dtype=ACC)
                    else:
                        p1 = tl.dot(tl.trans(g1), z1, out_dtype=ACC)
                        p2 = tl.dot(tl.trans(g2), z2, out_dtype=ACC)
                    po = t * spt + rk[:, None] * spn + rj[None, :]
                    tl.store(P1 + po, p1, mask=mk[:, None] & mj[None, :])
                    tl.store(P2 + po, p2, mask=mk[:, None] & mj[None, :])
            if FAI_DZ:
                tl.store(DZ1 + zo, a1.to(DZ1.dtype.element_ty), mask=mm[:, None] & mj[None, :])
                tl.store(DZ2 + zo, a2.to(DZ2.dtype.element_ty), mask=mm[:, None] & mj[None, :])

    @triton.jit
    def _k_gr_riduci(P1, P2, D1, D2, INIZIO, NT, spt, spn, sde, sdn,
                     R: tl.constexpr, r: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, ACC: tl.constexpr):
        # dC_f[e, n, :] = sum over e's tiles (in tile order) of P_f[t, n, :]; no tile -> exact 0
        e = tl.program_id(0)
        pn = tl.program_id(1)
        rn = pn * BN + tl.arange(0, BN)
        mn = rn < R
        rj = tl.arange(0, BR)
        mj = rj < r
        t0 = tl.load(INIZIO + e)
        nt = tl.load(NT + e)
        a1 = tl.zeros((BN, BR), dtype=ACC)
        a2 = tl.zeros((BN, BR), dtype=ACC)
        m = mn[:, None] & mj[None, :]
        for i in range(0, nt):
            po = (t0 + i) * spt + rn[:, None] * spn + rj[None, :]
            a1 += tl.load(P1 + po, mask=m, other=0.0)
            a2 += tl.load(P2 + po, mask=m, other=0.0)
        do = e * sde + rn[:, None] * sdn + rj[None, :]
        tl.store(D1 + do, a1.to(D1.dtype.element_ty), mask=m)
        tl.store(D2 + do, a2.to(D2.dtype.element_ty), mask=m)

    _GRUPPI_K = {"avanti": _k_gr_avanti, "indietro": _k_gr_indietro, "riduci": _k_gr_riduci,
                 "triton": triton, "tl": tl}
    return _GRUPPI_K


# tile sizes (env-overridable for the timing sweep); tl.dot needs every dim >= 16
_GR_BM = int(_os.environ.get("CARDAN_GRUPPI_BM", "64"))        # rows per tile (forward and backward)
_GR_BN = int(_os.environ.get("CARDAN_GRUPPI_BN", "128"))       # R-columns per forward program
_GR_BK = int(_os.environ.get("CARDAN_GRUPPI_BK", "64"))        # R-block of the backward K loop
_GR_RID_BN = int(_os.environ.get("CARDAN_GRUPPI_RID_BN", "64"))  # R-rows per reduction program


def _p2(n: int, lo: int = 16) -> int:
    v = lo
    while v < n:
        v *= 2
    return v


def _sul_dispositivo(t):
    """Multi-GPU: launch on the tensor's own device (Triton uses the CURRENT device); CPU/interpreter: no-op."""
    return torch.cuda.device(t.device) if t.is_cuda else _contextlib.nullcontext()


def tabella_riquadri(offs: torch.Tensor, S: int, BM: int) -> tuple:
    """Tile table on device, no host sync: te[t] = expert of tile t (-1 = unused), ts[t] = first row, plus inizio[e] =
    first tile of e and nt[e] = its tile count. n_max = S//BM + E is an upper bound known from shapes. Same rule as
    Piano.tabella (e2e_studente.py), rebuilt here so the module does not depend on it."""
    E = offs.numel() - 1
    cnt = offs[1:] - offs[:-1]
    nt = (cnt + BM - 1) // BM
    cum = torch.cumsum(nt, 0)
    inizio = cum - nt
    n_max = S // BM + E
    t = torch.arange(n_max, device=offs.device, dtype=cum.dtype)
    e = torch.searchsorted(cum, t, right=True)
    valido = e < E
    ec = torch.clamp(e, max=E - 1)
    s = offs[ec] + (t - inizio[ec]) * BM
    return (torch.where(valido, e, -1).to(torch.int32).contiguous(),
            torch.where(valido, s, 0).to(torch.int32).contiguous(),
            inizio.to(torch.int32).contiguous(), nt.to(torch.int32).contiguous(), n_max)


def _acc(k, dt):
    return k["tl"].float64 if dt == torch.float64 else k["tl"].float32


def _avanti(zg, zu, Cg, Cu, tab, offs):
    k = _kernel_gruppi()
    te, ts, _, _, n_max = tab
    S, r = zg.shape
    E, R, _ = Cg.shape
    yg = zg.new_empty(S, R)
    yu = zu.new_empty(S, R)
    if S == 0:
        return yg, yu
    BN = min(_GR_BN, _p2(R))
    with _sul_dispositivo(zg):
        k["avanti"][(n_max, k["triton"].cdiv(R, BN))](
            zg, zu, Cg, Cu, yg, yu, te, ts, offs,
            zg.stride(0), Cg.stride(0), Cg.stride(1), Cg.stride(2), yg.stride(0),
            R=R, r=r, BM=_GR_BM, BN=BN, BR=_p2(r), IEEE=zg.dtype == torch.float32, ACC=_acc(k, zg.dtype))
    return yg, yu


def _indietro(gg, gu, zg, zu, Cg, Cu, tab, offs, fai_dz: bool, fai_dc: bool):
    k = _kernel_gruppi()
    te, ts, inizio, nt, n_max = tab
    S, r = zg.shape
    E, R, _ = Cg.shape
    dzg = torch.empty_like(zg) if fai_dz else zg
    dzu = torch.empty_like(zu) if fai_dz else zu
    if fai_dc:
        Pg = zg.new_empty(n_max, R, r)
        Pu = zu.new_empty(n_max, R, r)
        dCg = torch.empty_like(Cg)
        dCu = torch.empty_like(Cu)
    else:
        Pg = Pu = zg
    BR = _p2(r)
    with _sul_dispositivo(zg):
        if S > 0:
            k["indietro"][(n_max,)](
                gg, gu, zg, zu, Cg, Cu, dzg, dzu, Pg, Pu, te, ts, offs,
                gg.stride(0), zg.stride(0), Cg.stride(0), Cg.stride(1), Cg.stride(2),
                Pg.stride(0) if fai_dc else 0, Pg.stride(1) if fai_dc else 0,
                R=R, r=r, BM=_GR_BM, BK=min(_GR_BK, _p2(R)), BR=BR, IEEE=zg.dtype == torch.float32,
                ACC=_acc(k, zg.dtype), FAI_DZ=fai_dz, FAI_DC=fai_dc)
        if fai_dc:
            BN = min(_GR_RID_BN, _p2(R))
            k["riduci"][(E, k["triton"].cdiv(R, BN))](
                Pg, Pu, dCg, dCu, inizio, nt, Pg.stride(0), Pg.stride(1), dCg.stride(0), dCg.stride(1),
                R=R, r=r, BN=BN, BR=BR, ACC=_acc(k, zg.dtype))
    return (dzg if fai_dz else None, dzu if fai_dz else None,
            dCg if fai_dc else None, dCu if fai_dc else None)


class _GruppiLowRank(torch.autograd.Function):
    """Same maths as _GroupedLowRank. `offs` = DEVICE tensor (E+1), `tab` = tabella_riquadri(offs, S, _GR_BM).
    Forward 1 launch (gate+up), backward 2 launches (fused dz+partial dC, per-expert reduction), no host sync."""

    @staticmethod
    def forward(ctx, zg, zu, Cg, Cu, offs, tab):
        zg, zu, Cg, Cu = zg.contiguous(), zu.contiguous(), Cg.contiguous(), Cu.contiguous()
        yg, yu = _avanti(zg, zu, Cg, Cu, tab, offs)
        ctx.save_for_backward(zg, zu, Cg, Cu, offs)
        ctx.tab = tab
        return yg, yu

    @staticmethod
    def backward(ctx, gg, gu):
        zg, zu, Cg, Cu, offs = ctx.saved_tensors
        gg = gg.to(zg.dtype).contiguous()
        gu = gu.to(zu.dtype).contiguous()
        n = ctx.needs_input_grad
        dzg, dzu, dCg, dCu = _indietro(gg, gu, zg, zu, Cg, Cu, ctx.tab, offs, n[0] or n[1], n[2] or n[3])
        return (dzg if n[0] else None, dzu if n[1] else None, dCg if n[2] else None, dCu if n[3] else None,
                None, None)

class CardanStrato(torch.nn.Module):
    """D_f [C_in, r] and C_f [E, R, r] for f in (gate, up); forward(x_ord, piano) -> (dg, du) in fp32 (or fp64 when
    x is fp64: the CPU reference recipes)."""

    def __init__(self, L: int, E: int, R: int, C_in: int, r: int, Dg0: torch.Tensor, Du0: torch.Tensor,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.L, self.E, self.R, self.C_in, self.r = int(L), int(E), int(R), int(C_in), int(r)
        for f, D0 in (("gate", Dg0), ("up", Du0)):
            if tuple(D0.shape) != (C_in, r):
                raise ValueError(f"L{L} {f}: D {tuple(D0.shape)} != {(C_in, r)}")
            setattr(self, f"D_{f}", torch.nn.Parameter(D0.detach().to(device=device, dtype=dtype).clone()))
            setattr(self, f"C_{f}", torch.nn.Parameter(torch.zeros(E, R, r, device=device, dtype=dtype)))

    def forward(self, x_ord: torch.Tensor, piano) -> tuple:
        dt = torch.float64 if x_ord.dtype == torch.float64 else torch.float32
        x = x_ord.to(dt)
        zg = x @ self.D_gate.to(dt)
        zu = x @ self.D_up.to(dt)
        if gruppi_attivo() and x.is_cuda and dt == torch.float32:     # 21/10 CARDAN_GRUPPI=1 (default off)
            chiave = (zg.shape[0], _GR_BM)
            tab = getattr(piano, "_cardan_tab", None)       # cached on the Piano: forward and recompute share it
            if tab is None or tab[0] != chiave:
                tab = (chiave, tabella_riquadri(piano.offs, zg.shape[0], _GR_BM))
                try:
                    piano._cardan_tab = tab
                except AttributeError:
                    pass
            return _GruppiLowRank.apply(zg, zu, self.C_gate.to(dt), self.C_up.to(dt), piano.offs, tab[1])
        offs = piano.offs.tolist()
        return _GroupedLowRank.apply(zg, zu, self.C_gate.to(dt), self.C_up.to(dt), offs)

    def correzione(self, f: str, e: int) -> torch.Tensor:
        """dW'_e = C_e D^T [R, C_in] (stored/rotated basis), fp64 on CPU: what the fold writes into the planes."""
        C = getattr(self, f"C_{f}").detach()[e].double().cpu()
        D = getattr(self, f"D_{f}").detach().double().cpu()
        return C @ D.T


def base_a1_ruotata(npz: Path, ruota) -> Dict[str, torch.Tensor]:
    """{gate, up: D_rot [2048, r] fp64} from a cardan_costruisci.py layer file (lrd = D^T fp16 in the ORIGINAL basis).
    x_rot = R x  =>  z = D^T x = (R D)^T x_rot, and the rows of (R D)^T are ruota(rows of D^T)."""
    z = np.load(npz)
    out = {}
    for f in FAMS:
        k = [n for n in z.files if n.endswith(f"ffn_{f}_lrd.weight")][0]
        DT = torch.from_numpy(z[k].view(np.float16).astype(np.float64))         # (r, 2048) = D^T
        out[f] = ruota(DT).T.contiguous()                                        # (2048, r) = R D
    return out


def installa(st, strati, rango: int, base_dir: Optional[Path], device, ruota=None, seme: int = 20261009) -> dict:
    """Registers `mlp.cardan` on every requested layer. D from the A.1 basis when available (rotated with the student's
    own rotation when the recipe rotates), otherwise a Gaussian basis whitened on nothing (unit columns). C = 0."""
    if ruota is None:
        import e2e_studente as S
        ruota = getattr(S, "ruota", None)
    mods = {}
    for L in strati:
        moe = st.moduli[L].mlp
        es = moe.esperti
        fg = es.fam["gate"]
        E, R, C_in = fg.E, fg.R, fg.C
        Ds = None
        if base_dir is not None and (Path(base_dir) / f"strato_{L:02d}.npz").exists():
            if st.ric.ruota:
                Ds = base_a1_ruotata(Path(base_dir) / f"strato_{L:02d}.npz", lambda t: ruota(t))
            else:
                z = np.load(Path(base_dir) / f"strato_{L:02d}.npz")
                Ds = {f: torch.from_numpy(z[[n for n in z.files if n.endswith(f"ffn_{f}_lrd.weight")][0]]
                                          .view(np.float16).astype(np.float64)).T.contiguous() for f in FAMS}
            Ds = {f: v[:, :rango] for f, v in Ds.items()}
        if Ds is None or Ds["gate"].shape[1] < rango:
            g = torch.Generator().manual_seed(seme + L)
            Ds = {f: torch.randn(C_in, rango, generator=g, dtype=torch.float64) / C_in ** 0.5 for f in FAMS}
        m = CardanStrato(L, E, R, C_in, rango, Ds["gate"], Ds["up"], device=device,
                         dtype=torch.float64 if st.ric.dtype_gemm == torch.float64 else torch.float32)
        moe.cardan = m
        mods[L] = m
    return mods


def parametri(mods: dict, chi: str) -> list:
    return [getattr(mods[L], f"{chi}_{f}") for L in sorted(mods) for f in FAMS]


def salva(mods: dict, p: Path, meta: dict) -> None:
    torch.save({"meta": meta, "rango": next(iter(mods.values())).r,
                "stato": {L: {k: v.detach().cpu() for k, v in m.state_dict().items()} for L, m in mods.items()}}, p)


def carica(mods: dict, p: Path) -> dict:
    d = torch.load(p, map_location="cpu", weights_only=False)
    for L, sd in d["stato"].items():
        mods[int(L)].load_state_dict(sd)
    return d["meta"]


def statistiche(mods: dict) -> dict:
    """rms of C and D per family (for the log): how far the correction has moved from 0."""
    out = {}
    for f in FAMS:
        c = torch.cat([getattr(m, f"C_{f}").detach().reshape(-1) for m in mods.values()])
        out[f"rms_C_{f}"] = float(c.float().pow(2).mean().sqrt())
    return out


def scrivi_json(mods: dict, p: Path, extra: dict) -> None:
    json.dump({"strati": sorted(mods), "rango": next(iter(mods.values())).r, "famiglie": list(FAMS),
               "dominio": "stored (rotated) basis of the TQ1_0R_1024 planes: dW'_e = C_e D^T", **extra},
              open(p, "w"), indent=1)
