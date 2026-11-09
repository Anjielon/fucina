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
