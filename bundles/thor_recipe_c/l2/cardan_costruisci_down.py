#!/usr/bin/env python3
"""cardan_costruisci_down.py — 25/10: CARDAN App. A.1 shared basis for the routed expert DOWN projection (ffn_down_exps),
the init of the new lever "CARDAN low-rank on expert DOWN" (cardan_kd_moduli.CardanDown, README_DOWN.md).

Same mathematics as cardan_costruisci.py / cardan_base.py (gate/up), transposed to the down shapes:
    W_e = down_proj[e] (bf16 teacher, [2048, 512])     Q_e = host ternary down (ALL planes: main + exps2, TQ1_0, plain basis)
    R_e = W_e - Q_e                                       Sigma = E[a a^T] of the down INPUT (SwiGLU output, 512 x 512)
    D = R^-T U_r (U_r = top-r of R^T (sum_e R_e^T R_e) R, Sigma = R R^T)     C_e = R_e Sigma D (D^T Sigma D)^-1
    correction W_e ~ Q_e + C_e D^T,  D [512, r] shared,  C_e [2048, r]
TQ1_0 is NOT rotated: everything is in the plain basis, which is also the basis of the student's down GEMM and of the fold.
Sigma: --sigma-dir DIR with Sigma_down_LL.npy ([512, 512] shared, or [E, 512, 512] per expert -> averaged); without it
Sigma = I (plain Frobenius, = SVD of the stacked residuals). No down-input statistics were captured as of 25/10: the
identity basis is the declared fallback (README_DOWN.md, open risk).
Output per layer: strato_LL.npz with blk.L.ffn_down_lrd.weight (r, 512) = D^T fp16 and blk.L.ffn_down_lrc.weight
(256, 2048, r) = C_e fp16 (stored as uint8 views, like cardan_costruisci.py), plus misure.json.
CPU only (torch threads), no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import cardan_base as CB  # noqa: E402  (local copy, pure functions)

F64 = torch.float64


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sigma_di(L: int, sigma_dir: Optional[str], C: int) -> tuple:
    """(Sigma [C, C] fp64 normalised to trace C, source label)."""
    if sigma_dir:
        p = Path(sigma_dir) / f"Sigma_down_{L:02d}.npy"
        if p.exists():
            S = torch.from_numpy(np.load(p).astype(np.float64))
            if S.dim() == 3:
                S = S.mean(0)
            S = 0.5 * (S + S.T)
            return S / torch.trace(S) * C, f"Sigma {p.name}"
    return torch.eye(C, dtype=F64), "identita"


def base_down(Rs: torch.Tensor, Sigma: torch.Tensor, r: int, lotto: int = 16) -> tuple:
    """Rs [E, H, FF] residuals. Returns (D16 [FF, r] fp16-valued fp64, C16 [E, H, r] fp16, E0, E1) with E0/E1 the
    Sigma-weighted residual energy before/after the correction."""
    lotti = [Rs[i:i + lotto] for i in range(0, Rs.shape[0], lotto)]
    E0 = float(sum(((Rb.double() @ Sigma) * Rb.double()).sum() for Rb in lotti))
    D16, coeff = CB.risolvi(lotti, Sigma, r)
    C16 = torch.cat([coeff(Rb) for Rb in lotti]).to(torch.float16)
    E1 = float(sum(CB.obiettivo(Rs[i:i + lotto], C16[i:i + lotto], D16, Sigma).sum()
                   for i in range(0, Rs.shape[0], lotto)))
    return D16, C16, E0, E1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--gguf", required=True, help="host GGUF whose ternary down planes are corrected")
    ap.add_argument("--sigma-dir", default=None, help="Sigma_down_LL.npy of the down input (default: identity)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thread", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.thread)
    sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    from sorgente import SorgenteHF, bf16_a_f32
    import comune as CMR
    from rd_ternary.formato import TensorKey
    srcp = CMR.gp.GGUFPlaneSource(a.gguf, bases=["ffn_down_exps"])
    lo, _, hi = a.strati.partition("-")
    strati = list(range(int(lo), int(hi or lo) + 1))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = SorgenteHF()
    rk = a.rango
    misure = json.loads((out / "misure.json").read_text()) if (out / "misure.json").exists() else {}
    for L in strati:
        f_npz = out / f"strato_{L:02d}.npz"
        if f_npz.exists() and str(L) in misure:
            continue
        t0 = time.time()
        u16, _ = src.leggi_hf(f"model.language_model.layers.{L}.mlp.experts.down_proj")   # [E, 2048, 512]
        en = srcp.entry(TensorKey(L, 0, "down"))
        E, Rr, C = en.n_experts, en.out, en.in_
        assert tuple(u16.shape) == (E, Rr, C), (u16.shape, (E, Rr, C))
        Rs = torch.empty(E, Rr, C, dtype=torch.float32)
        for e in range(E):
            tq = srcp.load(TensorKey(L, e, "down"))
            assert not tq.rotated, "ffn_down_exps expected TQ1_0 (plain basis)"
            P_, S_ = tq.planes, tq.scales.double()
            K_, R_, C_ = P_.shape; ng_ = S_.shape[-1]
            Q = sum((P_[k].double().reshape(R_, ng_, C_ // ng_) * S_[k].unsqueeze(-1)).reshape(R_, C_) for k in range(K_))
            Rs[e] = torch.from_numpy(bf16_a_f32(u16[e]).astype(np.float32)) - Q.float()
        Sigma, fonte = sigma_di(L, a.sigma_dir, C)
        D16, C16, E0, E1 = base_down(Rs, Sigma, rk)
        del Rs
        arrs = {f"blk.{L}.ffn_down_lrd.weight": D16.T.contiguous().to(torch.float16).numpy(),     # (r, 512)
                f"blk.{L}.ffn_down_lrc.weight": C16.numpy()}                                      # (256, 2048, r)
        np.savez(f_npz, **{k: v.view(np.uint8) for k, v in arrs.items()})
        misure[str(L)] = {"rango": rk, "sigma": fonte, "err_rel_prima": E0 ** .5,
                          "riduzione_errore_pesato": 1 - E1 / E0, "secondi": round(time.time() - t0, 1)}
        (out / "misure.json").write_text(json.dumps(misure, indent=1))
        log(f"strato {L}: down, riduzione errore pesato {100 * (1 - E1 / E0):.2f} % ({fonte}) [{time.time() - t0:.0f}s]")
    log("FINE", out)


if __name__ == "__main__":
    main()
