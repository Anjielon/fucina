#!/usr/bin/env python3
"""cardan_piega_drq.py — 23/10/2026: piega CARDAN + DRQ (Distributionally Robust Quantization, arXiv 2610.11226, 8/10/2026).

Stessa piega di cardan_piega_lotti.py (importata, NON modificata); dopo JARQ, sui codici di ogni lotto di esperti gira la
rifinitura DRQ del lavoro (Alg. 3.3 + App. C), con scale e gruppi FERMI (cambiano solo i trit, la taglia resta):
    L_DRQ(W^) = tr((W^-W) H (W^-W)^T) + alpha * S_C(W^),   S_C = max_{u^T C u = 1} ||(W^-W) u||^2,   C = (1-beta) H + beta I
  per giro: u = direzione peggiore (autovettore generalizzato, iterazione di potenza con Cholesky di C su CPU fp64:
  rocSOLVER su gfx1151 non e' affidabile); per ogni riga la migliore sostituzione di UN codice (+-1 dentro {-1,0,1})
  per riduzione di L + alpha ||(W^-W)u||^2; righe ordinate per riduzione; candidati = prime frazioni {1/32,1/16,1/8,
  1/4,1/2,1}; accettato il candidato con L_DRQ minima, se L_DRQ scende e la ricostruzione resta <= (1+tol) quella di
  partenza; al massimo 3 giri.
DICHIARATO rispetto al lavoro: W = bersaglio della piega (Q + C D^T, cioe' il modello continuo del congiunto), non i pesi
FP originali; H = Hessiana ruotata della piega (non attivazioni grezze), quindi niente sottoinsiemi di controllo separati;
8 sostituzioni singole per riga + coppie e frazioni {0,1/32,..,1} come App. C (v3); alpha/beta fissi da ambiente (il lavoro li sceglie su 64 Pile).
Ambiente: DRQ_ALPHA (0.3) DRQ_BETA (0.25) DRQ_TOL (0.03) DRQ_GIRI (3) DRQ_SPENTO=1 (solo JARQ, per il controllo al byte).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardan_piega as CP  # noqa: E402

ALPHA = float(os.environ.get("DRQ_ALPHA", "0.3"))
BETA = float(os.environ.get("DRQ_BETA", "0.25"))
TOL = float(os.environ.get("DRQ_TOL", "0.03"))
GIRI = int(os.environ.get("DRQ_GIRI", "3"))
FRAZ = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)   # App. C: {0,1/32,...,1}
CHUNK = int(os.environ.get('DRQ_CHUNK', '2'))
NS = 8                                                       # App. C: sostituzioni singole tenute per riga
STAT = {"lotti": 0, "giri_accettati": 0, "codici_cambiati": 0, "L0": 0.0, "L1": 0.0, "S0": 0.0, "S1": 0.0}


def _recon(E, Ha):                       # [...,B,R,C] x [B,C,C] -> [...,B]  tr(E H E^T), broadcast, una GEMM
    return ((E @ Ha) * E).sum((-1, -2))


def _peggiore(E, Li, iters=40, v0=None):
    """max_{u^T C u=1} ||E u||^2 = sigma_max(G)^2, G = E Li^T, Li = Lc^-1 (C = Lc Lc^T), u = Li^T v. Broadcast su [...,B]."""
    G = E @ Li.transpose(-1, -2)
    if v0 is None:
        v = torch.randn(G.shape[:-2] + (G.shape[-1], 1), dtype=E.dtype, generator=torch.Generator().manual_seed(0)).to(E.device)
    else:
        v = v0.unsqueeze(-1).expand(G.shape[:-2] + (G.shape[-1], 1)).clone()
    for _ in range(iters):
        v = G.transpose(-1, -2) @ (G @ v)
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    S = (G @ v).squeeze(-1).pow(2).sum(-1)
    return S, v.squeeze(-1), (Li.transpose(-1, -2) @ v).squeeze(-1)


def drq_lotto(W, H, T, S):
    """W [B,R,C] bersaglio, H [B,C,C], T int8 [B,1,R,C] (un piano), S [B,1,R,ng] valori fp16. Ritorna T rifinito.
    v3 (23/10, Angelo: tutto GPU, velocissimo, FEDELE): App. C del lavoro — per riga le 8 migliori sostituzioni singole
    (entrambi i versi) e le loro 28 COPPIE di pesi diversi (termine incrociato 2 d1 d2 (H_j1j2 + alpha u_j1 u_j2)); si
    tiene la singola o la coppia con la riduzione maggiore se positiva; righe ordinate; frazioni {0,1/32,..,1} valutate
    in UN lotto broadcast sulla GPU; accettazione: L_DRQ scende e recon <= (1+tol) quella iniziale. Lc^-1 su CPU fp64."""
    B, K, R, C = T.shape
    if K != 1:
        return T
    dev = W.device
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    W = W.to(dev, dt); H = H.to(dev, dt); Tl = T.clone(); Sv = S.to(dev, dt)
    ng = Sv.shape[-1]; gs = C // ng
    Ha = H + 1e-4 * torch.eye(C, dtype=dt, device=dev)
    nt0 = torch.get_num_threads(); torch.set_num_threads(int(os.environ.get("DRQ_FILI_CPU", "4")))
    Hc = H.double().cpu()
    Lc = torch.linalg.cholesky((1 - BETA) * Hc + BETA * torch.eye(C, dtype=torch.float64))   # CPU fp64 (rocSOLVER no)
    torch.set_num_threads(nt0)
    Lg = Lc.to(dev, dt)                                                                 # inversa: trsm sulla GPU (BLAS, affidabile)
    try:
        Li = torch.stack([torch.linalg.solve_triangular(Lg[i], torch.eye(C, dtype=dt, device=dev), upper=False)
                          for i in range(B)])                                           # trsm uno per esperto
    except RuntimeError:                                                                # ripiego: CPU fp64
        Li = torch.linalg.solve_triangular(Lc, torch.eye(C, dtype=torch.float64).expand(B, C, C), upper=False).to(dev, dt)
    sc = Sv[:, 0].repeat_interleave(gs, dim=-1)                                       # [B,R,C]
    dH = torch.diagonal(Ha, dim1=-2, dim2=-1)                                           # [B,C]
    NF = len(FRAZ)
    ar_b = torch.arange(B, device=dev).view(B, 1, 1)

    E = Tl[:, 0].to(dt) * sc - W
    S0, v, u = _peggiore(E, Li)
    L0 = _recon(E, Ha)
    Lcur, Scur = L0.clone(), S0.clone()
    Lmax = L0 * (1 + TOL)
    iu = torch.triu_indices(NS, NS, 1, device=dev)                                     # 28 coppie di 8
    for _ in range(GIRI):
        EH = E @ Ha
        Eu = (E @ u.unsqueeze(-1)).squeeze(-1)                                          # [B,R]
        t = Tl[:, 0].to(dt)
        sc_ = []
        for dlt in (1.0, -1.0):                                                         # punteggio di ogni sostituzione
            ok = ((t + dlt) >= -1) & ((t + dlt) <= 1)
            d = dlt * sc
            val = 2 * d * EH + d * d * dH.unsqueeze(1) + ALPHA * (2 * d * Eu.unsqueeze(-1) * u.unsqueeze(1) + (d * u.unsqueeze(1)) ** 2)
            sc_.append(torch.where(ok, val, torch.full_like(val, float("inf"))))
        tutto = torch.cat(sc_, -1)                                                      # [B,R,2C]: +1 poi -1
        v8, i8 = torch.topk(tutto, NS, dim=-1, largest=False)                           # 8 singole migliori per riga
        j8 = i8 % C; s8 = torch.where(i8 < C, 1.0, -1.0).to(dt)                         # colonna e verso
        d8 = s8 * torch.gather(sc, -1, j8)                                              # variazione del peso
        ja, jb = j8[..., iu[0]], j8[..., iu[1]]                                         # [B,R,28]
        Hab = Ha[ar_b, ja, jb]                                                          # H_j1j2
        ua, ub = u[ar_b, ja], u[ar_b, jb]
        coppia = v8[..., iu[0]] + v8[..., iu[1]] + 2 * d8[..., iu[0]] * d8[..., iu[1]] * (Hab + ALPHA * ua * ub)
        coppia = torch.where(ja != jb, coppia, torch.full_like(coppia, float("inf")))
        bc, ic = coppia.min(-1)                                                         # migliore coppia per riga
        usa_c = bc < v8[..., 0]
        best_d = torch.where(usa_c, bc, v8[..., 0])
        migl = best_d < 0
        nmigl = migl.sum(-1)
        rango = torch.argsort(torch.argsort(torch.where(migl, best_d, torch.full_like(best_d, float("inf"))), dim=-1), dim=-1)
        # aggiornamenti per riga: (j1, s1) sempre; (j2, s2) solo se coppia
        pa, pb = iu[0][ic].unsqueeze(-1), iu[1][ic].unsqueeze(-1)                       # posizioni nelle 8, [B,R,1]
        j1 = torch.where(usa_c, torch.gather(j8, -1, pa)[..., 0], j8[..., 0])
        s1 = torch.where(usa_c, torch.gather(s8, -1, pa)[..., 0], s8[..., 0])
        j2 = torch.gather(j8, -1, pb)[..., 0]; s2 = torch.gather(s8, -1, pb)[..., 0]
        TF = Tl[:, 0].to(torch.int16).unsqueeze(0).repeat(NF, 1, 1, 1)                 # [NF,B,R,C]
        for f, fr in enumerate(FRAZ):
            k = torch.ceil(nmigl.to(dt) * fr).long()
            sel = migl & (rango < k.unsqueeze(-1))
            bb, rr = sel.nonzero(as_tuple=True)
            TF[f, bb, rr, j1[bb, rr]] += s1[bb, rr].to(torch.int16)
            m2 = usa_c[bb, rr]
            TF[f, bb[m2], rr[m2], j2[bb[m2], rr[m2]]] += s2[bb[m2], rr[m2]].to(torch.int16)
        SF, LF = [], []                                                                 # candidati a gruppi (VRAM: LoTA accanto)
        for c0 in range(0, NF, CHUNK):
            EF = TF[c0:c0 + CHUNK].to(dt) * sc - W
            s_, _, _ = _peggiore(EF, Li, iters=12, v0=v); SF.append(s_); LF.append(_recon(EF, Ha)); del EF
        SF, LF = torch.cat(SF), torch.cat(LF)
        objF = torch.where(LF <= Lmax, LF + ALPHA * SF, torch.full_like(LF, float("inf")))
        mig, fi = objF.min(0)
        acc = mig < (Lcur + ALPHA * Scur)
        if not bool(acc.any()):
            break
        nuovo = TF[fi, torch.arange(B, device=dev)].to(Tl.dtype)
        Tl = torch.where(acc.view(B, 1, 1, 1), nuovo.unsqueeze(1), Tl)
        STAT["giri_accettati"] += int(acc.sum())
        E = Tl[:, 0].to(dt) * sc - W
        Scur, v, u = _peggiore(E, Li, iters=12, v0=v)
        Lcur = _recon(E, Ha)
    STAT["codici_cambiati"] += int((Tl != T).sum())
    STAT["L0"] += float(L0.sum()); STAT["S0"] += float(S0.sum())
    STAT["L1"] += float(Lcur.sum()); STAT["S1"] += float(Scur.sum()); STAT["lotti"] += 1
    return Tl


_jarq = CP.jarq_lotto


def jarq_drq(W, H, T, S, giri: int = 3, nu2: float = 1e-4):
    T2, S2, cat = _jarq(W, H, T, S, giri=giri, nu2=nu2)
    if os.environ.get("DRQ_SPENTO") != "1":
        T2 = drq_lotto(W, H, T2, S2)
        cat["drq"] = dict(STAT)
        CP.log(f"DRQ lotto {STAT['lotti']}: codici cambiati {STAT['codici_cambiati']}, giri accettati {STAT['giri_accettati']}, "
               f"recon {STAT['L0']:.4g} -> {STAT['L1']:.4g}, S {STAT['S0']:.4g} -> {STAT['S1']:.4g}")
    return T2, S2, cat


CP.jarq_lotto = jarq_drq

if __name__ == "__main__":
    import cardan_piega_lotti as PL
    sys.exit(PL.main())
