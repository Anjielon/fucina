#!/usr/bin/env python3
"""leve_g4/cardan_base.py — CARDAN (arXiv 2609.21137) App. A.1, «Whitened Shared-Basis Correction», in forma ESATTA.
9/10 (LEVE_APERTE_CURE.md). Sostituisce la base di cardan_costruisci.py del 7/10, che NON era quella del lavoro:

  7/10:  D = top-r autovettori ortonormali di  N = H (sum_e R_e^T R_e) H        (pesatura H^2, D ortonormale)
  lavoro (Eq. 4-6):  min_{D,C_e} sum_e w_e || Sigma^{1/2} (Delta_e - D C_e) ||_F^2,   Sigma = E[h h^T] = R R^T
         U_r = top-r autovettori di  R^T (sum_e w_e Delta_e Delta_e^T) R,   D = R^-T U_r,
         C_e = (D^T Sigma D)^-1 D^T Sigma Delta_e
  (Ky Fan / Eckart-Young sui residui sbiancati A_e = R^T Delta_e: ottimo per l'obiettivo di calibrazione).

Convenzioni nostre: W_e (righe = uscite 512, colonne = ingressi 2048), residuo R_e = W_e(bf16) - Q_e = Delta_e^T, correzione
W_e ~ Q_e + Cn_e D^T con Cn_e = C_e^T (512 x r) — e' la forma che il motore thor-cardan legge (lrd = D^T, lrc = Cn_e).
Funzioni pure (torch), nessun I/O: le prove CPU sono in test_cardan_base.py.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

F64 = torch.float64


def chol_sigma(Sigma: torch.Tensor, smorza_rel: float = 1e-6) -> torch.Tensor:
    """R triangolare inferiore con Sigma (+ smorzamento relativo minimo per la definita positivita') = R R^T."""
    S = 0.5 * (Sigma + Sigma.T).to(F64)
    d = torch.diagonal(S).mean()
    return torch.linalg.cholesky(S + smorza_rel * d * torch.eye(S.shape[0], dtype=F64, device=S.device))


def gram_sbiancata(Rlow: torch.Tensor, residui: Iterable[torch.Tensor], pesi: Optional[Iterable[float]] = None) -> torch.Tensor:
    """G = R^T (sum_e w_e Delta_e Delta_e^T) R con Delta_e = R_e^T: si accumula A_e A_e^T, A_e = R^T R_e^T, a lotti.
    residui: iterabile di tensori [n, 512, 2048] (lotti di esperti) o [512, 2048]."""
    C = Rlow.shape[0]
    G = torch.zeros(C, C, dtype=F64, device=Rlow.device)
    pesi = iter(pesi) if pesi is not None else None
    for Rb in residui:
        Rb = Rb.to(F64)
        if Rb.dim() == 2:
            Rb = Rb.unsqueeze(0)
        for Re in Rb:
            w = 1.0 if pesi is None else float(next(pesi))
            A = Rlow.T @ Re.T                                       # [2048, 512]
            G += w * (A @ A.T)
    return G


def base_condivisa(G: torch.Tensor, Rlow: torch.Tensor, r: int) -> torch.Tensor:
    """Eq. 5-6: U_r = top-r di G, D = R^-T U_r  [2048, r]."""
    ev, V = torch.linalg.eigh(0.5 * (G + G.T))
    U = V[:, torch.argsort(ev, descending=True)[:r]]
    return torch.linalg.solve_triangular(Rlow.T, U, upper=True)     # R^T D = U


def coefficienti(Re: torch.Tensor, D: torch.Tensor, Sigma: torch.Tensor, ridge_rel: float = 1e-12) -> torch.Tensor:
    """Eq. 6 nella nostra forma: Cn_e = R_e Sigma D (D^T Sigma D)^-1  [n?, 512, r] (D gia' quello che il motore vedra')."""
    D = D.to(F64)
    SD = Sigma.to(F64) @ D
    M = D.T @ SD
    M = M + ridge_rel * torch.trace(M) / M.shape[0] * torch.eye(M.shape[0], dtype=F64, device=M.device)
    Minv = torch.linalg.inv(M)
    return (Re.to(F64) @ SD) @ Minv


def obiettivo(Re: torch.Tensor, Cn: torch.Tensor, D: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """sum || (R_e - Cn_e D^T) Sigma^{1/2} ||_F^2  =  tr(E Sigma E^T) per esperto."""
    E = Re.to(F64) - Cn.to(F64) @ D.to(F64).T
    return ((E @ Sigma.to(F64)) * E).sum((-1, -2))


def base_7_10(Rs: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """La base del 7/10 (per il confronto nelle prove): top-r ortonormali di H (sum R_e^T R_e) H."""
    N = torch.zeros(H.shape[0], H.shape[0], dtype=F64)
    for Re in Rs.to(F64):
        RH = Re @ H.to(F64)
        N += RH.T @ RH
    ev, V = torch.linalg.eigh(N)
    return V[:, torch.argsort(ev, descending=True)[:r]]


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


def risolvi(Rs_lotti, Sigma: torch.Tensor, r: int, pesi=None) -> Tuple[torch.Tensor, "callable"]:
    """Base D (fp16, come la leggera' il motore) e funzione che dà Cn_e (fp16) per un lotto di residui."""
    Rl = chol_sigma(Sigma)
    D16 = fp16(base_condivisa(gram_sbiancata(Rl, Rs_lotti, pesi), Rl, r))
    return D16, (lambda Rb: fp16(coefficienti(Rb, D16, Sigma)))
