#!/usr/bin/env python3
"""paper_campionato.py — Lever 5b of SINTESI_1025/R5 (25/10): the KD term of the paper loss with UNBIASED sampled
distillation instead of the teacher top-k subset (RS-KD, "Sparse Logit Sampling", arXiv 2503.16870).

Why: the CARDAN paper loss (cardan_paper_perdita.perdita_paper, arXiv 2609.21137 Eq. 3) distils on Omega = the teacher's
top-256 logits, both distributions renormalised on Omega. RS-KD shows that a top-K target is a BIASED estimate of the
teacher distribution (it drops the tail, distorting both the gradient and the calibration), while importance sampling
gives an unbiased estimate at the same sparsity. Here the teacher logits are recomputed in full from the cached hidden
state, so the sampling is exact: per token, K indices v_1..v_K are drawn WITH replacement from the proposal
q = p_T^(tau) = softmax(z_T / tau) (RS-KD with proposal = teacher, importance weights p/q = 1):
    KL_hat = (1/K) sum_k [log p_T(v_k) - log p_S(v_k)]            E[KL_hat] = KL(p_T || p_S)   (full vocabulary)
    d(tau^2 KL_hat)/dz_S = tau * (p_S - p_hat),  p_hat = counts / K   E[...] = tau * (p_S - p_T)  (the exact gradient)
p_S is the full softmax of the student (temperature tau). The CE term is unchanged. Used ONLY in training
(--paper-campionato K): the held-out paper metric of the selection keeps the original top-256 definition.
"""
from __future__ import annotations

import torch


def generatore(device, seme: int, passo: int) -> torch.Generator:
    """Deterministic per step (a resumed run draws the same samples)."""
    g = torch.Generator(device=device)
    g.manual_seed((int(seme) * 1000003 + int(passo)) % (2 ** 63 - 1))
    return g


def perdita_paper_campionata(hS, WS, hT, WT, tgt, ok, tau: float, lam: float, K: int, pezzo: int, con_grad: bool,
                             logit_s, logit_t, per_testa, gen: torch.Generator) -> tuple:
    """Same contract as cardan_paper_perdita.perdita_paper (sum_t tau^2 KL_hat, sum_valid CE, n_tok, n_valid, dL/dhS),
    with L = mean_t(tau^2 KL_hat) + lam * mean_valid(CE) and KL_hat the K-sample unbiased estimate of the FULL KL."""
    T = hS.shape[0]
    nv = max(1, int(ok.sum()))
    kd_tot, ce_tot = 0.0, 0.0
    g = torch.zeros_like(hS, dtype=torch.float32) if con_grad else None
    for j in range(0, T, pezzo):
        zS = logit_s(hS[j:j + pezzo].float()).float()
        zT = logit_t(hT[j:j + pezzo]).float()
        lpT = torch.log_softmax(zT / tau, -1)
        del zT
        idx = torch.multinomial(lpT.exp(), K, replacement=True, generator=gen)     # [P, K] samples from p_T^(tau)
        lpS_t = torch.log_softmax(zS / tau, -1)
        kd_tot += float((tau * tau) * (lpT.gather(1, idx) - lpS_t.gather(1, idx)).sum() / K)
        del lpT
        m = ok[j:j + pezzo]
        t = tgt[j:j + pezzo]
        lp = torch.log_softmax(zS, -1)
        ce_tot += float(-(lp.gather(1, t[:, None])[:, 0] * m).sum())
        if con_grad:
            dz = lp.exp()
            dz[torch.arange(dz.shape[0], device=dz.device), t] -= 1.0
            dz *= (m.float() * (lam / nv))[:, None]
            dz.add_(lpS_t.exp(), alpha=tau / T)                                     # + tau/T * p_S (full vocabulary)
            dz.scatter_add_(1, idx, torch.full(idx.shape, -tau / (T * K), dtype=dz.dtype, device=dz.device))
            g[j:j + pezzo] += per_testa(dz).float()
        del zS, lp, lpS_t
    return kd_tot, ce_tot, T, nv, g
