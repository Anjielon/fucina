#!/usr/bin/env python3
"""cardan_paper_perdita.py — the end-to-end distillation loss of CARDAN as written in the paper (10/10 night, audit).

Paper: Ma, Fan, Liu, Li, "A Multi-Engine Dataflow for MoE Decoding on AWS Trainium" (CARDAN), arXiv 2609.21137,
§3.3 "End-to-end distillation", Eq. (3):

    L = tau^2 * D_KL( p_tea^Omega || p_stu^Omega ) + lambda * L_CE

  Omega       the indices of the TEACHER's 256 largest logits, per token;
  p^Omega     softmax over the logits at Omega, both divided by the scaling factor tau;
  L_CE        next-token cross-entropy of the student on the true text (full vocabulary).

The paper does NOT state the values of tau and lambda ("lambda is a model-quality alignment hyperparameter"): they are
arguments here and every run declares them. Per-token means: KD over all tokens, CE over the tokens with a valid next
token inside the segment (segments never leak into each other).

Gradient with respect to the student logits z (per token):
    d(tau^2 KL)/dz_Omega = tau * (p_stu^Omega - p_tea^Omega),  0 outside Omega
    d(CE)/dz             = softmax(z) - onehot(next)
then dL/dh = (dL/dz) @ W_S (the head), computed chunk by chunk so the full logits never live at once.
"""
from __future__ import annotations

import torch


def perdita_paper(hS, WS, hT, WT, tgt, ok, tau: float, lam: float, topk: int, pezzo: int, con_grad: bool,
                  logit_s, logit_t, per_testa) -> tuple:
    """One row. Returns (sum_t tau^2 KL_Omega, sum_valid CE, n_tok, n_valid, dL/dhS or None) where
    L = mean_t(tau^2 KL_Omega) + lam * mean_valid(CE). logit_s(h) = h @ W_S^T, logit_t(hT) = teacher logits (fp32),
    per_testa(q) = q @ W_S."""
    T = hS.shape[0]
    nv = max(1, int(ok.sum()))
    kd_tot, ce_tot = 0.0, 0.0
    g = torch.zeros_like(hS, dtype=torch.float32) if con_grad else None
    for j in range(0, T, pezzo):
        zS = logit_s(hS[j:j + pezzo].float()).float()
        zT = logit_t(hT[j:j + pezzo]).float()
        om = torch.topk(zT, topk, dim=-1).indices                       # Omega: teacher top-k
        lpT = torch.log_softmax(zT.gather(1, om) / tau, -1)
        lpS = torch.log_softmax(zS.gather(1, om) / tau, -1)
        pT = lpT.exp()
        kd_tot += float((tau * tau) * (pT * (lpT - lpS)).sum())
        m = ok[j:j + pezzo]
        t = tgt[j:j + pezzo]
        lp = torch.log_softmax(zS, -1)
        ce_tot += float(-(lp.gather(1, t[:, None])[:, 0] * m).sum())
        if con_grad:
            dz = lp.exp()
            dz[torch.arange(dz.shape[0], device=dz.device), t] -= 1.0
            dz *= (m.float() * (lam / nv))[:, None]
            dz.scatter_add_(1, om, (tau / T) * (lpS.exp() - pT))
            g[j:j + pezzo] += per_testa(dz).float()
        del zS, zT, lp
    return kd_tot, ce_tot, T, nv, g


def init_c_chiusa(D: torch.Tensor, sigma: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Paper App. A.1 Eq. (6): C_e = (D^T Sigma_x D)^-1 D^T Sigma_x Delta_e (fp64 on CPU: rocSOLVER rule)."""
    Dd, S, De = D.double().cpu(), sigma.double().cpu(), delta.double().cpu()
    return torch.linalg.solve(Dd.T @ S @ Dd, Dd.T @ S @ De)
