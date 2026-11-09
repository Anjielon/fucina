#!/usr/bin/env python3
"""Controllo del riferimento fp64 (27/9/2026): una ricorrenza token per token in fp64 (algoritmo indipendente dal
ripiego a blocchi, solo avanti) per verificare che gdn_fla_gfx1151.ripiego_fp64 sia davvero fp64.

Perche': il ripiego di transformers chiamato con tensori fp64 calcola in fp32 (`.to(torch.float32)` sugli ingressi,
`.float()` sulla maschera di decadimento; lo stesso fanno naive_recurrent/naive_chunk di fla). Il riferimento vero e'
G.ripiego_fp64 (stesso sorgente con float32 -> float64); qui lo si confronta con una matematica scritta da capo."""
from __future__ import annotations

import torch


def ricorrenza_fp64(q, k, v, g, beta, eps: float = 1e-6):
    """Delta rule con gate, token per token, fp64, q/k l2-normalizzati (solo avanti)."""
    q, k, v, g, beta = (t.double() for t in (q, k, v, g, beta))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + eps)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + eps)
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5
    S = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    o = torch.empty(B, T, H, V, dtype=torch.float64, device=q.device)
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        delta = (vt - torch.einsum("bhkv,bhk->bhv", S, kt)) * bt[..., None]
        S = S + kt[..., :, None] * delta[..., None, :]
        o[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t]) * scale
    return o
