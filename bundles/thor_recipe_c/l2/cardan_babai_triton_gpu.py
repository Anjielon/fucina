#!/usr/bin/env python3
"""cardan_babai_triton_gpu.py (21/10 giro 7): il ciclo colonna per colonna di cardan_piega_gpu.babai_lotto in UN kernel
Triton per gruppo invece di ~8 lanci torch per colonna (G = 128/256 colonne x ng gruppi x 3 giri x lotti: centinaia di
migliaia di lanci piccoli per famiglia di esperti, il motivo per cui 5 processi sulla stessa GPU rendono 2x).

Stessa matematica (Babai nearest-plane sull'alfabeto composito, all'indietro sulle colonne del gruppo):
    corr_l = sum_{m>l} (z_m - y_m) R[l, m] / R[l, l];  v = y_l - corr_l;  j = argmin_a |A_a - v|;  z_l = A_j
L'unica differenza e' l'ordine della somma di corr_l (riduzione Triton invece di torch.sum): scarto all'ultimo bit del
dtype, che puo' cambiare un argmin solo a pari distanza entro l'arrotondamento; si dichiara misurandolo (byte diversi,
quota catturata). Accettazione per riga, aggiornamento di T/What/HE e algebra (chol, risolvi_chol) restano quelli
dell'originale (babai_lotto copiato riga per riga, cambia solo il ciclo interno).
Attivazione: FORGIA_BABAI_TRITON=1 negli ingressi _gpu (cardan_piega_gpu_atomica.py, cardan_piega_densi_gpu_veloce.py),
che chiamano installa(CP) -> CP.babai_lotto = babai_lotto (jarq_lotto lo cerca per nome a ogni chiamata).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _babai_kernel(Y, RC, A, Z, IDX, R, NA,
                  sy_b, sy_r, sy_g, sr_b, sr_i, sr_j, sa_b, sa_r, sa_a, sz_b, sz_r, sz_g, si_b, si_r, si_g,
                  G: tl.constexpr, BR: tl.constexpr, GP: tl.constexpr, NAP: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    rows = tl.program_id(1) * BR + tl.arange(0, BR)
    rm = rows < R
    cols = tl.arange(0, GP)
    cm = cols < G
    aa = tl.arange(0, NAP)
    am = aa < NA
    Ablk = tl.load(A + b * sa_b + rows[:, None] * sa_r + aa[None, :] * sa_a, mask=rm[:, None] & am[None, :],
                   other=float("inf"))
    d = tl.zeros([BR, GP], dtype=Ablk.dtype)
    for t in range(0, G):
        l = G - 1 - t
        rrow = tl.load(RC + b * sr_b + l * sr_i + cols * sr_j, mask=cm & (cols > l), other=0.0)
        rll = tl.load(RC + b * sr_b + l * sr_i + l * sr_j)
        corr = tl.sum(d * rrow[None, :], axis=1) / rll
        yl = tl.load(Y + b * sy_b + rows * sy_r + l * sy_g, mask=rm, other=0.0)
        v = yl - corr
        j = tl.argmin(tl.abs(Ablk - v[:, None]), axis=1, tie_break_left=True)
        zl = tl.sum(tl.where(aa[None, :] == j[:, None], Ablk, 0.0), axis=1)
        tl.store(Z + b * sz_b + rows * sz_r + l * sz_g, zl, mask=rm)
        tl.store(IDX + b * si_b + rows * si_r + l * si_g, j.to(tl.int64), mask=rm)
        d = tl.where(cols[None, :] == l, (zl - yl)[:, None], d)


def babai_colonne(y: torch.Tensor, Rch: torch.Tensor, A: torch.Tensor):
    """y [B,R,G], Rch [B,G,G] (triangolare superiore), A [B,R,NA] -> (z [B,R,G], idx long [B,R,G])."""
    B, R, G = y.shape
    NA = A.shape[-1]
    z = torch.empty_like(y)
    idx = torch.empty(B, R, G, dtype=torch.long, device=y.device)
    GP = triton.next_power_of_2(G)
    BR = int(os.environ.get("FORGIA_BABAI_BR", "16" if GP <= 128 else "8"))
    NAP = max(2, triton.next_power_of_2(NA))
    grid = (B, triton.cdiv(R, BR))
    _babai_kernel[grid](y, Rch, A, z, idx, R, NA, *y.stride(), *Rch.stride(), *A.stride(), *z.stride(),
                        *idx.stride(), G=G, BR=BR, GP=GP, NAP=NAP, num_warps=4)
    return z, idx


def babai_lotto(W, H, T, S):
    """cardan_piega_gpu.babai_lotto con il ciclo sulle colonne in Triton (vedi sopra). Returns (T new, n. changed).
    Algebra (chol_ex, chol, risolvi_chol) e aiuti presi dal modulo passato a installa (predefinito cardan_piega_gpu)."""
    CP = _MOD[0]
    if CP is None:
        import cardan_piega_gpu as CP
    B, K, R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    dt = W.dtype
    tab = CP._tab(K, W.device)
    T = T.clone()
    What = CP.ricostruisci_lotto(T, S)
    HE = torch.bmm(W - What, H)
    cambi = 0
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        Hgg = H[:, sl, sl]
        Lg, info = CP.chol_ex(Hgg)
        if bool((info != 0).any()):
            bad = info != 0
            eye = torch.eye(G, dtype=dt, device=W.device)
            Hgg = Hgg.clone()
            Hgg[bad] = Hgg[bad] + 1e-3 * torch.diagonal(Hgg[bad], dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1) * eye
            Lg = CP.chol(Hgg)
        Rch = Lg.transpose(-1, -2).contiguous()
        zcur = What[:, :, sl]
        y = zcur + CP.risolvi_chol(HE[:, :, sl].transpose(-1, -2), Rch.transpose(-1, -2)).transpose(-1, -2)
        A = torch.einsum("bkr,ak->bra", S[:, :, :, g], tab.to(dt))
        z, idx = babai_colonne(y.contiguous(), Rch, A.contiguous())
        dz_new = z - y
        dz_old = zcur - y
        e_new = (torch.bmm(dz_new, Hgg) * dz_new).sum(-1)
        e_old = (torch.bmm(dz_old, Hgg) * dz_old).sum(-1)
        acc = (e_new < e_old - 1e-12 * e_old.abs()) & torch.isfinite(e_new)
        n = int(acc.sum())
        if n:
            cambi += n
            tr = tab[idx]
            for k in range(K):
                T[:, k, :, sl] = torch.where(acc.unsqueeze(-1), tr[..., k], T[:, k, :, sl])
            dl = torch.where(acc.unsqueeze(-1), z - zcur, torch.zeros_like(z))
            What[:, :, sl] = torch.where(acc.unsqueeze(-1), z, zcur)
            HE -= torch.bmm(dl, H[:, sl, :])
    return T, cambi


_MOD = [None]


def installa(CP) -> bool:
    """Sostituisce CP.babai_lotto (cardan_piega_gpu o cardan_piega: stessa firma) se FORGIA_BABAI_TRITON=1."""
    if os.environ.get("FORGIA_BABAI_TRITON") != "1":
        return False
    _MOD[0] = CP
    CP.babai_lotto = babai_lotto
    print("[babai] ciclo colonne Triton (cardan_babai_triton_gpu)", flush=True)
    return True
