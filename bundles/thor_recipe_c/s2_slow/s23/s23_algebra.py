#!/usr/bin/env python3
"""S2+S3 (piano casalingo di C, 13/10/2026) — ALGEBRA: i solutori della ricetta di C portati sul dispositivo, con TUTTE
le fattorizzazioni (Cholesky, inversa, autovalori) su CPU in fp64 e i prodotti/trsm sul dispositivo.

Perche' (memoria feedback_rocsolver_gfx1151_inaffidabile, cardan_piega.py 9/10): le fattorizzazioni rocSOLVER su gfx1151
danno NaN o «Memory access fault» coi dati veri; i trsm (rocBLAS) no. Stessa matematica dei sorgenti di riferimento,
copiati qui SOLO per renderli device-agnostici; i test (s23_test.py) li confrontano al bit (CPU fp64) con gli originali:
  gptq_bil_lotto   = raffina/solutori.gptq_bilanciato K=1 (GPTQ 2210.17323 sulla griglia bilanciata BIL), a lotti
  jarq_fedele_dev  = OBIETTIVO_C/codice/jarq_fedele.jarq_fedele (JARQ 2609.38599 Alg. 1, K=3, nu 0,6, tau 1e-5)
  drq_dev          = ptq2_costruttori.drq (DRQ 2610.11226 Alg. 1 + App. C, alpha 1, beta 0,005, tol 1 %), con lambda_max
                     per LANCZOS a riortogonalizzazione piena + Rayleigh-Ritz (App. E.3 dice «Krylov + Rayleigh-Ritz»; LOBPCG
                     di torch chiama rocSOLVER): stesso controllo di residuo, ripiego eigh CPU fp64
  fastkron_sd      = FRONTIERA fk_scale_date (FastKron 2608.06291 Alg. 6, curvatura A ⊗ B) con L_A, L_B fattorizzate su CPU
  scale_due_lati   = FRONTIERA maestro_kron.scale_due_lati (gradiente coniugato su tr(B E A E^T))
  passo_colonne    = s1b_comune.passo_colonne (ARB-LLM_RC 2410.03129 eq. 13, pesata H, asimmetrica), legame_gqa
Il JARQ a lotti degli ESPERTI e' quello di CARDAN (cardan_piega.jarq_lotto), importato dall'albero congelato.
dtype di lavoro: fp64 su CPU (prova e test), fp32 sul dispositivo (come la piega CARDAN); scale sempre arrotondate fp16.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import torch

F64 = torch.float64
EPS32 = float(torch.finfo(torch.float32).eps)
BIL_FATTORI = tuple(torch.linspace(0.35, 1.45, 23).tolist())        # = solutori.BIL_FATTORI


def dt_lavoro(dev) -> torch.dtype:
    return F64 if torch.device(dev).type == "cpu" else torch.float32


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


# ───────────────────────────── fattorizzazioni su CPU fp64 ─────────────────────────────
def _c64(M: torch.Tensor) -> torch.Tensor:
    return M.detach().to("cpu", F64)


def chol(M: torch.Tensor) -> torch.Tensor:
    """Cholesky inferiore (solleva se non SPD), fattorizzata su CPU fp64, nel dtype/dispositivo di M."""
    return torch.linalg.cholesky(_c64(M)).to(M.device, M.dtype)


def chol_ex(M: torch.Tensor):
    L, info = torch.linalg.cholesky_ex(_c64(M))
    return L.to(M.device, M.dtype), info.to(M.device)


def risolvi_chol(B: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """x con L L^T x = B; su CPU torch.cholesky_solve, sul dispositivo due trsm (rocBLAS)."""
    if L.device.type == "cpu":
        return torch.cholesky_solve(B, L)
    y = torch.linalg.solve_triangular(L, B, upper=False)
    return torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)


def risolvi(M: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(_c64(M), _c64(B)).to(M.device, M.dtype)


def cholesky_smorzata(M: torch.Tensor, percdamp: float = 0.01, passo: float = 0.0015):
    """= solutori.cholesky_smorzata (porting di G2PTQ fasterquant) su CPU fp64; ritorna (Minv, U, pd) nel dtype/disp. di M."""
    dev, dt = M.device, M.dtype
    M = _c64(M).clone()
    n = M.shape[0]
    dead = torch.diagonal(M) == 0
    if bool(dead.any()):
        M[dead, dead] = 1
    I = torch.eye(n, dtype=F64)
    dm = torch.diagonal(M).mean()
    pd = percdamp
    while 1 > pd > 0:
        try:
            Minv = torch.cholesky_inverse(torch.linalg.cholesky(M + pd * dm * I))
            U = torch.linalg.cholesky(Minv, upper=True)
            if bool(torch.isnan(U).any()):
                raise torch._C._LinAlgError
            return Minv.to(dev, dt), U.to(dev, dt), pd
        except torch._C._LinAlgError:
            pd += passo
    raise ValueError(f"damp_percent deve stare in (0,1): {pd}")


def inversa_smorzata(H: torch.Tensor, damp_rel: float = 0.01) -> torch.Tensor:
    """(H + damp_rel·mean diag·I)^-1 su CPU fp64 (bersaglio QEP: smorzamento = percdamp di GPTQ, dichiarato)."""
    Hc = _c64(H)
    Hc = 0.5 * (Hc + Hc.T)
    dm = torch.diagonal(Hc).clamp_min(0).mean().clamp_min(1e-300)
    L = torch.linalg.cholesky(Hc + damp_rel * dm * torch.eye(Hc.shape[0], dtype=F64))
    return torch.cholesky_inverse(L).to(H.device, H.dtype)


# ───────────────────────────── BIL K=1: GPTQ sulla griglia bilanciata, a lotti ─────────────────────────────
def _tab1(dev) -> torch.Tensor:
    return torch.tensor([[-1], [0], [1]], dtype=torch.int8, device=dev)          # = solutori.tabella_trit(1)


def _bil_livelli(s: torch.Tensor, tab: torch.Tensor):
    lev = s.unsqueeze(1) @ tab.to(s.dtype).T                                     # [N, 3] = S @ tab^T (K = 1)
    return torch.sort(lev, dim=1)


def _bil_vicino(w: torch.Tensor, Ls: torch.Tensor):
    L = Ls.shape[1]
    i = torch.searchsorted(Ls, w.contiguous()).clamp(1, L - 1)
    lo = Ls.gather(1, i - 1)
    hi = Ls.gather(1, i)
    su = (hi - w).abs() < (w - lo).abs()
    idx = torch.where(su, i, i - 1)
    return Ls.gather(1, idx), idx


def _bil_scegli_D(blk: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    """= solutori.bil_scegli_D con K=1 (m = 1). blk [N, G] -> D [N]."""
    amax = blk.abs().amax(1).clamp_min(1e-30)
    best_D, best_e = None, None
    for f in BIL_FATTORI:
        D = f * amax / 1.0
        Ls, _ = _bil_livelli(fp16(D), tab)
        q, _ = _bil_vicino(blk, Ls)
        e = ((q - blk) ** 2).sum(1)
        if best_D is None:
            best_D, best_e = D.clone(), e
        else:
            u = e < best_e
            best_D = torch.where(u, D, best_D)
            best_e = torch.where(u, e, best_e)
    return best_D


def gptq_bil_lotto(W: torch.Tensor, U: torch.Tensor, ng: int):
    """= solutori.gptq_bilanciato(W_b, H_b, K=1, ng) per ogni b del lotto, con U_b = chol sup. di H_b^-1 smorzata gia' data
    (cholesky_smorzata). W [B,R,C], U [B,C,C] -> T int8 [B,R,C], S [B,R,ng] (valori fp16, dtype di W)."""
    B, R, C = W.shape
    G = C // ng
    dev, dt = W.device, W.dtype
    tab = _tab1(dev)
    Wk = W.clone()
    T = torch.zeros(B, R, C, dtype=torch.int8, device=dev)
    S = torch.zeros(B, R, ng, dtype=dt, device=dev)
    dU = torch.diagonal(U, dim1=1, dim2=2)                                       # [B, C]
    for g in range(ng):
        a, b = g * G, (g + 1) * G
        D = _bil_scegli_D(Wk[:, :, a:b].reshape(B * R, G), tab)
        s = fp16(D)                                                               # bil_scale K=1
        S[:, :, g] = s.view(B, R)
        Ls, perm = _bil_livelli(s, tab)
        Err = torch.zeros(B, R, G, dtype=dt, device=dev)
        for jj in range(G):
            j = a + jj
            w = Wk[:, :, j].reshape(B * R, 1)
            q, idx = _bil_vicino(w, Ls)
            T[:, :, j] = tab[perm.gather(1, idx).squeeze(1)][:, 0].view(B, R)
            e = (w.squeeze(1) - q.squeeze(1)).view(B, R) / dU[:, j].unsqueeze(1)
            if jj + 1 < G:
                Wk[:, :, j + 1:b] -= e.unsqueeze(2) * U[:, j, j + 1:b].unsqueeze(1)
            Err[:, :, jj] = e
        if b < C:
            Wk[:, :, b:] -= torch.bmm(Err, U[:, a:b, b:])
    return T, S


def ricostruisci(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """T [..., R, C] (un piano), S [..., R, ng] -> [..., R, C] nel dtype di S."""
    ng = S.shape[-1]
    C = T.shape[-1]
    return (T.to(S.dtype).reshape(*T.shape[:-1], ng, C // ng) * S.unsqueeze(-1)).reshape(T.shape)


def obiettivo(W, What, H):
    E = W - What
    return ((E @ H) * E).sum(-1)


# ───────────────────────────── JARQ Alg. 1 fedele (densi, un tensore) ─────────────────────────────
NU, TAU, RIDGE, RCOND = 0.6, 1e-5, 1e-4, 1e-10                    # = jarq_fedele (2609.38599 §4, App. E.2)


def _scale_riga_lotto(Wt, Ha, T, ng, c):
    R, C = Wt.shape
    G = C // ng
    dt, dev = Wt.dtype, Wt.device
    Tt = T.to(dt).reshape(R, ng, G)
    Hg = Ha.reshape(ng, G, C)
    BH = torch.empty(R, ng, C, dtype=dt, device=dev)
    for g in range(ng):
        BH[:, g, :] = Tt[:, g, :] @ Hg[g]
    M = (BH.reshape(R, ng, ng, G) * Tt.unsqueeze(1)).sum(-1)
    rhs = torch.einsum("rgj,rj->rg", BH, Wt)
    Mp = (c * 0.5 * (M + M.transpose(1, 2))).float().cpu()                       # unita' del lavoro, FP32 (CPU)
    bp = (c * rhs).float().cpu()
    ev = torch.linalg.eigvalsh(Mp)
    lmax = ev.abs().amax(1).clamp_min(EPS32)
    instab = ~torch.isfinite(ev).all(1) | (ev[:, 0] <= RCOND * lmax)
    Lc, info = torch.linalg.cholesky_ex(Mp)
    instab |= info != 0
    I = torch.eye(ng).expand_as(Mp)
    Mp = torch.where(instab.view(-1, 1, 1), Mp + RIDGE * I, Mp)
    Lc, info = torch.linalg.cholesky_ex(Mp)
    if (info != 0).any():
        raise RuntimeError("JARQ: Cholesky fallita anche con ridge (App. E.2: errore numerico)")
    s = torch.cholesky_solve(bp.unsqueeze(-1), Lc).squeeze(-1).double()
    return s.to(dev, dt), int(instab.sum())


def _scale_righe(Wt, Ha, T, ng, c, lotto=256):
    out, nrid = [], 0
    for a in range(0, Wt.shape[0], lotto):
        s, n = _scale_riga_lotto(Wt[a:a + lotto], Ha, T[a:a + lotto], ng, c)
        out.append(s)
        nrid += n
    return torch.cat(out, 0), nrid


def _babai_fedele(W, Ha, T, S, attive, giro: int = 0):
    if KBEST > 1:
        return _babai_kbest(W, Ha, T, S, attive, giro)
    R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    dt, dev = W.dtype, W.device
    T = T.clone()
    What = ricostruisci(T, S)
    HE = (W - What) @ Ha
    cambi = 0
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        Hgg = Ha[sl, sl]
        Rch = chol(Hgg).T.contiguous()
        s = S[:, g]
        ok = attive & (s.abs() > EPS32)
        if not ok.any():
            continue
        zcur = What[:, sl]
        y = zcur + risolvi_chol(HE[:, sl].T.contiguous(), Rch.T.contiguous()).T
        z = torch.empty_like(y)
        q = torch.empty(R, G, dtype=dt, device=dev)
        ss = torch.where(ok, s, torch.ones_like(s))
        for l in range(G - 1, -1, -1):
            corr = ((z[:, l + 1:] - y[:, l + 1:]) * Rch[l, l + 1:]).sum(1) / Rch[l, l] if l < G - 1 else 0.0
            v = (y[:, l] - corr) / ss
            q[:, l] = torch.clamp(torch.round(v), -1, 1)
            z[:, l] = q[:, l] * ss
        dn, do = z - y, zcur - y
        e_new = ((dn @ Hgg) * dn).sum(1)
        e_old = ((do @ Hgg) * do).sum(1)
        acc = ok & torch.isfinite(e_new) & (e_new < e_old)
        n = int(acc.sum())
        if n:
            cambi += n
            T[acc, sl] = q[acc].to(torch.int8)
            dl = torch.zeros(R, G, dtype=dt, device=dev)
            dl[acc] = z[acc] - zcur[acc]
            What[acc, sl] = z[acc]
            HE -= dl @ Ha[sl, :]
    return T, cambi


# ───────────── K-best di Klein sopra il Babai di JARQ (OJBKQ, arXiv 2602.08376, feb 2026) — interruttore S23_KBEST ─────────────
# S23_KBEST=K (K >= 2): a ogni passo di codici di JARQ, per ogni gruppo, oltre al percorso di Babai deterministico (percorso 0,
#   «reference path» dell'Alg. 2 del lavoro: il punto di Babai e' sempre fra i candidati) si lanciano K percorsi di
#   back-substitution RANDOMIZZATI alla Klein (Eq. 13): q_l ~ exp(-alpha Rbar_ll^2 (c_l - v)^2) su v in {-1,0,1} (griglia
#   ternaria: il supporto e' la griglia, come B del lavoro), centro c_l aggiornato coi q gia' campionati (Alg. 3 riga 9).
#   alpha = ln(rho) / min_i Rbar_ii^2 con K = (e rho)^(2m/rho) (Liu-Ling-Stehle 2011 come citato dal lavoro), m = taglia del
#   gruppo. Rbar = R diag(s) (base scalata); il lavoro scrive Rbar_ii non al quadrato nell'Eq. 13 e r_ii^2 in alpha: qui si
#   usa la forma di Liu et al., dimensionalmente coerente (dichiarato). Per riga si tiene il candidato col MINIMO errore nella
#   metrica dell'Hessiana del gruppo (dz^T H_gg dz, lo stesso criterio di accettazione di Babai), poi la guardia «mai peggio»
#   esistente (e_new < e_old) e quella del JARQ esterno restano identiche. K <= 1 = spento: codice di prima AL BIT.
#   Seme fisso dichiarato: S23_KBEST_SEME (default 20261013), generatore CPU per (giro, gruppo) -> riproducibile e
#   indipendente dall'ordine/ripresa.
KBEST = int(os.environ.get("S23_KBEST", "1"))
KBEST_SEME = int(os.environ.get("S23_KBEST_SEME", "20261013"))


def klein_ln_rho(K: int, m: int) -> float:
    """ln(rho) con K = (e rho)^(2m/rho), rho > 1 (ramo decrescente; bisezione in ln rho)."""
    f = lambda lr: (2.0 * m / math.exp(lr)) * (1.0 + lr) - math.log(K)
    a, b = 0.0, 60.0
    if f(a) <= 0:
        return a
    for _ in range(200):
        c = 0.5 * (a + b)
        a, b = (c, b) if f(c) > 0 else (a, c)
    return 0.5 * (a + b)


def _gen(giro: int, g: int, extra: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(KBEST_SEME + 1_000_003 * giro + 7919 * g + 104_729 * extra)


def _klein_scegli(logit: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """indice campionato per riga da softmax(logit) [N, V] con l'uniforme u [N] gia' estratto (inversa della CDF). Tutto sul
    dispositivo di logit: nessuna sincronizzazione per colonna."""
    p = torch.softmax(logit, -1)
    return (p.cumsum(-1) < u.unsqueeze(-1)).sum(-1).clamp_max(p.shape[-1] - 1)


def _uniformi(gen: torch.Generator, forma, dev, dt) -> torch.Tensor:
    """uniformi estratti in BLOCCO sul generatore CPU (seme dichiarato), poi spostati una volta sul dispositivo: la sequenza
    e' la stessa su CPU e GPU."""
    return torch.rand(*forma, generator=gen, dtype=F64).to(dev, dt)


def _babai_kbest(W, Ha, T, S, attive, giro: int = 0):
    R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    dt, dev = W.dtype, W.device
    T = T.clone()
    What = ricostruisci(T, S)
    HE = (W - What) @ Ha
    cambi = 0
    lr = klein_ln_rho(KBEST, G)
    lv = torch.tensor([-1.0, 0.0, 1.0], dtype=dt, device=dev)
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        Hgg = Ha[sl, sl]
        Rch = chol(Hgg).T.contiguous()
        s = S[:, g]
        ok = attive & (s.abs() > EPS32)
        if not ok.any():
            continue
        zcur = What[:, sl]
        y = zcur + risolvi_chol(HE[:, sl].T.contiguous(), Rch.T.contiguous()).T
        ss = torch.where(ok, s, torch.ones_like(s))
        d2 = torch.diagonal(Rch).pow(2)
        w_l = lr * d2 / d2.min().clamp_min(1e-300)            # alpha Rbar_ll^2 = ln rho r_ll^2 / min r^2 (s si semplifica)
        U_k = _uniformi(_gen(giro, g), (KBEST, G, R), dev, dt)
        best_q = best_z = best_e = None
        for p in range(KBEST + 1):
            z = torch.empty_like(y)
            q = torch.empty(R, G, dtype=dt, device=dev)
            for l in range(G - 1, -1, -1):
                corr = ((z[:, l + 1:] - y[:, l + 1:]) * Rch[l, l + 1:]).sum(1) / Rch[l, l] if l < G - 1 else 0.0
                v = (y[:, l] - corr) / ss
                if p == 0:
                    q[:, l] = torch.clamp(torch.round(v), -1, 1)
                else:
                    q[:, l] = lv[_klein_scegli(-w_l[l] * (v.unsqueeze(1) - lv).pow(2), U_k[p - 1, l])]
                z[:, l] = q[:, l] * ss
            dn = z - y
            e = ((dn @ Hgg) * dn).sum(1)
            e = torch.where(torch.isfinite(e), e, torch.full_like(e, float("inf")))
            if p == 0:
                best_q, best_z, best_e = q, z, e
            else:
                m = e < best_e
                best_q = torch.where(m.unsqueeze(1), q, best_q)
                best_z = torch.where(m.unsqueeze(1), z, best_z)
                best_e = torch.where(m, e, best_e)
        q, z, e_new = best_q, best_z, best_e
        do = zcur - y
        e_old = ((do @ Hgg) * do).sum(1)
        acc = ok & torch.isfinite(e_new) & (e_new < e_old)
        n = int(acc.sum())
        if n:
            cambi += n
            T[acc, sl] = q[acc].to(torch.int8)
            dl = torch.zeros(R, G, dtype=dt, device=dev)
            dl[acc] = z[acc] - zcur[acc]
            What[acc, sl] = z[acc]
            HE -= dl @ Ha[sl, :]
    return T, cambi


def jarq_lotto_kbest_da(CP):
    """JARQ a lotti degli ESPERTI (= CP.jarq_lotto di CARDAN, stessa catena e stessa guardia «righe tornate all'ospite») col
    passo di codici K-best di Klein. CP = modulo cardan_piega congelato (helper: chol_ex/chol/risolvi_chol/_tab/...)."""
    def babai_lotto_kbest(W, H, T, S, giro):
        B, Kp, R, C = T.shape
        ng = S.shape[-1]
        G = C // ng
        dt = W.dtype
        tab = CP._tab(Kp, W.device)
        T = T.clone()
        What = CP.ricostruisci_lotto(T, S)
        HE = torch.bmm(W - What, H)
        cambi = 0
        lr = klein_ln_rho(KBEST, G)
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
            A = torch.einsum("bkr,ak->bra", S[:, :, :, g], tab.to(dt))           # [B, R, 3^K] valori in unita' di z
            seff = S[:, :, :, g].abs().sum(1).clamp_min(EPS32)                   # [B, R] passo della griglia (1 piano: = s)
            d2 = torch.diagonal(Rch, dim1=-2, dim2=-1).pow(2)                    # [B, G]
            w_l = lr * d2 / d2.min(-1, keepdim=True).values.clamp_min(1e-300)
            U_k = _uniformi(_gen(giro, g, 1), (KBEST, G, B * R), W.device, dt)
            best = None
            for p in range(KBEST + 1):
                z = torch.empty_like(y)
                idx = torch.empty(B, R, G, dtype=torch.long, device=W.device)
                for l in range(G - 1, -1, -1):
                    if l < G - 1:
                        corr = ((z[:, :, l + 1:] - y[:, :, l + 1:]) * Rch[:, l, l + 1:].unsqueeze(1)).sum(-1) \
                            / Rch[:, l, l].unsqueeze(1)
                    else:
                        corr = torch.zeros(B, R, dtype=dt, device=W.device)
                    v = y[:, :, l] - corr
                    if p == 0:
                        j = (A - v.unsqueeze(-1)).abs().argmin(-1)
                    else:
                        lg = -w_l[:, l].view(B, 1, 1) * ((A - v.unsqueeze(-1)) / seff.unsqueeze(-1)).pow(2)
                        j = _klein_scegli(lg.reshape(B * R, -1), U_k[p - 1, l]).view(B, R)
                    idx[:, :, l] = j
                    z[:, :, l] = A.gather(-1, j.unsqueeze(-1)).squeeze(-1)
                dz = z - y
                e = (torch.bmm(dz, Hgg) * dz).sum(-1)
                e = torch.where(torch.isfinite(e), e, torch.full_like(e, float("inf")))
                if best is None:
                    best = [z, idx, e]
                else:
                    m = e < best[2]
                    best = [torch.where(m.unsqueeze(-1), z, best[0]), torch.where(m.unsqueeze(-1), idx, best[1]),
                            torch.where(m, e, best[2])]
            z, idx, e_new = best
            dz_old = zcur - y
            e_old = (torch.bmm(dz_old, Hgg) * dz_old).sum(-1)
            acc = (e_new < e_old - 1e-12 * e_old.abs()) & torch.isfinite(e_new)
            n = int(acc.sum())
            if n:
                cambi += n
                tr = tab[idx]
                for k in range(Kp):
                    T[:, k, :, sl] = torch.where(acc.unsqueeze(-1), tr[..., k], T[:, k, :, sl])
                dl = torch.where(acc.unsqueeze(-1), z - zcur, torch.zeros_like(z))
                What[:, :, sl] = torch.where(acc.unsqueeze(-1), z, zcur)
                HE = HE - torch.bmm(dl, H[:, sl, :])
        return T, cambi

    def jarq_lotto_kbest(W, H, T, S, giri: int = 3, nu2: float = 1e-4):
        B, Kp, R, C = T.shape
        ng = S.shape[-1]
        dt = W.dtype
        Ha = H + nu2 * torch.eye(C, dtype=dt, device=H.device) if nu2 > 0 else H
        S = S.to(dt)
        f0 = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, S), Ha)
        Tl, Sl = T.clone(), S.clone()
        cat = {"ospite": float(f0.sum()), "kbest": KBEST, "kbest_seme": KBEST_SEME}
        for it in range(giri):
            Sl = CP._fp16(CP.scale_lotto(W, Ha, Tl, ng))
            Tl, c = babai_lotto_kbest(W, Ha, Tl, Sl, it + 1)
            cat[f"giro_{it + 1}"] = float(CP.obiettivo_lotto(W, CP.ricostruisci_lotto(Tl, Sl), Ha).sum())
            cat[f"cambi_{it + 1}"] = c
        f1 = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(Tl, Sl), Ha)
        peggio = f1 > f0
        if peggio.any():
            m = peggio.view(B, 1, R, 1)
            Tl = torch.where(m, T, Tl)
            Sl = torch.where(m, S, Sl)
        cat["righe_tornate_ospite"] = int(peggio.sum())
        return Tl, Sl, cat
    return jarq_lotto_kbest


def jarq_fedele_dev(W, H, T0, S0, c, K=3, nu=NU, tau=TAU):
    """= jarq_fedele.jarq_fedele per UN piano: T0 int8 [R,C], S0 [R,ng]. Ritorna (T, S fp16 nel dtype di W, catena)."""
    R, C = W.shape
    ng = S0.shape[-1]
    dt, dev = W.dtype, W.device
    Ha = H + (nu * nu / c) * torch.eye(C, dtype=dt, device=dev)
    T, S = T0.clone(), S0.clone().to(dt)
    attive = torch.ones(R, dtype=torch.bool, device=dev)
    Lprec = None
    cat = {"ospite": float(obiettivo(W, ricostruisci(T, S), Ha).sum()), "nu2_rel": nu * nu / c}
    for k in range(1, K + 1):
        Snew, nrid = _scale_righe(W[attive], Ha, T[attive], ng, c)
        S[attive] = Snew
        Lk = obiettivo(W, ricostruisci(T, S), Ha) * c
        if k >= 2:
            d = Lprec - Lk
            stop = attive & ((d / Lprec.abs().clamp_min(EPS32) <= tau) | (d.abs() <= tau))
            attive &= ~stop
        Lprec = Lk
        cat[f"L_scale_{k}"] = float(Lk.sum() / c)
        cat[f"ridge_{k}"] = nrid
        cat[f"attive_{k}"] = int(attive.sum())
        if not attive.any():
            break
        T, cambi = _babai_fedele(W, Ha, T, S, attive, k)
        cat[f"cambi_{k}"] = cambi
        cat[f"L_codici_{k}"] = float(obiettivo(W, ricostruisci(T, S), Ha).sum())
    S = fp16(S)
    cat["finale_fp16"] = float(obiettivo(W, ricostruisci(T, S), Ha).sum())
    return T, S, cat


# ───────────────────────────── DRQ (2610.11226), lambda_max per Lanczos ─────────────────────────────
DRQ_ALPHA, DRQ_BETA, DRQ_TOL, DRQ_GIRI, DRQ_K = 1.0, 0.005, 0.01, 3, 8          # = ptq2_costruttori
DRQ_FRAZ = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
DT_SALTI = (-2.0, -1.0, 1.0, 2.0)


def lanczos_max(M: torch.Tensor, v0: Optional[torch.Tensor] = None, kmax: int = 160, tol: float = 1e-10,
                seme: int = 20261013):
    """(lambda_max, v) di M simmetrica SPD per Lanczos con riortogonalizzazione piena (due passate) e Rayleigh-Ritz sulla
    tridiagonale (eigh CPU fp64, k <= kmax). Prodotti sul dispositivo; arresto quando |beta_k s_k| <= tol·theta."""
    n = M.shape[0]
    dev, dt = M.device, M.dtype
    if v0 is None:
        g = torch.Generator(device="cpu").manual_seed(seme)
        v = torch.randn(n, generator=g, dtype=F64).to(dev, dt)
    else:
        v = v0.to(dev, dt).clone()
    v = v / v.norm()
    V = torch.empty(min(kmax, n) + 1, n, dtype=dt, device=dev)
    V[0] = v
    al, be = [], []
    theta, y = None, v
    k = 0
    for k in range(min(kmax, n)):
        w = M @ V[k]
        a = float(w @ V[k])
        al.append(a)
        Vk = V[:k + 1]
        for _ in range(2):
            w = w - Vk.T @ (Vk @ w)
        b = float(w.norm())
        if (k + 1) % 8 == 0 or b <= 1e-14 * max(abs(a), 1e-300) or k + 1 == min(kmax, n):
            Tm = torch.diag(torch.tensor(al, dtype=F64))
            if k > 0:
                off = torch.tensor(be, dtype=F64)
                Tm += torch.diag(off, 1) + torch.diag(off, -1)
            ev, S_ = torch.linalg.eigh(Tm)
            theta = float(ev[-1])
            s = S_[:, -1]
            if abs(b * float(s[-1])) <= tol * max(abs(theta), 1e-300) or b <= 1e-14 * max(abs(a), 1e-300):
                y = s.to(dev, dt) @ Vk
                return theta, y / y.norm()
        if b <= 1e-300:
            break
        be.append(b)
        V[k + 1] = w / b
    Tm = torch.diag(torch.tensor(al, dtype=F64))
    if be:
        off = torch.tensor(be[:len(al) - 1], dtype=F64)
        Tm += torch.diag(off, 1) + torch.diag(off, -1)
    ev, S_ = torch.linalg.eigh(Tm)
    y = S_[:, -1].to(dev, dt) @ V[:len(al)]
    return float(ev[-1]), y / y.norm()


class Peggiore:
    """= ptq2_costruttori._Peggiore: S_C(W^) = lambda_max(L^-1 E^T E L^-T), C = (1-beta) H + beta I = L L^T (CPU)."""

    def __init__(self, H, beta):
        n = H.shape[0]
        Cm = (1 - beta) * _c64(H) + beta * torch.eye(n, dtype=F64)
        self.L = torch.linalg.cholesky(0.5 * (Cm + Cm.T)).to(H.device, H.dtype)
        self.v = None
        self.lanczos = 0
        self.ripieghi = 0

    def __call__(self, E):
        A = torch.linalg.solve_triangular(self.L, E.T.contiguous(), upper=False)
        M = A @ A.T
        M = 0.5 * (M + M.T)
        lam, v = lanczos_max(M, self.v)
        ok = math.isfinite(lam) and float((M @ v - lam * v).norm()) <= 1e-6 * max(lam, 1e-300)
        if ok:
            self.lanczos += 1
        else:
            self.ripieghi += 1
            ev, Vv = torch.linalg.eigh(_c64(M))
            lam, v = float(ev[-1]), Vv[:, -1].to(E.device, E.dtype)
        self.v = v / v.norm()
        u = torch.linalg.solve_triangular(self.L.T.contiguous(), self.v.unsqueeze(1), upper=True).squeeze(1)
        return lam, u


def _Lq(E, H):
    return float(((E @ H) * E).sum())


def drq_dev(W, Hs, Hcs, T0, S, alpha=DRQ_ALPHA, beta=DRQ_BETA, tol=DRQ_TOL, giri=DRQ_GIRI, kk=DRQ_K):
    """= ptq2_costruttori.drq per UN piano (T0 int8 [R,C], S [R,ng] fissa). Hs = H di ricerca grezza (X X^T / n), Hcs =
    H di controllo grezze (qui la meta' B: e' esattamente l'insieme di controllo del lavoro). Ritorna (T, catena)."""
    R, C = W.shape
    ng = S.shape[-1]
    G = C // ng
    dev, dt = W.device, W.dtype
    DTt = torch.tensor(DT_SALTI, dtype=dt, device=dev)
    sc = S.to(dt).repeat_interleave(G, dim=1)
    pegg = Peggiore(Hs, beta)
    T = T0.clone()
    E0 = sc * T.to(dt) - W
    L0 = _Lq(E0, Hs)
    L0c = [_Lq(E0, h) for h in Hcs]
    lam, u = pegg(E0)
    J = L0 + alpha * lam
    cat = {"L0": L0, "S0": lam, "J0": J, "giri": []}
    dH = torch.diagonal(Hs)
    tri = torch.triu(torch.ones(kk, kk, dtype=torch.bool, device=dev), 1).unsqueeze(0)
    for r in range(giri):
        E = sc * T.to(dt) - W
        Gm = E @ Hs
        p = E @ u
        T64 = T.to(dt)
        dJs = []
        for dts in DT_SALTI:
            tn = T64 + dts
            d = sc * dts
            dJ = 2 * d * Gm + d * d * dH + alpha * (2 * d * p.unsqueeze(1) * u + d * d * u * u)
            dJs.append(torch.where((tn >= -1) & (tn <= 1), dJ, torch.full_like(dJ, math.inf)))
        DJ = torch.stack(dJs, -1).reshape(R, C * len(DT_SALTI))
        k = min(kk, C)
        vals, idx = DJ.topk(k, dim=1, largest=False)
        col, dsalto = idx // len(DT_SALTI), DTt[idx % len(DT_SALTI)]
        dv = dsalto * sc.gather(1, col)
        Hab = Hs[col.unsqueeze(2), col.unsqueeze(1)]
        uab = u[col].unsqueeze(2) * u[col].unsqueeze(1)
        cop = vals.unsqueeze(2) + vals.unsqueeze(1) + 2 * dv.unsqueeze(2) * dv.unsqueeze(1) * (Hab + alpha * uab)
        valida = tri[:, :k, :k] & (col.unsqueeze(2) != col.unsqueeze(1))
        cop = torch.where(valida & torch.isfinite(cop), cop, torch.full_like(cop, math.inf))
        cmin, carg = cop.reshape(R, -1).min(1)
        smin = vals[:, 0]
        usa_cop = cmin < smin
        dec = torch.where(usa_cop, cmin, smin)
        righe = torch.nonzero(dec < 0).squeeze(1)
        if righe.numel() == 0:
            cat["giri"].append({"giro": r + 1, "righe": 0})
            break
        ordine = righe[torch.argsort(dec[righe])]
        a_i = torch.where(usa_cop, carg // k, torch.zeros_like(carg))
        b_i = carg % k
        prop = []
        for f in DRQ_FRAZ:
            n = max(1, int(math.ceil(f * ordine.numel())))
            if prop and prop[-1][0] == n:
                continue
            rr = ordine[:n]
            Tn = T.clone()
            ca, da = col[rr, a_i[rr]], dsalto[rr, a_i[rr]]
            Tn[rr, ca] = (Tn[rr, ca].to(dt) + da).to(torch.int8)
            rc = rr[usa_cop[rr]]
            if rc.numel():
                cb, db = col[rc, b_i[rc]], dsalto[rc, b_i[rc]]
                Tn[rc, cb] = (Tn[rc, cb].to(dt) + db).to(torch.int8)
            assert int(Tn.abs().max()) <= 1
            En = sc * Tn.to(dt) - W
            Ln = _Lq(En, Hs)
            okc = Ln <= (1 + tol) * L0 and all(_Lq(En, h) <= (1 + tol) * l0 for h, l0 in zip(Hcs, L0c))
            if not okc:
                prop.append((n, None, None, None))
                continue
            lam_n, u_n = pegg(En)
            Jn = Ln + alpha * lam_n
            prop.append((n, Jn, Tn, u_n) if Jn < J else (n, None, None, None))
        valide = [x for x in prop if x[1] is not None]
        cat["giri"].append({"giro": r + 1, "righe": int(righe.numel()), "provate": [x[0] for x in prop],
                            "valide": [x[0] for x in valide]})
        if not valide:
            break
        n, J, T, u = min(valide, key=lambda x: x[1])
        cat["giri"][-1]["scelta"] = n
        pegg(sc * T.to(dt) - W)
    E = sc * T.to(dt) - W
    cat.update({"L": _Lq(E, Hs), "J": J, "Lc0": L0c, "Lc": [_Lq(E, h) for h in Hcs],
                "trit_cambiati": float((T != T0).double().mean()), "lanczos": pegg.lanczos, "ripieghi_eigh": pegg.ripieghi})
    return T, cat


# ───────────────────────────── K1: FastKron a scale date + scale a due lati ─────────────────────────────
def L_unit(M: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """= solutori._L_unit: L unitaria inferiore dalla Cholesky superiore di M^-1 smorzata (CPU fp64)."""
    _, U, _ = cholesky_smorzata(M, damp)
    return (U / torch.diagonal(U).unsqueeze(1)).T.contiguous()


def fastkron_sd(Wres, A, B, S_dato, damp=0.01):
    """= fk_scale_date.fastkron_scale_date con Kl = 1 (FastKron 2608.06291 Alg. 6). Wres [R,C], A [C,C], B [R,R] (o None),
    S_dato [R,ng]. Ritorna T int8 [R,C]."""
    R, C = Wres.shape
    ng = S_dato.shape[-1]
    G = C // ng
    dev, dt = Wres.device, Wres.dtype
    LA = L_unit(A.to(dt), damp)
    LB = L_unit(B.to(dt), damp) if B is not None else torch.eye(R, dtype=dt, device=dev)
    W = Wres.to(dt).clone()
    tab = _tab1(dev).to(dt)
    ALF = torch.einsum("rg,ak->rga", S_dato.to(dt), tab)
    V = torch.zeros_like(W)
    IDX = torch.zeros(R, C, dtype=torch.long, device=dev)
    m, n = R, C
    ii = torch.arange(m, device=dev).view(-1, 1)
    jj = torch.arange(n, device=dev).view(1, -1)
    zero = torch.zeros((), dtype=dt, device=dev)

    def box(k, l):
        return max(0, k - n + 1), min(m, l), max(0, k - m + 1), min(n, l)

    def fk(k, l):
        if l == k + 1:
            r0 = max(0, k - n + 1)
            r1 = min(m, k + 1)
            if r0 >= r1:
                return
            ri = torch.arange(r0, r1, device=dev)
            cj = k - ri
            w = W[ri, cj]
            alf = ALF[ri, cj // G]
            j = (alf - w.unsqueeze(1)).abs().argmin(1)
            V[ri, cj] = alf.gather(1, j.unsqueeze(1)).squeeze(1)
            IDX[ri, cj] = j
            return
        s = (k + l) // 2
        fk(k, s)
        a0, a1, b0, b1 = box(k, s)
        t0, t1, u0, u1 = box(s, l)
        if a0 < a1 and b0 < b1 and t0 < t1 and u0 < u1:
            d = ii[a0:a1] + jj[:, b0:b1]
            msk = (d >= k) & (d < s)
            Dl = torch.where(msk, V[a0:a1, b0:b1] - W[a0:a1, b0:b1], zero)
            Up = LB[t0:t1, a0:a1] @ Dl @ LA[u0:u1, b0:b1].T
            d2 = ii[t0:t1] + jj[:, u0:u1]
            W[t0:t1, u0:u1] += torch.where((d2 >= s) & (d2 < l), Up, zero)
        fk(s, l)

    sys.setrecursionlimit(max(20000, sys.getrecursionlimit()))
    fk(0, m + n - 1)
    return tab[IDX][..., 0].to(torch.int8).contiguous()


def espandi(s: torch.Tensor, C: int) -> torch.Tensor:
    return s.repeat_interleave(C // s.shape[-1], dim=-1)


def scale_due_lati(W, T, s0, A, B, it=60):
    """= maestro_kron.scale_due_lati: min_s tr(B (W - s⊙T) A (W - s⊙T)^T) per gradiente coniugato; s [R, ng]."""
    R, C = W.shape
    ng = s0.shape[-1]
    Tf = T.to(W.dtype)
    gs = lambda M: M.reshape(R, ng, C // ng).sum(-1)  # noqa: E731
    Hv = lambda d: gs((B @ (espandi(d, C) * Tf) @ A) * Tf)  # noqa: E731
    s = s0.clone().to(W.dtype)
    r = gs((B @ W @ A) * Tf) - Hv(s)
    p = r.clone()
    rr = float((r * r).sum())
    r0 = rr
    for _ in range(it):
        Ap = Hv(p)
        den = float((p * Ap).sum())
        if den <= 0 or rr == 0:
            break
        a = rr / den
        s = s + a * p
        r = r - a * Ap
        rn = float((r * r).sum())
        if rn < 1e-12 * r0:
            break
        p = r + (rn / rr) * p
        rr = rn
    return s


def J_due_lati(E, A, B) -> float:
    return float(((B @ E @ A) * E).sum())


# ───────────────────────────── K2: scala per colonna piegata (forma chiusa) ─────────────────────────────
def passo_colonne(Qo, Hq, Clin, P=None, ridge=1e-9):
    """= s1b_comune.passo_colonne (sistema risolto su CPU fp64). Qo [R,C] effettivo SENZA colonna (base originale)."""
    A = Hq * (Qo.T @ Qo)
    bc = (Qo * Clin).sum(0)
    if P is not None:
        A = P.T @ A @ P
        bc = P.T @ bc
    A = _c64(A)
    bc = _c64(bc)
    A = 0.5 * (A + A.T)
    A = A + ridge * torch.diagonal(A).clamp_min(0).mean() * torch.eye(A.shape[0], dtype=F64)
    try:
        cm = torch.linalg.solve(A, bc)
    except RuntimeError:
        cm = torch.linalg.lstsq(A, bc.unsqueeze(-1)).solution.squeeze(-1)
    cm = cm.to(Qo.device, Qo.dtype)
    return P @ cm if P is not None else cm


def legame_gqa(n_head: int, n_kv: int, head_dim: int, dev="cpu", dt=F64) -> torch.Tensor:
    """= s1b_comune.legame_gqa: P [n_head*hd, n_kv*hd] (teste kv ripetute in blocchi contigui, come repeat_kv)."""
    rep = n_head // n_kv
    P = torch.zeros(n_head * head_dim, n_kv * head_dim, dtype=dt, device=dev)
    j = torch.arange(n_head * head_dim, device=dev)
    P[j, (j // head_dim // rep) * head_dim + j % head_dim] = 1.0
    return P


def scale_scrivibili(S: torch.Tensor, S0: torch.Tensor) -> bool:
    """come s1b_comune.scala_ok: finite, dentro fp16, nessuna scala non nulla dell'ospite diventata zero."""
    return bool(torch.isfinite(S).all() and (S.abs() <= 65504.0).all() and not ((S0 != 0) & (S == 0)).any())
