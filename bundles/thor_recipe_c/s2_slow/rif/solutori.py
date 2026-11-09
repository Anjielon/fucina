"""Risolutori delle leve di raffinamento sul formato a piani ternari W' = sum_k s_k T_k (scala per riga e gruppo).

Tutto nella base memorizzata (gia' ruotata): W [R,C] bersaglio, H [C,C] Hessiana (mean diag = 1), T int8 [K,R,C],
S fp32 [K,R,ng] (valori fp16). Ogni risolutore restituisce (T, S) con S ESATTAMENTE fp16 e la catena dell'obiettivo.

Il «composito» di un gruppo con K piani liberi prende |A| = 3^K valori  a = sum_k s_k t_k  (t in {-1,0,1}^K):
la decisione per elemento e' l'indice nell'alfabeto, poi si scompone nei trit dei piani.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import torch

import os
F64 = getattr(torch, os.getenv("LEVE_DT", "float64"))


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


def tabella_trit(K: int, dev=None) -> torch.Tensor:
    """[3^K, K] tutte le combinazioni di trit (ordine fisso)."""
    return torch.tensor(list(itertools.product((-1, 0, 1), repeat=K)), dtype=torch.int8, device=dev)


def ricostruisci(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    K, R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    w = torch.zeros(R, C, dtype=F64, device=T.device)
    for k in range(K):
        w += (T[k].to(F64).reshape(R, ng, G) * S[k].to(F64).unsqueeze(-1)).reshape(R, C)
    return w


def obiettivo(W: torch.Tensor, What: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """per riga: (w - ŵ) H (w - ŵ)ᵀ."""
    E = (W - What).to(F64)
    return ((E @ H) * E).sum(1)


# ═══════════════════════════════════════ passo delle SCALE (congiunto) ═══════════════════════════════════════
def scale_congiunte(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, Wfisso: torch.Tensor, ng: int,
                    righe_lotto: int = 128, ridge: float = 1e-4, legame: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Minimi quadrati CONGIUNTI di tutte le scale di una riga (tutti i piani liberi x tutti i gruppi), JARQ eq. 6:
    min_s || W - Wfisso - sum_{k,g} s_kg T_kg ||_H.  T [K,R,C] (solo i piani liberi).
    legame: se dato [K] (es. (9,3,1)), i K piani di un gruppo condividono UNA scala s_g con s_kg = legame_k * s_g
    (Tied Trit-Planes: il composito diventa un quantizzatore uniforme).  Ritorna S [K,R,ng] fp64 (non arrotondate)."""
    K, R, C = T.shape
    G = C // ng
    Hg = H.reshape(ng, G, C)
    out = torch.zeros(K, R, ng, dtype=F64, device=W.device)
    for a in range(0, R, righe_lotto):
        b = min(R, a + righe_lotto)
        Tt = T[:, a:b].to(F64).reshape(K, b - a, ng, G)
        if legame is not None:                                        # un solo «piano» composito c = sum legame_k t_k
            Tt = (Tt * legame.to(F64).view(K, 1, 1, 1)).sum(0, keepdim=True)
        Kk = Tt.shape[0]
        r_ = b - a
        BH = torch.empty(Kk, r_, ng, C, dtype=F64, device=W.device)                    # (B H) per riga: ng prodotti densi (BLAS)
        for g in range(ng):
            BH[:, :, g, :] = (Tt[:, :, g, :].reshape(Kk * r_, G) @ Hg[g]).reshape(Kk, r_, C)
        # M[r,(k,g),(l,q)] = sum_{j in q} BH[k,r,g,j] T[l,r,q,j]   -> matmul batched su (r, q)
        M = torch.empty(r_, Kk, ng, Kk, ng, dtype=F64, device=W.device)
        for l in range(Kk):
            prod = BH * Tt[l].reshape(1, r_, 1, C)                    # [Kk, r, ng, C]
            M[:, :, :, l, :] = prod.reshape(Kk, r_, ng, ng, G).sum(-1).permute(1, 0, 2, 3)
        rhs = torch.einsum("krgj,rj->rkg", BH, (W[a:b] - Wfisso[a:b]).to(F64))
        n = Kk * ng
        M = M.reshape(b - a, n, n)
        rhs = rhs.reshape(b - a, n)
        dia = torch.diagonal(M, dim1=1, dim2=2)
        vuoti = dia <= 0                                              # colonna senza codici: scala libera -> 0
        M = M + torch.diag_embed(vuoti.to(F64))
        sol = None
        try:
            L = torch.linalg.cholesky(M)
            sol = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        except RuntimeError:
            pass
        if sol is None or not torch.isfinite(sol).all():
            M2 = M + ridge * torch.diag_embed(dia.clamp_min(1e-12))
            sol = torch.linalg.solve(M2, rhs.unsqueeze(-1)).squeeze(-1)
        sol = torch.where(vuoti, torch.zeros_like(sol), sol).reshape(b - a, Kk, ng)
        if legame is not None:
            out[:, a:b] = legame.to(F64).view(K, 1, 1) * sol[:, 0].unsqueeze(0)
        else:
            out[:, a:b] = sol.permute(1, 0, 2)
    return out


# ═══════════════════════════════════════ passo dei CODICI (Babai per gruppo) ═══════════════════════════════════════
def _alfabeto(S: torch.Tensor, g: int, tab: torch.Tensor) -> torch.Tensor:
    """valori del composito per riga nel gruppo g: [R, 3^K] = S[:, :, g]ᵀ @ tabᵀ."""
    return S[:, :, g].T.to(F64) @ tab.to(F64).T


def codici_babai(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, S: torch.Tensor, Wfisso: torch.Tensor,
                 ordine_gruppi: Optional[List[int]] = None) -> Tuple[torch.Tensor, int]:
    """Un giro JARQ dei codici (eq. 7-10): per ogni gruppo, proposta di Babai «nearest plane» sul composito dei piani
    liberi (alfabeto finito, non uniforme), accettata riga per riga solo se abbassa l'errore del gruppo.
    Ritorna (T nuovo, n. di gruppi-riga cambiati)."""
    K, R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    tab = tabella_trit(K, W.device)
    T = T.clone()
    What = Wfisso.to(F64) + ricostruisci(T, S)
    E = W.to(F64) - What                                              # errore corrente
    HE = E @ H                                                        # [R, C]
    cambi = 0
    for g in (ordine_gruppi or range(ng)):
        sl = slice(g * G, (g + 1) * G)
        Hgg = H[sl, sl]
        Rch = torch.linalg.cholesky(Hgg).T.contiguous()               # superiore, RᵀR = Hgg
        zcur = What[:, sl]
        # ottimo continuo del gruppo dato il resto: y = zcur + Hgg^-1 (H E)_g
        y = zcur + torch.cholesky_solve(HE[:, sl].T.contiguous(), Rch.T.contiguous()).T
        A = _alfabeto(S, g, tab)                                      # [R, 3^K]
        z = torch.empty_like(y)
        idx = torch.empty(R, G, dtype=torch.long, device=W.device)
        for l in range(G - 1, -1, -1):                                # all'indietro: prima l'ultima coordinata
            if l < G - 1:
                corr = ((z[:, l + 1:] - y[:, l + 1:]) * Rch[l, l + 1:]).sum(1) / Rch[l, l]
            else:
                corr = torch.zeros(R, dtype=F64, device=W.device)
            v = y[:, l] - corr
            j = (A - v.unsqueeze(1)).abs().argmin(1)
            idx[:, l] = j
            z[:, l] = A.gather(1, j.unsqueeze(1)).squeeze(1)
        dz_new = z - y
        dz_old = zcur - y
        e_new = ((dz_new @ Hgg) * dz_new).sum(1)
        e_old = ((dz_old @ Hgg) * dz_old).sum(1)
        acc = (e_new < e_old - 1e-12 * e_old.abs()) & torch.isfinite(e_new)
        n = int(acc.sum())
        if n:
            cambi += n
            tr = tab[idx[acc]]                                        # [n, G, K]
            for k in range(K):
                T[k, acc, sl] = tr[:, :, k]
            dl = torch.zeros(R, G, dtype=F64, device=W.device)
            dl[acc] = z[acc] - zcur[acc]
            What[acc, sl] = z[acc]
            HE -= dl @ H[sl, :]                                       # E diminuisce di delta (solo colonne del gruppo)
    return T, cambi


def jarq(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, S: torch.Tensor, liberi: List[int],
         giri: int = 3, nu2: float = 0.0, legame: Optional[torch.Tensor] = None, tau: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """JARQ (arXiv 2609.38599, Alg. 1) adattato ai piani ternari: si alternano (i) scale congiunte di tutti i gruppi
    e piani liberi, arrotondate a fp16, (ii) proposte di Babai per gruppo sul composito; ospite = (T, S) di R2B3m.
    nu2 = penalita' sui pesi (nu^2 ||W - Ŵ||^2, aggiunta a H). Garanzia: se il risultato finale e' peggiore
    dell'ospite su una riga (puo' capitare solo per l'arrotondamento fp16), quella riga torna all'ospite."""
    K, R, C = T.shape
    ng = S.shape[-1]
    Ha = H + nu2 * torch.eye(C, dtype=F64, device=H.device) if nu2 > 0 else H
    fissi = [k for k in range(K) if k not in liberi]
    Wfisso = ricostruisci(T[fissi], S[fissi]) if fissi else torch.zeros(R, C, dtype=F64, device=W.device)
    Tl, Sl = T[liberi].clone(), S[liberi].clone().to(F64)
    f0 = obiettivo(W, Wfisso + ricostruisci(Tl, Sl), Ha)
    catena = {"ospite": float(f0.sum())}
    prec = f0.sum()
    for it in range(giri):
        Sl = fp16(scale_congiunte(W, Ha, Tl, Wfisso, ng, legame=legame))
        if legame is not None:                                        # legame esatto anche dopo fp16: s_k = legame_k * s_ultimo
            base = Sl[-1]
            Sl = fp16(legame.to(F64).view(-1, 1, 1) * base.unsqueeze(0))
        Tl, cambi = codici_babai(W, Ha, Tl, Sl, Wfisso)
        f = obiettivo(W, Wfisso + ricostruisci(Tl, Sl), Ha).sum()
        catena[f"giro_{it+1}"] = float(f)
        catena[f"cambi_{it+1}"] = cambi
        if it >= 1 and (prec - f) <= tau * prec:
            break
        prec = f
    f1 = obiettivo(W, Wfisso + ricostruisci(Tl, Sl), Ha)
    peggio = f1 > f0
    if peggio.any():
        Tl[:, peggio] = T[liberi][:, peggio]
        Sl[:, peggio] = S[liberi][:, peggio].to(F64)
    T2, S2 = T.clone(), S.clone().to(F64)
    T2[liberi], S2[liberi] = Tl, Sl
    catena["righe_tornate_ospite"] = int(peggio.sum())
    catena["finale"] = float(obiettivo(W, ricostruisci(T2, S2), Ha).sum())
    return T2, S2.float(), catena


# ═══════════════════════════════════════ #1 Tied Trit-Planes ═══════════════════════════════════════
def tied_init(W: torch.Tensor, Wfisso: torch.Tensor, K: int, ng: int, legame: torch.Tensor, giri_lloyd: int = 6):
    """Avvio del quantizzatore uniforme a 3^K livelli (c in {-(3^K-1)/2..}) con scala per gruppo: Lloyd NON pesato sul
    residuo W - Wfisso (come il risolutore alternato di PTQTP), poi scomposizione ternaria bilanciata di c."""
    R, C = W.shape
    G = C // ng
    X = (W - Wfisso).to(F64).reshape(R, ng, G)
    cmax = (3 ** K - 1) // 2
    s = X.abs().amax(-1, keepdim=True) / cmax
    s = torch.where(s > 0, s, torch.ones_like(s))
    for _ in range(giri_lloyd):
        c = torch.clamp(torch.round(X / s), -cmax, cmax)
        num = (c * X).sum(-1, keepdim=True)
        den = (c * c).sum(-1, keepdim=True)
        s = torch.where(den > 0, num / den.clamp_min(1e-30), s)
    c = torch.clamp(torch.round(X / s), -cmax, cmax).reshape(R, C).long()
    T = trit_bilanciati(c, K)
    S = (legame.to(F64).view(K, 1, 1) * s.reshape(1, R, ng))
    return T, fp16(S)


def trit_bilanciati(c: torch.Tensor, K: int) -> torch.Tensor:
    """c intero in [-(3^K-1)/2, (3^K-1)/2] -> K trit, cifra piu' significativa nel piano 0 (c = sum 3^(K-1-k) t_k)."""
    out = []
    x = c.clone()
    for k in range(K - 1, -1, -1):
        r = ((x + 1) % 3) - 1                                         # resto bilanciato in {-1,0,1}
        out.append(r)
        x = (x - r) // 3
    out.reverse()
    return torch.stack(out).to(torch.int8)


# ═══════════════════════════════════════ #11 SchurOpt ═══════════════════════════════════════
def schuropt(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, S: torch.Tensor, liberi: List[int], ng: int,
             passi: int = 4, lam: float = 0.0, Wref: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """SchurOpt (arXiv 2608.15567, Alg. 1) sul composito dei piani liberi: si visitano i gruppi (chunk di una scala)
    da sinistra a destra; il prefisso e' gia' discreto, il suffisso resta CONTINUO al suo ottimo. Per il chunk
    corrente: S = Pcc^-1 (complemento di Schur), T_lin = C_eff_c - C_eff_r G_rr^-1 G_rc, poi `passi` alternanze di
    (scale ottime per riga col complemento di Schur, eq. 16 senza zero-point) e discesa per coordinate sul
    composito (eq. 17, con il termine incrociato Φ = Q S). Avvio dai codici di R2B3m (raffinamento dell'ospite).
    Obiettivo: ½ tr(W G Wᵀ) - tr(C Wᵀ) con G = H + lam I, C = Wb H + lam Wref (Wb = bersaglio dei piani liberi)."""
    K, R, C = T.shape
    Gs = C // ng
    fissi = [k for k in range(K) if k not in liberi]
    Wfisso = ricostruisci(T[fissi], S[fissi]) if fissi else torch.zeros(R, C, dtype=F64, device=W.device)
    Wb = (W - Wfisso).to(F64)                                          # i piani liberi inseguono questo
    Gm = H + lam * torch.eye(C, dtype=F64, device=H.device)
    Cm = Wb @ H + (lam * (Wref - Wfisso) if (lam > 0 and Wref is not None) else lam * Wb)
    Tl, Sl = T[liberi].clone(), S[liberi].clone().to(F64)
    Kl = len(liberi)
    tab = tabella_trit(Kl, W.device).to(F64)                                    # [A, Kl]
    f0 = obiettivo(Wb, ricostruisci(Tl, Sl), H)
    P = torch.cholesky_inverse(torch.linalg.cholesky(Gm))             # (G[t:,t:])^-1, t = 0
    Wq = torch.zeros(R, C, dtype=F64, device=W.device)                                 # prefisso discreto (si riempie)
    catena = {"ospite": float(f0.sum())}
    for g in range(ng):
        t0 = g * Gs
        c = slice(0, Gs)                                              # indici relativi in P (che parte da t0)
        Pcc, Pcr = P[:Gs, :Gs], P[:Gs, Gs:]
        Ssc = torch.linalg.inv(Pcc)
        Ssc = 0.5 * (Ssc + Ssc.T)
        Kmat = -Pcr.T @ Ssc                                           # G_rr^-1 G_rc   [m-g, g]
        # C_eff = C - W_F G_F·   (prefisso discreto)
        if t0 > 0:
            WF = Wq[:, :t0]
            Ceff_c = Cm[:, t0:t0 + Gs] - WF @ Gm[:t0, t0:t0 + Gs]
            Ceff_r = Cm[:, t0 + Gs:] - WF @ Gm[:t0, t0 + Gs:]
        else:
            Ceff_c = Cm[:, t0:t0 + Gs]
            Ceff_r = Cm[:, t0 + Gs:]
        Tlin = Ceff_c - Ceff_r @ Kmat if Ceff_r.shape[1] else Ceff_c  # [R, Gs]
        # stato iniziale del chunk: codici e scale dell'ospite
        codes = Tl[:, :, t0:t0 + Gs].to(F64)                          # [Kl, R, Gs]
        sc = Sl[:, :, g].clone()                                      # [Kl, R]
        for it in range(passi):
            # (a) scale ottime per riga: min ½ qSqᵀ - t qᵀ con q = sum_k s_k code_k  ->  M s = b
            CS = torch.einsum("krj,ji->kri", codes, Ssc)              # [Kl, R, Gs]
            M = torch.einsum("kri,lri->rkl", CS, codes)               # [R, Kl, Kl]
            bb = torch.einsum("ri,kri->rk", Tlin, codes)
            dia = torch.diagonal(M, dim1=1, dim2=2)
            vuoti = dia <= 0
            M = M + torch.diag_embed(vuoti.to(F64)) + 1e-10 * torch.diag_embed(dia.abs())
            s_new = torch.linalg.solve(M + 1e-6 * M.diagonal(dim1=-2, dim2=-1).abs().mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-12) * torch.eye(M.shape[-1], device=M.device, dtype=M.dtype), bb.unsqueeze(-1)).squeeze(-1)   # casa 7/10: ridge contro M singolare (piani tutti zero in un gruppo)
            s_new = torch.where(vuoti, torch.zeros_like(s_new), s_new)
            sc = fp16(s_new.T.contiguous())                           # [Kl, R]
            # (b) discesa per coordinate sul composito: q = sum_k sc_k code_k
            A = sc.T @ tab.T                                          # [R, A]
            q = (codes * sc.unsqueeze(-1)).sum(0)                     # [R, Gs]
            Phi = q @ Ssc                                             # [R, Gs]
            idx = None
            for j in range(Gs):
                Sjj = Ssc[j, j]
                cross = Phi[:, j] - q[:, j] * Sjj - Tlin[:, j]
                val = 0.5 * Sjj * A * A + cross.unsqueeze(1) * A      # [R, A]
                jj = val.argmin(1)
                vnew = A.gather(1, jj.unsqueeze(1)).squeeze(1)
                d = vnew - q[:, j]
                if bool((d != 0).any()):
                    Phi += d.unsqueeze(1) * Ssc[j].unsqueeze(0)
                    q[:, j] = vnew
                codes[:, :, j] = tab[jj].T
        # rifit finale delle scale e chiusura del chunk
        CS = torch.einsum("krj,ji->kri", codes, Ssc)
        M = torch.einsum("kri,lri->rkl", CS, codes)
        bb = torch.einsum("ri,kri->rk", Tlin, codes)
        dia = torch.diagonal(M, dim1=1, dim2=2)
        vuoti = dia <= 0
        M = M + torch.diag_embed(vuoti.to(F64)) + 1e-10 * torch.diag_embed(dia.abs())
        s_new = torch.linalg.solve(M + 1e-6 * M.diagonal(dim1=-2, dim2=-1).abs().mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-12) * torch.eye(M.shape[-1], device=M.device, dtype=M.dtype), bb.unsqueeze(-1)).squeeze(-1)   # casa 7/10: ridge contro M singolare (piani tutti zero in un gruppo)
        sc = fp16(torch.where(vuoti, torch.zeros_like(s_new), s_new).T.contiguous())
        Tl[:, :, t0:t0 + Gs] = codes.to(torch.int8)
        Sl[:, :, g] = sc
        Wq[:, t0:t0 + Gs] = (codes * sc.unsqueeze(-1)).sum(0)
        P = P[Gs:, Gs:] - Pcr.T @ Ssc @ Pcr                           # P_next (eq. 14)
        P = 0.5 * (P + P.T)
    f1 = obiettivo(Wb, ricostruisci(Tl, Sl), H)
    catena["schur"] = float(f1.sum())
    peggio = f1 > f0
    if peggio.any():
        Tl[:, peggio] = T[liberi][:, peggio]
        Sl[:, peggio] = S[liberi][:, peggio].to(F64)
    catena["righe_tornate_ospite"] = int(peggio.sum())
    T2, S2 = T.clone(), S.clone().to(F64)
    T2[liberi], S2[liberi] = Tl, Sl
    catena["finale"] = float(obiettivo(W, ricostruisci(T2, S2), H).sum())
    return T2, S2.float(), catena


# ═══════════════════════════════════════ #12 QTEA: GPTQ ternario con decadimento dell'errore ═══════════════════════════════════════
def gptq_piano_decay(Wres: torch.Tensor, H: torch.Tensor, ng: int, lam_decay: float = 0.0,
                     riscala_colonne: bool = False, giri_riscala: int = 3,
                     damp: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """UN piano ternario su Wres con GPTQ colonna per colonna (ordine naturale), scala di gruppo = ottimo ternario
    esatto del gruppo al suo inizio (come la ricetta di casa), e i due pezzi di QTEA (arXiv 2609.00224):
      * error decay (eq. 8-9): l'errore propagato dalla colonna i e' moltiplicato per
        γ_i = exp(-λ · [H^-1]_ii / mean_k [H^-1]_kk · i/B), B = blocco di colonne (qui il gruppo di scala);
      * column-wise rescale (eq. 4-5) se richiesto: v_j per colonna, alternato con la riassegnazione dei trit
        (ritornato a parte: si puo' caricare solo se assorbito altrove, vedi applica.py).
    Ritorna (T [R,C] int8, S [R,ng] fp32-fp16, v [C] o None)."""
    R, C = Wres.shape
    G = C // ng
    Hd = H + damp * torch.eye(C, dtype=F64, device=H.device)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hd))
    U = torch.linalg.cholesky(Hinv, upper=True)                       # GPTQ: Hinv = Uᵀ U
    dmean = torch.diagonal(Hinv).mean()
    Wk = Wres.to(F64).clone()
    T = torch.zeros(R, C, dtype=torch.int8, device=Wres.device)
    S = torch.zeros(R, ng, dtype=F64, device=Wres.device)
    v = torch.ones(C, dtype=F64, device=Wres.device) if riscala_colonne else None
    for g in range(ng):
        a, b = g * G, (g + 1) * G
        blk = Wk[:, a:b]
        # scala ternaria esatta del gruppo (supporto top-k) sul blocco corrente
        ab = blk.abs()
        so, _ = torch.sort(ab, dim=1, descending=True)
        cs = torch.cumsum(so, 1)
        kk = torch.arange(1, G + 1, dtype=F64, device=Wres.device)
        i = torch.argmax(cs * cs / kk, 1, keepdim=True)
        s = fp16(torch.gather(cs, 1, i) / (i + 1).to(F64)).squeeze(1)  # [R]
        S[:, g] = s
        Err = torch.zeros(R, G, dtype=F64, device=Wres.device)
        for jj in range(G):
            j = a + jj
            w = Wk[:, j]
            if riscala_colonne:
                t = torch.clamp(torch.round(w / torch.where(s != 0, s, torch.ones_like(s))), -1, 1)
                for _ in range(giri_riscala):
                    nz = (t != 0) & (s != 0)
                    if nz.any():
                        v[j] = (w[nz].abs() / s[nz].abs()).mean()
                    t = torch.clamp(torch.round(w / torch.where(s * v[j] != 0, s * v[j], torch.ones_like(s))), -1, 1)
                q = t * s * v[j]
            else:
                t = torch.clamp(torch.round(w / torch.where(s != 0, s, torch.ones_like(s))), -1, 1)
                q = t * s
            T[:, j] = t.to(torch.int8)
            e = (w - q) / U[j, j]
            gam = math.exp(-lam_decay * float(Hinv[j, j] / dmean) * (jj / G)) if lam_decay > 0 else 1.0
            Wk[:, j + 1:b] -= gam * e.unsqueeze(1) * U[j, j + 1:b].unsqueeze(0)
            Err[:, jj] = gam * e
        if b < C:
            Wk[:, b:] -= Err @ U[a:b, b:]
    return T, S.float(), v


# ═══════════════════════════════════════ smorzamento con auto-incremento (porting fedele del codice ufficiale) ═══════════════════════════════════════
def cholesky_smorzata(M: torch.Tensor, percdamp: float = 0.01, passo: float = 0.0015):
    """Porting di G2PTQ gptq_utils/g2ptq_utils.py (fasterquant, righe 399-450): colonne morte -> diag 1, poi
    damp = percdamp * mean(diag) e, se Cholesky di Md o di Md^-1 fallisce (o da' NaN), percdamp += 0.0015 finche' < 1.
    Serve perche' le curvature salvate (A, B in fp16, mean diag 1) non sono piu' semidefinite: l'arrotondamento fp16 di
    una Gram mal condizionata (ssm_out: autovalori fino a -0.046 su 4096) supera lo smorzamento fisso 1%.
    Ritorna (Minv = Md^-1, U = chol(Minv) superiore, percdamp usato)."""
    n = M.shape[0]
    M = M.clone()
    dead = torch.diagonal(M) == 0
    if bool(dead.any()):
        M[dead, dead] = 1
    I = torch.eye(n, dtype=M.dtype, device=M.device)
    dm = torch.diagonal(M).mean()
    pd = percdamp
    while 1 > pd > 0:
        try:
            Minv = torch.cholesky_inverse(torch.linalg.cholesky(M + pd * dm * I))
            U = torch.linalg.cholesky(Minv, upper=True)
            if bool(torch.isnan(U).any()):
                raise torch._C._LinAlgError
            return Minv, U, pd
        except torch._C._LinAlgError:
            pd += passo
    raise ValueError(f"damp_percent deve stare in (0,1): {pd}")


# ═══════════════════════════════════════ #17 G2PTQ: GPTQ con compensazione del gradiente generalizzata ═══════════════════════════════════════
def _scale_gruppo_composito(blk: torch.Tensor, Kl: int) -> torch.Tensor:
    """Scale di avvio di un gruppo per Kl piani: ottimo ternario esatto (supporto top-k) del blocco, poi del resto,
    ... (la ricetta residua di casa).  blk [R, G] -> [Kl, R] (fp16)."""
    R, G = blk.shape
    res = blk.clone()
    out = []
    kk = torch.arange(1, G + 1, dtype=F64, device=blk.device)
    for _ in range(Kl):
        so, _ = torch.sort(res.abs(), dim=1, descending=True)
        cs = torch.cumsum(so, 1)
        i = torch.argmax(cs * cs / kk, 1, keepdim=True)
        s = fp16(torch.gather(cs, 1, i) / (i + 1).to(F64)).squeeze(1)
        t = torch.clamp(torch.round(res / torch.where(s != 0, s, torch.ones_like(s)).unsqueeze(1)), -1, 1)
        res = res - t * s.unsqueeze(1)
        out.append(s)
    return torch.stack(out)


def g2ptq_composito(Wres: torch.Tensor, H: torch.Tensor, Grad: torch.Tensor, ng: int, Kl: int = 2, alpha: float = 1.0,
                    damp: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Porting di G2PTQ (gptq_utils/g2ptq_utils.py: fasterquant + _run_torch_compensation, una sola «channel group»)
    al composito di Kl piani ternari: colonna per colonna, q = valore piu' vicino dell'alfabeto 3^Kl del gruppo
    (scale fissate all'inizio del gruppo), errore compensato col termine di GPTQ PIU' il termine del gradiente
    ESATTO della perdita globale (Grad), scalato per riga dalla regione di fiducia beta (Teorema 3, eq. 13-14,
    stessa formula del codice: beta = mean_t (1 - sqrt(max(1 - 2 alpha |dL_gptq| / c, 0))))."""
    R, C = Wres.shape
    G = C // ng
    dev = Wres.device
    Hinv_init, U, damp_usato = cholesky_smorzata(H.to(F64), damp)
    W = Wres.to(F64).clone()
    Gr = Grad.to(F64)
    tab = tabella_trit(Kl, dev).to(F64)
    # quantizzazione "finta" RTN (scale di avvio per gruppo sul peso originale) per la regione di fiducia
    Q0 = torch.empty_like(W)
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        sc = _scale_gruppo_composito(W[:, sl], Kl)
        A = sc.T @ tab.T
        j = (W[:, sl].unsqueeze(-1) - A.unsqueeze(1)).abs().argmin(-1)
        Q0[:, sl] = A.gather(1, j)
    eps = torch.finfo(torch.float32).tiny
    GHinv0 = Gr @ Hinv_init
    qe = Q0 - W
    dH = torch.diagonal(Hinv_init).unsqueeze(0)
    loss_gptq = ((qe * GHinv0 + qe ** 2 / 2) / dH).abs()
    c = (Gr * GHinv0).sum(1, keepdim=True) - GHinv0 ** 2 / dH
    c = c.clamp(min=2 * alpha * loss_gptq).clamp(min=eps)
    beta = (1 - torch.sqrt(torch.clamp(1 - (2 * alpha * loss_gptq) / c, min=0.0))).mean(-1) if alpha > 0 \
        else torch.zeros(R, dtype=F64, device=dev)
    Z = (Gr @ U.T) * beta.unsqueeze(1)
    GH = Z @ U
    T = torch.zeros(Kl, R, C, dtype=torch.int8, device=dev)
    S = torch.zeros(Kl, R, ng, dtype=F64, device=dev)
    for g in range(ng):
        a, b = g * G, (g + 1) * G
        sc = _scale_gruppo_composito(W[:, a:b], Kl)
        S[:, :, g] = sc
        A = sc.T @ tab.T                                              # [R, 3^Kl]
        for t in range(a, b):
            w = W[:, t]
            j = (A - w.unsqueeze(1)).abs().argmin(1)
            q = A.gather(1, j.unsqueeze(1)).squeeze(1)
            T[:, :, t] = tab[j].T.to(torch.int8)
            err = (w - q - GH[:, t]) / U[t, t]
            if t + 1 < C:
                W[:, t + 1:] -= GH[:, t + 1:]
                W[:, t + 1:] -= err.unsqueeze(1) * U[t, t + 1:].unsqueeze(0)
                GH[:, t + 1:] -= Z[:, t].unsqueeze(1) * U[t, t + 1:].unsqueeze(0)
    return T, S, {"beta_media": float(beta.mean()), "beta_max": float(beta.max()), "damp": damp_usato}


# ═══════════════════════════════════════ #8 FastKron: GPTQ a DUE lati (Hessiana A ⊗ B) ═══════════════════════════════════════
def _L_unit(M: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """L unitaria inferiore dalla Cholesky superiore di M^-1 (M smorzata): GPTQ aggiorna W[:, j'] += Δ_j L[j', j]."""
    _, U, _ = cholesky_smorzata(M, damp)
    return (U / torch.diagonal(U).unsqueeze(1)).T.contiguous()


def fastkron_composito(Wres: torch.Tensor, A: torch.Tensor, B: torch.Tensor, ng: int, Kl: int = 2,
                       damp: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """FastKron (arXiv 2608.06291, Alg. 6: antidiagonali in parallelo + divide-et-impera, equivalente per il Teorema 2.1
    a GPTQ sul peso vettorizzato con curvatura A ⊗ B) con il quantizzatore di base = valore piu' vicino del composito
    a Kl piani ternari, scale STATICHE per (riga, gruppo) fissate sul bersaglio (ricetta residua di casa).
    A [C,C] curvatura d'ingresso, B [R,R] curvatura d'uscita (YAQA: E[g gᵀ] della KL)."""
    R, C = Wres.shape
    G = C // ng
    dev = Wres.device
    LA = _L_unit(A.to(F64), damp)
    LB = _L_unit(B.to(F64), damp) if B is not None else torch.eye(R, dtype=F64, device=dev)
    W = Wres.to(F64).clone()
    tab = tabella_trit(Kl, dev).to(F64)
    S = torch.stack([_scale_gruppo_composito(W[:, g * G:(g + 1) * G], Kl) for g in range(ng)], dim=2)  # [Kl, R, ng]
    ALF = torch.einsum("krg,ak->rga", S, tab)                         # [R, ng, 3^Kl]
    V = torch.zeros_like(W)
    IDX = torch.zeros(R, C, dtype=torch.long, device=dev)
    m, n = R, C
    ii = torch.arange(m, device=dev).view(-1, 1)
    jj = torch.arange(n, device=dev).view(1, -1)

    def box(k, l):
        return max(0, k - n + 1), min(m, l), max(0, k - m + 1), min(n, l)

    def fk(k, l):
        if l == k + 1:
            r0 = max(0, k - n + 1); r1 = min(m, k + 1)
            if r0 >= r1:
                return
            ri = torch.arange(r0, r1, device=dev)
            cj = k - ri
            w = W[ri, cj]
            alf = ALF[ri, cj // G]                                    # [len, 3^Kl]
            j = (alf - w.unsqueeze(1)).abs().argmin(1)
            V[ri, cj] = alf.gather(1, j.unsqueeze(1)).squeeze(1)
            IDX[ri, cj] = j
            return
        s = (k + l) // 2
        fk(k, s)
        a0, a1, b0, b1 = box(k, s)                                    # sorgente: antidiagonali [k, s)
        t0, t1, u0, u1 = box(s, l)                                    # bersaglio: antidiagonali [s, l)
        if a0 < a1 and b0 < b1 and t0 < t1 and u0 < u1:
            d = ii[a0:a1] + jj[:, b0:b1]
            msk = (d >= k) & (d < s)
            Dl = torch.where(msk, V[a0:a1, b0:b1] - W[a0:a1, b0:b1], torch.zeros((), dtype=F64, device=dev))
            Up = LB[t0:t1, a0:a1] @ Dl @ LA[u0:u1, b0:b1].T
            d2 = ii[t0:t1] + jj[:, u0:u1]
            W[t0:t1, u0:u1] += torch.where((d2 >= s) & (d2 < l), Up, torch.zeros((), dtype=F64, device=dev))
        fk(s, l)

    import sys as _s
    _s.setrecursionlimit(max(10000, _s.getrecursionlimit()))
    fk(0, m + n - 1)
    T = tab[IDX].permute(2, 0, 1).to(torch.int8).contiguous()        # [Kl, R, C]
    return T, S, {}


def gptq_statico_composito(Wres: torch.Tensor, A: torch.Tensor, ng: int, Kl: int = 2, damp: float = 0.01):
    """GPTQ a un lato con le STESSE scale statiche di fastkron_composito (solo per il collaudo: con B = I i due
    devono coincidere al bit, Teorema 2.1)."""
    R, C = Wres.shape
    G = C // ng
    LA = _L_unit(A.to(F64), damp)
    W = Wres.to(F64).clone()
    tab = tabella_trit(Kl, W.device).to(F64)
    S = torch.stack([_scale_gruppo_composito(W[:, g * G:(g + 1) * G], Kl) for g in range(ng)], dim=2)
    ALF = torch.einsum("krg,ak->rga", S, tab)
    IDX = torch.zeros(R, C, dtype=torch.long, device=W.device)
    for j in range(C):
        alf = ALF[:, j // G]
        jx = (alf - W[:, j].unsqueeze(1)).abs().argmin(1)
        v = alf.gather(1, jx.unsqueeze(1)).squeeze(1)
        IDX[:, j] = jx
        W[:, j:] += (v - W[:, j]).unsqueeze(1) * LA[j:, j].unsqueeze(0)
    return tab[IDX].permute(2, 0, 1).to(torch.int8).contiguous(), S


# ═══════════════════════════════════════ BIL: GPTQ su griglia ternaria BILANCIATA (9/10, cura della costruzione lenta) ═══════════════════════════════════════
# Leva BIL (#26, 2609.05641 → griglia uniforme): K piani con scale s_k = fp16(3^(K-1-k) D) per gruppo formano la griglia
# a 3^K livelli del ternario bilanciato; ogni peso va al livello piu' vicino con la retroazione dell'errore di GPTQ
# (2210.17323, Alg. 1, aggiornamento pigro per gruppo). Stessa matematica di thor_v2/leve_g4/bil_gptq.py (7/10), che
# costava 181 s per attn_qkv (~10 h per 250 tensori) perche' cercava il livello piu' vicino a forza bruta su un tensore
# [righe, 128, 3^K] per ognuno dei 23 fattori di ogni gruppo. Qui: livelli ordinati per riga + searchsorted, che da' lo
# STESSO livello (il piu' vicino, esatto) in O(log 3^K). Nessuna approssimazione.
BIL_FATTORI = tuple(torch.linspace(0.35, 1.45, 23).tolist())


def bil_livelli(S: torch.Tensor, tab: torch.Tensor):
    """S [R, K] scale fp16 della riga nel gruppo, tab [L, K] trit -> (livelli ordinati [R, L], permutazione [R, L])."""
    lev = S.to(F64) @ tab.to(F64).T
    return torch.sort(lev, dim=1)


def bil_vicino(w: torch.Tensor, Ls: torch.Tensor):
    """w [R, n], Ls [R, L] crescente -> (valore del livello piu' vicino [R, n], indice nell'ordinamento [R, n])."""
    L = Ls.shape[1]
    i = torch.searchsorted(Ls, w.contiguous()).clamp(1, L - 1)
    lo = Ls.gather(1, i - 1)
    hi = Ls.gather(1, i)
    su = (hi - w).abs() < (w - lo).abs()
    idx = torch.where(su, i, i - 1)
    return Ls.gather(1, idx), idx


def bil_scale(D: torch.Tensor, K: int) -> torch.Tensor:
    """D [R] -> [R, K] scale fp16 3^(K-1-k) D (come il formato le leggera')."""
    p = torch.tensor([3.0 ** (K - 1 - k) for k in range(K)], dtype=F64, device=D.device)
    return fp16(D.to(F64).unsqueeze(1) * p.unsqueeze(0))


def bil_scegli_D(blk: torch.Tensor, K: int, tab: torch.Tensor) -> torch.Tensor:
    """blk [R, G] (pesi gia' compensati all'inizio del gruppo) -> D [R] che minimizza l'errore RTN euclideo sulla
    griglia bilanciata, cercando D = f max|w| / m sui 23 fattori di bil_gptq.py."""
    m = (3 ** K - 1) / 2
    amax = blk.abs().amax(1).clamp_min(1e-30)
    best_D, best_e = None, None
    for f in BIL_FATTORI:
        D = f * amax / m
        Ls, _ = bil_livelli(bil_scale(D, K), tab)
        q, _ = bil_vicino(blk, Ls)
        e = ((q - blk) ** 2).sum(1)
        if best_D is None:
            best_D, best_e = D.clone(), e
        else:
            u = e < best_e
            best_D = torch.where(u, D, best_D)
            best_e = torch.where(u, e, best_e)
    return best_D


def gptq_bilanciato(W: torch.Tensor, H: torch.Tensor, K: int, ng: int, damp: float = 0.01,
                    foem_beta: float = 0.0, foem_forma: str = "lavoro") -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """GPTQ (ordine naturale, aggiornamento pigro per gruppo di scala) sulla griglia bilanciata a 3^K livelli.
    W [R, C] bersaglio nella base memorizzata, H [C, C] (mean diag 1). Ritorna (T [K,R,C] int8, S [K,R,ng] fp16 in F64, catena).
    foem_beta > 0, foem_forma="lavoro" (default): FOEM (2507.11017) come nell'Algoritmo 1 / Eq. 19 del LAVORO:
        E_j = ((w~_j - q_j) - beta (w~_j - w0_j)) / U_jj
        W~[:, j:b] -= E_j U[j, j:b] + beta (W~[:, j:b] - W0[:, j:b])        (solo dentro il blocco)
        W~[:, b:]  -= E U[a:b, b:]                                         (aggiornamento pigro invariato)
      Forma INVARIANTE alla scala di H (i termini in H e H^-1 si elidono, §3.3): beta = 0,1 del lavoro si usa tale e quale.
    foem_forma="riferimento" (12/10/2026, CURA del difetto di f0 = 25,30): l'implementazione di RIFERIMENTO degli autori
    (GPTQModel, gptqmodel/quantization/foem.py e mlx_foem.py; il README ufficiale dice che i numeri attuali vengono da li'):
        E_j = ((w~_j - q_j) - beta (w~_j - w0_j)) / U_jj ;  W~[:, j+1:b] -= E_j U[j, j+1:b] ;
        POI  W~[:, j+1] -= beta (W~[:, j+1] - W0[:, j+1])   (SOLO la colonna successiva, una volta)
      beta adimensionale (0,1 del lavoro; GPTQModel 0,2, intervallo consigliato [0,1; 0,25]).
      DIFETTO della forma "lavoro" qui sotto: applica -beta (W~ - W0) a TUTTE le colonne j..b del blocco a OGNI colonna,
      quindi la colonna j+k viene riportata verso W0 di un fattore (1-beta)^k (0,9^127 ~ 1e-6 a fine gruppo): la
      compensazione GPTQ viene cancellata e i pesi latenti tornano al bf16 -> quasi RTN (errH x2, MMLU 25,30 su C).
    foem_forma="codice": il codice UFFICIALE (github Xingyu-Zheng/FOEM, LLM/weight-only/gptq.py fasterquant, ramo
    ours_v2 senza GPTAQ): a ogni colonna i del blocco
        W1[:, i:] -= err_i U[i, i:] + beta (W1[:, i:] - W0[:, i:]) (U[i:, i:]^T U[i:, i:])
    e a ogni blocco  W[:, b:] -= Err U[a:b, b:] + beta (W[:, b:] - W0[:, b:]) (U[b:, b:]^T U[b:, b:]),
    con U = chol(H^-1) superiore e W0 = pesi di partenza. Qui foem_beta e' il beta GIA' riportato alla scala di questa H
    (vedi foem_beta_efficace): il termine e' beta·(W~-W0)·H^-1 e dipende dalla scala di H."""
    R, C = W.shape
    G = C // ng
    tab = tabella_trit(K, W.device)
    _, U, pd = cholesky_smorzata(H.to(F64), percdamp=damp)
    Wk = W.to(F64).clone()
    W0 = W.to(F64).clone() if foem_beta > 0 else None
    T = torch.zeros(K, R, C, dtype=torch.int8, device=W.device)
    S = torch.zeros(K, R, ng, dtype=F64, device=W.device)
    for g in range(ng):
        a, b = g * G, (g + 1) * G
        D = bil_scegli_D(Wk[:, a:b], K, tab)
        Sg = bil_scale(D, K)                                          # [R, K]
        S[:, :, g] = Sg.T
        Ls, perm = bil_livelli(Sg, tab)
        Err = torch.zeros(R, G, dtype=F64, device=W.device)
        for jj in range(G):
            j = a + jj
            w = Wk[:, j:j + 1]
            q, idx = bil_vicino(w, Ls)
            T[:, :, j] = tab[perm.gather(1, idx).squeeze(1)].T
            e = (w.squeeze(1) - q.squeeze(1)) / U[j, j]
            if foem_beta > 0 and foem_forma == "riferimento":
                e = e - foem_beta * (w.squeeze(1) - W0[:, j]) / U[j, j]
                if jj + 1 < G:
                    Wk[:, j + 1:b] -= e.unsqueeze(1) * U[j, j + 1:b].unsqueeze(0)
                    Wk[:, j + 1] -= foem_beta * (Wk[:, j + 1] - W0[:, j + 1])
            elif foem_beta > 0 and foem_forma == "lavoro":
                e = e - foem_beta * (w.squeeze(1) - W0[:, j]) / U[j, j]
                Wk[:, j:b] -= e.unsqueeze(1) * U[j, j:b].unsqueeze(0) + foem_beta * (Wk[:, j:b] - W0[:, j:b])
            elif foem_beta > 0:
                Ui = U[j:b, j:b]
                rit = ((Wk[:, j:b] - W0[:, j:b]) @ Ui.T) @ Ui               # (W~ - W0) (U^T U) sulle colonne j..b
                Wk[:, j:b] -= e.unsqueeze(1) * U[j, j:b].unsqueeze(0) + foem_beta * rit
            elif jj + 1 < G:
                Wk[:, j + 1:b] -= e.unsqueeze(1) * U[j, j + 1:b].unsqueeze(0)
            Err[:, jj] = e
        if b < C:
            if foem_beta > 0 and foem_forma == "codice":
                Ub = U[b:, b:]
                Wk[:, b:] -= Err @ U[a:b, b:] + foem_beta * (((Wk[:, b:] - W0[:, b:]) @ Ub.T) @ Ub)
            else:
                Wk[:, b:] -= Err @ U[a:b, b:]
    return T, S, {"damp_usato": pd, "foem_beta": foem_beta, "foem_forma": foem_forma}


def foem_beta_efficace(beta: float, diag_media_grezza: float, token_per_campione: int = 2048) -> float:
    """beta del lavoro (definito sulla H del loro codice: H = (2/nsamples) sum_campioni X X^T = 2·seqlen·E[x x^T], seqlen
    2048, damp 1 % della media della diagonale) -> beta sulla NOSTRA H normalizzata (media della diagonale 1).
    H_loro = 2·seqlen·dbar·H_nostra  =>  beta·(W~-W0)·H_loro^-1 = [beta / (2·seqlen·dbar)]·(W~-W0)·H_nostra^-1.
    L'errore del 2/9 era qui: beta del repository usato tale e quale su una H di altra scala (beta·lambda_max(H^-1) > 2)."""
    return beta / (2.0 * token_per_campione * diag_media_grezza)


def foem_rho(H: torch.Tensor, beta_eff: float, damp: float = 0.01) -> float:
    """rho = beta_eff · lambda_max((H + damp·mean diag·I)^-1): fattore di contrazione del passo FOEM nella direzione piu'
    piatta di H. Stabile solo se rho < 2 (con rho < 1 non oscilla)."""
    ev = torch.linalg.eigvalsh(H.to(torch.float64) + damp * torch.diagonal(H).mean().to(torch.float64) * torch.eye(H.shape[0], dtype=torch.float64, device=H.device))
    return float(beta_eff / ev.min().clamp_min(1e-30))
