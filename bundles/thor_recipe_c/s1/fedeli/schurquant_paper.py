#!/usr/bin/env python3
"""SchurQuant FEDELE al lavoro (Lee, Son, Lee, Kim, Ahn — arXiv 2608.15567), nucleo matematico.

10/10/2026 sera. Nasce dall'audit di aderenza (ricerche/AUDIT_ADERENZA_PAPER.md): la leva `schur_pk/p2k` usava solo
l'ottimizzatore SchurOpt (R = 4, avvio dall'ospite, obiettivo GPTQ con l'Hessiana del manifesto) e NON SchurQuant.
Qui c'e' il metodo del lavoro, equazione per equazione:

  * statistiche sufficienti (Eq. 7-9):     G = X Ω Xᵀ + λ I,   C = Y Ω Xᵀ + λ W_ref
    (convenzione del codice: X [N, din] token-righe, quindi G = Xᵀ Ω X, C = Yᵀ Ω X; W [dout, din]);
  * riduzione di Schur sul gruppo corrente (Prop. 1, Eq. 11-13) con la ricorrenza dell'inversa (Eq. 14);
  * scala e zero-point esatti per riga (Prop. 2, Eq. 16), enumerando tutti i 2^b zero-point (App. B.2);
  * discesa per coordinate sui codici interi col termine incrociato Φ = Q S (Eq. 17);
  * Algoritmo 1 completo (avvio min-max per riga, R rifiniture, rifit finale, P ← P_rr − P_crᵀ S P_cr);
  * pesi per token della decisione del maestro (Eq. 18-19): α_i = 1 + (ρ−1) m_i normalizzati a media 1,
    ω_i = ᾱ_i / max(r_i, ε), r_i = ‖W_ref x_i − y_i‖₂, ρ = 8, ε = 1e-8 (App. D).

Variante THOR (dichiarata, NON nel lavoro): stesso ottimizzatore con il formato ternario a K piani di Thor —
codici simmetrici {-1,0,1} per piano, una scala fp16 per piano e per gruppo di g colonne (TQ1_0R_1024: g = 128),
insieme dei candidati per coordinata = i 3^K valori Σ_k a_k t_k; le K scale per riga si stimano col minimo esatto
della forma quadratica (generalizzazione di Prop. 2 con o = 0: K×K, esatta per codici fissi). Avvio: «min-max»
ternario per piani avidi (analogo del min-max del lavoro, dichiarato) oppure i piani dell'ospite.

Tutto in float64. Funziona su CPU e GPU (argomento device implicito nei tensori)."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

F64 = torch.float64


# ───────────────────────────────────── statistiche sufficienti (Eq. 7-9, 22) ─────────────────────────────────────
class Statistiche:
    """Accumula Xᵀ Ω X e Yᵀ Ω X a lotti. `finale(lam, Wref)` restituisce (G, C) dell'Eq. 8."""

    def __init__(self, din: int, dout: int, device="cpu"):
        self.XtX = torch.zeros(din, din, dtype=F64, device=device)
        self.YtX = torch.zeros(dout, din, dtype=F64, device=device)
        self.n = 0
        self.massa = 0.0

    @torch.no_grad()
    def aggiungi(self, X: torch.Tensor, Y: torch.Tensor, omega: Optional[torch.Tensor] = None) -> None:
        X = X.reshape(-1, X.shape[-1]).to(F64)
        Y = Y.reshape(-1, Y.shape[-1]).to(F64)
        if omega is None:
            Xw = X
            self.massa += float(X.shape[0])
        else:
            w = omega.reshape(-1).to(F64)
            Xw = X * w.unsqueeze(1)
            self.massa += float(w.sum())
        self.XtX += X.T @ Xw
        self.YtX += Y.T @ Xw
        self.n += X.shape[0]

    def finale(self, lam: float, Wref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        G = self.XtX.clone()
        C = self.YtX.clone()
        if lam > 0:
            G += lam * torch.eye(G.shape[0], dtype=F64, device=G.device)
            C += lam * Wref.to(F64)
        return 0.5 * (G + G.T), C


def obiettivo(W: torch.Tensor, G: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Eq. 9 a meno della costante: ½ tr(W G Wᵀ) − tr(C Wᵀ)."""
    W = W.to(F64)
    return 0.5 * ((W @ G) * W).sum() - (C * W).sum()


# ───────────────────────────────────── pesi per token (Eq. 18-19) ─────────────────────────────────────
def pesi_decisione(arg_q: torch.Tensor, arg_t: torch.Tensor, rho: float = 8.0) -> torch.Tensor:
    """ᾱ_i: α_i = 1 + (ρ − 1)·1[argmax_q ≠ argmax_t], normalizzati a media 1 (Eq. 18)."""
    m = (arg_q.reshape(-1) != arg_t.reshape(-1)).to(F64)
    a = 1.0 + (rho - 1.0) * m
    return a / a.mean()


def pesi_proiezione(abar: torch.Tensor, Wref: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """ω_i = ᾱ_i / max(r_i, ε), r_i = ‖W_ref x_i − y_i‖₂ (Eq. 19, residuo staccato, norma L2 non al quadrato)."""
    X = X.reshape(-1, X.shape[-1]).to(F64)
    Y = Y.reshape(-1, Y.shape[-1]).to(F64)
    r = torch.linalg.vector_norm(X @ Wref.to(F64).T - Y, dim=1)
    return abar.reshape(-1).to(F64) / r.clamp_min(eps)


# ───────────────────────────────────── quantizzatore affine del lavoro (Eq. 1, 16, 17) ─────────────────────────────────────
def minmax_affine(Wc: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Avvio per riga del chunk (Alg. 1 riga 9): a = (max − min)/(2^b − 1), o = round(−min/a), z = clip(round(w/a) + o)."""
    qmax = 2 ** bits - 1
    mn = torch.minimum(Wc.min(1).values, torch.zeros_like(Wc[:, 0]))
    mx = torch.maximum(Wc.max(1).values, torch.zeros_like(Wc[:, 0]))
    a = ((mx - mn) / qmax).clamp_min(1e-12)
    o = torch.round(-mn / a).clamp(0, qmax)
    z = torch.clamp(torch.round(Wc / a.unsqueeze(1)) + o.unsqueeze(1), 0, qmax)
    return z, a, o


def scala_zero_esatti(z: torch.Tensor, S: torch.Tensor, T: torch.Tensor, bits: int,
                      eps_a: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prop. 2 (Eq. 16): per codici fissi, per ogni zero-point o ∈ {0..2^b−1}:
    u = z − o, A = u S uᵀ, d = t uᵀ, a* = max(d/A, ε_a); si sceglie o* che minimizza ½a*²A − a*d."""
    qmax = 2 ** bits - 1
    cand = list(range(qmax + 1))
    best_v = None
    a_out = torch.zeros(z.shape[0], dtype=F64, device=z.device)
    o_out = torch.zeros(z.shape[0], dtype=F64, device=z.device)
    for o in cand:
        u = z - o
        A = ((u @ S) * u).sum(1)
        d = (T * u).sum(1)
        a = torch.where(A > 0, d / A.clamp_min(1e-300), torch.full_like(A, eps_a)).clamp_min(eps_a)
        v = 0.5 * a * a * A - a * d
        if best_v is None:
            best_v, a_out, o_out = v, a, torch.full_like(a, float(o))
        else:
            meglio = v < best_v
            best_v = torch.where(meglio, v, best_v)
            a_out = torch.where(meglio, a, a_out)
            o_out = torch.where(meglio, torch.full_like(a, float(o)), o_out)
    return a_out, o_out


def spazzata_affine(z: torch.Tensor, a: torch.Tensor, o: torch.Tensor, S: torch.Tensor, T: torch.Tensor,
                    bits: int) -> torch.Tensor:
    """Una spazzata di discesa per coordinate (Eq. 17), stato Φ = Q S aggiornato di rango uno."""
    qmax = 2 ** bits - 1
    lv = torch.arange(qmax + 1, dtype=F64, device=z.device)                 # codici candidati
    V = a.unsqueeze(1) * (lv.unsqueeze(0) - o.unsqueeze(1))                  # [R, 2^b] valori v_{i,k}
    q = a.unsqueeze(1) * (z - o.unsqueeze(1))
    Phi = q @ S
    z = z.clone()
    for j in range(z.shape[1]):
        Sjj = S[j, j]
        cross = Phi[:, j] - q[:, j] * Sjj - T[:, j]
        val = 0.5 * Sjj * V * V + cross.unsqueeze(1) * V
        k = val.argmin(1)
        vnew = V.gather(1, k.unsqueeze(1)).squeeze(1)
        dlt = vnew - q[:, j]
        if bool((dlt != 0).any()):
            Phi += dlt.unsqueeze(1) * S[j].unsqueeze(0)
            q[:, j] = vnew
        z[:, j] = k.to(F64)
    return z


def locale(q: torch.Tensor, S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Eq. 15: Σ_i ½ q_i S q_iᵀ − t_i q_iᵀ (per riga)."""
    return 0.5 * ((q @ S) * q).sum(1) - (T * q).sum(1)


def smorza(G: torch.Tensor, damp: float) -> torch.Tensor:
    """Alg. 1 riga 1 letta alla lettera: G + ε_damp·diag(G) (matrice diagonale della diagonale di G). La stessa G
    smorzata entra in P e nei coefficienti efficaci, cosi' la riduzione di Schur e' esatta per l'obiettivo smorzato."""
    return G + damp * torch.diag_embed(torch.diagonal(G).clamp_min(0))


def schuropt_affine(G: torch.Tensor, C: torch.Tensor, Wref: torch.Tensor, bits: int, g: int = 128, R: int = 16,
                    damp: float = 0.01, griglia_fissa: bool = False, curvatura: str = "schur",
                    traccia: Optional[list] = None) -> Dict[str, torch.Tensor]:
    """Algoritmo 1 del lavoro, per un solo strato lineare (quantizzatore affine asimmetrico, App. D).
    `curvatura="raw"` (suffisso fermo) e `griglia_fissa=True` servono solo alle prove dell'ablazione di Tab. 3."""
    dout, din = Wref.shape
    assert din % g == 0, f"din {din} non divisibile per g {g}"
    Gd = smorza(G.to(F64), damp)
    C = C.to(F64)
    Wref = Wref.to(F64)
    P = torch.cholesky_inverse(torch.linalg.cholesky(Gd))
    P = 0.5 * (P + P.T)
    W = Wref.clone()
    Z = torch.zeros(dout, din, dtype=F64, device=Gd.device)
    A = torch.zeros(dout, din // g, dtype=F64, device=Gd.device)
    O = torch.zeros(dout, din // g, dtype=F64, device=Gd.device)
    for gi in range(din // g):
        t = gi * g
        WF = W[:, :t]
        Cc = C[:, t:t + g] - (WF @ Gd[:t, t:t + g] if t > 0 else 0)
        Cr = C[:, t + g:] - (WF @ Gd[:t, t + g:] if t > 0 else 0)
        if curvatura == "raw":
            S = Gd[t:t + g, t:t + g]
            T = Cc - W[:, t + g:] @ Gd[t + g:, t:t + g] if Cr.shape[1] else Cc
            Pcr = None
        else:
            Pcc, Pcr = P[:g, :g], P[:g, g:]
            S = torch.linalg.inv(Pcc)
            S = 0.5 * (S + S.T)
            K = -Pcr.T @ S                                      # = G_rr⁻¹ G_rc (Eq. 14)
            T = Cc - Cr @ K if Cr.shape[1] else Cc
        z, a, o = minmax_affine(Wref[:, t:t + g], bits)
        for _ in range(R):
            if not griglia_fissa:
                a, o = scala_zero_esatti(z, S, T, bits)
            z = spazzata_affine(z, a, o, S, T, bits)
            if traccia is not None:
                traccia.append(float(locale(a.unsqueeze(1) * (z - o.unsqueeze(1)), S, T).sum()))
        if not griglia_fissa:
            a, o = scala_zero_esatti(z, S, T, bits)
        W[:, t:t + g] = a.unsqueeze(1) * (z - o.unsqueeze(1))
        Z[:, t:t + g], A[:, gi], O[:, gi] = z, a, o
        if Pcr is not None:
            P = P[g:, g:] - Pcr.T @ S @ Pcr
            P = 0.5 * (P + P.T)
    return {"W": W, "Z": Z, "A": A, "O": O}


# ───────────────────────────────────── variante THOR: ternario a K piani ─────────────────────────────────────
def tabella_trit(K: int, device="cpu") -> torch.Tensor:
    """[3^K, K] tutte le combinazioni di trit {-1,0,1}."""
    import itertools
    return torch.tensor(list(itertools.product((-1.0, 0.0, 1.0), repeat=K)), dtype=F64, device=device)


def scale_ternarie(codes: torch.Tensor, S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Minimo esatto in a ∈ R^K di Σ_i ½ q_iSq_iᵀ − t_iq_iᵀ con q_i = Σ_k a_ik c_ik (codici fissi): M a = b per riga.
    Piani tutti nulli in una riga → scala 0 (righe singolari risolte nel sottospazio con pseudo-inversa)."""
    CS = torch.einsum("krj,ji->kri", codes, S)
    M = torch.einsum("kri,lri->rkl", CS, codes)
    b = torch.einsum("ri,kri->rk", T, codes)
    a = (torch.linalg.pinv(M, hermitian=True) @ b.unsqueeze(-1)).squeeze(-1)
    return a.T.contiguous()                                    # [K, R]


def avvio_ternario_minmax(Wc: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analogo ternario del min-max per riga (dichiarato): piani avidi sul residuo, scala del piano k = max|res|·2/3
    (soglia del ternario uniforme a 3 livelli su [−max, max]: livelli −a, 0, a con a = 2/3·max copre il range con
    arrotondamento al piu' vicino), codici = round(res/a) in {-1,0,1}."""
    R, g = Wc.shape
    codes = torch.zeros(K, R, g, dtype=F64, device=Wc.device)
    sc = torch.zeros(K, R, dtype=F64, device=Wc.device)
    res = Wc.clone()
    for k in range(K):
        a = (res.abs().max(1).values * (2.0 / 3.0)).clamp_min(1e-12)
        c = torch.clamp(torch.round(res / a.unsqueeze(1)), -1, 1)
        codes[k], sc[k] = c, a
        res = res - a.unsqueeze(1) * c
    return codes, sc


def spazzata_ternaria(codes: torch.Tensor, sc: torch.Tensor, S: torch.Tensor, T: torch.Tensor,
                      tab: torch.Tensor) -> torch.Tensor:
    """Eq. 17 col insieme dei candidati del formato a K piani: per ogni coordinata i 3^K valori Σ_k a_k t_k."""
    V = sc.T @ tab.T                                           # [R, 3^K]
    q = (codes * sc.unsqueeze(-1)).sum(0)
    Phi = q @ S
    codes = codes.clone()
    for j in range(q.shape[1]):
        Sjj = S[j, j]
        cross = Phi[:, j] - q[:, j] * Sjj - T[:, j]
        val = 0.5 * Sjj * V * V + cross.unsqueeze(1) * V
        k = val.argmin(1)
        vnew = V.gather(1, k.unsqueeze(1)).squeeze(1)
        dlt = vnew - q[:, j]
        if bool((dlt != 0).any()):
            Phi += dlt.unsqueeze(1) * S[j].unsqueeze(0)
            q[:, j] = vnew
        codes[:, :, j] = tab[k].T
    return codes


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(F64)


def schuropt_ternario(G: torch.Tensor, C: torch.Tensor, Wref: torch.Tensor, K: int, g: int = 128, R: int = 16,
                      damp: float = 0.01, T0: Optional[torch.Tensor] = None, S0: Optional[torch.Tensor] = None,
                      scale_fp16: bool = True, traccia: Optional[list] = None) -> Dict[str, torch.Tensor]:
    """Algoritmo 1 del lavoro con il formato di Thor. T0 [K, R, din] int8 / S0 [K, R, din/g]: avvio dall'ospite
    (variante THOR-ospite); se None, avvio min-max ternario per piani avidi (variante THOR-minmax).
    Le scale si arrotondano a fp16 (formato del file) dopo ogni stima: la spazzata vede i valori veri del file."""
    dout, din = Wref.shape
    assert din % g == 0
    G = G.to(F64)
    C = C.to(F64)
    Wref = Wref.to(F64)
    tab = tabella_trit(K, G.device)
    Gd = smorza(G, damp)
    P = torch.cholesky_inverse(torch.linalg.cholesky(Gd))
    P = 0.5 * (P + P.T)
    W = Wref.clone()
    Tout = torch.zeros(K, dout, din, dtype=torch.int8, device=G.device)
    Sout = torch.zeros(K, dout, din // g, dtype=F64, device=G.device)
    for gi in range(din // g):
        t = gi * g
        Pcc, Pcr = P[:g, :g], P[:g, g:]
        S = torch.linalg.inv(Pcc)
        S = 0.5 * (S + S.T)
        Km = -Pcr.T @ S
        WF = W[:, :t]
        Cc = C[:, t:t + g] - (WF @ Gd[:t, t:t + g] if t > 0 else 0)
        Cr = C[:, t + g:] - (WF @ Gd[:t, t + g:] if t > 0 else 0)
        Tl = Cc - Cr @ Km if Cr.shape[1] else Cc
        if T0 is not None:
            codes = T0[:, :, t:t + g].to(F64).clone()
            sc = S0[:, :, gi].to(F64).clone()
        else:
            codes, sc = avvio_ternario_minmax(Wref[:, t:t + g], K)
        if scale_fp16:
            sc = fp16(sc)
        for _ in range(R):
            sc = scale_ternarie(codes, S, Tl)
            if scale_fp16:
                sc = fp16(sc)
            codes = spazzata_ternaria(codes, sc, S, Tl, tab)
            if traccia is not None:
                traccia.append(float(locale((codes * sc.unsqueeze(-1)).sum(0), S, Tl).sum()))
        sc = scale_ternarie(codes, S, Tl)
        if scale_fp16:
            sc = fp16(sc)
        W[:, t:t + g] = (codes * sc.unsqueeze(-1)).sum(0)
        Tout[:, :, t:t + g] = codes.to(torch.int8)
        Sout[:, :, gi] = sc
        P = P[g:, g:] - Pcr.T @ S @ Pcr
        P = 0.5 * (P + P.T)
    return {"W": W, "T": Tout, "S": Sout}


def ricostruisci(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    K, R, C = T.shape
    ng = S.shape[-1]
    return sum((T[k].to(F64).reshape(R, ng, C // ng) * S[k].to(F64).unsqueeze(-1)).reshape(R, C) for k in range(K))
