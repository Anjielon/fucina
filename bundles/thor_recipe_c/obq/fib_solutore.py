"""OBQ FIBRATO sulla metrica della KL4 — solutore puro (nessun I/O di modello).

LEVA ORIGINALE THOR (matematica fusa, 13/10/2026: ricerche/MATEMATICA_FUSA_1013/T1 Teorema 3-4 = T2 Lemma 1-2, verificata
in V_verifica_cpu.md §4). Deriva da: OBQ / Optimal Brain Quantization (Frantar & Alistarh, arXiv 2208.11580), GPTQ
(arXiv 2210.17323), K-FAC (arXiv 1503.05671), YAQA/FastKron (arXiv 2505.22988), GPTQ come algoritmo di Babai
(arXiv 2507.18553).

Parte FEDELE (OBQ/GPTQ): la compensazione colonna per colonna con il fattore di Cholesky superiore di H^-1 e lo
smorzamento 1 % della media della diagonale (IST-DASLab fasterquant), scale per gruppo FISSE.
Variante Thor DICHIARATA: la metrica non e' H (x) I (Hessiana dell'errore di ricostruzione, uscita isotropa) ma
    M = H (x) I + sum_i lam_i A_i (x) B_i ,
con A_i, B_i i fattori di Kronecker (K-FAC) della KL4 sulle risposte (A = E[x x^T] ingresso, B = E[g g^T] uscita,
Fisher EMPIRICA: Kunstner et al. 1905.12558, dichiarato). Teoremi usati (T1 Teorema 3/4):
  * con un solo compito (o B_i commutanti) M si spezza, nella base propria di B, nella somma diretta
    (+)_k (H + lam mu_k A): ogni direzione d'uscita k ha la sua Hessiana H_k = H + lam mu_k A;
  * la compensazione OBS lungo la direzione k usa chol(H_k^-1) (la stessa formula di GPTQ, Teorema 3);
  * la decisione discreta della colonna j e' un CVP sui trit {-1,0,+1} con Gram B_j = V diag(1/U_k[j,j]^2) V^T:
    risolta con discesa esatta a un trit per volta (costo esatto della variazione) partendo dall'arrotondamento.
Approssimazioni dichiarate (le stesse di V1): le prime NTOP = 8 direzioni di B hanno H_k propria; le altre R - 8
condividono una H media (bin di coda); con piu' compiti i coefficienti mu_ik si mediano nel bin.
lam = 0 => GPTQ a scale fisse (controllo esatto, stessa base, stesse scale).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

F64 = torch.float64
NTOP = 8


def psd(M: torch.Tensor) -> torch.Tensor:
    """Simmetrizza e, se serve, proietta sul cono PSD (fattori salvati in fp16 possono perdere la semidefinitezza),
    poi normalizza a media della diagonale = 1."""
    M = 0.5 * (M + M.T)
    ev, Q = torch.linalg.eigh(M)
    if float(ev[0]) < 0:
        M = (Q * ev.clamp_min(0)) @ Q.T
        M = 0.5 * (M + M.T)
        del Q
    d = float(torch.diagonal(M).mean())
    return M / (d if d > 0 else 1.0)


def rel(E: torch.Tensor, W: torch.Tensor, M: torch.Tensor) -> float:
    """errore relativo nella metrica d'ingresso M: sqrt(tr(E M E^T) / tr(W M W^T))."""
    return float(((E @ M) * E).sum() / ((W @ M) * W).sum()) ** 0.5


def relKF(E: torch.Tensor, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> float:
    """errore relativo nella metrica di Kronecker tr(B E A E^T) (la KL4 al secondo ordine, K-FAC)."""
    return float(((B @ E @ A) * E).sum() / ((B @ W @ A) * W).sum()) ** 0.5


def ricostruisci(T: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """T int8 [R, C], D scale [R, ng] -> W^ [R, C] f64."""
    R, C = T.shape
    G = C // D.shape[1]
    return D.to(F64).repeat_interleave(G, 1) * T.to(F64)


def _U_superiore(Hk: torch.Tensor, damp: float) -> torch.Tensor:
    C = Hk.shape[0]
    Hk = Hk + damp * torch.diagonal(Hk).mean() * torch.eye(C, dtype=F64)
    L = torch.linalg.cholesky(Hk)
    return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)


def obq_fibrato_rif(W: torch.Tensor, H: torch.Tensor, D: torch.Tensor,
                    metriche: Sequence[Tuple[float, torch.Tensor, torch.Tensor]] = (),
                    damp: float = 0.01, ntop: int = NTOP, T_init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """RIFERIMENTO (V1, al bit, nella base propria di B; O(R^2) per colonna). La corsa usa obq_fibrato (sotto).
    W [R, C] f64 (base memorizzata), H [C, C] (media diag 1), D scale fisse [R, ng] (valori fp16),
    metriche = [(lam, A [C,C], B [R,R])]. Ritorna T int8 [R, C] in {-1, 0, +1}.
    Con metriche vuote: GPTQ a scale fisse (B = I, una sola H)."""
    R, C = W.shape
    ng = D.shape[1]
    G = C // ng
    D = D.to(F64)
    metriche = [(float(l), A, B) for l, A, B in metriche if float(l) != 0.0]
    if not metriche:
        V = torch.eye(R, dtype=F64)
        bins: List[List[int]] = [list(range(R))]
        Hs = [H]
    else:
        Bmix = sum(l * B for l, _, B in metriche)
        _, V_ = torch.linalg.eigh(Bmix)
        del Bmix
        V = V_.flip(1).contiguous()
        del V_
        ntop = min(ntop, R - 1)
        coef = [torch.einsum("rk,rs,sk->k", V, B, V) for _, _, B in metriche]          # v_k^T B_i v_k
        bins = [[k] for k in range(ntop)] + [list(range(ntop, R))]
        Hs = []
        for ks in bins:
            Hk = H.clone()
            for (l, A, _), cf in zip(metriche, coef):
                Hk += l * float(cf[ks].clamp_min(0).mean()) * A
            Hs.append(Hk)
    U = [_U_superiore(Hk, damp) for Hk in Hs]
    del Hs
    nb = len(bins)
    kbin = torch.empty(R, dtype=torch.long)
    for b, ks in enumerate(bins):
        kbin[ks] = b
    Yt = V.T @ W                                                    # pesi compensati nella base propria di B
    T = torch.zeros(R, C, dtype=torch.int8)
    nt = min(ntop, R) if nb > 1 else 0
    Vn = V[:, :nt]
    code = [slice(ks[0], ks[-1] + 1) for ks in bins]                 # bin contigui: viste, niente copie
    for j in range(C):
        dj = torch.stack([U[b][j, j] for b in range(nb)])[kbin]     # [R]: diag di chol(H_k^-1) per direzione
        lamj = 1.0 / dj ** 2
        y = V @ Yt[:, j]
        Dj = D[:, j // G]
        if T_init is not None:
            t = T_init[:, j].to(F64).clone()
        else:
            t = torch.clamp(torch.round(y / Dj.clamp_min(1e-30)), -1, 1)
            t[Dj == 0] = 0
        if nb > 1:
            delta = float(lamj[nt:].mean())                        # B_j = delta I + Vn diag(ln - delta) Vn^T
            ln = lamj[:nt] - delta
            r = y - Dj * t
            z = Vn.T @ r
            Bii = delta + (Vn ** 2 * ln).sum(1)
            for _ in range(4 * nt + 16):                            # discesa esatta: un trit per volta
                Br = delta * r + Vn @ (ln * z)
                best = None
                for dt in (-1.0, 1.0):
                    tn = t + dt
                    df = -2 * dt * Dj * Br + (dt * Dj) ** 2 * Bii
                    df = torch.where((tn >= -1) & (tn <= 1) & (Dj > 0), df, torch.full_like(df, math.inf))
                    v, i = df.min(0)
                    if best is None or float(v) < best[0]:
                        best = (float(v), int(i), dt)
                if best[0] >= -1e-15:
                    break
                _, i, dt = best
                t[i] += dt
                r[i] -= dt * Dj[i]
                z -= dt * Dj[i] * Vn[i]
        T[:, j] = t.to(torch.int8)
        if j + 1 < C:
            et = (V.T @ (y - Dj * t)) / dj                            # errore OBS scalato, per direzione
            for b in range(nb):
                ks = code[b]
                Yt[ks, j + 1:] -= et[ks].unsqueeze(1) * U[b][j, j + 1:].unsqueeze(0)
    return T


def obq_fibrato(W: torch.Tensor, H: torch.Tensor, D: torch.Tensor,
                metriche: Sequence[Tuple[float, torch.Tensor, torch.Tensor]] = (),
                damp: float = 0.01, ntop: int = NTOP, blocco: int = 128) -> torch.Tensor:
    """Stessa matematica di obq_fibrato_rif, riscritta nella base ORIGINALE e a blocchi (lazy batch di GPTQ):
    la compensazione lungo le direzioni k del fibrato e' Y[:, j+1:] -= sum_k v_k a_k U_k[j, j+1:] + u_coda U_coda[j, j+1:],
    con a_k = v_k^T e / U_k[j,j] (prime ntop direzioni) e u_coda = (e - V_n V_n^T e) / U_coda[j,j] (la coda ha UNA H, quindi
    una sola diagonale: niente prodotto R x R per colonna). Costo per colonna O(R ntop + R blocco) invece di O(R^2).
    Coefficienti della coda: media di v^T B_i v sulle R - ntop direzioni = (tr B_i - sum_top v_k^T B_i v_k) / (R - ntop)
    (il riferimento tronca a zero ogni termine prima della media: differenza solo sugli autovalori negativi
    d'arrotondamento). Uguale al riferimento a meno dell'arrotondamento in virgola mobile (fib_test.py lo misura)."""
    R, C = W.shape
    G = C // D.shape[1]
    D = D.to(F64)
    metriche = [(float(l), A, B) for l, A, B in metriche if float(l) != 0.0]
    nt = min(ntop, R - 1) if metriche else 0
    if metriche:
        Bmix = sum(l * B for l, _, B in metriche)
        _, V_ = torch.linalg.eigh(Bmix)
        del Bmix
        Vn = V_[:, -nt:].flip(1).contiguous()
        del V_
        Hs = []
        cfs = []
        for l, A, B in metriche:
            ctop = torch.einsum("rk,rs,sk->k", Vn, B, Vn)
            ccoda = (float(torch.diagonal(B).sum()) - float(ctop.sum())) / (R - nt)
            cfs.append((l, A, ctop.clamp_min(0), max(ccoda, 0.0)))
        for k in range(nt + 1):
            Hk = H.clone()
            for l, A, ctop, ccoda in cfs:
                Hk += l * (float(ctop[k]) if k < nt else ccoda) * A
            Hs.append(Hk)
    else:
        Vn = torch.zeros(R, 0, dtype=F64)
        Hs = [H]
    U = [_U_superiore(Hk, damp) for Hk in Hs]                     # U[k], k < nt: direzioni; U[nt]: coda
    del Hs
    Utop = torch.stack(U[:nt]) if nt else torch.zeros(0, C, C, dtype=F64)
    Uc = U[-1]
    Y = W.clone()
    T = torch.zeros(R, C, dtype=torch.int8)
    for i1 in range(0, C, blocco):
        i2 = min(i1 + blocco, C)
        w = i2 - i1
        Yb = Y[:, i1:i2].clone()
        Atop = torch.zeros(nt, w, dtype=F64)
        Ecoda = torch.zeros(R, w, dtype=F64)
        for jj in range(w):
            j = i1 + jj
            y = Yb[:, jj]
            Dj = D[:, j // G]
            t = torch.clamp(torch.round(y / Dj.clamp_min(1e-30)), -1, 1)
            t[Dj == 0] = 0
            dc = float(Uc[j, j])
            if nt:
                dtop = Utop[:, j, j]
                delta = 1.0 / dc ** 2
                ln = 1.0 / dtop ** 2 - delta
                r = y - Dj * t
                z = Vn.T @ r
                Bii = delta + (Vn ** 2 * ln).sum(1)
                for _ in range(4 * nt + 16):                        # discesa esatta: un trit per volta
                    Br = delta * r + Vn @ (ln * z)
                    best = None
                    for dt in (-1.0, 1.0):
                        tn = t + dt
                        df = -2 * dt * Dj * Br + (dt * Dj) ** 2 * Bii
                        df = torch.where((tn >= -1) & (tn <= 1) & (Dj > 0), df, torch.full_like(df, math.inf))
                        v, i = df.min(0)
                        if best is None or float(v) < best[0]:
                            best = (float(v), int(i), dt)
                    if best[0] >= -1e-15:
                        break
                    _, i, dt = best
                    t[i] += dt
                    r[i] -= dt * Dj[i]
                    z -= dt * Dj[i] * Vn[i]
            T[:, j] = t.to(torch.int8)
            e = y - Dj * t
            if nt:
                c = Vn.T @ e
                a = c / dtop
                u = (e - Vn @ c) / dc
                Atop[:, jj] = a
            else:
                u = e / dc
            Ecoda[:, jj] = u
            if jj + 1 < w:
                Yb[:, jj + 1:] -= u.unsqueeze(1) * Uc[j, j + 1:i2].unsqueeze(0)
                if nt:
                    Yb[:, jj + 1:] -= Vn @ (a.unsqueeze(1) * Utop[:, j, j + 1:i2])
        if i2 < C:
            Y[:, i2:] -= Ecoda @ Uc[i1:i2, i2:]
            if nt:
                M = torch.einsum("kw,kwc->kc", Atop, Utop[:, i1:i2, i2:])
                Y[:, i2:] -= Vn @ M
    return T
