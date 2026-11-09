"""Verifica a posteriori dell'algebra GPU della piega esperti (D1, D4), senza rocSOLVER e senza cambiare i valori corretti.

Si installa SOPRA `cardan_piega_gpu` con PIEGA_ALGEBRA=gpu_sicura (lo fa `cardan_piega_gpu_atomica.py`):
- chol_ex / chol: ogni fattore GPU e' controllato con Freivalds in fp64 (k vettori casuali fissi: ||M v - L (L^T v)|| <=
  tol * ||M||_F ||v||, costo O(n^2) per matrice). Le matrici con info != 0, fattore non finito o residuo fuori soglia sono
  RIFATTE sulla CPU con LAPACK fp64 (`torch.linalg.cholesky_ex`, il calcolo dell'originale): un falso «non definita
  positiva» della GPU (D1, tappa 100 L16, pivot 1 su Hgg SPD) non applica piu' il damping ne' fa cadere il lavoratore; un
  fattore finito ma sbagliato non passa in silenzio. Le matrici buone restano quelle della GPU, AL BIT.
- risolvi (sistemi generali M x = B, solo le righe gia' fallite): residuo relativo controllato, fuori soglia -> CPU fp64.
- trasferimento delle Hessiane (sorgente di D1 col sospetto SDMA): somma int64 esatta (modulo 2^64, indipendente
  dall'ordine) della matrice sul device == quella dell'array host; diversa -> ricopia una volta, poi errore esplicito.
- normalizzazione (D4): media della diagonale sulla CPU con lo STESSO passo C+1 della vista dell'originale
  (`cardan_guardia_gpu.diagonale_cpu_stesso_passo`): H normalizzata identica al bit a `cardan_piega._normalizza`.
Ogni intervento scrive una riga JSON `{"verifica": ...}` su stderr; i contatori finali sono in CONTATORI.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

CONTATORI = {"chol_matrici": 0, "chol_cpu": 0, "chol_falso_non_spd": 0, "chol_fattore_errato": 0,
             "risolvi_cpu": 0, "h_trasferimenti": 0, "h_ricopiati": 0}
TOLL = float(os.environ.get("FORGIA_FREIVALDS_TOLL", "1e-12"))
K = int(os.environ.get("FORGIA_FREIVALDS_K", "4"))
_V = {}


def evento(tipo, **d):
    print(json.dumps({"verifica": tipo, "ora": round(time.time(), 3), **d}), file=sys.stderr, flush=True)


def _vettori(n, dev):
    k = (n, str(dev))
    if k not in _V:
        g = torch.Generator().manual_seed(20261021 + n)
        _V[k] = torch.randn(n, K, generator=g, dtype=torch.float64).to(dev)
    return _V[k]


def errore_freivalds(M64, L64):
    """Residuo relativo ||M V - L (L^T V)||_F / (||M||_F ||V||_F) per matrice, usando solo il triangolo inferiore di M
    (quello che legge la Cholesky: LAPACK e algebra_gpu ignorano il superiore)."""
    n = M64.shape[-1]
    V = _vettori(n, M64.device)
    S = torch.tril(M64) + torch.tril(M64, diagonal=-1).transpose(-1, -2)
    r = S @ V - L64 @ (L64.transpose(-1, -2) @ V)
    den = torch.linalg.matrix_norm(S) * torch.linalg.vector_norm(V)
    return torch.linalg.matrix_norm(r) / den.clamp_min(1e-300)


def controlla_fattori(M, L, info, cpu_ex=torch.linalg.cholesky_ex):
    """(M, L, info) GPU -> (L, info) con le sole matrici sospette rifatte da LAPACK fp64 su CPU."""
    lead = M.shape[:-2]
    n = M.shape[-1]
    M64 = M.double().reshape(-1, n, n)
    Lf = L.reshape(-1, n, n)
    inf = info.reshape(-1)
    CONTATORI["chol_matrici"] += M64.shape[0]
    ok = inf == 0
    finito = torch.isfinite(Lf).all(dim=(-2, -1))
    err = torch.full((M64.shape[0],), float("inf"), dtype=torch.float64, device=M64.device)
    if bool((ok & finito).any()):
        idx = (ok & finito).nonzero().flatten()
        err[idx] = errore_freivalds(M64[idx], Lf[idx].double())
    # L restituito nel dtype di M (fp32 in produzione, LEVE_DT): il solo arrotondamento di L a fp32 da' un residuo relativo
    # fino a ~sqrt(n)*eps32; la soglia e' quindi max(TOLL, 8*sqrt(n)*eps(dtype di L)). Un errore sotto soglia e' dello
    # stesso ordine dell'arrotondamento che il percorso usa comunque.
    soglia = max(TOLL, 8 * n ** 0.5 * torch.finfo(L.dtype).eps)
    male = ~ok | ~finito | ~(err <= soglia)
    if bool(male.any()):
        b = male.nonzero().flatten()
        Lc, ic = cpu_ex(M64[b].detach().to("cpu"))
        falso = (inf[b] != 0).cpu() & (ic == 0)
        sbagliato = (inf[b] == 0).cpu()
        CONTATORI["chol_cpu"] += b.numel()
        CONTATORI["chol_falso_non_spd"] += int(falso.sum())
        CONTATORI["chol_fattore_errato"] += int(sbagliato.sum())
        evento("CHOL_RIFATTA_CPU", matrici=b.numel(), su=int(M64.shape[0]), ordine=n,
               info_gpu=inf[b].tolist()[:16], info_cpu=ic.tolist()[:16],
               errore_gpu=[float(x) for x in err[b].tolist()[:16]], falsi_non_spd=int(falso.sum()), soglia=soglia,
               fattori_errati=int(sbagliato.sum()))
        Lf = Lf.clone()
        inf = inf.clone()
        Lf[b] = Lc.to(Lf.device, Lf.dtype)
        inf[b] = ic.to(inf.device, inf.dtype)
    return Lf.reshape(*lead, n, n), inf.reshape(lead)


def controlla_soluzione(M, B, X):
    """Solo per le poche righe in `risolvi` (gia' fallite in chol_ex): residuo relativo, fuori soglia -> CPU fp64."""
    M64, B64, X64 = M.double(), B.double(), X.double()
    r = torch.linalg.matrix_norm(M64 @ X64 - B64)
    den = torch.linalg.matrix_norm(M64) * torch.linalg.matrix_norm(X64) + torch.linalg.matrix_norm(B64)
    male = ~torch.isfinite(r) | ~(r <= 1e-8 * den.clamp_min(1e-300))
    if bool(male.any()):
        CONTATORI["risolvi_cpu"] += int(male.sum())
        evento("RISOLVI_RIFATTO_CPU", matrici=int(male.sum()), su=int(male.numel()))
        X = X.clone()
        X[male] = torch.linalg.solve(M64[male].cpu(), B64[male].cpu()).to(X.device, X.dtype)
    return X


def somma_bit(x):
    """Somma esatta modulo 2^64 dei bit (vista int64): indipendente dall'ordine di riduzione e dal device."""
    if isinstance(x, np.ndarray):
        return int(np.ascontiguousarray(x, np.float64).reshape(-1).view(np.int64).sum(dtype=np.int64))
    return int(x.contiguous().reshape(-1).view(torch.int64).sum())


def a_device_verificato(g, dev):
    """np.float64 host -> tensore sul device, con controllo esatto del trasferimento (una ricopia, poi errore)."""
    attesa = somma_bit(g)
    CONTATORI["h_trasferimenti"] += 1
    for tentativo in range(2):
        H = torch.from_numpy(g).to(dev)
        if somma_bit(H) == attesa:
            return H
        CONTATORI["h_ricopiati"] += 1
        evento("H_TRASFERIMENTO_DIVERSO", tentativo=tentativo, forma=list(g.shape))
    raise RuntimeError("Hessiana corrotta nel trasferimento verso il device per due volte: lavoratore fermato")


def installa(P, normalizza=True, algebra=True, trasferimento=True):
    """Installa le verifiche in un modulo cardan_piega_gpu gia' importato. Restituisce l'elenco delle sostituzioni."""
    fatte = []
    if normalizza:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cardan_guardia_gpu import diagonale_cpu_stesso_passo

        def _normalizza_gpu(H):
            H = 0.5 * (H + H.T)
            dg = diagonale_cpu_stesso_passo(H)
            if float(dg.max()) <= 0:
                return None
            morti = dg <= 0
            if morti.any():
                m = morti.to(H.device)
                H[m, m] = dg[~morti].mean().to(H.device)
                dg = diagonale_cpu_stesso_passo(H)
            return H / dg.mean().to(H.device)
        P._normalizza_gpu = _normalizza_gpu
        fatte.append("normalizza_stesso_passo")
    if trasferimento:
        H0 = P.Hessiane

        def di_gpu(self, e, dev, grezza=None):
            if self.gq is not None:
                g = grezza if grezza is not None else self.grezza(e)
                H = P._normalizza_gpu(a_device_verificato(g, dev))
                if H is not None:
                    return H, "Gq"
            return self.hs.to(dev), "H_ffn"
        H0.di_gpu = di_gpu
        fatte.append("trasferimento_H")
    if algebra and os.environ.get("PIEGA_ALGEBRA") == "gpu_sicura":
        risolvi0 = P.risolvi
        AG = P._AG

        def chol_ex(M):
            # = chol_ex di gpu_sicura (stessi pezzi da 8192, stesso _AG.chol_ex fp64), ma il controllo e' fatto sul
            # fattore fp64 PRIMA della conversione al dtype di M: soglia stretta (TOLL) invece dell'arrotondamento fp32.
            M64 = M.double(); lead = M64.shape[:-2]; Mb = M64.reshape(-1, *M64.shape[-2:])
            Ls, Is = [], []
            for i in range(0, Mb.shape[0], 8192):
                L, info = AG.chol_ex(Mb[i:i + 8192]); Ls.append(L); Is.append(info)
            L, info = controlla_fattori(Mb, torch.cat(Ls), torch.cat(Is))
            return L.reshape(*lead, *M64.shape[-2:]).to(M.dtype), info.reshape(lead)

        def chol(M):
            L, info = chol_ex(M)
            if bool((info != 0).any()):
                i = int((info.reshape(-1) != 0).nonzero()[0])
                raise torch.linalg.LinAlgError(f"chol: matrice non SPD anche per LAPACK fp64 CPU (lotto {i})")
            return L

        def risolvi(M, B):
            return controlla_soluzione(M, B, risolvi0(M, B))
        P.chol_ex, P.chol, P.risolvi = chol_ex, chol, risolvi
        fatte.append("freivalds_chol")
    return fatte
