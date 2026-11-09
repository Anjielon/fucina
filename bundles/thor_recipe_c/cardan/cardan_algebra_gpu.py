"""Algebra fp64 senza rocSOLVER; attivazione esplicita con PIEGA_GPU=1.

Le sole primitive linalg sulla GPU sono solve_triangular (rocBLAS TRSM).
Le fattorizzazioni fallite tornano sulla CPU per singola matrice del lotto.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import time
import warnings

import torch


class AlgebraGPU:
    def __init__(self, blocco=32, pausa=0.0):
        if blocco < 1 or pausa < 0:
            raise ValueError("Blocco positivo e pausa non negativa richiesti")
        self.blocco, self.pausa = blocco, pausa
        self.ripieghi = {}

    def _avvisa(self, operazione, quanti):
        self.ripieghi[operazione] = self.ripieghi.get(operazione, 0) + quanti
        warnings.warn(f"PIEGA GPU: {operazione}, ripiego CPU fp64 per {quanti} matrici",
                      RuntimeWarning, stacklevel=2)

    def riposa(self, device):
        if self.pausa and device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
            time.sleep(self.pausa)

    def chol_ex(self, M):
        if M.device.type == "cpu":
            return torch.linalg.cholesky_ex(M)
        if M.dtype != torch.float64:
            raise TypeError("La variante GPU richiede matrici fp64")
        return self._chol_blocchi(M)

    def _chol_blocchi(self, M):
        """Nucleo provabile anche su CPU, senza chiamare una fattorizzazione nativa."""
        n = M.shape[-1]
        A = M.reshape(-1, n, n)
        L = torch.zeros_like(A)
        info = torch.zeros(A.shape[0], dtype=torch.int32, device=A.device)
        for a in range(0, n, self.blocco):
            b = min(a + self.blocco, n)
            D = A[:, a:b, a:b] - L[:, a:b, :a] @ L[:, a:b, :a].transpose(-1, -2)
            tile = torch.zeros_like(D)
            # Pannello diagonale: Cholesky scalare vettorizzata sul lotto.
            for j in range(b - a):
                pivot = D[:, j, j] - (tile[:, j, :j] ** 2).sum(-1)
                valido = (pivot > 0) & torch.isfinite(pivot)
                info = torch.where((info == 0) & ~valido, a + j + 1, info)
                tile[:, j, j] = torch.sqrt(torch.where(valido, pivot, torch.ones_like(pivot)))
                if j + 1 < b - a:
                    v = D[:, j + 1:, j] - (
                        tile[:, j + 1:, :j] * tile[:, j:j + 1, :j]).sum(-1)
                    tile[:, j + 1:, j] = v / tile[:, j:j + 1, j]
            L[:, a:b, a:b] = tile
            if b < n:
                E = A[:, b:, a:b] - L[:, b:, :a] @ L[:, a:b, :a].transpose(-1, -2)
                L[:, b:, a:b] = torch.linalg.solve_triangular(
                    tile, E.transpose(-1, -2), upper=False).transpose(-1, -2)
            self.riposa(M.device)
        bad = (info != 0) | ~torch.isfinite(L).all(dim=(-1, -2))
        if bool(bad.any()):
            self._avvisa("cholesky", int(bad.sum()))
            lc, ic = torch.linalg.cholesky_ex(A[bad].detach().cpu())
            # Anche LAPACK puo' restituire info=0 con NaN nell'ingresso.
            ic = torch.where(~torch.isfinite(lc).all(dim=(-1, -2)) & (ic == 0), n + 1, ic)
            L[bad], info[bad] = lc.to(M.device), ic.to(M.device)
        # Il chiamante usa info per lo smorzamento; non eseguire TRSM su
        # fattori parziali/singolari prima che scale_lotto applichi il ridge.
        falliti = info != 0
        if bool(falliti.any()):
            L[falliti] = torch.eye(n, dtype=L.dtype, device=L.device)
        return L.reshape(M.shape), info.reshape(M.shape[:-2])

    def chol(self, M):
        L, info = self.chol_ex(M)
        if bool((info != 0).any()):
            raise torch.linalg.LinAlgError(
                f"Cholesky: matrice non SPD anche su CPU (info={info.cpu().tolist()})")
        return L

    def risolvi_chol(self, B, L):
        if L.device.type == "cpu":
            return torch.cholesky_solve(B, L)
        return self._risolvi_triangolari(B, L)

    def _risolvi_triangolari(self, B, L):
        y = torch.linalg.solve_triangular(L, B, upper=False)
        x = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)
        bad = ~torch.isfinite(x).all(dim=(-1, -2))
        if bool(bad.any()):
            self._avvisa("sistema triangolare", int(bad.sum()))
            n, k = B.shape[-2:]
            xf = x.reshape(-1, n, k)
            mask = bad.reshape(-1)
            lc = L.expand(*x.shape[:-2], n, n).reshape(-1, n, n)[mask].detach().cpu()
            bc = B.expand_as(x).reshape(-1, n, k)[mask].detach().cpu()
            xc = torch.cholesky_solve(bc, lc)
            if not bool(torch.isfinite(xc).all()):
                raise FloatingPointError("Sistema Cholesky non finito anche su CPU")
            xf[mask] = xc.to(x.device)
        self.riposa(L.device)
        return x

    def risolvi(self, M, B):
        if M.device.type == "cpu":
            return torch.linalg.solve(M, B)
        L, info = self.chol_ex(M)
        n, k = B.shape[-2:]
        A, rhs = M.reshape(-1, n, n), B.reshape(-1, n, k)
        good = info.reshape(-1) == 0
        x = torch.empty_like(rhs)
        if bool(good.any()):
            x[good] = self.risolvi_chol(rhs[good], L.reshape(-1, n, n)[good])
        if bool((~good).any()):
            self._avvisa("sistema non SPD", int((~good).sum()))
            x[~good] = torch.linalg.solve(A[~good].detach().cpu(),
                                        rhs[~good].detach().cpu()).to(M.device)
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("Sistema non finito anche dopo ripiego CPU")
        return x.reshape(B.shape)


@contextmanager
def percorso(CP, abilitato=None, algebra=None):
    """Sostituzioni solo nel processo chiamante; nessun file originale modificato."""
    if abilitato is None:
        abilitato = os.environ.get("PIEGA_GPU", "0") == "1"
    if algebra is None:
        algebra = AlgebraGPU(int(os.environ.get("PIEGA_GPU_BLOCCO", "32")),
                             float(os.environ.get("PIEGA_GPU_PAUSA", "0")))
    nomi = ("chol_ex", "chol", "risolvi_chol", "risolvi")
    prima = {n: getattr(CP, n) for n in nomi}
    cpu = CP.ALGEBRA_CPU
    CP.ALGEBRA_CPU = True
    try:
        if abilitato:
            for n in nomi:
                setattr(CP, n, getattr(algebra, n))
        yield algebra
    finally:
        for n, f in prima.items():
            setattr(CP, n, f)
        CP.ALGEBRA_CPU = cpu


def jarq_protetto(CP, W, H, T, S, **kwargs):
    """Un risultato non finito non puo' diventare un file di uscita valido."""
    if not all(bool(torch.isfinite(t).all()) for t in (W, H, S)):
        raise FloatingPointError("Ingresso JARQ non finito: impossibile ripiego affidabile")
    risultato = CP.jarq_lotto(W, H, T, S, **kwargs)
    if (bool(torch.isfinite(risultato[1]).all())
            and all(torch.isfinite(torch.tensor(v, dtype=torch.float64)).item()
                    for v in risultato[2].values())):
        return risultato
    warnings.warn("PIEGA GPU: JARQ non finito, ripiego CPU fp64 per il blocco di righe",
                  RuntimeWarning, stacklevel=2)
    with percorso(CP, False):
        tc, sc, cat = CP.jarq_lotto(W.cpu(), H.cpu(), T.cpu(), S.cpu(), **kwargs)
    if not bool(torch.isfinite(sc).all()) or not all(
            torch.isfinite(torch.tensor(v, dtype=torch.float64)).item() for v in cat.values()):
        raise FloatingPointError("JARQ non finito anche su CPU")
    return tc.to(T.device), sc.to(S.device), cat
