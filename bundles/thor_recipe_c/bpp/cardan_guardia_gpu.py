#!/usr/bin/env python3
"""Guardia D1/D4: nessun rocSOLVER device; fallback esplicito LAPACK fp64.

Si installa solo nel nuovo processo, mai nei processi della corsa viva.
Conserva il dtype dei fattori del percorso storico: non promette identita'
al bit del JARQ fp32 rispetto al riferimento fp64.
"""
from __future__ import annotations

import contextlib
import contextvars
import importlib
import json
import os
import sys
import time

_contesto = contextvars.ContextVar("cardan_guardia", default={})
contatori = {"matrici_cpu": 0}


@contextlib.contextmanager
def contesto(**dati):
    token = _contesto.set({**_contesto.get(), **dati})
    try:
        yield
    finally:
        _contesto.reset(token)


def evento(tipo, **dati):
    print(json.dumps({"guardia": tipo, "ora": time.time(), **_contesto.get(), **dati},
                     allow_nan=False), file=sys.stderr, flush=True)


def verifica_finito(x, nome):
    import torch
    mask = torch.isfinite(x)
    if not bool(mask.all()):
        posizione = (~mask).nonzero()[0].tolist()
        evento("NON_FINITO", tensore=nome, indice=posizione)
        raise ValueError(f"{nome}: valore non finito a {posizione}; input non riparato silenziosamente")


def diagonale_cpu_stesso_passo(H):
    """Il passo C+1 sceglie la stessa riduzione ATen del riferimento CPU."""
    import torch
    d = torch.diagonal(H)
    cpu = torch.empty_strided(d.shape, d.stride(), dtype=d.dtype, device="cpu")
    cpu.copy_(d)
    return cpu


def normalizza(H):
    import torch
    verifica_finito(H, "H_grezza")
    H = 0.5 * (H + H.T)
    d = diagonale_cpu_stesso_passo(H)
    if float(d.max()) <= 0:
        evento("H_SENZA_DIAGONALE_POSITIVA", ripiego="H_ffn del riferimento")
        return None
    morti = d <= 0
    if bool(morti.any()):
        m = morti.to(H.device)
        H[m, m] = d[~morti].mean().to(H.device)
        d = diagonale_cpu_stesso_passo(H)
    H = H / d.mean().to(H.device)
    verifica_finito(H, "H_normalizzata")
    return H


def fattorizza(M, gpu_factor=None):
    """Riprova su CPU solo le matrici rifiutate/non finite dal fattore GPU.

    Un input non finito e' un errore, non una matrice cui aggiungere ridge.
    Una matrice realmente non-SPD mantiene info != 0, per il damping JARQ.
    gpu_factor e' iniettabile per prove CPU senza aprire un device.
    """
    import torch
    verifica_finito(M, "M_prima_chol")
    M64 = M.double()
    if M.device.type == "cpu" and gpu_factor is None:
        L, info = torch.linalg.cholesky_ex(M64)
        return L.to(M.dtype), info
    if gpu_factor is None:
        sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/ROCSOLVER_RIPARA_1020")
        gpu_factor = importlib.import_module("algebra_gpu").chol_ex
    n = M.shape[-1]
    flat = M64.reshape(-1, n, n)
    fattori, esiti = [], []
    limite = int(os.environ.get("PIEGA_TRSM_LOTTO", "8192"))
    if limite < 1:
        raise ValueError("PIEGA_TRSM_LOTTO deve essere positivo")
    for a in range(0, len(flat), limite):
        Mb = flat[a:a + limite]
        L, info = gpu_factor(Mb)
        male = (info != 0) | ~torch.isfinite(L).all(dim=(-2, -1))
        if n <= 256:
            simmetrica = torch.tril(Mb) + torch.tril(Mb, diagonal=-1).transpose(-1, -2)
            residuo = (simmetrica - L @ L.transpose(-1, -2)).abs().amax(dim=(-2, -1))
            scala = simmetrica.abs().amax(dim=(-2, -1)).clamp_min(1e-300)
            male |= ~torch.isfinite(residuo) | (residuo > 64 * n * torch.finfo(M64.dtype).eps * scala)
        if bool(male.any()):
            if n > 256:
                evento("CHOLESKY_FALLITA_GRANDE", ordine=n)
                raise RuntimeError(f"fallback CPU vietato per matrice grande {n}x{n}")
            indici = (male.nonzero().flatten() + a).tolist()
            contatori["matrici_cpu"] += len(indici)
            evento("CHOLESKY_RIPROVA_CPU", indici_lotto=indici,
                   pivot_gpu=info[male].tolist(), dtype=str(M.dtype))
            Lc, ic = torch.linalg.cholesky_ex(Mb[male].detach().to("cpu", torch.float64))
            evento("CHOLESKY_ESITO_CPU", indici_lotto=indici, pivot_cpu=ic.tolist())
            L[male], info[male] = Lc.to(L.device), ic.to(info.device)
        validi = info == 0
        verifica_finito(L[validi], "L_dopo_chol")
        fattori.append(L)
        esiti.append(info)
    return (torch.cat(fattori).reshape(M.shape).to(M.dtype),
            torch.cat(esiti).reshape(M.shape[:-2]))


def chol(M):
    import torch
    L, info = fattorizza(M)
    if bool((info != 0).any()):
        i = (info.reshape(-1) != 0).nonzero()[0].item()
        evento("CHOLESKY_NON_SPD", lotto=i, pivot=int(info.reshape(-1)[i]))
        raise torch.linalg.LinAlgError(
            f"chol: matrice ricevuta non-SPD (lotto {i}, pivot {int(info.reshape(-1)[i])}); "
            "integrita' rispetto alla sorgente host non verificata dalla fattorizzazione")
    return L


def risolvi(M, B):
    import torch
    verifica_finito(M, "M_prima_solve")
    verifica_finito(B, "B_prima_solve")
    if M.device.type == "cpu":
        return torch.linalg.solve(M.double(), B.double()).to(B.dtype)
    L, info = fattorizza(M.double())
    X = torch.empty_like(B)
    buoni = info == 0
    if bool(buoni.any()):
        X[buoni] = risolvi_chol(B[buoni], L[buoni])
    if bool((~buoni).any()):
        evento("SOLVE_LU_CPU", matrici=int((~buoni).sum()))
        X[~buoni] = torch.linalg.solve(
            M[~buoni].detach().to("cpu", torch.float64),
            B[~buoni].detach().to("cpu", torch.float64)).to(B.device, B.dtype)
    verifica_finito(X, "X_dopo_solve")
    return X


def risolvi_chol(B, L):
    import torch
    verifica_finito(B, "B_prima_trsm")
    verifica_finito(L, "L_prima_trsm")
    lead = torch.broadcast_shapes(B.shape[:-2], L.shape[:-2])
    Bb = B.double().expand(*lead, *B.shape[-2:]).reshape(-1, *B.shape[-2:])
    Lb = L.double().expand(*lead, *L.shape[-2:]).reshape(-1, *L.shape[-2:])
    X = torch.empty_like(Bb)
    limite = int(os.environ.get("PIEGA_TRSM_LOTTO", "8192"))
    if limite < 1:
        raise ValueError("PIEGA_TRSM_LOTTO deve essere positivo")
    for a in range(0, len(Bb), limite):
        b, l = Bb[a:a + limite], Lb[a:a + limite]
        y = torch.linalg.solve_triangular(l, b, upper=False)
        X[a:a + limite] = torch.linalg.solve_triangular(l.transpose(-1, -2), y, upper=True)
    verifica_finito(X, "X_dopo_trsm")
    # Nessun falso raffinamento del sistema M con un fattore gia' arrotondato.
    return X.reshape(*lead, *B.shape[-2:]).to(B.dtype)


def installa(CP=None):
    if CP is None:
        import cardan_piega_gpu as CP
    CP.chol_ex, CP.chol = fattorizza, chol
    CP.risolvi, CP.risolvi_chol = risolvi, risolvi_chol
    CP._normalizza_gpu = normalizza
    return CP
