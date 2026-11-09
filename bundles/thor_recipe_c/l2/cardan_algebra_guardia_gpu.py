"""Ingresso compatibile alla guardia comune, senza fattorizzazioni duplicate.

Un ingresso non finito interrompe l'unita'. Per sole matrici piccole, un
fattore GPU fallito/inaccurato viene ricalcolato CPU fp64, con avviso.
"""
from __future__ import annotations

def installa(CP):
    import torch
    import cardan_guardia_gpu as G
    G.installa(CP)
    if getattr(CP.jarq_lotto, "_guardia_finito", False):
        return G.contatori
    jarq_originale = CP.jarq_lotto

    def jarq(W, H, T, S, **kwargs):
        if not all(bool(torch.isfinite(x).all()) for x in (W, H, S)):
            raise FloatingPointError("JARQ: ingresso non finito; nessun file pubblicato")
        t, s, cat = jarq_originale(W, H, T, S, **kwargs)
        if not bool(torch.isfinite(s).all()) or not all(
                bool(torch.isfinite(torch.tensor(v, dtype=torch.float64)).all()) for v in cat.values()):
            raise FloatingPointError("JARQ: risultato non finito; nessun file pubblicato")
        return t, s, cat

    jarq._guardia_finito = True
    CP.jarq_lotto = jarq
    return G.contatori
