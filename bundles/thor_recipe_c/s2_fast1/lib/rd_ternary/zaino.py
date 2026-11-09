"""Zaino a scelta multipla ESATTO (programmazione dinamica) sul margine di byte.

Il Lagrangiano trova i punti dell'inviluppo convesso; fra due punti il riempimento
greedy chiude il gap ma non garantisce l'ottimo. Quando i menu sono pochi (gruppi
di esperti col dKL misurato) o il margine e' piccolo, la DP e' esatta a meno della
granularita': i costi in byte sopra il minimo di ogni menu sono arrotondati PER
ECCESSO a multipli di `unit`, quindi la soluzione trovata sta sempre nel budget vero.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .allocazione import Allocation, _matrices, _totals


def knapsack_exact(menus: Sequence, budget: int, unit: Optional[int] = None,
                   max_cells: float = 5e7) -> Optional[Allocation]:
    """min sum d  s.v.  sum b <= budget.
    unit=None: granularita' automatica, la piu' fine con tabella N x (margine/unit) <= max_cells
    (esatta al byte quando il margine e' piccolo). unit fissato: None se la tabella eccede."""
    d, b = _matrices(menus)
    n, q = d.shape
    fin = np.isfinite(d)
    big = np.iinfo(np.int64).max // 4
    bmin = np.where(fin, b, big).min(1)
    slack = int(budget) - int(bmin.sum())
    if slack < 0:
        raise ValueError(f"budget {budget} sotto il minimo ottenibile {int(bmin.sum())}")
    if unit is None:
        unit = max(1, -(-n * (slack + 1) // int(max_cells)))
    cap = slack // unit
    if n * (cap + 1) > max_cells:
        return None
    cost = np.where(fin, -(-(b - bmin[:, None]) // unit), big)          # ceil, per eccesso
    dp = np.zeros(cap + 1)                                              # "<= c": monotona in c
    choice = np.zeros((n, cap + 1), dtype=np.int16)
    for i in range(n):
        new = np.full(cap + 1, np.inf)
        arg = np.zeros(cap + 1, dtype=np.int16)
        for j in range(q):
            c = int(cost[i, j]) if fin[i, j] else big
            if c > cap:
                continue
            cand = np.full(cap + 1, np.inf)
            cand[c:] = dp[: cap + 1 - c] + d[i, j]
            better = cand < new
            new[better] = cand[better]
            arg[better] = j
        dp, choice[i] = new, arg
    ch = np.zeros(n, dtype=np.int64)
    c = cap
    for i in range(n - 1, -1, -1):
        j = int(choice[i, c])
        ch[i] = j
        c -= int(cost[i, j])
    tb, td = _totals(d, b, ch)
    assert tb <= budget
    return Allocation(list(map(int, ch)), float("nan"), tb, td, int(budget), False)
