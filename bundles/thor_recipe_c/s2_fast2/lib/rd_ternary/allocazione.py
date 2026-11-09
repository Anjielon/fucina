"""Allocazione rate-distortion: knapsack a scelta multipla risolto col
Lagrangiano (bisezione su lambda) + riempimento greedy del budget residuo,
verifica di additivita' su coppie casuali e termini incrociati opzionali per layer.

Problema:  min sum_i d_i(q_i)   s.v.  sum_i b_i(q_i) <= B
Lagrangiano: per lambda fissato ogni tensore sceglie argmin_q d_i(q) + lambda b_i(q)
(scelta separabile, O(N*Q)). Le soluzioni lagrangiane sono OTTIME per il proprio
budget (punti dell'inviluppo convesso); il riempimento chiude il gap residuo.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .formato import TensorKey
from .menu import TensorMenu


@dataclass
class Allocation:
    choice: List[int]            # indice dell'opzione scelta per ogni menu
    lam: float
    total_bytes: int
    total_distortion: float
    budget: int
    filled: bool = False
    history: List[Tuple[float, int, float]] = field(default_factory=list)


def _matrices(menus: Sequence[TensorMenu]):
    q = max(len(m.options) for m in menus)
    d = np.full((len(menus), q), np.inf)
    b = np.full((len(menus), q), np.iinfo(np.int64).max // 4, dtype=np.int64)
    for n, m in enumerate(menus):
        for j, o in enumerate(m.options):
            d[n, j], b[n, j] = o.distortion, o.bytes
    return d, b


def lagrangian_choice(d: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """argmin d + lam*b, a parita' vince il MENO costoso (rende la scelta monotona in lambda)."""
    cost = d + lam * b.astype(np.float64)
    best = cost.min(1, keepdims=True)
    tie = np.isclose(cost, best, rtol=1e-12, atol=0.0) | (cost == best)
    bb = np.where(tie, b, np.iinfo(np.int64).max)
    return bb.argmin(1)


def _totals(d, b, ch):
    r = np.arange(len(ch))
    return int(b[r, ch].sum()), float(d[r, ch].sum())


def greedy_fill(d: np.ndarray, b: np.ndarray, ch: np.ndarray, budget: int) -> np.ndarray:
    """Usa il budget residuo: applica ripetutamente la mossa col miglior rapporto
    (riduzione di d)/(byte in piu') fra quelle che stanno nel budget.
    Ogni voce dell'heap porta la versione del tensore: le voci stantie si scartano."""
    ch = ch.copy()
    left = budget - _totals(d, b, ch)[0]
    n, q = d.shape
    ver = np.zeros(n, dtype=np.int64)

    def best_move(i, cap):
        cur_d, cur_b = d[i, ch[i]], b[i, ch[i]]
        best = None
        for j in range(q):
            if not np.isfinite(d[i, j]) or d[i, j] >= cur_d:
                continue
            db = int(b[i, j] - cur_b)
            if db > cap:
                continue
            r = math.inf if db <= 0 else (cur_d - d[i, j]) / db
            if best is None or r > best[0]:
                best = (r, j, db)
        return best

    heap = []
    for i in range(n):
        mv = best_move(i, left)
        if mv:
            heapq.heappush(heap, (-mv[0], i, int(ver[i]), mv[1], mv[2]))
    while heap:
        _, i, v, j, db = heapq.heappop(heap)
        if v != ver[i]:
            continue
        if db <= left:
            ch[i] = j
            left -= db
            ver[i] += 1
        mv = best_move(i, left)  # nuova mossa (o una piu' piccola se questa non ci stava)
        if mv:
            heapq.heappush(heap, (-mv[0], i, int(ver[i]), mv[1], mv[2]))
    return ch


def allocate(menus: Sequence[TensorMenu], budget: int, iters: int = 100, fill: bool = True) -> Allocation:
    d, b = _matrices(menus)
    ch_min = lagrangian_choice(d, b, 1e30)            # minimo dei byte
    if _totals(d, b, ch_min)[0] > budget:
        raise ValueError(f"budget {budget} sotto il minimo ottenibile {_totals(d, b, ch_min)[0]}")
    ch0 = lagrangian_choice(d, b, 0.0)
    if _totals(d, b, ch0)[0] <= budget:
        tb, td = _totals(d, b, ch0)
        return Allocation(list(map(int, ch0)), 0.0, tb, td, budget)
    lo, hi = 0.0, 1.0
    while _totals(d, b, lagrangian_choice(d, b, hi))[0] > budget:
        hi *= 4.0
    hist = []
    for _ in range(iters):  # bisezione geometrica (lambda copre molti ordini di grandezza)
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e3
        tbm, tdm = _totals(d, b, lagrangian_choice(d, b, mid))
        hist.append((mid, tbm, tdm))
        if tbm > budget:
            lo = mid
        else:
            hi = mid
        if lo > 0 and hi / lo < 1 + 1e-9:
            break
    ch = lagrangian_choice(d, b, hi)
    filled = False
    if fill:
        ch2 = greedy_fill(d, b, ch, budget)
        if _totals(d, b, ch2)[1] < _totals(d, b, ch)[1]:
            ch, filled = ch2, True
    tb, td = _totals(d, b, ch)
    return Allocation(list(map(int, ch)), hi, tb, td, budget, filled, hist)


# --------------------------------------------------------------------------- regimi accoppiati (g64)
G64_OPTIONS = ("g64", "+1g64")


def is_g64(name: str) -> bool:
    return name in G64_OPTIONS


def allocate_regimi(menus: Sequence[TensorMenu], budget: int, coupled: Sequence[Sequence[int]],
                    iters: int = 100, fill: bool = True) -> Allocation:
    """Knapsack a scelta multipla con un vincolo di REGIME per gruppo (6/10, scala 64).

    Le opzioni g64 cambiano il tipo GGML dell'intero tensore: per gli esperti di uno stesso
    (strato, proiezione) o TUTTI stanno in regime g64 (opzioni g64 / +1g64) o NESSUNO.
    `coupled` = gruppi di indici di menu legati (un tensore denso e' un gruppo da solo; i
    menu fuori da ogni gruppo sono liberi). Per lambda fissato si sceglie, per ogni gruppo,
    il regime col costo lagrangiano totale minore e, dentro il regime, l'opzione migliore
    di ogni membro (resta separabile fra gruppi); poi bisezione su lambda e riempimento
    greedy con le mosse ristrette al regime scelto per ciascun gruppo."""
    d, b = _matrices(menus)
    n, q = d.shape
    g64 = np.zeros((n, q), dtype=bool)
    for i, m in enumerate(menus):
        for j, o in enumerate(m.options):
            g64[i, j] = is_g64(o.name)
    grp = np.full(n, -1, dtype=np.int64)
    for gi, idx in enumerate(coupled):
        grp[list(idx)] = gi
    free = np.where(grp < 0)[0]
    ng = len(coupled)
    big = np.iinfo(np.int64).max // 4

    def choose(lam: float):
        cost = d + lam * b.astype(np.float64)
        c0 = np.where(g64, np.inf, cost)          # regime normale
        c1 = np.where(g64, cost, np.inf)          # regime g64
        m0, m1 = c0.min(1), c1.min(1)
        reg = np.zeros(n, dtype=bool)
        if ng:
            s0 = np.zeros(ng)
            s1 = np.zeros(ng)
            np.add.at(s0, grp[grp >= 0], m0[grp >= 0])
            np.add.at(s1, grp[grp >= 0], m1[grp >= 0])
            gsel = s1 < s0                          # a parita' (o inf) resta il regime normale
            reg[grp >= 0] = gsel[grp[grp >= 0]]
        if free.size:
            reg[free] = m1[free] < m0[free]
        dd = np.where(g64 == reg[:, None], d, np.inf)
        bb = np.where(g64 == reg[:, None], b, big)
        return lagrangian_choice(dd, bb, lam), reg

    ch_min, _ = choose(1e30)
    if _totals(d, b, ch_min)[0] > budget:
        raise ValueError(f"budget {budget} sotto il minimo ottenibile {_totals(d, b, ch_min)[0]}")
    ch, reg = choose(0.0)
    lam = 0.0
    hist: List[Tuple[float, int, float]] = []
    if _totals(d, b, ch)[0] > budget:
        lo, hi = 0.0, 1.0
        while _totals(d, b, choose(hi)[0])[0] > budget:
            hi *= 4.0
        for _ in range(iters):
            mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e3
            tbm, tdm = _totals(d, b, choose(mid)[0])
            hist.append((mid, tbm, tdm))
            if tbm > budget:
                lo = mid
            else:
                hi = mid
            if lo > 0 and hi / lo < 1 + 1e-9:
                break
        ch, reg = choose(hi)
        lam = hi
    filled = False
    if fill:
        dd = np.where(g64 == reg[:, None], d, np.inf)
        bb = np.where(g64 == reg[:, None], b, big)
        ch2 = greedy_fill(dd, bb, ch, budget)
        if _totals(d, b, ch2)[1] < _totals(d, b, ch)[1]:
            ch, filled = ch2, True
    tb, td = _totals(d, b, ch)
    return Allocation(list(map(int, ch)), lam, tb, td, budget, filled, hist)


# --------------------------------------------------------------------------- additivita'
@dataclass
class PairReport:
    i: int
    j: int
    kl_i: float
    kl_j: float
    kl_ij: float
    second_diff: float          # kl_ij - kl_i - kl_j (0 se additivo)
    rel_nonadd: float           # |second_diff| / (|kl_i| + |kl_j|)
    pred_i: float               # variazione prevista dal menu (d(q) - d(cur))
    pred_j: float
    rho: Optional[float]        # correlazione stimata (modello 2 rho sqrt(d_i d_j))
    same_layer: bool


def additivity_check(menus: Sequence[TensorMenu], alloc: Allocation,
                     eval_delta_kl: Callable[[Dict[TensorKey, str]], float],
                     n_pairs: int = 32, same_layer_frac: float = 0.5, seed: int = 0) -> Dict:
    """Perturba due tensori insieme vs separati.

    eval_delta_kl(assegnazione) deve applicare le opzioni indicate (gli altri tensori
    restano 'cur') e restituire KL(modello perturbato) - KL(modello cur), misurata
    sul set di validazione rispetto al teacher. Ogni singolo viene valutato una volta.
    """
    rng = random.Random(seed)
    moved = [n for n, m in enumerate(menus) if m.options[alloc.choice[n]].name != "cur"]
    if len(moved) < 2:
        return {"pairs": [], "median_rel_nonadd": 0.0, "calibration_median": None}
    by_layer: Dict[int, List[int]] = {}
    for n in moved:
        by_layer.setdefault(menus[n].key.layer, []).append(n)
    pairs = set()
    tries = 0
    while len(pairs) < n_pairs and tries < 50 * n_pairs:
        tries += 1
        if rng.random() < same_layer_frac:
            lay = [v for v in by_layer.values() if len(v) >= 2]
            if not lay:
                continue
            i, j = rng.sample(rng.choice(lay), 2)
        else:
            i, j = rng.sample(moved, 2)
        pairs.add((min(i, j), max(i, j)))
    cache: Dict[int, float] = {}

    def single(n):
        if n not in cache:
            cache[n] = eval_delta_kl({menus[n].key: menus[n].options[alloc.choice[n]].name})
        return cache[n]

    reps = []
    for i, j in sorted(pairs):
        oi, oj = menus[i].options[alloc.choice[i]], menus[j].options[alloc.choice[j]]
        ci, cj = menus[i].get("cur"), menus[j].get("cur")
        a, bb = single(i), single(j)
        ab = eval_delta_kl({menus[i].key: oi.name, menus[j].key: oj.name})
        sd = ab - a - bb
        den = 2 * (math.sqrt(max(oi.distortion, 0)) - math.sqrt(max(ci.distortion, 0))) * \
            (math.sqrt(max(oj.distortion, 0)) - math.sqrt(max(cj.distortion, 0)))
        rho = sd / den if abs(den) > 1e-30 else None
        reps.append(PairReport(i, j, a, bb, ab, sd, abs(sd) / max(abs(a) + abs(bb), 1e-30),
                               oi.distortion - ci.distortion, oj.distortion - cj.distortion,
                               rho, menus[i].key.layer == menus[j].key.layer))
    cal = [cache[n] / (menus[n].options[alloc.choice[n]].distortion - menus[n].get("cur").distortion)
           for n in cache if abs(menus[n].options[alloc.choice[n]].distortion - menus[n].get("cur").distortion) > 1e-30]
    return {"pairs": reps,
            "median_rel_nonadd": float(np.median([r.rel_nonadd for r in reps])) if reps else 0.0,
            "calibration_median": float(np.median(cal)) if cal else None}


def rho_per_layer(report: Dict, menus: Sequence[TensorMenu], clip: float = 1.0) -> Dict[int, float]:
    """rho condiviso per layer = mediana delle coppie stesso-layer (robusta)."""
    acc: Dict[int, List[float]] = {}
    for r in report["pairs"]:
        if r.same_layer and r.rho is not None:
            acc.setdefault(menus[r.i].key.layer, []).append(r.rho)
    return {lay: float(np.clip(np.median(v), -clip, clip)) for lay, v in acc.items()}


def allocate_with_cross(menus: Sequence[TensorMenu], budget: int, rho_layer: Dict[int, float],
                        sweeps: int = 4, iters: int = 60) -> Allocation:
    """Obiettivo con termini incrociati per layer (tutte le coppie dello stesso layer):
        sum_i d_i + sum_L rho_L sum_{i<j in L} 2 sqrt(d_i d_j) + lambda sum_i b_i
    Ottimizzato con ICM (Gauss-Seidel sulle scelte) partendo dalla soluzione
    separabile; lambda per bisezione. Contributo di i: 2 rho_L sqrt(d_i(q)) (S_L - u_i)."""
    d, b = _matrices(menus)
    sq = np.sqrt(np.clip(d, 0, None))
    layer = np.array([m.key.layer for m in menus])
    rho = np.array([rho_layer.get(int(l), 0.0) for l in layer])

    def solve(lam):
        ch = lagrangian_choice(d, b, lam)
        u = sq[np.arange(len(ch)), ch]
        S = {l: u[layer == l].sum() for l in np.unique(layer)}
        for _ in range(sweeps):
            changed = 0
            for i in range(len(menus)):
                l = layer[i]
                others = S[l] - u[i]
                cost = d[i] + lam * b[i] + 2 * rho[i] * sq[i] * others
                j = int(np.nanargmin(np.where(np.isfinite(cost), cost, np.inf)))
                if j != ch[i]:
                    S[l] += sq[i, j] - u[i]
                    u[i], ch[i] = sq[i, j], j
                    changed += 1
            if not changed:
                break
        return ch

    def total_obj(ch):
        tb, td = _totals(d, b, ch)
        u = sq[np.arange(len(ch)), ch]
        cross = 0.0
        for l in np.unique(layer):
            ul = u[layer == l]
            r = rho_layer.get(int(l), 0.0)
            cross += r * (ul.sum() ** 2 - (ul ** 2).sum())
        return tb, td + cross

    lo, hi = 0.0, 1.0
    if total_obj(solve(0.0))[0] <= budget:
        ch = solve(0.0)
        tb, td = total_obj(ch)
        return Allocation(list(map(int, ch)), 0.0, tb, td, budget)
    while total_obj(solve(hi))[0] > budget:
        hi *= 4.0
        if hi > 1e300:
            raise ValueError("budget non raggiungibile")
    for _ in range(iters):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6
        if total_obj(solve(mid))[0] > budget:
            lo = mid
        else:
            hi = mid
        if lo > 0 and hi / lo < 1 + 1e-6:
            break
    ch = solve(hi)
    tb, td = total_obj(ch)
    return Allocation(list(map(int, ch)), hi, tb, td, budget)
