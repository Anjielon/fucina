"""Margine di file per il GGUF scritto da scrivi_piano_gguf (R022).

I byte dei tensori del piano sono esatti; il file in piu' ha le voci nuove della tabella dei
tensori (nome + forma + tipo + offset), le mappe `*_exps2_map` (motore "piani-variabili") e
l'allineamento. `overhead_upper` ripete il tetto di `scrivi_piano_gguf.estimate_size` (stessa
formula) ma sulle SCELTE di un'allocazione, cosi' unisci puo' spendere il margine fino in fondo:
prima toglieva 1 MiB fisso, ora toglie il tetto vero (~0,1 MiB) e verifica a ciclo chiuso.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Tuple

from .allocazione import Allocation
from .menu import TensorMenu
from .motore import base_of


def overhead_upper(menus: Sequence[TensorMenu], choice: Sequence[int], profilo: str, n_exp: int = 256,
                   alignment: int = 32) -> int:
    k2: Dict[Tuple[int, str], int] = defaultdict(int)
    planes: Dict[Tuple[int, str], int] = {}
    for m, c in zip(menus, choice):
        name = m.options[c].name
        k = m.n_planes + {"+1": 1, "-1": -1, "+1g64": 1}.get(name, 0)
        if m.key.expert >= 0:
            k2[(m.key.layer, base_of(m.key))] += int(k >= 2)
        else:
            planes[(m.key.layer, m.key.kind)] = k - m.n_planes
    maps = sum(1 for v in k2.values() if 0 < v < n_exp) if profilo in ("piani-variabili", "scala-g64") else 0
    new_tensors = sum(max(0, v) for v in planes.values()) + sum(1 for v in k2.values() if v) + maps
    return maps * (4 * n_exp + alignment) + new_tensors * (128 + alignment) + 4096


def risolvi_nel_file(solve_fn: Callable[[int], Tuple[Allocation, object]], menus: Sequence[TensorMenu],
                     tetto_senza_overhead: int, profilo: str, n_exp: int, alignment: int,
                     max_giri: int = 6):
    """solve_fn(budget_tensori) -> (Allocation, extra). Cerca il budget piu' alto tale che
    budget + overhead(soluzione) <= tetto_senza_overhead. -> (alloc, extra, info)."""
    overhead = overhead_upper(menus, [0] * len(menus), profilo, n_exp, alignment) + 1024
    best = None
    storia = []
    for _ in range(max_giri):
        bud = tetto_senza_overhead - overhead
        al, extra = solve_fn(bud)
        ov = overhead_upper(menus, al.choice, profilo, n_exp, alignment)
        storia.append({"budget": bud, "total_bytes": al.total_bytes, "overhead_assunto": overhead,
                       "overhead_vero_max": ov})
        if ov <= overhead:
            if best is None or al.total_bytes >= best[0].total_bytes:
                best = (al, extra, ov)
            if overhead - ov <= 4096:
                break
            overhead = ov + 1024          # riprova con piu' budget; se sfora si tiene la precedente
        else:
            overhead = ov + 1024
    if best is None:         # non converso: tetto peggiore possibile (tutti i "+1" dove esistono)
        worst = [next((j for j, o in enumerate(m.options) if o.name == "+1"), 0) for m in menus]
        overhead = overhead_upper(menus, worst, profilo, n_exp, alignment)
        al, extra = solve_fn(tetto_senza_overhead - overhead)
        best = (al, extra, overhead)
        storia.append({"budget": tetto_senza_overhead - overhead, "total_bytes": al.total_bytes,
                       "overhead_assunto": overhead, "overhead_vero_max": "caso peggiore"})
    al, extra, ov = best
    return al, extra, {"overhead_max": ov, "giri": storia}
