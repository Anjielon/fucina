"""R022 — punteggio coi dKL MISURATI + vincolo "solo opzioni che il motore carica" in un colpo solo.

Prima di R022 `unisci.py --loadable-only --delta-kl` era rifiutato: il solutore misurato
(misurato.allocate_measured) lavorava per gruppo di misura e ignorava cosa il motore sa caricare;
quello caricabile (motore.solve_loadable) lavorava coi menu proxy. Qui le due cose si fondono
in UNITA' DI DECISIONE:

  unita' = componente connessa dei tensori legati fra loro da
      (a) un gruppo di misura (il dKL e' del gruppo: l'opzione si applica a tutti i membri);
      (b) solo col motore "oggi": lo stesso (strato, proiezione) di esperti (il 2° piano copre
          TUTTI gli esperti o nessuno; "-1"/"+1" valgono per l'intero tensore).
  Col motore "piani-variabili" il vincolo (b) cade: ogni gruppo di misura e' autonomo.

Opzioni di un'unita' = quelle misurate in OGNI gruppo dell'unita', presenti nei menu di ogni
membro e caricabili per ogni membro (motore.option_loadable: niente "+sNN", "+1" entro il
limite di piani del motore, "-1" solo con >= 2 piani). Il costo e' la somma dei byte esatti, il
punteggio la somma dei dKL misurati (linea di base "cur" sottratta, misure appaiate).
Un'unita' con tensori FUORI dalle misure resta ferma a "cur" (unmeasured="cur") oppure li
include col proxy calibrato (unmeasured="proxy", come misurato.py).
Un'unita' di esperti incompleta (< n_experts nel piano) col motore "oggi" si ferma a "cur".

Il solutore e' lo stesso di unisci (Lagrangiano + greedy + zaino esatto), seguito da un
riempimento greedy a byte VERI: lo zaino esatto arrotonda i costi per eccesso a una
granularita' (con ~10^4 unita' di esperti: ~100 KB) e lascerebbe sul tavolo parte del margine.

`se_k` > 0 penalizza ogni opzione != cur con k volte l'errore standard del dKL (in quadratura
fra i gruppi dell'unita'): contro la maledizione del vincitore quando le opzioni sono migliaia.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .allocazione import Allocation, _matrices, _totals, greedy_fill
from .menu import Option, TensorMenu
from .misurato import MeasuredResult, _option_index, risolvi_gruppi
from .motore import N_MAIN_DEFAULT, group_key, option_loadable


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _has(m: TensorMenu, name: str) -> bool:
    return any(o.name == name for o in m.options)


def allocate_measured_loadable(menus: Sequence[TensorMenu], deltas: Dict[str, Dict[str, float]], budget: int,
                               profilo: str = "oggi", n_main: int = N_MAIN_DEFAULT, n_experts: int = 256,
                               unmeasured: str = "cur", se: Optional[Dict[str, Dict[str, float]]] = None,
                               se_k: float = 0.0, scale_cur: str = "sorgente",
                               solver: Optional[Callable[[Sequence[TensorMenu], int], Allocation]] = None,
                               polish: bool = True) -> MeasuredResult:
    if unmeasured not in ("cur", "proxy"):
        raise ValueError(unmeasured)
    if scale_cur not in ("sorgente", "rifit"):
        raise ValueError(scale_cur)
    menus = list(menus)
    se = se or {}
    # Il dKL di un'opzione e' misurato rispetto al modello SORGENTE; la riga "cur" del file e' il guadagno del
    # solo rifit delle scale. Con scrivi_piano_gguf --scale-cur sorgente (default) i tensori fermi NON ricevono
    # quel guadagno: la linea di base e' 0. Con --scale-cur rifit lo ricevono: le opzioni si misurano contro "cur".
    def base_of_group(g: str) -> float:
        return float(deltas[g].get("cur", 0.0)) if scale_cur == "rifit" else 0.0

    n = len(menus)
    owner, members = risolvi_gruppi(menus, deltas)
    # ---- unita' di decisione
    uf = _UF(n)
    for idx in members.values():
        for i in idx[1:]:
            uf.union(idx[0], i)
    experts_of: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    if profilo == "oggi":
        for i, m in enumerate(menus):
            if m.key.expert >= 0:
                experts_of[group_key(m.key)].append(i)
        for idx in experts_of.values():
            for i in idx[1:]:
                uf.union(idx[0], i)
    comp: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    # ---- calibrazione del proxy (stessa definizione di misurato.py), solo se serve
    calib: List[float] = []
    if unmeasured == "proxy":
        for g, idx in members.items():
            for o, v in deltas[g].items():
                if o == "cur" or not all(_has(menus[i], o) for i in idx):
                    continue
                dp = sum(menus[i].get(o).distortion - menus[i].get("cur").distortion for i in idx)
                if abs(dp) > 1e-30 and math.isfinite(dp):
                    calib.append((float(v) - base_of_group(g)) / dp)
    c = float(np.median(calib)) if calib else None
    notes: List[str] = []
    dropped: Dict[str, int] = defaultdict(int)
    umenus: List[TensorMenu] = []
    uidx: List[List[int]] = []
    ugroups: List[List[str]] = []
    frozen = 0
    for root in sorted(comp):
        idx = comp[root]
        gs = sorted({owner[i] for i in idx if i in owner})
        free = [i for i in idx if i not in owner]
        cur_bytes = sum(menus[i].get("cur").bytes for i in idx)
        names: Optional[List[str]] = None
        if profilo == "oggi" and any(menus[i].key.expert >= 0 for i in idx):
            exp = [i for i in idx if menus[i].key.expert >= 0]
            per_group: Dict[Tuple[int, str], List[int]] = defaultdict(list)
            for i in exp:
                per_group[group_key(menus[i].key)].append(i)
            incompleto = any(len({menus[i].key.expert for i in v}) < n_experts or
                             len({menus[i].n_planes for i in v}) != 1 for v in per_group.values())
            if incompleto:
                names = []
                frozen += 1
        if names is None:
            usa_proxy = unmeasured == "proxy" and c is not None
            if gs:
                cand = set.intersection(*[set(deltas[g]) for g in gs]) - {"cur"}
                if free and not usa_proxy:
                    cand = set()                 # tensori non misurati nell'unita': fermi
            elif free and usa_proxy:
                cand = {o.name for o in menus[free[0]].options} - {"cur"}
            else:
                cand = set()
            names = []
            for o in sorted(cand, key=lambda x: (x != "+1", x != "-1", x)):
                if not all(_has(menus[i], o) for i in idx):
                    dropped[o] += 1
                    continue
                if not all(option_loadable(menus[i].key, menus[i].n_planes, o, profilo, n_main) for i in idx):
                    dropped[o] += 1
                    continue
                names.append(o)
        opts = [Option("cur", cur_bytes, 0.0)]
        for o in names:
            by = sum(menus[i].get(o).bytes for i in idx)
            d = 0.0
            var = 0.0
            for g in gs:
                d += float(deltas[g][o]) - base_of_group(g)
                var += float(se.get(g, {}).get(o, 0.0)) ** 2
            if free and c is not None:
                d += sum(c * (menus[i].get(o).distortion - menus[i].get("cur").distortion) for i in free)
            opts.append(Option(o, by, d + se_k * math.sqrt(var), 0, {"d_misurato": d}))
        umenus.append(TensorMenu(menus[idx[0]].key, menus[idx[0]].n_planes, menus[idx[0]].shape, opts))
        uidx.append(idx)
        ugroups.append(gs)
    if frozen:
        notes.append(f"{frozen} unita' di esperti incomplete ferme a 'cur' (motore {profilo})")
    if unmeasured == "proxy" and c is None:
        notes.append("nessuna calibrazione possibile: unita' non misurate ferme a 'cur'")
    from .unisci import solve
    solver = solver or (lambda ms, bud: solve(ms, bud))
    al = solver(umenus, budget)
    ch = np.array(al.choice, dtype=np.int64)
    if polish:
        d, b = _matrices(umenus)
        ch2 = greedy_fill(d, b, ch, budget)
        if _totals(d, b, ch2)[1] <= _totals(d, b, ch)[1]:
            ch = ch2
    # ---- espansione ai tensori originali
    choice = [0] * n
    gch: Dict[str, str] = {}
    gdb: Dict[str, int] = {}
    gdk: Dict[str, float] = {}
    for u, idx in enumerate(uidx):
        o = umenus[u].options[int(ch[u])]
        for i in idx:
            choice[i] = _option_index(menus[i], o.name)
        for g in ugroups[u]:
            gm = members[g]
            gch[g] = o.name
            gdb[g] = sum(menus[i].options[choice[i]].bytes - menus[i].get("cur").bytes for i in gm)
            gdk[g] = 0.0 if o.name == "cur" else float(deltas[g][o.name]) - base_of_group(g)
    tb = sum(menus[i].options[choice[i]].bytes for i in range(n))
    tdp = sum(menus[i].options[choice[i]].distortion for i in range(n))
    assert tb <= budget, (tb, budget)
    d_u, b_u = _matrices(umenus)
    pred = float(sum(umenus[u].options[int(ch[u])].extra.get("d_misurato", 0.0) for u in range(len(umenus))))
    per_tensor = Allocation(choice, al.lam, tb, tdp, budget, al.filled)
    res = MeasuredResult(per_tensor, gch, gdb, gdk, pred, c, unmeasured,
                         sum(len(i) for i, g in zip(uidx, ugroups) if not g), budget, notes)
    base_tot = sum(float(v.get("cur", 0.0)) for v in deltas.values())
    res.scale_cur = scale_cur                                 # type: ignore[attr-defined]
    res.cur_base_somma = base_tot                             # type: ignore[attr-defined]
    # previsione rispetto al modello SORGENTE: con rifit ogni gruppo porta il suo guadagno di base
    res.pred_vs_sorgente = pred + (base_tot if scale_cur == "rifit" else 0.0)   # type: ignore[attr-defined]
    res.units = len(umenus)                                   # type: ignore[attr-defined]
    res.dropped_options = dict(dropped)                       # type: ignore[attr-defined]
    res.frozen_units = frozen                                 # type: ignore[attr-defined]
    res.n_moved = int(sum(1 for u in range(len(umenus)) if int(ch[u]) != 0))   # type: ignore[attr-defined]
    return res
