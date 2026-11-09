"""Punteggio coi dKL MISURATI per gruppo di esperti, al posto del proxy imatrix.

Letteratura: MoPEQ (2509.02512) e GEMQ (2605.23078) assegnano la precisione per
esperto/gruppo con una sensibilita' MISURATA (o stimata sul modello intero) e
risolvono uno zaino sotto il vincolo di memoria. Qui: punteggio = dKL misurato del
gruppo con l'opzione applicata (KL(teacher||studente) dopo - prima, sullo stesso
set di validazione), costo = byte esatti dai menu, zaino entro il budget.

File dei delta (TSV/CSV o JSON):
  TSV/CSV  : righe  gruppo <TAB|,> opzione <TAB|,> delta_kl   (# commento; intestazione
             "group option delta_kl" facoltativa; colonne in piu' ignorate)
  JSON     : [{"group": "L12.E37", "option": "+1", "delta_kl": -0.0031}, ...]
             oppure {"groups": {"L12.E37": {"+1": -0.0031, "+s25": -0.0012}, ...}}
Gruppo = uno o piu' selettori separati da virgola:
  L12.E37        tutti i tensori dell'esperto 37 dello strato 12 (gate, up, down)
  L12.E37.down   un solo tensore            L12.E0-31.down   esperti 0..31, solo down
  L12.D.attn_qkv un denso                   L12.E*.gate      tutti i gate dello strato
L'opzione si applica a OGNI tensore del gruppo (deve esistere nei loro menu); "cur"
vale 0 per definizione — se il file la riporta, il suo valore e' la linea di base e
viene sottratto alle altre opzioni del gruppo (misure appaiate).

Tensori fuori da ogni gruppo (unmeasured):
  "cur"   : restano 'cur' (sicuro: si spende solo dove si e' misurato);
  "proxy" : entrano col proxy, convertito in unita' di KL col fattore di calibrazione
            c = mediana_gruppi( dKL_misurato / sum(d_proxy(opz) - d_proxy(cur)) ).
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .allocazione import Allocation, allocate
from .formato import TensorKey
from .menu import Option, TensorMenu

_SEL = re.compile(r"^L(\d+)\.(D|E(\*|\d+(?:-\d+)?))(?:\.(\*|[A-Za-z0-9_]+))?$")


def parse_selector(sel: str):
    """-> predicato su TensorKey."""
    m = _SEL.match(sel.strip())
    if not m:
        raise ValueError(f"selettore di gruppo non valido: {sel!r}")
    layer = int(m.group(1))
    kind = m.group(4)
    if m.group(2) == "D":
        if kind in (None, "*"):
            raise ValueError(f"{sel!r}: per un denso serve il tipo (L12.D.attn_qkv)")
        return lambda k: k.layer == layer and k.expert < 0 and k.kind == kind
    ex = m.group(3)
    if ex == "*":
        lo, hi = 0, 1 << 30
    elif "-" in ex:
        lo, hi = map(int, ex.split("-"))
    else:
        lo = hi = int(ex)
    return lambda k: (k.layer == layer and lo <= k.expert <= hi and (kind in (None, "*") or k.kind == kind))


def load_deltas(path: str) -> Dict[str, Dict[str, float]]:
    """-> {gruppo: {opzione: dKL}}"""
    with open(path) as f:
        txt = f.read()
    out: Dict[str, Dict[str, float]] = {}
    s = txt.lstrip()
    if s.startswith("{") or s.startswith("["):
        data = json.loads(s)
        if isinstance(data, dict):
            for g, opts in data["groups"].items():
                out.setdefault(g, {}).update({o: float(v) for o, v in opts.items()})
        else:
            for r in data:
                out.setdefault(r["group"], {})[r["option"]] = float(r["delta_kl"])
        return out
    delim = "\t" if "\t" in txt else ","
    for row in csv.reader([ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")],
                          delimiter=delim):
        row = [c.strip() for c in row]
        if row[0].lower() in ("group", "gruppo"):
            continue
        if len(row) < 3:
            raise ValueError(f"{path}: riga con meno di 3 colonne: {row}")
        out.setdefault(row[0], {})[row[1]] = float(row[2])
    return out


def risolvi_gruppi(menus: Sequence[TensorMenu], deltas: Dict[str, Dict[str, float]]):
    """-> (proprietario per indice di menu, membri per gruppo). Ogni selettore fissa lo strato:
    si scandiscono solo i tensori di quello strato (con ~10^4 gruppi la scansione di tutti i menu
    per ogni gruppo costava minuti)."""
    by_layer: Dict[int, List[int]] = {}
    for n, m in enumerate(menus):
        by_layer.setdefault(m.key.layer, []).append(n)
    owner: Dict[int, str] = {}
    members: Dict[str, List[int]] = {}
    for g in deltas:
        sels = g.split(",")
        preds = [parse_selector(s) for s in sels]
        layers = {int(_SEL.match(s.strip()).group(1)) for s in sels}
        pool = sorted(n for L in layers for n in by_layer.get(L, ()))
        idx = [n for n in pool if any(p(menus[n].key) for p in preds)]
        if not idx:
            raise ValueError(f"gruppo {g!r}: nessun tensore nei piani")
        for n in idx:
            if n in owner:
                raise ValueError(f"{menus[n].key.name()} in due gruppi: {owner[n]!r} e {g!r}")
            owner[n] = g
        members[g] = idx
    return owner, members


@dataclass
class MeasuredResult:
    allocation: Allocation               # per tensore (menu originali): esportabile con export_plan
    group_choice: Dict[str, str]
    group_bytes_delta: Dict[str, int]
    group_dkl: Dict[str, float]
    pred_dkl_total: float
    calibration: Optional[float]
    unmeasured: str
    n_unmeasured: int
    budget: int
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict:
        return {"score_mode": "measured_dkl", "unmeasured": self.unmeasured,
                "pred_dkl_total": self.pred_dkl_total, "calibration_dkl_per_proxy": self.calibration,
                "n_unmeasured_tensors": self.n_unmeasured,
                "groups": [{"group": g, "option": self.group_choice[g], "delta_bytes": self.group_bytes_delta[g],
                            "delta_kl": self.group_dkl[g]} for g in sorted(self.group_choice)],
                "notes": self.notes}


def _option_index(m: TensorMenu, name: str) -> int:
    for j, o in enumerate(m.options):
        if o.name == name:
            return j
    raise KeyError(f"{m.key.name()}: opzione {name!r} assente dal menu")


def allocate_measured(menus: Sequence[TensorMenu], deltas: Dict[str, Dict[str, float]], budget: int,
                      unmeasured: str = "cur", exact: bool = True,
                      exact_unit: Optional[int] = None) -> MeasuredResult:
    if unmeasured not in ("cur", "proxy"):
        raise ValueError(unmeasured)
    menus = list(menus)
    owner, members = risolvi_gruppi(menus, deltas)
    notes: List[str] = []
    # ---- menu di gruppo: byte = somma esatta, punteggio = dKL misurato (cur = 0)
    gmenus: List[TensorMenu] = []
    gnames: List[str] = []
    gopts: Dict[str, List[str]] = {}
    calib: List[float] = []
    for g, idx in members.items():
        meas = dict(deltas[g])
        base = meas.pop("cur", 0.0)
        if base:
            notes.append(f"{g}: linea di base 'cur'={base:g} sottratta")
        opts = [Option("cur", sum(menus[n].get("cur").bytes for n in idx), 0.0)]
        names = ["cur"]
        for oname, dkl in meas.items():
            jj = [_option_index(menus[n], oname) for n in idx]
            by = sum(menus[n].options[j].bytes for n, j in zip(idx, jj))
            opts.append(Option(oname, by, float(dkl) - base))
            names.append(oname)
            dproxy = sum(menus[n].options[j].distortion - menus[n].get("cur").distortion for n, j in zip(idx, jj))
            if abs(dproxy) > 1e-30 and math.isfinite(dproxy):
                calib.append((float(dkl) - base) / dproxy)
        gmenus.append(TensorMenu(menus[idx[0]].key, 0, (), opts))
        gnames.append(g)
        gopts[g] = names
    c = float(np.median(calib)) if calib else None
    # ---- tensori non misurati
    free = [n for n in range(len(menus)) if n not in owner]
    others: List[TensorMenu] = []
    for n in free:
        m = menus[n]
        cur = m.get("cur")
        if unmeasured == "cur" or c is None:
            others.append(TensorMenu(m.key, m.n_planes, m.shape, [Option("cur", cur.bytes, 0.0)]))
        else:
            others.append(TensorMenu(m.key, m.n_planes, m.shape,
                                     [Option(o.name, o.bytes, c * (o.distortion - cur.distortion)) for o in m.options]))
    if unmeasured == "proxy" and c is None:
        notes.append("nessuna calibrazione possibile: non misurati fermi a 'cur'")
    allm = gmenus + others
    al = allocate(allm, budget)
    if exact:
        from .zaino import knapsack_exact
        ex = knapsack_exact(allm, budget, exact_unit)
        if ex is not None and ex.total_distortion < al.total_distortion - 1e-15 * max(abs(al.total_distortion), 1e-30):
            ex.lam = al.lam
            al = ex
    # ---- espansione a scelta per tensore sui menu ORIGINALI
    choice = [0] * len(menus)
    gch, gdb, gdk = {}, {}, {}
    for gi, g in enumerate(gnames):
        oname = gopts[g][al.choice[gi]]
        gch[g] = oname
        gdb[g] = gmenus[gi].options[al.choice[gi]].bytes - gmenus[gi].get("cur").bytes
        gdk[g] = gmenus[gi].options[al.choice[gi]].distortion
        for n in members[g]:
            choice[n] = _option_index(menus[n], oname)
    for k, n in enumerate(free):
        oname = others[k].options[al.choice[len(gnames) + k]].name
        choice[n] = _option_index(menus[n], oname)
    tb = sum(menus[n].options[choice[n]].bytes for n in range(len(menus)))
    tdp = sum(menus[n].options[choice[n]].distortion for n in range(len(menus)))
    assert tb == al.total_bytes and tb <= budget, (tb, al.total_bytes, budget)
    per_tensor = Allocation(choice, al.lam, tb, tdp, budget, al.filled)
    return MeasuredResult(per_tensor, gch, gdb, gdk, float(al.total_distortion), c, unmeasured, len(free),
                          budget, notes)


def load_deltas_se(path: str) -> Dict[str, Dict[str, float]]:
    """-> {gruppo: {opzione: errore standard}} dalla 4a colonna di un TSV/CSV (come scrive dkl.py)
    o dal campo "se" di un JSON a righe. Dove manca: assente dal dizionario (se = 0)."""
    with open(path) as f:
        txt = f.read()
    out: Dict[str, Dict[str, float]] = {}
    s = txt.lstrip()
    if s.startswith("{") or s.startswith("["):
        data = json.loads(s)
        if isinstance(data, list):
            for r in data:
                if "se" in r:
                    out.setdefault(r["group"], {})[r["option"]] = float(r["se"])
        return out
    delim = "\t" if "\t" in txt else ","
    for row in csv.reader([ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")],
                          delimiter=delim):
        row = [c.strip() for c in row]
        if len(row) < 4 or row[0].lower() in ("group", "gruppo"):
            continue
        try:
            out.setdefault(row[0], {})[row[1]] = abs(float(row[3]))
        except ValueError:
            continue
    return out
