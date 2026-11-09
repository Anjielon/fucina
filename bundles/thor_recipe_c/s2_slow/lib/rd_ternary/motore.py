"""Cosa il MOTORE sa caricare: vincolo "--loadable-only" per l'allocatore e controlli
per scrivi_piano_gguf.

Fonte: thor-llama, src/models/qwen35moe.cpp (load_arch_tensors) + llama-graph.cpp
(build_moe_ffn). Due profili:

  "oggi"            ramo velocita-ternario (f4b7c6f) e esperti-recupero
  "piani-variabili" ramo cloud/piani-variabili (= oggi + mappa esperto->slot del 2° piano)

Cosa si carica (tutti i piani di un tensore si SOMMANO sullo stesso ingresso, l'ordine
non conta; i nomi devono essere contigui base, base2, base3, ... per il lettore):

  densi negli strati principali (0..n_main-1)
      attn_q, attn_k, attn_v, attn_output, attn_qkv, attn_gate, ssm_out,
      ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp             fino a 4 piani
  output (testa)                                               fino a 4 piani
  token_embd                                                   fino a 3 piani
  esperti ffn_{gate,up,down}_exps (strati principali)          fino a 2 piani
      "oggi":            il 2° piano copre TUTTI gli esperti dello strato/proiezione
                         (o un prefisso globale expert_count2, inutile senza permutazione)
      "piani-variabili": il 2° piano copre un SOTTOINSIEME QUALSIASI (tensore *_exps2_map)
  strato MTP (blk.n_main, nextn.*) e qualunque altro tensore    1 piano (nessun piano in piu')
  piani SPARSI (+sNN)                                           MAI (il motore non ha il formato)

Un "-1" e' sempre caricabile dove ci sono >= 2 piani (si rinumera). Per gli esperti in
"oggi" un -1 o un +1 vale solo per l'INTERO tensore (tutti gli esperti insieme).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .allocazione import Allocation, _matrices, _totals
from .formato import TensorKey
from .menu import Option, TensorMenu

PROFILI = ("oggi", "piani-variabili", "scala-g64")
# "scala-g64" (6/10) = piani-variabili + tipi TQ1_0_G64 / TQ1_0R_1024_G64 (ramo thor-llama
# cloud/scala-g64): opzioni "g64" e "+1g64"; per gli esperti il tipo e' dell'INTERO tensore
# (strato, proiezione): o tutti gli esperti in g64 o nessuno (allocazione.allocate_regimi).
MAPPE = ("piani-variabili", "scala-g64")     # profili con la mappa *_exps2_map
N_MAIN_DEFAULT = 40

DENSE_MAX4 = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out",
              "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
EXPERT_BASES = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
BASE_KIND = {v: k for k, v in EXPERT_BASES.items()}


def base_of(key: TensorKey) -> str:
    return EXPERT_BASES.get(key.kind, key.kind) if key.expert >= 0 else key.kind


def max_planes(layer: int, base: str, n_main: int = N_MAIN_DEFAULT) -> int:
    """Numero massimo di piani che il motore carica per (strato, nome base GGUF)."""
    if layer < 0:
        return {"output": 4, "token_embd": 3}.get(base, 1)
    if layer >= n_main:              # strato MTP: nessun piano in piu' nel loader
        return 1
    if base in DENSE_MAX4:
        return 4
    if base in BASE_KIND:
        return 2
    return 1


def option_loadable(key: TensorKey, n_planes: int, opt: str, profilo: str = "oggi",
                    n_main: int = N_MAIN_DEFAULT) -> bool:
    """L'opzione e' caricabile PER IL SINGOLO tensore/esperto? (per gli esperti in "oggi"
    serve inoltre l'uniformita' sul gruppo: vedi constrain_menus)."""
    if profilo not in PROFILI:
        raise ValueError(f"profilo motore {profilo!r}: atteso uno di {PROFILI}")
    if opt == "cur":
        return True
    if opt.startswith("+s"):
        return False
    lim = max_planes(key.layer, base_of(key), n_main)
    if opt == "+1":
        return n_planes + 1 <= lim
    if opt == "-1":
        return n_planes >= 2
    if opt in ("g64", "+1g64"):
        if profilo != "scala-g64" or lim < 1 or (key.layer >= n_main):
            return False      # strato MTP e motori senza i tipi a scala 64: no
        return n_planes + (1 if opt == "+1g64" else 0) <= lim
    return False


def group_key(key: TensorKey) -> Tuple[int, str]:
    return (key.layer, base_of(key))


@dataclass
class Constrained:
    menus: List[TensorMenu]                    # menu da dare al solutore
    members: List[List[int]]                   # indici dei menu originali coperti da ciascuno
    dropped_options: Dict[str, int]            # opzione -> quante volte tolta
    groups: int                                # gruppi di esperti aggregati
    frozen_groups: int                         # gruppi incompleti fermati a "cur"


def constrain_menus(menus: Sequence[TensorMenu], profilo: str = "oggi", n_main: int = N_MAIN_DEFAULT,
                    n_experts: int = 256) -> Constrained:
    """Toglie dai menu le opzioni che il motore non carica. In "oggi" gli esperti di uno
    stesso (strato, proiezione) diventano UN menu aggregato (byte e distorsioni sommati:
    il problema resta separabile), con le sole opzioni comuni e caricabili per tutti.
    Un gruppo con meno di n_experts esperti nel piano (es. --max-experts 16) si ferma a
    "cur": non si puo' cambiare solo una parte del tensore."""
    dropped: Dict[str, int] = defaultdict(int)
    out: List[TensorMenu] = []
    members: List[List[int]] = []
    groups: Dict[Tuple[int, str], List[int]] = defaultdict(list)

    def filt(m: TensorMenu) -> List[Option]:
        keep = []
        for o in m.options:
            if option_loadable(m.key, m.n_planes, o.name, profilo, n_main):
                keep.append(o)
            else:
                dropped[o.name] += 1
        return keep

    for n, m in enumerate(menus):
        if profilo == "oggi" and m.key.expert >= 0:
            groups[group_key(m.key)].append(n)
            continue
        out.append(TensorMenu(m.key, m.n_planes, m.shape, filt(m)))
        members.append([n])
    frozen = 0
    for (layer, base), idx in sorted(groups.items()):
        ms = [menus[i] for i in idx]
        experts = {m.key.expert for m in ms}
        names = None
        for m in ms:
            ok = {o.name for o in filt(m)}
            names = ok if names is None else names & ok
        if len(experts) < n_experts or len({m.n_planes for m in ms}) != 1:
            names = {"cur"}
            frozen += 1
        order = [o.name for o in ms[0].options if o.name in names]
        opts = []
        for name in order:
            sel = [m.get(name) for m in ms]
            opts.append(Option(name, sum(o.bytes for o in sel), sum(o.distortion for o in sel),
                               sum(o.nnz for o in sel), {"gruppo_esperti": len(ms)}))
        out.append(TensorMenu(TensorKey(layer, -1, base), ms[0].n_planes, ms[0].shape, opts))
        members.append(idx)
    return Constrained(out, members, dict(dropped), len(groups), frozen)


def expand_choice(menus: Sequence[TensorMenu], c: Constrained, choice: Sequence[int], budget: int,
                  lam: float = 0.0, filled: bool = False) -> Allocation:
    """Scelte sui menu vincolati -> scelte (indici) sui menu ORIGINALI."""
    full = [None] * len(menus)
    for cm, ch, idx in zip(c.menus, choice, c.members):
        name = cm.options[ch].name
        for i in idx:
            full[i] = next(j for j, o in enumerate(menus[i].options) if o.name == name)
    d, b = _matrices(menus)
    import numpy as np
    ch = np.array(full, dtype=np.int64)
    tb, td = _totals(d, b, ch)
    return Allocation(list(map(int, ch)), lam, tb, td, budget, filled)


def solve_loadable(menus: Sequence[TensorMenu], budget: int, profilo: str = "oggi",
                   n_main: int = N_MAIN_DEFAULT, n_experts: int = 256, solver=None) -> Tuple[Allocation, Constrained]:
    """Risolve il budget sui soli menu caricabili. solver(menus, budget) -> Allocation
    (default: allocazione.allocate)."""
    from .allocazione import allocate, allocate_regimi
    solver = solver or allocate
    c = constrain_menus(menus, profilo, n_main, n_experts)
    if profilo == "scala-g64" and any(o.name in ("g64", "+1g64") for m in c.menus for o in m.options):
        # regime g64 accoppiato per (strato, proiezione) degli esperti: zaino a regimi
        coupled: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for j, m in enumerate(c.menus):
            if m.key.expert >= 0:
                coupled[group_key(m.key)].append(j)
        al = allocate_regimi(c.menus, budget, list(coupled.values()))
    else:
        al = solver(c.menus, budget)
    return expand_choice(menus, c, al.choice, budget, al.lam, al.filled), c


def check_plan_loadable(plan: Dict, profilo: str = "oggi", n_main: int = N_MAIN_DEFAULT,
                        n_experts: int = 256) -> List[str]:
    """Elenco dei motivi per cui il piano NON e' caricabile (vuoto = caricabile)."""
    errs = []
    groups: Dict[Tuple[int, str], set] = defaultdict(set)
    cnt: Dict[Tuple[int, str], int] = defaultdict(int)
    for t in plan["tensors"]:
        k = TensorKey.parse(t["key"])
        opt = t["option"]
        if not option_loadable(k, int(t["n_planes"]), opt, profilo, n_main):
            errs.append(f"{t['key']}: opzione {opt} non caricabile ({profilo})")
        if k.expert >= 0:
            groups[group_key(k)].add(opt)
            cnt[group_key(k)] += 1
    if profilo == "scala-g64":
        for g, opts in groups.items():
            k64 = {o in ("g64", "+1g64") for o in opts}
            if k64 == {True, False} or (True in k64 and cnt[g] < n_experts):
                errs.append(f"blk.{g[0]}.{g[1]}: esperti misti fra scala 64 e scala madre ({sorted(opts)}, "
                            f"{cnt[g]} esperti) — il tipo GGUF e' unico per tensore")
    if profilo == "oggi":
        for g, opts in groups.items():
            if opts != {"cur"} and (len(opts) > 1 or cnt[g] < n_experts):
                errs.append(f"blk.{g[0]}.{g[1]}: opzioni per esperto {sorted(opts)} su {cnt[g]} esperti — "
                            f"in 'oggi' servono tutti i {n_experts} esperti con la stessa opzione")
    return errs
