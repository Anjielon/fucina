"""Unione di piani parziali (un processo per sottoinsieme di strati, --layers) e
risoluzione UNICA del budget globale.

Ogni piano_parte*.json porta, per ogni tensore, il menu completo (byte, distorsione):
le curve R-D. Il knapsack a scelta multipla e' separabile, quindi risolverlo su
tutti i menu insieme equivale a combinare le curve per strato allo stesso lambda
(inviluppo convesso) — qui lo si risolve direttamente sui menu, poi lo si chiude
col riempimento greedy e, se la taglia lo consente, con lo zaino esatto a
granularita' di byte fissata (vedi zaino.py).

Uso:
  python3 -m rd_ternary.unisci piano_parte0.json piano_parte1.json piano_parte2.json \
      --budget-rel 0.0 --expect-layers 0-40 --out piano_globale.json
  # punteggio coi dKL MISURATI per gruppo di esperti (MoPEQ/GEMQ) invece del proxy:
  python3 -m rd_ternary.unisci piano_parte*.json --delta-kl dkl_misurati.tsv --unmeasured cur \
      --budget-rel 0.02 --out piano_misurato.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .allocazione import Allocation, allocate, lagrangian_choice, _matrices
from .formato import TensorKey
from .menu import Option, TensorMenu
from .piano import PLAN_VERSION, cfg_from_dict, export_plan, load_plan

# margine (byte) per le voci nuove della tabella dei tensori e l'allineamento nel file GGUF
FILE_SLACK_BYTES = 1 << 20

# chiavi di configurazione che DEVONO coincidere fra le parti (il resto e' informativo)
_CFG_MUST_MATCH = ("densities", "packing", "sparse_coding", "damp", "refit_existing",
                   "min_planes_to_drop", "allow_plus_full", "safeguard", "mm_dtype")


def menus_from_plan(plan: Dict) -> List[TensorMenu]:
    """Ricostruisce i menu da un piano. Piani nuovi: "menu_full" (nnz ed extra per
    opzione). Piani vecchi: "menu" [nome, byte, d] + nnz/extra della sola opzione scelta."""
    out = []
    reserved = {"key", "n_planes", "shape", "option", "bytes", "delta_bytes", "distortion",
                "distortion_cur", "nnz", "menu", "menu_full"}
    for t in plan["tensors"]:
        key = TensorKey.parse(t["key"])
        if "menu_full" in t:
            opts = [Option(o["name"], int(o["bytes"]), float(o["distortion"]), int(o.get("nnz", 0)),
                           dict(o.get("extra", {}))) for o in t["menu_full"]]
        else:
            opts = [Option(n, int(b), float(d)) for n, b, d in t["menu"]]
            for o in opts:
                if o.name == t["option"]:
                    o.nnz = int(t.get("nnz", 0))
                    o.extra = {k: v for k, v in t.items() if k not in reserved}
        out.append(TensorMenu(key, int(t["n_planes"]), tuple(t["shape"]), opts))
    return out


def _same_menu(a: TensorMenu, b: TensorMenu, rtol: float = 1e-9) -> bool:
    if [o.name for o in a.options] != [o.name for o in b.options]:
        return False
    return all(x.bytes == y.bytes and math.isclose(x.distortion, y.distortion, rel_tol=rtol, abs_tol=0.0)
               for x, y in zip(a.options, b.options))


def merge_plans(plans: Sequence[Dict], names: Optional[Sequence[str]] = None) -> Tuple[List[TensorMenu], Dict, Dict]:
    """-> (menu ordinati per chiave, config comune, info). Errore se le configurazioni
    differiscono o se lo stesso tensore compare con menu diversi (doppioni identici:
    tenuti una volta, contati in info)."""
    names = list(names) if names is not None else [f"parte{i}" for i in range(len(plans))]
    if not plans:
        raise ValueError("nessun piano da unire")
    for p, n in zip(plans, names):
        if p.get("version") != PLAN_VERSION:
            raise ValueError(f"{n}: versione piano {p.get('version')} non supportata")
    cfg0 = plans[0]["config"]
    for p, n in zip(plans[1:], names[1:]):
        diff = {k: (cfg0.get(k), p["config"].get(k)) for k in _CFG_MUST_MATCH if cfg0.get(k) != p["config"].get(k)}
        if diff:
            raise ValueError(f"{n}: configurazione diversa da {names[0]}: {diff}")
    by_key: Dict[TensorKey, TensorMenu] = {}
    origin: Dict[TensorKey, str] = {}
    dup, missing_fisher, fast = 0, {}, set()
    for p, n in zip(plans, names):
        fast.add(bool(p["config"].get("fast_diag", False)))
        missing_fisher.update(p.get("missing_fisher", {}))
        for m in menus_from_plan(p):
            if m.key in by_key:
                if not _same_menu(by_key[m.key], m):
                    raise ValueError(f"{m.key.name()}: menu diversi in {origin[m.key]} e {n}")
                dup += 1
                continue
            by_key[m.key] = m
            origin[m.key] = n
    cfg = dict(cfg0)
    # parti col percorso veloce e parti senza: la rimaterializzazione usa il riferimento
    # (le differenze fra i due percorsi sono di arrotondamento, entro la tolleranza)
    cfg["fast_diag"] = fast == {True}
    menus = [by_key[k] for k in sorted(by_key)]
    layers = sorted({k.layer for k in by_key})
    info = {"merged_from": list(names), "n_tensors": len(menus), "duplicates_skipped": dup,
            "layers": layers, "missing_fisher": missing_fisher}
    return menus, cfg, info


def parse_layers(spec: Sequence[str]) -> List[int]:
    """'0-40' '3' '5,7' -> lista ordinata di interi."""
    out = set()
    for s in spec:
        for part in str(s).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    return sorted(out)


def layer_curves(menus: Sequence[TensorMenu], n_lambda: int = 64) -> Dict[int, List[Tuple[int, float]]]:
    """Curva R-D (inviluppo convesso inferiore) per strato: punti (byte, distorsione)
    ottenuti con la scelta lagrangiana su una griglia di lambda. E' la curva che il
    knapsack globale combina; serve a leggere DOVE vanno i byte."""
    d, b = _matrices(menus)
    layer = np.array([m.key.layer for m in menus])
    fin = np.isfinite(d)
    dd = np.where(fin, d, np.nan)
    bb = np.where(fin, b.astype(np.float64), np.nan)
    # griglia di lambda dalle pendenze reali dei menu
    slopes = []
    for n in range(len(menus)):
        f = fin[n]
        if f.sum() >= 2:
            db = np.subtract.outer(bb[n, f], bb[n, f])
            ddd = np.subtract.outer(dd[n, f], dd[n, f])
            ok = (db > 0) & (ddd < 0)
            slopes.extend((-ddd[ok] / db[ok]).tolist())
    if slopes:
        lo, hi = np.percentile(slopes, 0.5), np.percentile(slopes, 99.5)
        lams = np.concatenate([[0.0], np.geomspace(max(lo, 1e-300) / 4, max(hi, lo) * 4, n_lambda), [1e30]])
    else:
        lams = np.array([0.0, 1e30])
    curves: Dict[int, List[Tuple[int, float]]] = {}
    rows = np.arange(len(menus))
    for lam in lams:
        ch = lagrangian_choice(d, b, float(lam))
        cb, cd = b[rows, ch], d[rows, ch]
        for L in np.unique(layer):
            sel = layer == L
            curves.setdefault(int(L), []).append((int(cb[sel].sum()), float(cd[sel].sum())))
    return {L: sorted(set(v)) for L, v in curves.items()}


def solve(menus: Sequence[TensorMenu], budget: int, exact: bool = True, exact_unit: Optional[int] = None,
          exact_max_cells: float = 5e7) -> Allocation:
    """Lagrangiano + riempimento greedy; poi zaino esatto (costi a unita' di exact_unit byte,
    arrotondati per eccesso: la soluzione resta nel budget; None = granularita' automatica)
    e si tiene il migliore dei due."""
    from .zaino import knapsack_exact
    al = allocate(menus, budget)
    if exact:
        ex = knapsack_exact(menus, budget, exact_unit, max_cells=exact_max_cells)
        if ex is not None and ex.total_distortion < al.total_distortion - 1e-15 * abs(al.total_distortion):
            ex.lam = al.lam
            return ex
    return al


def _expand(paths: Iterable[str]) -> List[str]:
    out = []
    for p in paths:
        g = sorted(glob.glob(p))
        out.extend(g if g else [p])
    return out


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="+", help="piano_parte*.json (anche glob)")
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--budget-bytes", type=int, default=None)
    g.add_argument("--budget-rel", type=float, default=0.0, help="budget = byte 'cur' totali * (1 + x)")
    ap.add_argument("--expect-layers", nargs="*", default=None, help="es. 0-40: errore se ne manca uno")
    ap.add_argument("--allow-missing-layers", action="store_true", help="solo avviso se mancano strati")
    ap.add_argument("--delta-kl", default=None, help="dKL misurati per gruppo (TSV/JSON, vedi misurato.py)")
    ap.add_argument("--unmeasured", choices=("cur", "proxy"), default="cur",
                    help="tensori fuori dai gruppi misurati: fermi a 'cur' o col proxy calibrato")
    ap.add_argument("--exact-unit", type=int, default=None,
                    help="granularita' (byte) dello zaino esatto (default: automatica, la piu' fine che sta in memoria)")
    ap.add_argument("--no-exact", action="store_true", help="solo Lagrangiano + riempimento greedy")
    ap.add_argument("--curves", default=None, help="scrive le curve R-D per strato in questo JSON")
    ap.add_argument("--loadable-only", action="store_true",
                    help="solo opzioni che il motore carica (vedi motore.py): niente sparsi, limiti di piani, "
                         "esperti a gruppo intero col motore 'oggi'")
    ap.add_argument("--motore", choices=("oggi", "piani-variabili", "scala-g64"), default="oggi",
                    help="profilo del motore per --loadable-only")
    ap.add_argument("--n-strati", type=int, default=None,
                    help="strati principali (lo strato MTP e' il successivo); default da --gguf o 40")
    ap.add_argument("--n-esperti", type=int, default=256, help="esperti per strato (gruppi completi)")
    ap.add_argument("--gguf", default=None, help="GGUF sorgente: legge strati/esperti e serve a --max-file-bytes")
    ap.add_argument("--max-file-bytes", type=int, default=None,
                    help="tetto del FILE GGUF risultante (es. 9663676416 = 9 GiB): il budget dei tensori "
                         "si riduce perche' il file ci stia (richiede --gguf)")
    ap.add_argument("--file-bytes", type=int, default=None,
                    help="R022: taglia (byte) del GGUF sorgente, in alternativa a --gguf (che serve solo a "
                         "leggerla): con --max-file-bytes fissa il budget dei tensori senza avere il file")
    ap.add_argument("--allineamento", type=int, default=32, help="allineamento dei tensori nel GGUF (default 32)")
    ap.add_argument("--scale-cur", choices=("sorgente", "rifit"), default="sorgente",
                    help="R022: come scrivi_piano_gguf scrive i tensori 'cur': scale del sorgente (default: la riga 'cur' "
                         "del dkl.tsv NON e' linea di base) o rifittate (le opzioni si misurano contro 'cur')")
    ap.add_argument("--se-k", type=float, default=0.0,
                    help="R022: penalita' k*se sul dKL di ogni opzione != cur (maledizione del vincitore); 0 = spenta")
    # R033 (motore TQZ1 al volo, thor-llama cloud/tq1z-al-volo): il tetto e' la RAM OCCUPATA a modello caricato
    ap.add_argument("--max-ram-bytes", type=int, default=None,
                    help="tetto di RAM al volo (es. 9663676416 = 9 GiB): lo zaino lavora in byte di RAM (blob TQZ1 + "
                         "indice) invece che in byte di file; richiede --gguf (sorgente in chiaro) o --gguf-z-sorgente")
    ap.add_argument("--gguf-z-sorgente", default=None, help="GGUF-Z della sorgente (RAM esatta dall'indice, istantanea)")
    ap.add_argument("--margine-ram", type=int, default=16 * 2**20, help="margine sotto --max-ram-bytes (default 16 MiB)")

    a = ap.parse_args(argv)

    paths = _expand(a.plans)
    plans = [load_plan(p) for p in paths]
    menus, cfg_d, info = merge_plans(plans, paths)
    if a.expect_layers:
        want = set(parse_layers(a.expect_layers))
        miss = sorted(want - set(info["layers"]))
        if miss:
            msg = f"strati mancanti nell'unione: {miss}"
            if not a.allow_missing_layers:
                raise SystemExit(msg)
            print(f"[avviso] {msg}", file=sys.stderr)
    ram_info = None
    if a.max_ram_bytes:
        # R033: byte di file -> byte di RAM al volo. Rapporto per tipo misurato sulla sorgente (TQ1_0 se in=512, cioe'
        # le down; TQ1_0R_1024 altrimenti). E' una stima per le opzioni NUOVE: il piano scritto si verifica poi ESATTO con
        # `python3 -m rd_ternary.ram_tq1z --gguf piano.gguf`.
        from . import ram_tq1z
        if a.gguf_z_sorgente:
            src_ram = ram_tq1z.ram_gguf_z(a.gguf_z_sorgente)
            rapp = {str(ram_tq1z.T_TQ1_0R_1024): src_ram["ternari_ram"] / src_ram["ternari_grezzi"],
                    str(ram_tq1z.T_TQ1_0): src_ram["ternari_ram"] / src_ram["ternari_grezzi"]}
        elif a.gguf:
            src_ram = ram_tq1z.ram_gguf(a.gguf)
            rapp = src_ram["rapporto_per_tipo"]
        else:
            raise SystemExit("--max-ram-bytes richiede --gguf o --gguf-z-sorgente")
        tipo_di = lambda m: ram_tq1z.T_TQ1_0 if int(m.shape[-1]) % 1024 else ram_tq1z.T_TQ1_0R_1024   # noqa: E731
        base_file = sum(m.get("cur").bytes for m in menus)
        menus = ram_tq1z.menus_in_ram(menus, lambda m: rapp.get(str(tipo_di(m)), 1.0))
        base_ram = sum(m.get("cur").bytes for m in menus)
        fuori = src_ram["ram_al_volo"] - base_ram          # RAM fuori dal piano (non ternari + piani non nel menu)
        ram_info = {"max_ram_bytes": a.max_ram_bytes, "ram_sorgente_al_volo": src_ram["ram_al_volo"],
                    "ram_fuori_piano": fuori, "rapporti": rapp, "piano_cur_file": base_file, "piano_cur_ram": base_ram,
                    "margine": a.margine_ram, "tetto_piano_ram": a.max_ram_bytes - fuori - a.margine_ram}
        if a.budget_bytes is None and a.budget_rel == 0.0:
            a.budget_bytes = ram_info["tetto_piano_ram"]
        else:
            a.budget_bytes = min(a.budget_bytes if a.budget_bytes is not None else int(base_ram * (1.0 + a.budget_rel)),
                                 ram_info["tetto_piano_ram"])
        if a.max_file_bytes:
            raise SystemExit("--max-ram-bytes e --max-file-bytes insieme: scegli il tetto (RAM al volo o file)")
    base = sum(m.get("cur").bytes for m in menus)
    budget = a.budget_bytes if a.budget_bytes is not None else int(base * (1.0 + a.budget_rel))
    cfg = cfg_from_dict(cfg_d)
    extra = {"merged_from": info["merged_from"], "missing_fisher": info["missing_fisher"]}
    if ram_info is not None:
        extra["ram_budget"] = ram_info
    n_main, n_exp = a.n_strati, a.n_esperti
    file_size, alignment = a.file_bytes, a.allineamento
    if a.gguf:
        from .gguf_grezzo import read_gguf
        from .scrivi_piano_gguf import n_main_layers
        src = read_gguf(a.gguf)
        n_main = n_main if n_main is not None else n_main_layers(src)
        n_exp = int(src.kv.get(f"{src.kv.get('general.architecture')}.expert_count", n_exp) or n_exp)
        file_size, alignment = src.file_size, src.alignment
    if a.max_file_bytes and file_size is None:
        raise SystemExit("--max-file-bytes richiede --gguf o --file-bytes")
    cap_senza_overhead = None
    # senza --budget-bytes/--budget-rel espliciti il tetto e' il file: si spende TUTTO il margine
    esplicito = a.budget_bytes is not None or a.budget_rel != 0.0
    if a.max_file_bytes:
        fixed = file_size - base            # byte del file fuori dal piano
        cap_senza_overhead = a.max_file_bytes - fixed
        extra["file_budget"] = {"max_file_bytes": a.max_file_bytes, "bytes_fuori_piano": fixed,
                                "file_bytes_sorgente": file_size, "tetto_tensori_senza_overhead": cap_senza_overhead}
        if not a.loadable_only:
            cap = cap_senza_overhead - FILE_SLACK_BYTES
            extra["file_budget"].update({"margine": FILE_SLACK_BYTES, "tetto_tensori": cap})
            budget = min(budget, cap) if esplicito else cap
            cap_senza_overhead = None
    n_main = n_main if n_main is not None else 40

    def _loadable(bud: int):
        from . import motore
        if esplicito:
            bud = min(bud, budget)
        if a.delta_kl:
            from .misurato import load_deltas, load_deltas_se
            from .misurato_caricabile import allocate_measured_loadable
            res = allocate_measured_loadable(
                menus, load_deltas(a.delta_kl), bud, a.motore, n_main, n_exp, a.unmeasured,
                se=load_deltas_se(a.delta_kl), se_k=a.se_k, scale_cur=a.scale_cur,
                solver=lambda ms, b: solve(ms, b, exact=not a.no_exact, exact_unit=a.exact_unit))
            ex = res.summary()
            ex["score_mode"] = "measured_dkl_loadable"
            ex["vincoli_motore"] = {"motore": a.motore, "n_strati": n_main, "n_esperti": n_exp,
                                    "opzioni_tolte": res.dropped_options, "unita_di_decisione": res.units,
                                    "unita_spostate": res.n_moved, "unita_esperti_ferme": res.frozen_units,
                                    "se_k": a.se_k, "scale_cur": a.scale_cur,
                                    "cur_base_somma": res.cur_base_somma,
                                    "pred_dkl_vs_sorgente": res.pred_vs_sorgente}
            return res.allocation, ex
        al, cons = motore.solve_loadable(menus, bud, a.motore, n_main, n_exp,
                                         solver=lambda ms, b: solve(ms, b, exact=not a.no_exact,
                                                                    exact_unit=a.exact_unit))
        return al, {"score_mode": "proxy",
                    "vincoli_motore": {"motore": a.motore, "n_strati": n_main, "n_esperti": n_exp,
                                       "opzioni_tolte": cons.dropped_options, "gruppi_esperti": cons.groups,
                                       "gruppi_fermi": cons.frozen_groups}}

    if a.loadable_only:
        if cap_senza_overhead is not None:
            from .taglia_file import risolvi_nel_file
            alloc, ex, info_file = risolvi_nel_file(_loadable, menus, cap_senza_overhead, a.motore, n_exp, alignment)
            extra["file_budget"]["overhead_max"] = info_file["overhead_max"]
            extra["file_budget"]["giri"] = info_file["giri"]
            budget = cap_senza_overhead - info_file["overhead_max"]
        else:
            alloc, ex = _loadable(budget)
        extra.update(ex)
        what = "dKL misurato" if a.delta_kl else "proxy"
        note = f"unione di {len(paths)} piani, punteggio {what}, solo opzioni caricabili ({a.motore})"
    elif a.delta_kl:
        from .misurato import allocate_measured, load_deltas
        res = allocate_measured(menus, load_deltas(a.delta_kl), budget, unmeasured=a.unmeasured,
                                exact=not a.no_exact, exact_unit=a.exact_unit)
        alloc = res.allocation
        extra.update(res.summary())
        note = f"unione di {len(paths)} piani, punteggio dKL misurato ({a.delta_kl})"
    else:
        alloc = solve(menus, budget, exact=not a.no_exact, exact_unit=a.exact_unit)
        extra["score_mode"] = "proxy"
        note = f"unione di {len(paths)} piani, punteggio proxy"
    if a.curves:
        with open(a.curves, "w") as f:
            json.dump({str(k): v for k, v in layer_curves(menus).items()}, f)
    plan = export_plan(menus, alloc, cfg, a.out, note=note, extra=extra)
    counts: Dict[str, int] = {}
    for t in plan["tensors"]:
        counts[t["option"]] = counts.get(t["option"], 0) + 1
    rep = {"plans": len(paths), "n_tensors": len(menus), "layers": f"{info['layers'][0]}..{info['layers'][-1]} "
           f"({len(info['layers'])})", "duplicates_skipped": info["duplicates_skipped"],
           "base_bytes": base, "budget": budget, "total_bytes": alloc.total_bytes,
           "pred_distortion_base": plan["pred_distortion_base"], "pred_distortion": plan["pred_distortion_total"],
           "options": counts, "score_mode": extra["score_mode"], "fast_diag": cfg_d["fast_diag"],
           "missing_fisher": len(info["missing_fisher"])}
    if "file_budget" in extra:
        fb = extra["file_budget"]
        rep["file_budget"] = {k: v for k, v in fb.items() if k != "giri"}
        if a.max_file_bytes and fb.get("overhead_max") is not None:
            file_max = file_size - base + alloc.total_bytes + fb["overhead_max"]
            rep["file_budget"].update({"file_stimato_max": file_max,
                                       "margine_residuo_byte": a.max_file_bytes - file_max,
                                       "delta_byte_sul_file_sorgente": file_max - file_size})
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
