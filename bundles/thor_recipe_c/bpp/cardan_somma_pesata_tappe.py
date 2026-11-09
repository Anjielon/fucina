#!/usr/bin/env python3
"""Weighted completion time of a stage ("tappa") measured from its REAL logs (D9 closure, 22/10).

For one stage directory of the joint-CARDAN chain it reconstructs, from timestamps written by the chain itself
(dopo.log, piega_d/piega.out, patch.out mtimes, sonda.out, mmlu log mtime), when each deliverable became available:
  esperti  = expert planes patched (piega_e/patch.out mtime)
  densi    = dense planes patched (piega_d/patch.out mtime)
  sonda    = probe verdict (sonda.out mtime)
  mmlu     = MMLU-2000 log complete (mmlu_CONGIUNTO.log mtime)
and the critical-path seconds of each phase. sum(peso * fine) is computed from the stage start (first dopo.log line).
Phases that waited on an EXTERNAL cause (contention from other runs, a chain bug) are reported, not hidden: pass
--escludi PHASE to also print the sum with that phase's duration replaced by the reference stage's duration.

  python3 cardan_somma_pesata_tappe.py DIR_A DIR_B [--pesi esperti=1,densi=1,sonda=1,mmlu=3]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re


def _ts_riga(line):
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", line)
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() if m else None


def _mtime(p):
    return os.stat(p).st_mtime if os.path.exists(p) else None


def _densi_finestre(p, giorno):
    """[(start, end)] of each dense-fold invocation from piega.out ([HH:MM:SS] lines; day from the stage)."""
    out, cur = [], None
    if not os.path.exists(p):
        return out
    for line in open(p, errors="replace"):
        m = re.match(r"\[(\d\d:\d\d:\d\d)\]", line)
        if not m:
            continue
        t = dt.datetime.strptime(f"{giorno} {m.group(1)}", "%Y-%m-%d %H:%M:%S").timestamp()
        if cur is None:
            cur = [t, t]
        cur[1] = t
        if "FINE piega densi" in line:
            out.append(tuple(cur))
            cur = None
    return out


def misura(d: Path):
    righe = [l for l in open(d / "dopo.log", errors="replace") if _ts_riga(l)]
    t0 = _ts_riga(righe[0])
    giorno = dt.datetime.fromtimestamp(t0).strftime("%Y-%m-%d")
    fine = {"esperti": _mtime(d / "piega_e/patch.out"), "densi": _mtime(d / "piega_d/patch.out"),
            "sonda": _mtime(d / "sonda.out"), "mmlu": _mtime(d / "mmlu_CONGIUNTO.log")}
    finestre = _densi_finestre(d / "piega_d/piega.out", giorno)
    # the dense phase is on the critical path only from max(expert patch, start of its LAST invocation): v10 starts it
    # from BASE while the experts fold, then a second call finds everything committed (D36 provenance) and patches.
    ultimo = finestre[-1][0] if finestre else fine["esperti"]
    densi_sul_percorso = fine["densi"] - max(fine["esperti"], ultimo)
    attesa_tra = max(0.0, ultimo - fine["esperti"])                # expert patch -> last dense start (audit, waits)
    eventi = [l.strip()[:160] for l in righe if re.search(r"ERRORE|rilancio|STALLO", l)]
    return {"dir": str(d), "inizio": dt.datetime.fromtimestamp(t0).isoformat(),
            "fine_relativa_s": {k: (round(v - t0, 1) if v else None) for k, v in fine.items()},
            "densi_finestre": [(dt.datetime.fromtimestamp(a).strftime("%H:%M:%S"), round(b - a, 1))
                               for a, b in finestre],
            "densi_sul_percorso_critico_s": round(densi_sul_percorso, 1),
            "attesa_fra_patch_esperti_e_ultimo_avvio_densi_s": round(attesa_tra, 1),
            "mmlu_s": round(fine["mmlu"] - max(fine["densi"], fine["esperti"]), 1),
            "eventi_esterni_o_guasti": eventi}


def somma(m, pesi):
    return sum(pesi[k] * m["fine_relativa_s"][k] for k in pesi if m["fine_relativa_s"].get(k) is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tappe", nargs="+")
    ap.add_argument("--pesi", default="esperti=1,densi=1,sonda=1,mmlu=3")
    a = ap.parse_args()
    pesi = {k: float(v) for k, v in (x.split("=") for x in a.pesi.split(","))}
    ris = []
    for t in a.tappe:
        m = misura(Path(t))
        m["somma_pesata_s"] = round(somma(m, pesi), 1)
        ris.append(m)
    print(json.dumps({"pesi": pesi, "tappe": ris}, indent=1))


if __name__ == "__main__":
    main()
