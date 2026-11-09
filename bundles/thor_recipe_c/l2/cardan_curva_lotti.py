#!/usr/bin/env python3
"""Physical throughput curve of the expert fold workers from REAL production logs (D10/D16 closure, 22/10).

Input: the per-worker logs of cardan_piega_lotti.py of one stage (piega_<a>-<b>.out: "[HH:MM:SS] L.. f: ... [S s, lotti]")
and optionally the dense-fold log (piega_d/piega.out) whose active window is a second class running concurrently.
Every family k has an interval [end_k - S_k, end_k]. For each family we compute the time-weighted mean number of
expert workers active during its interval (a worker is active between its first family start and its last family end)
and whether the dense fold overlapped it. Families are grouped by rounded concurrency, giving seconds per family per
worker and the aggregate throughput (families/s = concurrency / seconds) for "experts alone" vs "experts + dense".
Restarted workers (rc 143 relaunch) are separate activity spans.

  python3 cardan_curva_lotti.py STAGE_DIR [--giorno 2026-10-21] [--out curva.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import statistics


def _t(giorno, hms, prec=None):
    t = dt.datetime.strptime(f"{giorno} {hms}", "%Y-%m-%d %H:%M:%S").timestamp()
    while prec is not None and t < prec - 43200:          # past midnight
        t += 86400
    return t


def leggi_lavoratori(d: Path, giorno):
    fam, spans = [], []
    for p in sorted((d / "piega_e").glob("piega_*.out")):
        span, prec = None, None
        for line in open(p, errors="replace"):
            m = re.match(r"\[(\d\d:\d\d:\d\d)\] L(\d+) (gate|up): .*\[([\d.]+) s, lotti\]", line)
            if not m:
                continue
            fine = _t(giorno, m.group(1), prec)
            prec = fine
            s = float(m.group(4))
            inizio = fine - s
            if span is not None and inizio - span[1] > 120:   # gap > 2 min = a relaunch: new activity span
                spans.append(tuple(span)); span = None
            span = [inizio, fine] if span is None else [span[0], fine]
            fam.append({"file": p.name, "strato": int(m.group(2)), "famiglia": m.group(3), "inizio": inizio,
                        "fine": fine, "secondi": s})
        if span:
            spans.append(tuple(span))
    return fam, spans


def finestra_densi(d: Path, giorno):
    p = d / "piega_d" / "piega.out"
    if not p.exists():
        return []
    out, cur, prec = [], None, None
    for line in open(p, errors="replace"):
        m = re.match(r"\[(\d\d:\d\d:\d\d)\]", line)
        if not m:
            continue
        t = _t(giorno, m.group(1), prec); prec = t
        cur = [t, t] if cur is None else [cur[0], t]
        if "FINE piega densi" in line:
            if cur[1] - cur[0] > 30:
                out.append(tuple(cur))
            cur = None
    return out


def media_attivi(a, b, spans):
    if b <= a:
        return 0.0
    tot = 0.0
    for s0, s1 in spans:
        tot += max(0.0, min(b, s1) - max(a, s0))
    return tot / (b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tappa")
    ap.add_argument("--giorno", default=None)
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.tappa)
    giorno = a.giorno or dt.datetime.fromtimestamp((d / "dopo.log").stat().st_mtime).strftime("%Y-%m-%d")
    first = open(d / "dopo.log").readline()
    m = re.match(r"\[(\d{4}-\d\d-\d\d)", first)
    giorno = a.giorno or (m.group(1) if m else giorno)
    fam, spans = leggi_lavoratori(d, giorno)
    densi = finestra_densi(d, giorno)
    gruppi = {}
    for f in fam:
        n = media_attivi(f["inizio"], f["fine"], spans)
        sovr = sum(max(0.0, min(f["fine"], b) - max(f["inizio"], a_)) for a_, b in densi) / f["secondi"]
        k = (round(n), "esperti+densi" if sovr > 0.5 else "esperti")
        gruppi.setdefault(k, []).append(f["secondi"])
    curva = []
    for (n, classe), ss in sorted(gruppi.items()):
        med = statistics.median(ss)
        curva.append({"lavoratori_esperti": n, "classi": classe, "famiglie": len(ss), "secondi_mediana": med,
                      "secondi_min": min(ss), "secondi_max": max(ss),
                      "famiglie_al_secondo_aggregato": round(n / med, 5) if n else None})
    ris = {"tappa": str(d), "famiglie": len(fam), "intervalli_lavoratori": len(spans),
           "finestre_densi_s": [round(b - a_, 1) for a_, b in densi], "curva": curva}
    print(json.dumps(ris, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(ris, indent=1))


if __name__ == "__main__":
    main()
