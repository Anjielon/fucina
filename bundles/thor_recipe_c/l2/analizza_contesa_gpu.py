#!/usr/bin/env python3
"""Contesa MISURATA sulla corsa prioritaria (D10): tempo di passo del congiunto per contesto di compresenza GPU.

Ingressi passivi: registro del trainer (righe `[ISO] {"passo": .., "s": ..}`) e campioni di `cardan_code_gpu.py campiona`
(code KFD per processo con riga di comando, processi llama Vulkan, gpu_busy, MemAvailable, pswpin). Per ogni passo,
finestra [fine - s, fine]: contesto = insieme dei TIPI di altri processi GPU presenti in tutti i campioni della finestra
(passi con contesto che cambia a meta' = "misto", esclusi dalle medie). Uscita: righe per passo + medie per contesto e
rapporto col passo da solo -> rallentamento della corsa prioritaria dovuto a ciascuna compresenza.
Nessun segnale, nessuna scrittura fuori da --out.
"""
import argparse
import datetime as dt
import json
import re
import statistics
import sys


def tipo(cmd: str) -> str:
    for k in ("cardan_piega_densi_gpu", "cardan_piega_gpu_atomica", "cardan_piega_gpu", "cardan_piega.py",
              "cardan_congiunto", "profila_strati_gpu", "cardan_costruisci", "cardan_kd"):
        if k in cmd:
            return k.replace(".py", "")
    return "altro_torch"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registro", required=True)
    ap.add_argument("--campioni", required=True)
    ap.add_argument("--corsa", default="cardan_congiunto", help="tipo della corsa prioritaria (escluso dal contesto)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    campioni = []
    for r in open(a.campioni):
        try:
            c = json.loads(r)
        except ValueError:
            continue
        t = dt.datetime.strptime(c["ora"], "%Y-%m-%d %H:%M:%S")
        altri = sorted(tipo(p["cmd"]) for p in c["processi"].values() if tipo(p["cmd"]) != a.corsa)
        if c.get("llama_vulkan"):
            altri += ["llama_vulkan"] * int(c["llama_vulkan"])
        campioni.append((t, tuple(altri), c.get("gpu_busy", {}), c.get("uso", {})))
    passi = []
    for r in open(a.registro):
        m = re.match(r"\[(\S+)\] (\{.*\"passo\".*\})$", r.strip())
        if not m:
            continue
        d = json.loads(m.group(2))
        fine = dt.datetime.fromisoformat(m.group(1)); inizio = fine - dt.timedelta(seconds=d["s"])
        fin = [c for c in campioni if inizio <= c[0] <= fine]
        if not fin:
            continue
        ctx = {c[1] for c in fin}
        busy = [v for c in fin for v in c[2].values()]
        passi.append({"passo": d["passo"], "s": d["s"], "fine": m.group(1), "campioni": len(fin),
                      "contesto": list(next(iter(ctx))) if len(ctx) == 1 else "misto",
                      "contesti": [list(x) for x in sorted(ctx)], "gpu_busy_medio": statistics.mean(busy) if busy else None})
    per = {}
    for p in passi:
        if p["contesto"] != "misto":
            per.setdefault(json.dumps(p["contesto"]), []).append(p["s"])
    base = statistics.median(per.get("[]", [float("nan")]))
    sint = {k: {"passi": len(v), "s_mediana": statistics.median(v), "s_media": statistics.mean(v),
                "rallentamento_vs_sola": statistics.median(v) / base if base == base else None}
            for k, v in sorted(per.items())}
    out = {"corsa": a.corsa, "passo_solo_mediana_s": base, "per_contesto": sint, "passi": passi}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in sint.items():
        print(f"{k:60s} passi {v['passi']:3d}  mediana {v['s_mediana']:7.1f} s  x{v['rallentamento_vs_sola'] or float('nan'):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
