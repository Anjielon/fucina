#!/usr/bin/env python3
"""Precondizione di lancio (regola fucina E67, 28/9): una corsa E2E che parte dalla riga R del teacher deve trovare il
blocco che contiene R presente e NON consumato; altrimenti aspetta per sempre a GPU ferma (TT2, 28/9 21:3x).
Uso: verifica_righe_teacher.py <cartella teacher> <riga iniziale>   -> esce 0 se ok, 2 se il lancio va rifiutato."""
import json, sys
from pathlib import Path

d, r = Path(sys.argv[1]), int(sys.argv[2])
blocchi = [b for b in json.load(open(d / "piano.json"))["blocchi"] if b["nome"].startswith("train/")]
b = next((b for b in blocchi if b["da"] <= r < b["a"]), None)
if b is None:
    sys.exit(f"⛔ riga {r}: nessun blocco train nel piano ({len(blocchi)} blocchi)") or 2
base = d / b["nome"]
if Path(str(base) + ".consumato").exists():
    liberi = [x["da"] for x in blocchi if (d / x["nome"]).exists()]
    print(f"⛔ riga {r}: {b['nome']} gia' CONSUMATO; prima riga disponibile: {min(liberi) if liberi else 'nessuna'}")
    sys.exit(2)
if not base.exists():
    print(f"⛔ riga {r}: {b['nome']} non ancora prodotto dal teacher"); sys.exit(2)
print(f"ok: riga {r} in {b['nome']} (righe {b['da']}-{b['a']}), presente e non consumato")
