#!/usr/bin/env python3
"""Verifica AVVERSARIA del ripristino di applica_patch_tempo.applica_vera (25/9) — su COPIE, solo CPU.

La catena vera viene interrotta dopo 3 patch in quattro modi: eccezione non-SystemExit (Ctrl-C =
KeyboardInterrupt, OSError), SIGTERM, SIGKILL. Si guarda se i file BERSAGLIO della copia tornano ai byte
originali (o, per SIGKILL, se la corsa dopo se ne accorge e rifiuta invece di applicare sopra).

    python3 strumenti/collaudo_ripristino_giornale.py --lavoro <scratch> [--modulo <applica_patch_tempo.py>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ESITI: list = []


def ok(nome, cond, dettaglio=""):
    ESITI.append((nome, bool(cond)))
    print(f"  {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)


def carica(modulo: Path):
    spec = importlib.util.spec_from_file_location("applica_in_prova", modulo)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def copia(A, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in Path(A.THOR_V2).glob("*.py"):
        shutil.copy2(f, dest / f.name)
    return dest


def stato(c: Path, nomi) -> dict:
    return {n: (c / n).read_bytes() for n in nomi}


def figlio(modulo: Path, cartella: Path, modo: str) -> None:
    """Nel processo figlio: prova generale, poi applica_vera con una catena che dopo 3 patch si interrompe."""
    A = carica(modulo)
    A.processi_in_conflitto = lambda *a, **k: []          # la guardia qui non e' in prova (copie in scratch)

    def catena_interrotta(c):
        for p in A.ORDINE[:3]:
            subprocess.run([sys.executable, str(Path(A.QUI) / f"patch_tempo_{p}.py"), "--cartella", str(c)],
                           check=True, capture_output=True)
        if modo == "keyboard":
            raise KeyboardInterrupt
        if modo == "oserror":
            raise OSError("disco sparito (simulato)")
        if modo in ("sigterm", "sigkill", "sighup"):
            os.kill(os.getpid(), {"sigterm": signal.SIGTERM, "sigkill": signal.SIGKILL, "sighup": signal.SIGHUP}[modo])
            import time
            time.sleep(5)
    with tempfile.TemporaryDirectory(prefix="avv_ripristino_") as d:
        foto, uscita = A.prova_generale(cartella, Path(d))
        A.applica_vera(cartella, foto, uscita, catena_fn=catena_interrotta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulo", type=Path, default=Path(__file__).resolve().parent / "applica_patch_tempo.py")
    ap.add_argument("--lavoro", type=Path)
    ap.add_argument("--figlio", default="")
    ap.add_argument("--cartella", type=Path)
    a = ap.parse_args()
    if a.figlio:
        figlio(a.modulo.resolve(), a.cartella.resolve(), a.figlio)
        return 0
    A = carica(a.modulo.resolve())
    nomi = A.bersagli_catena()
    lav = a.lavoro.resolve()
    for modo in ("keyboard", "oserror", "sigterm", "sighup", "sigkill"):
        c = copia(A, lav / f"copia_{modo}")
        prima = stato(c, nomi)
        r = subprocess.run([sys.executable, __file__, "--modulo", str(a.modulo.resolve()), "--figlio", modo,
                            "--cartella", str(c)], capture_output=True, text=True)
        dopo = stato(c, nomi)
        diversi = [n for n in nomi if dopo[n] != prima[n]]
        if modo != "sigkill":
            ok(f"{modo}: catena interrotta dopo 3 patch → bersagli ai byte originali (rc {r.returncode})", not diversi,
               f"parzialmente patchati: {diversi}" if diversi else (r.stdout + r.stderr).strip().splitlines()[-1][:120])
            ok(f"{modo}: nessun giornale rimasto", not (c / "strumenti" / ".giornale_applica_patch_tempo").exists())
        else:   # SIGKILL non si intercetta: i bersagli restano a meta', e deve restare il GIORNALE
            ok(f"sigkill: bersagli a meta' (rc {r.returncode}, come atteso: SIGKILL non si intercetta) e GIORNALE presente",
               diversi and (c / "strumenti" / ".giornale_applica_patch_tempo" / "giornale.json").exists(),
               f"a meta': {diversi}")
        if modo == "sigkill":
            # SIGKILL non si intercetta: conta che la corsa DOPO se ne accorga (non applichi sopra a una catena a meta')
            r2 = subprocess.run([sys.executable, str(a.modulo.resolve()), "--cartella", str(c)],
                                capture_output=True, text=True, env={**os.environ})
            testo = (r2.stdout + r2.stderr)
            ok("sigkill: la corsa successiva RIFIUTA (giornale di una catena interrotta) invece di applicare sopra",
               r2.returncode != 0 and "giornale" in testo.lower(), testo.strip().splitlines()[-1][:160] if testo.strip() else "")
            r3 = subprocess.run([sys.executable, str(a.modulo.resolve()), "--cartella", str(c), "--ripristina-giornale"],
                                capture_output=True, text=True)
            ok("sigkill: --ripristina-giornale riporta i bersagli ai byte originali e toglie il giornale",
               r3.returncode == 0 and stato(c, nomi) == prima and not any((c / "strumenti").glob(".giornale*")),
               (r3.stdout + r3.stderr).strip().splitlines()[-1][:160] if (r3.stdout + r3.stderr).strip() else "")
    n = sum(v for _, v in ESITI)
    print(f"\n{'✅ TUTTI' if n == len(ESITI) else '⛔ FALLITI ' + str(len(ESITI) - n) + ' di'} {len(ESITI)} controlli")
    return 0 if n == len(ESITI) else 1


if __name__ == "__main__":
    raise SystemExit(main())
