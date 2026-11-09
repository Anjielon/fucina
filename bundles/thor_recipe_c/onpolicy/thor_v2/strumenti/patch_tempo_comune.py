"""Motore comune delle patch di VELOCITA' della forgia scorrevole (25/9 notte).

Piano e numeri: thor_v2/PIANO_TEMPO_SCORREVOLE.md. Collaudo: strumenti/collaudo_patch_tempo.py.

Ogni patch e' un dizionario {file relativo a thor_v2: [(aggancio, sostituto), ...]} piu' una
MARCA (stringa unica che il sostituto porta dentro). Stesse regole di patch_ripartenza.py:
  - ogni aggancio deve comparire ESATTAMENTE una volta nel testo corrente, altrimenti rifiuto;
  - se la marca e' gia' nel file la patch risulta gia' applicata: rifiuto (niente doppioni);
  - il risultato di OGNI file deve passare ast.parse;
  - nessuna scrittura parziale: prima si calcolano tutti i file, poi si scrivono in file
    temporanei accanto agli originali e solo alla fine si rinominano (os.replace).
Gli agganci delle patch sono scelti DISGIUNTI: le patch si applicano in qualunque ordine
(verificato dal collaudo, che le applica da sole e tutte insieme).

Uso di una patch (tutte uguali):
    python3 strumenti/patch_tempo_<nome>.py                  # applica ai file veri di thor_v2
    python3 strumenti/patch_tempo_<nome>.py --cartella COPIA # applica a una copia
    python3 strumenti/patch_tempo_<nome>.py --prova          # controlla soltanto, non scrive
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

THOR_V2 = Path(__file__).resolve().parent.parent


def applica_testo(testo: str, sostituzioni: list, marca: str, nome: str) -> str:
    if marca in testo:
        raise SystemExit(f"{nome}: patch gia' applicata (marca {marca!r} presente): niente da fare")
    for vecchio, nuovo in sostituzioni:
        n = testo.count(vecchio)
        if n != 1:
            raise SystemExit(f"{nome}: aggancio trovato {n} volte (atteso 1): {vecchio[:110]!r}")
        testo = testo.replace(vecchio, nuovo)
    if marca not in testo:
        raise SystemExit(f"{nome}: la marca {marca!r} non compare nel risultato (patch malformata)")
    ast.parse(testo)
    return testo


def esegui_patch(descrizione: str, piano: dict, marca: str, argv=None) -> int:
    ap = argparse.ArgumentParser(description=descrizione)
    ap.add_argument("--cartella", type=Path, default=THOR_V2, help="cartella con i file da patchare (default thor_v2)")
    ap.add_argument("--prova", action="store_true", help="controlla agganci e sintassi, non scrive")
    args = ap.parse_args(argv)
    nuovi = {}
    for rel, sostituzioni in piano.items():
        p = args.cartella / rel
        nuovi[p] = applica_testo(p.read_text(), sostituzioni, marca, rel)
    n_sost = sum(len(s) for s in piano.values())
    if args.prova:
        print(f"prova OK ({marca}): {n_sost} agganci unici in {len(piano)} file, risultato valido — NULLA scritto "
              f"in {args.cartella}")
        return 0
    temporanei = {}
    try:
        for p, testo in nuovi.items():
            q = p.with_name(p.name + ".patch_tmp")
            q.write_text(testo)
            temporanei[p] = q
        for p, q in temporanei.items():
            os.replace(q, p)
    finally:
        for q in temporanei.values():
            if q.exists():
                q.unlink()
    print(f"patch applicata ({marca}): {n_sost} sostituzioni in " + ", ".join(str(p) for p in nuovi))
    return 0
