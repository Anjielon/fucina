#!/usr/bin/env python3
"""Integrazione additiva giro 1: famiglie indipendenti, ripresa, guardia algebra.

Stesse opzioni di cardan_fucina_gpu.py, piu' --tentativi-lavoratori (default 2)
e --correzione-gpu (sperimentale: GEMM C D^T fp64 device, da collaudare).
Non importa torch prima delle guardie della CLI originale.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # priorita' 0 (21/10): ogni worker inductor apre /dev/kfd e prende una coda SDMA
from pathlib import Path
import sys
import uuid

import cardan_fucina_gpu as F
import pianificatore_gpu as PG
import pianificatore_guardia_gpu as E

QUI = Path(__file__).resolve().parent
_prepara = F.prepara
_raccogli = F.raccogli
_esegui = E.esegui


def prepara(a, correzione_gpu=False):
    piano, unita = _prepara(a)
    if a.fase == "esperti":
        nuove = []
        for u in unita:
            for i, fam in enumerate(("gate", "up")):
                nomi = [n for n in u.dati["file_attesi"] if f".ffn_{fam}_exps" in n]
                if not nomi:
                    raise ValueError(f"nessun piano per {u.chiave}/{fam}")
                dati = {"strato": u.dati["strato"], "famiglia": fam, "file_attesi": nomi}
                if "forme" in u.dati:
                    dati.update(forme=[u.dati["forme"][i]], gq=u.dati["gq"])
                nuove.append(PG.Unita(f"{u.chiave}-{fam}", dati))
        unita = nuove
    nomi_codice = ("cardan_guardia_gpu.py", "cardan_esperti_guardia_gpu.py",
                  "cardan_unita_guardia_gpu.py", "cardan_fucina_guardia_gpu.py",
                  "pianificatore_guardia_gpu.py", "pianificatore_misurato_gpu.py",
                  "cardan_obiettivo_gpu.py", "cardan_telemetria_gpu.py", "cardan_classi_gpu.py")
    codici = {n: PG.sha_file(QUI / n) for n in nomi_codice}
    piano["config"]["correzione_gpu"] = correzione_gpu
    piano["config"]["codici_guardia"] = codici
    from cardan_classi_gpu import collega
    collega(piano["config"], unita)
    piano["factory"] = "cardan_unita_guardia_gpu:crea"
    piano["unita"] = [dataclasses.asdict(u) for u in unita]
    piano["ricetta_sha256"] = PG.firma_json({
        "ricetta_base": piano["ricetta_sha256"], "correzione_gpu": correzione_gpu,
        "codici_guardia": codici, "unita_esperti": "strato-famiglia"})
    del piano["firma"]
    piano["firma"] = PG.firma_json(piano)
    return piano, unita


def raccogli(root, piano, unita):
    if piano["config"]["fase"] != "esperti":
        return _raccogli(root, piano, unita)
    root = Path(root)
    dest = root / "pubblicato"
    descrizione = PG.Unita("pubblicato", {"unita": [u.chiave for u in unita]})
    for u in unita:
        if not PG.completa(root / "unita" / u.chiave, piano["firma"], u):
            raise RuntimeError(f"manca unita' {u.chiave}")
    if PG.completa(dest, piano["firma"], descrizione):
        return dest
    stage = root / f".pubblicazione-{uuid.uuid4().hex}"
    (stage / "tensori").mkdir(parents=True)
    records, visti = [], set()
    for u in unita:
        d = root / "unita" / u.chiave
        rr = [json.loads(r) for r in (d / "registro.jsonl").read_text().splitlines() if r]
        chiave = (u.dati["strato"], u.dati["famiglia"])
        if len(rr) != 1 or (rr[0]["strato"], rr[0]["famiglia"]) != chiave or chiave in visti:
            raise RuntimeError(f"registro duplicato/incompleto: {u.chiave}")
        visti.add(chiave)
        files = list((d / "tensori").glob("*.npy"))
        if {f.name for f in files} != set(u.dati["file_attesi"]):
            raise RuntimeError(f"piani mancanti/estranei: {u.chiave}")
        for f in files:
            os.link(f, stage / "tensori" / f.name)
        records.extend(rr)
    with open(stage / "registro.jsonl", "x") as out:
        for r in sorted(records, key=lambda r: (r["strato"], r["famiglia"])):
            out.write(json.dumps(r, allow_nan=False) + "\n")
    PG.pubblica_unita(stage, dest, piano["firma"], descrizione, {})
    return dest


def esegui_con_ripresa(unita, factory, config, root, firma, posti, budget, *, tentativi=2):
    """Fail-fast del coordinatore storico; un solo worker al secondo tentativo."""
    if tentativi < 1:
        raise ValueError("--tentativi-lavoratori deve essere positivo")
    for i in range(tentativi):
        try:
            return _esegui(unita, factory, config, root, firma, posti, budget)
        except E.ErroreLavoratore as e:
            if i + 1 == tentativi:
                raise
            print(json.dumps({"guardia": "RIPRESA_DOPO_WORKER_MORTO",
                              "tentativo": i + 2, "errore": str(e),
                              "politica": "una sola unita' in volo; risultati completi conservati"}),
                  file=sys.stderr, flush=True)
            PG.guardia_corse()
            PG.guardia_host(*PG.misura_host())
            device = next(d for d, n in posti.items() if n > 0)
            posti = {d: int(d == device) for d in posti}


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tentativi-lavoratori", type=int, default=2)
    ap.add_argument("--correzione-gpu", action="store_true",
                    default=os.environ.get("FORGIA_CORREZIONE_GPU") == "1")
    a, resto = ap.parse_known_args(argv)
    if a.tentativi_lavoratori < 1:
        ap.error("--tentativi-lavoratori deve essere positivo")
    if "-h" in resto or "--help" in resto:
        print(ap.format_help())
    originale_prepara, originale_raccogli, originale_esegui = F.prepara, F.raccogli, PG.esegui
    try:
        F.prepara = lambda arg: prepara(arg, a.correzione_gpu)
        F.raccogli = raccogli
        PG.esegui = lambda *arg, **kw: esegui_con_ripresa(
            *arg, tentativi=a.tentativi_lavoratori, **kw)
        return F.main(resto)
    finally:
        F.prepara, F.raccogli, PG.esegui = originale_prepara, originale_raccogli, originale_esegui


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as e:
        print(f"ERRORE fucina guardia GPU: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
