#!/usr/bin/env python3
"""Misura esplicita isolato/compresenza su massimo due unita' REALI.

Nessuna stima sintetica e nessuna prova con corse protette. Produce un JSON
per --rendimenti della stessa ricetta. La misura non e' un collaudo numerico.
L'out nuovo conserva tutti i risultati e i registri delle prove.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import cardan_obiettivo_gpu as O
import pianificatore_gpu as PG


def esporta(corse, out):
    tempi, avvii, firme_device = [], {}, {}
    ricetta = None
    sorgenti = []
    classi = {}
    if len({Path(p).resolve() for p in corse}) != len(corse):
        raise ValueError("stessa corsa ripetuta: duplicazione di campioni")
    for root in map(Path, corse):
        piano = json.loads((root / "piano.json").read_text())
        lancio = json.loads((root / "lancio.json").read_text())
        if ricetta is not None and ricetta != lancio["ricetta_sha256"]:
            raise ValueError("corse di ricette diverse")
        ricetta = lancio["ricetta_sha256"]
        h = json.loads((root / "rendimenti_misurati.json").read_text())
        if h["firma"] != piano["firma"] or piano["firma"] != lancio["firma"]:
            raise ValueError("rendimenti/piano non coerenti")
        for v in piano["unita"]:
            u = PG.Unita(**v)
            if not PG.completa(root / "unita" / u.chiave, piano["firma"], u):
                raise ValueError(f"calibrazione incompleta: {root}/{u.chiave}")
            c = piano["config"].get("pianificazione", {}).get("classi", {}).get(u.chiave, u.chiave)
            if u.chiave in classi and classi[u.chiave] != c:
                raise ValueError("classi diverse fra le prove")
            classi[u.chiave] = c
        eventi = [json.loads(line) for line in (root / "decisioni.jsonl").read_text().splitlines()]
        if any(r["evento"] == "DRAIN" for r in eventi):
            raise ValueError("corsa in DRAIN: rendimenti non certificabili")
        osservati = []
        devices = {}
        for r in eventi:
            if r["evento"] == "avvio":
                devices[r["pid"]] = r["device"]
            if r["evento"] != "completamento" or not r["campione_ammesso"]:
                continue
            if r["chiave"] not in {u["chiave"] for u in piano["unita"]}:
                raise ValueError("completamento non presente nel piano")
            d = devices[r["pid"]]
            tipo = r["misure"]["device_tipo"]
            if str(d) in firme_device and firme_device[str(d)] != tipo:
                raise ValueError("identita' device diversa fra le prove")
            firme_device[str(d)] = tipo
            osservati.append({"contesto": r["contesto"], "device": d,
                              "classe": classi[r["chiave"]], "secondi": r["wall_s"]})
        # Legge gli eventi, NON somma snapshot contenenti baseline gia' importate.
        tempi.extend(osservati)
        avvii.update(h.get("avvio_s", {}))
        sorgenti.append({"path": str(root.resolve()), "eventi_sha256": PG.sha_file(
            root / "decisioni.jsonl"), "campioni": len(osservati)})
    if not tempi:
        raise ValueError("nessun completamento ammissibile misurato")
    r = {"schema": 1, "ricetta_sha256": ricetta,
         "dispositivi": sorted(set(firme_device.values())), "device_firme": firme_device,
         "tempi": O.Tempi(tempi).esporta(), "avvio_s": avvii, "classi": classi,
         "sorgenti": sorgenti, "tipo": "completamenti_wall_ondate",
         "nota": "Solo ricetta/device/classi misurati; non certifica numerica o corse protette."}
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"non sovrascrivo {out}")
    PG.scrivi_json(out, r)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--piano")
    ap.add_argument("--profilo")
    ap.add_argument("--out", required=True, help="directory nuova delle prove, oppure JSON con --esporta")
    ap.add_argument("--gpu", default="all")
    ap.add_argument("--processi-per-gpu", type=int, default=2, choices=(1, 2))
    ap.add_argument("--esporta", nargs="+", help="solo estrazione da corse completate, nessuna GPU")
    ap.add_argument("--unita", help="una o due chiavi del piano, separate da virgola")
    a = ap.parse_args(argv)
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    if a.esporta:
        esporta(a.esporta, a.out)
        return 0
    if not a.piano or not a.profilo or not a.unita:
        ap.error("misura richiede --piano --profilo --unita (massimo due)")
    PG.guardia_corse()
    for _ in range(2):
        PG.guardia_host(*PG.misura_host())
    piano = json.loads(Path(a.piano).read_text())
    profilo = json.loads(Path(a.profilo).read_text())
    if profilo["ricetta_sha256"] != piano["ricetta_sha256"]:
        raise ValueError("profilo di altra ricetta")
    chiavi = a.unita.split(",")
    if len(set(chiavi)) != len(chiavi) or not 1 <= len(chiavi) <= 2:
        raise ValueError("richieste una o due unita' distinte")
    unita = [u for u in piano["unita"] if u["chiave"] in chiavi]
    if len(unita) != len(chiavi):
        raise ValueError("chiavi non presenti nel piano")
    root = Path(a.out)
    if root.exists():
        raise FileExistsError(f"non sovrascrivo {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    disco = profilo["disco_byte"] * len(unita)
    if shutil.disk_usage(root.parent).free < 8 * disco + PG.GIB:
        raise RuntimeError("spazio insufficiente per calibrazioni + riserva")
    import torch
    ids = PG.seleziona_gpu(a.gpu, torch.cuda.device_count())
    if any(PG.tipo_device(torch, d) not in profilo["dispositivi"] for d in ids):
        raise ValueError("profilo non misurato sui device richiesti")
    root.mkdir()
    corse = []

    def misura(nome, gpu, n, uu):
        PG.guardia_corse()
        PG.guardia_host(*PG.misura_host())
        d = root / nome
        d.mkdir()
        p = {**piano, "unita": uu}
        p["firma"] = PG.firma_json({k: v for k, v in p.items() if k != "firma"})
        PG.scrivi_json(d / "lancio.json", p)
        PG.main(["--piano", str(d / "lancio.json"), "--profilo", a.profilo,
                 "--out", str(d), "--gpu", gpu, "--processi-per-gpu", str(n),
                 "--calibra-compresenza", "--attesa-massima-s", "120"])
        corse.append(d)

    for d in ids:
        # Una prova per unita'/device fornisce baseline isolate autentiche.
        for i, u in enumerate(unita):
            misura(f"isolata-gpu{d}-u{i}", str(d), 1, [u])
        if a.processi_per_gpu == 2 and len(unita) == 2:
            misura(f"coppia-gpu{d}", str(d), 2, unita)
    if len(ids) >= 2 and len(unita) == 2:
        for i, d in enumerate(ids):
            for dd in ids[i + 1:]:
                misura(f"coppia-gpu{d}-{dd}", f"{d},{dd}", 1, unita)
    r = esporta(corse, root / "rendimenti.json")
    print(json.dumps({"rendimenti": str(root / "rendimenti.json"),
                      "campioni": len(r["tempi"]), "corse": len(corse)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
