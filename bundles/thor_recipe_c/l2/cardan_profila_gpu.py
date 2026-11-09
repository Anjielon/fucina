#!/usr/bin/env python3
"""Estrae un profilo dai picchi REALI di una calibrazione completata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pianificatore_gpu as PG


def rss_con_cache(misura):
    cache = misura.get("input_densi_cumulativi") or {}
    limite = cache.get("cache_h_limite_byte", 0)
    residente = cache.get("cache_h_residente_byte", 0)
    if (type(limite) is not int or type(residente) is not int or
            not 0 <= residente <= limite):
        raise ValueError("misure della cache Hessiane non valide")
    # Una calibrazione corta potrebbe non aver riempito la LRU.
    return misura["picco_rss_byte"] + limite - residente


def code_profilo(misure):
    if not any("code_campioni" in m for m in misure):
        return {}
    for m in misure:
        if (any(type(m.get(k)) is not int or m[k] < 0
                for k in ("code_compute", "code_sdma", "code_campioni")) or
                m["code_campioni"] < 1 or m["code_compute"] < 1):
            raise ValueError("misure KFD mancanti/incomplete: niente profilo di code ipotizzato")
    return {"code_compute": max(m["code_compute"] for m in misure),
            "code_sdma": max(m["code_sdma"] for m in misure),
            "code_campioni": sum(m["code_campioni"] for m in misure),
            "code_metodo": "massimi KFD campionati, inclusi caricamento e discendenti"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corsa", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.corsa)
    piano = json.loads((root / "lancio.json").read_text())
    misure = []
    disco = []
    for v in piano["unita"]:
        u = PG.Unita(**v)
        d = root / "unita" / u.chiave
        if not PG.completa(d, piano["firma"], u):
            raise RuntimeError(f"calibrazione incompleta: {u.chiave}")
        m = json.loads((d / "completa.json").read_text())
        misure.append(m["misure"])
        disco.append(sum(f["byte"] for f in m["file"].values()) + 65536)
    if not misure:
        raise RuntimeError("nessuna misura")
    profilo = {"ricetta_sha256": piano["ricetta_sha256"],
               "gpu_byte": max(m["picco_gpu_byte"] for m in misure),
               "rss_byte": max(rss_con_cache(m) for m in misure),
               "rss_picco_osservato_byte": max(m["picco_rss_byte"] for m in misure),
               "fili": piano["config"]["fili"], "disco_byte": max(disco),
               "dispositivi": sorted({m["device_tipo"] for m in misure}),
               "sorgente": str(root.resolve()), "unita_misurate": [u["chiave"] for u in piano["unita"]],
               **code_profilo(misure),
               "nota": "Picchi osservati + quota cache H non ancora residente; "
                       "non certificazione numerica; il pianificatore aggiunge 25%."}
    PG.Budget(profilo["gpu_byte"], profilo["rss_byte"], profilo["fili"], profilo["disco_byte"])
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"non sovrascrivo {out}")
    PG.scrivi_json(out, profilo)
    print(json.dumps(profilo, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
