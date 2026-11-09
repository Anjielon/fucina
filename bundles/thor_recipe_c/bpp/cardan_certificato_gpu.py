"""Cancello numerico comune: profilo di memoria != certificato numerico."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pianificatore_gpu as PG


def verifica(profilo, collaudo):
    pp = json.loads(Path(profilo).read_text())
    cc = json.loads(Path(collaudo).read_text())
    if (cc.get("schema") != 2 or cc.get("fase") != "piega" or cc.get("ok") is not True or
            cc.get("ricetta_sha256") != pp["ricetta_sha256"] or
            not pp.get("dispositivi") or
            not set(pp["dispositivi"]).issubset(cc.get("dispositivi", []))):
        raise ValueError("collaudo assente/fallito/obsoleto o riferito ad altro codice/device/lotto")
    if cc.get("fase") == "piega":
        appaiate = cc.get("metriche", {}).get("appaiate")
        if not appaiate or any(not math.isfinite(r["scarto_catturata"]) or
                              r["scarto_catturata"] > cc["toll_metrica"] for r in appaiate):
            raise ValueError("manca il confronto numerico appaiato per ogni tensore/famiglia")
        if cc.get("verifica_indipendente", {}).get("ok") is not True:
            raise ValueError("manca la ricostruzione indipendente delle energie dai tensori")
        audit = cc["verifica_indipendente"]
        if not audit.get("rapporto") or not audit.get("sha256"):
            raise ValueError("audit senza rapporto e SHA: il solo ok non e' una prova")
        if PG.sha_file(audit["rapporto"]) != audit["sha256"]:
            raise ValueError("audit indipendente modificato")
        # Questa verifica legge i piani: non avviarla durante corse protette
        # o sotto swap, nemmeno per validare un certificato esistente.
        PG.guardia_corse()
        PG.guardia_host(*PG.misura_host())
        from cardan_verifica_energie_gpu import verifica_rapporto
        if not verifica_rapporto(audit["rapporto"], cc["riferimento"], cc["candidato"],
                                cc["toll_metrica"])["ok"]:
            raise ValueError("audit indipendente non superato")
    if (not math.isfinite(cc["toll_byte"]) or not 0 <= cc["toll_byte"] <= 1 or
            not cc.get("tensori") or any(
                not math.isfinite(r["frazione_byte"]) or not 0 <= r["frazione_byte"] <= cc["toll_byte"]
                for r in cc["tensori"])):
        raise ValueError("almeno un tensore supera la tolleranza byte")
    return cc
