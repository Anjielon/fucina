"""Classi strutturali dei carichi; non sono certificati numerici o di picco.

I tempi si trasferiscono solo fra unita' della stessa ricetta e struttura.
Contesti/device senza campioni restano vietati dal modello misurato.
"""
from __future__ import annotations

import copy
import hashlib
import json


def descrittore(fase, dati):
    if fase == "densi":
        return {"fase": fase, "forma": dati["forma"], "lotto": len(dati["nomi"]),
                "operatori": sorted(n.split(".")[-2] for n in dati["nomi"])}
    if fase == "esperti" and "forme" in dati:
        return {"fase": fase, "forme": dati["forme"],
                "famiglia": dati.get("famiglia", "gate+up"),
                "gq": dati["gq"]}
    if fase == "a1" and "forma_h" in dati:
        return {"fase": fase, "forma_h": dati["forma_h"],
                "tipo_strato": dati["tipo_strato"]}
    # Piani vecchi privi delle forme: nessuna equivalenza inventata.
    return None


def collega(config, unita):
    fase = config["fase"]
    classi, descrizioni = {}, {}
    for u in unita:
        d = descrittore(fase, u.dati)
        if d is None:
            classe = u.chiave
        else:
            testo = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
            classe = fase + "-" + hashlib.sha256(testo.encode()).hexdigest()
            descrizioni[classe] = d
        classi[u.chiave] = classe
    config["pianificazione"] = {**config.get("pianificazione", {}),
                               "classi": classi, "descrittori_classi": descrizioni}


def unisci(config, rendimenti):
    """Conserva le classi del piano anche nelle calibrazioni senza --rendimenti."""
    piano = copy.deepcopy(config.get("pianificazione", {}))
    classi = piano.get("classi", {})
    osservate = rendimenti.get("classi", {})
    if any(k in classi and classi[k] != c for k, c in osservate.items()):
        raise ValueError("classi dei rendimenti diverse dal piano")
    return {**piano, **rendimenti, "classi": {**osservate, **classi}}


def opzioni_rendimenti(fase, esperti=None, densi=None):
    nome = {"esperti": esperti, "densi": densi}[fase]
    return ["--rendimenti", nome] if nome else []
