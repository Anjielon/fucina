"""Cancello fail-closed: energie dichiarate non certificano byte diversi."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pianificatore_gpu as PG


def verifica_audit(path, riferimento, candidato, checkpoint, gguf, tolleranza):
    from cardan_verifica_energie_gpu import verifica_rapporto

    report = json.loads(Path(path).read_text())
    for nome, file in (("checkpoint", checkpoint), ("gguf", gguf)):
        if report.get(nome) != PG.identita_file(file):
            raise ValueError(f"audit riferito ad altro {nome}")
    for file, identita in report.get("hessiane", {}).items():
        if PG.identita_file(file) != identita:
            raise ValueError(f"H cambiata dopo l'audit: {file}")
    if not report.get("hessiane"):
        raise ValueError("audit privo delle Hessiane di provenienza")
    if not math.isfinite(tolleranza) or tolleranza < 0:
        raise ValueError("tolleranza audit non valida")
    esito = verifica_rapporto(path, riferimento, candidato, tolleranza)
    if not esito["ok"]:
        raise ValueError("energie ricostruite dai byte fuori tolleranza")
    return esito


def accettabile(byte_diversi, frazione, toll_byte, energie_coerenti, audit):
    if byte_diversi == 0:
        return True
    return bool(energie_coerenti and frazione <= toll_byte and audit and audit.get("ok") is True)
