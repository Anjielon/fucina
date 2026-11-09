#!/usr/bin/env python3
"""e2e_cancello_fase_v — cancello GPU delle cure della fase V (2/10/2026, ramo tt5-fase-v). File NUOVO.

Cosa prova. Le cure V1-V3 (`--pv-dd-solo-salti`, `--pv-pm-ordine-strati`, `--pv-golosa-pigra`, FASE_V.md) promettono
le STESSE decisioni del passo V con meno lavoro. Sulla CPU lo provano e2e_test_fase_v.py (giocattoli, motore torch);
qui si prova sulla GPU vera, nel passo V vero di TT5 (motore triton2, densi piegati, riuso, GDN veloce): lo stesso
passo V di decisione si esegue DUE volte dallo stesso stato,
    1. SPENTO (le tre cure spente, come la corsa di oggi)    -> esito, registro dei salti tenuti, stato V-Bop, byte
    2. stato rimesso AL BIT (V-Bop, buffer del passo, regione di fiducia, registri, byte: impronta verificata)
    3. ACCESO (le tre cure accese)                           -> stesse grandezze
e si confrontano: esito (tolti orologio e contatori di lavoro), righe del registro tenuto, sha256 dello stato V-Bop,
impronta dei byte. Con --pv-tempi-v (sempre acceso nel cancello) il JSON porta i secondi per sotto-fase dei due giri.

Come si accende: variabile d'ambiente, nessun campo di config (la ripresa non cambia):
    E2E_CANCELLO_V=<file JSON dell'esito>
    E2E_CANCELLO_V_MODO=corsa (default) | esci
  - corsa: il cancello scatta al PRIMO passo V di decisione della corsa (con --pv-passi-accumulo 4: il prossimo passo
    multiplo di 4) e la corsa CONTINUA. Uguali: si prosegue con lo stato dell'acceso (= spento). Diversi: si rimette lo
    stato dello SPENTO (quello di oggi, salvato alla fine del giro 1) e le cure si spengono per il resto del processo.
    Costo: un passo V di decisione in piu' (il giro spento), una volta.
  - esci: in una COPIA della cartella della corsa (mai quella vera): il primo passo V diventa di decisione, cancello,
    JSON, uscita (codice 0 uguali, 3 diversi). Nessuna ripresa scritta.
Il controllo del determinismo (spento due volte) si chiede con E2E_CANCELLO_V_CONTROLLO=1 (un giro in piu').
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch

_FATTO = {"scattato": False, "cure_spente": False}
CURE = ("pv_dd_solo_salti", "pv_pm_ordine_strati", "pv_golosa_pigra")
# chiavi dell'esito che dicono QUANTO lavoro, non COSA si e' deciso (escluse dal confronto)
_LAVORO = ("tempi_v", "riuso", "dd_saltati")


def attivo() -> bool:
    return bool(os.getenv("E2E_CANCELLO_V", "").strip()) and not _FATTO["scattato"]


def modo() -> str:
    m = os.getenv("E2E_CANCELLO_V_MODO", "corsa").strip() or "corsa"
    if m not in ("corsa", "esci"):
        raise ValueError(f"E2E_CANCELLO_V_MODO={m!r}: corsa | esci")
    return m


def cure_spente() -> bool:
    """Dopo un cancello ROSSO in modo corsa: le cure restano spente per il resto del processo."""
    return _FATTO["cure_spente"]


# ─────────────────────────────── istantanea / ripristino dello stato del passo V ───────────────────────────────
def _istantanea(pv, reg_f, impronta: Callable) -> dict:
    sv = pv.stato
    return {
        "bop": {k: {n: getattr(z, n).clone() for n in ("cand", "m", "r", "H", "eta", "sm", "sr", "sH")}
                for k, z in sv.bop_.items()},
        "buf": {k: {n: getattr(b, n).clone() for n in ("cand", "prop", "S", "Q", "H") if getattr(b, n) is not None}
                for k, b in sv.buffer_.items()},
        "int": {n: copy.deepcopy(getattr(sv, n)) for n in ("passi_bop", "n_verifica", "n_propose", "n_curvatura", "modo",
                                                           "passi_nella_finestra", "n_curvatura_passo")},
        "reg_f": copy.deepcopy(reg_f.__dict__),
        "n_registri": len(pv.registri),
        "bop_prima": pv._bop_prima,
        "impronta": impronta(),
    }


def _ripristina(pv, reg_f, ist: dict, impronta: Callable, reg_tenuto) -> None:
    import e2e_pv_codici as PV
    if reg_tenuto is not None and len(reg_tenuto):
        PV.annulla(pv.fam, reg_tenuto)                 # i byte tornano quelli di partenza (registro esatto)
    sv = pv.stato
    for k, d in ist["bop"].items():
        for n, t in d.items():
            getattr(sv.bop_[k], n).copy_(t)
    for k, d in ist["buf"].items():
        for n, t in d.items():
            getattr(sv.buffer_[k], n).copy_(t)
    for n, v in ist["int"].items():
        setattr(sv, n, copy.deepcopy(v))
    sv.famiglie_ricentro = None
    sv.salta_dd_superflui = False
    reg_f.__dict__.clear()
    reg_f.__dict__.update(copy.deepcopy(ist["reg_f"]))
    del pv.registri[ist["n_registri"]:]
    pv._bop_prima = ist["bop_prima"]
    imp = impronta()
    if imp != ist["impronta"]:
        raise RuntimeError("cancello fase V: i byte non sono tornati quelli di partenza dopo il ripristino")


def _sha_bop(pv) -> str:
    h = hashlib.sha256()
    for k in sorted(pv.stato.bop_):
        z = pv.stato.bop_[k]
        for n in ("cand", "m", "r", "H", "eta", "sm", "sr", "sH"):
            t = getattr(z, n).detach().contiguous().cpu()
            h.update(f"{k}.{n}".encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else t.numpy().tobytes())
    return h.hexdigest()


def _decisione(out: dict) -> str:
    """L'esito senza le chiavi di LAVORO, in JSON canonico (i float con repr: confronto AL BIT)."""
    d = {k: v for k, v in out.items() if k not in _LAVORO}
    return json.dumps(d, sort_keys=True, default=repr)


def allinea_pigra(out_spento: dict, out_acceso: dict) -> dict:
    """V3 (golosa pigra): nelle voci delle dosi che l'acceso ha rifiutato SENZA misurare l'obiettivo, lo spento ha il
    valore misurato. Copia dell'esito spento con quelle voci scritte come le scrive l'acceso (obiettivo None,
    obiettivo_saltato True) — solo se lo spento le aveva davvero rifiutate (tenuta False), altrimenti invariato
    (e il confronto sara' rosso, come deve)."""
    da, db = out_spento.get("dosi"), out_acceso.get("dosi")
    if not da or not db or len(da) != len(db):
        return out_spento
    o = copy.deepcopy(out_spento)
    for x, y in zip(o["dosi"], db):
        if y.get("obiettivo_saltato") and not x.get("tenuta", True):
            x["obiettivo"] = None
            x["obiettivo_saltato"] = True
    return o


def _giro(chiama: Callable, ca, pv, nome: str) -> dict:
    t0 = time.time()
    n0 = len(pv.registri)
    pv.tempi = {}
    try:
        out = chiama(ca)
    finally:
        tempi, pv.tempi = pv.tempi, None
    reg = pv.registri[-1] if len(pv.registri) == n0 + 1 else None
    righe = reg.righe.tobytes() if reg is not None else b""
    return {"nome": nome, "out": out, "decisione": _decisione(out), "reg": reg,
            "sha_registro": hashlib.sha256(righe).hexdigest(), "salti_registro": 0 if reg is None else len(reg),
            "sha_bop": _sha_bop(pv), "secondi": time.time() - t0, "tempi": tempi,
            "riuso": out.get("riuso"), "dd_saltati": out.get("dd_saltati")}


def esegui(chiama: Callable, ca, pv, reg_f, impronta: Callable, log: Callable) -> dict:
    """`chiama(ca_x)` = il passo V di decisione (`_passo_v_dosi`) con la config `ca_x`. Ritorna l'esito da usare."""
    _FATTO["scattato"] = True
    dest = Path(os.environ["E2E_CANCELLO_V"])
    m = modo()
    ca_off = dataclasses.replace(ca, **{c: False for c in CURE}, pv_tempi_v=True)
    ca_on = dataclasses.replace(ca, pv_dd_solo_salti=True, pv_pm_ordine_strati=bool(ca.pv_punto_medio),
                                pv_golosa_pigra=bool(ca.pv_pm_golosa), pv_tempi_v=True)
    log(f"CANCELLO FASE V ({m}): giro SPENTO, poi stato rimesso al bit, poi giro ACCESO "
        f"({', '.join(c for c in CURE if getattr(ca_on, c))}); esito in {dest}")
    ist0 = _istantanea(pv, reg_f, impronta)
    giri = []
    controllo = os.getenv("E2E_CANCELLO_V_CONTROLLO", "0") == "1"
    a = _giro(chiama, ca_off, pv, "spento")
    giri.append(a)
    ist_off = _istantanea(pv, reg_f, impronta)          # lo stato di OGGI dopo il passo V (per il rosso in corsa)
    _ripristina(pv, reg_f, ist0, impronta, a["reg"])
    if controllo:
        c = _giro(chiama, ca_off, pv, "spento_bis")
        giri.append(c)
        _ripristina(pv, reg_f, ist0, impronta, c["reg"])
    b = _giro(chiama, ca_on, pv, "acceso")
    giri.append(b)
    uguali = {"decisione": _decisione(allinea_pigra(a["out"], b["out"])) == b["decisione"], "registro": a["sha_registro"] == b["sha_registro"],
              "bop": a["sha_bop"] == b["sha_bop"], "impronta_byte": impronta() == ist_off["impronta"]}
    ok = all(uguali.values())
    esito = {"data": time.strftime("%Y-%m-%dT%H:%M:%S"), "modo": m, "ok": ok, "uguali": uguali,
             "controllo_determinismo": (giri[1]["decisione"] == a["decisione"] and giri[1]["sha_bop"] == a["sha_bop"])
             if controllo else None,
             "giri": [{k: v for k, v in g.items() if k not in ("out", "reg", "decisione")} for g in giri],
             "risparmio_s": a["secondi"] - b["secondi"],
             "decisione_spento": json.loads(a["decisione"]), "decisione_acceso": None if ok else json.loads(b["decisione"])}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".parziale")
    tmp.write_text(json.dumps(esito, indent=1, default=repr))
    os.replace(tmp, dest)
    log(f"CANCELLO FASE V: {'VERDE' if ok else 'ROSSO'} {uguali}; spento {a['secondi']:.0f} s, acceso "
        f"{b['secondi']:.0f} s (risparmio {a['secondi'] - b['secondi']:.0f} s); {dest}")
    if m == "esci":
        log("CANCELLO FASE V: modo esci, nessuna ripresa scritta")
        sys.stdout.flush()
        os._exit(0 if ok else 3)
    if ok:
        return b["out"]
    # ROSSO in corsa: si torna allo stato dello SPENTO (quello che la corsa di oggi avrebbe avuto) e cure spente
    _ripristina(pv, reg_f, ist0, impronta, b["reg"])
    import e2e_pv_codici as PV
    if a["reg"] is not None:
        PV.rigioca(pv.fam, [a["reg"]])
    sv = pv.stato
    for k, d in ist_off["bop"].items():
        for n, t in d.items():
            getattr(sv.bop_[k], n).copy_(t)
    for k, d in ist_off["buf"].items():
        for n, t in d.items():
            getattr(sv.buffer_[k], n).copy_(t)
    for n, v in ist_off["int"].items():
        setattr(sv, n, copy.deepcopy(v))
    reg_f.__dict__.clear()
    reg_f.__dict__.update(copy.deepcopy(ist_off["reg_f"]))
    if a["reg"] is not None:
        pv.registri.append(a["reg"])
    pv._bop_prima = ist_off["bop_prima"]
    if impronta() != ist_off["impronta"]:
        raise RuntimeError("cancello fase V ROSSO: lo stato dello spento non e' tornato (impronta dei byte)")
    _FATTO["cure_spente"] = True
    log("CANCELLO FASE V ROSSO: stato dello SPENTO rimesso, cure V1-V3 spente per il resto del processo")
    return a["out"]
