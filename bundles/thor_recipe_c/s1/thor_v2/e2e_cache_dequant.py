#!/usr/bin/env python3
"""e2e_cache_dequant — cache su disco delle tabelle dequantizzate dello studente (embedding e testa), 1/10/2026, ramo
tt5-ottimizzato (AUDIT TT5 cura 9). File NUOVO.

## Perche'
`Studente._somma_piani` dequantizza token_embd e output (Q5_K + piani ternari 2/3 in R2B3m) con numpy a un filo: 11 min
47 s all'avvio di TT5 il 1/10, 21 min il 30/9 [M, registro]. Embedding e testa NON si allenano: a ogni avvio/ripresa il
risultato e' lo stesso array. Qui lo si scrive una volta (fp32 .npy) e si rilegge.

## Al bit
- La chiave e' lo sha256 dei BYTE GREZZI di tutti i piani della tabella (nome, tipo, forma, dati del GGUF) + lo sha256 di
  `dequant_gguf_torch.py` (il codice che dequantizza) + una versione di questa ricetta (somma dei piani nell'ordine del
  motore, fp32). Un GGUF diverso, un piano in piu' o un dequantizzatore cambiato = chiave diversa = si ricalcola.
- Il file .npy conserva dtype, forma e byte; accanto un .json con lo sha256 dei byte dell'array. Alla lettura lo sha si
  ricontrolla: se non torna (file rovinato) si ricalcola e si riscrive. Scrittura atomica (.parziale, fsync, rename).
- Spento (variabile E2E_CACHE_DEQUANT assente o vuota) = `_somma_piani` di sempre, identico.

Prova: `e2e_test_cache_dequant.py` (sha256 dell'array dalla cache == quello calcolato, byte per byte, per token_embd e
output di R2B3m).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

VERSIONE = "somma_piani_fp32_v1"
_QUI = Path(__file__).resolve().parent


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for pezzo in iter(lambda: f.read(1 << 24), b""):
            h.update(pezzo)
    return h.hexdigest()


def sha_array(a: np.ndarray) -> str:
    """sha256 di dtype, forma e byte (ordine C) dell'array."""
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{tuple(a.shape)}|".encode())
    c = np.ascontiguousarray(a)
    mv = memoryview(c.reshape(-1).view(np.uint8))
    for i in range(0, len(mv), 1 << 26):
        h.update(mv[i:i + (1 << 26)])
    return h.hexdigest()


def chiave(tens: dict, piani: list) -> str:
    """sha256 dei byte grezzi dei piani (nell'ordine della somma) + codice del dequantizzatore + versione."""
    h = hashlib.sha256()
    h.update(VERSIONE.encode())
    h.update(_sha_file(_QUI / "dequant_gguf_torch.py").encode())
    for nome in piani:
        t = tens[nome]
        raw = np.ascontiguousarray(np.asarray(t.data))
        h.update(f"|{nome}|{t.tensor_type.name}|{tuple(int(x) for x in raw.shape)}|{raw.dtype.str}|".encode())
        mv = memoryview(raw.reshape(-1).view(np.uint8))
        for i in range(0, len(mv), 1 << 26):
            h.update(mv[i:i + (1 << 26)])
    return h.hexdigest()


def somma_piani(tens: dict, nome_g: str, piani: list, cartella: Path, calcola: Callable[[], np.ndarray],
                log: Callable = print) -> np.ndarray:
    """L'array di `calcola()` (la dequantizzazione di sempre), dalla cache se c'e' ed e' integra."""
    cartella.mkdir(parents=True, exist_ok=True)
    k = chiave(tens, piani)
    # 1/10: nomi espliciti (nome_g contiene un punto: with_suffix avrebbe tagliato "weight-<chiave>")
    p_npy, p_js = cartella / f"{nome_g}-{k[:24]}.npy", cartella / f"{nome_g}-{k[:24]}.json"
    if p_npy.exists() and p_js.exists():
        try:
            meta = json.loads(p_js.read_text())
            if meta.get("chiave") == k:
                a = np.load(p_npy, allow_pickle=False)
                if sha_array(a) == meta.get("sha256_array"):
                    log(f"cache dequant: {nome_g} letto da {p_npy.name} ({a.nbytes / 2 ** 30:.2f} GiB, sha al bit)")
                    return a
                log(f"cache dequant: {p_npy.name} con sha diverso dal registro: ricalcolo")
        except (OSError, ValueError) as e:
            log(f"cache dequant: {p_npy.name} illeggibile ({e!r}): ricalcolo")
    a = calcola()
    sha = sha_array(a)
    tmp = p_npy.with_name(p_npy.name + ".parziale")
    with open(tmp, "wb") as f:
        np.save(f, a, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p_npy)
    tmp_j = p_js.with_name(p_js.name + ".parziale")
    tmp_j.write_text(json.dumps({"chiave": k, "nome": nome_g, "piani": list(piani), "dtype": a.dtype.str,
                                 "forma": list(a.shape), "sha256_array": sha, "versione": VERSIONE}, indent=1))
    os.replace(tmp_j, p_js)
    log(f"cache dequant: {nome_g} scritto in {p_npy.name} ({a.nbytes / 2 ** 30:.2f} GiB)")
    return a
