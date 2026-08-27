#!/usr/bin/env python3
"""RIPRISTINO SELETTIVO — annulla una cura senza ricopiare il modello.

⛔ NATA IL 27/8: la cura QESC aveva peggiorato ODINO (PPL 5.81 → 6.88) e per
tornare indietro serviva ricopiare 88 GiB… con soli 41 GiB liberi. Ma la cura
aveva toccato SOLO 35 tensori piccoli (8 MB l'uno): si ripristinano quelli.

METODO (verificato sul campo):
  1. confronto i due file tensore per tensore → elenco esatto dei diversi;
  2. controllo a campione che TUTTO IL RESTO sia identico (guardia);
  3. copio i tensori buoni UNO PER VOLTA (mai piu' di 8 MB in RAM);
  4. rileggo e verifico ognuno dopo la scrittura;
  5. verifica finale: zero differenze residue.
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
sys.path.insert(0, str(Path(__file__).parent))
from gguf import GGUFReader
from patch_gguf import ChirurgoGGUF


def differenze(buono: str, corrente: str, filtro: str = "") -> list[str]:
    """I tensori in cui i due file differiscono (opzionalmente filtrati)."""
    tb = {t.name: t for t in GGUFReader(buono).tensors}
    tc = {t.name: t for t in GGUFReader(corrente).tensors}
    assert set(tb) == set(tc), "i due file non hanno gli stessi tensori"
    div = []
    for n in sorted(tb):
        if filtro and filtro not in n:
            continue
        if not np.array_equal(np.asarray(tb[n].data), np.asarray(tc[n].data)):
            div.append(n)
    return div


def guardia(buono: str, corrente: str, esclusi: set[str], campione: int = 40) -> int:
    """Quanti tensori NON attesi risultano diversi (deve essere 0)."""
    tb = {t.name: t for t in GGUFReader(buono).tensors}
    tc = {t.name: t for t in GGUFReader(corrente).tensors}
    altri = [n for n in tb if n not in esclusi]
    random.seed(0)
    ko = 0
    for n in random.sample(altri, min(campione, len(altri))):
        if not np.array_equal(np.asarray(tb[n].data)[:200000], np.asarray(tc[n].data)[:200000]):
            ko += 1
    return ko


def ripristina(buono: str, corrente: str, nomi: list[str]) -> int:
    """Copia i tensori indicati dal file buono a quello corrente, uno alla volta."""
    tb = {t.name: t for t in GGUFReader(buono).tensors}
    c = ChirurgoGGUF(corrente)
    fatti = 0
    for n in nomi:
        atteso = np.asarray(tb[n].data)
        attuale = c.leggi(n)
        if np.array_equal(atteso.reshape(-1), attuale.reshape(-1)):
            continue
        c.scrivi(n, atteso.reshape(attuale.shape))
        assert np.array_equal(c.leggi(n).reshape(-1), atteso.reshape(-1)), f"verifica fallita: {n}"
        fatti += 1
        del atteso, attuale
    return fatti


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("buono"); ap.add_argument("corrente")
    ap.add_argument("--filtro", default="", help="es. ffn_gate_inp per i soli router")
    ap.add_argument("--applica", action="store_true")
    a = ap.parse_args()
    div = differenze(a.buono, a.corrente, a.filtro)
    print(f"tensori diversi: {len(div)}")
    ko = guardia(a.buono, a.corrente, set(div))
    print(f"guardia (tensori inattesi diversi): {ko} — {'OK' if ko == 0 else 'ATTENZIONE'}")
    if a.applica and ko == 0:
        n = ripristina(a.buono, a.corrente, div)
        resto = differenze(a.buono, a.corrente, a.filtro)
        print(f"✅ ripristinati {n} · differenze residue: {len(resto)}")
    elif a.applica:
        print("⛔ guardia fallita: NON ripristino")
