#!/usr/bin/env python3
"""Pianificazione del refinement "a seam" (ICBQ, arXiv 2608.09595 — PIANO.md
24/9 14:1x) ADATTATA dalla granularita' di ICBQ (blocco/layer singolo) alla
NOSTRA (finestra fissa W=2 layer, no-overlap — `ottimizzazione_globale_loop.
pianifica_finestre`). Funzione PURA, zero GPU/torch: si verifica da sola.

Algorithm 1 di ICBQ (outer driver) triggera un refinement ogni K unita' di
lavoro, con `w_min = max(0, chunk_start - 1)` — l'OVERLAP di UNA unita'
all'indietro e' quello che fa si' che l'ultima unita' del chunk precedente
venga ri-ottimizzata col nuovo confine fresco (la "seam"). Qui l'unita' e'
la FINESTRA (non il layer singolo: le nostre finestre sono gia' coppie fisse
[0,1],[2,3],... — l'equivalente della "coppia (theta_i,theta_i+1)" di ICBQ,
non del blocco singolo `b`), quindi CHUNK_FINESTRE e' espresso in finestre.

Esempio: 12 strati, W=2 -> 6 finestre [0,1]..[10,11]. CHUNK_FINESTRE=2:
- trigger dopo la finestra 1 (indice 1, le finestre 0 e 1 fatte): range [0,2)
  (w_min=max(0,0-1)=0 — nessun overlap al primissimo chunk, come in ICBQ)
- trigger dopo la finestra 3 (indici 2,3 fatte): range [1,4) — INCLUDE la
  finestra 1 (l'ultima del chunk precedente: la "seam"), overlap=1 finestra
- trigger dopo la finestra 5 (ultima, indici 4,5 fatte): range [3,6)
"""
from __future__ import annotations


def pianifica_refine(n_finestre: int, chunk_finestre: int) -> "list[tuple[int, int]]":
    """Ritorna una lista di range [w_min_idx, w_max_idx) — INDICI DI FINESTRA
    (non layer) — uno per ogni trigger di refinement, nello stesso ordine in
    cui vanno eseguiti (il chunk-start avanza SOLO dopo ogni trigger, mai
    prima: stesso `b_cs` di Algorithm 1)."""
    assert n_finestre > 0 and chunk_finestre > 0
    piano: list[tuple[int, int]] = []
    chunk_start = 0
    for i in range(n_finestre):
        raggiunto_k = (i + 1) - chunk_start >= chunk_finestre
        ultima = i == n_finestre - 1
        if raggiunto_k or ultima:
            w_min = max(0, chunk_start - 1)
            w_max = i + 1
            piano.append((w_min, w_max))
            chunk_start = i + 1
    return piano


def _test() -> bool:
    ok = True

    # 6 finestre, chunk=2 -> tre trigger, overlap=1 dal secondo in poi
    r = pianifica_refine(6, 2)
    atteso = [(0, 2), (1, 4), (3, 6)]
    ok &= r == atteso
    print(f"6 finestre, chunk=2: {r}  atteso {atteso}  {'OK' if r == atteso else 'FALLITO'}")

    # 6 finestre, chunk=3 -> due trigger
    r = pianifica_refine(6, 3)
    atteso = [(0, 3), (2, 6)]
    ok &= r == atteso
    print(f"6 finestre, chunk=3: {r}  atteso {atteso}  {'OK' if r == atteso else 'FALLITO'}")

    # 6 finestre, chunk=1 -> refine di OGNI finestra appena fatta (caso limite,
    # overlap=1 sempre dal secondo trigger — il piu' aggressivo/costoso)
    r = pianifica_refine(6, 1)
    atteso = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    ok &= r == atteso
    print(f"6 finestre, chunk=1: {r}  atteso {atteso}  {'OK' if r == atteso else 'FALLITO'}")

    # 1 finestra sola -> un solo trigger, l'intero range
    r = pianifica_refine(1, 4)
    atteso = [(0, 1)]
    ok &= r == atteso
    print(f"1 finestra, chunk=4: {r}  atteso {atteso}  {'OK' if r == atteso else 'FALLITO'}")

    # ogni range deve coprire ESATTAMENTE fino all'ultima finestra fatta finora
    # (nessun buco, nessuna finestra mai refinata "in anticipo")
    for n, k in [(12, 2), (12, 3), (12, 4), (7, 3), (9, 4)]:
        piano = pianifica_refine(n, k)
        assert piano[-1][1] == n, f"n={n} k={k}: l'ultimo range non arriva a {n}: {piano}"
        for (a, b) in piano:
            assert 0 <= a < b <= n, f"n={n} k={k}: range invalido {(a, b)}"
    print(f"copertura/validita' range: OK ({len([(12,2),(12,3),(12,4),(7,3),(9,4)])} casi)")

    print("✅ TUTTI I TEST PASSATI" if ok else "⛔ ALCUNI TEST FALLITI")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)
