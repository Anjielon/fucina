#!/usr/bin/env python3
"""LA DISTRIBUZIONE DEL MARGINE AL CONFINE DEL TOP-K — senza GPU.

Il contributo piu' difendibile del lavoro, e finora ne avevamo solo la
mediana (0.0003). La letteratura ha il concetto, il nome e la MEDIA; quello
che manca e' la distribuzione — ed e' la parte che conta, perche' una media su
una distribuzione asimmetrica non e' la mediana e non puo' dire che meta' dei
token stanno a 0.0003, un ordine di grandezza sotto la quota uniforme 1/256.

Due lavori definiscono esattamente questa grandezza e non la misurano mai:
ReMoE (arXiv 2605.27081, App. C.2) come premessa non misurata di un lemma di
stabilita', in spazio delle PROBABILITA' come noi; e arXiv 2608.11212 in
spazio dei LOGIT, ridotta a un singolo numero di rilevabilita'.

Si calcola dalle catture gia' su disco: nessuna GPU, nessun modello caricato.
"""
from __future__ import annotations

import glob
import re

import numpy as np

CATTURA = "/tmp/rotte_rif"          # piano spento: il modello come viene servito
N_USED, E = 8, 256


def carica(L: int) -> np.ndarray | None:
    p = glob.glob(f"{CATTURA}/*-{L}.f32")
    if not p:
        return None
    a = np.fromfile(p[0], dtype=np.float32)
    if a.size == 0 or a.size % E:
        return None
    return a.reshape(-1, E).astype(np.float64)


def main() -> None:
    tutti = []
    per_strato = {}
    for L in range(64):
        A = carica(L)
        if A is None:
            continue
        ord_ = np.sort(A, axis=1)[:, ::-1]
        g = ord_[:, N_USED - 1] - ord_[:, N_USED]      # k-esimo meno (k+1)-esimo
        per_strato[L] = g
        tutti.append(g)
    if not tutti:
        print("⛔ nessuna cattura delle probabilita' del router")
        return
    g = np.concatenate(tutti)
    quota = 1.0 / E

    print(f"MARGINE AL CONFINE DEL TOP-K — {len(per_strato)} strati, "
          f"{len(g):,} token·strato\n")
    print(f"  quota uniforme di un esperto (1/{E}) = {quota:.6f}\n")
    print("  percentile        margine   quante volte piu' stretto della quota")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        v = float(np.percentile(g, q))
        print(f"  {q:3d}%          {v:.6f}   {quota / max(v, 1e-12):8.1f}×")
    print(f"\n  media           {g.mean():.6f}   {quota / max(g.mean(), 1e-12):8.1f}×")
    print(f"  mediana         {np.median(g):.6f}   "
          f"{quota / max(np.median(g), 1e-12):8.1f}×")
    print(f"\n  ⇒ media / mediana = {g.mean() / max(np.median(g), 1e-12):.2f}: "
          "la distribuzione e' asimmetrica,")
    print("    e riportare la media al posto della mediana la fa sembrare"
          f" {g.mean() / max(np.median(g), 1e-12):.1f} volte")
    print("    piu' larga di quanto sia per meta' dei token.")

    # quanto piccolo e' "piccolo": frazione sotto soglie di rilievo pratico
    print("\n  FRAZIONE DI TOKEN SOTTO UNA SOGLIA")
    for s in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3):
        print(f"    margine < {s:.0e} : {float((g < s).mean()) * 100:6.2f}%")
    # un ULP di bf16 vicino a 0.004 vale ~1/256 * 2^-8 ≈ 1.5e-5
    ulp = quota * 2 ** -8
    print(f"\n    sotto un ULP di bf16 alla quota tipica ({ulp:.2e}): "
          f"{float((g < ulp).mean()) * 100:.2f}% dei token")
    print("    ⇒ e' la meta' quantitativa di cio' che sglang #35916 afferma")
    print("      qualitativamente sulla stessa topologia, su pesi sintetici.")

    print("\n  PER STRATO (mediana del margine)")
    ls = sorted(per_strato)
    for i in range(0, len(ls), 5):
        print("    " + "  ".join(
            f"{L:2d}:{np.median(per_strato[L]):.5f}" for L in ls[i:i + 5]))


if __name__ == "__main__":
    main()
