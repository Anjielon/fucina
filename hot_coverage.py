#!/usr/bin/env python3
"""QUANTO TRAFFICO COPRONO I 28 CALDI, A 256 CONTRO 512 ESPERTI?

La ricerca avversaria ha declassato l'ipotesi "5.5% di esperti e' troppo poco"
sotto l'ipotesi "la frequenza e' un cattivo criterio a 512". Ma ha anche
notato che NESSUNO ha mai pubblicato la CDF della massa instradata in funzione
del numero di esperti — ed e' una misura da pochi minuti sui due imatrix.

Se i top-28 di ODINO coprono molto meno traffico dei top-28 di Tony, la
questione della copertura si riapre; se coprono simile, muore del tutto e
resta solo la selezione.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/build-llamacpp-tq1/gguf-py")))
from gguf import GGUFReader

MODELLI = {
    "Tony 35B (256 esperti)": "/mnt/models/gguf/tony-tern/tony-imatrix.gguf",
    "ODINO 397B (512 esperti)": os.path.expanduser("~/odino-lab/imatrix/Ornith-1.5-397B-imatrix.gguf"),
}
K = 28

for nome, path in MODELLI.items():
    conteggi = {}
    try:
        for t in GGUFReader(path).tensors:
            if t.name.endswith(".ffn_gate_exps.weight.counts"):
                L = int(t.name.split(".")[1])
                conteggi[L] = np.array(t.data).astype(np.float64).ravel()
    except Exception as e:
        print(f"{nome}: ⛔ {type(e).__name__}: {e}")
        continue
    if not conteggi:
        print(f"{nome}: ⛔ nessun `counts`")
        continue
    E = len(next(iter(conteggi.values())))
    quote = []
    for L, c in sorted(conteggi.items()):
        tot = c.sum()
        if tot > 0:
            quote.append(np.sort(c)[::-1][:K].sum() / tot * 100)
    q = np.array(quote)
    print(f"\n{nome} — {len(quote)} strati, {E} esperti, top-{K} = {K/E*100:.1f}% degli esperti")
    print(f"  quota di traffico dei {K} piu' caldi:")
    print(f"    minimo {q.min():5.2f}% · mediana {np.median(q):5.2f}% · "
          f"media {q.mean():5.2f}% · massimo {q.max():5.2f}%")
    print(f"  primi 5 strati:  " + " ".join(f"{v:5.1f}" for v in q[:5]))
    print(f"  ultimi 5 strati: " + " ".join(f"{v:5.1f}" for v in q[-5:]))
