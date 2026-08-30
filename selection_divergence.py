#!/usr/bin/env python3
"""LA CLASSIFICA PER FREQUENZA E' QUELLA PER IMPATTO?

La forgia sceglie i 28 esperti da correggere per FREQUENZA di instradamento.
Ma il danno/beneficio atteso di correggere l'esperto e scala come
freq_e x ||W_e||^2: se le norme variano fra esperti, i 28 piu' frequenti NON
sono i 28 a massimo impatto, e la correzione va in parte ai bersagli sbagliati.

AWQ Tabella 1 mostra cosa succede col criterio sbagliato: proteggere l'1%
sbagliato e' PEGGIO che non proteggere nulla. Se la divergenza fra le due
classifiche e' piu' grande su ODINO (512 esperti) che su Tony (256), l'ipotesi
"la frequenza e' un buon criterio a 256 e cattivo a 512" e' corroborata senza
riforgiare nulla.

I file sono permutati caldi-primi (verificato 28/28 su tutti gli strati di
Tony), quindi: posizione i = i-esimo piu' frequente, e le frequenze ordinate
decrescenti si allineano alle posizioni. Nessuna GPU; un esperto alla volta.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/build-llamacpp-tq1/gguf-py")))
from gguf import GGUFReader, quants as GQ

K = 28
CASI = {
    "Tony (256 esperti)": (
        "/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf",
        "/mnt/models/gguf/tony-tern/tony-imatrix.gguf",
        [0, 10, 20, 30, 39]),
    "ODINO (512 esperti)": (
        "/mnt/models/gguf/odino-v31/ODINO-397B-v31.gguf",
        os.path.expanduser("~/odino-lab/imatrix/Ornith-1.5-397B-imatrix.gguf"),
        [0, 15, 30, 45, 59]),
}

def norme_per_esperto(t, n_esp, quanti):
    d = np.asarray(t.data)
    per = d.shape[0] // n_esp
    out = np.empty(quanti)
    for pos in range(quanti):
        W = GQ.dequantize(d[pos*per:(pos+1)*per], t.tensor_type)
        out[pos] = float(np.square(W, dtype=np.float64).sum())
        del W
    return out

for nome, (mod, imat, strati) in CASI.items():
    try:
        r = GGUFReader(mod)
    except Exception as e:
        print(f"{nome}: ⛔ modello non leggibile: {e}")
        continue
    byname = {t.name: t for t in r.tensors}
    freq = {}
    for t in GGUFReader(imat).tensors:
        if t.name.endswith(".ffn_gate_exps.weight.counts"):
            freq[int(t.name.split(".")[1])] = np.sort(
                np.array(t.data).astype(np.float64).ravel())[::-1]
    print(f"\n{nome}")
    print("  strato   sovrapposizione top-28   impatto perso scegliendo per frequenza")
    sov_tot = []
    for L in strati:
        t = byname.get(f"blk.{L}.ffn_gate_exps.weight")
        if t is None or L not in freq:
            continue
        E = int(t.shape[2])
        # norme di TUTTI gli esperti (serve la classifica completa per impatto)
        n2 = norme_per_esperto(t, E, E)
        f = freq[L][:E]
        impatto = f * n2
        top_imp = set(np.argsort(-impatto)[:K].tolist())
        top_freq = set(range(K))
        sov = len(top_imp & top_freq)
        # quanta parte dell'impatto massimo raccoglie la scelta per frequenza
        presa = impatto[:K].sum()
        massima = np.sort(impatto)[::-1][:K].sum()
        persa = (1 - presa / max(massima, 1e-12)) * 100
        sov_tot.append(sov)
        print(f"  {L:5d}          {sov:2d}/{K}                     {persa:5.1f}%")
    if sov_tot:
        print(f"  media sovrapposizione: {np.mean(sov_tot):.1f}/{K}")
