#!/usr/bin/env python3
"""LA PERMUTAZIONE E' GIUSTA ALLO STRATO DEL DISASTRO?

Nessuna statistica dell'attivazione spiega il picco allo strato 20 (rango
effettivo +0.284, curtosi -0.186). La conclusione registrata PRIMA della
misura era: allora si insegue come difetto.

Primo sospetto: la forgia riordina gli esperti caldi-primi e il secondo piano
copre i primi 28 posti. Se a QUEL strato la permutazione fosse sbagliata, la
correzione andrebbe agli esperti sbagliati e la maschera coprirebbe i freddi.
Si verifica confrontando l'ordine nel file con i conteggi dell'imatrix.
"""
import re, sys
import os
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/build-llamacpp-tq1/gguf-py")))
from gguf import GGUFReader, quants as GQ

TONY = "/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf"
SORG = "/mnt/models/gguf/ornith-1.5/Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf"
IMAT = "/mnt/models/gguf/tony-tern/tony-imatrix.gguf"
K = 28

def router(r, L):
    t = next((x for x in r.tensors if x.name == f"blk.{L}.ffn_gate_inp.weight"), None)
    if t is None: return None
    W = GQ.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
    ne = [int(x) for x in t.shape]
    W = W.reshape(ne[1], ne[0])
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)

caldi = {}
for t in GGUFReader(IMAT).tensors:
    if t.name.endswith(".ffn_gate_exps.weight.counts"):
        L = int(t.name.split(".")[1])
        caldi[L] = np.array(t.data).astype(np.float64).ravel()

rt, rs = GGUFReader(TONY), GGUFReader(SORG)
print("Per ogni strato: i primi 28 posti del FILE sono davvero i 28 piu' caldi?\n")
print("  strato   quanti dei 28 sono fra i piu' caldi   quota traffico coperta")
righe = []
for L in sorted(caldi):
    A, B = router(rt, L), router(rs, L)
    if A is None or B is None or A.shape != B.shape: continue
    perm = np.argmax(A @ B.T, axis=1)      # posizione i del file -> esperto originale
    if len(set(perm.tolist())) != len(perm): continue
    c = caldi[L]
    veri = set(np.argsort(-c)[:K].tolist())
    primi = set(perm[:K].tolist())
    ok = len(veri & primi)
    quota = c[list(primi)].sum() / max(c.sum(), 1e-9) * 100
    massimo = c[np.argsort(-c)[:K]].sum() / max(c.sum(), 1e-9) * 100
    righe.append((L, ok, quota, massimo))
DANNO = {0:10.95, 5:8.73, 10:8.81, 15:20.12, 20:37.80, 25:22.51, 30:8.69, 35:8.77, 39:8.74}
for L, ok, q, mx in righe:
    if L in DANNO:
        segno = "✓" if ok == K else "⛔"
        print(f"  {L:5d}      {ok:2d}/{K}  {segno}                   "
              f"{q:5.2f}% (massimo possibile {mx:5.2f}%)   danno {DANNO[L]:6.2f}")
peggio = [r for r in righe if r[1] < K]
print(f"\n  strati con permutazione IMPERFETTA: {len(peggio)} su {len(righe)}")
if peggio:
    print("   " + ", ".join(f"{L}:{ok}/{K}" for L, ok, _, _ in peggio[:12]))
    print("\n  ⇒ Se gli strati imperfetti coincidono coi danneggiati, TROVATO.")
else:
    print("\n  ⇒ La permutazione e' PERFETTA ovunque: il difetto non e' li'.")
