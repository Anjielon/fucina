#!/usr/bin/env python3
"""L'ERRORE CHE CONTA E' ||dW·X||, NON ||dW||.

Finora abbiamo sempre misurato quanto il secondo piano cambia i PESI. Quello
che il modello sente e' quanto cambia l'USCITA, sotto le attivazioni vere di
quello strato: la stessa perturbazione dei pesi puo' valere molto o nulla a
seconda di come e' orientata rispetto alla covarianza dell'ingresso.

Prudente sulla memoria: uno strato per volta, campione ridotto, float32,
e si ferma se la RAM disponibile scende. ODINO sta girando.
"""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/build-llamacpp-tq1/gguf-py")))
from gguf import GGUFReader, quants as GQ

M = "/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf"
DUMP = "/tmp/ingresso_down"
DANNO = {0:10.9524, 5:8.7277, 10:8.8125, 15:20.1161, 20:37.8030,
         25:22.5109, 30:8.6860, 35:8.7743, 39:8.7370}
N_CAMP, N_ESP = 256, 6

def ram_gb():
    with open("/proc/meminfo") as f:
        for r in f:
            if r.startswith("MemAvailable"):
                return int(r.split()[1]) / 1048576
    return 99.0

def esperto(t, pos):
    d = np.asarray(t.data); E = int(t.shape[2]); per = d.shape[0] // E
    ind, out = int(t.shape[0]), int(t.shape[1])
    return GQ.dequantize(d[pos*per:(pos+1)*per], t.tensor_type).astype(np.float32).reshape(out, ind)

r = GGUFReader(M)
byname = {t.name: t for t in r.tensors}
print("Errore RELATIVO sull'uscita di `down`, sotto le attivazioni vere\n")
print("  strato   ||dW||/||W1||   ||dW·X||/||W1·X||   rapporto   danno")
righe = []
for L in sorted(DANNO):
    if ram_gb() < 5:
        print(f"  ⛔ RAM sotto 5 GiB: mi fermo allo strato {L}"); break
    a, b = byname.get(f"blk.{L}.ffn_down_exps.weight"), byname.get(f"blk.{L}.ffn_down_exps2.weight")
    p = glob.glob(f"{DUMP}/*-{L}.f32")
    if a is None or b is None or not p: continue
    n_ff = int(a.shape[0])
    X = np.fromfile(p[0], dtype=np.float32)
    if X.size % n_ff: continue
    X = X.reshape(-1, n_ff)[:N_CAMP].T            # (n_ff, campioni)
    ew, ey = [], []
    for pos in range(min(N_ESP, int(b.shape[2]))):
        W1 = esperto(a, pos); W2 = esperto(b, pos)
        nw1 = np.linalg.norm(W1)
        if nw1 <= 0: continue
        ew.append(np.linalg.norm(W2) / nw1)
        Y1 = W1 @ X; dY = W2 @ X
        ny1 = np.linalg.norm(Y1)
        if ny1 > 0: ey.append(np.linalg.norm(dY) / ny1)
        del W1, W2, Y1, dY
    del X
    if not ew or not ey: continue
    mw, my = float(np.mean(ew)), float(np.mean(ey))
    righe.append((L, mw, my, my/max(mw,1e-9)))
    print(f"  {L:5d}   {mw:12.4f}   {my:16.4f}   {my/max(mw,1e-9):8.3f}   {DANNO[L]:7.2f}")

if len(righe) >= 4:
    y = np.array([DANNO[L] for L,_,_,_ in righe])
    for nome, i in (("||dW||", 1), ("||dW·X||", 2), ("rapporto", 3)):
        x = np.array([r[i] for r in righe])
        rho = float(np.corrcoef(x, y)[0,1])
        print(f"\n  correlazione {nome:9s} / danno: {rho:+.3f}" + ("  ⭐" if abs(rho)>0.7 else ""))
    print("\n  Se ||dW·X|| correla e ||dW|| no, l'errore proiettato E' il meccanismo,")
    print("  e spiega perche' misurare i pesi non poteva vederlo.")
