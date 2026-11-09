#!/bin/bash
# L1 di analisi_leve_conoscenza.md (28/9/2026): l'esperto condiviso ALLARGATO gira su llama.cpp di serie?
# SOLO CPU (nessuna GPU vista: il log deve dire "no usable GPU found"). Quattro file minuscoli qwen35moe:
#   orig = e2e_modello_minuscolo.py · A = copia con FS_EXTRA=0 (deve essere == orig al bit)
#   B = +64 neuroni, colonne nuove del down a ZERO (atteso: KLD 0 contro A, stessa scelta 100 %)
#   C = +64 neuroni, down nuovo casuale (atteso: KLD > 0 -> i neuroni nuovi sono vivi nel grafo)
# e il controllo al byte: in B i tensori di A sono identici, quelli del condiviso sono il PREFISSO.
set -euo pipefail
QUI=$(cd "$(dirname "$0")" && pwd)
LAV=${1:-@@ROOT:depot@@/leve_conoscenza/allarga}
P=${LLAMA_PPL:-@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/llama-perplexity}
mkdir -p "$LAV"; cd "$LAV"
R=(env XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
   systemd-run --user --scope -q -p CPUQuota=200% nice -n 19 ionice -c3
   env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= GGML_VK_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2)
"${R[@]}" python3 "$QUI/../e2e_modello_minuscolo.py" orig.gguf > /dev/null
"${R[@]}" env FS_EXTRA=0 python3 "$QUI/prova_condiviso_allargato_modello.py" A.gguf > /dev/null
"${R[@]}" env FS_EXTRA=64 DOWN_EXTRA=zero python3 "$QUI/prova_condiviso_allargato_modello.py" B.gguf > /dev/null
"${R[@]}" env FS_EXTRA=64 DOWN_EXTRA=casuale python3 "$QUI/prova_condiviso_allargato_modello.py" C.gguf > /dev/null
cmp -s orig.gguf A.gguf && echo "A == orig al bit: SI" || { echo "A != orig: FALLITO"; exit 1; }
"${R[@]}" python3 - <<'PY'
import sys; sys.path.insert(0,"@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from gguf import GGUFReader
import numpy as np
A={t.name:t for t in GGUFReader("A.gguf").tensors}; B={t.name:t for t in GGUFReader("B.gguf").tensors}
ok=True
for k,ta in A.items():
    da=np.asarray(ta.data); db=np.asarray(B[k].data)
    if k.endswith(("gate_shexp.weight","up_shexp.weight")): s=db[:da.shape[0]].tobytes()==da.tobytes()
    elif k.endswith("down_shexp.weight"): s=db[:, :da.shape[1]].tobytes()==da.tobytes()
    else: s=da.tobytes()==db.tobytes()
    ok&=s
print(f"byte: {len(A)} tensori, vecchi identici o prefisso: {ok}"); sys.exit(0 if ok else 1)
PY
head -c 200000 @@ROOT:odino@@/collaudo/corpus/wiki.test.raw > testo.txt
"${R[@]}" "$P" -m A.gguf -f testo.txt -c 256 --chunks 8 -t 2 -ngl 0 --device none --kl-divergence-base A.kld > logA.txt 2>&1
for M in B C; do
  "${R[@]}" "$P" -m $M.gguf -f testo.txt -c 256 --chunks 8 -t 2 -ngl 0 --device none --kl-divergence-base A.kld --kl-divergence > log$M.txt 2>&1
  grep -q "no usable GPU found" log$M.txt || { echo "$M: GPU vista?"; exit 1; }
  echo "== $M: $(grep -E 'Mean    KLD' log$M.txt) | $(grep -E 'Maximum KLD' log$M.txt) | $(grep -E 'Same top p' log$M.txt)"
done
