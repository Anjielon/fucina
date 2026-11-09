#!/bin/bash
# diagnosi crollo K01: dump llama.cpp (CPU) per k01, base, maestro — in sequenza, una corsia
cd @@ROOT:depot@@/diagnosi_k01 || exit 2
for m in k01 base maestro; do
  echo "[$(date +%T)] inizio $m"
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= @@ROOT:venv@@/bin/python3 @@ROOT:forge@@/thor_v2/crollo_k01.py llama $m > llama_$m.log 2>&1
  echo "[$(date +%T)] fine $m rc=$?"
done
