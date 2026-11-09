#!/bin/bash
# 30/9 04:1x DIAGNOSI: stessa EMA p4 esportata col codice VECCHIO: se i byte == p4e, l export non e la causa del calo MMLU.
# esporta la EMA come TT3-p2e e la manda al Dell (sonda padre 18, poi MMLU) dietro RB2 via lucchetto.
S=@@ROOT:forge@@/thor_v2/strumenti; D=@@ROOT:depot@@/e2e/allena/TT3; COD=@@ROOT:depot@@/e2e/codice/TT3-df753ee/thor_v2
PY=@@ROOT:venv@@/bin/python3; A2=@@ROOT:models@@/gguf/tony-tern/R1/Thor-T-R1.gguf; OUT=@@ROOT:models@@/gguf/tony-tern/TT3; L=@@ROOT:depot@@/tutto_ternario/log
M=$D/migliore_ema_000004; N=TT3-p4e-codvecchio; G=$OUT/Thor-T-$N.gguf; mkdir -p $OUT
echo "[$(date '+%F %T')] export $N da $M"
"$S/governa_export.sh" systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=14G -p MemorySwapMax=2G nice -n 19 ionice -c3 \
  env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  bash -c "cd $COD && $PY e2e_esporta.py --sorgente $A2 --corsa $D --istantanea $(basename $M) --uscita $G.scale --device cpu" > $L/export_$N.log 2>&1 || { echo "⛔ export fallito"; exit 1; }
"$S/governa_export.sh" systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=14G nice -n 19 ionice -c3 \
  env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 bash -c \
  "cd $COD && $PY e2e_pv_codici.py esporta --scalato $G.scale --dst $G --registri $M/registri_pv.npz" >> $L/export_$N.log 2>&1 || { echo "⛔ export codici fallito"; rm -f $G.scale $G; exit 1; }
rm -f $G.scale
echo "[$(date '+%F %T')] export $N fatto ($(sha256sum $G | cut -c1-16))"
