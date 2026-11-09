#!/bin/bash
# 28/9 20:3x — EB3 FERMA, GPU del Corsair in esclusiva a tutto_ternario (decisione del coordinatore con Angelo).
# Come lancia.sh ma con la GPU: torch ROCm (HSA_OVERRIDE_GFX_VERSION=11.5.1, il valore giusto per i wheel di casa),
# TT_DEVICE=cuda. Memoria host sempre confinata. Uso: [VAR=...] lancia_gpu.sh <nome> <script.py> [argomenti...]
set -u
N=$1; shift
Q=@@ROOT:depot@@/tutto_ternario
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
mkdir -p $Q/log
echo "[$(date '+%F %T')] lancio GPU tt-$N: TT_RICETTA=${TT_RICETTA:-20set} TT_TENSORI=${TT_TENSORI:-} $*" >> $Q/log/lanci.log
EXTRA=()
for v in TT_RICETTA TT_TENSORI TT_ROUND TT_HESS_OUT; do [ -n "${!v:-}" ] && EXTRA+=(--setenv=$v="${!v}"); done
systemd-run --user --collect --unit="tt-$N" \
  -p CPUQuota=800% -p Nice=5 -p MemoryHigh=12G -p MemoryMax=16G -p MemorySwapMax=2G \
  -p WorkingDirectory=$Q/codice -p StandardOutput="append:$Q/log/$N.log" -p StandardError="append:$Q/log/$N.log" \
  --setenv=OMP_NUM_THREADS=8 --setenv=MKL_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=8 --setenv=TT_THREADS=8 \
  --setenv=HSA_OVERRIDE_GFX_VERSION=11.5.1 --setenv=TT_DEVICE=cuda --setenv=PYTHONUNBUFFERED=1 "${EXTRA[@]}" \
  @@ROOT:venv@@/bin/python3 "$@"
