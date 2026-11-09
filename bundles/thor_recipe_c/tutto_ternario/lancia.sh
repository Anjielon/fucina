#!/bin/bash
# Lanciatore UNICO dei lavori pesanti del progetto tutto_ternario (28/9). Vincoli del coordinatore:
# CPU 400 %, Nice 19, MemoryHigh 10G / MemoryMax 14G, 4 thread per ogni libreria, IO 100 MB/s per disco, niente GPU.
# In piu' la guardia GPU (guardia_gpu.sh): se gpu_busy_percent resta < 40 % per 120 s il lavoro va in SIGSTOP (e
# riprende quando la GPU torna >= 60 % per 30 s); ogni evento finisce in guardia_<unita>.log.
# Uso: lancia.sh <nome> <script.py> [argomenti...]      -> unita' tt-<nome>, registro ../log/<nome>.log
set -u
N=$1; shift
Q=@@ROOT:depot@@/tutto_ternario
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
mkdir -p $Q/log
B=$(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null)
echo "[$(date '+%F %T')] lancio tt-$N (GPU ora ${B}%): $*" >> $Q/log/lanci.log
systemd-run --user --collect --unit="tt-$N" \
  -p CPUQuota=400% -p Nice=19 -p MemoryHigh=10G -p MemoryMax=14G -p MemorySwapMax=2G \
  -p "IOReadBandwidthMax=/dev/nvme1n1 100M" -p "IOWriteBandwidthMax=/dev/nvme1n1 100M" \
  -p "IOReadBandwidthMax=/dev/nvme0n1 100M" -p "IOWriteBandwidthMax=/dev/nvme0n1 100M" \
  -p WorkingDirectory=$Q/codice -p StandardOutput="append:$Q/log/$N.log" -p StandardError="append:$Q/log/$N.log" \
  --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=NUMEXPR_NUM_THREADS=4 --setenv=TT_THREADS=4 --setenv=HIP_VISIBLE_DEVICES= --setenv=CUDA_VISIBLE_DEVICES= \
  --setenv=ROCR_VISIBLE_DEVICES= --setenv=PYTHONUNBUFFERED=1 \
  @@ROOT:venv@@/bin/python3 "$@" || exit $?
nohup setsid bash $Q/codice/guardia_gpu.sh "tt-$N" >> $Q/log/guardia_tt-$N.log 2>&1 < /dev/null &
echo "tt-$N avviata, guardia GPU pid $!"
