#!/bin/bash
# Cattura il prossimo blocco di EB3 (GPU < 5% per 4 min con l'unita' attiva): pile Python con variabili locali e pile
# native (fili di rocr), stato KFD, poi esce. Serve a sapere DOVE si ferma: tre blocchi il 28/9, tutti al passo 21.
export XDG_RUNTIME_DIR=/run/user/1000
OUT=@@ROOT:depot@@/e2e/allena/EB3/blocchi; mkdir -p "$OUT"; n=0
while true; do
  systemctl --user is-active --quiet e2e-allena-EB3 || { n=0; sleep 20; continue; }
  g=$(cat /sys/class/drm/card*/device/gpu_busy_percent)
  if [ "$g" -lt 5 ]; then n=$((n+1)); else n=0; fi
  if [ "$n" -ge 12 ]; then
    P=$(systemctl --user show -p MainPID --value e2e-allena-EB3); T=$(date +%Y%m%d-%H%M%S)
    timeout 120 py-spy dump --pid "$P" --locals > "$OUT/pile_locali_$T.txt" 2>&1
    timeout 120 py-spy dump --pid "$P" --native > "$OUT/pile_native_$T.txt" 2>&1
    ls /sys/class/kfd/kfd/proc/ > "$OUT/kfd_$T.txt" 2>&1
    tail -5 @@ROOT:depot@@/e2e/allena/EB3/registro.log > "$OUT/registro_$T.txt"
    echo "BLOCCO EB3 catturato $T (pid $P): $OUT/pile_locali_$T.txt"
    exit 0
  fi
  sleep 20
done
