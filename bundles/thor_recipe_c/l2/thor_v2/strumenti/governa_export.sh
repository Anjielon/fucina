#!/bin/bash
# 29/9 06:5x — l'export CPU del sorvegliante ferma la GPU di TT2 (misurato due volte: SIGSTOP dell'export -> GPU 0 -> 100 % in 10 s;
# con pressione IO bassa, 10 %, quindi non e' solo IO). Governatore indipendente dalla causa: esegue il comando dato e, se la GPU
# resta sotto 10 % per 30 s mentre il comando vive, lo mette in pausa (SIGSTOP al gruppo) finche' la GPU non torna sopra 50 % per 60 s.
# Uso: governa_export.sh <comando...>   (exit code = quello del comando)
gpu() { if [ -n "${GOV_GPU_FILE:-}" ]; then cat "$GOV_GPU_FILE"; else cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | sort -n | tail -n 1; fi; }
setsid "$@" & P=$!
basso=0; fermo=0; alto=0; pause=0
while kill -0 $P 2>/dev/null; do
  g=$(gpu); g=${g:-100}
  if [ $fermo = 0 ]; then
    if [ "$g" -lt 10 ]; then basso=$((basso+5)); else basso=0; fi
    if [ $basso -ge 30 ]; then kill -STOP -- -$P 2>/dev/null; fermo=1; alto=0; pause=$((pause+1)); echo "[$(date +%T)] governatore: export in pausa (GPU $g %)"; fi
  else
    if [ "$g" -gt 50 ]; then alto=$((alto+5)); else alto=0; fi
    if [ $alto -ge 60 ]; then kill -CONT -- -$P 2>/dev/null; fermo=0; basso=0; echo "[$(date +%T)] governatore: export ripreso"; fi
  fi
  sleep 5
done
wait $P; rc=$?
echo "[$(date +%T)] governatore: fine, pause $pause, rc $rc"; exit $rc
