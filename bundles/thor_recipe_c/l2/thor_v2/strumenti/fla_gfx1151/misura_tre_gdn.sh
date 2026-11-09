#!/bin/bash
# Percorso VERO della corsa (e2e_allena.py misura, ricetta B_ind32, strati 0 e 0,1) con le tre varianti del GDN
# (E2E_GDN = ripiego | veloce | fla) + un secondo ripiego di controllo (rumore di fondo del confronto dtheta),
# Teacher: q8-tutte-riga (in q8-EB-pieno i blocchi 0-1 sono gia' consumati da EB2: misura aspetterebbe per sempre).
# IN SEQUENZA nello stesso lancio: un solo processo GPU di prova alla volta accanto a EB2 (27/9/2026).
set -uo pipefail
QUI=$(cd "$(dirname "$0")" && pwd)
T2=$(cd "$QUI/../.." && pwd)                                # thor_v2
OUT=$QUI/esiti/misura_tre
mkdir -p "$OUT"
EB2=$(systemctl --user show e2e-allena-EB2 -p MainPID --value 2>/dev/null || echo 0)
for V in ripiego veloce fla ripiego:ripiego2; do
  GDN=${V%%:*}; ETI=${V#*:}
  echo "=== $(date +%T) E2E_GDN=$GDN etichetta $ETI: attendo la GPU (solo EB2 accanto)"
  GPU_ESCLUDI_PID="$EB2" "$QUI/../gpu_occupata.sh" --attendi 5400 || { echo "GPU mai libera: salto $ETI"; continue; }
  ( cd "$T2" && env -u HSA_OVERRIDE_GFX_VERSION GPU_ESCLUDI_PID="$EB2" E2E_GDN="$GDN" E2E_MISURA_ETICHETTA="$ETI" \
      E2E_MISURA_DTHETA="$OUT" @@ROOT:venv@@/bin/python3 e2e_allena.py misura \
      --teacher "${TEACHER:-@@ROOT:lavoro@@/e2e/teacher/q8-tutte-riga}" --strati 0 --strati 0,1 --ripetizioni 2 --ricetta B_ind32 \
      --uscita "$OUT/misura_$ETI.json" 2>&1 | grep -v "UserWarning\|warn(" | tail -25 )
  echo "=== $(date +%T) fine $ETI rc=$?"
done
