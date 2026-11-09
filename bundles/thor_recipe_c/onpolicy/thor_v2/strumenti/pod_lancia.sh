#!/bin/bash
# 27/9 10:3x, ordine di Angelo: «Fermati appena l'utilizzo del pod raggiunge il 30% della sua soglia».
# Unica porta per lanciare una sessione del pod: legge la quota (premium_interactions) e RIFIUTA dal 20% in su (27/9 10:4x: da 30 a 20, Angelo).
# Uso: pod_lancia.sh <argomenti di claude-pod esegui...>   -> stampa l'id; rc 3 = quota >= 20%.
set -u
F=@@ROOT:home@@/claude-pod-lavoro/FERMO_FINO_A_DOMANI    # 27/9 10:3x Angelo: «per oggi non aprire piu' nulla» (solo cose veramente importanti)
if [ -f "$F" ] && [ "$(cut -c1-10 "$F")" = "$(date +%F)" ]; then
  echo "FERMO per oggi: $(cat "$F") (nessuna eccezione: Angelo 12:3x «Nessuna nuova sessione sul pod»)" >&2; exit 3
fi
Q=$(claude-pod consumo 2>/dev/null | grep -oE "usato [0-9.]+ %" | grep -oE "[0-9.]+")
if [ -z "$Q" ] || python3 -c "import sys;sys.exit(0 if float('$Q')>=20 else 1)"; then
  echo "QUOTA POD ${Q:-illeggibile}% >= 20%: nessun lancio (ordine di Angelo)" >&2; exit 3
fi
exec claude-pod esegui "$@"
