#!/bin/bash
# 30/9 — UN pezzo MMLU sul Dell (va chiamato SOTTO aragorn_lucchetto.sh). Argomenti: NOME GGUF PEZZO PROFILO NTASK
set -uo pipefail
N=$1; F=$2; P=$3; PR=$4; NT=$5
OUT=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/completo
log() { echo "[$(date '+%F %T')] $*"; }
[ -s "$OUT/$N.log" ] && grep -q "Final result" "$OUT/$N.log" && { log "$N gia' fatto, salto"; exit 0; }
log "lucchetto preso: $N ($F, $P, $PR)"
timeout 120 ssh aragorn "del C:\\tcq\\thor_mmlu\\log\\$N.fine 2>nul & schtasks /create /f /tn mmluc_$N /tr \"C:\\tcq\\thor_mmlu\\banco_pezzo.bat $N $F $P $PR\" /sc once /st 23:59 /rl limited && schtasks /run /tn mmluc_$N" >/dev/null \
  || { log "⛔ schtasks fallito"; exit 3; }
log "MMLU PARTITO sul Dell"
T0=$(date +%s)
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$N.fine" 2>/dev/null | grep -q FATTO; do
  [ $(( $(date +%s) - T0 )) -gt 14400 ] && { log "⛔ pezzo oltre 4 h"; exit 4; }
  sleep 60
done
timeout 600 scp -q "aragorn:C:/tcq/thor_mmlu/log/$N.log" "$OUT/$N.log" || { log "⛔ scp registro fallito"; exit 3; }
timeout 60 scp -q "aragorn:C:/tcq/thor_mmlu/log/$N.fine" "$OUT/$N.fine" 2>/dev/null
RA=$(grep -cP "^\d+\t" "$OUT/$N.log"); RM=$(grep -c "^MCLP" "$OUT/$N.log")
FR=$(grep -o "Final result.*" "$OUT/$N.log" | tr -d '\r')
log "MMLU FINITO: $N righe acc $RA · MCLP $RM · $FR · $(( ($(date +%s) - T0) / 60 )) min"
[ "$RA" -eq "$NT" ] && [ "$RM" -eq "$NT" ] || { log "⛔ righe attese $NT"; mv "$OUT/$N.log" "$OUT/$N.log.incompleto"; exit 5; }
