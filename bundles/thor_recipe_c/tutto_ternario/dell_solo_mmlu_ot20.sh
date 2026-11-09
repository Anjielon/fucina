#!/bin/bash
# 30/9: solo MMLU 2000 (sonda gia' fatta) con banco_ot20.bat (esperti 0-19 in RAM): per R2B4, OOM al compito 1202 con 0-15.
# Va chiamato SOTTO il lucchetto: aragorn_lucchetto.sh dell_solo_mmlu_ot20.sh NOME /percorso/FILE.gguf
set -uo pipefail
N=$1; G=$2; F=$(basename "$G"); S=@@ROOT:forge@@/thor_v2/strumenti
REG_DIR=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn; REG=$REG_DIR/$N.log; RIF=$REG_DIR/R2B3.log
log() { echo "[$(date '+%F %T')] $*"; }
pulisci() { timeout 60 ssh aragorn "del C:\\tcq\\thor_mmlu\\gguf\\$F" >/dev/null 2>&1; log "GGUF $F tolto dal Dell"; }
[ -s "$REG" ] && { log "⛔ $REG esiste gia'"; exit 2; }
timeout 60 scp -q @@ROOT:depot@@/tutto_ternario/codice/banco_ot20.bat aragorn:C:/tcq/thor_mmlu/banco_ot20.bat || { log "⛔ copia bat"; exit 3; }
SHA=$(sha256sum "$G" | cut -d' ' -f1); timeout 3600 scp -q "$G" "aragorn:C:/tcq/thor_mmlu/gguf/$F" || { log "⛔ copia"; pulisci; exit 3; }
A=$(timeout 600 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$F SHA256" | sed -n 2p | tr -d '\r ')
[ "$A" = "$SHA" ] || { log "⛔ sha diverso"; pulisci; exit 3; }; log "COPIATO sul Dell, sha verificato"
timeout 120 ssh aragorn "schtasks /create /f /tn thor_mmlu_$N /tr \"C:\\tcq\\thor_mmlu\\banco_ot20.bat $N $F\" /sc once /st 23:59 /rl limited && schtasks /run /tn thor_mmlu_$N" >/dev/null || { log "⛔ schtasks"; pulisci; exit 3; }
log "MMLU PARTITO sul Dell (ot20)"; T0=$(date +%s)
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$N.fine" 2>/dev/null | grep -q FATTO; do [ $(( $(date +%s) - T0 )) -gt 14400 ] && { log "⛔ oltre 4 h"; break; }; sleep 60; done
timeout 60 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$N.fine"
timeout 600 scp -q "aragorn:C:/tcq/thor_mmlu/log/$N.log" "$REG" || log "⛔ registro non riportato"
NA=$(tr -d '\r' < "$REG" | grep -cE '^[0-9]+	[0-9]+\.[0-9]+$'); NM=$(grep -c '^MCLP	' "$REG"); FR=$(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' "$REG" | tail -1)
log "MMLU FINITO: righe acc $NA · MCLP $NM · ${FR:-Final result ASSENTE}"
if [ "$NA" -eq 2000 ] && [ "$NM" -eq 2000 ] && [ -n "$FR" ]; then
  nice -n 19 env OMP_NUM_THREADS=2 python3 "$S/mmlu_quattro_numeri.py" "$REG" --confronta "$RIF" --json "$REG_DIR/$N.json" | tail -5
fi
pulisci
