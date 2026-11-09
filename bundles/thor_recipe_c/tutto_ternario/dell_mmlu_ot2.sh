#!/bin/bash
# MMLU 2000 sul Dell con banco_ot2.bat (esperti 0-15 e 37-38 in RAM), GGUF gia' copiato e verificato. Sotto il lucchetto.
N=$1; F=$2; REG=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/$N.log
log() { echo "[$(date '+%F %T')] $*"; }
timeout 120 ssh aragorn "del C:\\tcq\\thor_mmlu\\log\\$N.fine 2>nul & schtasks /create /f /tn thor_mmlu_$N /tr \"C:\\tcq\\thor_mmlu\\banco_ot2.bat $N $F\" /sc once /st 23:59 /rl limited && schtasks /run /tn thor_mmlu_$N" >/dev/null || { log "⛔ schtasks fallito"; exit 3; }
log "MMLU PARTITO sul Dell (banco_ot2)"
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$N.fine" 2>/dev/null | grep -q FATTO; do sleep 60; done
timeout 600 scp -q "aragorn:C:/tcq/thor_mmlu/log/$N.log" "$REG"
log "MMLU FINITO: $(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' "$REG" | tail -n 1) righe $(tr -d '\r' < "$REG" | grep -cE '^[0-9]+	[0-9.]+$')"
timeout 60 ssh aragorn "del C:\\tcq\\thor_mmlu\\gguf\\$F" >/dev/null 2>&1; log "GGUF $F tolto dal Dell"
