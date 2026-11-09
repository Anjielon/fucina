#!/bin/bash
# Copia (sha verificato) + MMLU 2000 sul Dell con banco_ot.bat, senza sonda. Uso: dell_solo_mmlu.sh NOME GGUF
N=$1; G=$2; F=$(basename "$G"); log() { echo "[$(date '+%F %T')] $*"; }
SHA=$(sha256sum "$G" | cut -d' ' -f1); log "copia $F (sha $SHA)"
timeout 3600 scp -q "$G" "aragorn:C:/tcq/thor_mmlu/gguf/$F" || { log "⛔ copia fallita"; exit 3; }
SHA_A=$(timeout 600 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$F SHA256" | sed -n 2p | tr -d '\r ')
[ "$SHA" = "$SHA_A" ] || { log "⛔ sha diverso sul Dell"; exit 3; }
log "COPIATO, sha verificato"
# 29/9: niente "| bash -s": il primo ssh dello script si mangiava il resto da stdin (MMLU mai aspettato)
bash <(sed "s/banco_ot2.bat/banco_ot.bat/" @@ROOT:depot@@/tutto_ternario/codice/dell_mmlu_ot2.sh) "$N" "$F" < /dev/null
