#!/bin/bash
# 27/9 21:3x — misura continua dell'MMLU (log-score calibrato, Brier, margine; bootstrap appaiato) su ogni registro MMLU di EB3
# appena completo (2000 righe MCLP) sul Dell: confronto con l'EMA16 di EB2 e con B. Solo CPU (nice 19, 4 fili). Esito nei cicli.
set -u
R=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn; O=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/continuo_EB3; mkdir -p $O
cd @@ROOT:forge@@/thor_v2
log() { echo "[$(date +%F' '%T)] $*"; }
log "continuo EB3 acceso"
while true; do
  for L in $(ls $R/EB3-*.log 2>/dev/null); do
    N=$(basename $L .log); [ -s $O/$N.json ] && continue
    [ "$(grep -c '^MCLP' $L 2>/dev/null)" = 2000 ] || continue
    log "misura continua di $N"
    if env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 19 python3 strumenti/mmlu_continuo.py \
         --aggiungi "$N=$L" --json $O/$N.json > $O/$N.txt 2>&1; then
      RIS=$(python3 -c "
import json,sys
d=json.load(open('$O/$N.json'))['delte']
out=[]
for k in ('$N−e16 (Aragorn)','$N−B (Aragorn)'):
    x=d.get(k,{})
    out.append(k+': '+', '.join(f'{m} {x[m][\"stima\"]:+.4f} (z {x[m][\"z\"]:.1f})' for m in ('acc','logp_scelte_T','brier','margine_T') if m in x))
print(' | '.join(out))")
      log "$N: $RIS"
      printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "MMLU continuo $N (bootstrap appaiato; regola: leva valida se log-score calibrato >= +0,0072 e Brier <= -0,0049 A PASSO): $RIS" >> @@ROOT:forge@@/stato/cicli.tsv
    else
      log "⛔ misura continua di $N fallita (vedi $O/$N.txt)"; touch $O/$N.json
    fi
  done
  sleep 120
done
