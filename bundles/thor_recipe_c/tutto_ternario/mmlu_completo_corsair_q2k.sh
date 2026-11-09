#!/bin/bash
# 30/9 14:00 — GPU del Corsair ferma: fa i pezzi MMLU del Q2_K DALLA CODA (14000 -> 04000) mentre il Dell va dalla testa.
# Stessa copia isolata MCLP e stesse opzioni di mmlu_logprob_lancia.sh; parita' Corsair/Dell verificata sul pezzo 00000 (78,15 = 78,15).
# Scrive nel registro comune solo se il Dell non l'ha gia' fatto (il pezzo del Dell salta se trova il registro con Final result).
set -uo pipefail
B=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/sorgente/build/bin; MC=@@ROOT:forge@@/mc_v2
OUT=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/completo; G=@@ROOT:models@@/gguf/thor-rivale/Ornith-1.5-35B-A3B-Q2_K.gguf
log() { echo "[$(date '+%F %T')] $*"; }
cd @@ROOT:forge@@/thor_v2; unset HSA_OVERRIDE_GFX_VERSION
for p in 14000 12000 10000 08000 06000 04000; do
  N=MMLUC-Q2K-$p; NT=$([ $p = 14000 ] && echo 42 || echo 2000)
  [ -s $OUT/$N.log ] && grep -q "Final result" $OUT/$N.log && { log "$N gia' fatto (Dell): mi fermo, il Dell mi ha raggiunto"; break; }
  if ! @@ROOT:forge@@/thor_v2/strumenti/gpu_occupata.sh >/dev/null; then log "GPU occupata da altri: mi fermo"; break; fi
  T=$OUT/.$N.corsair.tmp; log "Corsair $N parte"
  LD_LIBRARY_PATH=$B $B/llama-perplexity -m $G -ngl 999 -c 4096 -np 4 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f $MC/mmlu_$p.bin > $T 2>&1
  RA=$(grep -cE '^[0-9]+	[0-9]+\.[0-9]+$' $T); RM=$(grep -c '^MCLP	' $T); FR=$(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' $T | tail -1)
  if [ "$RA" = "$NT" ] && [ "$RM" = "$NT" ] && [ -n "$FR" ]; then
    if [ -s $OUT/$N.log ]; then log "$N: il Dell l'ha finito prima, tengo il suo"; rm -f $T; break; fi
    echo "# eseguito sul CORSAIR (Vulkan AMD), non sul Dell" >> $T; mv $T $OUT/$N.log; log "MMLU FINITO: $N (Corsair) · $FR"
  else log "⛔ $N incompleto: acc $RA MCLP $RM"; mv $T $OUT/$N.corsair.incompleto; fi
done
log "fine corsia Corsair"
