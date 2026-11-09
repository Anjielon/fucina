#!/bin/bash
# 1/10: copia sul Dell R2B3m, QAT2-neutro, QAT2 con verifica sha256 (certutil). Esce !=0 al primo errore.
set -u; G=@@ROOT:models@@/gguf/tony-tern
for f in "$G/QAT2/Thor-T-QAT2-neutro.gguf" "$G/R2B3m/Thor-T-R2B3m.gguf" "$G/QAT2/Thor-T-QAT2.gguf"; do
  b=$(basename "$f"); s=$(sha256sum "$f" | cut -c1-64)
  scp -q "$f" "aragorn:C:/tcq/thor_mmlu/gguf/$b" || { echo "ERRORE copia $b"; exit 3; }
  a=$(ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$b SHA256" | sed -n 2p | tr -d '\r ')
  [ "$a" = "$s" ] || { echo "ERRORE sha $b $a != $s"; exit 3; }
  echo "$(date +%H:%M) OK $b $s"
done
echo FATTO
