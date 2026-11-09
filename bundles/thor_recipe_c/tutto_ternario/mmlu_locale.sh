#!/bin/bash
# 28/9 tutto_ternario — MMLU 2000 coi logprob in LOCALE (GPU del Corsair, EB3 ferma per decisione di Angelo):
# stessa copia isolata di llama-perplexity (patch MCLP), stesso file dei 2000 compiti (sha af987cc3…), stessi argomenti
# di mmlu_logprob_lancia.sh / catena_v2.sh, piu' -ub 512 esplicito (= default). Salta solo la guardia «GPU occupata»
# (la GPU e' nostra). Uso: mmlu_locale.sh NOME GGUF    Registro: @@ROOT:models@@/gguf/tony-tern/mmlu_logprob/NOME.log
set -uo pipefail
NOME=${1:?}; GGUF=${2:?}
SB=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob; B=$SB/sorgente/build/bin
BANCO=@@ROOT:forge@@/mc_v2/mmlu_00000.bin
SHA_BANCO=af987cc3369c15f90df9f390def043fd9fa4787e075b3b2170ff697a9dabbc6a
REG=$SB/$NOME.log
grep -qa "MCLP<TAB>" "$B/libllama-perplexity-impl.so" || { echo "⛔ manca la patch MCLP"; exit 2; }
[ "$(sha256sum "$BANCO" | cut -d' ' -f1)" = "$SHA_BANCO" ] || { echo "⛔ banco diverso"; exit 2; }
[ -s "$REG" ] && { echo "⛔ $REG esiste gia'"; exit 2; }
cd @@ROOT:forge@@/thor_v2 || exit 2
unset HSA_OVERRIDE_GFX_VERSION
T0=$(date +%s)
echo "[$(date '+%F %T')] MMLU locale $NOME · $GGUF (sha $(sha256sum "$GGUF" | cut -c1-16)) → $REG"
LD_LIBRARY_PATH=$B "$B/llama-perplexity" -m "$GGUF" -ngl 999 -c 4096 -np 4 -ub 512 -dev Vulkan0 \
  --multiple-choice --multiple-choice-tasks 100000 -f "$BANCO" > "$REG" 2>&1
rc=$?
N_ACC=$(grep -cE '^[0-9]+	[0-9]+\.[0-9]+$' "$REG"); N_MCLP=$(grep -c '^MCLP	' "$REG")
FINALE=$(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' "$REG" | tail -1)
echo "[$(date '+%F %T')] rc=$rc · acc $N_ACC · MCLP $N_MCLP · ${FINALE:-Final ASSENTE} · $(( $(date +%s) - T0 )) s"
[ "$N_ACC" -eq 2000 ] && [ "$N_MCLP" -eq 2000 ] && [ -n "$FINALE" ]
