#!/bin/bash
# 26/9 — MMLU (mc_v2/mmlu_00000.bin, 2000 compiti) con la copia isolata di llama-perplexity che stampa, per ogni
# compito, una riga «MCLP» con i logprob delle quattro scelte (depurazione PriDe: B contro B-150 perde per lettera).
#
# Binario: @@ROOT:models@@/gguf/tony-tern/mmlu_logprob/sorgente/build/bin/llama-perplexity, costruito da costruisci.sh
# (stesso albero di @@ROOT:models@@/build/llamacpp-tq1-k1024 + perplexity_mclp.patch, che aggiunge SOLO stampe).
# Opzioni = catena_v2.sh riga 77 (le stesse di valuta_a3_densi.sh e thor/cancello_v20_downaggr3.sh):
#   -m GGUF -ngl 999 -c 4096 -np 4 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f mc_v2/mmlu_00000.bin
# Il registro ha le stesse righe di quello della catena (task/acc, Final result) PIU' le righe MCLP:
#   MCLP <TAB> indice compito 0-based (= posizione nel file) <TAB> giusta 0-based <TAB> scelta 0-based <TAB> n scelte
#        <TAB> logprob[0..n-1] (log-verosimiglianza media per token usata per il punteggio, %.9g)
#        <TAB> token valutati per scelta (1 = log p(token-lettera | prompt) sulla softmax dell'intero vocabolario)
# Le righe «n<TAB>acc» restano quelle di prima: strumenti/mmlu_per_domanda.py legge questo registro come gli altri.
#
# Uso:  mmlu_logprob_lancia.sh NOME GGUF        es. mmlu_logprob_lancia.sh B @@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf
# Registro: @@ROOT:models@@/gguf/tony-tern/mmlu_logprob/NOME.log (mai sovrascritto: un NOME gia' usato = rifiuto).
# La GPU e' UNA corsia: si parte solo se nessun altro la occupa (strumenti/gpu_occupata.sh, esenti i server della casa).
set -uo pipefail
NOME=${1:?uso: mmlu_logprob_lancia.sh NOME GGUF}
GGUF=${2:?uso: mmlu_logprob_lancia.sh NOME GGUF}
[[ "$NOME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "⛔ NOME=$NOME: solo lettere, cifre, . _ -"; exit 2; }

SB=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob
B=$SB/sorgente/build/bin
BANCO=@@ROOT:forge@@/mc_v2/mmlu_00000.bin
SHA_BANCO=af987cc3369c15f90df9f390def043fd9fa4787e075b3b2170ff697a9dabbc6a   # 26/9, = mmlu_per_domanda.md
REG=$SB/$NOME.log

# ── 1. ingressi
[ -x "$B/llama-perplexity" ] || { echo "⛔ manca $B/llama-perplexity: lanciare prima $SB/costruisci.sh"; exit 2; }
grep -qa "MCLP<TAB>" "$B/libllama-perplexity-impl.so" || {
  echo "⛔ $B/libllama-perplexity-impl.so non contiene la patch MCLP: non e' la copia isolata"; exit 2; }
[ -f "$GGUF" ] || { echo "⛔ GGUF inesistente: $GGUF"; exit 2; }
[ "$(sha256sum "$BANCO" | cut -d' ' -f1)" = "$SHA_BANCO" ] || { echo "⛔ $BANCO non e' piu' il file dei 2000 compiti"; exit 2; }
[ -s "$REG" ] && { echo "⛔ $REG esiste gia': scegliere un altro NOME"; exit 2; }

# ── 2. una corsia pesante per volta (stessa guardia di lancia_forgia_scorrevole.sh: nome del processo, poi VRAM vera)
OCCUPATO=$(ps -eo pid=,comm=,args= | awk '
  $2 ~ /^python3/ { for (i = 4; i <= NF; i++) if ($i ~ /(^|\/)forgia_(globale_40_strati|scorrevole)\.py$/) { print; next } }
  $2 == "catena_v2.sh" || $2 ~ /^llama-perplexi/ { print }')
if [ -n "$OCCUPATO" ]; then
  echo "$OCCUPATO"; echo "⛔ una forgia, una catena o un banco e' gia' in corso (sopra): niente partenza"; exit 3
fi
if ! GPU_ALTRI=$(@@ROOT:forge@@/thor_v2/strumenti/gpu_occupata.sh); then
  echo "$GPU_ALTRI" | head -5; echo "⛔ la GPU e' occupata da processi che non sono la casa (sopra): niente partenza"; exit 3
fi

# ── 3. il banco, con l'ambiente della catena (catena_v2.sh: cd thor_v2, niente HSA_OVERRIDE_GFX_VERSION)
cd @@ROOT:forge@@/thor_v2 || exit 2
unset HSA_OVERRIDE_GFX_VERSION
echo "[$(date '+%d/%m %H:%M:%S')] MMLU logprob $NOME · $GGUF → $REG"
LD_LIBRARY_PATH=$B "$B/llama-perplexity" -m "$GGUF" -ngl 999 -c 4096 -np 4 -dev Vulkan0 \
  --multiple-choice --multiple-choice-tasks 100000 -f "$BANCO" > "$REG" 2>&1
rc=$?

# ── 4. controllo del registro: 2000 righe acc, 2000 righe MCLP, indici 0..1999 in ordine, Final result presente
N_ACC=$(grep -cE '^[0-9]+	[0-9]+\.[0-9]+$' "$REG")
N_MCLP=$(grep -c '^MCLP	' "$REG")
ORDINE=$(awk -F'\t' '$1 == "MCLP" { if ($2 != n) { print "fuori ordine alla riga " NR; exit } n++ } END { if (!n) print "nessuna" }' n=0 "$REG")
FINALE=$(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' "$REG" | tail -1)
echo "[$(date '+%d/%m %H:%M:%S')] rc=$rc · righe acc $N_ACC · righe MCLP $N_MCLP ${ORDINE:+· $ORDINE} · ${FINALE:-Final result ASSENTE}"
[ "$rc" -eq 0 ] && [ "$N_ACC" -eq 2000 ] && [ "$N_MCLP" -eq 2000 ] && [ -z "$ORDINE" ] && [ -n "$FINALE" ] || exit 5
