#!/bin/bash
# Taratura del bersaglio di consegna con lo strumento STANDARD di llama.cpp (26/9/2026 sera, Angelo: «prima di inventare,
# hai consultato la letteratura?»): llama-perplexity --kl-divergence-base / --kl-divergence, KL(teacher Q8 || modello) per
# token + percentili + accordo del token top-1. Riferimento: Dutta et al. 2024, "Accuracy is Not All You Need"
# (arXiv 2407.09141): la KL dal modello pieno misura la fedelta' di una quantizzazione meglio dell'accuratezza.
# Testo: wikitext-2 test (lo stesso della PPL 7,9330 del rivale), -c 4096, --chunks 20: il file .kld della base occupa
# ~497 KB per token valutato (vocabolario 248.320, uint16) -> 20 x 2047 token ~ 20 GB su @@ROOT:models@@ (76 GB liberi):
# limite dichiarato, i 72 chunk interi (~73 GB) non ci stanno.
# Accanto a EB2: un solo contesto GPU in piu', i modelli uno alla volta.
set -u
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)} DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}
B=@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin
TESTO=@@ROOT:odino@@/collaudo/corpus/wiki.test.raw
CHUNKS=${CHUNKS:-20}
D=@@ROOT:lavoro@@/e2e/fedelta/kld
BASE=@@ROOT:models@@/gguf/tony-tern/taratura_teacherQ8_wiki_c4096_ch${CHUNKS}.kld
mkdir -p "$D"
TEACHER=@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf
declare -A M=([B]=@@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf
              [rivale_Q2_K]=@@ROOT:models@@/gguf/thor-rivale/Ornith-1.5-35B-A3B-Q2_K.gguf
              [v20_downaggr3]=@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf)
ORDINE=${ORDINE:-"B rivale_Q2_K v20_downaggr3"}
# UB: i modelli ternari (TQ1_0R_1024) con -ub 2048 ACCANTO a EB2 hanno fatto scadere il ring di calcolo (reset alle
# 21:50:53 e 21:57:36, recuperati; EB2 intatto): dispatch piu' corti con -ub 512. L'ubatch non cambia i numeri, solo il taglio.
UB=${UB:-2048}
comune=(-ngl 999 -c 4096 -b 2048 -ub "$UB" -dev Vulkan0 -f "$TESTO" --chunks "$CHUNKS")

if [ ! -s "$BASE" ]; then
  echo "── $(date '+%H:%M:%S') base: teacher Q8 -> $BASE"
  LD_LIBRARY_PATH=$B "$B/llama-perplexity" -m "$TEACHER" "${comune[@]}" --kl-divergence-base "$BASE" \
    > "$D/base_teacherQ8.log" 2>&1 || { echo "⛔ base fallita"; tail -5 "$D/base_teacherQ8.log"; exit 2; }
  echo "   PPL teacher: $(grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' "$D/base_teacherQ8.log" | tail -1) · $(du -h "$BASE" | cut -f1)"
fi
for n in $ORDINE; do
  echo "── $(date '+%H:%M:%S') $n"
  LD_LIBRARY_PATH=$B "$B/llama-perplexity" -m "${M[$n]}" "${comune[@]}" --kl-divergence-base "$BASE" --kl-divergence \
    > "$D/kld_$n.log" 2>&1 || { echo "⛔ $n fallito"; tail -5 "$D/kld_$n.log"; continue; }
  sed -n '/====== Perplexity statistics/,$p' "$D/kld_$n.log" | grep -vE "^llama_|^ggml_|^$" | head -40
done
echo "── $(date '+%H:%M:%S') fine"
