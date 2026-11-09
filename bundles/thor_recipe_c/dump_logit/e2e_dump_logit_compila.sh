#!/usr/bin/env bash
# Compila e2e_dump_logit contro la build llama.cpp del fork k1024 (librerie condivise; RUNPATH
# alla cartella della build, come strumenti/dump_strati). Scritto il 25/9/2026.
# Solo CPU: g++ a nice 19 / ionice idle, un processo. Non tocca la build (sola lettura).
#
# Uso:  ./e2e_dump_logit_compila.sh [USCITA]      (default: accanto al sorgente)
#       LLAMA_K1024=/altra/build ./e2e_dump_logit_compila.sh
set -euo pipefail
QUI="$(cd "$(dirname "$0")" && pwd)"
LL="${LLAMA_K1024:-@@ROOT:models@@/build/llamacpp-tq1-k1024}"
USCITA="${1:-$QUI/e2e_dump_logit}"
LIB="$LL/build/bin"

for f in "$LL/include/llama.h" "$LL/ggml/include/ggml.h" "$LL/vendor/nlohmann/json.hpp" \
         "$LIB/libllama.so" "$LIB/libggml.so" "$LIB/libggml-base.so"; do
    [[ -e "$f" ]] || { echo "ERRORE: manca $f" >&2; exit 1; }
done

# stessi -O3 -DNDEBUG di llama-perplexity (build/tools/perplexity/.../flags.make)
nice -n 19 ionice -c3 g++ -std=c++17 -O3 -DNDEBUG -DLLAMA_SHARED -DGGML_SHARED \
    -Wall -Wextra -Wno-unused-function \
    -I"$LL/include" -I"$LL/ggml/include" -I"$LL/vendor" \
    "$QUI/e2e_dump_logit.cpp" -o "$USCITA" \
    -L"$LIB" -lllama -lggml -lggml-base -Wl,-rpath,"$LIB"

echo "compilato: $USCITA"
sha256sum "$QUI/e2e_dump_logit.cpp" "$USCITA"
echo "librerie:"
ldd "$USCITA" | grep -E "llama|ggml"
