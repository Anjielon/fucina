#!/bin/bash
# KLD standard (llama-perplexity, stessa base/testo della taratura) di un checkpoint E2E: export su CPU + misura accanto
# alla corsa (-ub 512, un contesto GPU in piu'). Uso: kld_checkpoint.sh <cartella migliore_XXXXXX>
set -u
IST=$1; CORSA=$(dirname "$IST"); NOME=$(basename "$CORSA")-$(basename "$IST")
QUI=@@ROOT:forge@@/thor_v2; PY=@@ROOT:venv@@/bin/python3
GGUF=@@ROOT:models@@/gguf/tony-tern/Thor-T-$NOME.gguf
if [ ! -s "$GGUF" ]; then
  XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=16G -p MemorySwapMax=2G \
    nice -n 19 ionice -c3 env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 \
    "$PY" "$QUI/e2e_esporta.py" --sorgente @@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf --corsa "$CORSA" \
    --istantanea "$(basename "$IST")" --uscita "$GGUF.scale" --device cpu > @@ROOT:lavoro@@/e2e/fedelta/kld/export_$NOME.log 2>&1 \
    || { echo "⛔ export fallito"; exit 3; }
  if [ -s "$IST/registri_pv.npz" ]; then          # checkpoint PV: anche i codici cambiati, verificati con ricarica al bit
    env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice -n 19 "$PY" "$QUI/e2e_pv_codici.py" esporta \
      --scalato "$GGUF.scale" --dst "$GGUF" --registri "$IST/registri_pv.npz" >> @@ROOT:lavoro@@/e2e/fedelta/kld/export_$NOME.log 2>&1 \
      || { echo "⛔ export dei codici fallito"; rm -f "$GGUF.scale" "$GGUF"; exit 3; }
    rm -f "$GGUF.scale"
  else
    mv "$GGUF.scale" "$GGUF"
  fi
fi
B=@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin
BASE=@@ROOT:models@@/gguf/tony-tern/taratura_teacherQ8_wiki_c4096_ch20.kld
LD_LIBRARY_PATH=$B "$B/llama-perplexity" -m "$GGUF" -ngl 999 -c 4096 -b 2048 -ub 512 -dev Vulkan0 \
  -f @@ROOT:odino@@/collaudo/corpus/wiki.test.raw --chunks 20 --kl-divergence-base "$BASE" --kl-divergence \
  > @@ROOT:lavoro@@/e2e/fedelta/kld/kld_$NOME.log 2>&1 || { echo "⛔ KLD fallita"; exit 4; }
R=$(grep -E "Mean *KLD|Same top p" @@ROOT:lavoro@@/e2e/fedelta/kld/kld_$NOME.log | tr -s ' ' | tr '\n' ' ')
printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "KLD standard $NOME: $R (B 0,3506 · Q2_K 0,2201 · bersaglio ~0,1115)" >> @@ROOT:forge@@/stato/cicli.tsv
echo "$NOME: $R"
