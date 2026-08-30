#!/bin/bash
# LA MONTAGNA E' DELLO STRATO O DELLA PROIEZIONE?
#
# Due profili misurati, finora raccontati come due storie:
#   per proiezione: up 8.4226 · gate 8.5187 · up+gate 8.3493 · down 11.2405
#   per strato:     0 -> 10.95 · 20 -> 37.80 · 30 -> 8.69 · 39 -> 8.74
#
# Se al solo strato 20 il danno viene tutto da `down`, le due storie sono UNA:
# `down` scrive dritto nel flusso residuo, e allo strato 20 quel flusso conta
# piu' che altrove. Se invece anche `up` da solo esplode li', allora lo strato
# 20 e' fragile a QUALUNQUE perturbazione e la proiezione non c'entra — due
# fenomeni distinti, e il rimedio e' diverso.
#
# Controllo incluso: la stessa scomposizione su uno strato SANO (30). Se li'
# tutte e tre le proiezioni sono innocue, la differenza e' dello strato.
$HOME/odino-lab/odino/spazio_per.sh 12 gpu || exit 1
B=$HOME/build-llamacpp-tq1/build
M=/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf
W=$HOME/odino-lab/collaudo/corpus/wiki.test.raw
export LD_LIBRARY_PATH=$B/bin
m() {
  local ETICHETTA="$1"; shift
  local P
  P=$(env "$@" $B/bin/llama-perplexity -m $M -ngl 999 -c 2048 -b 2048 \
        --chunks 12 -f $W 2>&1 | grep -oP 'Final estimate: PPL = \K[0-9.]+')
  printf '%-34s PPL = %s\n' "$ETICHETTA" "${P:-ERRORE}"
}
echo "── lo strato malato, scomposto"
m "20: tutte e tre"      ODINO_P2_LAYERS=20-20
m "20: solo up"          ODINO_P2_LAYERS=20-20 ODINO_NO_P2_GATE=1 ODINO_NO_P2_DOWN=1
m "20: solo gate"        ODINO_P2_LAYERS=20-20 ODINO_NO_P2_UP=1 ODINO_NO_P2_DOWN=1
m "20: solo down"        ODINO_P2_LAYERS=20-20 ODINO_NO_P2_UP=1 ODINO_NO_P2_GATE=1
m "20: up+gate"          ODINO_P2_LAYERS=20-20 ODINO_NO_P2_DOWN=1
echo
echo "── controllo: lo strato sano, scomposto uguale"
m "30: tutte e tre"      ODINO_P2_LAYERS=30-30
m "30: solo down"        ODINO_P2_LAYERS=30-30 ODINO_NO_P2_UP=1 ODINO_NO_P2_GATE=1
echo "🏁 CHI FA LA MONTAGNA COMPLETO"
