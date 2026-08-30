#!/bin/bash
# IL DANNO DIPENDE DA QUANTI ROUTER STANNO A VALLE?
#
# Se il meccanismo e' la deriva dell'instradamento, un solo strato acceso deve
# fare tanto piu' danno quanto piu' e' PROFONDO il resto del modello sotto di
# lui: lo strato 0 ha 40 router a valle da disturbare, lo strato 39 ne ha uno.
#
# ⚠️ Lo strato 40 NON ha il secondo piano (i tensori `_exps2` coprono 0-39):
#    "solo strato 40 = identico al riferimento" era una tautologia, non un
#    risultato. Il vero ultimo strato e' il 39.
#
# Predizione secca, falsificabile: PPL(solo 0) > PPL(solo 10) > PPL(solo 20)
# > PPL(solo 30) > PPL(solo 39). Se invece il danno e' piatto, o peggiora in
# fondo, l'ipotesi dell'instradamento muore come le altre otto.
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
  printf '%-28s PPL = %s\n' "$ETICHETTA" "${P:-ERRORE}"
}
echo "── un solo strato acceso, dal primo all'ultimo"
m "spento (riferimento)"   ODINO_NO_P2=1
for L in 0 5 10 15 20 25 30 35 39; do
  m "solo strato $L"       ODINO_P2_LAYERS=$L-$L
done
echo
echo "── conferma delle due misure sorprendenti"
m "36-39 (i buoni)"        ODINO_P2_LAYERS=36-39
m "20-39 (i cattivi)"      ODINO_P2_LAYERS=20-39
m "tutti (0-39)"           ODINO_P2_LAYERS=0-39
echo "🏁 DANNO PER STRATO COMPLETO"
