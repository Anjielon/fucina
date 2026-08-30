#!/bin/bash
# SI PUO' AGGIRARE LA MONTAGNA?
#
# Misurato (un solo strato acceso per volta, 12 blocchi, modello 35B):
#   spento 8.7204 · str.0 10.9524 · str.5 8.7277 · str.10 8.8125
#   str.15 20.1161 · str.20 37.8030 · str.25 22.5109
#   str.30 8.6860 · str.35 8.7743 · str.39 8.7370
#
# Il danno NON si accumula con la profondita': e' una montagna stretta
# centrata sullo strato 20, piu' una collina sullo strato 0. La banda 20-39
# misura 37.61, cioe' praticamente il solo strato 20 (37.80): gli altri
# diciannove non contribuiscono.
#
# Se e' davvero locale, accendere il secondo piano OVUNQUE TRANNE la montagna
# deve dare il guadagno di fedelta' senza il danno. Prova secca: le bande
# sane, misurate una per una. Se ognuna e' <= riferimento, il caso e' fatto e
# vale la pena aggiungere al motore un interruttore di ESCLUSIONE.
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
  printf '%-30s PPL = %s\n' "$ETICHETTA" "${P:-ERRORE}"
}
echo "── le bande sane, una per una"
m "spento (riferimento)"     ODINO_NO_P2=1
m "1-12 (sotto la montagna)" ODINO_P2_LAYERS=1-12
m "28-39 (sopra)"            ODINO_P2_LAYERS=28-39
m "30-39"                    ODINO_P2_LAYERS=30-39
echo
echo "── confini della montagna: dove comincia e dove finisce"
m "13-27 (la montagna sola)"  ODINO_P2_LAYERS=13-27
m "16-24 (il cuore)"          ODINO_P2_LAYERS=16-24
m "19-21"                     ODINO_P2_LAYERS=19-21
echo "🏁 AGGIRAMENTO COMPLETO"
