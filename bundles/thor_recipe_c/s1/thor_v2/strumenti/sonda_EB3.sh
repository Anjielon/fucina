#!/bin/bash
# 27/9 21:0x — sonda parlata (3 domande x 7 semi, ragionamento acceso; stessa di sonda_consegna_EB2-p12/ema16) sulle istantanee
# EMA di EB3, un passo si' e uno no (il primo, poi ogni 2). Il sorvegliante di EB3 cancella l'export precedente sul Corsair e il
# GGUF sul Dell dopo l'MMLU: qui si tiene un COLLEGAMENTO FISSO appena l'export compare (nome fuori dal suo schema di pulizia),
# poi, sotto il lucchetto del Dell: copia, sonda (sonda_parlata.bat), risultati in ~/tony-forgia, pulizia. Nessuna quota Claude.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
USC=@@ROOT:models@@/gguf/tony-tern; L=@@ROOT:forge@@/thor_v2/strumenti/aragorn_lucchetto.sh; OUT=@@ROOT:forge@@
FATTI=@@ROOT:lavoro@@/e2e/allena/sonda_EB3.fatti; touch $FATTI; PRIMO=""
log() { echo "[$(date +%F' '%T)] $*"; }
esegui_sonda() {  # <nome> <gguf locale collegato>
  local N=$1 G=$2 F=$(basename $2)
  timeout 1800 scp -q "$G" "aragorn:C:/tcq/thor_mmlu/gguf/$F" || { log "⛔ copia di $F sul Dell fallita"; return 1; }
  local a b; a=$(stat -c %s "$G"); b=$(ssh aragorn "for %I in (C:\\tcq\\thor_mmlu\\gguf\\$F) do @echo %~zI" | tr -d '\r')
  [ "$a" = "$b" ] || { log "⛔ taglia diversa sul Dell ($a contro $b)"; return 1; }
  ssh aragorn "del C:\\tcq\\thor_mmlu\\log\\sonda_$N.fine 2>nul & schtasks /create /f /tn sonda_$N /tr \"C:\\tcq\\thor_mmlu\\sonda_parlata.bat $N $F 23:59\" /sc once /st 23:59 /sd 01/01/2030 /rl limited && schtasks /run /tn sonda_$N" >/dev/null || return 1
  log "sonda $N partita sul Dell"
  until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\sonda_$N.fine" 2>/dev/null | grep -q FATTO; do sleep 60; done
  for f in sonda_$N.txt sonda_$N.log sonda_${N}_grezze.jsonl; do scp -q aragorn:C:/tcq/thor_mmlu/log/$f $OUT/ 2>/dev/null; done
  ssh aragorn "del C:\\tcq\\thor_mmlu\\gguf\\$F" >/dev/null 2>&1
  local R; R=$(grep -E "giuste [0-9]+/21" $OUT/sonda_$N.txt | tail -1 | tr -d '\r\n')   # 28/9: niente a-capo in cicli.tsv
  log "sonda $N: $R"
  printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "Sonda parlata $N (Dell, 3x7 semi, ragionamento): $R (bersaglio >= 19/21, 0 mute; EMA16 19/21, p12 20/21)" >> @@ROOT:forge@@/stato/cicli.tsv
}
export -f esegui_sonda log; export OUT
log "sonda EB3 accesa"
while true; do
  for G in $(ls $USC/Thor-T-EB3-passo_*-ema.gguf 2>/dev/null); do
    N=$(basename $G .gguf | sed 's/^Thor-T-//'); P=$((10#$(echo $N | sed -E 's/.*passo_0*([0-9]+)-ema/\1/')))
    grep -qx "$N" $FATTI && continue
    [ -z "$PRIMO" ] && PRIMO=$P
    if [ $(( (P - PRIMO) % 2 )) -ne 0 ]; then echo "$N" >> $FATTI; log "$N: saltata (una si' e una no)"; continue; fi
    # l'export e' finito quando il sorvegliante annuncia l'MMLU di QUESTO nome (scrive codici in coda: la taglia non basta)
    grep -q "MMLU su Aragorn di $N " @@ROOT:lavoro@@/e2e/allena/sorvegliante_EB3.log || continue
    K=$USC/Sonda-$N.gguf; ln -f $G $K && echo "$N" >> $FATTI && log "collegato $K"
    $L bash -c "esegui_sonda '$N' '$K'"; rm -f $K
  done
  sleep 30
done
