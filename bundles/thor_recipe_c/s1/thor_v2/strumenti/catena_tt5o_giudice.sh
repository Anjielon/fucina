#!/bin/bash
# 1/10 sera — GIUDICE di TT5 ripartita (TT5-ottimizzato): ogni 2 passi la EMA del passo (istantanea passo_N/ema) si
# esporta sulla CPU del Corsair (<= 4 fili, nice 19: la GPU e' di TT5) e va al Dell: sonda parlata, poi MMLU 2000
# (banco_ot.bat), McNemar appaiato contro R2B3m misurato sul Dell con lo STESSO binario. Zero token.
# Due modi (due unita' transienti):
#   raccogli : copia SUBITO ogni passo_N/ema con N pari in $STAGE (le istantanee tenute sono 3: non si perdono)
#   giudica  : per ogni N pari in ordine: export -> (QAT2 finito, Dell libero, lucchetto) -> sonda -> MMLU -> regole
# Priorita' del Dell (decisione 1/10): prima la misura QAT2 in corso (catena_QAT2_dell.sh, qat2_rumore_dell.sh), poi i
# checkpoint di TT5; le prove PR2 solo a Dell fermo (le prende il lucchetto aragorn.lock come tutti).
#
# REGOLE SCRITTE PRIMA (BUDGET.md §7 Corsia B, coordinatore 1/10 22:2x), riportate nel registro all'avvio:
#  R1 ADOTTA il primo checkpoint con MMLU >= 75,85, McNemar esatto p < 0,05 contro R2B3m (stesso binario), sonda 0 mute
#     e non peggiore del padre (18/21, Fisher unilaterale alfa 0,05). -> riga ADOTTA in cicli.tsv (la sostituzione del
#     modello di casa la decide Angelo: la catena NON tocca nient'altro).
#  R2 REGRESSIONE: due misure consecutive con MMLU significativamente SOTTO R2B3m (p < 0,05) -> ferma e2e-allena-TT5
#     (systemctl --user stop) e lo scrive; si torna all'ultimo checkpoint non peggiore (decisione di Angelo).
#     (Sonda fallita = misura non valida per l'MMLU, conta come allarme: due sonde fallite di fila -> stesso stop.)
#  R3 PLATEAU: 3 misure di fila (6 passi) senza miglioramento significativo (p < 0,05 contro R2B3m) -> PLATEAU scritto
#     in cicli.tsv con la curva; la corsa NON si ferma da sola (lo decide Angelo).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=@@ROOT:forge@@/stato; T=@@ROOT:forge@@/thor_v2/strumenti
COD=@@ROOT:depot@@/e2e/codice/TT5-ottimizzato/thor_v2; PY=@@ROOT:venv@@/bin/python3
D=@@ROOT:depot@@/e2e/allena/TT5; I=@@ROOT:depot@@/e2e_istantanee/TT5
BASE=@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf
C=@@ROOT:depot@@/e2e/catena_tt5o; STAGE=$C/stage; OUT=@@ROOT:models@@/gguf/tony-tern/TT5o; mkdir -p $C $STAGE $OUT
REG=$C/registro.log; ESITI=$C/esiti.tsv
PRIMO=${PRIMO:-4}; OGNI=${OGNI:-2}; PADRE=${SONDA_PADRE:-18}
log(){ echo "[$(date '+%F %T')] $*" >> $REG; }
riga(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $S/cicli.tsv; log "$1 $2"; }
dell(){ timeout 120 ssh aragorn "$1"; }
compito(){   # NOME_TASK "COMANDO" FILE_FINE -> RC (data FUTURA + cancellazione: niente ripartenze alle 23:59)
  local tn=$1 cmd=$2 fine=$3 t0; t0=$(date +%s)
  dell "del C:\\tcq\\thor_mmlu\\log\\$fine 2>nul & schtasks /create /f /tn $tn /tr \"$cmd\" /sc once /st 23:59 /sd 01/01/2030 /rl limited && schtasks /run /tn $tn" >/dev/null || { RC=schtasks; return 9; }
  sleep 20
  until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$fine" 2>/dev/null | grep -q FATTO; do
    [ $(( $(date +%s) - t0 )) -gt 18000 ] && { RC=oltre5h; dell "schtasks /end /tn $tn & schtasks /delete /f /tn $tn" >/dev/null; return 8; }
    sleep 60; done
  RC=$(timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$fine" | grep -o 'rc=[0-9-]*' | tail -1)
  dell "schtasks /delete /f /tn $tn" >/dev/null
}
dell_libero(){ ! timeout 30 ssh aragorn "powershell -NoProfile -Command \"Get-Process llama-* -ErrorAction SilentlyContinue\"" | grep -q llama; }
qat2_vivo(){ pgrep -f "catena_QAT2_dell.sh|qat2_rumore_dell.sh" >/dev/null; }
bin_sha(){ timeout 300 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\sorgente\\build\\bin\\Release\\llama-perplexity.exe SHA256" | sed -n 2p | tr -d '\r '; }
mclp_ok(){ [ -s "$1" ] && [ "$(grep -c '^MCLP	' "$1")" -eq 2000 ] && grep -qE 'Final result: [0-9.]+' "$1"; }
# TT5 «viva» anche durante una fermata programmata al confine di passo (file $C/pausa, messo da confine_tt5_piegati.sh)
tt5_vivo(){ [ -f $C/pausa ] || systemctl --user is-active -q e2e-allena-TT5 || [ "$(systemctl --user show -p ActiveState --value e2e-allena-TT5)" = activating ]; }

raccogli(){
  local N=$PRIMO
  log "raccogli: da passo $N ogni $OGNI"
  while :; do
    local P; P=$(printf 'passo_%06d' $N)
    if [ -f $I/$P/ema/meta.json ] && grep -q "istantanea del passo $N:" $D/registro.log 2>/dev/null; then
      if [ ! -d $STAGE/$P ]; then
        cp -a $I/$P/ema $STAGE/$P.parziale && mv $STAGE/$P.parziale $STAGE/$P && log "raccolto $P (ema, $(du -sh $STAGE/$P | cut -f1))"
      fi
      N=$((N + OGNI)); continue
    fi
    # il passo N saltato (istantanea SALTATA per spazio o errore) o gia' oltre: si passa al successivo pari
    if grep -qE "istantanea del passo $N (SALTATA|NON salvata)" $D/registro.log 2>/dev/null; then
      log "⛔ istantanea del passo $N non scritta: niente giudizio per $N"; N=$((N + OGNI)); continue; fi
    if ! tt5_vivo; then sleep 300; tt5_vivo || { log "raccogli: TT5 fermo, esco"; exit 0; }; fi
    sleep 120
  done
}

baseline(){   # -> $BASE_LOG valido e binario registrato
  until ! qat2_vivo && dell_libero; do sleep 300; done
  local q=$S/qat2/R2B3m_q2bin.log b; b=$(bin_sha)
  if mclp_ok $q && [ -f $S/qat2/bin_sha.txt ] && [ "$(cat $S/qat2/bin_sha.txt)" = "$b" ]; then BASE_LOG=$q; fi
  if [ -z "${BASE_LOG:-}" ] && mclp_ok $q && [ ! -f $S/qat2/bin_sha.txt ]; then
    # QAT2 l'ha misurato sul binario qat2-bias senza annotarne lo sha: si accetta se nessuna build e' piu' recente del log
    BASE_LOG=$q; log "baseline: $q (sha del binario non annotato da QAT2; binario ora $b)"; fi
  if [ -z "${BASE_LOG:-}" ]; then
    log "baseline: R2B3m sullo stesso binario non disponibile: la misuro io (R2B3m_tt5o)"
    exec 9>@@ROOT:forge@@/stato/aragorn.lock; flock 9
    compito thor_mmlu_R2B3m_tt5o "C:\\tcq\\thor_mmlu\\banco_ot.bat R2B3m_tt5o Thor-T-R2B3m.gguf" R2B3m_tt5o.fine
    scp -q "aragorn:C:/tcq/thor_mmlu/log/R2B3m_tt5o.log" $C/R2B3m_tt5o.log; flock -u 9
    mclp_ok $C/R2B3m_tt5o.log || { riga ERRORE "TT5o giudice: baseline R2B3m fallita ($RC)"; exit 3; }
    BASE_LOG=$C/R2B3m_tt5o.log
  fi
  BIN=$b; echo "$BIN" > $C/bin_sha_baseline.txt
  riga ATTIVO "TT5o giudice: baseline R2B3m = $BASE_LOG ($(grep -oE 'Final result: [0-9.]+' $BASE_LOG | tail -1)), binario Dell ${BIN:0:16}"
}

giudica(){
  local N=$PRIMO reg=0 sonde_ko=0 senza=0
  [ -f $ESITI ] || printf 'passo\tmmlu\tbase\tvinte\tperse\tp\tsonda\tverdetto\n' > $ESITI
  riga ATTIVO "TT5o giudice avviato (passi pari da $PRIMO). Regole scritte prima: R1 adotta >=75,85 p<0,05 vs R2B3m stesso binario + sonda 0 mute non peggiore del padre ${PADRE}/21; R2 due regressioni significative di fila -> stop TT5; R3 6 passi (3 misure) senza miglioramento significativo -> PLATEAU (decide Angelo)"
  BASE_LOG=""
  while :; do
    local P G NOME; P=$(printf 'passo_%06d' $N); NOME=TT5o-p${N}e; G=$OUT/Thor-T-$NOME.gguf
    until [ -d $STAGE/$P ]; do
      if ! tt5_vivo && ! pgrep -f "catena_tt5o_giudice.sh raccogli" >/dev/null; then log "giudica: TT5 e raccolta fermi, esco"; exit 0; fi
      sleep 120; done
    # export (CPU <= 4 fili; la GPU e' di TT5)
    if [ ! -s $G ]; then
      log "export $NOME"
      local LIB; LIB=$(df -B1G --output=avail $OUT | tail -1 | tr -d ' ')
      [ "$LIB" -ge 25 ] || { riga ERRORE "TT5o export $NOME: $OUT $LIB GiB liberi"; exit 3; }
      systemd-run --user --scope --quiet -p CPUQuota=400% -p MemoryMax=16G -p MemorySwapMax=2G nice -n 19 \
        env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 bash -c "cd $COD && $PY e2e_esporta.py --sorgente $BASE --corsa $STAGE --istantanea $P \
        --uscita $G.scale --device cpu" > $C/export_$NOME.log 2>&1 \
        || { riga ERRORE "TT5o export $NOME fallito ($C/export_$NOME.log)"; N=$((N + OGNI)); continue; }
      if [ -s $STAGE/$P/registri_pv.npz ]; then
        systemd-run --user --scope --quiet -p CPUQuota=400% -p MemoryMax=16G nice -n 19 \
          env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 bash -c \
          "cd $COD && $PY e2e_pv_codici.py esporta --scalato $G.scale --dst $G --registri $STAGE/$P/registri_pv.npz" \
          >> $C/export_$NOME.log 2>&1 || { riga ERRORE "TT5o export codici $NOME fallito"; rm -f $G.scale $G; N=$((N + OGNI)); continue; }
        rm -f $G.scale
      else mv $G.scale $G; fi
      log "export $NOME fatto ($(sha256sum $G | cut -c1-16))"
    fi
    [ -n "$BASE_LOG" ] || baseline
    until ! qat2_vivo && dell_libero; do sleep 300; done
    local b; b=$(bin_sha)
    if [ "$b" != "$BIN" ]; then riga ATTIVO "TT5o giudice: binario Dell cambiato (${BIN:0:12} -> ${b:0:12}): rimisuro la baseline"; BASE_LOG=""; rm -f $C/R2B3m_tt5o.log; baseline; fi
    exec 9>@@ROOT:forge@@/stato/aragorn.lock; flock 9
    local F; F=$(basename $G); local SHA; SHA=$(sha256sum $G | cut -d' ' -f1)
    timeout 3600 scp -q $G "aragorn:C:/tcq/thor_mmlu/gguf/$F" && \
      [ "$(timeout 600 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$F SHA256" | sed -n 2p | tr -d '\r ')" = "$SHA" ] \
      || { flock -u 9; riga ERRORE "TT5o $NOME: copia sul Dell fallita"; sleep 600; continue; }
    compito sonda_$NOME "C:\\tcq\\thor_mmlu\\sonda_parlata.bat $NOME $F 23:59" sonda_$NOME.fine
    for f in sonda_$NOME.txt sonda_$NOME.log; do timeout 300 scp -q "aragorn:C:/tcq/thor_mmlu/log/$f" $C/ 2>/dev/null; done
    local RS GS MS PS; RS=$(grep -E "mute [0-9]+/21" $C/sonda_$NOME.txt 2>/dev/null | tail -1 | tr -d '\r')
    GS=$(echo "$RS" | grep -oE "giuste [0-9]+" | grep -oE "[0-9]+"); MS=$(echo "$RS" | grep -oE "mute [0-9]+" | grep -oE "[0-9]+")
    PS=$(python3 -c "from scipy.stats import fisher_exact as f; a=int('${GS:-0}'); b=int('$PADRE'); print(round(f([[a,21-a],[b,21-b]],alternative='less')[1],4))")
    local SONDA="giuste ${GS:-?}/21 mute ${MS:-?} Fisher p $PS"
    if [ -z "$GS" ] || [ "${MS:-99}" -gt 0 ] || awk -v p="$PS" 'BEGIN{exit !(p<0.05)}'; then
      dell "del C:\\tcq\\thor_mmlu\\gguf\\$F" >/dev/null; flock -u 9
      sonde_ko=$((sonde_ko + 1)); senza=$((senza + 1))
      printf '%s\t-\t-\t-\t-\t-\t%s\tSONDA_KO\n' $N "$SONDA" >> $ESITI
      riga ATTIVO "TT5o $NOME: sonda NON superata ($SONDA): niente MMLU (sonde KO di fila: $sonde_ko)"
      if [ $sonde_ko -ge 2 ]; then systemctl --user stop e2e-allena-TT5
        riga CHIUSO "TT5o R2: due sonde fallite di fila (ultima passo $N) -> e2e-allena-TT5 FERMATO. Ultimo checkpoint non peggiore: vedere $ESITI"; exit 0; fi
      N=$((N + OGNI)); continue
    fi
    sonde_ko=0
    compito thor_mmlu_$NOME "C:\\tcq\\thor_mmlu\\banco_ot.bat $NOME $F" $NOME.fine
    timeout 600 scp -q "aragorn:C:/tcq/thor_mmlu/log/$NOME.log" $C/$NOME.log
    dell "del C:\\tcq\\thor_mmlu\\gguf\\$F" >/dev/null; flock -u 9
    if ! mclp_ok $C/$NOME.log; then riga ERRORE "TT5o $NOME: MMLU incompleto ($RC)"; N=$((N + OGNI)); continue; fi
    python3 $T/mcnemar_mclp.py $BASE_LOG $C/$NOME.log --json $C/mcnemar_$NOME.json > /dev/null
    read -r ACC ACC0 V PP PV < <(python3 -c "import json;d=json.load(open('$C/mcnemar_$NOME.json'));print(f\"{d['acc_B']:.2f} {d['acc_A']:.2f} {d['B_giusta_A_sbagliata']} {d['A_giusta_B_sbagliata']} {d['p_mcnemar_esatto']:.4g}\")")
    local VERD=neutro
    if awk -v a=$ACC -v b=$ACC0 -v p=$PV 'BEGIN{exit !(a>b && p<0.05)}'; then VERD=meglio; senza=0; else senza=$((senza + 1)); fi
    if awk -v a=$ACC -v b=$ACC0 -v p=$PV 'BEGIN{exit !(a<b && p<0.05)}'; then VERD=peggio; reg=$((reg + 1)); else reg=0; fi
    if [ $VERD = meglio ] && awk -v a=$ACC 'BEGIN{exit !(a>=75.85)}'; then VERD=ADOTTA; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $N $ACC $ACC0 $V $PP $PV "$SONDA" $VERD >> $ESITI
    riga ATTIVO "TT5o $NOME: MMLU $ACC contro R2B3m $ACC0 (stesso binario), vinte/perse $V/$PP, p=$PV; sonda $SONDA -> $VERD"
    if [ $VERD = ADOTTA ]; then riga CHIUSO "TT5o R1 ADOTTA $NOME ($G): MMLU $ACC >= 75,85, p=$PV < 0,05, sonda ok. Sostituzione del modello: decisione di Angelo"; fi
    if [ $reg -ge 2 ]; then systemctl --user stop e2e-allena-TT5
      riga CHIUSO "TT5o R2: due regressioni significative di fila (passo $N: $ACC vs $ACC0, p=$PV) -> e2e-allena-TT5 FERMATO. Ultimo checkpoint non peggiore: vedere $ESITI"; exit 0; fi
    if [ $senza -ge 3 ]; then riga ATTIVO "TT5o R3 PLATEAU: $senza misure di fila (6 passi) senza miglioramento significativo; curva in $ESITI: $(tail -n +2 $ESITI | awk -F'\t' '{printf "p%s %s; ", $1, $2}'). Decide Angelo"; senza=0; fi
    # disco: si tengono gli ADOTTA e gli ultimi 2 export
    ls -t $OUT/Thor-T-TT5o-p*e.gguf 2>/dev/null | tail -n +3 | while read -r x; do
      n=$(basename $x .gguf | sed 's/Thor-T-TT5o-p//; s/e$//'); grep -qP "^$n\t.*\tADOTTA$" $ESITI || rm -f "$x"; done
    rm -rf $STAGE/$P
    N=$((N + OGNI))
  done
}

case "${1:-}" in
  raccogli) raccogli ;;
  giudica) giudica ;;
  *) echo "uso: $0 raccogli|giudica"; exit 2 ;;
esac
