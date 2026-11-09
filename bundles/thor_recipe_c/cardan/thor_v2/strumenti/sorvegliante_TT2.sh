#!/bin/bash
# 29/9 00:0x — a ogni nuovo migliore_XXXXXX di TT2 (held-out pieno, passi pari): export su CPU col codice di TT2
# (collaudato: passo 0 = A2 al bit) e poi, sul Dell sotto il lucchetto, sonda PRIMA (cancello 0 mute, >=10/21) e MMLU 2000.
# Export a 8 fili: il freno della GPU di TT2 e' la PRESSIONE I/O (swap 34 G + lettura/scrittura di 16 GB), non i fili; export CORTO = freno corto (29/9 00:57, provato con SIGSTOP: io full 40 %->0,4 %, GPU 0->100).
# Cancello del banco costoso: l'MMLU parte solo se l'accuratezza mc held-out del passo supera quella dell'ultimo misurato.
set -u
D=@@ROOT:depot@@/e2e/allena/TT2; COD=@@ROOT:depot@@/e2e/codice/TT2-7c07fd6/thor_v2; PY=@@ROOT:venv@@/bin/python3
OUT=@@ROOT:models@@/gguf/tony-tern/TT2; L=@@ROOT:depot@@/tutto_ternario/log; S=@@ROOT:forge@@/thor_v2/strumenti
A2=@@ROOT:depot@@/tutto_ternario/gguf/Thor-T-A2.gguf; ULT=""; ACC_ULT=0; KL4_ULT=9
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
log() { echo "[$(date '+%F %T')] $*"; }
log "sorvegliante TT2 acceso"
# 29/9 06:3x: lo stato del cancello si RICOSTRUISCE dai log del Dell (prima ripartiva da 0 a ogni riavvio: il cancello si apriva da solo)
for f in $L/dell_TT2-p*.log; do
  [ -f "$f" ] && grep -q "MMLU FINITO" "$f" || continue
  PP=$(basename "$f" .log | sed 's/.*-p//'); A=$(grep -E "held-out mc al passo $PP:" $D/registro.log | tail -n 1 | grep -oE "accuratezza [0-9.]+" | grep -oE "[0-9.]+")
  [ -n "$A" ] && awk -v a="$A" -v b="$ACC_ULT" 'BEGIN{exit !(a>b)}' && ACC_ULT=$A
  K=$(grep -E "held-out mc al passo $PP:" $D/registro.log | tail -n 1 | grep -oE "KL4 [0-9.]+" | cut -d" " -f2)
  [ -n "$K" ] && awk -v a="$K" -v b="$KL4_ULT" 'BEGIN{exit !(a<b)}' && KL4_ULT=$K
done
log "cancello: ultimo misurato al Dell KL4 ${KL4_ULT} (mc ${ACC_ULT} %)"
while true; do
  M=$(ls -d $D/migliore_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -n 1)
  if [ -n "$M" ] && [ "$M" != "$ULT" ] && [ "$(basename $M)" != migliore_000000 ]; then
    sleep 120
    P=$((10#$(basename $M | cut -d_ -f2))); N=TT2-p$P; G=$OUT/Thor-T-$N.gguf
    if [ -f $L/dell_$N.log ]; then log "$N gia' esportato e mandato al Dell: salto"; ULT=$M; continue; fi
    ACC=$(grep -E "held-out mc al passo $P:" $D/registro.log | tail -n 1 | grep -oE "accuratezza [0-9.]+" | grep -oE "[0-9.]+")
    KL4=$(grep -E "held-out mc al passo $P:" $D/registro.log | tail -n 1 | grep -oE "KL4 [0-9.]+" | cut -d" " -f2)
    log "nuovo $M (passo $P, KL4 ${KL4:-?} contro ultimo misurato ${KL4_ULT}; mc ${ACC:-?} %)"
    t0=$(date +%s)
    "$S/governa_export.sh" systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=14G -p MemorySwapMax=2G nice -n 19 ionice -c3 \
      env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      bash -c "cd $COD && $PY e2e_esporta.py --sorgente $A2 --corsa $D --istantanea $(basename $M) --uscita $G.scale --device cpu" \
      > $L/export_$N.log 2>&1 || { log "⛔ export $N fallito"; ULT=$M; continue; }
    if [ -s $M/registri_pv.npz ]; then
      R=$M/registri_pv.npz
      "$S/governa_export.sh" systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=14G nice -n 19 ionice -c3 \
        env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 bash -c \
        "cd $COD && $PY e2e_pv_codici.py esporta --scalato $G.scale --dst $G --registri $R" >> $L/export_$N.log 2>&1 \
        || { log "⛔ export codici $N fallito"; rm -f $G.scale $G; ULT=$M; continue; }
      rm -f $G.scale
    else mv $G.scale $G; fi
    log "export $N fatto in $(( $(date +%s)-t0 )) s ($(sha256sum $G | cut -c1-16))"
    # 29/9 07:5x: cancello sulla KL4 (monotona con l'MMLU su A2, p2, p4, EB3 p20: 0,750/0,581/0,494/0,329 -> 56,9/64,7/~67/74,7),
    # non sull'accuratezza mc a 400 risposte (p2 50,00 > p4 49,00 ma MMLU appaiato p4 > p2: rumore di +-2,5 punti)
    if [ -n "$KL4" ] && awk -v a="$KL4" -v b="$KL4_ULT" 'BEGIN{exit !(a<b)}'; then
      SONDA_PADRE=9 "$S/aragorn_lucchetto.sh" bash @@ROOT:depot@@/tutto_ternario/codice/dell_variante_sonda_prima.sh "$N" "$G" >> $L/dell_$N.log 2>&1
      grep -q "MMLU FINITO" $L/dell_$N.log && { ACC_ULT=$ACC; KL4_ULT=$KL4; }
      log "Dell $N: $(grep -E 'cancello|MMLU FINITO' $L/dell_$N.log | tr '\n' ' ' | cut -c1-240)"
    else log "KL4 held-out non migliore dell'ultimo misurato: niente Dell per $N"; fi
    ls -t $OUT/Thor-T-TT2-p*.gguf 2>/dev/null | tail -n +3 | xargs -r rm -f
    ULT=$M
  fi
  sleep 60
done
