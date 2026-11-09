#!/bin/bash
# 30/9 02:5x — sorvegliante di TT3 dopo le cure (codice TT3c-cfb718f). Sostituisce sorvegliante_TT3.sh, che:
#  (1) esportava col codice VECCHIO TT3-df753ee (la corsa ora gira con TT3c);
#  (2) guardava solo il grezzo migliore_*, mentre dal passo 2 la EMA e' migliore (p2: 0,436 vs 0,506; p4: 0,4265 vs 0,4304);
#  (3) usava il cancello KL4, che non predice l'MMLU (TT2 p4->p6: KL4 -12 %, MMLU fermo).
# Ora: a ogni nuovo passo P con un migliore, esporta la EMA se esiste migliore_ema_P (altrimenti il grezzo), e la manda
# SEMPRE al Dell (sonda prima, poi MMLU 2000): il giudice e' l'MMLU appaiato. Il passo 4 (TT3-p4e) e' esportato a mano.
set -u
D=@@ROOT:depot@@/e2e/allena/TT3; COD=@@ROOT:depot@@/e2e/codice/TT3c-cfb718f/thor_v2; PY=@@ROOT:venv@@/bin/python3
OUT=@@ROOT:models@@/gguf/tony-tern/TT3; L=@@ROOT:depot@@/tutto_ternario/log; S=@@ROOT:forge@@/thor_v2/strumenti
A2=@@ROOT:models@@/gguf/tony-tern/R1/Thor-T-R1.gguf; ULT=4
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
log() { echo "[$(date '+%F %T')] $*"; }
log "sorvegliante TT3c acceso (ultimo passo gestito: $ULT)"
while true; do
  P=$(ls -d $D/migliore_[0-9]* $D/migliore_ema_[0-9]* 2>/dev/null | grep -oE "[0-9]{6}$" | sort -u | tail -n 1)
  P=$((10#${P:-0}))
  if [ "$P" -gt "$ULT" ]; then
    sleep 120
    if [ -d $D/migliore_ema_$(printf %06d $P) ]; then M=$D/migliore_ema_$(printf %06d $P); N=TT3-p${P}e
    else M=$D/migliore_$(printf %06d $P); N=TT3-p$P; fi
    G=$OUT/Thor-T-$N.gguf
    if [ -f $L/dell_$N.log ]; then log "$N gia' mandato al Dell: salto"; ULT=$P; continue; fi
    avail=$(df -BG --output=avail @@ROOT:models@@ | tail -1 | tr -dc 0-9)
    if [ "$avail" -lt 20 ]; then log "⛔ @@ROOT:models@@ $avail GB liberi (< 20 = 2x taglia): export $N rimandato"; sleep 600; continue; fi
    log "nuovo $M (passo $P) -> $N"; t0=$(date +%s)
    "$S/governa_export.sh" systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=14G -p MemorySwapMax=2G nice -n 19 ionice -c3 \
      env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      bash -c "cd $COD && $PY e2e_esporta.py --sorgente $A2 --corsa $D --istantanea $(basename $M) --uscita $G.scale --device cpu" \
      > $L/export_$N.log 2>&1 || { log "⛔ export $N fallito"; ULT=$P; continue; }
    if [ -s $M/registri_pv.npz ]; then
      "$S/governa_export.sh" systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=14G nice -n 19 ionice -c3 \
        env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 bash -c \
        "cd $COD && $PY e2e_pv_codici.py esporta --scalato $G.scale --dst $G --registri $M/registri_pv.npz" >> $L/export_$N.log 2>&1 \
        || { log "⛔ export codici $N fallito"; rm -f $G.scale $G; ULT=$P; continue; }
      rm -f $G.scale
    else mv $G.scale $G; fi
    log "export $N fatto in $(( $(date +%s)-t0 )) s ($(sha256sum $G | cut -c1-16))"
    SONDA_PADRE=18 "$S/aragorn_lucchetto.sh" bash @@ROOT:depot@@/tutto_ternario/codice/dell_variante_sonda_prima.sh "$N" "$G" >> $L/dell_$N.log 2>&1
    log "Dell $N: $(grep -E 'cancello|MMLU FINITO' $L/dell_$N.log | tr '\n' ' ' | cut -c1-240)"
    ULT=$P
  fi
  sleep 60
done
