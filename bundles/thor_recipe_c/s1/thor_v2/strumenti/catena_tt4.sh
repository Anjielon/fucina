#!/bin/bash
# 30/9 06:0x — TT4 corta: dopo il passo 2 (ripresa_000002 completa e stabile) ferma TT4, esporta la EMA del passo 2
# (o il grezzo se la EMA non esiste) col codice TT3c e la manda al Dell (sonda padre 19 = R2, poi MMLU 2000).
# NIENTE governatore: dopo lo stop la GPU e' a 0 e governa_export.sh lascerebbe l'export in pausa per sempre.
# Criterio scritto prima: successo se MMLU(TT4-p2) >= R2 73,15 + 1 con McNemar p<0,05; poi si aggiungono le tabelle ternarie.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=@@ROOT:forge@@/thor_v2/strumenti; D=@@ROOT:depot@@/e2e/allena/TT4; COD=@@ROOT:depot@@/e2e/codice/TT3c-cfb718f/thor_v2
PY=@@ROOT:venv@@/bin/python3; BASE=@@ROOT:models@@/gguf/tony-tern/R2/Thor-T-R2.gguf; OUT=@@ROOT:models@@/gguf/tony-tern/TT4
L=@@ROOT:depot@@/tutto_ternario/log; R=$D/ripresa_000002.pt
until [ -f $R ] && grep -q '"passo": 2' $D/ripresa_ultima.json 2>/dev/null; do sleep 60; done
s1=0; s2=1; until [ "$s1" = "$s2" ]; do s1=$(stat -c %s $R); sleep 60; s2=$(stat -c %s $R); done; sleep 30
echo "[$(date '+%F %T')] ripresa_000002 stabile: fermo TT4"
systemctl --user stop e2e-allena-TT4; echo "[$(date '+%F %T')] TT4: $(systemctl --user is-active e2e-allena-TT4)"
if [ -d $D/migliore_ema_000002 ]; then M=$D/migliore_ema_000002; N=TT4-p2e; else M=$D/migliore_000002; N=TT4-p2; fi
[ -d $M ] || { echo "⛔ nessun migliore al passo 2 ($(ls $D | tr '\n' ' '))"; exit 1; }
G=$OUT/Thor-T-$N.gguf; mkdir -p $OUT
echo "[$(date '+%F %T')] export $N da $M"
systemd-run --user --scope --quiet -p CPUQuota=1600% -p MemoryMax=16G -p MemorySwapMax=2G nice -n 10 \
  env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  bash -c "cd $COD && $PY e2e_esporta.py --sorgente $BASE --corsa $D --istantanea $(basename $M) --uscita $G.scale --device cpu" > $L/export_$N.log 2>&1 \
  || { echo "⛔ export fallito"; exit 1; }
if [ -s $M/registri_pv.npz ]; then
  systemd-run --user --scope --quiet -p CPUQuota=1600% -p MemoryMax=16G nice -n 10 \
    env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=16 bash -c \
    "cd $COD && $PY e2e_pv_codici.py esporta --scalato $G.scale --dst $G --registri $M/registri_pv.npz" >> $L/export_$N.log 2>&1 \
    || { echo "⛔ export codici fallito"; rm -f $G.scale $G; exit 1; }
  rm -f $G.scale
else mv $G.scale $G; fi
echo "[$(date '+%F %T')] export $N fatto ($(sha256sum $G | cut -c1-16))"
SONDA_PADRE=19 "$S/aragorn_lucchetto.sh" bash @@ROOT:depot@@/tutto_ternario/codice/dell_variante_sonda_prima.sh "$N" "$G" >> $L/dell_$N.log 2>&1
echo "[$(date '+%F %T')] Dell $N: $(grep -E 'cancello|MMLU FINITO' $L/dell_$N.log | tr '\n' ' ' | cut -c1-240)"
