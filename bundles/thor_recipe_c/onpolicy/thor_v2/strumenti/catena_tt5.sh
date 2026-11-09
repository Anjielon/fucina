#!/bin/bash
# 30/9 — TT5 corta: dopo il passo 2 (ripresa_000002 completa e stabile) ferma TT5, esporta la EMA del passo 2 (o il grezzo
# se la EMA non esiste) col codice TT5 e la manda al Dell (sonda padre 18 = R2B3/R2B3m, poi MMLU 2000).
# NIENTE governatore: dopo lo stop la GPU e' a 0 e governa_export.sh lascerebbe l'export in pausa per sempre.
# CRITERIO SCRITTO PRIMA: successo se MMLU(TT5-p2) >= 75,85 (R2B3 74,85 + 1) con McNemar appaiato p < 0,05 contro R2B3
# (@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/R2B3.log). Altrimenti R2B3m resta il migliore.
# 30/9 (TT4 morto al salvataggio): se la corsa si ferma PRIMA del passo 2 (unita' non piu' attiva senza ripresa_000002),
# la catena lo dice e si ferma invece di aspettare per sempre.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=@@ROOT:forge@@/thor_v2/strumenti; D=@@ROOT:depot@@/e2e/allena/TT5; COD=$(cat $D/codice.txt)/thor_v2
PY=@@ROOT:venv@@/bin/python3; BASE=@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf; OUT=@@ROOT:models@@/gguf/tony-tern/TT5
L=@@ROOT:depot@@/tutto_ternario/log; R=$D/ripresa_000002.pt
riga() { printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%dT%H:%M')" "$1" "$2" >> @@ROOT:forge@@/stato/cicli.tsv; }
until [ -f $R ] && grep -q '"passo": 2' $D/ripresa_ultima.json 2>/dev/null; do
  stato() { systemctl --user show -p ActiveState --value e2e-allena-TT5 2>/dev/null; }
  if [ "$(stato)" != active ]; then
    sleep 240                                           # RestartSec=180: un riavvio automatico fa in tempo a ripartire
    if [ "$(stato)" != active ] && [ "$(stato)" != activating ] && [ ! -f $R ]; then
      echo "[$(date '+%F %T')] ⛔ e2e-allena-TT5 fermo senza ripresa_000002"; riga CHIUSO "⛔ TT5 fermo prima del passo 2 (catena_tt5 esce): vedere journalctl --user -u e2e-allena-TT5"; exit 1
    fi
  fi
  sleep 60
done
s1=0; s2=1; until [ "$s1" = "$s2" ]; do s1=$(stat -c %s $R); sleep 60; s2=$(stat -c %s $R); done; sleep 30
echo "[$(date '+%F %T')] ripresa_000002 stabile: fermo TT5"
systemctl --user stop e2e-allena-TT5; echo "[$(date '+%F %T')] TT5: $(systemctl --user is-active e2e-allena-TT5)"
if [ -d $D/migliore_ema_000002 ]; then M=$D/migliore_ema_000002; N=TT5-p2e; else M=$D/migliore_000002; N=TT5-p2; fi
[ -d $M ] || { echo "⛔ nessun migliore al passo 2 ($(ls $D | tr '\n' ' '))"; riga CHIUSO "⛔ TT5: nessun migliore al passo 2"; exit 1; }
G=$OUT/Thor-T-$N.gguf; mkdir -p $OUT
LIB=$(df -B1G --output=avail $OUT | tail -1 | tr -d ' ')
[ "$LIB" -ge 25 ] || { echo "⛔ $OUT: $LIB GiB liberi, ne servono 25 (scalato + finale)"; riga CHIUSO "⛔ TT5 export: disco $OUT pieno"; exit 1; }
echo "[$(date '+%F %T')] export $N da $M"
systemd-run --user --scope --quiet -p CPUQuota=1600% -p MemoryMax=16G -p MemorySwapMax=2G nice -n 10 \
  env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  bash -c "cd $COD && $PY e2e_esporta.py --sorgente $BASE --corsa $D --istantanea $(basename $M) --uscita $G.scale --device cpu" > $L/export_$N.log 2>&1 \
  || { echo "⛔ export fallito"; riga CHIUSO "⛔ TT5 export fallito ($L/export_$N.log)"; exit 1; }
if [ -s $M/registri_pv.npz ]; then
  systemd-run --user --scope --quiet -p CPUQuota=1600% -p MemoryMax=16G nice -n 10 \
    env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=16 bash -c \
    "cd $COD && $PY e2e_pv_codici.py esporta --scalato $G.scale --dst $G --registri $M/registri_pv.npz" >> $L/export_$N.log 2>&1 \
    || { echo "⛔ export codici fallito"; rm -f $G.scale $G; riga CHIUSO "⛔ TT5 export codici fallito"; exit 1; }
  rm -f $G.scale
else mv $G.scale $G; fi
echo "[$(date '+%F %T')] export $N fatto ($(sha256sum $G | cut -c1-16))"
riga APERTO "TT5 $N esportato ($G, $(stat -c %s $G) B): al Dell (sonda padre 18, poi MMLU 2000). Criterio: >= 75,85 con p<0,05 vs R2B3."
SONDA_PADRE=18 "$S/aragorn_lucchetto.sh" bash @@ROOT:depot@@/tutto_ternario/codice/dell_variante_sonda_prima.sh "$N" "$G" >> $L/dell_$N.log 2>&1
echo "[$(date '+%F %T')] Dell $N: $(grep -E 'cancello|MMLU FINITO' $L/dell_$N.log | tr '\n' ' ' | cut -c1-240)"
riga CHIUSO "TT5 $N Dell: $(grep -E 'cancello|MMLU FINITO' $L/dell_$N.log | tr '\n' ' ' | cut -c1-200)"
