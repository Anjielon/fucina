#!/bin/bash
# 10/10 23:00 (scadenza 11/10 21:00): gradienti della KL4 (risposta, calibrazione SELRISP disgiunta dal banco) invece della KL24 generale: lezione SEL_pred/FINALE-A/CARDAN-DENSI.
# PROVA LEVE vs PAPER (9/10): gradiente ESATTO della KL del modello intero + fattori di Kronecker dei 250 densi, misurati
# su PV12P (non su R2B3m), maestro q8-maestro (stesso maestro della KL-4 MMLU), 32 righe train, seme 7. Poi predittore
# K-FAC espanso in PV12P: decide se H1 (bil sulle 169 famiglie a 3 piani) peggiora e H2 (81 riallocate) migliora.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
P=/mnt/models/gguf/tony-tern/AUDIT/prova_leve; OUT=$P/grad_eb882K_kl4; mkdir -p $OUT
GU=/mnt/models/gguf/tony-tern/TUTTE/eb882K.gguf
TV=/mnt/models/gguf/tony-tern/RILQT/codice_shexp/recupero_7dabd6f/thor_v2   # studente CURATO 9/10: *4 ed exps2 con mappe, controllo di copertura
PY=/home/angelo/venv-catq/bin/python3; S=/home/angelo/tony-forgia/stato
riga(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $S/cicli.tsv; }
LIB=$(df --output=avail -B1G /mnt/models | tail -1); [ "$LIB" -ge 20 ] || { riga ERRORE "PROVA-GRAD-EB882K: /mnt/models $LIB GiB liberi, servono 20 (15 di g/A/B + 2,5 di meta')"; exit 3; }
export TMPDIR=$P/_tmp TRITON_CACHE_DIR=$P/_tmp/triton TORCHINDUCTOR_CACHE_DIR=$P/_tmp/inductor; mkdir -p $TMPDIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $P/cache_dequant   # niente scritture in /mnt/deposito
cp $0 $OUT/comando.sh
. /home/angelo/tony-forgia/lucchetto.sh; guardia_ram || exit 3; prendi_gpu "PROVA-GRAD-EB882K $$" 55; trap molla_gpu EXIT
riga ATTIVO "PROVA-GRAD-EB882K-KL4: gradienti KL4 + Kronecker dei densi di eb882K (896 domande di calibrazione) avviati"
t0=$SECONDS
cd /mnt/models/build/leve/raffina
GRAD_META=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True E2E_GGUF=$GU E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 \
E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 E2E_CACHE_DEQUANT=$P/cache_dequant PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TV" \
nice -n 5 "$PY" gradienti_densi.py --thor-v2 "$TV" --gguf $GU --teacher-cache /mnt/deposito/e2e_teacher/q8-maestro --out $OUT --seme 7 --obiettivo kl4 --calib-jsonl /mnt/models/gguf/tony-tern/SELRISP/calib/calib_mmlu.jsonl --calib-maestro /mnt/models/gguf/tony-tern/SELRISP/calib/mclp_maestro.log --calib-n 896 --calib-pacco 4096 > $OUT/esecuzione.out 2>&1
c=$?
molla_gpu; trap - EXIT
riga ATTIVO "PROVA-GRAD-EB882K: esito $c in $(( (SECONDS-t0)/60 )) min, $(ls $OUT/g_*.npy 2>/dev/null | wc -l) gradienti ($(tail -1 $OUT/esecuzione.out | cut -c1-150))"
[ $c = 0 ] || exit 4
exit 0   # il predittore in eb882K lo esegue la fase C di combinazione_eb882K.sh
