#!/bin/bash
# 10/10 (Angelo: «tutte le leve possibili... soprattutto ai modelli piu' piccoli»). La pipeline di combinazione_0910.sh
# (COMBINAZIONE_LEVE.md) sull'ospite eb882K (obiettivo B, 9.470.184.864 B, tetto 9.492.293.427):
#   B (GPU) 9 leve generiche in K su eb882K: jarq_pk jarq_p2k schur_pk schur_p2k tjarq_p1fermo bil_foem tied_jarq
#     bil_tjarq bil_tjarq_p1fermo (solo tensori .npy; BIL-H2 di PV12P = BIL sulle famiglie scelte: qui la selezione per
#     famiglia lo copre con bil_tjarq/bil_foem/bil_tjarq_p1fermo)
#   C (CPU) predittore K-FAC espanso IN eb882K (gradienti esatti AUDIT/prova_leve/grad_eb882K, affidabilita' dalle due meta')
#   D (CPU) selezione per famiglia + CARDAN eb882K in parallelo (piega gia' fatta: esperti, disgiunti dai densi) + tutte-meno-una
#   E (GPU) per candidato: GGUF (stessa taglia di eb882K), KL 24 righe + tg contro eb882K stessa sessione, MMLU-2000 SEMPRE
#     + McNemar e KL4 contro eb882K (74,90) e contro eb882K-CARDAN (75,15); cancello velocita'. GGUF cancellato dopo.
#   F (CPU) additivita' (misurato contro previsto, contributi tutte-meno-una)
# Lanciato da ~/tony-forgia/stato/leve_eb882k_1010.sh (che aspetta il turno e fa prima i gradienti).
set -uo pipefail
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus PYTHONDONTWRITEBYTECODE=1
TT=/mnt/models/gguf/tony-tern; C=$TT/COMBINAZIONE_eb882K; S=/home/angelo/tony-forgia/stato; F=/home/angelo/tony-forgia
PY=/home/angelo/venv-catq/bin/python3; COD=/mnt/models/build/leve/raffina; CB0=$TT/COMBINAZIONE
EB=$TT/TUTTE/eb882K.gguf; ON=eb882K; TETTO=9492293427
RIF_MMLU=$TT/CARDAN/kd_r16_eb882K/mmlu_eb882K.log; RIF_CA=$TT/CARDAN/kd_r16_eb882K/mmlu_CARDAN.log
RIF_KL=$TT/CARDAN/kd_r16_eb882K/kl/kl_eb882K.json
GR=$TT/AUDIT/prova_leve/grad_eb882K_kl4;   # 10/10 23:00: gradienti KL4 (risposta), non KL24
GR_KL24=$TT/AUDIT/prova_leve/grad_eb882K; CARD=$TT/CARDAN/kd_r16_eb882K/piega
K4PY=$TT/LEVE_RAFF_PV12P/diagnosi/kl4_appaiato.py; MCN=$F/thor_v2/strumenti/mcnemar_mclp.py
export TMPDIR=$C/_tmp; mkdir -p $TMPDIR $C/misura
cp "$0" $C/comando_combinazione.sh
exec >> $C/combinazione.log 2>&1
riga(){ local t; t=$(printf '%s' "$2" | tr '\t\n' '  '); printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$t" >> $S/cicli.tsv; echo "[$(date '+%F %T')] $1 $t"; }
cpu(){ HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= LEVE_THREADS=6 nice -n 10 "$@"; }
libero(){ df -BG --output=avail /mnt/models | tail -1 | tr -dc 0-9; }

ls $GR/g_*.npy >/dev/null 2>&1 && [ -s $GR/scale_kron.tsv ] || { riga ERRORE "COMBINAZIONE-EB882K: mancano i gradienti in eb882K ($GR)"; exit 2; }
riga ATTIVO "COMBINAZIONE-EB882K: gradienti eb882K presenti ($(ls $GR/g_*.npy | wc -l)), disco $(libero) GiB"

# ── B: leve generiche in K su eb882K (GPU; registro riprendibile per tensore) ──
LEVE_B="jarq_pk jarq_p2k schur_pk schur_p2k tjarq_p1fermo bil_foem tied_jarq bil_tjarq bil_tjarq_p1fermo"
# 11/10 10:30: le stesse leve calcolate sul Dell (stato/dell_leve_eb882k.sh) rientrano nei registri: qui si salta il fatto.
# Si aspetta fino a 15 min la fine naturale della linea Dell, poi la si ferma e si aspetta l'ultimo rientro.
if systemctl --user is-active -q dell-leve-eb882k.service; then
  w=0; while systemctl --user is-active -q dell-leve-eb882k.service && [ $w -lt 45 ]; do w=$((w+1)); sleep 20; done
  touch $S/FERMA_DELL_LEVE
  while systemctl --user is-active -q dell-leve-eb882k.service; do sleep 20; done
  riga ATTIVO "COMBINAZIONE-EB882K B: linea Dell chiusa; tensori gia' nei registri: $(for L in $LEVE_B; do printf '%s %s ' $L $(cat $C/$L/registro.jsonl 2>/dev/null | wc -l); done)"
fi
. $F/lucchetto.sh
until guardia_ram; do sleep 120; done
prendi_gpu "COMBINAZIONE-EB882K-B $$" 20; trap 'molla_gpu' EXIT
for L in $LEVE_B; do
  t1=$SECONDS
  LEVE_OSPITE=$EB LEVE_OUT=$C LEVE_DT=float32 LEVE_THREADS=6 LEVE_FOEM_FORMA=codice nice -n 5 $PY -u $COD/applica.py --leva $L --device cuda > $C/applica_$L.log 2>&1 \
    || { riga ERRORE "COMBINAZIONE-EB882K B: $L codice $? ($(tail -2 $C/applica_$L.log | tr '\n' ' ' | cut -c1-200))"; continue; }
  riga ATTIVO "COMBINAZIONE-EB882K B: $L su eb882K in $(( (SECONDS - t1) / 60 )) min GPU ($(wc -l < $C/$L/registro.jsonl) tensori)"
done
molla_gpu; trap - EXIT

# ── C: predittore IN eb882K su tutte le leve (CPU) ──
t1=$SECONDS; VARS=""
for L in $LEVE_B; do [ -d $C/$L/tensori ] && VARS="$VARS $L@$ON"; done
( cd $TT/AUDIT/prova_leve && OSPITE_GGUF=$EB OSPITE_NOME=$ON GRAD_DIR=$GR PUNTO=$ON COMB_DIR=$C cpu python3 predittore_kfac.py $C/kfac_comb.json $VARS > $C/kfac_comb.log 2>&1 ) \
  || { riga ERRORE "COMBINAZIONE-EB882K C: predittore fallito ($(tail -2 $C/kfac_comb.log | tr '\n' ' ' | cut -c1-200))"; exit 3; }
riga ATTIVO "COMBINAZIONE-EB882K C: predittore in eb882K in $(( (SECONDS - t1) / 60 )) min CPU: $(grep -E '^[a-zA-Z_0-9]+@' $C/kfac_comb.log | awk '{printf "%s %s; ", $1, $NF}')"

# ── D: selezione per famiglia + CARDAN eb882K in parallelo + tutte-meno-una ──
CA=""; [ "$(ls $CARD/tensori/blk.*.ffn_*_exps*.weight.npy 2>/dev/null | wc -l)" -ge 80 ] && CA="--cardan $CARD/tensori"
OSPITE_NOME=$ON COMB_DIR=$C cpu python3 $CB0/seleziona.py $C/kfac_comb.json $C --nome pred $CA --meno-una > $C/selezione.log 2>&1 \
  || { riga ERRORE "COMBINAZIONE-EB882K D: selezione fallita ($(tail -2 $C/selezione.log | tr '\n' ' '))"; exit 4; }
riga ATTIVO "COMBINAZIONE-EB882K D: $(head -1 $C/selezione.log) | CARDAN ${CA:+si}${CA:-no (piega incompleta)}"

# ── E: misura per candidato, uno alla volta (disco) ──
ORD=(); [ -n "$CA" ] && ORD+=(SEL_pred_CARDAN); ORD+=(SEL_pred)
for d in $C/SEL_pred_meno_*; do [ -d "$d" ] && ORD+=("$(basename $d)"); done
export C_TETTO=$TETTO C_RIF_KL=$RIF_KL C_RIF_NOME=$ON C_RIF_MMLU=$RIF_MMLU C_MMLU=1 C_MMLU_SEMPRE=1 C_RIF_BENCH=$ON=$EB
tgv(){ grep -E "tg128" "$1" 2>/dev/null | awk -F'|' '{print $(NF-1)}' | tr -d ' ' | head -1; }   # "54.40±1.32"
for N in "${ORD[@]}"; do
  w=0; until [ "$(libero)" -ge 25 ]; do
    [ $w = 0 ] && riga ATTIVO "COMBINAZIONE-EB882K E: $N attende disco (/mnt/models $(libero) GiB, servono 25: GGUF 8,8 + dump KL)"
    w=$((w + 1)); [ $w -gt 90 ] && { riga ERRORE "COMBINAZIONE-EB882K E: disco insufficiente da 3 h, mi fermo prima di $N"; exit 5; }; sleep 120; done
  G=$C/$N/Thor-T-eb882K-$N.gguf
  cpu $PY $COD/patch_gguf.py --dir $C/$N/tensori --out $G --base $EB > $C/$N/patch.log 2>&1 || { riga ERRORE "COMBINAZIONE-EB882K E: patch $N fallita"; continue; }
  [ "$(stat -c %s $G)" = "$(stat -c %s $EB)" ] && [ "$(stat -c %s $G)" -le $TETTO ] \
    || { riga ERRORE "COMBINAZIONE-EB882K E: $N taglia $(stat -c %s $G) B diversa da eb882K o sopra il tetto B"; rm -f $G; continue; }
  bash $S/combo_misura.sh $N $C/misura "$N=$G"
  rm -rf $C/misura/kl/_lavoro_$N
  # cancello velocita': stessa sessione, tg candidato non sotto eb882K oltre 2 sigma combinate (byte e tipi identici:
  # una differenza vera va spiegata, non accettata)
  vr=$(tgv $C/misura/bench_$ON.txt); vc=$(tgv $C/misura/bench_$N.txt)
  vel=$(python3 -c "
import sys
try:
    a,sa=[float(x) for x in '$vr'.split('±')]; b,sb=[float(x) for x in '$vc'.split('±')]
    print('OK' if b >= a - 2*(sa*sa+sb*sb)**.5 else 'SOTTO', f'{b:.2f} vs {a:.2f}')
except Exception: print('IGNOTA', '$vc vs $vr')")
  if [ -f $C/misura/mmlu_$N.log ]; then
    cpu $PY $MCN $RIF_CA $C/misura/mmlu_$N.log --json $C/misura/mcnemar_${N}_vs_CARDAN.json > /dev/null 2>&1
    riga ATTIVO "COMBINAZIONE-EB882K E: $N velocita' $vel t/s | KL4 vs eb882K: $(cpu $PY $K4PY $RIF_MMLU $C/misura/mmlu_$N.log 2>&1 | tail -1 | sed 's|.*/||') | vs CARDAN: $(cpu $PY $K4PY $RIF_CA $C/misura/mmlu_$N.log 2>&1 | tail -1 | sed 's|.*/||') McNemar $(python3 -c "import json;d=json.load(open('$C/misura/mcnemar_${N}_vs_CARDAN.json'));print(round(d['acc_B'],2),'vs',round(d['acc_A'],2),'vinte/perse',d['B_giusta_A_sbagliata'],d['A_giusta_B_sbagliata'],'p',round(d['p_mcnemar_esatto'],4))" 2>&1)"
  fi
  if [ "${vel%% *}" = SOTTO ]; then
    riga ATTIVO "COMBINAZIONE-EB882K E: $N velocita' SOTTO eb882K ($vel): GGUF tenuto per il bench ABBA a carico basso"
  else
    sha256sum $G > $G.sha256 && rm -f $G
  fi
done

# ── F: additivita' ──
OSPITE_NOME=$ON ADD_RIF_LOG=$RIF_MMLU ADD_CARDAN_LOG=$RIF_CA cpu python3 $CB0/additivita.py $C > $C/additivita.log 2>&1
riga CHIUSO "COMBINAZIONE-EB882K: misure finite; additivita' in $C/additivita.log: $(tail -3 $C/additivita.log | tr '\n' ';' | cut -c1-300)"
