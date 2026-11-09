#!/bin/bash
# 21/10 22:5x v10 = v9 + sonda in parallelo alla MMLU.
# 21/10 19:2x v9 = v8 + piega esperti con cardan_piega_lotti.py (identica AL BYTE all'originale cardan_piega.py, x2,8 per processo,
# fucina/levers/fold_prefetch_batches.yaml), massimo 3 processi (6 code KFD invece di 10). PIEGA_PIANI=<gguf> (sperimentale):
# piani di partenza da quel GGUF e bersaglio = BASE + correzione (piega cumulata); vuoto = come v8.
# 21/10 17:xx v8 (giro 2 guardia): COPIA DELLA VEGLIA con: HSA_ENABLE_SDMA=0 di default (0 code SDMA per piega), registro_unico.jsonl (D6),
# AUDIT D15 (energie ricalcolate dai byte pubblicati, non dal registro autodichiarato) dopo la piega esperti, niente cancellazione di
# tensori/ (D: l'inventario vieta di cancellare intermedi; si toglie solo GE, contenuto interamente in G).
# 21/10 12:3x COPIA CON VEGLIA (dopo_tappa_misura_gpu_1021_veglia.sh): identica all'originale salvo riserva RAM (D3),
# rilancio dei lavoratori morti (D2) e scadenza d'inattivita' delle pieghe esperti (stallo GPU tappa 150).
# 21/10 VERSIONE GPU (Angelo: tutto GPU, massima velocita'): congiunto su S7-T550-g10; esperti = cardan_piega_gpu.py
# PIEGA_ALGEBRA=gpu_sicura su N processi paralleli per strati (N da RAM disponibile); densi = cardan_piega_densi_gpu.py fp32 lotto 8.
# Collaudi: COLLAUDO_PIEGA_GPU_1020 (esperti stessa quota; densi 2,2e-4 byte diversi, catturata 65,64 vs 65,58 %).
# 18/10 — dopo l'allenamento del CARDAN CONGIUNTO (cardan_congiunto_1018.sh, unita' cardan-congiunto-1018):
# piega esperti (cardan_piega.py ORIGINALE, come il ripiego del wrapper C4) -> patch_gguf -> piega densi (cardan_piega_densi.py,
# CPU, come D2) -> patch_gguf -> taglia = ospite al byte + verifica al byte (cambiano SOLO gate/up esperti e densi attn/gdn)
# -> sonda 21 seed -> MMLU-2000 int8 (comando del banco C) -> McNemar + KL4 contro D2 71,75 e contro S4b 69,25.
# Comandi derivati da s5/cardan_ospite_c_veloce.sh:121-153 e stato/c_densi_cardan_1013.sh:247-283. Idempotente (ripresa).
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0} TORCHINDUCTOR_COMPILE_THREADS=1  # priorita' 0 (21/10): ogni worker inductor apre /dev/kfd e prende una coda SDMA
OC=@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C; COD=$OC/CARDAN_C0B/codice; F=@@ROOT:forge@@; TT=@@ROOT:models@@/gguf/tony-tern
[ "${PIEGA_DENSI_VELOCE:-1}" = 1 ] && DENSI_PY=cardan_piega_densi_gpu_veloce.py || DENSI_PY=cardan_piega_densi_gpu.py   # giro 7: una sola apertura GGUF, delta mmap (prep 213 s -> ~10 s, byte identici)
W=$OC/CARDAN_CONGIUNTO_S7T550_1020; E=$W/allena; BASE=$OC/S7_CONGIUNTO_T550_1020/Thor-T-C-S7-T550-g10.gguf
PY=@@ROOT:venv@@/bin/python3; B=@@ROOT:models@@/build/thor-c0t/build-c0t/bin; PATCH=@@ROOT:models@@/build/leve/raffina/patch_gguf.py
MCN=$F/thor_v2/strumenti/mcnemar_mclp.py; K4PY=$TT/LEVE_RAFF_PV12P/diagnosi/kl4_appaiato.py; SONDA=$OC/TRIO/sonda.sh
RD2=$OC/DENSI2_SU_N3_1017/mmlu_DC.log; RS4=$OC/S7_CONGIUNTO_T550_1020/Thor-T-C-S7-T550-g10.mmlu.log
SENZA_GPU="HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES="
CE=$E/cardan_migliore.pt; CD=$E/residui_delta.pt; O=$W
if [ -n "${TAPPA:-}" ]; then   # 18/10: misura di una tappa conservata (allena/tappe), uscite in $W/tappa_<p>
  O=$W/tappa_$TAPPA; mkdir -p $O; CE=$E/tappe/cardan_$TAPPA.pt; CD=$O/residui_delta.pt
  [ -s $CD ] || python3 $F/stato/tappa_a_delta_1018.py $E/tappe/residui_$TAPPA.pt $CD $TAPPA > $O/delta.out 2>&1 || exit 2; fi
PE=$O/piega_e; PD=$O/piega_d; GE=$PE/Thor-T-C-S4b-CONG-E.gguf; G=$PD/Thor-T-C-S4b-CONGIUNTO.gguf; mkdir -p $PE $PD
LOG=$O/dopo.log; riga(){ echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
giuste(){ grep -o 'giuste [0-9]*/21' "$1" 2>/dev/null | tail -1 | grep -o '[0-9]*/21'; }
# SOLO_PIEGA=1 (18/10): piega sulla CPU mentre l'allenamento usa la GPU (niente attesa, niente lucchetto); si ferma dopo
# la verifica al byte. La misura (sonda+MMLU) si lancia dopo con lo stesso script senza SOLO_PIEGA.
DEVP=cuda; THR=6
if [ -n "${SOLO_PIEGA:-}${MISURA_ORA:-}" ]; then DEVP=${DEV_PIEGA:-cpu}; THR=${THR_PIEGA:-12}; [ "$DEVP" = cpu ] && export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
else
riga "attesa fine allenamento (cardan-congiunto-1018)"
while systemctl --user is-active -q cardan-congiunto-1018; do sleep 60; done
[ -s $E/esportazione.json ] && [ -s $E/cardan_migliore.pt ] && [ -s $E/residui_delta.pt ] \
  || { riga "CHIUSO/ERRORE: esportazione assente ($(tail -1 $W/corsa.log | cut -c1-200))"; exit 2; }
riga "esportazione: $(head -c 300 $E/esportazione.json | tr '\n' ' ')"
. $F/lucchetto.sh; prendi_gpu "C-CONGIUNTO-DOPO $$" 30; trap molla_gpu EXIT
fi
# v10: piega DENSI anticipata dalla BASE in parallelo agli esperti (i densi del GE sono byte identici alla base: cambia
# solo gate/up degli esperti). Misurato T300 21/10: 507 s sovrapposti invece che in coda. Il passo densi sotto la aspetta
# e poi trova tutto fatto (salta le proiezioni gia' nel registro).
if [ ! -s $G ]; then mkdir -p $PD
  OMP_NUM_THREADS=6 LEVE_THREADS=6 nice -n 5 $PY -X faulthandler $COD/${DENSI_PY:-cardan_piega_densi_gpu.py} \
    --delta $CD --gguf $BASE --out $PD --device cuda --dtype float32 --lotto 8 >> $PD/piega.out 2>&1 & PDENSI=$!
fi
# 1 piega esperti (originale, seriale)
if [ ! -s $G ] && [ ! -s $GE ]; then
  # VEGLIA (D3): riserva host di ${VEGLIA_RISERVA_GIB:-8} GiB prima di contare 3 GiB a processo (MemAvailable/3 portava allo swap)
  NP=$(( ( $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) - ${VEGLIA_RISERVA_GIB:-8} ) / 3 )); [ $NP -gt ${VEGLIA_MAX_NP:-2} ] && NP=${VEGLIA_MAX_NP:-2}; [ $NP -lt 1 ] && NP=1
  riga "piega esperti GPU: $NP processi (lotti, identica al byte${PIEGA_PIANI:+, PIANI $PIEGA_PIANI})"; pids=(); fr=$((40 / NP))
  lancia(){ if [ -n "${PIEGA_PIANI:-}" ]; then PP="--gguf $PIEGA_PIANI --bersaglio $BASE"; else PP="--gguf $BASE"; fi
    PIEGA_ALGEBRA=cpu OMP_NUM_THREADS=4 PIEGA_PREP_FILI=6 LEVE_DT=float32 LEVE_THREADS=4 nice -n 5 $PY -X faulthandler \
      $COD/${PIEGA_MOTORE:-cardan_piega_lotti.py} --pt $CE $PP --out $PE --device cuda --lotto 16 --strati $1 >> $PE/piega_$1.out 2>&1 & }
  declare -A rng prove; for i in $(seq 0 $((NP-1))); do a=$((i*fr)); b=$(( i == NP-1 ? 39 : (i+1)*fr-1 ))
    lancia $a-$b; pids+=($!); rng[$!]=$a-$b; prove[$a-$b]=0; done
  # VEGLIA (stallo 21/10 tappa 150, D2): niente `wait` cieco. Ogni 30 s: un lavoratore uscito con errore viene rilanciato
  # sullo stesso intervallo (la piega salta le famiglie gia' scritte) fino a VEGLIA_RIPROVE volte; se nessun nuovo file
  # in tensori/ ne' riga nel registro per VEGLIA_INATTIVITA_S (default 1800 s = 4x la famiglia piu' lenta misurata,
  # 420 s) -> STALLO: TERM poi KILL ai SOLI lavoratori di questa piega, uscita 4 (l'orchestratore delle tappe riprende
  # il congiunto). Nessun altro processo toccato.
  impronta(){ { ls -l --time-style=+%s $PE/tensori 2>/dev/null; stat -c %s $PE/registro.jsonl 2>/dev/null; } | md5sum; }
  ult=$(impronta); t_ult=$(date +%s)
  while :; do vivi=(); for p in "${pids[@]}"; do
      if kill -0 $p 2>/dev/null; then vivi+=($p); continue; fi
      wait $p; rc=$?; r=${rng[$p]}; [ $rc = 0 ] && continue
      if [ ${prove[$r]} -lt ${VEGLIA_RIPROVE:-2} ]; then prove[$r]=$((prove[$r]+1)); riga "piega esperti $r: uscita $rc, rilancio ${prove[$r]}/${VEGLIA_RIPROVE:-2}"
        lancia $r; vivi+=($!); rng[$!]=$r
      else riga "ERRORE piega esperti GPU $r (rc $rc dopo ${prove[$r]} rilanci, $PE/piega_$r.out)"; kill ${vivi[@]} ${pids[@]} 2>/dev/null; exit 3; fi
    done; pids=("${vivi[@]}"); [ ${#pids[@]} = 0 ] && break
    sleep ${VEGLIA_PASSO_S:-30}; ora=$(impronta)
    if [ "$ora" != "$ult" ]; then ult=$ora; t_ult=$(date +%s)
    elif [ $(( $(date +%s) - t_ult )) -ge ${VEGLIA_INATTIVITA_S:-1800} ]; then
      riga "STALLO piega esperti: nessuna uscita da $(( $(date +%s) - t_ult )) s (${#pids[@]} lavoratori vivi: ${pids[*]}); TERM poi KILL"
      kill ${pids[@]} 2>/dev/null; sleep ${VEGLIA_PASSO_S:-30}; kill -9 ${pids[@]} 2>/dev/null; exit 4; fi
  done
  [ "$(ls $PE/tensori/*.npy | wc -l)" = 80 ] || { riga "ERRORE piega esperti: $(ls $PE/tensori/*.npy | wc -l)/80 tensori"; exit 3; }
  env $SENZA_GPU nice -n 10 $PY $PATCH --dir $PE/tensori --out $GE --base $BASE > $PE/patch.out 2>&1 || { riga "ERRORE patch esperti"; exit 3; }
fi
[ -s $G ] || [ "$(stat -c %s $GE)" = "$(stat -c %s $BASE)" ] || { riga "ERRORE taglia piegato esperti"; exit 3; }
# AUDIT D15: energie ricalcolate dai BYTE pubblicati (cardan_verifica_energie_gpu.py, riferimento = candidato = questa piega) e
# confrontate col registro: un byte ternario alterato con registro intatto fa fallire lo scarto. Su CPU, 1 filo, 2 famiglie (AUDIT_D15=0 per saltarlo).
if [ "${AUDIT_D15:-1}" = 1 ] && [ -z "${PIEGA_PIANI:-}" ] && [ ! -s $G ] && [ ! -s $PE/audit_byte.json ]; then
  riga "audit D15: energie dai byte (${AUDIT_D15_UNITA:-2} famiglie, CPU)"
  rm -rf $PE/audit_byte.tmp.json
  # v10 (21/10 23:2x): il controllo accetta al massimo --max-unita famiglie: gli si da' un SOTTOINSIEME vero (strato 5,
  # gate+up: stessi byte e stesse righe di registro) invece del registro intero di 80 (falliva sempre: «oltre --max-unita»)
  SUB=$PE/audit_sub; rm -rf $SUB; mkdir -p $SUB/tensori
  for f in gate up; do for x in exps exps2; do [ -e $PE/tensori/blk.5.ffn_${f}_${x}.weight.npy ] && ln -s $PE/tensori/blk.5.ffn_${f}_${x}.weight.npy $SUB/tensori/; done; done
  $PY -c "
import json
r={}
for l in open('$PE/registro.jsonl'):
    if l.strip():
        x=json.loads(l)
        if x['strato']==5: r[x['famiglia']]=l if l.endswith(chr(10)) else l+chr(10)
open('$SUB/registro.jsonl','w').writelines(r.values())"
  for t in $(seq 1 40); do
    env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PIEGA_ORIGINALE_DIR=$COD nice -n 19 $PY $COD/verifica_energie_cpu1.py \
      --gguf $BASE --checkpoint $CE --fase esperti --riferimento $SUB --candidato $SUB --toll-report ${AUDIT_D15_TOLL:-1e-4} \
      --max-unita ${AUDIT_D15_UNITA:-2} --out $PE/audit_byte.tmp.json > $PE/audit_byte.out 2>&1 && { mv $PE/audit_byte.tmp.json $PE/audit_byte.json; break; }
    grep -q "AMMISSIONE NEGATA" $PE/audit_byte.out && { sleep 30; continue; }; riga "ERRORE audit D15 ($PE/audit_byte.out)"; exit 3
  done
  [ -s $PE/audit_byte.json ] || { riga "ERRORE audit D15: ammissione negata 40 volte ($PE/audit_byte.out)"; exit 3; }
  riga "audit D15: $($PY -c "import json;d=json.load(open('$PE/audit_byte.json'));print(len(d['metriche']),'famiglie, scarto massimo registro',max(max(m[n]['scarto_report'] for n in ('riferimento','candidato')) for m in d['metriche']))")"
fi
[ -s $PE/registro.jsonl ] && riga "piega esperti: $($PY -c "
import json,os; R='$PE/registro_unico.jsonl'; R=R if os.path.exists(R) else '$PE/registro.jsonl'
r={}
for x in open(R):
    if x.strip(): j=json.loads(x); r[(j['strato'],j['famiglia'])]=j
r=list(r.values())
d=sum(x['energia_correzione_H'] for x in r); f=sum(x['residuo_dopo_piega_H'] for x in r); print(f'{len(r)} famiglie, catturata {100*(1-f/d):.1f} %')")"
# 2 piega densi (CPU, come D2)
[ -n "${PDENSI:-}" ] && { wait $PDENSI || riga "densi anticipati: uscita non zero (si rifanno sotto)"; }
if [ ! -s $G ]; then
  OMP_NUM_THREADS=6 LEVE_THREADS=6 nice -n 5 $PY -X faulthandler $COD/${DENSI_PY:-cardan_piega_densi_gpu.py} \
    --delta $CD --gguf $GE --out $PD --device cuda --dtype float32 --lotto 8 >> $PD/piega.out 2>&1 || { riga "ERRORE piega densi ($PD/piega.out)"; exit 3; }
  env $SENZA_GPU LEVE_OSPITE=$GE nice -n 10 $PY $PATCH --dir $PD/tensori --out $G --base $GE > $PD/patch.out 2>&1 || { riga "ERRORE patch densi"; exit 3; }
fi
[ "$(stat -c %s $G)" = "$(stat -c %s $BASE)" ] || { riga "ERRORE taglia finale"; exit 3; }
# 3 verifica al byte contro S4b: solo gate/up esperti + densi attn/gdn
env $SENZA_GPU nice -n 19 $PY - $BASE $G > $O/verifica_byte.txt 2>&1 <<'Q' || { riga "ERRORE verifica al byte ($O/verifica_byte.txt)"; exit 3; }
import re, sys, numpy as np
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
from gguf import GGUFReader
a, c = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta, tc = {t.name: t for t in a.tensors}, {t.name: t for t in c.tensors}
assert list(ta) == list(tc), "nomi/ordine diversi"
for n in ta:
    x, y = ta[n], tc[n]
    assert (x.tensor_type, list(x.shape), x.data_offset, x.n_bytes) == (y.tensor_type, list(y.shape), y.data_offset, y.n_bytes), n
camb = [n for n in ta if not np.array_equal(np.asarray(ta[n].data).view(np.uint8), np.asarray(tc[n].data).view(np.uint8))]
fam = r"blk\.\d+\.(ffn_gate_exps|ffn_up_exps|attn_q|attn_k|attn_v|attn_output|attn_qkv|attn_gate|ssm_out|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight"
fuori = [n for n in camb if not re.fullmatch(fam, n)]
print(f"tensori {len(ta)}; cambiati {len(camb)}; fuori famiglia {fuori}")
sys.exit(0 if camb and not fuori else 1)
Q
riga "verifica al byte: $(tail -1 $O/verifica_byte.txt)"; rm -f $GE   # GE e' contenuto in G; tensori/ conservati
[ -n "${SOLO_PIEGA:-}" ] && { riga "SOLO_PIEGA: pronto $G (misura: TAPPA=$TAPPA senza SOLO_PIEGA)"; exit 0; }
# 4 sonda 21 seed
# v10 (21/10 22:5x): sonda IN PARALLELO alla MMLU (entrambe Vulkan, nessuna coda KFD); il suo esito si scrive dopo la MMLU
[ -n "$(giuste $O/sonda.out)" ] || { GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_TQ1R_Q8=1 bash $SONDA $G C-S4b-CONGIUNTO > $O/sonda.out 2>&1 & PSONDA=$!; }
# 5 MMLU-2000 int8
L=$O/mmlu_CONGIUNTO.log
[ "$(grep -c '^MCLP	' $L 2>/dev/null)" = 2000 ] || LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 nice -n 5 \
  $B/llama-perplexity -m $G -ngl 999 -c 4096 -np 4 -ub 512 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 \
  -f $F/mc_v2/mmlu_00000.bin > $L 2>&1
[ -n "${PSONDA:-}" ] && wait $PSONDA; riga "sonda: $(giuste $O/sonda.out) (D2 14/21)"
[ "$(grep -c '^MCLP	' $L)" = 2000 ] || { riga "ERRORE MMLU incompleto"; exit 4; }
molla_gpu; trap - EXIT
for n in D2:$RD2 S4b:$RS4; do k=${n%%:*}; r=${n#*:}
  python3 $MCN $r $L --json $O/mcnemar_vs_$k.json > /dev/null 2>&1
  env $SENZA_GPU python3 $K4PY $r $L > $O/kl4_vs_$k.txt 2>&1
  riga "vs $k: $(python3 -c "import json; d=json.load(open('$O/mcnemar_vs_$k.json')); print(f\"{d['acc_B']:.2f} vs {d['acc_A']:.2f}, vinte/perse {d['B_giusta_A_sbagliata']}/{d['A_giusta_B_sbagliata']}, p {d['p_mcnemar_esatto']:.4f}\")") | KL4 $(tail -1 $O/kl4_vs_$k.txt | cut -c1-100)"
done
riga "CHIUSO CARDAN congiunto da S4b: MMLU-2000 $(grep -o 'Final result: [0-9.]*' $L | tail -1) | $G"
