#!/bin/bash
# 1/11 12:20: continuazione 50->150 (ripresa da ottimizzatori_ultimo=50): T50 77,25 vs base 76,05, p 0,10 = dentro il rumore -> 3 tappe prima del giudizio (regola 27/10). --ore-max 9, tappe 100 e 150.
# 1/11: catena POST di C (T400POST: scale gate/up/down + F32 + residui densi rango 16, tutto dispiegabile esatto)
# applicata al B attuale (eb882K-SEL_pred_CARDAN, 76,05). Differenza da C: ssm_alpha/beta di B sono F32 -> niente
# E2E_DENSI_SSM_AB (prova a secco CPU 1/11: passo 0 kl24 0,248, gradienti F32 non nulli; OOM al backward solo per il tetto 9G).
# Ordine: lucchetto GPU (fra due prove del banco) -> MMLU-2000 del B base con la pipeline delle tappe -> 50 passi ->
# tappa 50 dispiegata + MMLU + McNemar contro il B base.
set -uo pipefail
. /home/angelo/tony-forgia/lucchetto.sh
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
W=/mnt/models/gguf/tony-tern/B_POST_1101; E=$W/allena; OC=/mnt/models/gguf/tony-tern/OBIETTIVO_C; COD=$OC/SINTESI_1025/BPP/codice; PY=/home/angelo/venv-catq/bin/python3
G0=/mnt/models/gguf/tony-tern/BANCO_1031/Thor-T-eb882K-SEL_pred_CARDAN.gguf; B=/mnt/models/build/thor-rexppiani2/build-rexp/bin
F=/home/angelo/tony-forgia; mkdir -p $E/tappe $W/base
riga(){ echo "[$(date '+%F %T')] $*" | tee -a $W/lanciatore.log; }
prendi_gpu "BPOST-B $$" 30; trap molla_gpu EXIT
guardia_ram || { until guardia_ram; do sleep 60; done; }
if [ "$(grep -c '^MCLP' $W/base/mmlu_B_base.log 2>/dev/null)" != 2000 ]; then
  riga "MMLU-2000 del B base (banco mescolato, Q8, rexppiani2)"
  LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 nice -n 5 $B/llama-perplexity -m $G0 -ngl 999 -c 4096 -np 4 -ub 512     -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f $F/mc_v2/mmlu2000_mescolato.bin > $W/base/mmlu_B_base.log.mesc 2>&1
  python3 $F/stato/parziale_mmlu.py --indice $F/mc_v2/mmlu2000_mescolato.indice.tsv --rimappa $W/base/mmlu_B_base.log $W/base/mmlu_B_base.log.mesc > $W/base/parziale.out 2>&1
  riga "B base: $(awk -F'\t' '$1=="MCLP"{n++; if($3==$4)k++} END{printf "%.2f (%d)", 100*k/n, n}' $W/base/mmlu_B_base.log)"
fi
ARGV="--uscita $E --gguf $G0 --nome B-POST --scale-down-lr 1e-4 --strati 0-39 --rango-esperti 0 --famiglie attn,gdn --rango-densi 16 --lr-densi 1e-5 --lr-gamma 1e-4 --tau 1.0 --lambda-ce 0.5 --topk 256 --kl4-peso 1 --kl-peso 0 --kl4-n 896 --modo congiunto --passi ${BPOST_PASSI:-50} --heldout-ogni 50 --pazienza 1000 --selezione kl24_paper --guardia-kl24 0.02 --tappe-dir $E/tappe --fuso --riprendi --kl4-rotazioni 4 --kl4-rotazioni-modo ciclo --kl4-rotazioni-dir $OC/SINTESI_1025/L2/kl4_rotazioni --allinea-peso 0.3 --allinea-modo rapporto --allinea-metrica cka --allinea-strati 0-39 --allinea-cache $OC/SINTESI_1025/L2/STATI_MAESTRO_wt103_1024tok --scale-gateup-lr 1e-4 --ricetta B3 --vram-tetto 60 --ore-max 9 --allena-f32 tutte"
ENV_VELOCE="E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 E2E_CACHE_DEQUANT=$W/cache_dequant E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton"
ENV_RIF="E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa E2E_MOTORE=triton2 E2E_SENZA_COESISTENZA=1 E2E_CACHE_DEQUANT=$W/cache_dequant E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton"
cd $COD
LIVELLI=("CARDAN_GRUPPI=1 E2E_RIUSO=1" "CARDAN_GRUPPI=1" ""); i=0; BASEENV=$ENV_VELOCE; rif=0; rc=1
riga "allenamento POST su B: ${BPOST_PASSI:-50} passi"
while :; do
  LIV=${LIVELLI[$i]}; echo "[$(date +%T)] livello [$LIV] base: $BASEENV" >> $E/corsa.log
  env $LIV $BASEENV HSA_ENABLE_SDMA=0 TORCHINDUCTOR_COMPILE_THREADS=1 $PY cardan_congiunto_scale.py $ARGV >> $E/corsa.log 2>&1
  rc=$?; echo "[$(date +%T)] livello [$LIV] uscita $rc" >> $E/corsa.log
  if [ $rc = 6 ] && [ $rif = 0 ]; then BASEENV=$ENV_RIF; rif=1; continue; fi
  if [ $rc = 2 ] && [ $i -lt 2 ]; then i=$((i + 1)); continue; fi
  break
done
riga "allenamento uscita $rc"
[ $rc = 0 ] || exit $rc
BPOST_TAPPE="100 150" BPOST_NON_ATTENDERE=1 bash $F/stato/bpost_b_tappe_1101.sh
riga "catena finita: $(tail -1 $W/bpost_tappe.log)"
printf '%s\tATTIVO\tB-POST-T150 (continuazione 50->150): %s\n' "$(date +%FT%H:%M)" "$(grep 'T50 (3)' $W/bpost_tappe.log | tail -1 | cut -c1-200)" >> $F/stato/cicli.tsv
