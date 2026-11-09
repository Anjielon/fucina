#!/bin/bash
# 18/10 — dopo l'allenamento del CARDAN CONGIUNTO (cardan_congiunto_1018.sh, unita' cardan-congiunto-1018):
# piega esperti (cardan_piega.py ORIGINALE, come il ripiego del wrapper C4) -> patch_gguf -> piega densi (cardan_piega_densi.py,
# CPU, come D2) -> patch_gguf -> taglia = ospite al byte + verifica al byte (cambiano SOLO gate/up esperti e densi attn/gdn)
# -> sonda 21 seed -> MMLU-2000 int8 (comando del banco C) -> McNemar + KL4 contro D2 71,75 e contro S4b 69,25.
# Comandi derivati da s5/cardan_ospite_c_veloce.sh:121-153 e stato/c_densi_cardan_1013.sh:247-283. Idempotente (ripresa).
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
OC=@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C; COD=$OC/CARDAN_C0B/codice; F=@@ROOT:forge@@; TT=@@ROOT:models@@/gguf/tony-tern
W=$OC/CARDAN_CONGIUNTO_1018; E=$W/allena; BASE=$OC/OBQ_FIBRATO_S4_1014/Thor-T-C-S23-s2-OBQFIB.gguf
PY=@@ROOT:venv@@/bin/python3; B=@@ROOT:models@@/build/thor-c0t/build-c0t/bin; PATCH=@@ROOT:models@@/build/leve/raffina/patch_gguf.py
MCN=$F/thor_v2/strumenti/mcnemar_mclp.py; K4PY=$TT/LEVE_RAFF_PV12P/diagnosi/kl4_appaiato.py; SONDA=$OC/TRIO/sonda.sh
RD2=$OC/DENSI2_SU_N3_1017/mmlu_DC.log; RS4=$OC/OBQ_FIBRATO_S4_1014/mmlu_info_OBQ.log
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
# 1 piega esperti (originale, seriale)
if [ ! -s $G ] && [ ! -s $GE ]; then
  PIEGA_ALGEBRA=cpu OMP_NUM_THREADS=$THR LEVE_DT=float32 LEVE_THREADS=$THR PIEGA_ORIGINALE_DIR=$COD nice -n 5 $PY -X faulthandler \
    $COD/cardan_piega.py --pt $CE --gguf $BASE --out $PE --device $DEVP --lotto 16 > $PE/piega.out 2>&1 \
    || { riga "ERRORE piega esperti ($PE/piega.out)"; exit 3; }
  env $SENZA_GPU nice -n 10 $PY $PATCH --dir $PE/tensori --out $GE --base $BASE > $PE/patch.out 2>&1 || { riga "ERRORE patch esperti"; exit 3; }
fi
[ -s $G ] || [ "$(stat -c %s $GE)" = "$(stat -c %s $BASE)" ] || { riga "ERRORE taglia piegato esperti"; exit 3; }
[ -s $PE/registro.jsonl ] && riga "piega esperti: $($PY -c "
import json; r=[json.loads(x) for x in open('$PE/registro.jsonl') if x.strip()]
d=sum(x['energia_correzione_H'] for x in r); f=sum(x['residuo_dopo_piega_H'] for x in r); print(f'{len(r)} famiglie, catturata {100*(1-f/d):.1f} %')")"
# 2 piega densi (CPU, come D2)
if [ ! -s $G ]; then
  env $SENZA_GPU PIEGA_ALGEBRA=cpu OMP_NUM_THREADS=$THR LEVE_THREADS=$THR nice -n 10 $PY -X faulthandler $COD/cardan_piega_densi.py \
    --delta $CD --gguf $GE --out $PD --device cpu >> $PD/piega.out 2>&1 || { riga "ERRORE piega densi ($PD/piega.out)"; exit 3; }
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
riga "verifica al byte: $(tail -1 $O/verifica_byte.txt)"; rm -f $GE; rm -rf $PE/tensori $PD/tensori
[ -n "${SOLO_PIEGA:-}" ] && { riga "SOLO_PIEGA: pronto $G (misura: TAPPA=$TAPPA senza SOLO_PIEGA)"; exit 0; }
# 4 sonda 21 seed
[ -n "$(giuste $O/sonda.out)" ] || GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_TQ1R_Q8=1 bash $SONDA $G C-S4b-CONGIUNTO > $O/sonda.out 2>&1
riga "sonda: $(giuste $O/sonda.out) (D2 14/21)"
# 5 MMLU-2000 int8
L=$O/mmlu_CONGIUNTO.log
[ "$(grep -c '^MCLP	' $L 2>/dev/null)" = 2000 ] || LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 nice -n 5 \
  $B/llama-perplexity -m $G -ngl 999 -c 4096 -np 4 -ub 512 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 \
  -f $F/mc_v2/mmlu_00000.bin > $L 2>&1
[ "$(grep -c '^MCLP	' $L)" = 2000 ] || { riga "ERRORE MMLU incompleto"; exit 4; }
molla_gpu; trap - EXIT
for n in D2:$RD2 S4b:$RS4; do k=${n%%:*}; r=${n#*:}
  python3 $MCN $r $L --json $O/mcnemar_vs_$k.json > /dev/null 2>&1
  env $SENZA_GPU python3 $K4PY $r $L > $O/kl4_vs_$k.txt 2>&1
  riga "vs $k: $(python3 -c "import json; d=json.load(open('$O/mcnemar_vs_$k.json')); print(f\"{d['acc_B']:.2f} vs {d['acc_A']:.2f}, vinte/perse {d['B_giusta_A_sbagliata']}/{d['A_giusta_B_sbagliata']}, p {d['p_mcnemar_esatto']:.4f}\")") | KL4 $(tail -1 $O/kl4_vs_$k.txt | cut -c1-100)"
done
riga "CHIUSO CARDAN congiunto da S4b: MMLU-2000 $(grep -o 'Final result: [0-9.]*' $L | tail -1) | $G"
