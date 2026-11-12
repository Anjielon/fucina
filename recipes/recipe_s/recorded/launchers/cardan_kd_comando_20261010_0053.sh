#!/bin/bash
# CARDAN-KD on ANY host (9/10 evening, generalised copy of cardan_corsa.sh for the <9 GiB host eb882K): same recipe as kd_r16 on PV12P,
# CARDAN-KD (9/10): distillation of CARDAN (arXiv 2609.21137 §3.3) in the TERNARY flow on the host, then FOLD into the
# existing expert planes/scales (same tensors, same bytes: the engine runs it at the host's speed), then the proofs.
#   0  preflight (A.1 basis complete, CPU tests already green: codice/test_cardan_kd.py)
#   1  GPU lock (waits: BIL and the kernel benches keep their turn)
#   2  realised KL int8 of PV12P and COMBO A on the SAME engine (thor-rexppiani2 + e2e_dump_logit_p2, 24 held-out rows)
#   3  training (cardan_kd.py): identity gate at the bit (C = 0) + coherence gate KL torch step 0 vs engine (3 %)
#   4  fold (cardan_piega.py, GPU fp32 batched JARQ) + patch_gguf on PV12P (same names/types/shapes/bytes)
#   5  realised KL int8 of the folded GGUF + post-fold coherence gate (KL torch of the folded GGUF vs engine, 3 %)
#   6  llama-bench tg128/pp512 int8, PV12P and CARDAN in the same session
# MMLU-2000 is NOT run here: the command is written at the end (comando_mmlu.sh).
# Launch: systemd-run --user --unit cardan-kd --collect -p MemoryMax=20G bash /mnt/models/gguf/tony-tern/CARDAN/cardan_corsa.sh
# Env: CK_ORE (training cap, default 3), CK_PASSI (default 300), CK_RANGO (16), CK_OUT (default CARDAN/kd_r16)
#      CK_DA_PIEGA=1: resume FROM STEP 4 (training already done: cardan_migliore.pt + kl/kl_PV12P.json, kl_COMBO_A.json)
#      CK_MMLU=1: run comando_mmlu.sh at the end + KL on the MMLU answer (kl4_appaiato.py) vs PV12P
set -uo pipefail
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus PYTHONDONTWRITEBYTECODE=1
R=/mnt/models/gguf/tony-tern/CARDAN; C=$R/codice
export TMPDIR=$R/_tmp TRITON_CACHE_DIR=$R/_tmp/triton TORCHINDUCTOR_CACHE_DIR=$R/_tmp/inductor
S=/home/angelo/tony-forgia/stato; F=/home/angelo/tony-forgia
B=/mnt/models/build/thor-rexppiani2/build-rexp/bin
PY=/home/angelo/venv-catq/bin/python3
KLPY=/mnt/models/gguf/tony-tern/R026/v12/_lav/kl_realizzata.py
DUMP=/mnt/models/gguf/tony-tern/R026/_bin/e2e_dump_logit_p2
TEACH=/mnt/deposito/e2e_teacher/q8-maestro
PV=${CK_OSPITE:?CK_OSPITE (host GGUF) required}; NO=${CK_NOME:?CK_NOME (host label) required}; export NO
RIF_MMLU=${CK_RIF_MMLU:-}   # int8 MMLU log of the host on the same binary; empty = measured here before CARDAN's
BASE=${CK_BASE:?CK_BASE (A.1 basis of THIS host, cardan_costruisci.py --gguf host) required}
O=${CK_OUT:-$R/kd_r${CK_RANGO:-16}}; E=$O/allena; P=$O/piega; G=$O/Thor-T-$NO-CARDAN.gguf
LIMITE=${CK_LIMITE:-13085779616}
mkdir -p $E $P $O/kl $TMPDIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR
exec >> $O/corsa.log 2>&1
riga(){ local t; t=$(printf '%s' "$2" | tr '\t\n' '  ' | sed 's/rc=/codice /g'); printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$t" >> $S/cicli.tsv; echo "[$(date '+%F %T')] $1 $t"; }
cp "$0" $O/comando_$(date +%Y%m%d_%H%M).sh
klenv(){ LD_LIBRARY_PATH=$B GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_TQ1R_Q8=1 nice -n 5 timeout 7200 $PY $KLPY --thor-v2 $F/thor_v2 \
           --cache $TEACH --dump-bin $DUMP --out $O/kl --righe 24 "$@"; }
klval(){ $PY -c "import json, math; v = json.load(open('$O/kl/kl_$1.json'))['kl_media']; assert math.isfinite(v) and v > 0; print(v)"; }

# ── 0 preflight ──
e=0
for f in $C/cardan_kd.py $C/cardan_kd_moduli.py $C/cardan_piega.py /mnt/models/build/leve/raffina/patch_gguf.py \
         $B/llama-bench $B/llama-perplexity $KLPY $DUMP $PV $TEACH $F/lucchetto.sh; do [ -e $f ] || { echo "manca $f"; e=1; }; done
until [ "$(ls $BASE/strato_*.npz 2>/dev/null | wc -l)" = 40 ] && [ -s $BASE/info.json ]; do
  systemctl --user is-active -q ${CK_BASE_UNITA:-cardan-base-a1} || { [ "$(ls $BASE/strato_*.npz 2>/dev/null | wc -l)" = 40 ] && break; echo "base A.1 incompleta e cardan-base-a1 finita"; e=1; break; }
  sleep 60; done
LIB=$(df -BG --output=avail /mnt/models | tail -1 | tr -dc 0-9); [ "${LIB:-0}" -ge 16 ] || { echo "disco /mnt/models ${LIB} GiB < 16"; e=1; }
[ $e = 0 ] || { riga ERRORE "CARDAN-KD: preflight fallito ($O/corsa.log)"; exit 2; }
{ echo "codice $(cd $C && sha256sum cardan_kd.py cardan_kd_moduli.py cardan_piega.py recupero_7dabd6f/thor_v2/e2e_studente.py | cut -c1-16 | tr '\n' ' ')"
  echo "motore $(git -C /mnt/models/build/thor-rexppiani2 rev-parse HEAD) llama-perplexity $(sha256sum $B/llama-perplexity | cut -c1-16)"
  echo "ospite $PV $(stat -c %s $PV) B"; } > $O/provenienza.txt

# ── 1 GPU ──
. $F/lucchetto.sh
until guardia_ram; do sleep 60; done
riga ATTIVO "CARDAN-KD: pronta (base A.1 40 strati, prove CPU K1-K5 verdi), attende il lucchetto GPU (ora di $(cat $L_FILE/proprietario 2>/dev/null))"
prendi_gpu "CARDAN $$" 50
fine(){ local c=$?; molla_gpu; [ $c = 0 ] || riga ERRORE "CARDAN-KD uscita con codice $c (log $O/corsa.log)"; }
trap fine EXIT
T0=$(date +%s)
riga ATTIVO "CARDAN-KD $NO: GPU presa, KL realizzata int8 dell'ospite sullo stesso motore (rexppiani2)"

# ── 2 engine KL of the host and of COMBO A ──
if [ "${CK_DA_PIEGA:-0}" = 1 ]; then riga ATTIVO "CARDAN-KD: ripresa DALLA PIEGA (allenamento gia' fatto: $E/cardan_migliore.pt)"
else
klenv $NO=$PV > $O/kl_partenza.out 2>&1 || { riga ERRORE "CARDAN-KD: KL della partenza fallita ($(tail -2 $O/kl_partenza.out | tr '\n' ' ' | cut -c1-200))"; exit 2; }
rm -rf $O/kl/_lavoro_*
fi
KLP=$(klval $NO) || { riga ERRORE "CARDAN-KD $NO: kl_$NO.json illeggibile"; exit 2; }
riga ATTIVO "CARDAN-KD $NO: KL realizzata int8 (24 righe q8-maestro, rexppiani2): $NO $KLP"

# ── 3 training ──
# 9/10 sera: kernel nuovi dell'agente KERNEL QAT, solo se dichiarati pronti e coerenti (file segnale) al momento della partenza
KN=/home/angelo/tony-forgia/stato/KERNEL_NUOVI_PRONTI
if [ -s $KN ] && [ -d $C/recupero_7dabd6f/thor_v2_kqat ] && grep -q E2E_KERNEL_NUOVI $C/cardan_kd.py; then export E2E_KERNEL_NUOVI=1; riga ATTIVO "CARDAN-KD $NO: allenamento con E2E_KERNEL_NUOVI=1 ($(head -c 150 $KN))"
else riga ATTIVO "CARDAN-KD $NO: allenamento con i kernel di oggi (interruttore E2E_KERNEL_NUOVI non ancora pronto)"; fi
cd $C
if [ "${CK_DA_PIEGA:-0}" != 1 ]; then
RIP=""; [ "${CK_RIPRENDI:-0}" = 1 ] && RIP="--riprendi"
E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 \
E2E_CACHE_DEQUANT=$R/cache_dequant E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton \
  timeout $(( ${CK_ORE:-3} * 3600 + 3600 )) nice -n 5 $PY cardan_kd.py --uscita $E --gguf $PV --nome $NO \
    --teacher $TEACH --kl-partenza-motore $KLP --soglia-coerenza 0.03 --strati 0-39 --rango ${CK_RANGO:-16} --base-a1 $BASE \
    --passi ${CK_PASSI:-300} --righe-passo 1 --heldout-righe 24 --heldout-ogni 50 --pazienza 3 --ore-max ${CK_ORE:-3} $RIP \
    > $E/allena.out 2>&1
CK=$?
if [ $CK = 2 ]; then riga ERRORE "CARDAN-KD: cancello d'identita'/coerenza fallito, niente allenamento: $(grep -h -o 'ERRORE:.*' $E/registro.log $E/allena.out 2>/dev/null | tail -1 | cut -c1-250)"; exit 2; fi
fi
riga ATTIVO "CARDAN-KD: allenamento uscito con codice ${CK:-ripresa} in $(( ($(date +%s) - T0) / 60 )) min: $(grep -o 'FINE:.*' $E/registro.log 2>/dev/null | tail -1 | cut -c1-200) | $(grep -o 'coerenza: KL torch.*' $E/registro.log | tail -1 | cut -c1-140)"
[ -s $E/cardan_migliore.pt ] && [ -s $E/cardan.json ] || { riga CHIUSO "CARDAN-KD: nessun passo batte il passo 0 sulla KL held-out: niente piega ($E)"; exit 0; }

# ── 4 fold + GGUF ──
t4=$(date +%s)
w=0; until [ "$(df -BG --output=avail /mnt/models | tail -1 | tr -dc 0-9)" -ge 24 ]; do w=$((w+1)); [ $w = 1 ] && riga ATTIVO "CARDAN-KD $NO: piega in attesa di 24 GiB liberi su /mnt/models"; [ $w -gt 90 ] && { riga ERRORE "CARDAN-KD $NO: disco < 24 GiB da 3 h"; exit 3; }; sleep 120; done
PIEGA_ALGEBRA=${PIEGA_ALGEBRA:-cpu} OMP_NUM_THREADS=6 LEVE_DT=float32 LEVE_THREADS=6 nice -n 5 $PY -X faulthandler $C/cardan_piega.py --pt $E/cardan_migliore.pt --gguf $PV --out $P --device cuda --lotto 16 > $P/piega.out 2>&1 \
  || { riga ERRORE "CARDAN-KD: piega fallita ($(tail -2 $P/piega.out | tr '\n' ' ' | cut -c1-200))"; exit 3; }
CAPT=$($PY - $P/registro.jsonl <<'Q'
import json, sys
r = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
d = sum(x["energia_correzione_H"] for x in r); f = sum(x["residuo_dopo_piega_H"] for x in r)
fs = [x["catturata"] for x in r]
print(f"{len(r)} famiglie, energia della correzione catturata {100*(1-f/d):.1f} % (per famiglia min {100*min(fs):.1f} max {100*max(fs):.1f})")
Q
)
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= nice -n 10 $PY /mnt/models/build/leve/raffina/patch_gguf.py --dir $P/tensori --out $G --base $PV > $P/patch.out 2>&1 \
  || { riga ERRORE "CARDAN-KD: patch_gguf fallito ($(tail -2 $P/patch.out | tr '\n' ' ' | cut -c1-200))"; exit 3; }
SZ=$(stat -c %s $G); [ "$SZ" = "$(stat -c %s $PV)" ] && [ "$SZ" -le $LIMITE ] || { riga ERRORE "CARDAN-KD: GGUF $SZ B (atteso = ospite $NO, tetto $LIMITE)"; exit 3; }
riga ATTIVO "CARDAN-KD: piega nei piani esistenti in $(( ($(date +%s) - t4) / 60 )) min: $CAPT; GGUF $G ($SZ B = $NO al byte di taglia, stessi tensori)"

# ── 5 engine KL of the folded GGUF + post-fold coherence gate ──
klenv CARDAN=$G > $O/kl_cardan.out 2>&1; KC=$?; rm -rf $O/kl/_lavoro_*; [ $KC = 0 ] || { riga ERRORE "CARDAN-KD: KL realizzata del GGUF piegato fallita"; exit 4; }
KLC=$(klval CARDAN) || exit 4
VS=$($PY - $O/kl <<'Q'
import json, sys
d = sys.argv[1]; c = json.load(open(f"{d}/kl_CARDAN.json")); out = []
import os
for n in (os.environ["NO"],):
    b = json.load(open(f"{d}/kl_{n}.json"))
    dd = [c['per_riga'][k]['kl_media'] - b['per_riga'][k]['kl_media'] for k in c['per_riga'] if k in b['per_riga']]
    m = sum(dd) / len(dd); se = (sum((x - m) ** 2 for x in dd) / (len(dd) - 1)) ** .5 / len(dd) ** .5
    out.append(f"vs {n} {b['kl_media']:.5f}: {100*m/b['kl_media']:+.2f} % ({m/se:+.1f} se, {sum(x<0 for x in dd)}/{len(dd)} righe meglio)")
print(f"KL {c['kl_media']:.5f} " + "; ".join(out))
Q
)
riga ATTIVO "CARDAN-KD: KL realizzata int8 del piegato: $VS"
E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 \
E2E_CACHE_DEQUANT=$R/cache_dequant E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton \
  nice -n 5 $PY cardan_kd.py --uscita $O/coerenza_piegato --gguf $G --nome CARDAN-piegato --teacher $TEACH \
    --kl-partenza-motore $KLC --soglia-coerenza 0.03 --strati 0 --rango 1 --passi 0 --heldout-righe 24 > $O/coerenza_piegato.out 2>&1
CC=$?
COER=$(grep -o 'coerenza: KL torch.*' $O/coerenza_piegato/registro.log 2>/dev/null | tail -1 | cut -c1-160)
[ $CC = 0 ] && riga ATTIVO "CARDAN-KD: cancello torch<->motore sul GGUF PIEGATO passato ($COER)" \
            || riga ERRORE "CARDAN-KD: cancello torch<->motore sul GGUF piegato FALLITO (codice $CC: $COER)"

# ── 6 speed, same session ──
tg(){ grep -E "tg128" "$1" 2>/dev/null | awk -F'|' '{print $(NF-1)}' | tr -d ' ' | head -1; }
pp(){ grep -E "pp512" "$1" 2>/dev/null | awk -F'|' '{print $(NF-1)}' | tr -d ' ' | head -1; }
for v in OSPITE=$PV CARDAN=$G OSPITE_bis=$PV; do n=${v%%=*}; g=${v#*=}
  LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 nice -n 5 $B/llama-bench -m "$g" -ngl 999 -dev Vulkan0 -p 512 -n 128 -r 5 > $O/bench_$n.txt 2>&1; done
riga ATTIVO "CARDAN-KD: velocita' int8 tg128 $NO $(tg $O/bench_OSPITE.txt) CARDAN $(tg $O/bench_CARDAN.txt) $NO(bis) $(tg $O/bench_OSPITE_bis.txt) t/s; pp512 $(pp $O/bench_OSPITE.txt) / $(pp $O/bench_CARDAN.txt) (carico $(awk '{print $1}' /proc/loadavg))"
molla_gpu; trap - EXIT
# ── MMLU command: host int8 reference (if not given) + CARDAN, same binary/parameters as combo_misura.sh, inside the lock ──
[ -n "$RIF_MMLU" ] || RIF_MMLU=$O/mmlu_$NO.log
cat > $O/comando_mmlu.sh <<M
#!/bin/bash
. $F/lucchetto.sh; prendi_gpu "CARDAN-MMLU-$NO \$\$" 30; trap molla_gpu EXIT
mm(){ LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 nice -n 5 $B/llama-perplexity -m "\$1" -ngl 999 -c 4096 -np 4 -ub 512 \\
  -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f $F/mc_v2/mmlu_00000.bin > "\$2" 2>&1
  [ "\$(grep -c '^MCLP	' "\$2")" = 2000 ] || { echo "MMLU incompleto: \$2"; exit 1; }; }
[ "\$(grep -c '^MCLP	' $RIF_MMLU 2>/dev/null)" = 2000 ] || mm $PV $RIF_MMLU
mm $G $O/mmlu_CARDAN.log
python3 $F/thor_v2/strumenti/mcnemar_mclp.py $RIF_MMLU $O/mmlu_CARDAN.log --json $O/mcnemar_vs_$NO.json
M
chmod +x $O/comando_mmlu.sh
riga CHIUSO "CARDAN-KD $NO (KD nel flusso ternario + piega nei piani, zero byte e zero tensori in piu'): $VS | piega: $CAPT | velocita' tg128 $(tg $O/bench_OSPITE.txt) -> $(tg $O/bench_CARDAN.txt) t/s | MMLU: bash $O/comando_mmlu.sh"
if [ "${CK_MMLU:-0}" = 1 ]; then
  bash $O/comando_mmlu.sh > $O/comando_mmlu.out 2>&1 || { riga ERRORE "CARDAN-KD $NO: MMLU fallito ($(tail -2 $O/comando_mmlu.out | tr '\n' ' ' | cut -c1-200))"; exit 5; }
  MC=$($PY -c "import json; d=json.load(open('$O/mcnemar_vs_$NO.json')); print(f\"{d['acc_B']:.2f} vs $NO {d['acc_A']:.2f}, vinte/perse {d['B_giusta_A_sbagliata']}/{d['A_giusta_B_sbagliata']}, p {d['p_mcnemar_esatto']:.4f}\")")
  K4=$(HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= $PY /mnt/models/gguf/tony-tern/LEVE_RAFF_PV12P/diagnosi/kl4_appaiato.py $RIF_MMLU $O/mmlu_CARDAN.log 2>&1 | tail -1 | sed 's|^.*log: ||')
  riga CHIUSO "CARDAN-KD $NO MMLU-2000 int8 $MC | KL risposta MMLU: $K4 | file $(stat -c %s $G) B (tetto B 9.492.293.427)"
fi
exit 0
