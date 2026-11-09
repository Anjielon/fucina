#!/bin/bash
# CONSEGNA DI THOR-T — certificazione completa in un comando (26/9/2026 sera).
# Definizione di Angelo: «Thor FUNZIONANTE che batte NETTAMENTE il rivale». Cinque cancelli, in quest'ordine:
#   1. EXPORT   e2e_esporta.py su CPU (scope confinato, verificato al bit; 330 s misurati su EB1/migliore_000000)
#   2. SONDA    sonda_parlata_k1024.sh: thinking acceso, 3 domande x 7 seed, verdetto a TASSO accanto a B
#               (Thor-v2-ThorT-scor150, il modello prima dell'E2E), stessi seed. PASS: zero muti e giuste >= B.
#               (feedback_sonda_multi_seed_obbligatoria, feedback_sonda_prima_del_banco_sempre)
#   3. MMLU     mmlu_logprob_lancia.sh sul Corsair (stesso banco mc_v2/mmlu_00000.bin, 2000 compiti, stessi quattro
#               numeri di oggi via mmlu_quattro_numeri.py). PASS: grezza (= Final result, la stessa del 78,15 del rivale)
#               >= 80,05 = 78,15 + 2 sigma (sigma ~0,95 con 2000 compiti).
#               Durata misurata oggi sul Corsair: B 15 min 50 s (B.log 05:46:51 -> 06:02:41), teacher 12 min 40 s.
#   4. PPL      llama-perplexity wikitext-2 (odino-lab/collaudo/corpus/wiki.test.raw, 72 chunk, -c 4096 -b 2048 -ub 2048,
#               build tq1-k1024) = ESATTAMENTE il metodo del rivale Q2_K 7,9330 +- 0,0559 (thor/coda_gpu_rivale_q2k.sh,
#               21/9; 6 min sul Corsair). PASS: PPL < 7,9330; NETTA se sotto di 2 sigma della differenza.
#               CONSEGNA_PPL_SOLO_RIPORTO=1 -> la PPL si riporta ma non entra nel verdetto.
#   5. F0c      e2e_fedelta.py f0c: il guadagno misurato in torch si trasferisce a llama.cpp (rho >= 0,8).
#               ⚠️ F0c tiene lo studente in torch E lancia llama.cpp: con il training acceso sono TRE contesti GPU
#               (feedback_tre_candidati_insieme_hang_gpu: tre carichi insieme hanno piantato il driver). Quindi:
#               --f0c auto (default): se il training e' attivo F0c e' RINVIATO (verdetto PROVVISORIO, exit 4, comando
#               stampato per dopo); --f0c accanto: lo lancia comunque; --f0c solo: salta 1-4 e fa solo F0c
#               (anche in «solo» F0c aspetta un training spento: per forzarlo accanto usare --f0c accanto).
#
# Ogni pezzo GPU e' un'unita' systemd --user (la shell ha HSA_OVERRIDE_GFX_VERSION=11.5.0 SBAGLIATO: gfx1150 invece di
# gfx1151 -> "device kernel image is invalid") con GPU_ESCLUDI_PID = PID vivo della corsa di training, riletto a ogni
# passo: la verifica gira ACCANTO al training (2 carichi GPU: training + un solo pezzo di verifica alla volta).
# Durata prevista con GPU libera: export 6 + sonda B 10 (una volta sola, poi in cache) + sonda 10 + MMLU 16 + PPL 6
# + F0c ~25 = ~75 min; accanto al training ~1,5x (~1 h 50).
#
# Uso:  consegna_verifica.sh @@ROOT:lavoro@@/e2e/allena/EB2/migliore_000016 [--f0c auto|accanto|solo] [--unita e2e-allena-EB2]
# Esito: @@ROOT:lavoro@@/e2e/consegna/<corsa>-<istantanea>.json ; exit 0 CONSEGNA, 1 NON consegna, 4 provvisorio, 2-3 errore.
set -uo pipefail
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
unset HSA_OVERRIDE_GFX_VERSION

IST=${1:?uso: consegna_verifica.sh CARTELLA_migliore_XXXXXX [--f0c auto|accanto|solo] [--unita UNITA]}; shift
F0C=auto; UNITA_TRAIN=e2e-allena-EB2
while [ $# -gt 0 ]; do
  case "$1" in --f0c) F0C=${2:?}; shift 2 ;; --unita) UNITA_TRAIN=${2:?}; shift 2 ;; *) echo "⛔ argomento $1"; exit 2 ;; esac
done
case "$F0C" in auto|accanto|solo) ;; *) echo "⛔ --f0c auto|accanto|solo"; exit 2 ;; esac

QUI=@@ROOT:forge@@/thor_v2
FORGIA=@@ROOT:forge@@
PY=@@ROOT:venv@@/bin/python3
B=@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin
SORG=@@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf
USC_DIR=@@ROOT:models@@/gguf/tony-tern/e2e_checkpoint              # lo stesso di valuta_checkpoint_aragorn.sh: un export solo
WIKI=@@ROOT:odino@@/collaudo/corpus/wiki.test.raw
MMLU_REG=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob
ESITI=@@ROOT:lavoro@@/e2e/consegna
PPL_RIV=7.9330; SIG_RIV=0.0559; MMLU_RIV=78.15; MMLU_SOGLIA=80.05

IST=${IST%/}
[ -f "$IST/meta.json" ] && [ -f "$IST/theta.pt" ] || { echo "⛔ $IST non e' un'istantanea (meta.json/theta.pt)"; exit 2; }
CORSA=$(dirname "$IST"); NOME="$(basename "$CORSA")-$(basename "$IST")"
GGUF=$USC_DIR/Thor-T-$NOME.gguf
CFG=$CORSA/config.json
TEACHER=$($PY -c "import json;print(json.load(open('$CFG'))['teacher'])")
RICETTA=$($PY -c "import json;print(json.load(open('$CFG'))['ricetta'])")
mkdir -p "$USC_DIR" "$ESITI"
LOG=$ESITI/$NOME.log
exec > >(tee -a "$LOG") 2>&1
echo "════ CONSEGNA $NOME · $(date '+%d/%m %H:%M:%S') · ricetta $RICETTA · teacher $TEACHER · f0c=$F0C"

pid_train(){ systemctl --user show "$UNITA_TRAIN" -p MainPID --value 2>/dev/null | grep -v '^0$' ; }
# un pezzo GPU: unita' --user, attesa del suo esito, GPU_ESCLUDI_PID col PID vivo del training
gpu_unita(){ # $1 nome unita', resto = comando
  local u=$1; shift
  local p; p=$(pid_train)
  if ! GPU_ESCLUDI_PID="${p:-}" "$QUI/strumenti/gpu_occupata.sh" --attendi 900 >/dev/null; then
    echo "⛔ GPU occupata da altro oltre al training (dopo 15 min di attesa): $(GPU_ESCLUDI_PID="${p:-}" "$QUI/strumenti/gpu_occupata.sh" | head -3)"
    return 3
  fi
  systemctl --user reset-failed "$u" 2>/dev/null
  systemd-run --user --unit "$u" --collect --wait --quiet -p Nice=5 --setenv=GPU_ESCLUDI_PID="${p:-}" \
    --setenv=XDG_RUNTIME_DIR=/run/user/1000 --setenv=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    --working-directory "$QUI" "$@"
}

declare -A ESITO DETT
ESITO=([export]=NON_FATTO [sonda]=NON_FATTO [mmlu]=NON_FATTO [ppl]=NON_FATTO [f0c]=NON_FATTO)

if [ "$F0C" != solo ]; then
# ── 1. EXPORT (CPU)
if [ -s "$GGUF" ] && [ -s "$GGUF.provenienza.json" ]; then
  echo "── 1/5 export: gia' presente $GGUF (con provenienza), si riusa"
  ESITO[export]=PASS; DETT[export]="riusato"
else
  LIB=$(df -BG --output=avail "$USC_DIR" | tail -1 | tr -dc 0-9)
  [ "$LIB" -ge 12 ] || { echo "⛔ $USC_DIR: ${LIB} GiB liberi, ne servono 12"; exit 2; }
  echo "── 1/5 export su CPU $(date '+%H:%M:%S')"
  t0=$(date +%s)
  if systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=16G -p MemorySwapMax=2G \
       nice -n 19 ionice -c3 env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 \
       "$PY" "$QUI/e2e_esporta.py" --sorgente "$SORG" --corsa "$CORSA" --istantanea "$(basename "$IST")" \
       --uscita "$GGUF" --device cpu && [ -s "$GGUF.provenienza.json" ]; then
    ESITO[export]=PASS; DETT[export]="$(( $(date +%s) - t0 )) s"
  else
    ESITO[export]=FAIL; DETT[export]="e2e_esporta fallito"
  fi
fi
echo "   export: ${ESITO[export]} (${DETT[export]})"

if [ "${ESITO[export]}" = PASS ]; then
# ── 2. SONDA PARLATA (B di riferimento una volta sola, stessi seed)
sonda(){ # $1 gguf $2 nome -> stampa "giuste muti"
  local n=$2 f=$FORGIA/sonda_$2.txt
  if [ ! -s "$f" ]; then
    local p; p=$(pid_train)
    GPU_ESCLUDI_PID="${p:-}" "$QUI/strumenti/gpu_occupata.sh" --attendi 900 >/dev/null || return 3
    "$FORGIA/sonda_parlata_k1024.sh" "$1" "$n" >/dev/null 2>&1
  fi
  [ -s "$f" ] || return 3
  local g m
  g=$(grep -oE 'giuste [0-9]+/[0-9]+' "$f" | tail -1 | grep -oE '[0-9]+' | head -1)
  m=$(grep -oE 'mute [0-9]+/' "$f" | tail -1 | grep -oE '[0-9]+')
  echo "${g:-} ${m:-} $(grep -oE 'D1=[0-9]+/[0-9]+ D2=[0-9]+/[0-9]+ D3=[0-9]+/[0-9]+' "$f" | tail -1)"
}
echo "── 2/5 sonda parlata $(date '+%H:%M:%S')"
read -r GB MB TB < <(sonda "$SORG" "consegna_B-scor150" || echo "")
read -r GC MC TC < <(sonda "$GGUF" "consegna_$NOME" || echo "")
if [ -z "${GC:-}" ] || [ -z "${GB:-}" ]; then
  ESITO[sonda]=FAIL; DETT[sonda]="sonda non eseguita (B='${GB:-}' candidato='${GC:-}')"
elif [ "${MC:-1}" -eq 0 ] && [ "$GC" -ge "$GB" ]; then
  ESITO[sonda]=PASS; DETT[sonda]="giuste $GC/21 ($TC) contro B $GB/21 ($TB), mute 0"
else
  ESITO[sonda]=FAIL; DETT[sonda]="giuste $GC/21 ($TC) contro B $GB/21 ($TB), mute $MC"
fi
echo "   sonda: ${ESITO[sonda]} (${DETT[sonda]})"

# ── 3. MMLU coi logprob sul Corsair
echo "── 3/5 MMLU logprob $(date '+%H:%M:%S')"
REG=$MMLU_REG/consegna_$NOME.log
if [ ! -s "$REG" ]; then
  gpu_unita "consegna-mmlu" "$QUI/strumenti/mmlu_logprob_lancia.sh" "consegna_$NOME" "$GGUF" >/dev/null 2>&1
fi
if [ -s "$REG" ] && grep -q "Final result" "$REG"; then
  Q=$("$PY" "$QUI/strumenti/mmlu_quattro_numeri.py" "$REG" --json "$ESITI/$NOME.mmlu.json" 2>/dev/null | tail -3 | tr '\n' ' ')
  GR=$(grep -oE 'Final result: [0-9.]+' "$REG" | tail -1 | awk '{print $3}')
  if awk -v g="$GR" -v s="$MMLU_SOGLIA" 'BEGIN{exit !(g >= s)}'; then ESITO[mmlu]=PASS; else ESITO[mmlu]=FAIL; fi
  DETT[mmlu]="grezza $GR (soglia $MMLU_SOGLIA = rivale $MMLU_RIV + 2 sigma) · $Q"
else
  ESITO[mmlu]=FAIL; DETT[mmlu]="banco non completato ($REG)"
fi
echo "   MMLU: ${ESITO[mmlu]} (${DETT[mmlu]})"

# ── 4. PPL wikitext (metodo del rivale)
echo "── 4/5 PPL wikitext $(date '+%H:%M:%S')"
PPLLOG=$ESITI/ppl_$NOME.log
if ! grep -q "Final estimate" "$PPLLOG" 2>/dev/null; then
  gpu_unita "consegna-ppl" bash -c "LD_LIBRARY_PATH=$B $B/llama-perplexity --model '$GGUF' -ngl 999 -c 4096 -b 2048 -ub 2048 \
    -dev Vulkan0 -f '$WIKI' > '$PPLLOG' 2>&1"
fi
read -r PPL SPPL < <(grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' "$PPLLOG" 2>/dev/null | tail -1 | awk '{print $3, $5}')
if [ -n "${PPL:-}" ]; then
  VER=$(awk -v p="$PPL" -v s="$SPPL" -v r="$PPL_RIV" -v sr="$SIG_RIV" 'BEGIN{
    d = r - p; sd = sqrt(s*s + sr*sr); z = (sd > 0) ? d / sd : 0
    if (p < r) printf "PASS %.2f %s", z, (z >= 2 ? "NETTA" : "non netta"); else printf "FAIL %.2f peggiore", z }')
  ESITO[ppl]=${VER%% *}; DETT[ppl]="PPL $PPL +- $SPPL contro rivale $PPL_RIV +- $SIG_RIV: ${VER#* } sigma"
else
  ESITO[ppl]=FAIL; DETT[ppl]="PPL non letta ($PPLLOG)"
fi
echo "   PPL: ${ESITO[ppl]} (${DETT[ppl]})"
fi   # export PASS
fi   # F0C != solo

# ── 5. F0c
F0CJ=$ESITI/f0c_$NOME.json
CMD_F0C="$PY $QUI/e2e_fedelta.py f0c --teacher $TEACHER --sorgente $SORG --esportato $GGUF --corsa $CORSA --istantanea $(basename "$IST") --ricetta $RICETTA --uscita $F0CJ"
echo "── 5/5 F0c $(date '+%H:%M:%S')"
if [ ! -s "$GGUF" ]; then
  ESITO[f0c]=FAIL; DETT[f0c]="GGUF esportato assente"
elif [ "$F0C" != accanto ] && [ -n "$(pid_train)" ]; then
  ESITO[f0c]=RINVIATO; DETT[f0c]="training attivo: F0c = 3 contesti GPU (rischio hang). Dopo: $(readlink -f "$0") $IST --f0c solo"
else
  [ -s "$F0CJ" ] || gpu_unita "consegna-f0c" $CMD_F0C >/dev/null 2>&1
  if [ -s "$F0CJ" ]; then
    read -r PASSA RHO < <("$PY" -c "import json;d=json.load(open('$F0CJ'));print(d.get('passa'), round(d.get('rho',float('nan')),4))")
    [ "$PASSA" = True ] && ESITO[f0c]=PASS || ESITO[f0c]=FAIL
    DETT[f0c]="rho $RHO (soglia 0,8)"
  else
    ESITO[f0c]=FAIL; DETT[f0c]="F0c non completato"
  fi
fi
echo "   F0c: ${ESITO[f0c]} (${DETT[f0c]})"

# ── verdetto unico (nel caso --f0c solo, si fondono gli esiti gia' scritti)
J=$ESITI/$NOME.json
export C_J="$J" C_MODO="$F0C" C_PPL_RIPORTO="${CONSEGNA_PPL_SOLO_RIPORTO:-0}" C_IST="$IST" C_GGUF="$GGUF"
for k in export sonda mmlu ppl f0c; do export "C_E_$k=${ESITO[$k]}" "C_D_$k=${DETT[$k]:-}"; done
"$PY" - <<'EOF'
import json, os, datetime, sys
E = os.environ
j, ppl_riporto = E["C_J"], E["C_PPL_RIPORTO"] == "1"
nuovi = {k: (E[f"C_E_{k}"], E[f"C_D_{k}"]) for k in ("export", "sonda", "mmlu", "ppl", "f0c")}
vecchi = json.load(open(j))["cancelli"] if os.path.exists(j) else {}
canc = {k: ({"esito": e, "dettaglio": d} if e != "NON_FATTO" or k not in vecchi else vecchi[k]) for k, (e, d) in nuovi.items()}
richiesti = ["export", "sonda", "mmlu", "f0c"] + ([] if ppl_riporto else ["ppl"])
es = [canc[k]["esito"] for k in richiesti]
if all(e == "PASS" for e in es):
    v, rc = "CONSEGNA: Thor-T funzionante che batte nettamente il rivale", 0
elif "FAIL" in es:
    v, rc = "NON CONSEGNA: " + ", ".join(k for k in richiesti if canc[k]["esito"] == "FAIL") + " non passano", 1
else:
    v, rc = "PROVVISORIO: manca " + ", ".join(k for k in richiesti if canc[k]["esito"] != "PASS"), 4
out = {"istantanea": E["C_IST"], "gguf": E["C_GGUF"], "cancelli": canc, "richiesti": richiesti, "verdetto": v,
       "rc": rc, "scritto": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(out, open(j, "w"), indent=1, ensure_ascii=False)
for k in canc:
    print(f"  {k:7s} {canc[k]['esito']:9s} {canc[k]['dettaglio']}")
print("VERDETTO:", v)
sys.exit(rc)
EOF
RC=$?
echo "esito in $J"
exit $RC
