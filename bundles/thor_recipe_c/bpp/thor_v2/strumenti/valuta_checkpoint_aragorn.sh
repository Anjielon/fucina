#!/bin/bash
# 26/9 sera — MMLU coi logprob di un checkpoint E2E di Thor-T valutato su ARAGORN (RTX 4060, Vulkan), cosi' il Corsair
# non ferma mai l'allenamento per valutare. Stesso albero sorgente del banco del Corsair (mmlu_logprob/sorgente: llama.cpp
# 521a64cd + TQ1_0R_1024 + patch MCLP) compilato su Aragorn in C:\tcq\thor_mmlu\sorgente\build; stesso file dei 2000
# compiti (sha verificato); stessi argomenti salvo la posizione degli esperti degli strati 0-9 (in RAM: 8 GB di VRAM).
# Lanciatore su Aragorn: C:\tcq\thor_mmlu\banco_ot.bat (tutti gli strati su Vulkan0 = RTX 4060, GDN fuso come sul
# Corsair). ⚠️ Collaudo 26/9: i logprob di Aragorn NON coincidono compito per compito con quelli del Corsair (|Δ| medio
# ~0,45 nat anche su sola CPU): il confronto VALIDO e' APPAIATO contro B misurato SU ARAGORN (B_aragorn.log), stesso banco.
# Unica modifica al sorgente su Aragorn: common/arg.cpp legge -f in binario (mmlu_logprob/aragorn_arg_binario.patch).
#
# Passi: (1) export del GGUF su CPU del Corsair (e2e_esporta.py --device cpu, scope confinato, niente GPU);
# (2) copia su Aragorn con sha256 verificato; (3) banco lanciato con schtasks (MAI Start-Process da ssh: muore con la
# sessione); (4) attesa, registro riportato indietro e controllato (2000 acc, 2000 MCLP in ordine, Final result);
# (5) i quattro numeri (mmlu_quattro_numeri.py) contro B.
#
# Uso:  valuta_checkpoint_aragorn.sh @@ROOT:lavoro@@/e2e/allena/EB2/migliore_000016
#       valuta_checkpoint_aragorn.sh --gguf NOME /percorso/file.gguf     (salta l'export: per il collaudo)
set -uo pipefail
QUI=@@ROOT:forge@@/thor_v2
SORG=@@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf
USC_DIR=@@ROOT:models@@/gguf/tony-tern/e2e_checkpoint
REG_DIR=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn
RIF=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/B_aragorn.log      # B misurato su Aragorn: riferimento appaiato
RIF_CORSAIR=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/B.log
PY=@@ROOT:venv@@/bin/python3
mkdir -p "$USC_DIR" "$REG_DIR"

if [ "${1:-}" = "--gguf" ]; then
  NOME=${2:?nome}; GGUF=${3:?gguf}
else
  IST=${1:?uso: valuta_checkpoint_aragorn.sh CARTELLA_migliore_XXXXXX}
  IST=${IST%/}
  [ -f "$IST/meta.json" ] || { echo "⛔ $IST/meta.json assente"; exit 2; }
  CORSA=$(dirname "$IST"); NOME="$(basename "$CORSA")-$(basename "$IST")"
  GGUF=$USC_DIR/Thor-T-$NOME.gguf
  if [ ! -s "$GGUF" ]; then
    LIB=$(df -BG --output=avail "$USC_DIR" | tail -1 | tr -dc 0-9)
    [ "$LIB" -ge 12 ] || { echo "⛔ $USC_DIR: ${LIB} GiB liberi, ne servono 12"; exit 2; }
    echo "[$(date '+%H:%M:%S')] export $NOME su CPU (scope confinato)"
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
      systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=16G -p MemorySwapMax=2G \
      nice -n 19 ionice -c3 env -u HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 \
      "$PY" "$QUI/e2e_esporta.py" --sorgente "$SORG" --corsa "$CORSA" --istantanea "$(basename "$IST")" \
      --uscita "$GGUF" --device cpu || { echo "⛔ export fallito"; exit 3; }
  fi
fi
[ -s "$GGUF" ] || { echo "⛔ GGUF assente: $GGUF"; exit 2; }
REG=$REG_DIR/$NOME.log
[ -s "$REG" ] && { echo "⛔ $REG esiste gia'"; exit 2; }
BASE=$(basename "$GGUF")

# (2) copia con sha
SHA=$(sha256sum "$GGUF" | cut -d' ' -f1)
echo "[$(date '+%H:%M:%S')] copia $BASE su Aragorn"
scp -q "$GGUF" "aragorn:C:/tcq/thor_mmlu/gguf/$BASE" || { echo "⛔ copia fallita"; exit 3; }
SHA_A=$(ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$BASE SHA256" | sed -n 2p | tr -d '\r ')
[ "$SHA" = "$SHA_A" ] || { echo "⛔ sha diverso su Aragorn: $SHA_A != $SHA"; exit 3; }

# (3) banco via schtasks
ssh aragorn "schtasks /create /f /tn thor_mmlu_$NOME /tr \"C:\\tcq\\thor_mmlu\\banco_ot.bat $NOME $BASE\" /sc once /st 23:59 /sd 01/01/2030 /rl limited && schtasks /run /tn thor_mmlu_$NOME" >/dev/null \
  || { echo "⛔ schtasks fallito"; exit 3; }
echo "[$(date '+%H:%M:%S')] banco $NOME partito su Aragorn (banco_ot.bat)"

# (4) attesa della fine (il .bat scrive FATTO nel file .fine)
until ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$NOME.fine" 2>/dev/null | grep -q FATTO; do sleep 60; done
ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$NOME.fine"
scp -q "aragorn:C:/tcq/thor_mmlu/log/$NOME.log" "$REG" || { echo "⛔ registro non riportato"; exit 3; }
N_ACC=$(tr -d '\r' < "$REG" | grep -cE '^[0-9]+	[0-9]+\.[0-9]+$')
N_MCLP=$(grep -c '^MCLP	' "$REG")
FINALE=$(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' "$REG" | tail -1)
echo "[$(date '+%H:%M:%S')] righe acc $N_ACC · righe MCLP $N_MCLP · ${FINALE:-Final result ASSENTE}"
[ "$N_ACC" -eq 2000 ] && [ "$N_MCLP" -eq 2000 ] && [ -n "$FINALE" ] || exit 5

# (5) i quattro numeri: contro B su Aragorn (confronto valido, appaiato) e, per informazione, contro B del Corsair
[ -s "$RIF" ] || { echo "⛔ manca $RIF (B misurato su Aragorn)"; exit 5; }
nice -n 19 env OMP_NUM_THREADS=2 python3 "$QUI/strumenti/mmlu_quattro_numeri.py" "$REG" --confronta "$RIF" \
  --json "$REG_DIR/$NOME.json"
nice -n 19 env OMP_NUM_THREADS=2 python3 "$QUI/strumenti/mmlu_quattro_numeri.py" "$REG" --confronta "$RIF_CORSAIR" | tail -1
