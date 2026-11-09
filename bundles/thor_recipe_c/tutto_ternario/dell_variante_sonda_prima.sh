#!/bin/bash
# 28/9 tutto_ternario — sonda parlata PRIMA (cancello: 0 mute, >=10/21) e poi MMLU 2000 (sonda_parlata.bat, 3 domande x 7 semi)
# di UNA variante sul Dell, in UNA sola presa del lucchetto (una copia sola del GGUF), poi il GGUF si toglie dal Dell.
# Stessi comandi di valuta_checkpoint_aragorn.sh e di esegui_sonda (sonda_EB3.sh), stessi .bat, stesso banco di EB3 p20.
# Va chiamato SOTTO il lucchetto:  aragorn_lucchetto.sh dell_variante.sh NOME /percorso/FILE.gguf
set -uo pipefail
N=$1; G=$2; F=$(basename "$G")
Q=@@ROOT:depot@@/tutto_ternario/log
REG_DIR=@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn
REG=$REG_DIR/$N.log
RIF_EB3=$REG_DIR/EB3-passo_000020.log
S=@@ROOT:forge@@/thor_v2/strumenti
log() { echo "[$(date '+%F %T')] $*"; }
pulisci() { timeout 60 ssh aragorn "del C:\\tcq\\thor_mmlu\\gguf\\$F" >/dev/null 2>&1; log "GGUF $F tolto dal Dell"; }
[ -s "$REG" ] && { log "⛔ $REG esiste gia'"; exit 2; }
log "lucchetto preso: $N ($F)"

# spazio sul Dell (il GGUF + 1 GiB di margine)
LIB=$(timeout 60 ssh aragorn "powershell -NoProfile -Command \"(Get-PSDrive C).Free\"" | tr -dc 0-9)
SERVE=$(( $(stat -c %s "$G") + 1073741824 ))
[ -n "$LIB" ] && [ "$LIB" -gt "$SERVE" ] || { log "⛔ spazio sul Dell: ${LIB:-?} byte liberi, ne servono $SERVE"; exit 4; }

# (1) copia con sha256
SHA=$(sha256sum "$G" | cut -d' ' -f1)
log "copia $F sul Dell (sha $SHA)"
timeout 3600 scp -q "$G" "aragorn:C:/tcq/thor_mmlu/gguf/$F" || { log "⛔ copia fallita"; pulisci; exit 3; }
SHA_A=$(timeout 600 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$F SHA256" | sed -n 2p | tr -d '\r ')
[ "$SHA" = "$SHA_A" ] || { log "⛔ sha diverso sul Dell: $SHA_A"; pulisci; exit 3; }
log "COPIATO sul Dell, sha verificato"

# (2) sonda parlata (limite orario dello script sul Dell = stesso giorno: dopo le 23:25 si aspetta la mezzanotte)
if [ "$(date +%H%M)" -ge 2325 ]; then
  log "sono le $(date +%H:%M): la sonda parte dopo mezzanotte (limite orario del .bat)"
  until [ "$(date +%H%M)" -lt 2300 ]; do sleep 30; done
fi
timeout 120 ssh aragorn "del C:\\tcq\\thor_mmlu\\log\\sonda_$N.fine 2>nul & schtasks /create /f /tn sonda_$N /tr \"C:\\tcq\\thor_mmlu\\sonda_parlata.bat $N $F 23:59\" /sc once /st 23:59 /rl limited && schtasks /run /tn sonda_$N" >/dev/null \
  || { log "⛔ schtasks sonda fallito"; pulisci; exit 3; }
log "SONDA PARTITA sul Dell"
T0=$(date +%s)
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\sonda_$N.fine" 2>/dev/null | grep -q FATTO; do
  [ $(( $(date +%s) - T0 )) -gt 7200 ] && { log "⛔ sonda oltre 2 h"; break; }
  sleep 60
done
for f in sonda_$N.txt sonda_$N.log sonda_${N}_grezze.jsonl sonda_${N}_server.log; do timeout 300 scp -q "aragorn:C:/tcq/thor_mmlu/log/$f" "$Q/" 2>/dev/null; done
log "SONDA FINITA: $(grep -E "giuste [0-9]+/21" "$Q/sonda_$N.txt" 2>/dev/null | tail -1 | tr -d '\r\n')"
R_S=$(grep -E "mute [0-9]+/21" "$Q/sonda_$N.txt" 2>/dev/null | tail -1 | tr -d '\r')
G_S=$(echo "$R_S" | grep -oE "giuste [0-9]+" | grep -oE "[0-9]+"); M_S=$(echo "$R_S" | grep -oE "mute [0-9]+" | grep -oE "[0-9]+")
log "cancello sonda: giuste ${G_S:-?}/21 mute ${M_S:-?}/21 (EB3 p20: 19/21, 0 mute)"
# 29/9: niente soglia a mano. Muta -> stop; altrimenti stop solo se significativamente peggiore del modello di PARTENZA
# (SONDA_PADRE = giuste su 21 del padre; Fisher esatto unilaterale, alfa 0,05). Regola in fucina (probe gate, E64).
P_S=$(python3 -c "from scipy.stats import fisher_exact as f; a=int('${G_S:-0}'); b=int('${SONDA_PADRE:-0}'); print(round(f([[a,21-a],[b,21-b]],alternative='less')[1],4))")
log "Fisher contro il padre (${SONDA_PADRE:-0}/21): p = $P_S"
if [ -z "$G_S" ] || [ "${M_S:-99}" -gt 0 ] || awk -v p="$P_S" 'BEGIN{exit !(p<0.05)}'; then log "⛔ sonda NON superata: niente MMLU"; pulisci; exit 5; fi
# (3) MMLU 2000 coi logprob (banco_ot.bat, come EB3 p20)
timeout 120 ssh aragorn "schtasks /create /f /tn thor_mmlu_$N /tr \"C:\\tcq\\thor_mmlu\\banco_ot.bat $N $F\" /sc once /st 23:59 /rl limited && schtasks /run /tn thor_mmlu_$N" >/dev/null \
  || { log "⛔ schtasks MMLU fallito"; pulisci; exit 3; }
log "MMLU PARTITO sul Dell"
T0=$(date +%s)
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$N.fine" 2>/dev/null | grep -q FATTO; do
  [ $(( $(date +%s) - T0 )) -gt 14400 ] && { log "⛔ MMLU oltre 4 h"; break; }
  sleep 60
done
timeout 60 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$N.fine"
timeout 600 scp -q "aragorn:C:/tcq/thor_mmlu/log/$N.log" "$REG" || log "⛔ registro MMLU non riportato"
N_ACC=$(tr -d '\r' < "$REG" | grep -cE '^[0-9]+	[0-9]+\.[0-9]+$')
N_MCLP=$(grep -c '^MCLP	' "$REG")
FINALE=$(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' "$REG" | tail -1)
log "MMLU FINITO: righe acc $N_ACC · MCLP $N_MCLP · ${FINALE:-Final result ASSENTE}"
if [ "$N_ACC" -eq 2000 ] && [ "$N_MCLP" -eq 2000 ] && [ -n "$FINALE" ]; then
  nice -n 19 env OMP_NUM_THREADS=2 python3 "$S/mmlu_quattro_numeri.py" "$REG" --confronta "$RIF_EB3" --json "$REG_DIR/$N.json"
fi

pulisci
