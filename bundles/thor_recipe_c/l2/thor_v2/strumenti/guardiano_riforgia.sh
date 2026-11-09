#!/bin/bash
# 28/9 — guardiano della riforgia piena: legge il verdetto di OGNI finestra dal log e FERMA la corsa da solo se la
# finestra peggiora la conoscenza (held-out MMLU validation) o sfora il tetto della calibrazione. Nato dalla notte
# del 28/9: pilota finito alle 02:12, GPU ferma fino alle 08:36 perche' la decisione aspettava Claude. Qui la
# decisione e' scritta PRIMA e la prende il guardiano: nessuna attesa di persone.
# Uso: guardiano_riforgia.sh <log> <unita'> [tetto_conoscenza=1.0] [tetto_calib=1.05] [prima_finestra_giudicata=0]
# 28/9 12:5x: il 5o argomento salta le finestre gia' giudicate (il log si rilegge dall'inizio): serve per riarmare con
# soglie piu' strette senza fermare la corsa per una finestra vecchia.
set -u
LOG=$1; UNITA=$2; TC=${3:-1.0}; TK=${4:-1.05}; DA=${5:-0}
CICLI=@@ROOT:forge@@/stato/cicli.tsv
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
tail -n +1 -F "$LOG" 2>/dev/null | grep --line-buffered -E "controllo finestra|VERDETTO PILOTA" | while read -r riga; do
  fin=$(echo "$riga" | grep -oE "\[[0-9]+, [0-9]+\]" | head -1)
  primo=$(echo "$fin" | grep -oE "[0-9]+" | head -1)
  [ "${primo:-0}" -lt "$DA" ] && continue
  js=$(echo "$riga" | sed 's/^[^{]*//')
  giudizio=$(python3 -c "
import json,sys
v=json.loads(sys.argv[1]); c=v.get('conoscenza_fine_su_attuale',9); k=v.get('calib_fine_su_attuale',9)
ok = c <= $TC and k <= $TK
print(('OK' if ok else 'FERMA')+f' conoscenza {c:.4f} calibrazione {k:.4f} esito {v.get(\"esito\")}')" "$js" 2>&1)
  echo "[$(date '+%F %T')] finestra $fin: $giudizio"
  case "$giudizio" in
    OK*) printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "Riforgia $UNITA finestra $fin: $giudizio (guardiano: prosegue)" >> "$CICLI" ;;
    *)   printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "⛔ Riforgia $UNITA finestra $fin: $giudizio -> guardiano FERMA la corsa (finestre gia' commesse restano su disco)" >> "$CICLI"
         systemctl --user stop "$UNITA"; echo "[$(date '+%F %T')] $UNITA fermata"; exit 0 ;;
  esac
done
