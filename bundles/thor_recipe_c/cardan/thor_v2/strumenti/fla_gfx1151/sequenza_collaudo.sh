#!/bin/bash
# Collaudi GPU della ricetta fla gfx1151 in sequenza (27/9/2026): UN processo di prova alla volta accanto a EB2, mai
# durante il riavvio di EB2 (e2e-riavvio-accumulo la ferma dopo la ripresa del passo 12 e la riavvia): una prova si
# lancia solo se EB2 e' viva e caricata (> 30 GiB) e non e' a ridosso della fine del passo 12 (vecchio PID dopo le 09:02).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
QUI=$(cd "$(dirname "$0")" && pwd); THOR=$(cd "$QUI/../.." && pwd)
PID_VECCHIO=${PID_VECCHIO:-3443154}
TEACHER=@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno
PY=@@ROOT:venv@@/bin/python3
eb2_pronta() {
  local p; p=$(systemctl --user show e2e-allena-EB2 -p MainPID --value 2>/dev/null); [ "${p:-0}" -gt 0 ] || return 1
  local b=0 f; for f in /sys/class/kfd/kfd/proc/$p/vram_*; do [ -r "$f" ] && b=$((b + $(cat "$f"))); done
  [ $((b / 1048576)) -gt 30000 ] || return 1
  if [ "$p" = "$PID_VECCHIO" ] && [ "$(date +%H%M)" -ge 0902 ]; then return 1; fi
  echo "$p"
}
attendi() {
  while :; do
    p=$(eb2_pronta) && GPU_ESCLUDI_PID="$p" "$QUI/../gpu_occupata.sh" >/dev/null && { echo "$p"; return; }
    sleep 20
  done
}
corri() {   # corri NOME comando... (systemd --user, attende la fine)
  local u=$1; shift; local p; p=$(attendi)
  echo "$(date +%T) via $u (EB2 $p)"
  systemd-run --user --unit="$u" --collect --wait -p Nice=10 -p CPUQuota=400% --working-directory="$THOR" \
    --setenv=PYTHONUNBUFFERED=1 --setenv=GPU_ESCLUDI_PID="$p" "$@" 2>&1 | tail -3
  echo "$(date +%T) fine $u"
}
corri e2e-fla-abilita /usr/bin/env -u HSA_OVERRIDE_GFX_VERSION $PY "$QUI/prova_abilita.py"
corri e2e-fla-corsa /usr/bin/env -u HSA_OVERRIDE_GFX_VERSION $PY "$QUI/prova_corsa.py" --teacher $TEACHER
for modo in ripiego fla; do
  corri e2e-fla-misura-$modo /usr/bin/env -u HSA_OVERRIDE_GFX_VERSION $PY "$QUI/prova_misura.py" --gdn $modo \
    --teacher $TEACHER --strati 0 --strati 0,1 --ripetizioni 2 --ricetta B_ind32 --uscita "$QUI/esiti/misura_$modo.json"
done
echo "$(date +%T) SEQUENZA FINITA"
