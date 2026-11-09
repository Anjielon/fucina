#!/bin/bash
# 27/9 notte: decisione automatica di rho dopo la 1a misura held-out con PV (passo 8). Regola dichiarata prima del dato:
# KL held-out p8 <= 0,3690 (traiettoria sole scale: 0,399 -> 0,376 in 6 passi = -0,0038/passo -> ~0,368 al passo 8)
# => ripresa con --pv-rho 0.03 (prende tutti i codici convenienti: 24.709 contro 7.267 al passo 7). Altrimenti resta 0,01.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
D=@@ROOT:lavoro@@/e2e/allena/EB2; S=@@ROOT:forge@@/stato
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$D/ripresa_ultima.json'))['passo']>=8 else 1)" 2>/dev/null; do sleep 30; done
sleep 60
KL=$(grep -oE "held-out al passo 8: KL [0-9.]+" $D/registro.log | tail -1 | awk '{print $NF}')
if [ -n "$KL" ] && python3 -c "import sys;sys.exit(0 if float('$KL')<=0.3690 else 1)"; then
  systemctl --user stop e2e-allena-EB2
  for i in $(seq 1 60); do systemctl --user is-active -q e2e-allena-EB2 || break; sleep 5; done
  systemctl --user reset-failed e2e-allena-EB2 2>/dev/null
  systemd-run --user --unit=e2e-allena-EB2 --collect -p Nice=5 -p Restart=on-failure -p RestartSec=60 -p StartLimitBurst=3 \
    -p StartLimitIntervalSec=3600 --setenv=E2E_SENZA_COESISTENZA=1 --working-directory @@ROOT:forge@@/thor_v2 \
    @@ROOT:venv@@/bin/python3 @@ROOT:forge@@/thor_v2/e2e_allena.py allena --corsa EB2 --riprendi \
    --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno --ricetta B_ind32 --heldout-ogni 2 --rotazione --K0 524288 --pv-codici --pv-rho 0.03
  M="held-out p8 KL $KL <= 0,3690 -> ripartita con rho 0,03"
else
  M="held-out p8 KL ${KL:-?} > 0,3690 -> rho resta 0,01"
fi
echo "619 | $(date +%H:%M) DECISIONE RHO automatica: $M" >> $S/linee.txt
printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "decisione rho automatica: $M" >> $S/cicli.tsv
