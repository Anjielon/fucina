#!/bin/bash
# 27/9: alla ripresa del passo >= 10 la corsa PV riparte con held-out ogni 4 passi (sorveglianza: stessa matematica dei passi;
# ogni passo V ha comunque la sua sonda). ~15% di tempo in meno. Nessuna quota Claude.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
D=@@ROOT:lavoro@@/e2e/allena/EB2; S=@@ROOT:forge@@/stato
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$D/ripresa_ultima.json'))['passo']>=10 else 1)" 2>/dev/null; do sleep 30; done
sleep 90
systemctl --user stop e2e-allena-EB2
for i in $(seq 1 60); do systemctl --user is-active -q e2e-allena-EB2 || break; sleep 5; done
systemctl --user reset-failed e2e-allena-EB2 2>/dev/null
systemd-run --user --unit=e2e-allena-EB2 --collect -p Nice=5 -p Restart=on-failure -p RestartSec=60 -p StartLimitBurst=3 \
  -p StartLimitIntervalSec=3600 --setenv=E2E_SENZA_COESISTENZA=1 --working-directory @@ROOT:forge@@/thor_v2 \
  @@ROOT:venv@@/bin/python3 @@ROOT:forge@@/thor_v2/e2e_allena.py allena --corsa EB2 --riprendi \
  --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno --ricetta B_ind32 --heldout-ogni 4 --rotazione --K0 524288 --pv-codici --pv-rho 0.03
echo "622 | $(date +%H:%M) cadenza held-out 2 -> 4 applicata alla ripresa del passo 10 (corsa PV ripartita)" >> $S/linee.txt
printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "cadenza held-out 2 -> 4 dalla ripresa del passo 10 (~15% di tempo in meno, stessa matematica)" >> $S/cicli.tsv
