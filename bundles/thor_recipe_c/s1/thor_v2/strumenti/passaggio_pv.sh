#!/bin/bash
# 27/9 notte: passaggio automatico EB2 (sole scale) -> EB2 con PV sui codici, alla ripresa del passo >= 6, senza quota Claude.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
D=@@ROOT:lavoro@@/e2e/allena/EB2; LOG=@@ROOT:forge@@/stato/linee.txt
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$D/ripresa_ultima.json'))['passo']>=6 else 1)" 2>/dev/null; do sleep 30; done
sleep 90
P=$(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['passo'])")
systemctl --user stop e2e-allena-EB2
for i in $(seq 1 60); do systemctl --user is-active -q e2e-allena-EB2 || break; sleep 5; done
systemctl --user reset-failed e2e-allena-EB2 2>/dev/null
systemd-run --user --unit=e2e-allena-EB2 --collect -p Nice=5 -p Restart=on-failure -p RestartSec=60 -p StartLimitBurst=3 \
  -p StartLimitIntervalSec=3600 --setenv=E2E_SENZA_COESISTENZA=1 --working-directory @@ROOT:forge@@/thor_v2 \
  @@ROOT:venv@@/bin/python3 @@ROOT:forge@@/thor_v2/e2e_allena.py allena --corsa EB2 --riprendi \
  --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno --ricetta B_ind32 --heldout-ogni 2 --rotazione --K0 524288 --pv-codici
echo "615 | $(date +%H:%M) PASSAGGIO AUTOMATICO: EB2 fermata alla ripresa del passo $P e ripartita con --pv-codici (P+V), teacher lasciato in pausa (E2E_SENZA_COESISTENZA=1)" >> "$LOG"
printf "%s\tATTIVO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "CORSA PV (scale+codici) ripartita dalla ripresa del passo $P di EB2: prima discesa PV al prossimo held-out" >> @@ROOT:forge@@/stato/cicli.tsv
