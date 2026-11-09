#!/bin/bash
# 27/9 09:4x: A15 (ripresa a meta' finestra di accumulo nel ciclo completo) ROSSO -> accumulo rinviato. Si riavvia EB2 dalla
# ripresa del passo 12 con il solo GDN v3 (E2E_GDN=veloce, copia congelata 8ff7c3f, verificata nel percorso vero) e ripresa a
# ogni passo (le leve successive entrano a fine di qualunque passo). Stessa matematica del passo P e del passo V di prima.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
R=@@ROOT:forge@@; ST=$R/stato; E=@@ROOT:lavoro@@/e2e/allena/EB2
scrivi(){ echo "6xx | $(date +%H:%M) $*" >> $ST/linee.txt; printf "%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "$1" "${*:2}" >> $ST/cicli.tsv; echo "$*"; }
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$E/ripresa_ultima.json'))['passo']>=12 else 1)" 2>/dev/null; do sleep 20; done
sleep 30
systemctl --user stop e2e-allena-EB2; sleep 15
systemctl --user reset-failed e2e-allena-EB2 2>/dev/null
systemd-run --user --unit=e2e-allena-EB2 -p Nice=5 --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce --working-directory $R/thor_v2 \
  @@ROOT:venv@@/bin/python3 $R/thor_v2/e2e_allena.py allena --corsa EB2 --riprendi --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta B_ind32 --heldout-ogni 4 --rotazione --K0 524288 --pv-codici --pv-rho 0.03 --ripresa-ogni 1
sleep 120
scrivi ATTIVO "EB2 riavviata dal passo 12 con GDN v3 e ripresa a ogni passo: $(grep -E 'GDN|ripresa' $E/registro.log | tail -3 | cut -c1-160 | tr '\n' ' ')"
