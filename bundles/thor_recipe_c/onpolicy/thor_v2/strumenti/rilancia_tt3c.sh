#!/bin/bash
# 29/9 23:2x: a passo 3 chiuso (ripresa_000003.pt scritta e stabile) ferma TT3 e lo rilancia con le cure (codice TT3c-cfb718f).
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
D=@@ROOT:depot@@/e2e/allena/TT3; R=$D/ripresa_000003.pt
until [ -f $R ] && grep -q '000003' $D/ripresa_ultima.json 2>/dev/null; do sleep 30; done
s1=0; s2=1; until [ "$s1" = "$s2" ]; do s1=$(stat -c %s $R); sleep 60; s2=$(stat -c %s $R); done
sleep 30
echo "[$(date '+%F %T')] ripresa_000003 stabile ($(stat -c %s $R) byte): fermo TT3"
systemctl --user stop e2e-allena-TT3; sleep 10; systemctl --user reset-failed e2e-allena-TT3 2>/dev/null
COD=@@ROOT:depot@@/e2e/codice/TT3c-cfb718f @@ROOT:forge@@/thor_v2/strumenti/lancia_tt3.sh --riprendi --riprendi-da-ema --lr 2e-4 --pv-riuso --pv-ricentra-obiettivo
RC=$?; echo "[$(date "+%F %T")] rilancio rc=$RC"
