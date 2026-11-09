#!/bin/bash
# 30/9 04:3x: criterio fallito (TT3-p4e MMLU 68,45 < 72,95): TT3 si ferma dopo che ripresa_000005 e' completa e stabile.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
D=@@ROOT:depot@@/e2e/allena/TT3; R=$D/ripresa_000005.pt
until [ -f $R ] && grep -q '000005' $D/ripresa_ultima.json 2>/dev/null; do sleep 30; done
s1=0; s2=1; until [ "$s1" = "$s2" ]; do s1=$(stat -c %s $R); sleep 60; s2=$(stat -c %s $R); done; sleep 30
echo "[$(date '+%F %T')] ripresa_000005 stabile ($(stat -c %s $R) byte): fermo TT3"
systemctl --user stop e2e-allena-TT3; RC=$?; echo "[$(date '+%F %T')] stop rc=$RC; attivo: $(systemctl --user is-active e2e-allena-TT3)"
