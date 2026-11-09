#!/bin/bash
# 30/9 08:1x: TT4 era partito sopra il banco Vulkan di v34a (88 GiB in VRAM, messo in pausa dalla guardia): TT4 bloccato
# all'avvio (GPU 0 %, 41 MB). Fermato. Riparte quando l'unita' banco-mmid-dup e' finita e la GPU e' ferma da 60 s.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
while systemctl --user is-active --quiet banco-mmid-dup; do sleep 60; done
ok=0; until [ $ok -ge 12 ]; do g=$(cat /sys/class/drm/card*/device/gpu_busy_percent | sort -n | tail -1); [ "$g" -lt 10 ] && ok=$((ok+1)) || ok=0; sleep 5; done
echo "[$(date '+%F %T')] banco finito, GPU ferma: TT4 da capo"
rm -rf @@ROOT:depot@@/e2e/allena/TT4 @@ROOT:depot@@/e2e_istantanee/TT4; systemctl --user reset-failed e2e-allena-TT4 2>/dev/null
@@ROOT:forge@@/thor_v2/strumenti/lancia_tt4.sh; echo "[$(date '+%F %T')] lancia_tt4 rc=$?"
