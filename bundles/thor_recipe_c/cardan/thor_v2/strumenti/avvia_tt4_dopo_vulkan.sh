#!/bin/bash
# 30/9 06:0x: parte TT4 (2 passi da R2) quando il collaudo Vulkan ODINO scrive "GPU LIBERA dopo Vulkan" in cicli.tsv
# e la GPU e' davvero ferma (<10 % per 60 s). Poi arma la catena stop/export/Dell.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=@@ROOT:forge@@/thor_v2/strumenti
until grep -q "GPU LIBERA dopo Vulkan" @@ROOT:forge@@/stato/cicli.tsv; do sleep 30; done
ok=0; until [ $ok -ge 12 ]; do g=$(cat /sys/class/drm/card*/device/gpu_busy_percent | sort -n | tail -1); [ "$g" -lt 10 ] && ok=$((ok+1)) || ok=0; sleep 5; done
echo "[$(date '+%F %T')] GPU libera: lancio TT4"
$S/lancia_tt4.sh; RC=$?; echo "[$(date '+%F %T')] lancia_tt4 rc=$RC"
[ $RC = 0 ] || exit $RC
systemd-run --user --unit=catena-tt4 --quiet bash -c "$S/catena_tt4.sh >> @@ROOT:depot@@/tutto_ternario/log/catena_tt4.log 2>&1"
echo "[$(date '+%F %T')] catena-tt4: $(systemctl --user is-active catena-tt4)"
