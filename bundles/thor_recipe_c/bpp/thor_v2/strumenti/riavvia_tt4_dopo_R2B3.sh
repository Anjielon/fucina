#!/bin/bash
# 30/9 06:2x: TT4 fermato dopo 25 min per dare la CPU a R2B3 (a 8 thread la GPU di TT4 scendeva al 30 %).
# Quando la costruzione R2B3 (unita' costruisci-R2B3) e' finita, TT4 riparte DA CAPO (cartella pulita) e la catena resta armata.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
until grep -qE "SONDA PARTITA|COPIATO" @@ROOT:depot@@/tutto_ternario/log/dell_R2B3.log 2>/dev/null || ! systemctl --user is-active --quiet catena-R2B3; do sleep 60; done   # R2B3 al Dell, oppure la catena R2B3 e' finita (anche se bocciato)
echo "[$(date '+%F %T')] R2B3 al Dell; riparto TT4 da capo"
rm -rf @@ROOT:depot@@/e2e/allena/TT4 @@ROOT:depot@@/e2e_istantanee/TT4
systemctl --user reset-failed e2e-allena-TT4 2>/dev/null
@@ROOT:forge@@/thor_v2/strumenti/lancia_tt4.sh; echo "[$(date '+%F %T')] lancia_tt4 rc=$?"
