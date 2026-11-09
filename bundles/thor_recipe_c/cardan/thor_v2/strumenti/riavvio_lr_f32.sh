#!/bin/bash
# 27/9 13:0x: a fine passo 15 (ripresa a ogni passo) riavvio con lr dei tensori F32 1e-5 -> 1e-4: a 1e-5 Adam sposta norme
# e router di ~1e-5 relativo a passo (niente nei ~5 passi di oggi); 1e-4 ~ stesso passo relativo delle scale theta (lr 1e-3 in log).
# Held-out rapido a ogni passo: se sale, si torna indietro alla ripresa precedente.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
E=@@ROOT:lavoro@@/e2e/allena/EB2
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$E/ripresa_ultima.json'))['passo']>=15 else 1)" 2>/dev/null; do sleep 20; done
sleep 30
@@ROOT:forge@@/thor_v2/strumenti/riavvio_ora.sh --lr-f32 1e-4 --lora-nonesperti 16 --lora-tipi Q8_0 --lr-lora 1e-4
