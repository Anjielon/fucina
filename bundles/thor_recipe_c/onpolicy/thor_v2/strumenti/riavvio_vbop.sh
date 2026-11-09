#!/bin/bash
# 27/9 14:2x: a fine passo 16 (anticipato dal 17 su richiesta di Angelo: le statistiche dell'accumulo 14-16 si buttano, V-Bop parte comunque da zero) si passa a V-Bop
# (momento persistente per gruppo con isteresi, sintetico 27 contro 2 salti giusti) + regione di fiducia adattiva per rho.
# Resta tutto il resto: lr F32 1e-4, LoRA Q8_0, CAKLD, GDN v5, triton2, EMA, held-out rapido a ogni passo.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
E=@@ROOT:lavoro@@/e2e/allena/EB2
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$E/ripresa_ultima.json'))['passo']>=16 else 1)" 2>/dev/null; do sleep 20; done
sleep 30
@@ROOT:forge@@/thor_v2/strumenti/riavvio_ora.sh --lr-f32 1e-4 --lora-nonesperti 16 --lora-tipi Q8_0 --lr-lora 1e-4 \
  --pv-modo vbop --pv-passi-accumulo 1 --pv-rho-adattivo
