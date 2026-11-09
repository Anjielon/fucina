#!/bin/bash
# 27/9: GDN veloce v5 (copia congelata 9b99f34) nel percorso vero, stesse condizioni della misura a tre (q8-tutte-riga, B_ind32,
# strati 0 e 0,1, EB2 accanto), dθ confrontato col ripiego gia' salvato in @@ROOT:forge@@/thor_v2/strumenti/fla_gfx1151/esiti/misura_tre.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
EB2=$(systemctl --user show e2e-allena-EB2 -p MainPID --value)
cd @@ROOT:forge@@/thor_v2 && env -u HSA_OVERRIDE_GFX_VERSION GPU_ESCLUDI_PID="$EB2" E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MISURA_ETICHETTA=veloce5   E2E_MISURA_DTHETA=@@ROOT:forge@@/thor_v2/strumenti/fla_gfx1151/esiti/misura_tre @@ROOT:venv@@/bin/python3 e2e_allena.py misura --teacher @@ROOT:lavoro@@/e2e/teacher/q8-tutte-riga   --strati 0 --strati 0,1 --ripetizioni 2 --ricetta B_ind32 --uscita @@ROOT:forge@@/thor_v2/strumenti/fla_gfx1151/esiti/misura_tre/misura_veloce5.json 2>&1 | grep -v "UserWarning\|warn(" | tail -15
@@ROOT:venv@@/bin/python3 @@ROOT:forge@@/thor_v2/strumenti/fla_gfx1151/confronta_dtheta.py @@ROOT:forge@@/thor_v2/strumenti/fla_gfx1151/esiti/misura_tre ripiego 2>&1 | tail -12
