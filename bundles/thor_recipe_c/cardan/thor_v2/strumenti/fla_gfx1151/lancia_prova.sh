#!/bin/bash
# Lancia una prova GPU della copia di fla come unita' systemd --user, accanto alla corsa EB2 e a nient'altro (27/9/2026).
# Uso: lancia_prova.sh NOME_UNITA [argomenti di prova_equivalenza.py] ; variabili: DIAG_SERIALIZZA=1 -> AMD_SERIALIZE_KERNEL=3
set -euo pipefail
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
QUI=$(cd "$(dirname "$0")" && pwd)
UNITA=${1:?nome unita}; shift
EB2=$(systemctl --user show e2e-allena-EB2 -p MainPID --value 2>/dev/null || echo 0)
# un solo processo GPU di prova accanto alla corsa: aspetta che nessun altro (tranne la corsa e i server di casa) tenga VRAM
if ! GPU_ESCLUDI_PID="$EB2" "$QUI/../gpu_occupata.sh" --attendi "${ATTESA_S:-1800}"; then
  echo "GPU occupata da un altro processo di prova: non lancio"; exit 3
fi
F32DOT=${F32DOT:-ieee}
ENVS=(--setenv=TRITON_CACHE_DIR="$QUI/cache_triton/$F32DOT" --setenv=FLA_CACHE_MODE=disabled --setenv=PYTHONUNBUFFERED=1)
[ "$F32DOT" != ieee ] && ENVS+=(--setenv=TRITON_F32_DEFAULT="$F32DOT")
[ "${DIAG_SERIALIZZA:-0}" = 1 ] && ENVS+=(--setenv=AMD_SERIALIZE_KERNEL=3)
[ -n "${FLA_GFX1151_CUMSUM64:-}" ] && ENVS+=(--setenv=FLA_GFX1151_CUMSUM64="$FLA_GFX1151_CUMSUM64")
mkdir -p "$QUI/esiti"
systemd-run --user --unit="$UNITA" --collect -p Nice=10 -p CPUQuota=400% --working-directory="$QUI" "${ENVS[@]}" \
  /usr/bin/env -u HSA_OVERRIDE_GFX_VERSION @@ROOT:venv@@/bin/python3 "$QUI/${PROVA:-prova_equivalenza.py}" "$@"
echo "lanciata $UNITA: journalctl --user -u $UNITA -f"
