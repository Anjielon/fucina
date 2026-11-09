#!/bin/bash
# 27/9 — cancello del motore «triton2» (kernel esperti) sulla GPU: strumenti/cancello_kernel.py.
# Strati 0,1,38,39 (default), ricetta B, 8192 token veri: ms/token per strato vecchio («triton») contro nuovo
# («triton2»), microprova dei kernel per famiglia, dtheta come G1 (quota nulli+segno sbagliato contro l'indietro
# fp32 <= 1e-3; uscita e dL/dx del nuovo entro 1e-5 dal vecchio). NESSUN file del modello scritto.
# SOLO a GPU libera (strumenti/gpu_occupata.sh --attendi, e di nuovo dentro il python).
# Argomenti in piu' passano al python, es.:  --token 16384  --ricetta C  --blocchi '{"BS2": 128, "BN2": 64}'
# Dalla shell interattiva, se la GPU non si vede, lanciarlo come unita' utente:
#   export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
#   systemd-run --user --collect --unit=cancello-kernel -p Nice=5 thor_v2/strumenti/cancello_kernel.sh
set -uo pipefail
QUI="$(cd "$(dirname "$0")" && pwd)"
cd "$QUI/.."
D=${CANCELLO_DIR:-@@ROOT:models@@/gguf/tony-tern/e2e_studente}
T=$(date +%Y%m%d-%H%M%S)
mkdir -p "$D"
if ! occ=$("$QUI/gpu_occupata.sh" --attendi 120); then
  echo "⛔ cancello kernel NON partito: GPU occupata: $occ"; exit 3
fi
unset TRITON_INTERPRET CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
nice -n 5 @@ROOT:venv@@/bin/python3 strumenti/cancello_kernel.py --uscita "$D/cancello_kernel_$T.json" "$@" \
  2>&1 | tee "$D/cancello_kernel_$T.log"
rc=${PIPESTATUS[0]}
echo "rc=$rc · $D/cancello_kernel_$T.json · $D/cancello_kernel_$T.log"
exit "$rc"
