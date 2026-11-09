#!/bin/bash
# 27/9: cancello bf16x3 (ramo bf16x3, b883ce9) sulla GPU accanto a EB2, DOPO la misura v5 (una prova GPU per volta).
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
while systemctl --user is-active -q e2e-misura-v5; do sleep 15; done
EB2=$(systemctl --user show e2e-allena-EB2 -p MainPID --value)
env -u HSA_OVERRIDE_GFX_VERSION GPU_ESCLUDI_PID="$EB2" bash @@ROOT:home@@/claude-pod-lavoro/bf16x3/thor_v2/strumenti/cancello_b3.sh
echo "cancello rc=$?"
