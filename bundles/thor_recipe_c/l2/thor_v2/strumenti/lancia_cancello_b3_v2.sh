#!/bin/bash
# 27/9 14:2x: cancello bf16x3 v2 (ramo bf16x3fix d2797af: kernel Triton con accumulatore per riquadro, 0,35-0,47x fp32, errore
# 4,4e-6) sui 40 strati, DOPO la KLD del passo 16 del sorvegliante (mai tre lavori GPU insieme: ring timeout delle 10:31).
# Scoperta 14:22 (pod triton3): nell'indietro gli esperti sono solo il 7-16% dello strato; il resto sono GEMM non-esperte fp32
# -> bf16x3 e' la leva vera di velocita' del passo.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
K=@@ROOT:lavoro@@/e2e/fedelta/kld/kld_EB2-migliore_ema_000016.log
until grep -qE "Mean *KLD" $K 2>/dev/null; do sleep 30; done
sleep 60
B=@@ROOT:home@@/claude-pod-lavoro/bf16x3fix/thor_v2
for f in stato p0_dopo_valutazioni.sh mappa_completa_gguf_hf_perm.npz e2e_dump_logit; do [ -e $B/$f ] || ln -s @@ROOT:forge@@/thor_v2/$f $B/$f; done
EB2=$(systemctl --user show e2e-allena-EB2 -p MainPID --value)
env -u HSA_OVERRIDE_GFX_VERSION GPU_ESCLUDI_PID="$EB2" bash $B/strumenti/cancello_b3.sh
echo "cancello rc=$?"
