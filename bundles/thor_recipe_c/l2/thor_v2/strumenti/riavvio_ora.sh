#!/bin/bash
# 27/9 12:2x: riavvio IMMEDIATO dalla ripresa piu' recente con tutte le leve + motore triton2 (cancello superato, al bit) +
# held-out rapido (16 righe) a OGNI passo per misurare la discesa passo per passo (pieno ogni 16).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
R=@@ROOT:forge@@; E=@@ROOT:lavoro@@/e2e/allena/EB2
systemctl --user stop e2e-allena-EB2; sleep 15; systemctl --user reset-failed e2e-allena-EB2 2>/dev/null
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm
systemd-run --user --unit=e2e-allena-EB2 -p Nice=5 --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce \
  --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --setenv=E2E_MOTORE=${MOTORE:-triton2} --working-directory $R/thor_v2 \
  @@ROOT:venv@@/bin/python3 $R/thor_v2/e2e_allena.py allena --corsa EB2 --riprendi --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta B_ind32 --heldout-ogni 1 --heldout-righe-rapide 16 --heldout-pieno-ogni 16 --rotazione --K0 524288 --pv-codici --pv-rho 0.03 \
  --pv-passi-accumulo 4 --perdita cakld --gamma 0.552 --allena-f32 $F32 --lr-f32 1e-5 --ema-theta 0.9 --ripresa-ogni 1 "$@"
