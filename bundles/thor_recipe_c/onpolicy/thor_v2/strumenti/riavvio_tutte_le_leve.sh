#!/bin/bash
# 27/9 10:3x: riavvio di EB2 con TUTTE le leve verificate, a fine passo (ripresa a ogni passo), DOPO il cancello bf16x3
# (una prova GPU per volta; il cancello decide se entra anche B3). Leve: GDN veloce v5 (copia congelata 9b99f34, dθ 2,2e-5),
# CAKLD gamma 0,552, tensori F32 allenabili (13 famiglie, lr 1e-5), accumulo V a finestra 4 (corretto, A19/A28 verdi),
# held-out rapido 16 righe ogni 4 passi e pieno ogni 16, EMA di theta 0,9, rumore del gradiente (sempre acceso).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
R=@@ROOT:forge@@; ST=$R/stato; E=@@ROOT:lavoro@@/e2e/allena/EB2; C=@@ROOT:lavoro@@/e2e/allena/cancello_b3/esito_cancello_b3.json
scrivi(){ echo "6xx | $(date +%H:%M) $*" >> $ST/linee.txt; printf "%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "$1" "${*:2}" >> $ST/cicli.tsv; echo "$*"; }
while systemctl --user is-active -q e2e-cancello-b3; do sleep 20; done
RIC="--ricetta B_ind32"; NOTA="B_ind32 (cancello B3 non superato o assente)"
if python3 -c "import json,sys;d=json.load(open('$C'));sys.exit(0 if d.get('verdetto',{}).get('ammessa') else 1)" 2>/dev/null; then
  RIC="--ricetta B3_ind32 --testa-kl bf16x3 --logit-maestro bf16"; NOTA="B3_ind32 + testa bf16x3 + logit maestro bf16 (cancello ammesso)"
fi
P0=$(python3 -c "import json;print(json.load(open('$E/ripresa_ultima.json'))['passo'])")
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$E/ripresa_ultima.json'))['passo']>$P0 else 1)" 2>/dev/null; do sleep 20; done
sleep 30
systemctl --user stop e2e-allena-EB2; sleep 15; systemctl --user reset-failed e2e-allena-EB2 2>/dev/null
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm
systemd-run --user --unit=e2e-allena-EB2 -p Nice=5 --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce \
  --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --working-directory $R/thor_v2 \
  @@ROOT:venv@@/bin/python3 $R/thor_v2/e2e_allena.py allena --corsa EB2 --riprendi --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  $RIC --heldout-ogni 4 --heldout-righe-rapide 16 --heldout-pieno-ogni 16 --rotazione --K0 524288 --pv-codici --pv-rho 0.03 \
  --pv-passi-accumulo 4 --perdita cakld --gamma 0.552 --allena-f32 $F32 --lr-f32 1e-5 --ema-theta 0.9 --ripresa-ogni 1
sleep 240
scrivi ATTIVO "EB2 riavviata con tutte le leve ($NOTA): $(tail -6 $E/registro.log | cut -c1-140 | tr '\n' ' ')"
