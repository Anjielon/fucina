#!/bin/bash
# 28/9 sera — TT2: recupero E2E del modello A2 (tutti i 250 densi ternari dal bf16, ricetta 20/9 + Hessiane 28/9).
# Corsa NUOVA dal GGUF A2 (theta0 = scale del file: EB3 p20 e' gia' dentro). Stessi argomenti dell'ultima EB3 tranne:
# --gguf A2, --densi-ternari tutte (densi come esperto unico, §47, ramo densi-ternari 7c07fd6, prove A-E verdi),
# niente LoRA (non ammessa sui ternari e in A2 non restano Q8_0 densi), niente --attribuzione (era nel punto dei blocchi
# di oggi; 13 % del passo), strati allenati tutti (i densi ternari sono in tutti i 40).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
COD=${COD:-@@ROOT:depot@@/e2e/codice/TT2-7c07fd6}; PY=@@ROOT:venv@@/bin/python3
G=@@ROOT:depot@@/tutto_ternario/gguf/Thor-T-A2.gguf; MC=@@ROOT:depot@@/e2e_teacher_mc
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm
[ "$(sha256sum $G | cut -d' ' -f1)" = c9eaccdcf55b20432f952a28a7f6eaf57648a01e9aa1ba8dac9c66f90946efc8 ] || { echo "sha di A2 diverso"; exit 2; }
python3 @@ROOT:forge@@/thor_v2/strumenti/verifica_righe_teacher.py @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno 512 || exit 2   # precondizione E67
mkdir -p @@ROOT:depot@@/e2e/allena/TT2 @@ROOT:depot@@/e2e_istantanee/TT2
echo "$COD" > @@ROOT:depot@@/e2e/allena/TT2/codice.txt
systemd-run --user --unit=e2e-allena-TT2 -p Nice=5 -p RestartForceExitStatus=75 -p RestartSec=180 -p StartLimitBurst=3 -p StartLimitIntervalSec=21600 \
  --setenv=E2E_ALLENA_DIR=@@ROOT:depot@@/e2e/allena --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce \
  --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --setenv=E2E_MOTORE=triton2 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --working-directory "$COD/thor_v2" "$PY" "$COD/thor_v2/e2e_allena.py" allena --corsa TT2 --gguf "$G" --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta B3_ind32 --testa-kl bf16x3 --logit-maestro bf16 --heldout-ogni 1 --heldout-righe-rapide 16 --heldout-pieno-ogni 2 --rotazione \
  --non-consumare --K0 524288 --pv-codici --pv-rho 0.03 --pv-passi-accumulo 1 --pv-modo vbop --pv-rho-adattivo \
  --perdita cakld --gamma 0.552 --allena-f32 "$F32" --lr-f32 5e-5 --ema-theta 0.9 --ripresa-ogni 1 \
  --ottimizzatore-theta adamw --pv-accetta guadagno --pv-z-guadagno-min 1 \
  --istantanea-ogni-passo --istantanee-dir @@ROOT:depot@@/e2e_istantanee/TT2 --istantanee-tieni 2 --istantanee-libero-min-gib 12 \
  --teacher-mc $MC/q8-mc --teacher-mc-heldout $MC/q8-mc-heldout --righe-mc 8 --righe-mc-permutate --peso-token risposta \
  --beta-risposta 0.1 --gamma-lettere 1 \
  --peso-dominio it_chat=1,strumenti=1,cot=1,edu=1.5,it_web=1.5,it_wiki=1.5,it_pensiero=1.5 \
  --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 --pv-ricentra --pv-ricentra-giri 3 \
  --densi-ternari tutte --righe-base-da 512 "$@"  # 28/9 21:3x: righe 0-511 del teacher consumate da EB2 (senza, la corsa aspetta per sempre)
