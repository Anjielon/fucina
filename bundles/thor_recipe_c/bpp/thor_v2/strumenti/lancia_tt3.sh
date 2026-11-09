#!/bin/bash
# 29/9 pomeriggio — TT3: recupero E2E partendo da R1 (A2 + secondo piano ternario su tutti i 250 densi, MMLU 71,85 senza
# allenamento) invece che da A2. PREPARATO, NON LANCIATO. Stessi argomenti di TT2 (lancia_tt2.sh + quelli dati a mano a
# TT2: --pv-punto-medio --pv-pm-primitiva famiglia --pv-pm-max 12 --pv-pm-lettere --pv-pm-golosa), tranne:
#   --gguf R1 (o R2 con G=... e SHA=...), corsa/unita'/cartelle TT3, codice congelato TT3-df753ee (ramo secondo-piano-e2e:
#   i *2 sono famiglie dense proprie: scale theta allenate, codici nel passo V con f = 13..22, export), niente --riprendi.
# Prove del codice (CPU, tutte verdi): e2e_test_secondo_piano.py I-VI.
# Precondizioni controllate qui: sha del GGUF, TT2 FERMO (la GPU e' una sola), >= 16 GiB liberi su @@ROOT:depot@@.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
COD=${COD:-@@ROOT:depot@@/e2e/codice/TT3-df753ee}; PY=@@ROOT:venv@@/bin/python3
G=${G:-@@ROOT:models@@/gguf/tony-tern/R1/Thor-T-R1.gguf}
SHA=${SHA:-b102ed9d3ebd4e260e7bb419c934c2180aefd3b0fd4895265a0feff7b2c0b5c6}     # R1 (cicli.tsv 29/9 11:19)
MC=@@ROOT:depot@@/e2e_teacher_mc
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm
[ "$(sha256sum "$G" | cut -d' ' -f1)" = "$SHA" ] || { echo "sha di $G diverso da $SHA"; exit 2; }
if systemctl --user is-active --quiet e2e-allena-TT2; then echo "TT2 e' ancora attivo: la GPU e' sua"; exit 2; fi
LIBERI=$(df -B1G --output=avail @@ROOT:depot@@ | tail -1 | tr -d ' ')
[ "$LIBERI" -ge 16 ] || { echo "@@ROOT:depot@@: $LIBERI GiB liberi, ne servono 16"; exit 2; }
[ -f "$COD/thor_v2/e2e_studente.py" ] && grep -q FAMIGLIE_DENSE2 "$COD/thor_v2/e2e_studente.py" || { echo "codice $COD senza secondo piano"; exit 2; }
python3 @@ROOT:forge@@/thor_v2/strumenti/verifica_righe_teacher.py @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno 512 || exit 2   # precondizione E67
mkdir -p @@ROOT:depot@@/e2e/allena/TT3 @@ROOT:depot@@/e2e_istantanee/TT3
echo "$COD" > @@ROOT:depot@@/e2e/allena/TT3/codice.txt
systemd-run --user --unit=e2e-allena-TT3 -p Nice=5 -p RestartForceExitStatus=75 -p RestartSec=180 -p StartLimitBurst=3 -p StartLimitIntervalSec=21600 \
  --setenv=E2E_ALLENA_DIR=@@ROOT:depot@@/e2e/allena --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce \
  --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --setenv=E2E_MOTORE=triton2 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --working-directory "$COD/thor_v2" "$PY" "$COD/thor_v2/e2e_allena.py" allena --corsa TT3 --gguf "$G" --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta B3_ind32 --testa-kl bf16x3 --logit-maestro bf16 --heldout-ogni 1 --heldout-righe-rapide 16 --heldout-pieno-ogni 2 --rotazione \
  --non-consumare --K0 524288 --pv-codici --pv-rho 0.03 --pv-passi-accumulo 1 --pv-modo vbop --pv-rho-adattivo \
  --perdita cakld --gamma 0.552 --allena-f32 "$F32" --lr-f32 5e-5 --ema-theta 0.9 --ripresa-ogni 1 \
  --ottimizzatore-theta adamw --pv-accetta guadagno --pv-z-guadagno-min 1 \
  --istantanea-ogni-passo --istantanee-dir @@ROOT:depot@@/e2e_istantanee/TT3 --istantanee-tieni 2 --istantanee-libero-min-gib 12 \
  --teacher-mc $MC/q8-mc --teacher-mc-heldout $MC/q8-mc-heldout --righe-mc 8 --righe-mc-permutate --peso-token risposta \
  --beta-risposta 0.1 --gamma-lettere 1 \
  --peso-dominio it_chat=1,strumenti=1,cot=1,edu=1.5,it_web=1.5,it_wiki=1.5,it_pensiero=1.5 \
  --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 --pv-ricentra --pv-ricentra-giri 3 \
  --pv-punto-medio --pv-pm-primitiva famiglia --pv-pm-max 12 --pv-pm-lettere --pv-pm-golosa \
  --densi-ternari tutte --righe-base-da 512 "$@"
