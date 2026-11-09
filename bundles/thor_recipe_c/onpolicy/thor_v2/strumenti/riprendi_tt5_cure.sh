#!/bin/bash
# 1/10 sera — TT5 RIPRESA da ripresa_000002 con le cure del rapporto pod tt5-lentezza (cure 1, 2, 7): stessa matematica
# dell'allenamento sulle scale, meno lavoro attorno. Ordine di Angelo: «TT5 DEVE SCENDERE».
#  cura 1  --heldout-pieno-ogni 8   (era 2: il pieno costava ~1549 s a passo)        campo di ESECUZIONE
#  cura 2  NON applicabile: il codice vieta pv_passi_accumulo>1 con pv_modo vbop (e2e_allena.py:529). Resta da misurare 2b (pv-pm-max, pv-ricentra-giri).
#  cura 7  istantanea ogni passo resta (serve alla ripresa); costo ~45 s, non si tocca
# NON in questo lancio (richiedono codice): cura 3 (B3 fp16), cura 5 (cache embedding/testa a 21 min per avvio).
# Argomenti identici a lancia_tt5.sh tranne i due sopra e --riprendi. Il passo 2 e' gia' fatto (R2B3m + 2 passi).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
COD=$(cat @@ROOT:depot@@/e2e/allena/TT5/codice.txt); PY=@@ROOT:venv@@/bin/python3
G=@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf; SHA=56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43
MC=@@ROOT:depot@@/e2e_teacher_mc
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm
[ "$(sha256sum "$G" | cut -d' ' -f1)" = "$SHA" ] || { echo "sha di $G diverso"; exit 2; }
R=@@ROOT:depot@@/e2e/allena/TT5/ripresa_000002.pt
[ "$(sha256sum "$R" | cut -d' ' -f1)" = "$(python3 -c "import json;print(json.load(open('@@ROOT:depot@@/e2e/allena/TT5/ripresa_ultima.json'))['sha256'])")" ] || { echo "ripresa_000002 sha diverso dal registro"; exit 2; }
for u in $(systemctl --user list-units --state=active --no-legend 'e2e-allena-*' 'catena-R2B4*' | awk '{print $1}'); do echo "GPU occupata: $u"; exit 2; done
LIBERI=$(df -B1G --output=avail @@ROOT:depot@@ | tail -1 | tr -d ' '); [ "$LIBERI" -ge 30 ] || { echo "@@ROOT:depot@@: $LIBERI GiB"; exit 2; }
python3 @@ROOT:forge@@/thor_v2/strumenti/verifica_righe_teacher.py @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno 512 || exit 2
systemd-run --user --unit=e2e-allena-TT5 -p Nice=5 -p RestartForceExitStatus=75 -p RestartSec=180 -p StartLimitBurst=3 -p StartLimitIntervalSec=21600 \
  --setenv=E2E_ALLENA_DIR=@@ROOT:depot@@/e2e/allena --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce \
  --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --setenv=E2E_MOTORE=triton2 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --working-directory "$COD/thor_v2" "$PY" "$COD/thor_v2/e2e_allena.py" allena --corsa TT5 --gguf "$G" --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta B3_ind32 --testa-kl bf16x3 --logit-maestro bf16 --heldout-ogni 1 --heldout-righe-rapide 16 --heldout-pieno-ogni 8 --rotazione \
  --non-consumare --K0 524288 --pv-codici --pv-rho 0.03 --pv-passi-accumulo 1 --pv-modo vbop --pv-rho-adattivo \
  --perdita cakld --gamma 0.552 --allena-f32 "$F32" --lr-f32 5e-5 --ema-theta 0.9 --ripresa-ogni 1 \
  --ottimizzatore-theta adamw --pv-accetta guadagno --pv-z-guadagno-min 1 \
  --istantanea-ogni-passo --istantanee-dir @@ROOT:depot@@/e2e_istantanee/TT5 --istantanee-tieni 2 --istantanee-libero-min-gib 12 \
  --teacher-mc $MC/q8-mc --teacher-mc-heldout $MC/q8-mc-heldout --righe-mc 8 --righe-mc-permutate --peso-token risposta \
  --beta-risposta 0.1 --gamma-lettere 1 \
  --peso-dominio it_chat=1,strumenti=1,cot=1,edu=1.5,it_web=1.5,it_wiki=1.5,it_pensiero=1.5 \
  --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 --pv-ricentra --pv-ricentra-giri 3 \
  --pv-punto-medio --pv-pm-primitiva famiglia --pv-pm-max 12 --pv-pm-lettere --pv-pm-golosa \
  --densi-ternari tutte --righe-base-da 512 --pv-riuso --riprendi "$@"
