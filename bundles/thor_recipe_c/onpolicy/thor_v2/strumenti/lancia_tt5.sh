#!/bin/bash
# 30/9 — TT5: distillazione CORTA (2 passi, poi stop da catena_tt5.sh) partendo da R2B3m (74,85: R2B + TERZO piano ternario
# su tutti i 250 densi + tabelle token_embd/output a 3 piani + testa MTP ternaria, 8.846.968.608 B). Stessa ricetta e stessi
# argomenti di TT4 (lancia_tt4.sh), stesso lr; cambiano SOLO: il GGUF di partenza, il codice e la corsa.
# Motivo: TT3 da +1 nei primi 2 passi (TT3-p2e 72,95 vs R1 71,85, p 0,043) e poi perde; TT4 (da R2) e' morto al salvataggio
# del passo 2 (disco pieno, 30/9 11:58): la distillazione riparte dal modello MIGLIORE, cosi' un guadagno si somma a lui.
# CRITERIO SCRITTO PRIMA DEL LANCIO: successo se MMLU 2000 (Dell, banco_ot, mmlu_00000) di TT5-p2 >= 75,85 (= R2B3 74,85 + 1)
# con McNemar appaiato p < 0,05 contro R2B3 (@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/R2B3.log; R2B3m ha logit
# identici al bit a R2B3) e sonda non peggiore del padre (18/21, Fisher unilaterale alfa 0,05, 0 mute).
# Codice: @@ROOT:depot@@/e2e/codice/TT5-<hash> = ramo tt5-terzo-piano (da cfb718f = TT3c): terzo piano dei densi come famiglia
# propria (theta, codici V, export), embedding/testa = somma dei 3 piani, salvataggio che non muore a disco pieno.
# Prove CPU: e2e_test_terzo_piano.py (esito in @@ROOT:depot@@/e2e/prove_tt5/terzo_piano/esito_test_terzo_piano.json).
# PREPARATO, NON LANCIATO. Lancio: bash lancia_tt5.sh  (poi, SUBITO dopo, la catena: vedi in fondo)
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
COD=${COD:-$(cat @@ROOT:depot@@/e2e/codice/TT5.attuale 2>/dev/null)}; PY=@@ROOT:venv@@/bin/python3
G=${G:-@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf}
SHA=${SHA:-56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43}     # R2B3m
MC=@@ROOT:depot@@/e2e_teacher_mc
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm
[ -n "$COD" ] && [ -f "$COD/thor_v2/e2e_studente.py" ] || { echo "codice TT5 assente ($COD)"; exit 2; }
grep -q FAMIGLIE_DENSE3 "$COD/thor_v2/e2e_studente.py" || { echo "codice $COD senza terzo piano"; exit 2; }
grep -q _attendi_spazio "$COD/thor_v2/e2e_allena.py" || { echo "codice $COD senza salvataggio a prova di disco pieno"; exit 2; }
ESITO=@@ROOT:depot@@/e2e/prove_tt5/terzo_piano/esito_test_terzo_piano.json
python3 - "$ESITO" <<'PYX' || { echo "prove del terzo piano non tutte verdi ($ESITO)"; exit 2; }
import json, sys
import glob
p = {}
import os
for f in sorted(glob.glob(sys.argv[1].replace(".json", "*.json")), key=os.path.getmtime):   # il giro piu' recente vince
    p.update(json.load(open(f))["prove"])
chieste = ["I_senza_p3_identico_a_cfb718f", "II_p3_a_zero_uguale_a_senza", "III_contro_llama_cpu", "IV_export_rilettura_al_bit",
           "V_cd_mutazioni_export", "V_ab_mutazioni_avanti", "VI_passo_V_sui_p3", "VII_testa_embedding_tre_piani"]
manca = [k for k in chieste if not p.get(k, {}).get("ok")]
if manca:
    print("rosse o mancanti:", manca)
    sys.exit(1)
PYX
python3 -c "import json,sys; sys.exit(0 if json.load(open('@@ROOT:depot@@/e2e/prove_tt5/ritorno/esito_ritorno_tt5.json'))['TUTTO_VERDE'] else 1)" \
  || { echo "andata e ritorno su R2B3m non verde (@@ROOT:depot@@/e2e/prove_tt5/ritorno/esito_ritorno_tt5.json)"; exit 2; }
[ "$(sha256sum "$G" | cut -d' ' -f1)" = "$SHA" ] || { echo "sha di $G diverso da $SHA"; exit 2; }
# GPU libera: nessun allenamento, nessuna prova di R2B4, nessun dump/banco sulla GPU
for u in $(systemctl --user list-units --state=active --no-legend 'e2e-allena-*' 'catena-R2B4*' 'prova-c-R2B4*' | awk '{print $1}'); do
  echo "GPU occupata: $u attiva"; exit 2; done
if pgrep -x llama-perplexit >/dev/null || pgrep -f "dump_logit" >/dev/null; then echo "GPU/CPU occupata da un dump o banco"; exit 2; fi
LIBERI=$(df -B1G --output=avail @@ROOT:depot@@ | tail -1 | tr -d ' ')
[ "$LIBERI" -ge 30 ] || { echo "@@ROOT:depot@@: $LIBERI GiB liberi, ne servono 30"; exit 2; }
python3 @@ROOT:forge@@/thor_v2/strumenti/verifica_righe_teacher.py @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno 512 || exit 2   # precondizione E67
mkdir -p @@ROOT:depot@@/e2e/allena/TT5 @@ROOT:depot@@/e2e_istantanee/TT5
echo "$COD" > @@ROOT:depot@@/e2e/allena/TT5/codice.txt
systemd-run --user --unit=e2e-allena-TT5 -p Nice=5 -p RestartForceExitStatus=75 -p RestartSec=180 -p StartLimitBurst=3 -p StartLimitIntervalSec=21600 \
  --setenv=E2E_ALLENA_DIR=@@ROOT:depot@@/e2e/allena --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce \
  --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --setenv=E2E_MOTORE=triton2 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --working-directory "$COD/thor_v2" "$PY" "$COD/thor_v2/e2e_allena.py" allena --corsa TT5 --gguf "$G" --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta B3_ind32 --testa-kl bf16x3 --logit-maestro bf16 --heldout-ogni 1 --heldout-righe-rapide 16 --heldout-pieno-ogni 2 --rotazione \
  --non-consumare --K0 524288 --pv-codici --pv-rho 0.03 --pv-passi-accumulo 1 --pv-modo vbop --pv-rho-adattivo \
  --perdita cakld --gamma 0.552 --allena-f32 "$F32" --lr-f32 5e-5 --ema-theta 0.9 --ripresa-ogni 1 \
  --ottimizzatore-theta adamw --pv-accetta guadagno --pv-z-guadagno-min 1 \
  --istantanea-ogni-passo --istantanee-dir @@ROOT:depot@@/e2e_istantanee/TT5 --istantanee-tieni 2 --istantanee-libero-min-gib 12 \
  --teacher-mc $MC/q8-mc --teacher-mc-heldout $MC/q8-mc-heldout --righe-mc 8 --righe-mc-permutate --peso-token risposta \
  --beta-risposta 0.1 --gamma-lettere 1 \
  --peso-dominio it_chat=1,strumenti=1,cot=1,edu=1.5,it_web=1.5,it_wiki=1.5,it_pensiero=1.5 \
  --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 --pv-ricentra --pv-ricentra-giri 3 \
  --pv-punto-medio --pv-pm-primitiva famiglia --pv-pm-max 12 --pv-pm-lettere --pv-pm-golosa \
  --densi-ternari tutte --righe-base-da 512 --pv-riuso "$@"
# Catena (subito dopo): systemd-run --user --unit=catena-tt5 bash -c "@@ROOT:forge@@/thor_v2/strumenti/catena_tt5.sh >> @@ROOT:depot@@/tutto_ternario/log/catena_tt5.log 2>&1"
