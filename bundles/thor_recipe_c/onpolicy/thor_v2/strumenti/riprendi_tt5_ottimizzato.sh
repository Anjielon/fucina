#!/bin/bash
# 1/10 notte — TT5 RIPRESA da ripresa_000002 col codice TT5-ottimizzato (ramo tt5-ottimizzato, vedi
# thor_v2/ottimizzazioni_tt5/PRONTO_RIAVVIO.md). Decisioni di Angelo (1/10 22:1x): tutte le leve verificate insieme.
#  V-Bop ogni 4 passi   --pv-passi-accumulo 4 (con --pv-modo vbop: statistiche + EMA a ogni passo, decisione ogni 4)  DECISIONE
#  indietro diviso      RICETTA=B3_ind2 --cambia-ricetta SOLO se il cancello GPU (cancello_indietro.sh, config TT5) l'ha ammessa
#  valutazioni          --heldout-ogni 2 (pieno resta ogni 8)                                           ESECUZIONE
#  cache dequant        E2E_CACHE_DEQUANT (embedding/testa al bit dalla cache)                          ESECUZIONE
#  scritture asincrone  --scritture-asincrone (ripresa + V-Bop da un filo, file al bit)                 ESECUZIONE
#  tempi per fase       --tempi-fasi                                                                    ESECUZIONE
#  istantanee           --istantanee-tieni 3 (la catena giudice le raccoglie ogni 2 passi)              ESECUZIONE
# Il resto e' IDENTICO a riprendi_tt5_cure.sh (stessa riga di TT5 dal 21:25).
#  densi piegati        PIEGATI=x2|f32 (E2E_DENSI_PIEGATI, ramo densi-piegati) SOLO se il suo cancello l'ha ammesso
# Uso: RICETTA=B3_ind2 [PIEGATI=x2 PIEGA_GEMM=auto] bash riprendi_tt5_ottimizzato.sh   (default B3_ind32)
#  2/10 CANCELLO_V=<json> -> E2E_CANCELLO_V (cancello GPU delle cure della fase V, FASE_V.md); argomenti in piu' dopo --riprendi
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
COD=@@ROOT:depot@@/e2e/codice/TT5-ottimizzato; PY=@@ROOT:venv@@/bin/python3
RICETTA=${RICETTA:-B3_ind32}
G=@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf; SHA=56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43
MC=@@ROOT:depot@@/e2e_teacher_mc; CACHE=@@ROOT:depot@@/e2e/cache_dequant
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm
[ "$(git -C $COD rev-parse --abbrev-ref HEAD)" = tt5-ottimizzato ] || { echo "codice $COD non sul ramo tt5-ottimizzato"; exit 2; }
[ -z "$(git -C $COD status --porcelain -- thor_v2/*.py)" ] || { echo "codice $COD con modifiche non committate"; exit 2; }
[ "$(sha256sum "$G" | cut -d' ' -f1)" = "$SHA" ] || { echo "sha di $G diverso"; exit 2; }
# 2/10: dall'ULTIMA ripresa scritta (ripresa_ultima.json), non piu' fissa al passo 2 (confine di passo programmato)
R=@@ROOT:depot@@/e2e/allena/TT5/$(python3 -c "import json;print(json.load(open('@@ROOT:depot@@/e2e/allena/TT5/ripresa_ultima.json'))['file'])")
[ "$(sha256sum "$R" | cut -d' ' -f1)" = "$(python3 -c "import json;print(json.load(open('@@ROOT:depot@@/e2e/allena/TT5/ripresa_ultima.json'))['sha256'])")" ] || { echo "$R: sha diverso dal registro"; exit 2; }
for u in $(systemctl --user list-units --state=active --no-legend 'e2e-allena-*' 'catena-R2B4*' 'cancello-indietro-*' | awk '{print $1}'); do echo "GPU occupata: $u"; exit 2; done
LIBERI=$(df -B1G --output=avail @@ROOT:depot@@ | tail -1 | tr -d ' '); [ "$LIBERI" -ge 30 ] || { echo "@@ROOT:depot@@: $LIBERI GiB"; exit 2; }
python3 @@ROOT:forge@@/thor_v2/strumenti/verifica_righe_teacher.py @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno 512 || exit 2
EXTRA=()
if [ "$RICETTA" != B3_ind32 ]; then
  E=@@ROOT:depot@@/e2e/prove_tt5/cancello_indietro_TT5/esito_cancello_indietro.json
  python3 -c "import json,sys; v=json.load(open('$E'))['verdetto']; sys.exit(0 if v.get('$RICETTA',{}).get('ammessa') else 1)" \
    || { echo "$RICETTA non ammessa dal cancello ($E)"; exit 2; }
  EXTRA+=(--cambia-ricetta)
fi
D=@@ROOT:depot@@/e2e/allena/TT5
[ -f $D/codice_prima_del_1ott.txt ] || cp $D/codice.txt $D/codice_prima_del_1ott.txt
echo "$COD" > $D/codice.txt
systemd-run --user --unit=e2e-allena-TT5 -p Nice=5 -p RestartForceExitStatus=75 -p RestartSec=180 -p StartLimitBurst=3 -p StartLimitIntervalSec=21600 \
  --setenv=E2E_ALLENA_DIR=@@ROOT:depot@@/e2e/allena --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce \
  --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --setenv=E2E_MOTORE=triton2 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=E2E_CACHE_DEQUANT=$CACHE ${PIEGATI:+--setenv=E2E_DENSI_PIEGATI=$PIEGATI} ${PIEGA_GEMM:+--setenv=E2E_PIEGA_GEMM=$PIEGA_GEMM} \
  ${CANCELLO_V:+--setenv=E2E_CANCELLO_V=$CANCELLO_V} \
  --working-directory "$COD/thor_v2" "$PY" "$COD/thor_v2/e2e_allena.py" allena --corsa TT5 --gguf "$G" --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta "$RICETTA" --testa-kl bf16x3 --logit-maestro bf16 --heldout-ogni 2 --heldout-righe-rapide 16 --heldout-pieno-ogni 8 --rotazione \
  --non-consumare --K0 524288 --pv-codici --pv-rho 0.03 --pv-passi-accumulo 4 --pv-modo vbop --pv-rho-adattivo \
  --perdita cakld --gamma 0.552 --allena-f32 "$F32" --lr-f32 5e-5 --ema-theta 0.9 --ripresa-ogni 1 \
  --ottimizzatore-theta adamw --pv-accetta guadagno --pv-z-guadagno-min 1 \
  --istantanea-ogni-passo --istantanee-dir @@ROOT:depot@@/e2e_istantanee/TT5 --istantanee-tieni 3 --istantanee-libero-min-gib 12 \
  --teacher-mc $MC/q8-mc --teacher-mc-heldout $MC/q8-mc-heldout --righe-mc 8 --righe-mc-permutate --peso-token risposta \
  --beta-risposta 0.1 --gamma-lettere 1 \
  --peso-dominio it_chat=1,strumenti=1,cot=1,edu=1.5,it_web=1.5,it_wiki=1.5,it_pensiero=1.5 \
  --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 --pv-ricentra --pv-ricentra-giri 3 \
  --pv-punto-medio --pv-pm-primitiva famiglia --pv-pm-max 12 --pv-pm-lettere --pv-pm-golosa \
  --densi-ternari tutte --righe-base-da 512 --pv-riuso --tempi-fasi --scritture-asincrone "${EXTRA[@]}" --riprendi "$@"
