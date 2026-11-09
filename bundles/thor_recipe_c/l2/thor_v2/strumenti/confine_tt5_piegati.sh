#!/bin/bash
# 2/10 — CONFINE DI PASSO programmato per TT5 (richiesta del coordinatore, densi piegati d2958b5). Deterministico, senza
# presidio. Quando la ripresa del passo in corso e' scritta (ripresa_ultima.json passa a un passo > PARTENZA) ferma
# e2e-allena-TT5 (MainPID verificato), rifa' la sola FEDELTA' dei densi piegati (x2, gemm triton, B3_ind2) col nuovo
# pavimento di rumore (max di Bk2_ind32 e «riordina»), e riavvia TT5 SEMPRE:
#   - fedele                     -> git merge --ff-only densi-piegati in TT5-ottimizzato, riavvio con PIEGATI=x2 PIEGA_GEMM=triton
#   - non fedele / errore / tempo -> riavvio identico a ora (riprendi_tt5_ottimizzato.sh, RICETTA=B3_ind2)
#   - rumore «riordina» da solo > KL 0,0013 -> soglie NON toccate: riavvio senza piegati, riga DECISIONE PER ANGELO
# La velocita' dei piegati e' gia' misurata (cancello del 1/10 22:57: x0,856 / x0,858, indietro x0,746 / x0,734):
# FASI=fedelta non la rifa', quindi qui «ammesso» = «fedele».
# Tetto: TT5 di nuovo in corsa entro ~40 min dalla fermata (cancello oltre 32 min -> fermato e riavvio senza piegati).
# La catena giudice non si ferma: durante la pausa esiste @@ROOT:depot@@/e2e/catena_tt5o/pausa.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=@@ROOT:forge@@/stato; T=@@ROOT:forge@@/thor_v2/strumenti; D=@@ROOT:depot@@/e2e/allena/TT5
OTT=@@ROOT:depot@@/e2e/codice/TT5-ottimizzato; PIE=@@ROOT:depot@@/e2e/codice/TT5-piegati
U=@@ROOT:depot@@/e2e/prove_tt5/cancello_densi_piegati_B3ind2_v2; PAUSA=@@ROOT:depot@@/e2e/catena_tt5o/pausa
PARTENZA=${PARTENZA:-$(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['passo'])")}
TETTO_S=${TETTO_S:-1920}
riga(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $S/cicli.tsv; echo "[$(date '+%F %T')] $1 $2"; }
riavvia(){   # $1 = "piegati" | "" ; riprova una volta se il primo avvio viene rifiutato
  local env=(RICETTA=B3_ind2) n
  [ "$1" = piegati ] && env+=(PIEGATI=x2 PIEGA_GEMM=triton)
  for n in 1 2; do
    env "${env[@]}" bash $T/riprendi_tt5_ottimizzato.sh > $U/riavvio_$n.log 2>&1
    sleep 300
    if systemctl --user is-active -q e2e-allena-TT5; then
      rm -f $PAUSA
      riga ATTIVO "Confine TT5: riavviata ${1:+con i densi piegati x2 triton }dalla $(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['file'])") (attiva dopo 5 min, tentativo $n)"
      return 0
    fi
    riga ERRORE "Confine TT5: riavvio tentativo $n non attivo ($(tail -2 $U/riavvio_$n.log | tr '\n' ' ' | cut -c1-200); journal: $(journalctl --user -u e2e-allena-TT5 -n 3 --no-pager | tail -2 | tr '\n' ' ' | cut -c1-200))"
    [ "$1" = piegati ] && { env=(RICETTA=B3_ind2); riga ATTIVO "Confine TT5: ripiego al riavvio senza piegati"; }
  done
  rm -f $PAUSA; riga ERRORE "⛔ Confine TT5: TT5 NON ripartita: serve una persona"; exit 3
}
mkdir -p $U
riga ATTIVO "Confine TT5 ARMATO: alla prima ripresa dopo il passo $PARTENZA fermo TT5, fedelta' densi piegati v2 (x2 triton, B3_ind2, pavimento = max(Bk2, riordina)), poi riavvio (piegati solo se fedeli); tetto cancello ${TETTO_S} s"
# 1) attesa del confine (nessuna interruzione a meta' passo)
until [ "$(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['passo'])" 2>/dev/null || echo 0)" -gt "$PARTENZA" ]; do
  systemctl --user is-active -q e2e-allena-TT5 || [ "$(systemctl --user show -p ActiveState --value e2e-allena-TT5)" = activating ] \
    || { sleep 600; systemctl --user is-active -q e2e-allena-TT5 || { riga ERRORE "Confine TT5: TT5 non attiva prima del confine: niente fermata, esco"; exit 2; }; }
  sleep 60
done
P=$(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['passo'])")
# 2) fermata (unita' per nome, MainPID letto un istante prima e verificato sparito)
touch $PAUSA
PID=$(systemctl --user show -p MainPID --value e2e-allena-TT5)
T0=$(date +%s)
systemctl --user stop e2e-allena-TT5
for i in $(seq 1 60); do [ "$PID" = 0 ] || ! kill -0 "$PID" 2>/dev/null && break; sleep 5; done
if [ "$PID" != 0 ] && kill -0 "$PID" 2>/dev/null; then riga ERRORE "Confine TT5: MainPID $PID ancora vivo dopo lo stop: niente cancello, riavvio"; riavvia ""; exit 0; fi
riga ATTIVO "Confine TT5: ripresa del passo $P scritta, e2e-allena-TT5 fermata (MainPID $PID uscito). Cancello fedelta' densi piegati v2"
# 3) cancello (python diretto: lo script .sh rifiuta con le unita' catena-* attive, che devono restare vive)
cd $PIE/thor_v2
env -u TRITON_INTERPRET -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES systemd-run --user --unit=cancello-densi-piegati-v2 \
  --collect -p Nice=5 -p RuntimeMaxSec=$TETTO_S --setenv=E2E_GDN=veloce --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 \
  --setenv=E2E_MOTORE=triton2 --setenv=E2E_SENZA_COESISTENZA=1 --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OMP_NUM_THREADS=4 \
  --setenv=E2E_CACHE_DEQUANT=@@ROOT:depot@@/e2e/cache_dequant --working-directory $PIE/thor_v2 \
  bash -c "@@ROOT:venv@@/bin/python3 e2e_cancello_piegati.py --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno --uscita $U --ricetta B3_ind2 --modi x2 --gemm triton --strati-tempo 0,1 --strati-tempo 38,39 --strati-kl 0-39 --salta-banco --salta-tempo > $U/cancello.log 2>&1"
sleep 5
while systemctl --user is-active -q cancello-densi-piegati-v2; do sleep 20; done
E=$U/esito_cancello_piegati.json
read -r FED RIORD KLX < <(python3 - "$E" <<'PYX' 2>/dev/null || echo "none none none"
import json, sys
d = json.load(open(sys.argv[1]))
f = d.get("fedelta", {}).get("x2", {})
r = d.get("rumore_densi", {}).get("kl", {}).get("media")
print(f.get("fedele"), r, f.get("kl_B_B3", {}).get("media"))
PYX
)
riga ATTIVO "Confine TT5: cancello piegati v2 in $(( $(date +%s) - T0 )) s dalla fermata: fedele=$FED, KL piegato $KLX, KL rumore riordina $RIORD ($(grep -E "fedelta' piegato|rumore densi" $U/cancello.log | tr '\n' ' ' | cut -c1-300))"
if [ "$RIORD" != None ] && [ "$RIORD" != none ] && awk -v r="$RIORD" 'BEGIN{exit !(r>0.0013)}'; then
  riga ATTIVO "DECISIONE PER ANGELO: il rumore «riordina» da solo ha KL $RIORD > 0,0013: soglie NON cambiate, densi piegati NON applicati"
  riavvia ""; exit 0
fi
if [ "$FED" = True ]; then
  if git -C $OTT merge --ff-only densi-piegati > $U/merge.log 2>&1; then
    riga ATTIVO "Confine TT5: densi piegati FEDELI -> merge --ff-only densi-piegati in tt5-ottimizzato ($(git -C $OTT log --oneline -1 | cut -c1-80))"
    riavvia piegati; exit 0
  fi
  riga ERRORE "Confine TT5: merge --ff-only fallito ($(tail -2 $U/merge.log | tr '\n' ' ')): riavvio senza piegati"
fi
riavvia ""
