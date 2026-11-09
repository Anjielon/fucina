#!/bin/bash
# 2/10 — CONFINE DI PASSO per TT5: cure della fase V + scritture in sosta (decisione del coordinatore 2/10 ~06:00;
# ~/tony-forgia/thor_v2/ottimizzazioni_tt5/FASE_V.md). Modello: confine_tt5_piegati.sh. Deterministico, senza presidio.
#  1. attesa: ripresa_ultima.json passa a un passo > PARTENZA (nessuna interruzione a meta' passo)
#  2. fermata di e2e-allena-TT5 (MainPID verificato sparito), pausa della catena giudice
#  3. prova GPU AL BIT della sosta (e2e_prova_sosta.py cuda, ~1 min, < 2 GB VRAM) dal ramo tt5-fase-v
#  4. git merge --ff-only tt5-fase-v in TT5-ottimizzato (le cure V1-V3 spente = codice di prima al bit, prova F0)
#  5. riavvio con riprendi_tt5_ottimizzato.sh (RICETTA=B3_ind2 PIEGATI=x2 PIEGA_GEMM=triton, come ora) +
#     CANCELLO_V=<json> (cancello GPU in modo corsa al prossimo passo V di decisione) + --pv-tempi-v
#     + --scritture-sosta-dispositivo SOLO se la prova del punto 3 e' verde
#  Ripieghi: merge fallito -> riavvio identico a ora; riavvio non attivo -> riavvio senza le opzioni nuove (stesso
#  codice); ancora non attivo -> branch riportato al commit di prima (reset --keep) e riavvio identico a ora.
#  NON accende nulla che cambi le decisioni (curvatura ogni 4, pm-max 6, curvatura fp16: decisioni di Angelo).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=@@ROOT:forge@@/stato; T=@@ROOT:forge@@/thor_v2/strumenti; D=@@ROOT:depot@@/e2e/allena/TT5
OTT=@@ROOT:depot@@/e2e/codice/TT5-ottimizzato; FV=@@ROOT:depot@@/e2e/codice/TT5-V; RAMO=tt5-fase-v
U=@@ROOT:depot@@/e2e/prove_fase_v/confine; PAUSA=@@ROOT:depot@@/e2e/catena_tt5o/pausa
PY=@@ROOT:venv@@/bin/python3
PARTENZA=${PARTENZA:-$(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['passo'])")}
JS=$D/cancello_fase_v.json
riga(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $S/cicli.tsv; echo "[$(date '+%F %T')] $1 $2"; }
attiva_dopo(){ sleep 300; systemctl --user is-active -q e2e-allena-TT5; }
lancia(){   # $1 = etichetta; il resto: argomenti in piu' per riprendi; env CANCELLO_V gia' impostato o no
  local et=$1; shift
  env RICETTA=B3_ind2 PIEGATI=x2 PIEGA_GEMM=triton ${CV:+CANCELLO_V=$CV} bash $T/riprendi_tt5_ottimizzato.sh "$@" \
    > $U/riavvio_$et.log 2>&1
  attiva_dopo
}
mkdir -p $U
riga ATTIVO "Confine TT5 fase V ARMATO: alla prima ripresa dopo il passo $PARTENZA fermo TT5, prova GPU della sosta, merge --ff-only $RAMO, riavvio con cancello V (corsa) + --pv-tempi-v (+ sosta se verde)"
# 1) attesa del confine (TT5 puo' anche riavviarsi da sola nel frattempo: si aspetta)
until [ "$(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['passo'])" 2>/dev/null || echo 0)" -gt "$PARTENZA" ]; do
  st=$(systemctl --user show -p ActiveState --value e2e-allena-TT5)
  if [ "$st" != active ] && [ "$st" != activating ]; then
    sleep 600; st=$(systemctl --user show -p ActiveState --value e2e-allena-TT5)
    [ "$st" = active ] || [ "$st" = activating ] || { riga ERRORE "Confine TT5 fase V: TT5 non attiva ($st) prima del confine: niente fermata, esco"; exit 2; }
  fi
  sleep 60
done
P=$(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['passo'])")
# 2) fermata
touch $PAUSA
PID=$(systemctl --user show -p MainPID --value e2e-allena-TT5)
systemctl --user stop e2e-allena-TT5
for i in $(seq 1 60); do { [ "$PID" = 0 ] || ! kill -0 "$PID" 2>/dev/null; } && break; sleep 5; done
if [ "$PID" != 0 ] && kill -0 "$PID" 2>/dev/null; then
  riga ERRORE "Confine TT5 fase V: MainPID $PID ancora vivo dopo lo stop: niente modifiche, riavvio identico"
  CV= lancia identico; rm -f $PAUSA; exit 0
fi
riga ATTIVO "Confine TT5 fase V: ripresa del passo $P scritta, e2e-allena-TT5 fermata (MainPID $PID uscito)"
# 3) prova GPU della sosta (dal ramo, non dal codice della corsa)
SOSTA=()
if systemd-run --user --unit=prova-sosta-gpu --collect --wait -p RuntimeMaxSec=600 -p Nice=5 \
     --setenv=E2E_SOSTA_DIR=$U --working-directory $FV/thor_v2 \
     bash -c "$PY e2e_prova_sosta.py cuda > $U/prova_sosta_gpu.json 2> $U/prova_sosta_gpu.err" >/dev/null 2>&1 \
   && python3 -c "import json,sys; sys.exit(0 if json.load(open('$U/prova_sosta_gpu.json'))['ok'] else 1)" 2>/dev/null; then
  SOSTA=(--scritture-sosta-dispositivo)
  riga ATTIVO "Confine TT5 fase V: prova GPU della sosta VERDE (file al bit, etichette cpu; picco RSS $(python3 -c "import json;v=json.load(open('$U/prova_sosta_gpu.json'))['vie'];print(round(v['sosta']['picco_rss_mib_in_piu']),'MiB contro',round(v['cpu_oggi']['picco_rss_mib_in_piu']),'MiB')" 2>/dev/null)) -> --scritture-sosta-dispositivo"
else
  riga ATTIVO "Confine TT5 fase V: prova GPU della sosta NON verde ($(tail -c 300 $U/prova_sosta_gpu.err 2>/dev/null | tr '\n' ' ')): sosta NON accesa"
fi
# 4) merge (fast-forward soltanto)
VECCHIO=$(git -C $OTT rev-parse HEAD)
if [ -n "$(git -C $OTT status --porcelain -- thor_v2/*.py)" ] || ! git -C $OTT merge --ff-only $RAMO > $U/merge.log 2>&1; then
  riga ERRORE "Confine TT5 fase V: merge --ff-only $RAMO NON fatto ($(tail -2 $U/merge.log 2>/dev/null | tr '\n' ' ')): riavvio identico a ora"
  CV= lancia identico && { rm -f $PAUSA; riga ATTIVO "Confine TT5 fase V: riavviata identica"; exit 0; }
  rm -f $PAUSA; riga ERRORE "⛔ Confine TT5 fase V: TT5 NON ripartita: serve una persona"; exit 3
fi
riga ATTIVO "Confine TT5 fase V: merge --ff-only $RAMO in tt5-ottimizzato ($VECCHIO -> $(git -C $OTT log --oneline -1 | cut -c1-70))"
# 5) riavvio
rm -f "$JS"
if CV=$JS lancia cure --pv-tempi-v "${SOSTA[@]}"; then
  rm -f $PAUSA
  riga ATTIVO "Confine TT5 fase V: TT5 riavviata dalla $(python3 -c "import json;print(json.load(open('$D/ripresa_ultima.json'))['file'])") col codice $(git -C $OTT rev-parse --short HEAD), cancello V in corsa (esito $JS al prossimo passo V di decisione), --pv-tempi-v ${SOSTA[*]}"
  exit 0
fi
riga ERRORE "Confine TT5 fase V: riavvio con le opzioni nuove NON attivo ($(tail -2 $U/riavvio_cure.log | tr '\n' ' ' | cut -c1-200); journal: $(journalctl --user -u e2e-allena-TT5 -n 3 --no-pager | tail -2 | tr '\n' ' ' | cut -c1-200)): riavvio senza opzioni nuove"
systemctl --user stop e2e-allena-TT5 2>/dev/null; systemctl --user reset-failed e2e-allena-TT5 2>/dev/null
if CV= lancia senza_opzioni; then rm -f $PAUSA; riga ATTIVO "Confine TT5 fase V: riavviata col codice nuovo SENZA opzioni nuove (= codice di prima al bit)"; exit 0; fi
systemctl --user stop e2e-allena-TT5 2>/dev/null; systemctl --user reset-failed e2e-allena-TT5 2>/dev/null
git -C $OTT reset --keep "$VECCHIO" > $U/ritorno.log 2>&1
riga ERRORE "Confine TT5 fase V: anche senza opzioni non parte: tt5-ottimizzato riportato a $VECCHIO ($(tail -1 $U/ritorno.log)), riavvio identico"
if CV= lancia ritorno; then rm -f $PAUSA; riga ATTIVO "Confine TT5 fase V: riavviata col codice di prima"; exit 0; fi
rm -f $PAUSA; riga ERRORE "⛔ Confine TT5 fase V: TT5 NON ripartita: serve una persona"; exit 3
