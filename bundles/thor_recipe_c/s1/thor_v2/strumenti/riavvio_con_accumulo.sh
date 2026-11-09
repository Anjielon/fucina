#!/bin/bash
# 27/9: mette in servizio l'accumulo delle statistiche V (finestra 4 passi) sulla corsa EB2, senza intervento a mano.
# (1) aspetta i test di innesto (e2e_test_allena A3 A4 A5 A13 A14 A15) nel worktree; se non sono TUTTI verdi si ferma e lo scrive.
# (2) unisce il ramo pv-accumulo-wip in master. (3) aspetta la ripresa del passo 12. (4) ferma EB2 e la riavvia identica
# + --pv-passi-accumulo 4 e --ripresa-ogni 1 (le leve successive, es. CAKLD, entrano a fine di qualunque passo) (stessa matematica del passo P; il passo V decide ogni 4 passi con 4x le statistiche: T11 115 vs 5 salti giusti).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
W=/tmp/claude-1000/-home-angelo-MOGAVIS/12e4589a-507e-45a0-b690-66b786e2b26f/scratchpad/wt-accumulo
R=@@ROOT:forge@@; ST=$R/stato; E=@@ROOT:lavoro@@/e2e/allena/EB2
scrivi(){ echo "6xx | $(date +%H:%M) $*" >> $ST/linee.txt; printf "%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "$1" "${*:2}" >> $ST/cicli.tsv; echo "$*"; }
until grep -q "allena rc=" $W/esito_allena.log 2>/dev/null; do sleep 30; done
if ! grep -q "allena rc=0" $W/esito_allena.log || grep -q "❌" $W/esito_allena.log; then
  scrivi FERMO "accumulo PV: test di innesto NON tutti verdi, nessun riavvio: $(grep -E '❌|rc=' $W/esito_allena.log | tail -3 | tr '\n' ' ')"; exit 1
fi
cd $R && git merge --no-ff -q -m "thor_v2: accumulo delle statistiche V su N passi VERIFICATO (e2e_test_pv_codici T1-T12 verdi, T11 finestra 4: 115 salti giusti contro 5, precisione 1,0; T12 ripresa a meta' finestra al bit; e2e_test_allena A3 A4 A5 A13 A14 A15 verdi)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FEsCw8PTJjca1mUwu6jMzX" pv-accumulo-wip || { scrivi FERMO "accumulo PV: merge in master fallito"; git merge --abort; exit 1; }
grep -q "pv-passi-accumulo" $R/thor_v2/e2e_allena.py || { scrivi FERMO "accumulo PV: dopo il merge manca --pv-passi-accumulo"; exit 1; }
scrivi CHIUSO "accumulo PV unito in master ($(git -C $R log --oneline -1 | cut -c1-8)); aspetto la ripresa del passo 12 per riavviare EB2"
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$E/ripresa_ultima.json'))['passo']>=12 else 1)" 2>/dev/null; do sleep 20; done
sleep 30   # la scrittura della ripresa e' atomica; margine per il log del migliore
systemctl --user stop e2e-allena-EB2; sleep 15
systemctl --user reset-failed e2e-allena-EB2 2>/dev/null
systemd-run --user --unit=e2e-allena-EB2 -p Nice=5 --setenv=E2E_SENZA_COESISTENZA=1 --setenv=E2E_GDN=veloce --working-directory $R/thor_v2 \
  @@ROOT:venv@@/bin/python3 $R/thor_v2/e2e_allena.py allena --corsa EB2 --riprendi --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
  --ricetta B_ind32 --heldout-ogni 4 --rotazione --K0 524288 --pv-codici --pv-rho 0.03 --pv-passi-accumulo 4 --ripresa-ogni 1
sleep 900
scrivi ATTIVO "EB2 riavviata dal passo 12 con accumulo V a finestra 4 e GDN v3: $(grep -E 'ripresa|finestra|accumulo' $E/registro.log | tail -3 | cut -c1-160 | tr '\n' ' ')"
