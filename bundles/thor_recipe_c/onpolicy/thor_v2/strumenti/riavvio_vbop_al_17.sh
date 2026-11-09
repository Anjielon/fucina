#!/bin/bash
# 27/9 (ramo vbop2): accende V-Bop su EB2 al passo 17 SENZA perdere statistiche. NON ARMATO: si lancia a mano, dopo il
# merge di vbop2 in master (riavvio_ora.sh esegue il codice di @@ROOT:forge@@).
# Perche' al 17: la finestra di accumulo aperta alla ripresa del 14 (passi 14, 15, 16) si CHIUDE al passo V 16
# ((t+1) % 4 == 0): la ripresa_000016 non ha finestra aperta (accumulo None), quindi il passaggio a V-Bop non scarta
# nulla. Da una ripresa con finestra aperta le statistiche accumulate si SCARTANO (prova A30/A31: mai lette, V-Bop parte
# vuoto); si perde il segnale V di quei passi, non il passo P (theta, Adam, F32 restano).
# Aspetta ripresa_ultima.json al passo >= 16, ferma e riavvia con V-Bop (argparse: l'ultimo --pv-passi-accumulo vince).
set -u
E=@@ROOT:lavoro@@/e2e/allena/EB2; S=@@ROOT:forge@@/thor_v2/strumenti
grep -q "pv_modo" @@ROOT:forge@@/thor_v2/e2e_allena.py || { echo "master senza V-Bop: prima il merge di vbop2"; exit 1; }
while :; do
  P=$(python3 -c "import json; print(json.load(open('$E/ripresa_ultima.json'))['passo'])" 2>/dev/null || echo 0)
  [ "$P" -ge 16 ] && break
  sleep 30
done
if grep -q "passo V $P: salti" "$E/registro.log"; then
  echo "$(date -Is) ripresa $P: finestra chiusa, nessuna statistica scartata"
else
  echo "$(date -Is) ripresa $P: finestra APERTA, le sue statistiche V si scartano (V-Bop parte vuoto)"
fi
exec "$S/riavvio_ora.sh" --pv-modo vbop --pv-passi-accumulo 1 "$@"
