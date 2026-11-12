#!/bin/bash
# 11/10 10:30 (recupero scadenza, 122 min dovuti): le 9 leve generiche di B (fase B di combinazione_eb882K.sh, ~1,5 h di GPU
# del Corsair) calcolate sul DELL (RTX 4060, CUDA) mentre la GPU del Corsair misura CARDAN fedele e A3 e fa i gradienti.
# Prova di fedelta' 10:24 (jarq_p2k, 2 tensori): errH Dell = Corsair a 1e-11, codici diversi su ~1e-5 dei byte (arrotondamenti).
# Rientro nel formato di applica.py (registro.jsonl + tensori), con riga "macchina": "dell-cuda": la fase B sul Corsair salta i fatti.
# Fermata: file stato/FERMA_DELL_LEVE (combinazione_eb882K.sh lo crea prima della sua fase B).
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=/home/angelo/tony-forgia/stato; TT=/mnt/models/gguf/tony-tern; C=$TT/COMBINAZIONE_eb882K; Q=/mnt/lavoro/pacco_dell
PY=/home/angelo/venv-catq/bin/python3; DPY='C:\Users\angel\AppData\Local\Programs\Python\Python310\python.exe'
LEVE="jarq_pk jarq_p2k schur_pk schur_p2k tjarq_p1fermo bil_foem tied_jarq bil_tjarq bil_tjarq_p1fermo"
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= LEVE_OSPITE=$TT/TUTTE/eb882K.gguf LEVE_DT=float32 LEVE_THREADS=6
riga(){ local t; t=$(printf '%s' "$2" | tr '\t\n' '  '); printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$t" >> $S/cicli.tsv; echo "[$(date '+%F %T')] $1 $t"; }
exec >> $Q/corsa.log 2>&1
rm -f $S/FERMA_DELL_LEVE; T0=$(date +%s)
riga ATTIVO "DELL-LEVE-EB882K: le 9 leve di B sul Dell (CUDA) in parallelo alla GPU del Corsair; rientro nel registro di combinazione_eb882K"
# 1 pacchi: impacchetta in fondo e sposta sul Dell man mano (disco /mnt/lavoro stretto)
nice -n 10 $PY $Q/codice/impacchetta.py $Q/pacchi > $Q/impacchetta.log 2>&1 &
PK=$!
sposta(){ for f in $(ls $Q/pacchi/*.npz 2>/dev/null | grep -v '\.tmp\.npz$'); do
  scp -q "$f" aragorn:C:/pacco_dell/pacchi/ && rm -f "$f"; done; }
while kill -0 $PK 2>/dev/null; do sposta; sleep 10; done; sposta
grep -q FINE $Q/impacchetta.log || { riga ERRORE "DELL-LEVE-EB882K: impacchettamento fallito ($(tail -2 $Q/impacchetta.log | tr '\n' ' ' | cut -c1-200))"; exit 2; }
N=$(ssh aragorn "dir /b C:\pacco_dell\pacchi\*.npz" 2>/dev/null | tr -d '\r' | grep -c 'weight.npz$')
[ "$N" = 250 ] || { riga ERRORE "DELL-LEVE-EB882K: sul Dell $N pacchi su 250"; exit 2; }
riga ATTIVO "DELL-LEVE-EB882K: 250 pacchi sul Dell in $(( ($(date +%s) - T0) / 60 )) min; parte il calcolo"
# 2 calcolo sul Dell (in fondo) + rientro periodico
ssh aragorn "cd C:\pacco_dell && set LEVE_DT=float32&& set LEVE_THREADS=8&& set LEVE_FOEM_FORMA=codice&& $DPY codice\dell_leve.py pacchi uscita cuda $(echo $LEVE | tr ' ' ',')" > $Q/dell.log 2>&1 &
DS=$!
rientra(){ for L in $LEVE; do
    mkdir -p $Q/uscita/$L
    for n in $(ssh aragorn "dir /b C:\pacco_dell\uscita\\$L\*.npz" 2>/dev/null | tr -d '\r' | grep 'weight.npz$'); do
      [ -s $Q/uscita/$L/$n ] || scp -q aragorn:C:/pacco_dell/uscita/$L/$n $Q/uscita/$L/ ; done
    ls $Q/uscita/$L/*.npz >/dev/null 2>&1 && nice -n 10 $PY $Q/codice/rientra_dell.py $Q/pacchi $Q/uscita/$L $C/$L > /dev/null 2>>$Q/rientro.err
  done; }
while kill -0 $DS 2>/dev/null; do
  if [ -f $S/FERMA_DELL_LEVE ]; then for L in $LEVE; do ssh aragorn "type nul > C:\pacco_dell\uscita\FERMA_$L" 2>/dev/null; done; fi
  sleep 240; rientra; done
rientra
FATTE=""; for L in $LEVE; do FATTE="$FATTE $L $(wc -l < $C/$L/registro.jsonl 2>/dev/null || echo 0)"; done
riga ATTIVO "DELL-LEVE-EB882K: fine in $(( ($(date +%s) - T0) / 60 )) min; tensori nel registro:$FATTE (su 250; la fase B del Corsair fa il resto)"
