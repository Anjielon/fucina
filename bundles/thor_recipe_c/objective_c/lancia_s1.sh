#!/bin/bash
# 11/10/2026 — Obiettivo C, RAMO B: S1 fedele (OSAQ 2605.04738 -> AWQ 2306.00978 [+ L2 2609.21450 su down] -> segni
# 2609.21450, filtro CPU) sui 250 densi + 60 alpha/beta, poi S2 a/b/c sopra S1; poi la leva 3 isolata (solo OSAQ).
# SOLO CPU, 6 fili, nice 10. Guardie: available >= 8 GiB (attende), @@ROOT:models@@ >= 26 GiB liberi (24 restano alla coda
# GPU, uscita totale < 2 GB). Registro: cicli.tsv, nome «C S1».
set -u
REG=@@ROOT:forge@@/stato/cicli.tsv
COD=@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice
OUT=@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti_s1
OUT2=@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti_s1_osaq
riga() { printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> "$REG"; }
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= LEVE_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
libero=$(df -BG --output=avail @@ROOT:models@@ | tail -1 | tr -dc 0-9)
if [ "$libero" -lt 26 ]; then riga ERRORE "C S1: @@ROOT:models@@ ha ${libero} GiB liberi (servono >= 26: 24 alla coda GPU), non parto"; exit 1; fi
attesa=0
while :; do
  av=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$av" -ge 8 ] && break
  [ $attesa -eq 0 ] && riga ATTIVO "C S1: in attesa di RAM (available ${av} GiB < 8), controllo ogni 60 s"
  attesa=1; sleep 60
done
mkdir -p "$OUT" "$OUT2"
riga ATTIVO "C S1: parte su CPU (6 fili, nice 10). Fase 1: S1 fedele (OSAQ eq.8-14 -> AWQ griglia 20 RTN del formato su attn_norm e up->down [+L2 2609] -> segni 2609 filtro CPU) + S2 a/b/c sui 250 densi + 60 alpha/beta da bf16 puro, H' = D^-1 H D^-1; uscita $OUT. Fase 2: solo OSAQ (leva 3 isolata) in $OUT2"
t0=$(date +%s)
nice -n 10 /usr/bin/python3 $COD/s1_fedele.py --out "$OUT" > "$OUT/corsa.log" 2>&1
rc=$?
min=$(( ($(date +%s) - t0) / 60 ))
if [ $rc -ne 0 ]; then riga ERRORE "C S1: fase 1 uscita $rc dopo $min min, vedi $OUT/corsa.log"; exit $rc; fi
s1=$(/usr/bin/python3 $COD/s1_sintesi.py "$OUT" --json "$OUT/sintesi.json")
riga ATTIVO "C S1: fase 1 chiusa in $min min: $s1. Parte la fase 2 (solo OSAQ)"
t1=$(date +%s)
nice -n 10 /usr/bin/python3 $COD/s1_fedele.py --out "$OUT2" --solo-osaq > "$OUT2/corsa.log" 2>&1
rc=$?
min2=$(( ($(date +%s) - t1) / 60 ))
if [ $rc -ne 0 ]; then riga ERRORE "C S1: fase 2 (solo OSAQ) uscita $rc dopo $min2 min, vedi $OUT2/corsa.log"; exit $rc; fi
s2=$(/usr/bin/python3 $COD/s1_sintesi.py "$OUT2" --json "$OUT2/sintesi.json")
riga CHIUSO "C S1: S1 completo ($min min) $s1 || solo OSAQ ($min2 min) $s2. GGUF: c0t_scrivi.py --base C0B --tensori varianti_s1/norme varianti_s1/<v>/tensori (norme F32 nel manifesto). Nota: ricerche/S1_C_1011.md"
