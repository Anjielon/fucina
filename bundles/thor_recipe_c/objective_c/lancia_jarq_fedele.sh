#!/bin/bash
# 11/10/2026 — Obiettivo C: sequenza FEDELE JARQ (a f_gptq1, b f_gptq1jarq, c f_bil1jarq) sui 250 densi, SOLO CPU.
# Guardie: free 'available' >= 8 GiB (attende, controllo ogni 60 s), @@ROOT:models@@ >= 5 GiB liberi. Registro: cicli.tsv
set -u
REG=@@ROOT:forge@@/stato/cicli.tsv
OUT=@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti_fedeli
riga() { printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> "$REG"; }
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= LEVE_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
libero=$(df -BG --output=avail @@ROOT:models@@ | tail -1 | tr -dc 0-9)
if [ "$libero" -lt 5 ]; then riga ERRORE "C JARQ-FEDELE: @@ROOT:models@@ ha solo ${libero} GiB liberi (servono >= 5), non parto"; exit 1; fi
attesa=0
while :; do
  av=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$av" -ge 8 ] && break
  [ $attesa -eq 0 ] && riga ATTIVO "C JARQ-FEDELE: in attesa di RAM (available ${av} GiB < 8), controllo ogni 60 s"
  attesa=1; sleep 60
done
riga ATTIVO "C JARQ-FEDELE: parte su CPU (6 fili, nice 10): a=f_gptq1 (GPTQ trits ufficiale) b=a+JARQ Alg.1 fedele c=BIL K=1+JARQ fedele; 250 densi da bf16 puro, H manifesto R2B3; uscita $OUT"
t0=$(date +%s)
nice -n 10 /usr/bin/python3 @@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice/jarq_fedele.py --out "$OUT" > "$OUT/corsa.log" 2>&1
rc=$?
min=$(( ($(date +%s) - t0) / 60 ))
if [ $rc -ne 0 ]; then riga ERRORE "C JARQ-FEDELE: uscita $rc dopo $min min, vedi $OUT/corsa.log"; exit $rc; fi
sint=$(/usr/bin/python3 - "$OUT/registro.jsonl" <<'PY'
import json, math, sys
R = [json.loads(l) for l in open(sys.argv[1])]
g = lambda k: math.exp(sum(math.log(r[k]) for r in R) / len(R))
print(f"{len(R)} tensori, errH medio geom.: TT2 {g('errH_tt2'):.4f} | gptq1 {g('errH_gptq1'):.4f} -> +JARQ {g('errH_gptq1jarq'):.4f} | BIL1 {g('errH_bil1'):.4f} -> +JARQ {g('errH_bil1jarq'):.4f}")
PY
)
riga CHIUSO "C JARQ-FEDELE: costruite su CPU in $min min: $sint. Prossimo: c0_scrivi.py su C0B per a/b/c (PTQ contro PTQ, stesso ospite), poi MMLU appaiato; verifica finale E2E da (a) contro E2E da (b), in coda dopo E2E-C1"
