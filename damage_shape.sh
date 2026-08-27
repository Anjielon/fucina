#!/bin/bash
# IL DANNO E' DIFFUSO O ESPLOSIVO?
#
# Una perplessita' di 37.80 contro 8.72 puo' nascere in due modi opposti:
#   (a) DIFFUSO — ogni blocco peggiora un po'. Il modello e' complessivamente
#       piu' incerto: danno vero, spalmato.
#   (b) ESPLOSIVO — quasi tutti i blocchi restano normali e uno o due vanno a
#       fuoco. La media e' dominata da pochi token, e la cura e' tutt'altra:
#       non "il modello e' peggiorato" ma "in certi punti si rompe".
#
# La distinzione conta anche per la non-additivita': se lo strato 20 da solo
# esplode su pochi blocchi e tutti gli strati insieme degradano in modo
# diffuso, allora non c'e' nessuna "cancellazione" — sono due fenomeni
# diversi che la media confonde in un numero solo.
#
# llama-perplexity stampa il valore progressivo blocco per blocco: basta
# tenerlo invece di buttarlo.
/home/angelo/odino-lab/odino/spazio_per.sh 12 gpu || exit 1
B=/home/angelo/build-llamacpp-tq1/build
M=/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf
W=/home/angelo/odino-lab/collaudo/corpus/wiki.test.raw
export LD_LIBRARY_PATH=$B/bin
D=/tmp/danno_forma; mkdir -p $D
for CFG in "rif:ODINO_NO_P2=1" "solo20:ODINO_P2_LAYERS=20-20" \
           "tutti:ODINO_P2_LAYERS=0-39" "coda:ODINO_P2_LAYERS=30-39"; do
  N=${CFG%%:*}; V=${CFG#*:}
  env $V $B/bin/llama-perplexity -m $M -ngl 999 -c 2048 -b 2048 \
      --chunks 12 -f $W > $D/$N.txt 2>&1
  echo "  $N fatto"
done
python3 - <<'PY'
import math
import re
from pathlib import Path
D = Path("/tmp/danno_forma")
print("\nPERPLESSITA' PROGRESSIVA, BLOCCO PER BLOCCO\n")
serie = {}
for n in ("rif", "solo20", "tutti", "coda"):
    t = (D / f"{n}.txt").read_text(errors="ignore")
    v = [float(x) for x in re.findall(r"\[\d+\]([0-9.]+)", t)]
    if v:
        serie[n] = v
if not serie:
    print("⛔ nessun valore progressivo trovato — formato diverso dall'atteso")
    raise SystemExit
n_max = max(len(v) for v in serie.values())
print("  blocco " + "".join(f"{k:>10s}" for k in serie))
for i in range(n_max):
    riga = f"  {i + 1:6d} "
    for k, v in serie.items():
        riga += f"{v[i]:10.3f}" if i < len(v) else " " * 10
    print(riga)

# il progressivo e' una media cumulativa in log: il contributo del blocco i
# si ricava dalla differenza fra due cumulate consecutive
print("\nCONTRIBUTO DEL SINGOLO BLOCCO (non cumulato)\n")
print("  blocco " + "".join(f"{k:>10s}" for k in serie))
sing = {}
for k, v in serie.items():
    s = []
    for i, x in enumerate(v):
        if i == 0:
            s.append(x)
        else:
            s.append(math.exp((i + 1) * math.log(x) - i * math.log(v[i - 1])))
    sing[k] = s
for i in range(n_max):
    riga = f"  {i + 1:6d} "
    for k in serie:
        riga += f"{sing[k][i]:10.2f}" if i < len(sing[k]) else " " * 10
    print(riga)

print("\nLETTURA")
if "rif" not in serie:
    print("  ⛔ manca il riferimento: nessun rapporto calcolabile")
    raise SystemExit
for k in serie:
    if k == "rif":
        continue
    n = min(len(sing[k]), len(sing["rif"]))
    r = sorted((sing[k][i] / sing["rif"][i] for i in range(n)), reverse=True)
    peggiore, mediano = r[0], r[len(r) // 2]
    forma = "ESPLOSIVO" if peggiore > 4 * mediano else "diffuso"
    print(f"  {k:8s}: blocco peggiore x{peggiore:6.2f} · mediano x{mediano:5.2f}"
          f"  => {forma}")
PY
echo "🏁 FORMA DEL DANNO COMPLETA"
