#!/bin/bash
# IS THE DAMAGE DIFFUSE OR EXPLOSIVE?
#
# A perplexity of 37.80 against 8.72 can arise in two opposite ways:
#   (a) DIFFUSE — every chunk gets a little worse. The model is uniformly
#       more uncertain: real damage, spread out.
#   (b) EXPLOSIVE — nearly every chunk stays normal and one or two catch
#       fire. The mean is dominated by a handful of tokens, and the remedy is
#       entirely different: not "the model got worse" but "it breaks in
#       specific places".
#
# The distinction also matters for non-additivity: if layer 20 alone explodes
# on a few chunks while all layers together degrade diffusely, then there is
# no "cancellation" — they are two different phenomena that the mean collapses
# into a single number.
#
# llama-perplexity prints the running value chunk by chunk: it is enough
# tenerlo invece di buttarlo.
$HOME/odino-lab/odino/spazio_per.sh 12 gpu || exit 1
B=$HOME/build-llamacpp-tq1/build
M=/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf
W=$HOME/odino-lab/collaudo/corpus/wiki.test.raw
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
