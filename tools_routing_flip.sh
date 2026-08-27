#!/bin/bash
# QUANTE SCELTE DI ESPERTO CAMBIANO col secondo piano acceso? — versione corretta
#
# ⛔ La prima versione non misurava NULLA: catturava `ffn_moe_topk`, che e' di
#    INTERI, e il dump scrive solo F32 (common/debug.cpp). Col piano acceso
#    finiva nel filtro un nodo di conversione che il mio stesso codice crea e
#    che col piano spento NON ESISTE → "0 su 0". Un confronto contro il nulla
#    che si presentava come un risultato.
#
# Correzione: si cattura `ffn_moe_probs` — le probabilita' del router, F32,
# presenti in ENTRAMBE le configurazioni — e la scelta si ricalcola in numpy.
# In piu' si ottiene il MARGINE: quanto sono al pelo le scelte che cambiano.
/home/angelo/odino-lab/odino/spazio_per.sh 12 gpu || exit 1
B=/home/angelo/build-llamacpp-tq1/build
M=/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf
F=$(ls /home/angelo/odino-lab/guarigione/cal_blocchi/b*.txt | head -1)
export LD_LIBRARY_PATH=$B/bin
for CFG in spento:ODINO_NO_P2=1 acceso:IGNORA=1; do
  N=${CFG%%:*}; V=${CFG#*:}
  D=/tmp/instrad2_$N; rm -rf $D; mkdir -p $D
  env $V MOGAVIS_DUMP_DIR=$D MOGAVIS_DUMP_FILTER=ffn_moe_probs \
    $B/bin/llama-eval-callback -m $M -ngl 999 -c 2048 -n 1 -f $F > /dev/null 2>&1
  echo "  $N: $(ls $D/*.f32 2>/dev/null | wc -l) file salvati"
done
# ⛔ guardia: se una delle due gambe e' vuota, NON stampare percentuali.
A=$(ls /tmp/instrad2_spento/*.f32 2>/dev/null | wc -l)
C=$(ls /tmp/instrad2_acceso/*.f32 2>/dev/null | wc -l)
if [ "$A" -eq 0 ] || [ "$C" -eq 0 ]; then
  echo "⛔ una delle due catture e' VUOTA ($A / $C) — nessun confronto possibile"
  exit 1
fi
python3 - <<'PY'
import numpy as np, glob, re, os
N_USED = 8
print("\nQUANTE SCELTE DI ESPERTO CAMBIANO?\n")
righe = []
for f in sorted(glob.glob("/tmp/instrad2_spento/*.f32")):
    g = f.replace("_spento", "_acceso")
    if not os.path.exists(g):
        continue
    m = re.search(r"-(\d+)\.f32", f)
    if not m:
        continue
    L = int(m.group(1))
    a = np.fromfile(f, dtype=np.float32)
    b = np.fromfile(g, dtype=np.float32)
    n = min(a.size, b.size)
    if n == 0:
        continue
    a, b = a[:n], b[:n]
    # la forma e' (token, esperti): il numero di esperti si deduce dal file
    # gemello piu' corto solo se e' multiplo — altrimenti si salta lo strato.
    for E in (256, 128, 512, 64):
        if n % E == 0:
            break
    else:
        continue
    A = a.reshape(-1, E); B = b.reshape(-1, E)
    sa = np.argsort(-A, axis=1)[:, :N_USED]
    sb = np.argsort(-B, axis=1)[:, :N_USED]
    # insiemi diversi, non slot diversi: un riordino dentro gli 8 non conta
    diversi = sum(1 for i in range(sa.shape[0])
                  if set(sa[i].tolist()) != set(sb[i].tolist()))
    # margine: distanza fra l'8° e il 9° in probabilita' (piano spento)
    ord_a = np.sort(A, axis=1)[:, ::-1]
    margine = float(np.median(ord_a[:, N_USED - 1] - ord_a[:, N_USED]))
    righe.append((L, diversi, sa.shape[0], margine))

righe.sort()
for L, d, n, mg in righe:
    if L in (0, 1, 2, 5, 10, 20, 30, 40):
        print(f"  strato {L:2d}: {d:5d} token su {n:5d} cambiano insieme "
              f"({d/max(n,1)*100:5.1f}%)   margine mediano 8°-9°: {mg:.5f}")
td = sum(r[1] for r in righe); tn = sum(r[2] for r in righe)
print(f"\n  TOTALE: {td} su {tn} ({td/max(tn,1)*100:.2f}%)  su {len(righe)} strati")
if righe:
    primi = [r for r in righe if r[0] < len(righe) // 3]
    ultimi = [r for r in righe if r[0] >= 2 * len(righe) // 3]
    def pc(g):
        return sum(x[1] for x in g) / max(sum(x[2] for x in g), 1) * 100
    print(f"  primo terzo: {pc(primi):5.2f}%   ultimo terzo: {pc(ultimi):5.2f}%")
    print("  Se cresce con la profondita', l'instradamento DERIVA — ed e' il")
    print("  meccanismo che nessuna misura di fedelta' poteva vedere.")
PY
