#!/usr/bin/env python3
"""30/9 — MMLU completo (14.042) R2B3m vs Q2_K dai registri MCLP del Dell: accuratezza, ±1σ, McNemar appaiato, tabella per pezzo."""
import math, re, datetime
from pathlib import Path
from scipy.stats import binomtest
A = Path("@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn"); C = A / "completo"
PEZZI = ["00000", "02000", "04000", "06000", "08000", "10000", "12000", "14000"]
NT = {p: (42 if p == "14000" else 2000) for p in PEZZI}
def leggi(f):
    r = {}
    for riga in open(f, errors="replace"):
        i = riga.find("MCLP\t")
        if i < 0: continue
        c = riga[i:].split("\t")
        if not c[1].strip().isdigit(): continue
        r[int(c[1])] = int(c[2]) == int(c[3])
    return r
def reg(mod, p):
    if mod == "R2B3m" and p == "00000": return A / "R2B3.log"   # logit identici al bit (verifica pod 30/9)
    return C / f"MMLUC-{'Q2K' if mod == 'Q2K' else 'R2B3m'}-{p}.log"
tab, tot = [], {"R2B3m": [], "Q2K": []}
righe = ["| pezzo | domande | R2B3m | Q2_K | Δ |", "|---|---|---|---|---|"]
for p in PEZZI:
    x = {}
    for m in tot:
        f = reg(m, p)
        x[m] = leggi(f) if f.exists() else {}
    ok = all(len(x[m]) == NT[p] for m in tot)
    if not ok:
        righe.append(f"| {p} | {NT[p]} | {len(x['R2B3m'])} fatte | {len(x['Q2K'])} fatte | — |"); continue
    for m in tot: tot[m] += [x[m][i] for i in range(NT[p])]
    a, b = (100 * sum(x[m].values()) / NT[p] for m in ("R2B3m", "Q2K"))
    righe.append(f"| {p} | {NT[p]} | {a:.2f} | {b:.2f} | {a-b:+.2f} |")
n = len(tot["R2B3m"])
out = [f"# MMLU completo sul Dell — R2B3m contro Q2_K ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
       f"Domande con entrambi i modelli: **{n} / 14042**", ""] + righe + [""]
if n:
    for m in tot:
        acc = sum(tot[m]) / n
        out.append(f"- **{m}**: {100*acc:.2f} ± {100*math.sqrt(acc*(1-acc)/n):.2f} (1σ)")
    g = sum(1 for r, q in zip(tot["R2B3m"], tot["Q2K"]) if r and not q)
    l = sum(1 for r, q in zip(tot["R2B3m"], tot["Q2K"]) if q and not r)
    pv = binomtest(g, g + l, 0.5).pvalue if g + l else 1.0
    out.append(f"- McNemar appaiato (esatto): R2B3m giusta/Q2_K sbagliata {g}, contrario {l}, p = {pv:.3g}")
Path("@@ROOT:forge@@/stato/mmlu_completo_R2B3m_Q2K.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
if n == 14042:
    with open("@@ROOT:forge@@/stato/cicli.tsv", "a") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%dT%H:%M}\tCHIUSO\tMMLU completo 14.042 sul Dell: " +
                " · ".join(o[2:] for o in out if o.startswith("- ")) + "\n")
