#!/usr/bin/env python3
"""McNemar ESATTO appaiato fra due registri MCLP dello stesso banco (stesso ordine dei compiti), letti con
mmlu_depurato.leggi_registro (sola lettura). Stampa accuratezze, b (A giusta/B sbagliata), c (contrario), p bilaterale.
Uso: mcnemar_mclp.py REGISTRO_A REGISTRO_B [--json USCITA]"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mmlu_depurato as MD  # noqa: E402


def p_esatto(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


ap = argparse.ArgumentParser(); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--json'); x = ap.parse_args()
A, B = MD.leggi_registro(Path(x.a)), MD.leggi_registro(Path(x.b))
assert (A['giusta'] == B['giusta']).all(), 'banchi diversi (risposte giuste non coincidono)'
ga, gb = A['scelta'] == A['giusta'], B['scelta'] == B['giusta']
b, c = int((ga & ~gb).sum()), int((~ga & gb).sum())
out = {'A': x.a, 'B': x.b, 'n': int(len(ga)), 'acc_A': 100 * float(ga.mean()), 'acc_B': 100 * float(gb.mean()),
       'delta_B_meno_A': 100 * float(gb.mean() - ga.mean()), 'A_giusta_B_sbagliata': b, 'B_giusta_A_sbagliata': c,
       'p_mcnemar_esatto': p_esatto(b, c), 'scelte_uguali': int((A['scelta'] == B['scelta']).sum()),
       'finale_A': A['finale'], 'finale_B': B['finale']}
print(json.dumps(out, indent=1))
if x.json:
    Path(x.json).write_text(json.dumps(out, indent=1))
