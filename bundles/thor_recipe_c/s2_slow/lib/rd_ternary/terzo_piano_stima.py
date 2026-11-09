"""R022b — il TERZO piano degli esperti vale la pena? Stima dal dkl.tsv di casa, senza GPU e senza misurare «+2».

Il motore di `thor-llama cloud/rexp-piani2` carica un terzo piano per esperto, ma il raccoglitore dKL misura solo «+1»
(2° piano). Prima di spendere GPU per misurare «+2», si puo' chiedere ai dati gia' presenti: quanto e' concentrato il
beneficio del 2° piano fra gli esperti?

Modello (DICHIARATO, non misurato): il beneficio del k-esimo piano di un esperto e' il beneficio del 2° moltiplicato per
rho^(k-2), con rho = (riduzione di errore di peso del 3° piano) / (riduzione del 2°). Su pesi sintetici con scala per gruppo
da 128 (thor_v2/rd_ternary/..., `scala_vera`/sim del 4/10): rho = 0,157 (gaussiana) · 0,189 (Laplace) · 0,256 (Student t3).
A costo uguale per piano (stessi byte), lo zaino sceglie per beneficio decrescente: il 3° piano di un esperto entra solo se
rho·b_e supera il beneficio b* dell'ultimo 2° piano preso, cioe' per gli esperti con b_e > b*/rho.

Uso:
  python3 -m rd_ternary.terzo_piano_stima --delta-kl dkl.tsv --budget-esperti 4000000000 [--rho 0.16 0.19 0.26 0.4] [--se-k 0]
Esce con un JSON: quanti esperti col 2° piano, quanti col 3°, beneficio previsto contro il solo 2° piano a pari byte,
e la concentrazione del beneficio (quota del totale nei top 1/5/10/20/50 % degli esperti).
Il numero che decide e' `guadagno_rel_vs_solo_2piano`: sotto ~2-3 % il formato 3° piano non vale la corsa «+2»;
sopra ~10 % merita di misurare «+2» per davvero (il rho del modello e' un'assunzione).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

COSTO_ESPERTO_R2B3M = 679_936      # gate + up (TQ1_0R_1024, 512x2048) + down (TQ1_0, 2048x512): un piano, un esperto
_ESPERTO = re.compile(r"^L(\d+)\.E(\d+)$")


def benefici_esperti(deltas: Dict[str, Dict[str, float]], se: Optional[Dict[str, Dict[str, float]]] = None,
                     se_k: float = 0.0, scale_cur: str = "sorgente") -> Dict[str, float]:
    """{gruppo L<s>.E<e>: beneficio >= 0 del 2° piano} = -(dKL(+1) - base) - k*se; i non positivi valgono 0."""
    out: Dict[str, float] = {}
    for g, opts in deltas.items():
        if not _ESPERTO.match(g) or "+1" not in opts:
            continue
        base = float(opts.get("cur", 0.0)) if scale_cur == "rifit" else 0.0
        b = -(float(opts["+1"]) - base) - se_k * float((se or {}).get(g, {}).get("+1", 0.0))
        out[g] = max(b, 0.0)
    return out


def concentrazione(b: np.ndarray, quote: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50)) -> Dict[str, float]:
    s = np.sort(b)[::-1]
    tot = float(s.sum()) or 1.0
    return {f"top_{int(q * 100)}%": round(float(s[:max(1, int(round(q * len(s))))].sum()) / tot, 4) for q in quote}


def zaino_piani(b: np.ndarray, budget: int, costo: int, rho: float, max_piani: int = 3) -> Dict:
    """Greedy per beneficio decrescente (costo uguale per ogni piano): vale come ottimo perche' il beneficio del piano k
    di un esperto e' decrescente in k (rho < 1), quindi il 2° di un esperto precede sempre il suo 3°."""
    n = len(b)
    voci = [(float(b[i]) * rho ** (k - 2), i, k) for i in range(n) for k in range(2, max_piani + 1)]
    voci.sort(key=lambda v: -v[0])
    cap = budget // costo
    scelte = voci[:cap]
    n2 = sum(1 for v in scelte if v[2] == 2)
    n3 = sum(1 for v in scelte if v[2] == 3)
    ben = sum(v[0] for v in scelte)
    soglia = scelte[-1][0] if scelte else 0.0
    return {"piani_totali": len(scelte), "esperti_con_2piano": n2, "esperti_con_3piano": n3, "beneficio": ben,
            "beneficio_ultimo_piano_preso": soglia}


def stima(b: np.ndarray, budget: int, costo: int = COSTO_ESPERTO_R2B3M, rhos: Sequence[float] = (0.157, 0.189, 0.256)) -> Dict:
    solo2 = zaino_piani(b, budget, costo, 0.0, max_piani=2)        # rho = 0: nessun 3° piano ha valore
    res = {"esperti": int(len(b)), "costo_per_piano_esperto_byte": costo, "budget_byte": int(budget),
           "piani_che_entrano": int(budget // costo), "concentrazione_beneficio_2piano": concentrazione(b),
           "solo_2piano": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in solo2.items()}, "con_3piano": {}}
    for r in rhos:
        z = zaino_piani(b, budget, costo, r, max_piani=3)
        rel = z["beneficio"] / solo2["beneficio"] - 1.0 if solo2["beneficio"] > 0 else 0.0
        res["con_3piano"][f"rho={r}"] = {"esperti_con_2piano": z["esperti_con_2piano"], "esperti_con_3piano": z["esperti_con_3piano"],
                                         "quota_esperti_3piano": round(z["esperti_con_3piano"] / max(len(b), 1), 4),
                                         "guadagno_rel_vs_solo_2piano": round(rel, 4)}
    return res


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta-kl", required=True)
    ap.add_argument("--budget-esperti", type=int, required=True, help="byte per i piani in piu' degli esperti (margine meno i densi)")
    ap.add_argument("--costo-esperto", type=int, default=COSTO_ESPERTO_R2B3M)
    ap.add_argument("--rho", nargs="*", type=float, default=[0.157, 0.189, 0.256, 0.4])
    ap.add_argument("--se-k", type=float, default=0.0)
    ap.add_argument("--scale-cur", choices=("sorgente", "rifit"), default="sorgente")
    a = ap.parse_args(argv)
    from .misurato import load_deltas, load_deltas_se
    d = load_deltas(a.delta_kl)
    be = benefici_esperti(d, load_deltas_se(a.delta_kl), a.se_k, a.scale_cur)
    if not be:
        raise SystemExit("nessun gruppo L<strato>.E<esperto> con opzione +1 nel file (--gruppi esperto?)")
    res = stima(np.array(list(be.values())), a.budget_esperti, a.costo_esperto, a.rho)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
