#!/usr/bin/env python3
"""I quattro numeri MMLU di UN registro MCLP (26/9/2026, per i checkpoint E2E valutati su Aragorn).

Stesse definizioni di `mmlu_depurato.py` (importato in sola lettura: stesse funzioni, nessuna copia del calcolo):
grezza = scelta sui 4 logprob (= Final result); glob = depurata con preferenza globale BILANCIATA per lettera giusta
(stima principale); lib = depurata senza etichette; margine = margine medio (nat) della giusta dopo la depurazione glob;
massa = mediana di P(A)+P(B)+P(C)+P(D). Stima puntuale (niente bootstrap: gli IC stanno nel rapporto completo).

Con --confronta REGISTRO: scelte uguali compito per compito, |delta logprob| massimo e medio, delta dei quattro numeri —
per il collaudo di riproducibilita' di un secondo banco (Aragorn, Vulkan su NVIDIA) contro il Corsair (Vulkan su AMD).

Uso: mmlu_quattro_numeri.py REGISTRO [--confronta REGISTRO_RIFERIMENTO] [--json USCITA]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mmlu_depurato as MD  # noqa: E402  (sola lettura)


def quattro(reg: Path, oro: np.ndarray) -> dict:
    r = MD.leggi_registro(reg)
    lp = r["lp32"].astype(np.float64)
    S = MD.logsm(lp)
    grezza = MD.valuta(S[None], oro)
    ones = np.ones((1, MD.N))
    PRg, _ = MD.prior_celle(ones, S, np.zeros(MD.N, dtype=np.int64), 1, oro, None)
    glob = MD.valuta(S[None] - PRg, oro)
    PRl = MD.logsm(S.mean(axis=0))[None, None, :]
    lib = MD.valuta(S[None] - PRl, oro)
    mx = lp.max(axis=1)
    massa = np.exp(mx + np.log(np.exp(lp - mx[:, None]).sum(axis=1)))
    return {"registro": str(reg), "finale": r["finale"], "grezza": 100 * float(grezza["giusta"].mean()),
            "glob": 100 * float(glob["giusta"].mean()), "lib": 100 * float(lib["giusta"].mean()),
            "margine_glob": float(glob["margine"].mean()), "massa_mediana": 100 * float(np.median(massa)),
            "_lp": lp, "_scelta": lp.argmax(axis=1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("registro")
    ap.add_argument("--confronta", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    mappa = json.loads(MD.MAPPA.read_text(encoding="utf-8"))
    assert mappa["sha256_compiti"] == MD.SHA_BANCO
    oro = np.array([MD.LETTERE.index(x) for x in mappa["lettera"]])
    q = quattro(Path(a.registro), oro)
    out = {k: v for k, v in q.items() if not k.startswith("_")}
    print(f"{Path(a.registro).name}: grezza {q['grezza']:.2f} · depurata glob {q['glob']:.2f} · lib {q['lib']:.2f}"
          f" · margine glob {q['margine_glob']:+.3f} nat · massa mediana {q['massa_mediana']:.2f} % · {q['finale']}")
    if a.confronta:
        c = quattro(Path(a.confronta), oro)
        d = np.abs(q["_lp"] - c["_lp"])
        out["confronto"] = {"riferimento": a.confronta, "scelte_uguali": int((q["_scelta"] == c["_scelta"]).sum()),
                            "delta_lp_max": float(d.max()), "delta_lp_medio": float(d.mean()),
                            **{f"delta_{k}": q[k] - c[k] for k in ("grezza", "glob", "lib", "margine_glob",
                                                                   "massa_mediana")}}
        x = out["confronto"]
        print(f"contro {Path(a.confronta).name}: scelte uguali {x['scelte_uguali']}/{MD.N} · |Δlogprob| max "
              f"{x['delta_lp_max']:.3g} medio {x['delta_lp_medio']:.3g} · Δ grezza {x['delta_grezza']:+.2f} · Δ glob "
              f"{x['delta_glob']:+.2f} · Δ lib {x['delta_lib']:+.2f} · Δ margine {x['delta_margine_glob']:+.3f} · "
              f"Δ massa {x['delta_massa_mediana']:+.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
