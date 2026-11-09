#!/usr/bin/env python3
"""e1_esporta_residui.py — residui_<nome>.pt (salvati da e1_recupero_lettera a ogni passo / al migliore) ->
residui_tq1.npz + residui.json, SENZA lo studente.

Serve alla corsa notturna (stato/e1_corsa.sh) se l'allenamento viene interrotto (tetto di tempo del processo,
guasto) dopo aver salvato i latenti ma prima dell'esportazione: il lavoro gia' fatto non si perde. Le matrici del file
dipendono solo da A, B, gamma e dalla config (rango, piani): le permutazioni dello studente (perm_x, inv_y) non
entrano nell'esportazione (residui_ternari.matrici_file). Stesso formato, stessi byte di e1_recupero_lettera
(prova: test_e1_esporta.py: stessi byte dell'esportazione del driver).

Uso: e1_esporta_residui.py --pt OUT/residui_migliore.pt --uscita OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import residui_ternari as RT   # noqa: E402


def moduli_da_pt(path) -> tuple:
    d = torch.load(path, map_location="cpu", weights_only=False)   # file nostro (salva_latenti): meta con scalari numpy
    cfg = RT.ConfigResidui(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["cfg"].items()})
    mods = {}
    for nome, sd in d["latenti"].items():
        fuori, r = sd["B"].shape
        dentro = sd["A"].shape[1]
        if r != cfg.rango:
            raise SystemExit(f"{nome}: rango {r} != cfg {cfg.rango}")
        m = RT.ResiduoTernario(nome, fuori, dentro, cfg)
        m.load_state_dict(sd)
        mods[nome] = m
    return mods, cfg, d.get("meta", {})


def esporta(pt, uscita) -> dict:
    out = Path(uscita)
    mods, cfg, meta = moduli_da_pt(pt)
    tr, info = RT.tensori_gguf(mods)
    np.savez(out / "residui_tq1.npz", **tr)
    passo = meta.get("passo")
    json.dump({"tensori": info, "rango": cfg.rango, "piani": cfg.piani, "rp": RT.RP, "famiglie": list(cfg.famiglie),
               "strati": RT.strati_di(mods), "passo": passo, "da": str(pt),
               "uso_motore": "cur = mul_mat(W,x)[+piani] + mul_mat(<fam>_lrb, mul_mat(<fam>_lra, x))"},
              open(out / "residui.json", "w"), indent=1)
    return {"n_tensori": len(tr), "byte": int(sum(v.nbytes for v in tr.values())), "passo": passo}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(esporta(a.pt, a.uscita)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
