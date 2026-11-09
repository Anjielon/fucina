#!/usr/bin/env python3
"""Piega esperti fp64, PIEGA_GPU=1: Cholesky GPU senza rocSOLVER.

Interfaccia dell'originale. Senza la variabile esegue il percorso originale.
Preparazione GGUF/Hessiane e codifica restano quelle originali sulla CPU.
Il nome distinto evita conflitti con la variante concorrente di preparazione.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

import cardan_piega as CP
from cardan_algebra_gpu import jarq_protetto, percorso


def main(argv=None):
    if os.environ.get("PIEGA_GPU", "0") != "1":
        with percorso(CP, False):
            return CP.main(argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=1)
    ap.add_argument("--esperti", type=int)
    a = ap.parse_args(argv)
    if a.lotto < 1 or (a.esperti is not None and a.esperti < 1):
        ap.error("--lotto e --esperti devono essere positivi")
    dev = torch.device(a.device)
    os.environ["LEVE_DT"] = "float64"
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    lo, _, hi = a.strati.partition("-")
    td = Path(a.out) / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    originale = CP.jarq_lotto

    def protetto(*args, **kwargs):
        CP.jarq_lotto = originale
        try:
            return jarq_protetto(CP, *args, **kwargs)
        finally:
            CP.jarq_lotto = protetto

    stream = torch.cuda.Stream(device=dev, priority=0) if dev.type == "cuda" else None
    contesto = torch.cuda.stream(stream) if stream is not None else torch.no_grad()
    with torch.no_grad(), contesto, percorso(CP), open(Path(a.out) / "registro.jsonl", "a") as reg:
        CP.jarq_lotto = protetto
        try:
            for L in range(int(lo), int(hi or lo) + 1):
                stato = d["stato"].get(L, d["stato"].get(str(L)))
                if stato is None or all((td / f"blk.{L}.ffn_{f}_exps.weight.npy").exists()
                                        for f in ("gate", "up")):
                    continue
                CP.piega_strato(L, stato, a.gguf, dev, torch.float64, a.lotto, td, reg,
                                esperti=a.esperti)
        finally:
            CP.jarq_lotto = originale
    CP.log("FINE piega GPU fp64", a.out)


if __name__ == "__main__":
    main()
