#!/usr/bin/env python3
"""prova_piega_veloce_cpu.py (22/10) — CPU-only byte check of cardan_piega_veloce against cardan_piega_lotti.

Runs the lotti fold (reference, imported unchanged) and the veloce fold with the given switches on the SAME inputs, on
the CPU (HIP_VISIBLE_DEVICES= is enforced), in fp32 (the GPU working dtype) or fp64, then compares every tensor byte
and the registro energies. Also counts aten ops / matmul FLOP / bytes per family (TorchDispatchMode) for both, which
is device independent and feeds the time model of PIEGA_VELOCE_1022.md.
Usage: HIP_VISIBLE_DEVICES= python prova_piega_veloce_cpu.py --out DIR [--strato 3] [--esperti 32] [--dtype float32]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

assert os.environ.get("HIP_VISIBLE_DEVICES", "x") == "" , "CPU only: run with HIP_VISIBLE_DEVICES="
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardan_piega as CP  # noqa: E402
import cardan_piega_lotti as PL  # noqa: E402
import cardan_piega_veloce as PV  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

W_ = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_CONGIUNTO_S7T550_1020"
BASE = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/S7_CONGIUNTO_T550_1020/Thor-T-C-S7-T550-g10.gguf"


class Conta(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.op = Counter(); self.flop = Counter(); self.byte = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        n = str(func.overloadpacket.__name__)
        self.op[n] += 1
        if n in ("bmm", "mm", "matmul"):
            a, b = args[0], args[1]
            self.flop[n] += 2 * a.numel() * b.shape[-1]
        tens = [x for x in list(args) + [out] if isinstance(x, torch.Tensor)]
        self.byte[n] += sum(x.numel() * x.element_size() for x in tens)
        return out


def corsa(modulo, nome, a, dt, stato):
    out = Path(a.out) / nome
    td = out / "tensori"; td.mkdir(parents=True, exist_ok=True)
    c = Conta()
    t0 = time.time()
    with open(out / "registro.jsonl", "w") as reg, ThreadPoolExecutor(PL.PREP_FILI) as pool:
        with c:
            modulo.piega_strato(a.strato, stato, BASE, torch.device("cpu"), dt, 16, td, reg, pool, esperti=a.esperti)
    return time.time() - t0, c, [json.loads(x) for x in open(out / "registro.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--strato", type=int, default=3)
    ap.add_argument("--esperti", type=int, default=32)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--tappa", default="350")
    ap.add_argument("--salta-rif", action="store_true")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    os.environ["LEVE_DT"] = a.dtype
    d = torch.load(f"{W_}/allena/tappe/cardan_{a.tappa}.pt", map_location="cpu", weights_only=False)
    stato = d["stato"].get(a.strato, d["stato"].get(str(a.strato)))
    ris = {"strato": a.strato, "esperti": a.esperti, "dtype": a.dtype, "motore_veloce": PV._motore()}
    if not a.salta_rif:
        s, c, r = corsa(PL, "rif", a, dt, stato)
        ris["rif"] = {"s": round(s, 1), "op": sum(c.op.values()), "flop_matmul": sum(c.flop.values()),
                      "byte_op": sum(c.byte.values()), "top_op": c.op.most_common(12), "registro": r}
        print("rif", ris["rif"]["s"], "s", flush=True)
    s, c, r = corsa(PV, "veloce", a, dt, stato)
    ris["veloce"] = {"s": round(s, 1), "op": sum(c.op.values()), "flop_matmul": sum(c.flop.values()),
                     "byte_op": sum(c.byte.values()), "top_op": c.op.most_common(12), "registro": r}
    print("veloce", ris["veloce"]["s"], "s", flush=True)
    rif = Path(a.out) / "rif" / "tensori"; vel = Path(a.out) / "veloce" / "tensori"
    ok = True
    for p in sorted(rif.glob("*.npy")):
        q = vel / p.name
        u = q.exists() and np.array_equal(np.load(p), np.load(q))
        ok &= u
        print("IDENTICO" if u else "DIVERSO ", p.name)
    if "rif" in ris:
        for x, y in zip(sorted(ris["rif"]["registro"], key=lambda z: z["famiglia"]),
                        sorted(ris["veloce"]["registro"], key=lambda z: z["famiglia"])):
            for k in ("energia_correzione_H", "residuo_dopo_piega_H", "catturata", "catturata_sole_scale", "hessiane"):
                u = x[k] == y[k]
                ok &= u
                print("registro", x["famiglia"], k, "UGUALE" if u else f"DIVERSO {x[k]} vs {y[k]}")
    ris["identico"] = bool(ok)
    (Path(a.out) / "esito.json").write_text(json.dumps(ris, indent=1, default=str))
    print("ESITO", "IDENTICO" if ok else "DIVERSO")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
