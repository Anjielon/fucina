#!/usr/bin/env python3
"""Leve di raffinamento sugli ESPERTI gate/up di R2B3m (1 piano ternario, TQ1_0R_1024), strati con statistiche
per esperto gia' catturate nel flusso dello studente quantizzato (pod-schur-esperti: Gq_LL = Σ w x_q x_qᵀ per esperto,
pesi del router veri; Wstar_LL = bersaglio di ricostruzione del maestro, SchurQuant braccio B).

leve:  jarq_e      JARQ (1 piano: scale congiunte dei 16 gruppi + Babai per gruppo), bersaglio bf16, H = Gq
       jarq_es     idem con bersaglio W* (ricostruzione del maestro dal flusso quantizzato, QEP/SchurQuant)
       schur_e     SchurOpt, bersaglio bf16, H = Gq
Uscita: <OUT>/<leva>/tensori/blk.L.ffn_{gate,up}_exps.weight.npy (uint8 [256, 512, byte_riga]) per patch_gguf.py.
uso: applica_esperti.py --leva jarq_e --strati 0-31 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comune as CM  # noqa: E402
import solutori as SO  # noqa: E402
from awsrc_codec import esperti_bf16  # noqa: E402

GQ = "@@ROOT:models@@/gguf/tony-tern/pod-schur-esperti"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leva", required=True, choices=("jarq_e", "jarq_es", "schur_e"))
    ap.add_argument("--strati", default="0-31")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--esperti", type=int, default=256)
    a = ap.parse_args()
    a0, b0 = (int(x) for x in a.strati.split("-"))
    dev = torch.device(a.device)
    out = CM.OUT_BASE / a.leva
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    reg = out / "registro.jsonl"
    sys.path.insert(0, CM.TT)
    from sorgente import SorgenteHF
    from rd_ternary.formato import TensorKey
    sh = SorgenteHF()
    for L in range(a0, b0 + 1):
        fatti = [td / f"blk.{L}.ffn_{f}_exps.weight.npy" for f in ("gate", "up")]
        if all(p.exists() for p in fatti):
            continue
        t0 = time.time()
        src = CM.gp.GGUFPlaneSource(CM.R2B3M, layers=[L])
        Gq = torch.from_numpy(np.load(f"{GQ}/Gq_{L:02d}.npy", mmap_mode="r")[: a.esperti].copy())
        Wst = np.load(f"{GQ}/Wstar_{L:02d}.npy", mmap_mode="r") if a.leva == "jarq_es" else None
        for fi, fam in enumerate(("gate", "up")):
            Wall = esperti_bf16(sh, L, fam)
            e_ = src.entry(TensorKey(L, 0, fam))
            raw_all = np.array(e_.tensors[0].data).reshape(e_.n_experts, e_.out, -1).copy()
            e0s, e1s = [], []
            for e in range(a.esperti):
                tt = src.load(TensorKey(L, e, fam))
                T, S = tt.planes.to(dev), tt.scales.float().to(dev).to(SO.F64)
                H = Gq[e].to(dev).double()
                H = 0.5 * (H + H.T)
                dg = torch.diagonal(H)
                if float(dg.max()) <= 0:                               # esperto mai instradato: resta com'e'
                    continue
                morti = dg <= 0
                H[morti, morti] = dg[~morti].mean()
                H = H / torch.diagonal(H).mean()
                Hr = CM.ruota_H(H).to(SO.F64)
                if a.leva == "jarq_es":
                    Wt = torch.from_numpy(np.asarray(Wst[e, fi * 512:(fi + 1) * 512], dtype=np.float32)).to(dev).double()
                else:
                    Wt = torch.from_numpy(Wall[e]).to(dev).double()
                Wr = CM.ruota(Wt).to(SO.F64)
                f0 = float(SO.obiettivo(Wr, SO.ricostruisci(T, S), Hr).sum())
                if a.leva in ("jarq_e", "jarq_es"):
                    T2, S2, cat = SO.jarq(Wr, Hr, T, S, [0], giri=3, nu2=1e-4)
                else:
                    T2, S2, cat = SO.schuropt(Wr, Hr, T, S, [0], S.shape[-1], passi=4)
                f1 = float(SO.obiettivo(Wr, SO.ricostruisci(T2, S2.to(SO.F64)), Hr).sum())
                e0s.append(f0); e1s.append(f1)
                raw_all[e] = CM.gp.encode_rows(T2[0].cpu().numpy().astype(np.int8),
                                               S2[0].cpu().to(torch.float16).numpy(), e_.fmt.type_id)
            np.save(td / f"blk.{L}.ffn_{fam}_exps.weight.npy", raw_all)
            with open(reg, "a") as f:
                f.write(json.dumps({"strato": L, "famiglia": fam, "obiettivo_prima": sum(e0s), "obiettivo_dopo": sum(e1s),
                                    "rel": (sum(e1s) / max(sum(e0s), 1e-30)) ** 0.5, "secondi": round(time.time() - t0, 1)}) + "\n")
            CM.log(f"L{L} {fam}: obiettivo relativo {(sum(e1s) / max(sum(e0s), 1e-30)) ** 0.5:.4f} ({time.time() - t0:.0f} s)")
        del Gq


if __name__ == "__main__":
    main()
