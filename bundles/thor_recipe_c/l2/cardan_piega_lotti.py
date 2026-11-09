#!/usr/bin/env python3
"""cardan_piega_lotti.py — 21/10 sera (Angelo: tutto parallelo, massima velocita', zero errori).

Same fold as cardan_piega.py (imported, NOT modified: same functions, same order of every accumulation), with the CPU
preparation of each batch moved OFF the critical path. Measured on layer 3 / 64 experts (py-spy, 21/10 18:33): the
fold was CPU-bound — FWHT rotation of the fp64 Hessians 29 %, GGUF memmap reads 15 %, normalisation + fp64 copies 16 %,
while the GPU algebra (Babai, scales) was only ~15 %. Here a thread pool prepares the planes, the rotated Hessians and the
targets of batch i+1 (and i+2) while the GPU folds batch i. Prepared objects are consumed in the original order, so the
result is byte-identical to cardan_piega.py by construction; `--confronta DIR` checks it against a serial run's tensori/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardan_piega as CP  # noqa: E402
if os.environ.get("FORGIA_BABAI_TRITON") == "1":            # 21/10: Babai in un kernel Triton per gruppo (non al bit: ordine somme)
    import cardan_babai_triton_gpu as _BT  # noqa: E402
    _BT.installa(CP)

PREP_FILI = int(os.environ.get("PIEGA_PREP_FILI", "6"))
ANTICIPO = int(os.environ.get("PIEGA_ANTICIPO", "2"))       # batches prepared ahead of the GPU


def _prepara(src, hs, CM, TensorKey, L, f, e, Cf, D, src_b=None):
    """Everything cardan_piega.piega_strato computes on the CPU for one expert, in the same way.
    src_b (21/10, experiment): target = dequant(bersaglio) + C D^T while the planes start from the host (src):
    the extra target is the difference bersaglio - host, in the stored (rotated) basis."""
    tt = src.load(TensorKey(L, e, f))
    H, fonte = hs.di(e)
    Hr = CM.ruota_H(H)
    Wd = Cf[e] @ D.T
    if src_b is not None:
        tb = src_b.load(TensorKey(L, e, f))
        qb = CP.ricostruisci_lotto(tb.planes[None], tb.scales.double()[None])[0]
        qh = CP.ricostruisci_lotto(tt.planes[None], tt.scales.double()[None])[0]
        Wd = Wd + (qb - qh)
    return tt.planes, tt.scales.double(), Hr, Wd, fonte


def apri(L, gguf):
    """Per-layer setup of cardan_piega.piega_strato (GGUF source + Hessians): 27 % of the main thread, now prefetched."""
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    return CM.gp.GGUFPlaneSource(gguf, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"]), \
        CP.Hessiane(L, CP.GQ_DIR, CP.HFFN_DIR)


def piega_strato(L, cardan_stato: dict, gguf: str, dev, dt, lotto: int, td: Path, reg, pool, esperti=None,
                 aperto=None, bersaglio=None):
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    from rd_ternary.formato import TensorKey
    gp = CM.gp
    src, hs = aperto if aperto is not None else apri(L, gguf)
    src_b = gp.GGUFPlaneSource(bersaglio, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"]) if bersaglio else None
    for f in ("gate", "up"):
        t0 = time.time()
        e_ = src.entry(TensorKey(L, 0, f))
        tid = e_.fmt.type_id
        raw1 = np.array(e_.tensors[0].data).reshape(e_.n_experts, e_.out, -1).copy()
        raw2 = (np.array(e_.tensors[1].data).reshape(int(e_.tensors[1].shape[2]), e_.out, -1).copy()
                if len(e_.tensors) > 1 else None)
        D = cardan_stato[f"D_{f}"].double()
        Cf = cardan_stato[f"C_{f}"].double()
        nE = e_.n_experts if esperti is None else min(esperti, e_.n_experts)
        per_K = {}
        for e in range(nE):
            if float(Cf[e].abs().max()) == 0.0:
                continue
            per_K.setdefault(e_.n_planes_of(e), []).append(e)
        tot = {"dW": 0.0, "fold": 0.0, "scale_sole": 0.0, "esperti": 0, "fonti": {}}
        lotti = [(K, lista[i:i + lotto]) for K, lista in per_K.items() for i in range(0, len(lista), lotto)]
        fut = {}

        def lancia(j):
            if j < len(lotti) and j not in fut:
                fut[j] = [pool.submit(_prepara, src, hs, CM, TensorKey, L, f, e, Cf, D, src_b) for e in lotti[j][1]]

        for j in range(ANTICIPO + 1):
            lancia(j)
        for j, (K, ee) in enumerate(lotti):
            prep = [x.result() for x in fut.pop(j)]
            lancia(j + ANTICIPO + 1)
            Ts, Ss, Hs, Ws = [], [], [], []
            for planes, scales, Hr, Wd, fonte in prep:              # same order as the serial loop
                tot["fonti"][fonte] = tot["fonti"].get(fonte, 0) + 1
                Ts.append(planes); Ss.append(scales); Hs.append(Hr); Ws.append(Wd)
            T = torch.stack(Ts).to(dev)
            S = torch.stack(Ss).to(dev, dt)
            H = torch.stack(Hs).to(dev, dt)
            Q0 = CP.ricostruisci_lotto(T, S)
            W = Q0 + torch.stack(Ws).to(dev, dt)
            Ha = H + 1e-4 * torch.eye(H.shape[-1], dtype=dt, device=dev)
            e_dw = CP.obiettivo_lotto(W, Q0, Ha).sum(-1)
            Ssc = CP._fp16(CP.scale_lotto(W, Ha, T, S.shape[-1]))
            e_sc = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, Ssc), Ha).sum(-1)
            T2, S2, cat = CP.jarq_lotto(W, H, T, S, giri=3, nu2=1e-4)
            e_f = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum(-1)
            tot["dW"] += float(e_dw.sum()); tot["fold"] += float(e_f.sum()); tot["scale_sole"] += float(e_sc.sum())
            tot["esperti"] += len(ee)
            T2c, S2c = T2.cpu(), S2.cpu().to(torch.float16)
            for jj, e in enumerate(ee):
                for k in range(K):
                    b = gp.encode_rows(T2c[jj, k].numpy().astype(np.int8), S2c[jj, k].numpy(), tid)
                    tp, sp = gp.decode_rows(b, tid, e_.in_)
                    assert np.array_equal(tp, T2c[jj, k].numpy()) and np.array_equal(
                        sp.view(np.uint16), S2c[jj, k].numpy().view(np.uint16)), (L, f, e, k)
                    if k == 0:
                        raw1[e] = b
                    else:
                        raw2[e_.slot(1, e)] = b
        # atomic publication (a crash never leaves a partial .npy that the resume would skip); exps2 first, main last:
        # the resume test looks at the main file only
        for nome_, arr_ in ((f"blk.{L}.ffn_{f}_exps2.weight.npy", raw2), (f"blk.{L}.ffn_{f}_exps.weight.npy", raw1)):
            if arr_ is None:
                continue
            tmp_ = td / (nome_ + ".parziale.npy")
            np.save(tmp_, arr_); os.replace(tmp_, td / nome_)
        cap = 1 - tot["fold"] / max(tot["dW"], 1e-300)
        cap_s = 1 - tot["scale_sole"] / max(tot["dW"], 1e-300)
        riga = {"strato": L, "famiglia": f, "esperti_piegati": tot["esperti"], "energia_correzione_H": tot["dW"],
                "residuo_dopo_piega_H": tot["fold"], "catturata": cap, "catturata_sole_scale": cap_s,
                "hessiane": tot["fonti"], "secondi": round(time.time() - t0, 1), "motore": "lotti"}
        reg.write(json.dumps(riga) + "\n"); reg.flush()
        CP.log(f"L{L} {f}: piegati {tot['esperti']} esperti, catturata {100 * cap:.1f} % (sole scale {100 * cap_s:.1f} %) "
               f"[{riga['secondi']} s, lotti]")


def confronta(a: Path, b: Path) -> bool:
    ok = True
    for p in sorted(a.glob("*.npy")):
        q = b / p.name
        uguali = q.exists() and np.array_equal(np.load(p), np.load(q))
        ok &= uguali
        print(f"{'IDENTICO' if uguali else 'DIVERSO '} {p.name}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=16)
    ap.add_argument("--esperti", type=int, default=None)
    ap.add_argument("--bersaglio", default=None, help="(experiment) GGUF whose experts + C D^T are the target; planes from --gguf")
    ap.add_argument("--confronta", default=None, help="tensori/ of a serial cardan_piega.py run: byte check")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    os.environ.setdefault("LEVE_DT", "float64" if dt == torch.float64 else "float32")
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    lo, _, hi = a.strati.partition("-")
    strati = [L for L in range(int(lo), int(hi or lo) + 1) if L in d["stato"] or str(L) in d["stato"]]
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / "registro.jsonl", "a") as reg, ThreadPoolExecutor(PREP_FILI) as pool:
        da_fare = [L for L in strati
                   if not all((td / f"blk.{L}.ffn_{f}_exps.weight.npy").exists() for f in ("gate", "up"))]
        if os.environ.get("PIEGA_ORDINE") == "inverso":            # 21/10: lavoratore aggiunto che va incontro all'altro
            da_fare = da_fare[::-1]
        prossimo = pool.submit(apri, da_fare[0], a.gguf) if da_fare else None
        for i, L in enumerate(da_fare):
            aperto = prossimo.result()
            prossimo = pool.submit(apri, da_fare[i + 1], a.gguf) if i + 1 < len(da_fare) else None
            if all((td / f"blk.{L}.ffn_{f}_exps.weight.npy").exists() for f in ("gate", "up")):
                continue                                       # fatto nel frattempo da un altro lavoratore
            stato = d["stato"].get(L, d["stato"].get(str(L)))
            piega_strato(L, stato, a.gguf, dev, dt, a.lotto, td, reg, pool, esperti=a.esperti, aperto=aperto, bersaglio=a.bersaglio)
    CP.log(f"FINE piega (lotti) {out} in {time.time() - t0:.1f} s")
    if a.confronta:
        sys.exit(0 if confronta(Path(a.confronta), td) else 1)


if __name__ == "__main__":
    main()
