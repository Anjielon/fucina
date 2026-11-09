#!/usr/bin/env python3
"""cardan_piega_down.py — 25/10: FOLD of the distilled CARDAN DOWN correction (cardan_kd_moduli.CardanDown, trained by
cardan_congiunto_scale.py --down-lowrank-rango R) into the EXISTING TQ1_0 ternary planes and fp16 scales of the routed
experts' ffn_down_exps (+ ffn_down_exps2 of the hot experts), keeping the host's tensor names, types, shapes and byte
counts (the engine runs it unchanged, at the host's speed). Modelled on cardan_piega.py (gate/up, TQ1_0R_1024).

For each layer L and expert e (TQ1_0 = plain basis, block 256, one fp16 scale per 256 weights -> ng = FF/256 = 2):
    host    Q_e = sum_k s_k T_k   (ALL planes; with --scale-pt the plane-1 scales are the TRAINED ones,
                                   s = fp16(d0 * exp(theta)), exactly StratoEsperti._scala / scale_down_esporta.py)
    target  W*_e = Q_e + C_e D_down^T                      [2048, 512]
    fold    batched JARQ of cardan_piega.jarq_lotto (joint LS of the fp16 scales + Babai nearest-plane codes, accepted
            row by row only if they lower the error; rows that end worse than the host go back to the host)
    metric  H_e = down-input second moment E[a a^T] from --hessiane-down DIR (H_down_LL.npy, [512,512] shared or
            [E,512,512] per expert); absent = identity (plain Frobenius). As of 25/10 no down-input Hessian has been
            captured: identity is the declared fallback (README_DOWN.md, open risk).
Experts with C_e = 0 keep their bytes (or, with --scale-pt, are re-encoded with the trained scales and unchanged trits,
like scale_down_esporta.py). This script therefore REPLACES scale_down_esporta.py when both levers were trained (they
write the same file name): pass --scale-pt scale_<passo>.pt here and do not run scale_down_esporta.py.
Output: <out>/tensori/blk.L.ffn_down_exps.weight.npy (+ ..._exps2.weight.npy), uint8, host shapes, for
raffina/patch_gguf.py; <out>/registro.jsonl with the captured fraction 1 - |W* - fold|_H^2 / |C_e D^T|_H^2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import cardan_piega as CP  # noqa: E402  (jarq_lotto, scale_lotto, ricostruisci_lotto, obiettivo_lotto, _fp16, RAFF)

log = CP.log


class HessianeDown:
    """Per-expert (or shared) down-input metric, normalised to mean diagonal 1; identity when no file is given."""

    def __init__(self, L: int, C: int, hdir: Optional[str]):
        self.C = C
        self.h = None
        self.fonte = "identita"
        if hdir:
            p = Path(hdir) / f"H_down_{L:02d}.npy"
            if p.exists():
                self.h = np.load(p, mmap_mode="r")
                self.fonte = f"{p.name} ({'per esperto' if self.h.ndim == 3 else 'condivisa'})"

    def di(self, e: int) -> torch.Tensor:
        if self.h is not None:
            H = torch.from_numpy(np.asarray(self.h[e] if self.h.ndim == 3 else self.h, np.float64))
            Hn = CP._normalizza(H.clone())
            if Hn is not None:
                return Hn
        return torch.eye(self.C, dtype=torch.float64)


def scale_ospite(S: torch.Tensor, theta: Optional[torch.Tensor]) -> torch.Tensor:
    """S [K, R, ng] fp16 values (fp64) -> plane-1 scales replaced by fp16(d0 * exp(theta)) when theta is given."""
    if theta is None:
        return S
    S = S.clone()
    d0 = S[0]
    S[0] = (d0 * torch.exp(theta.double().reshape(d0.shape))).to(torch.float16).double()
    return S


def piega_strato_down(L: int, stato: dict, src, dev, dt, lotto: int, td: Path, reg, theta: Optional[torch.Tensor] = None,
                      hdir: Optional[str] = None, esperti: Optional[int] = None, giri: int = 3) -> dict:
    """Folds one layer. `src` = gguf_piani.GGUFPlaneSource (or a duck-typed fake in the tests). Returns the totals."""
    from rd_ternary.formato import TensorKey
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    gp = CM.gp
    t0 = time.time()
    en = src.entry(TensorKey(L, 0, "down"))
    tid = en.fmt.type_id
    raw1 = np.array(en.tensors[0].data).reshape(en.n_experts, en.out, -1).copy()
    raw2 = (np.array(en.tensors[1].data).reshape(int(en.tensors[1].shape[2]), en.out, -1).copy()
            if len(en.tensors) > 1 else None)
    D = stato["D_down"].double()
    Cd = stato["C_down"].double()
    hs = HessianeDown(L, en.in_, hdir)
    nE = en.n_experts if esperti is None else min(esperti, en.n_experts)
    per_K, solo_scale = {}, []
    for e in range(nE):
        if float(Cd[e].abs().max()) == 0.0:
            if theta is not None:
                solo_scale.append(e)
            continue
        per_K.setdefault(en.n_planes_of(e), []).append(e)
    tot = {"dW": 0.0, "fold": 0.0, "scale_sole": 0.0, "esperti": 0, "solo_scale": len(solo_scale)}

    def scrivi(e, k, Tk: np.ndarray, Sk: np.ndarray):
        b = gp.encode_rows(Tk.astype(np.int8), Sk.astype(np.float16), tid)
        tp, sp = gp.decode_rows(b, tid, en.in_)                                   # bit-exact round trip
        assert np.array_equal(tp, Tk) and np.array_equal(sp.view(np.uint16), Sk.astype(np.float16).view(np.uint16)), \
            (L, e, k)
        if k == 0:
            raw1[e] = b
        else:
            raw2[en.slot(1, e)] = b

    for e in solo_scale:                                  # C_e = 0, trained scales only (= scale_down_esporta.py)
        tt = src.load(TensorKey(L, e, "down"))
        S = scale_ospite(tt.scales.double(), theta[e])
        scrivi(e, 0, tt.planes[0].numpy(), S[0].numpy())
    for K, lista in per_K.items():
        for i in range(0, len(lista), lotto):
            ee = lista[i:i + lotto]
            Ts, Ss, Hs, Ws = [], [], [], []
            for e in ee:
                tt = src.load(TensorKey(L, e, "down"))
                assert not tt.rotated, "ffn_down_exps expected TQ1_0 (plain basis)"
                Ts.append(tt.planes)
                Ss.append(scale_ospite(tt.scales.double(), None if theta is None else theta[e]))
                Hs.append(hs.di(e))
                Ws.append(Cd[e] @ D.T)
            T = torch.stack(Ts).to(dev)
            S = torch.stack(Ss).to(dev, dt)
            H = torch.stack(Hs).to(dev, dt)
            Q0 = CP.ricostruisci_lotto(T, S)
            W = Q0 + torch.stack(Ws).to(dev, dt)
            Ha = H + 1e-4 * torch.eye(H.shape[-1], dtype=dt, device=dev)
            e_dw = CP.obiettivo_lotto(W, Q0, Ha).sum(-1)
            Ssc = CP._fp16(CP.scale_lotto(W, Ha, T, S.shape[-1]))
            e_sc = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, Ssc), Ha).sum(-1)
            T2, S2, _ = CP.jarq_lotto(W, H, T, S, giri=giri, nu2=1e-4)
            e_f = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum(-1)
            tot["dW"] += float(e_dw.sum()); tot["fold"] += float(e_f.sum()); tot["scale_sole"] += float(e_sc.sum())
            tot["esperti"] += len(ee)
            T2c, S2c = T2.cpu(), S2.cpu().to(torch.float16)
            for j, e in enumerate(ee):
                for k in range(K):
                    scrivi(e, k, T2c[j, k].numpy().astype(np.int8), S2c[j, k].numpy())
    np.save(td / f"blk.{L}.ffn_down_exps.weight.npy", raw1)
    if raw2 is not None:
        np.save(td / f"blk.{L}.ffn_down_exps2.weight.npy", raw2)
    cap = 1 - tot["fold"] / max(tot["dW"], 1e-300)
    cap_s = 1 - tot["scale_sole"] / max(tot["dW"], 1e-300)
    riga = {"strato": L, "famiglia": "down", "esperti_piegati": tot["esperti"], "solo_scale": tot["solo_scale"],
            "energia_correzione_H": tot["dW"], "residuo_dopo_piega_H": tot["fold"], "catturata": cap,
            "catturata_sole_scale": cap_s, "metrica": hs.fonte, "scale_allenate": theta is not None,
            "secondi": round(time.time() - t0, 1)}
    if reg is not None:
        reg.write(json.dumps(riga) + "\n"); reg.flush()
    log(f"L{L} down: piegati {tot['esperti']} esperti (+{tot['solo_scale']} solo scale), catturata {100 * cap:.1f} % "
        f"(sole scale {100 * cap_s:.1f} %) [{riga['secondi']} s]")
    return riga


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="cardan_down_migliore.pt (or tappe/cardan_down_<passo>.pt)")
    ap.add_argument("--gguf", required=True, help="host GGUF the run trained on (--gguf of cardan_congiunto_scale.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-pt", default=None, help="scale_<passo>.pt of the SAME step (--scale-down-lr runs)")
    ap.add_argument("--hessiane-down", default=None, help="DIR with H_down_LL.npy (default: identity metric)")
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=16)
    ap.add_argument("--giri", type=int, default=3)
    ap.add_argument("--esperti", type=int, default=None, help="(probes) only the first N experts")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    os.environ.setdefault("LEVE_DT", "float64" if dt == torch.float64 else "float32")
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    th = torch.load(a.scale_pt, map_location="cpu") if a.scale_pt else None
    lo, _, hi = a.strati.partition("-")
    strati = [L for L in range(int(lo), int(hi or lo) + 1) if L in d["stato"] or str(L) in d["stato"]]
    if th is not None:                                    # layers with trained scales but no down correction
        strati = sorted(set(strati) | {int(L) for L in th if int(lo) <= int(L) <= int(hi or lo)})
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    with open(out / "registro.jsonl", "a") as reg:
        for L in strati:
            if (td / f"blk.{L}.ffn_down_exps.weight.npy").exists():
                continue
            stato = d["stato"].get(L, d["stato"].get(str(L)))
            theta = None if th is None else th.get(L, th.get(str(L)))
            src = CM.gp.GGUFPlaneSource(a.gguf, layers=[L], bases=["ffn_down_exps"])
            if stato is None:                             # scales only on this layer: zero correction
                from rd_ternary.formato import TensorKey
                en = src.entry(TensorKey(L, 0, "down"))
                stato = {"D_down": torch.zeros(en.in_, 1), "C_down": torch.zeros(en.n_experts, en.out, 1)}
            piega_strato_down(L, stato, src, dev, dt, a.lotto, td, reg, theta=theta, hdir=a.hessiane_down,
                              esperti=a.esperti, giri=a.giri)
    log("FINE piega down", out)


if __name__ == "__main__":
    main()
