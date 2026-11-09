#!/usr/bin/env python3
"""ternarizza_down_h.py — 25/10: RE-TERNARIZATION of the routed expert DOWN planes (ffn_down_exps, + ffn_down_exps2 of the
hot experts) toward the bf16 teacher under the real per-expert down-input metric.

Why: the expert down planes of the host come from an old legacy pipeline and were never re-ternarized with input
statistics; putting the student's ternary downs into the full-precision teacher costs ~20 MMLU points.

For each layer L and expert e (TQ1_0 = plain basis, block 256, one fp16 scale per 256 weights -> ng = 512/256 = 2):
    target  W_e  = bf16 down_proj[e] of the teacher (thor-src-hf, mlp.experts.down_proj, [2048, 512], identity map to
                   GGUF order; checked per layer by the host/teacher correlation)
    start   Q_e  = host codes and fp16 scales, ALL planes (main + exps2), unchanged plane count
    metric  H_e  = E[a a^T] of the down input a (SwiGLU output) captured in the TEACHER flow
                   (hessiane-down/Hdown_LL.npy, stored as a MEAN: sum / count, cattura_hessiane_down_thor.py:292),
                   shrunk toward the count-weighted layer mean for rarely routed experts:
                   H'_e = (n_e H_e + n0 Hbar) / (n_e + n0)   (n0 = --ritiro, 0 = no shrinkage)
                   then normalised to mean diagonal 1 (cardan_piega._normalizza) + nu2 I.
    solver  batched JARQ of cardan_piega.jarq_lotto (the same one used by cardan_piega.py / cardan_piega_down.py):
            joint LS of the fp16 group scales of all planes alternated with Babai nearest-plane codes, row by row
            acceptance; rows whose final error is not lower than the host's go back to the host (bytes preserved).
Report (registro.jsonl, one line per layer):
    err_H_rel_*   = sum_e n_e tr(E_e H_e E_e^T) / sum_e n_e tr(W_e H_e W_e^T)  (raw H_e, count weighted = the
                    expected squared output error of the layer's down GEMMs before the router weight, relative)
    err_Hopt_rel_* = same with the optimisation metric H'_e (normalised, unweighted over experts)
    err_F_rel_*   = plain Frobenius |W - Q| / |W| (informative only, NOT optimised)
    rows accepted, experts changed, seconds.
Output: <out>/tensori/blk.L.ffn_down_exps.weight.npy (+ ..._exps2.weight.npy if the host has it), uint8, host shapes
and byte counts, for @@ROOT:models@@/build/leve/raffina/patch_gguf.py.
ffn_down_shexp is NOT handled: no shared-expert down-input statistics exist (Hdown covers the routed experts only).
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
import cardan_piega as CP  # noqa: E402  (jarq_lotto, ricostruisci_lotto, obiettivo_lotto, _normalizza, RAFF)

log = CP.log
HDOWN_DIR = "@@ROOT:models@@/gguf/tony-tern/hessiane-down"
SORGENTE_DIR = "@@ROOT:depot@@/tutto_ternario/codice"       # sorgente.SorgenteHF (bf16 teacher reader)
F64 = torch.float64


class MetricaDown:
    """Raw per-expert down-input second moments (mean form), counts, shrinkage and normalisation."""

    def __init__(self, L: int, hdir: str, ritiro: float):
        self.h = np.load(Path(hdir) / f"Hdown_{L:02d}.npy", mmap_mode="r")               # [E, FF, FF] float32
        self.n = np.load(Path(hdir) / f"Hdown_{L:02d}_conta.npy").astype(np.float64)       # [E]
        assert self.h.ndim == 3 and self.h.shape[0] == self.n.shape[0], (self.h.shape, self.n.shape)
        self.ritiro = float(ritiro)
        w = self.n / max(self.n.sum(), 1.0)
        acc = np.zeros(self.h.shape[1:], np.float64)
        for e in range(self.h.shape[0]):                     # count-weighted layer mean (= pooled E[a a^T])
            if w[e] > 0:
                acc += w[e] * np.asarray(self.h[e], np.float64)
        self.hbar = torch.from_numpy(0.5 * (acc + acc.T))

    def grezza(self, e: int) -> torch.Tensor:
        H = torch.from_numpy(np.asarray(self.h[e], np.float64))
        return 0.5 * (H + H.T)

    def ottimizzazione(self, e: int) -> torch.Tensor:
        ne = self.n[e]
        H = self.grezza(e)
        if self.ritiro > 0 or ne == 0:
            n0 = self.ritiro if self.ritiro > 0 else 1.0
            H = (ne * H + n0 * self.hbar) / (ne + n0)
        Hn = CP._normalizza(H.clone())
        return Hn if Hn is not None else torch.eye(H.shape[0], dtype=F64)


def _err(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """tr((W-Q) H (W-Q)^T) per batch element, fp64. W, Q [B,R,C], H [B,C,C]."""
    E = (W - Q).double()
    return (torch.bmm(E, H.double()) * E).sum((-1, -2))


def init_rtn(W: torch.Tensor, K: int, ng: int) -> tuple:
    """Fresh greedy residual ternarization of W [B,R,C]: per plane and group, s = mean|res| (absmean, BitNet b1.58),
    t = clip(round(res / s), -1, 1). Returns (T int8 [B,K,R,C], S fp16-valued [B,K,R,ng])."""
    B, R, C = W.shape
    G = C // ng
    res = W.reshape(B, R, ng, G).clone()
    Ts, Ss = [], []
    for _ in range(K):
        s = CP._fp16(res.abs().mean(-1).clamp_min(1e-8))
        t = (res / s.unsqueeze(-1)).round().clamp(-1, 1)
        res = res - t * s.unsqueeze(-1)
        Ts.append(t.reshape(B, R, C).to(torch.int8)); Ss.append(s)
    return torch.stack(Ts, 1), torch.stack(Ss, 1)


def ternarizza_strato(L: int, src, W_all: np.ndarray, met: MetricaDown, dev, dt, lotto: int, td: Path, reg,
                      giri: int = 3, nu2: float = 1e-4, esperti: Optional[int] = None, init: str = "ospite") -> dict:
    """One layer. src = gguf_piani.GGUFPlaneSource restricted to ffn_down_exps; W_all [E, 2048, 512] float32 teacher."""
    from rd_ternary.formato import TensorKey
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    gp = CM.gp
    t0 = time.time()
    en = src.entry(TensorKey(L, 0, "down"))
    tid = en.fmt.type_id
    E_, R_, C_ = en.n_experts, en.out, en.in_
    assert tuple(W_all.shape) == (E_, R_, C_), (W_all.shape, (E_, R_, C_))
    raw1 = np.array(en.tensors[0].data).reshape(E_, R_, -1).copy()
    raw2 = (np.array(en.tensors[1].data).reshape(int(en.tensors[1].shape[2]), R_, -1).copy()
            if len(en.tensors) > 1 else None)
    nE = E_ if esperti is None else min(esperti, E_)
    per_K: dict = {}
    for e in range(nE):
        per_K.setdefault(en.n_planes_of(e), []).append(e)
    tot = {k: 0.0 for k in ("H0", "H1", "HW", "O0", "O1", "OW", "F0", "F1", "FW")}
    righe_acc, esp_cambiati, corr = 0, 0, []

    for K, lista in sorted(per_K.items()):
        for i in range(0, len(lista), lotto):
            ee = lista[i:i + lotto]
            Ts, Ss, Hs, Hr = [], [], [], []
            for e in ee:
                tt = src.load(TensorKey(L, e, "down"))
                assert not tt.rotated, "ffn_down_exps expected TQ1_0 (plain basis)"
                assert tt.planes.shape[0] == K, (L, e, tt.planes.shape, K)
                Ts.append(tt.planes)
                Ss.append(tt.scales.double())
                Hs.append(met.ottimizzazione(e))
                Hr.append(met.grezza(e))
            T = torch.stack(Ts).to(dev)
            S = torch.stack(Ss).to(dev, dt)
            H = torch.stack(Hs).to(dev, dt)
            W = torch.from_numpy(np.ascontiguousarray(W_all[ee])).to(dev, dt)
            Q0 = CP.ricostruisci_lotto(T, S)
            # permutation / orientation guard: the host must already be a (coarse) copy of the teacher
            for j in range(len(ee)):
                a, b = W[j].double().flatten(), Q0[j].double().flatten()
                corr.append(float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30)))
            Ha = (H + nu2 * torch.eye(C_, dtype=dt, device=dev))
            if init in ("ospite", "migliore"):
                T2, S2, _ = CP.jarq_lotto(W, H, T, S, giri=giri, nu2=nu2)      # rows worse than host -> host
            else:
                T2, S2 = T, S
            if init in ("rtn", "migliore"):
                # second start from a fresh ternarization of the teacher; per row keep whichever of the two ends
                # lower (the host-start result already is <= host row by row; the rtn result must beat it)
                Tr, Sr = init_rtn(W, K, S.shape[-1])
                Tr, Sr, _ = CP.jarq_lotto(W, H, Tr, Sr.to(dt), giri=giri, nu2=nu2)
                f_cur = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha)
                f_r = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(Tr, Sr), Ha)
                m = (f_r < f_cur).view(len(ee), 1, R_, 1)
                T2 = torch.where(m, Tr, T2); S2 = torch.where(m, Sr, S2)
            Q1 = CP.ricostruisci_lotto(T2, S2)
            Hrd = torch.stack(Hr).to(dev, F64)
            wn = torch.tensor([met.n[e] for e in ee], dtype=F64, device=dev)
            tot["H0"] += float((wn * _err(W, Q0, Hrd)).sum()); tot["H1"] += float((wn * _err(W, Q1, Hrd)).sum())
            tot["HW"] += float((wn * _err(W, torch.zeros_like(W), Hrd)).sum())
            tot["O0"] += float(_err(W, Q0, Ha).sum()); tot["O1"] += float(_err(W, Q1, Ha).sum())
            tot["OW"] += float(_err(W, torch.zeros_like(W), Ha).sum())
            tot["F0"] += float(((W - Q0).double() ** 2).sum()); tot["F1"] += float(((W - Q1).double() ** 2).sum())
            tot["FW"] += float((W.double() ** 2).sum())
            camb = (T2 != T).any(1).any(-1) | (S2.to(torch.float16) != S.to(torch.float16)).any(1).any(-1)  # [B, R]
            righe_acc += int(camb.sum())
            T2c, S2c, cc = T2.cpu(), S2.cpu().to(torch.float16), camb.cpu().numpy()
            for j, e in enumerate(ee):
                if not cc[j].any():
                    continue                                        # expert untouched: host bytes kept
                esp_cambiati += 1
                for k in range(K):
                    Tk = T2c[j, k].numpy().astype(np.int8)
                    Sk = S2c[j, k].numpy()
                    b = gp.encode_rows(Tk, Sk, tid)
                    tp, sp = gp.decode_rows(b, tid, C_)              # bit-exact round trip
                    assert np.array_equal(tp, Tk) and np.array_equal(sp.view(np.uint16), Sk.view(np.uint16)), (L, e, k)
                    dst = raw1 if k == 0 else raw2
                    slot = e if k == 0 else en.slot(1, e)
                    dst[slot][cc[j]] = b[cc[j]]                     # only accepted rows change; the rest keep host bytes
    np.save(td / f"blk.{L}.ffn_down_exps.weight.npy", raw1)
    if raw2 is not None:
        np.save(td / f"blk.{L}.ffn_down_exps2.weight.npy", raw2)
    cmin, cmed = (float(np.min(corr)), float(np.median(corr))) if corr else (float("nan"), float("nan"))
    riga = {"strato": L, "famiglia": "down", "esperti": nE, "esperti_cambiati": esp_cambiati,
            "righe_accettate": righe_acc, "righe_totali": nE * R_,
            "err_H_rel_prima": tot["H0"] / max(tot["HW"], 1e-300), "err_H_rel_dopo": tot["H1"] / max(tot["HW"], 1e-300),
            "err_Hopt_rel_prima": tot["O0"] / max(tot["OW"], 1e-300),
            "err_Hopt_rel_dopo": tot["O1"] / max(tot["OW"], 1e-300),
            "err_F_rel_prima": (tot["F0"] / max(tot["FW"], 1e-300)) ** .5,
            "err_F_rel_dopo": (tot["F1"] / max(tot["FW"], 1e-300)) ** .5,
            "corr_ospite_bf16_min": cmin, "corr_ospite_bf16_mediana": cmed,
            "piani_per_esperto": {str(k): len(v) for k, v in per_K.items()},
            "metrica": f"Hdown_{L:02d} (media, flusso teacher), ritiro n0={met.ritiro:g}, nu2={nu2:g}",
            "giri": giri, "init": init, "secondi": round(time.time() - t0, 1)}
    riga["riduzione_err_H"] = 1 - riga["err_H_rel_dopo"] / max(riga["err_H_rel_prima"], 1e-300)
    if reg is not None:
        reg.write(json.dumps(riga) + "\n"); reg.flush()
    log(f"L{L} down: |W-Q|_H^2/|W|_H^2 {riga['err_H_rel_prima']:.5f} -> {riga['err_H_rel_dopo']:.5f} "
        f"(-{100 * riga['riduzione_err_H']:.1f} %), opt {riga['err_Hopt_rel_prima']:.5f} -> {riga['err_Hopt_rel_dopo']:.5f}, "
        f"F {riga['err_F_rel_prima']:.4f} -> {riga['err_F_rel_dopo']:.4f}, righe {righe_acc}/{nE * R_}, "
        f"corr min {cmin:.3f} [{riga['secondi']} s]")
    if corr and cmed < 0.5:
        raise SystemExit(f"L{L}: host/teacher median correlation {cmed:.3f} < 0.5: wrong orientation or permutation")
    return riga


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="host GGUF whose ffn_down_exps planes are re-ternarized")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--hessiane-down", default=HDOWN_DIR, help="DIR with Hdown_LL.npy + Hdown_LL_conta.npy")
    ap.add_argument("--ritiro", type=float, default=512.0,
                    help="shrinkage pseudo-count n0 toward the layer mean H (0 = raw per-expert H)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=16)
    ap.add_argument("--giri", type=int, default=3)
    ap.add_argument("--nu2", type=float, default=1e-4)
    ap.add_argument("--thread", type=int, default=8)
    ap.add_argument("--init", default="ospite", choices=["ospite", "rtn", "migliore"],
                    help="ospite: JARQ from the host codes (default); rtn: JARQ from a fresh absmean ternarization; "
                         "migliore: both, best per row (always <= host row by row)")
    ap.add_argument("--esperti", type=int, default=None, help="(probes) only the first N experts")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.thread)
    dev = torch.device(a.device)
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    os.environ.setdefault("LEVE_DT", "float64" if dt == torch.float64 else "float32")
    sys.path.insert(0, CP.RAFF)
    sys.path.insert(0, SORGENTE_DIR)
    import comune as CM
    from sorgente import SorgenteHF, bf16_a_f32
    hf = SorgenteHF()
    lo, _, hi = a.strati.partition("-")
    strati = list(range(int(lo), int(hi or lo) + 1))
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    (out / "comando.json").write_text(json.dumps({"argv": sys.argv, "args": vars(a)}, indent=1))
    with open(out / "registro.jsonl", "a") as reg:
        for L in strati:
            if (td / f"blk.{L}.ffn_down_exps.weight.npy").exists():
                log(f"L{L}: already done, skipped")
                continue
            u16, sha = hf.leggi_hf(f"model.language_model.layers.{L}.mlp.experts.down_proj")    # [E, 2048, 512]
            W_all = bf16_a_f32(u16)
            src = CM.gp.GGUFPlaneSource(a.gguf, layers=[L], bases=["ffn_down_exps"])
            met = MetricaDown(L, a.hessiane_down, a.ritiro)
            ternarizza_strato(L, src, W_all, met, dev, dt, a.lotto, td, reg, giri=a.giri, nu2=a.nu2,
                              esperti=a.esperti, init=a.init)
            del u16, W_all, src, met
    log("FINE ternarizza down", out)


if __name__ == "__main__":
    main()
