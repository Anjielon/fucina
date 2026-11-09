#!/usr/bin/env python3
"""cardan_costruisci_gpu.py — 21/10 (Angelo: «versione GPU perfetta e veloce di tutto»). Base CARDAN App. A.1 (arXiv
2609.21137, Eq. 5-6) come cardan_costruisci.py + cardan_base.py, stessa uscita (npz + info.json + misure.json), con:

  1 dequantizzazione dei piani, deruota (FWHT 1024) e residui R_e sulla GPU (fp64 per la ricostruzione, R_e fp32 come
    l'originale);
  2 Gram sbiancata RISCRITTA in forma equivalente esatta:
        originale  G = sum_e (R^T R_e^T)(R^T R_e^T)^T            (2 GEMM 2048x2048x512 per esperto)
        qui        G = R^T S R,  S = sum_e R_e^T R_e = Rcat^T Rcat   (1 GEMM su [256*512, 2048] a pezzi, poi 2 GEMM 2048^3)
     fp64 sulla GPU (rocBLAS dgemm, nessun rocSOLVER);
  3 cholesky / eigh / solve_triangular / inv (matrici 2048x2048 e 16x16) in fp64 sulla CPU: rocSOLVER su gfx1151 non e'
    affidabile (feedback_rocsolver_gfx1151_inaffidabile) e su queste taglie la CPU impiega ~1 s;
  4 coefficienti C_e = R_e Sigma D (D^T Sigma D)^-1 in fp64 sulla GPU, a lotto unico;
  5 misure E0/E1 in forma chiusa equivalente (tr(H S), termini incrociati), invece di (E Sigma)*E per esperto.
Le differenze dall'originale sono solo di arrotondamento (ordine delle somme): il collaudo confronta D16/C16 al byte e le
riduzioni d'errore con cardan_costruisci.py sugli stessi ingressi. Uso identico a cardan_costruisci.py (+ --device)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
import torch  # noqa: E402

F64 = torch.float64


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--gguf", default="@@ROOT:models@@/gguf/tony-tern/PV12P/Thor-T-pv12p.gguf")
    ap.add_argument("--hdir", default="@@ROOT:models@@/gguf/tony-tern/hessiane-densi-grande")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thread", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fili-lettura", type=int, default=8)
    ap.add_argument("--pezzo", type=int, default=32, help="esperti per GEMM della Gram (memoria GPU)")
    a = ap.parse_args()
    torch.set_num_threads(a.thread)
    dev = torch.device(a.device)
    from gguf import GGUFReader  # noqa: F401  (stesso ambiente dell'originale)
    from sorgente import SorgenteHF, bf16_a_f32
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina"); sys.path.insert(0, "@@ROOT:forge@@/thor_v2/leve_g4")
    import comune as CMR
    from rd_ternary.formato import TensorKey, fwht
    import cardan_base as CB
    srcp = CMR.gp.GGUFPlaneSource(a.gguf, bases=["ffn_gate_exps", "ffn_up_exps"])
    segni = torch.from_numpy(CMR.gp.sign1024().astype(np.float32)).to(dev, F64)

    def deruota(Wp):                                   # = comune.deruota, sul device
        y = fwht(Wp, 1024)
        return (y.reshape(*Wp.shape[:-1], Wp.shape[-1] // 1024, 1024) * segni).reshape(Wp.shape)

    lo, _, hi = a.strati.partition("-")
    strati = list(range(int(lo), int(hi or lo) + 1))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = SorgenteHF()
    rk = a.rango
    misure = json.loads((out / "misure.json").read_text()) if (out / "misure.json").exists() else {}
    for L in strati:
        f_npz = out / f"strato_{L:02d}.npz"
        if f_npz.exists() and str(L) in misure:
            continue
        t0 = time.time()
        H = torch.from_numpy(np.load(f"{a.hdir}/H_ffn_{L:02d}.npy").astype(np.float64))
        H = H / torch.trace(H) * H.shape[0]
        Rl = CB.chol_sigma(H)                                                    # CPU fp64 (come l'originale)
        Hg, Rlg = H.to(dev), Rl.to(dev)
        u16, _ = src.leggi_hf(f"model.language_model.layers.{L}.mlp.experts.gate_up_proj")
        arrs, m = {}, {}
        for fam, sl in (("gate", slice(0, 512)), ("up", slice(512, 1024))):
            Rs = torch.empty(256, 512, 2048, dtype=torch.float32, device=dev)
            with ThreadPoolExecutor(a.fili_lettura) as pool:      # lettura dei piani in parallelo (I/O + numpy)
                tqs = list(pool.map(lambda e: srcp.load(TensorKey(L, e, fam)), range(256)))
            for e0 in range(256):
                tq = tqs[e0]
                P_, S_ = tq.planes.to(dev), tq.scales.to(dev, F64)
                K_, R_, C_ = P_.shape; ng_ = S_.shape[-1]
                Wp = (P_.to(F64).reshape(K_, R_, ng_, C_ // ng_) * S_.unsqueeze(-1)).sum(0).reshape(R_, C_)
                Q = deruota(Wp) if tq.rotated else Wp
                W = torch.from_numpy(bf16_a_f32(u16[e0, sl]).astype(np.float32)).to(dev)
                Rs[e0] = W - Q.float()
            S = torch.zeros(2048, 2048, dtype=F64, device=dev)
            for i in range(0, 256, a.pezzo):
                Rb = Rs[i:i + a.pezzo].to(F64).reshape(-1, 2048)
                S += Rb.T @ Rb
            E0 = float((Hg * S).sum())                                           # sum_e tr(R_e H R_e^T)
            G = (Rlg.T @ S @ Rlg).cpu()
            D = CB.base_condivisa(G, Rl, rk)                                     # CPU fp64: eigh + solve_triangular
            D16 = CB.fp16(D)
            SD = H @ D16.to(F64)
            M = D16.to(F64).T @ SD
            M = M + 1e-12 * torch.trace(M) / M.shape[0] * torch.eye(M.shape[0], dtype=F64)
            Minv = torch.linalg.inv(M)                                           # 16x16, CPU fp64
            SDg, Minvg = SD.to(dev), Minv.to(dev)
            C16 = torch.empty(256, 512, rk, dtype=torch.float16)
            x1 = x2 = 0.0
            Mg = (D16.to(F64).T @ H @ D16.to(F64)).to(dev)                       # D^T Sigma D senza ridge (obiettivo)
            for i in range(0, 256, a.pezzo):
                RSD = Rs[i:i + a.pezzo].to(F64) @ SDg                            # [n, 512, r]
                Cn = CB.fp16(RSD @ Minvg)
                C16[i:i + a.pezzo] = Cn.to(torch.float16).cpu()
                x1 += float((Cn * RSD).sum())                                    # sum tr(Cn D^T Sigma R_e^T)
                x2 += float(((Cn @ Mg) * Cn).sum())                              # sum tr(Cn D^T Sigma D Cn^T)
            E1 = E0 - 2 * x1 + x2
            del Rs, S, tqs
            arrs[f"blk.{L}.ffn_{fam}_lrd.weight"] = D16.T.contiguous().to(torch.float16).numpy()
            arrs[f"blk.{L}.ffn_{fam}_lrc.weight"] = C16.numpy()
            m[fam] = {"err_rel_prima": E0 ** .5, "riduzione_errore_pesato": 1 - E1 / E0}
        tmp = out / f"strato_{L:02d}.tmp.npz"
        np.savez(tmp, **{k: v.view(np.uint8) for k, v in arrs.items()})
        tmp.replace(f_npz)
        misure[str(L)] = {"rango": rk, **m, "secondi": round(time.time() - t0, 1), "versione": "gpu"}
        aggiorna_json(out / "misure.json", lambda d, L=L: {**d, str(L): misure[str(L)]})    # D5: unione, non sovrascrittura
        log(f"strato {L}: riduzione errore pesato gate {100*m['gate']['riduzione_errore_pesato']:.2f} % "
            f"up {100*m['up']['riduzione_errore_pesato']:.2f} % [{time.time()-t0:.0f}s]")
    info = {"tensori": {}}
    for L in strati:
        for fam in ("gate", "up"):
            info["tensori"][f"blk.{L}.ffn_{fam}_lrd.weight"] = {"type": 1, "ne": [2048, rk]}
            info["tensori"][f"blk.{L}.ffn_{fam}_lrc.weight"] = {"type": 1, "ne": [rk, 512, 256]}
    aggiorna_json(out / "info.json", lambda d: {**d, "tensori": {**d.get("tensori", {}), **info["tensori"]}})
    log("FINE", out)


def aggiorna_json(p: Path, f):
    """D5 (giro 6): lettura-modifica-scrittura sotto flock (piu' processi --strati sulla stessa --out) e
    pubblicazione atomica (tmp + fsync + replace): nessuna voce di un altro processo persa, nessun JSON a meta'."""
    import fcntl
    import os
    with open(p.with_name(p.name + ".lock"), "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {}
        tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
        with open(tmp, "w") as fo:
            fo.write(json.dumps(f(d), indent=1)); fo.flush(); os.fsync(fo.fileno())
        os.replace(tmp, p)


if __name__ == "__main__":
    main()
