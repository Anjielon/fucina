#!/usr/bin/env python3
"""leve_g4/cardan_costruisci.py — 9/10: base ESATTA del lavoro (cardan_base.py, App. A.1) e ospite PV12P con i
secondi piani degli esperti; e' SOLO l'inizializzazione: il lavoro (Tab. 6) la fa seguire dalla distillazione (Eq. 3).
7/10: — leva #20 (CARDAN, arXiv 2609.21137 §3.1/§3.3, "training-free initialization") su R2B3m:
per ogni strato L e famiglia f in {gate, up}, base condivisa D_L,f (2048 x r) e coefficienti privati C_e (512 x r), in forma
chiusa sulle attivazioni (H = Hessiana d'ingresso del blocco FFN, hessiane-densi-grande/H_ffn_LL, la stessa per tutti gli
esperti dello strato: le Hessiane per esperto esistono solo per 3 strati):
    R_e = W_e(bf16) - Q_e(R2B3m)     N = sum_e (R_e H)^T (R_e H)     D = primi r autovettori di N
    C_e = R_e H D (D^T H D)^-1        (correzione: W_e ~ Q_e + C_e D^T)
Uscita: npz + info json per e1_scrivi_residui_gguf.scrivi (tensori F16 nuovi):
    blk.L.ffn_{gate,up}_lrd.weight  ne [2048, r]        (numpy (r, 2048): riga j = colonna j di D)
    blk.L.ffn_{gate,up}_lrc.weight  ne [r, 512, 256]    (numpy (256, 512, r) = C_e)
e misure (errore pesato relativo prima/dopo per famiglia e strato).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
import torch  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--gguf", default="@@ROOT:models@@/gguf/tony-tern/PV12P/Thor-T-pv12p.gguf",
                    help="9/10: ospite (default PV12P, base A); i suoi esperti a 2 piani entrano nel residuo")
    ap.add_argument("--hdir", default="@@ROOT:models@@/gguf/tony-tern/hessiane-densi-grande")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thread", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.thread)
    from gguf import GGUFReader
    from sorgente import SorgenteHF, bf16_a_f32
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina"); sys.path.insert(0, "@@ROOT:forge@@/thor_v2/leve_g4")
    import comune as CMR
    import cardan_base as CB
    from rd_ternary.formato import TensorKey
    srcp = CMR.gp.GGUFPlaneSource(a.gguf, bases=["ffn_gate_exps", "ffn_up_exps"])
    lo, _, hi = a.strati.partition("-")
    strati = list(range(int(lo), int(hi or lo) + 1))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = GGUFReader(a.gguf)
    tt = {t.name: t for t in r.tensors}
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
        u16, _ = src.leggi_hf(f"model.language_model.layers.{L}.mlp.experts.gate_up_proj")
        arrs, m = {}, {}
        for fam, sl in (("gate", slice(0, 512)), ("up", slice(512, 1024))):
            # 9/10: Q_e dall'ospite con TUTTI i suoi piani (PV12P: 2° piano exps2 degli esperti caldi, via mappa), poi la
            # base ESATTA di CARDAN App. A.1 (cardan_base.py: Gram sbiancata, D = R^-T U_r, C_e ai minimi quadrati).
            Rs = torch.empty(256, 512, 2048, dtype=torch.float32)
            for e0 in range(256):
                tq = srcp.load(TensorKey(L, e0, fam)); P_, S_ = tq.planes, tq.scales.double()
                K_, R_, C_ = P_.shape; ng_ = S_.shape[-1]
                Wp = sum((P_[k].double().reshape(R_, ng_, C_ // ng_) * S_[k].unsqueeze(-1)).reshape(R_, C_) for k in range(K_))
                Q = CMR.deruota(Wp) if tq.rotated else Wp
                Rs[e0] = torch.from_numpy(bf16_a_f32(u16[e0, sl]).astype(np.float32)) - Q.float()
            lotti = [Rs[i:i + 16] for i in range(0, 256, 16)]
            E0 = float(sum(((Rb.double() @ H) * Rb.double()).sum() for Rb in lotti))
            D16, coeff = CB.risolvi(lotti, H, rk)
            C16 = torch.cat([coeff(Rb) for Rb in lotti]).to(torch.float16)                  # (256, 512, r)
            E1 = float(sum(CB.obiettivo(Rs[i:i + 16], C16[i:i + 16], D16, H).sum() for i in range(0, 256, 16)))
            del Rs, lotti
            arrs[f"blk.{L}.ffn_{fam}_lrd.weight"] = D16.T.contiguous().to(torch.float16).numpy()   # (r, 2048)
            arrs[f"blk.{L}.ffn_{fam}_lrc.weight"] = C16.numpy()
            m[fam] = {"err_rel_prima": E0 ** .5, "riduzione_errore_pesato": 1 - E1 / E0}
        np.savez(f_npz, **{k: v.view(np.uint8) for k, v in arrs.items()})
        misure[str(L)] = {"rango": rk, **m, "secondi": round(time.time() - t0, 1)}
        (out / "misure.json").write_text(json.dumps(misure, indent=1))
        log(f"strato {L}: riduzione errore pesato gate {100*m['gate']['riduzione_errore_pesato']:.2f} % "
            f"up {100*m['up']['riduzione_errore_pesato']:.2f} % [{time.time()-t0:.0f}s]")
    # info per lo scrittore
    info = {"tensori": {}}
    for L in strati:
        for fam in ("gate", "up"):
            info["tensori"][f"blk.{L}.ffn_{fam}_lrd.weight"] = {"type": 1, "ne": [2048, rk]}
            info["tensori"][f"blk.{L}.ffn_{fam}_lrc.weight"] = {"type": 1, "ne": [rk, 512, 256]}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    log("FINE", out)


if __name__ == "__main__":
    main()
