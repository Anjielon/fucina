#!/usr/bin/env python3
"""Test of strumenti/gdn_qat.py against the fp64 reference (copy of transformers' fallback with dtype fp64, from
strumenti/prova_gdn_veloce.py): output and dq, dk, dv, dg, dbeta, for gdn3 (asserted: error <= 2 x the error of v5 fp32
vs fp64, the criterion of the GDN v5 gate) and gdn1 (reported).
CPU: TRITON_INTERPRET=1 HIP_VISIBLE_DEVICES= python3 e2e_test_gdn_qat.py     GPU: --gpu [FILE.json] (B=2 T=4096 H=32)"""
import json
import os
import sys
from pathlib import Path

if "--gpu" not in sys.argv:
    os.environ.setdefault("TRITON_INTERPRET", "1")
QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(QUI / "strumenti"))
import torch  # noqa: E402

import e2e_studente  # noqa: E402,F401  (interpreter patch for numpy 2)
import e2e_kernel_qat as K  # noqa: E402
import prova_gdn_veloce as P  # noqa: E402
from strumenti import gdn_qat as GQ  # noqa: E402


def dati(B, T, H, D, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, T, H, D, device=dev, generator=g)
    k = torch.randn(B, T, H, D, device=dev, generator=g)
    v = torch.randn(B, T, H, D, device=dev, generator=g)
    gg = -torch.nn.functional.softplus(torch.randn(B, T, H, device=dev, generator=g)) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev, generator=g))
    do = torch.randn(B, T, H, D, device=dev, generator=g) * 1e-6
    return q, k, v, gg, beta, do


def corre(f, xs, do):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    o, _ = f(*xs)
    o.backward(do.to(o.dtype))
    return [o.detach().double()] + [x.grad.double() for x in xs]


def main():
    gpu = "--gpu" in sys.argv
    dev = "cuda" if gpu else "cpu"
    B, T, H, D = (2, 4096, 32, 128) if gpu else (1, 130, 2, 128)
    q, k, v, gg, beta, do = dati(B, T, H, D, dev)
    xs = [q, k, v, gg, beta]
    rif = corre(lambda *a: P.ripiego_dtype(*a, use_qk_l2norm_in_kernel=True, dt=torch.float64),
                [x.double() for x in xs], do.double())
    nomi = ["o", "dq", "dk", "dv", "dg", "dbeta"]
    out = {}

    def errori(r):
        return {n: float((a - b).abs().max() / b.abs().max()) for n, a, b in zip(nomi, r, rif)}
    K.imposta("")
    out["v5_fp32"] = errori(corre(lambda *a: GQ.V5.chunk_gated_delta_rule_veloce(*a, use_qk_l2norm_in_kernel=True), xs, do))
    print("v5", out["v5_fp32"], flush=True)
    for voce in ("gdn3", "gdn1"):
        K.imposta(voce)
        out[voce] = errori(corre(lambda *a: GQ.chunk_gated_delta_rule_qat(*a, use_qk_l2norm_in_kernel=True), xs, do))
        print(voce, out[voce], flush=True)
    K.imposta("")
    if gpu:                                   # time of forward + backward, median of 5 (after a warm-up)
        import time
        tempi = {}
        for voce, f in (("v5", GQ.V5.chunk_gated_delta_rule_veloce), ("gdn3", GQ.chunk_gated_delta_rule_qat),
                        ("gdn1", GQ.chunk_gated_delta_rule_qat), ("gdn3_triton", GQ.chunk_gated_delta_rule_qat),
                        ("gdn1_triton", GQ.chunk_gated_delta_rule_qat), ("gdn3_ricT", GQ.chunk_gated_delta_rule_qat)):
            K.imposta("" if voce == "v5" else voce.split("_")[0])
            GQ._BMM["via"] = "triton" if voce.endswith("triton") else "blas"
            GQ._RIC["via"] = "triton" if voce.endswith("ricT") else "ibrida"
            ts = []
            for r in range(6):
                torch.cuda.synchronize()
                t0 = time.time()
                corre(lambda *a: f(*a, use_qk_l2norm_in_kernel=True), xs, do)
                torch.cuda.synchronize()
                ts.append(time.time() - t0)
            tempi[voce] = {"ms": sorted(ts[1:])[2] * 1e3, "us_token": sorted(ts[1:])[2] * 1e6 / (B * T)}
            print("tempo", voce, tempi[voce], flush=True)
        K.imposta("")
        out["tempi_avanti_indietro"] = tempi
    ok = all(out["gdn3"][n] <= 2 * out["v5_fp32"][n] + 1e-7 for n in nomi)
    out["gdn3_entro_2x_v5"] = ok
    if len(sys.argv) > 2:
        Path(sys.argv[-1]).write_text(json.dumps(out, indent=1))
    assert ok, out
    print("G1 ok")


if __name__ == "__main__":
    main()
