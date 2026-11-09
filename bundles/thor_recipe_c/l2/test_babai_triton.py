#!/usr/bin/env python3
"""Collaudo di cardan_babai_triton_gpu: ciclo colonne Triton contro il ciclo torch di cardan_piega_gpu.babai_lotto.
Su CPU: TRITON_INTERPRET=1 python3 test_babai_triton.py ; su GPU: python3 test_babai_triton.py --device cuda."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardan_piega_gpu as CP  # noqa: E402
import cardan_babai_triton_gpu as BT  # noqa: E402


def colonne_torch(y, Rch, A):
    B, R, G = y.shape
    dt = y.dtype
    z = torch.empty_like(y)
    idx = torch.empty(B, R, G, dtype=torch.long, device=y.device)
    for l in range(G - 1, -1, -1):
        if l < G - 1:
            corr = ((z[:, :, l + 1:] - y[:, :, l + 1:]) * Rch[:, l, l + 1:].unsqueeze(1)).sum(-1) / Rch[:, l, l].unsqueeze(1)
        else:
            corr = torch.zeros(B, R, dtype=dt, device=y.device)
        v = y[:, :, l] - corr
        j = (A - v.unsqueeze(-1)).abs().argmin(-1)
        idx[:, :, l] = j
        z[:, :, l] = A.gather(-1, j.unsqueeze(-1)).squeeze(-1)
    return z, idx


def caso(B, R, G, K, dt, dev, seme):
    g = torch.Generator().manual_seed(seme)
    M = torch.randn(B, G, 3 * G, generator=g, dtype=torch.float64)
    H = M @ M.transpose(-1, -2) / (3 * G) + 1e-3 * torch.eye(G, dtype=torch.float64)
    Rch = torch.linalg.cholesky(H).transpose(-1, -2).contiguous()
    S = torch.rand(B, K, R, generator=g, dtype=torch.float64) * 0.1
    tab = CP._tab(K, "cpu").double()
    A = torch.einsum("bkr,ak->bra", S, tab)
    y = torch.randn(B, R, G, generator=g, dtype=torch.float64) * 0.1
    return [x.to(dev, dt).contiguous() for x in (y, Rch, A)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grande", action="store_true", help="forme vere: B=16 R=2048 G=128/256")
    a = ap.parse_args()
    dev = torch.device(a.device)
    forme = [(2, 5, 8, 1), (3, 17, 16, 2), (1, 9, 12, 3), (2, 6, 32, 4)]
    if a.grande:
        forme = [(16, 2048, 128, 1), (16, 2048, 128, 2), (16, 2048, 256, 1), (8, 4096, 128, 3)]
    ok = True
    for dt in (torch.float64, torch.float32):
        for i, (B, R, G, K) in enumerate(forme):
            y, Rch, A = caso(B, R, G, K, dt, dev, 100 + i)
            t0 = time.time(); z0, i0 = colonne_torch(y, Rch, A)
            if dev.type == "cuda": torch.cuda.synchronize()
            t1 = time.time(); z1, i1 = BT.babai_colonne(y, Rch, A)
            if dev.type == "cuda": torch.cuda.synchronize()
            t2 = time.time()
            if a.grande:                                                     # tempo a regime (secondo lancio)
                t1 = time.time(); z1, i1 = BT.babai_colonne(y, Rch, A)
                if dev.type == "cuda": torch.cuda.synchronize()
                t2 = time.time()
            div = (i0 != i1).float().mean().item()
            dz = (z0 - z1).abs().max().item()
            soglia = 0 if dt == torch.float64 else 1e-3
            esito = div <= soglia and (div > 0 or dz == 0)
            ok &= esito
            print(f"{str(dt):14s} B{B} R{R} G{G} K{K}: idx diversi {div:.2e}, max|dz| {dz:.1e}, torch {t1 - t0:.3f} s, "
                  f"triton {t2 - t1:.3f} s {'OK' if esito else 'NO'}", flush=True)
    print("TUTTO OK" if ok else "DIFFERENZE OLTRE SOGLIA")
    return 0 if ok else 1


if __name__ == "__main__" and os.environ.get("TEST_LOTTO_INTERO") != "1":
    sys.exit(main())


def test_lotto_intero(dev="cpu"):
    """babai_lotto completo (accettazione, T, cambi) originale contro Triton, dati sintetici piccoli, fp64 e fp32."""
    ok = True
    for dt in (torch.float64, torch.float32):
        g = torch.Generator().manual_seed(7)
        B, K, R, C, ng = 2, 2, 6, 32, 2
        M = torch.randn(B, C, 3 * C, generator=g, dtype=torch.float64)
        H = (M @ M.transpose(-1, -2) / (3 * C) + 1e-3 * torch.eye(C, dtype=torch.float64)).to(dev, dt)
        T = torch.randint(-1, 2, (B, K, R, C), generator=g).to(torch.int8).to(dev)
        S = CP._fp16(torch.rand(B, K, R, ng, generator=g, dtype=torch.float64) * 0.1).to(dev, dt)
        W = CP.ricostruisci_lotto(T, S) + 0.03 * torch.randn(B, R, C, generator=g, dtype=torch.float64).to(dev, dt)
        T0, c0 = CP.babai_lotto(W, H, T, S)
        T1, c1 = BT.babai_lotto(W, H, T, S)
        e = torch.equal(T0, T1) and c0 == c1
        ok &= e
        print(f"babai_lotto {dt}: cambi {c0} vs {c1}, T uguali {torch.equal(T0, T1)} {'OK' if e else 'NO'}")
    return ok


if __name__ == "__main__" and os.environ.get("TEST_LOTTO_INTERO") == "1":
    sys.exit(0 if test_lotto_intero() else 1)
