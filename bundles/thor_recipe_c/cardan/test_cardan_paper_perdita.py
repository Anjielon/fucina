#!/usr/bin/env python3
"""CPU tests of cardan_paper_perdita (10/10 night): value against a direct formula of Eq. (3), gradient against
autograd, chunking invariance, Eq. (6) closed form against least squares."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cardan_paper_perdita as P  # noqa: E402


def caso(T=37, H=24, V=500, k=64, tau=2.0, lam=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    WS = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.3
    WT = WS + 0.05 * torch.randn(V, H, generator=g, dtype=torch.float64)
    hS = torch.randn(T, H, generator=g, dtype=torch.float64)
    hT = hS + 0.2 * torch.randn(T, H, generator=g, dtype=torch.float64)
    tgt = torch.randint(0, V, (T,), generator=g)
    ok = torch.ones(T, dtype=torch.bool); ok[-1] = False; ok[10] = False
    return WS, WT, hS, hT, tgt, ok, k, tau, lam


def riferimento(hS, WS, hT, WT, tgt, ok, k, tau, lam):
    zS, zT = hS @ WS.T, hT @ WT.T
    om = torch.topk(zT, k, -1).indices
    pT = torch.softmax(zT.gather(1, om) / tau, -1)
    lS = torch.log_softmax(zS.gather(1, om) / tau, -1)
    kd = (tau ** 2) * (pT * (pT.log() - lS)).sum(-1)
    ce = -torch.log_softmax(zS, -1).gather(1, tgt[:, None])[:, 0]
    return kd.mean() + lam * (ce * ok).sum() / ok.sum()


def test_valore_e_gradiente():
    WS, WT, hS, hT, tgt, ok, k, tau, lam = caso()
    for pezzo in (5, 37, 64):
        kd, ce, T, nv, g = P.perdita_paper(hS, WS, hT, WT, tgt, ok, tau, lam, k, pezzo, True,
                                           lambda h: h.double() @ WS.T, lambda h: h.double() @ WT.T,
                                           lambda q: q.double() @ WS)
        L = kd / T + lam * ce / nv
        h = hS.clone().requires_grad_(True)
        R = riferimento(h, WS, hT, WT, tgt, ok, k, tau, lam)
        R.backward()
        assert abs(L - float(R)) < 1e-6 * abs(float(R)), (L, float(R))  # fp32 inside
        assert torch.allclose(g.double(), h.grad, atol=1e-6), float((g.double() - h.grad).abs().max())


def test_tau1_topk_pieno_e_kl_piena():
    WS, WT, hS, hT, tgt, ok, _, _, _ = caso(V=80)
    kd, _, T, _, _ = P.perdita_paper(hS, WS, hT, WT, tgt, ok, 1.0, 0.0, 80, 16, False,
                                     lambda h: h.double() @ WS.T, lambda h: h.double() @ WT.T, lambda q: q @ WS)
    pT = torch.softmax(hT @ WT.T, -1)
    kl = (pT * (pT.log() - torch.log_softmax(hS @ WS.T, -1))).sum()
    assert abs(kd - float(kl)) < 1e-6 * abs(float(kl))


def test_init_c_chiusa():
    g = torch.Generator().manual_seed(1)
    H, I, r = 20, 12, 4
    X = torch.randn(300, H, generator=g, dtype=torch.float64)
    S = X.T @ X / 300
    D = torch.randn(H, r, generator=g, dtype=torch.float64)
    De = torch.randn(H, I, generator=g, dtype=torch.float64)
    C = P.init_c_chiusa(D, S, De)
    R = torch.linalg.cholesky(S)
    Cls = torch.linalg.lstsq(R.T @ D, R.T @ De).solution
    assert torch.allclose(C, Cls, atol=1e-9)


if __name__ == "__main__":
    test_valore_e_gradiente(); test_tau1_topk_pieno_e_kl_piena(); test_init_c_chiusa()
    print("test_cardan_paper_perdita: 3/3 OK")
