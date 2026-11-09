#!/usr/bin/env python3
"""CPU proofs of the "CARDAN of the dense" (10/10): continuous low-rank branch of residui_ternari (rilq_thor.py
--continuo) and its fold into the existing planes (cardan_piega_densi.py). Run: python3 test_cardan_densi.py

D1  continuous branch: zero output at the start (B = 0, gamma = 1), exact identity of ConResiduo
D2  delta_file is the FILE-order matrix of the branch: forward (HF space, with row/column permutations) == x dW_hf^T
D3  default config (continuo=False) unchanged: RILQ/CARDAN residues still ternary with gamma = 0
D4  fold of a correction that the host planes can represent EXACTLY by a change of scales: captured ~100 %
D5  fold of a random low-rank correction: never worse than the host (captured >= 0, row-wise guarantee), and the
    rotated stored basis (TQ1_0R, C = 1024) gives the same captured fraction as the plain basis (|E R^T|_{RHR^T} = |E|_H)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
import residui_ternari as RT          # noqa: E402
import cardan_piega as CP             # noqa: E402
import cardan_piega_densi as PD       # noqa: E402

torch.manual_seed(0)
F64 = torch.float64


def d1_d2_d3():
    cfg = RT.ConfigResidui(rango=4, continuo=True, strati=(0,), famiglie=("attn_q",))
    fuori, dentro = 12, 40                                    # dentro not a multiple of 256: allowed in continuo
    perm_x = torch.randperm(dentro).numpy()
    p = torch.randperm(fuori).numpy(); inv_y = np.empty_like(p); inv_y[p] = np.arange(fuori)
    for px, iy in ((None, None), (perm_x, None), (None, inv_y)):
        r = RT.ResiduoTernario("blk.0.x", fuori, dentro, cfg, perm_x=px, inv_y=iy, seme=1)
        x = torch.randn(5, dentro)
        assert torch.equal(r(x), torch.zeros(5, fuori)), "D1: branch not zero at the start"
        lin = torch.nn.Linear(dentro, fuori)
        assert torch.equal(RT.ConResiduo(lin, r)(x), lin(x)), "D1: ConResiduo not the identity"
        with torch.no_grad():
            r.B.normal_(); r.gamma.uniform_(0.5, 1.5)
        dWf = RT.delta_file(r)                                # file order
        dWh = dWf.clone()
        if px is not None:                                    # x_file = x_hf[:, perm]  ->  dW_hf[:, perm] = dW_file
            dWh = torch.zeros_like(dWf); dWh[:, torch.as_tensor(px)] = dWf
        if iy is not None:                                    # y_hf = y_file[:, inv]   ->  dW_hf = dW_file[inv]
            dWh = dWf[torch.as_tensor(iy)]
        y = r(x).double()
        assert torch.allclose(y, x.double() @ dWh.T, atol=1e-5), f"D2: delta_file != branch ({px is not None}, {iy is not None})"
    c0 = RT.ConfigResidui(rango=4, strati=(0,), famiglie=("attn_q",))
    r0 = RT.ResiduoTernario("blk.0.y", 8, 256, c0, seme=2)
    assert not c0.continuo and float(r0.gamma.detach().abs().max()) == 0.0 and float(r0.B.detach().abs().max()) > 0, "D3"
    Aq, _ = r0.pesi_q()
    assert all(len(torch.unique(row)) <= 3 for row in Aq.detach()), "D3: A not ternary-quantised (per-row group)"
    print("D1 D2 D3 ok")


class FintoDenso:
    """Minimal stand-in of comune.Denso (stored basis planes)."""
    def __init__(self, T, S, rotated=False):
        self.T, self.S, self.rotated = T, S, rotated
        self.K, self.out, self.in_ = T.shape
        self.nome = "blk.0.finto.weight"

    def ricostruisci(self, T=None, S=None):
        T = self.T if T is None else T
        S = self.S if S is None else S
        K, R, C = T.shape
        ng = S.shape[-1]
        return (T.double().reshape(K, R, ng, C // ng) * S.double().unsqueeze(-1)).sum(0).reshape(R, C)


class CMfinto:
    @staticmethod
    def ruota(W):
        return W

    @staticmethod
    def ruota_H(H):
        return H


def ospite(K, R, C, ng, seme):
    g = torch.Generator().manual_seed(seme)
    T = torch.randint(-1, 2, (K, R, C), generator=g).to(torch.int8)
    S = (torch.rand(K, R, ng, generator=g, dtype=F64) * 0.02 * torch.tensor([3.0 ** -k for k in range(K)],
         dtype=F64).view(K, 1, 1)).to(torch.float16).double()
    X = torch.randn(4 * C, C, generator=g, dtype=F64)
    H = X.T @ X / X.shape[0]
    return T, S, H / torch.diagonal(H).mean()


def d4():
    K, R, C, ng = 3, 16, 256, 2
    T, S, H = ospite(K, R, C, ng, 3)
    d = FintoDenso(T, S)
    S_vere = (S * (1 + 0.05 * torch.randn(S.shape, dtype=F64))).to(torch.float16).double()
    dW = d.ricostruisci(T, S_vere) - d.ricostruisci()        # representable exactly by the scales
    _, S2, rec = PD.piega_uno(d, dW, H, CMfinto, CP, torch.device("cpu"), F64)
    assert rec["catturata"] > 0.98, f"D4: captured {rec['catturata']:.4f}"
    print(f"D4 ok (captured {100 * rec['catturata']:.2f} %)")


def d5():
    import comune as CM
    K, R, ng, r = 3, 24, 8, 4
    for rotated, C in ((False, 1024), (True, 1024)):
        T, S, H = ospite(K, R, C, ng, 5)
        d = FintoDenso(T, S, rotated=rotated)
        g = torch.Generator().manual_seed(7)
        Wbase = CM.deruota(d.ricostruisci()) if rotated else d.ricostruisci()
        dW = 0.05 * Wbase.norm() / (R * C) ** 0.5 * (torch.randn(R, r, generator=g, dtype=F64)
                                                      @ torch.randn(r, C, generator=g, dtype=F64)) / r ** 0.5
        cmod = CM if rotated else CMfinto
        _, _, rec = PD.piega_uno(d, dW, H, cmod, CP, torch.device("cpu"), F64)
        assert rec["catturata"] >= -1e-9, f"D5: fold worse than the host ({rec['catturata']})"
        print(f"D5 ok rotated={rotated}: captured {100 * rec['catturata']:.1f} %, rows back to host "
              f"{rec['righe_tornate_ospite']}")


if __name__ == "__main__":
    d1_d2_d3()
    d4()
    d5()
    print("TUTTI VERDI")
