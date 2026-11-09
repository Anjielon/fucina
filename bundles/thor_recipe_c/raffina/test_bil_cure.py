"""Test CPU della cura BIL (9/10): il livello piu' vicino via searchsorted e' ESATTAMENTE quello della forza bruta di
leve_g4/bil_gptq.py, GPTQ bilanciato = riferimento a forza bruta al bit, trit che ricostruiscono il livello scelto,
JARQ in serie mai peggiore di BIL per riga, K=4 (famiglie col piano 4 di PV12P).
uso: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python -m pytest -q test_bil_cure.py
"""
import os
import sys

os.environ.setdefault("LEVE_DT", "float64")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import solutori as SO  # noqa: E402

torch.manual_seed(0)


def _H(C, seed=1):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * C, C, generator=g, dtype=torch.float64) @ torch.diag(torch.linspace(0.2, 2.0, C, dtype=torch.float64))
    H = X.T @ X
    return H / torch.diagonal(H).mean()


def _ref_gptq_bil(W, H, K, ng, damp=0.01):
    """Riferimento a FORZA BRUTA (argmin su tutti i 3^K livelli, come bil_gptq.py del 7/10), stessa Cholesky."""
    R, C = W.shape
    G = C // ng
    tab = SO.tabella_trit(K)
    _, U, _ = SO.cholesky_smorzata(H, percdamp=damp)
    Wk = W.clone()
    T = torch.zeros(K, R, C, dtype=torch.int8)
    S = torch.zeros(K, R, ng, dtype=torch.float64)
    m = (3 ** K - 1) / 2
    for g in range(ng):
        a, b = g * G, (g + 1) * G
        blk = Wk[:, a:b]
        amax = blk.abs().amax(1).clamp_min(1e-30)
        bD, be = None, None
        for f in SO.BIL_FATTORI:
            D = f * amax / m
            lev = SO.bil_scale(D, K) @ tab.double().T
            q = lev.gather(1, (lev[:, None, :] - blk[:, :, None]).abs().argmin(2))
            e = ((q - blk) ** 2).sum(1)
            if bD is None:
                bD, be = D.clone(), e
            else:
                u = e < be
                bD[u], be[u] = D[u], e[u]
        Sg = SO.bil_scale(bD, K)
        S[:, :, g] = Sg.T
        lev = Sg @ tab.double().T
        Err = torch.zeros(R, G, dtype=torch.float64)
        for jj in range(G):
            j = a + jj
            w = Wk[:, j]
            i = (lev - w[:, None]).abs().argmin(1)
            q = lev.gather(1, i[:, None])[:, 0]
            T[:, :, j] = tab[i].T
            e = (w - q) / U[j, j]
            Wk[:, j + 1:b] -= e[:, None] * U[j, j + 1:b][None, :]
            Err[:, jj] = e
        if b < C:
            Wk[:, b:] -= Err @ U[a:b, b:]
    return T, S


def test_vicino_uguale_forza_bruta():
    for K in (2, 3, 4):
        tab = SO.tabella_trit(K)
        D = torch.rand(50, dtype=torch.float64) + 0.01
        Ls, _ = SO.bil_livelli(SO.bil_scale(D, K), tab)
        w = torch.randn(50, 300, dtype=torch.float64) * D[:, None] * 3 ** K / 1.5
        q, _ = SO.bil_vicino(w, Ls)
        lev = SO.bil_scale(D, K) @ tab.double().T
        qb = lev.gather(1, (lev[:, None, :] - w[:, :, None]).abs().argmin(2))
        assert torch.equal((q - w).abs(), (qb - w).abs())


def test_gptq_bilanciato_uguale_riferimento_e_trit_coerenti():
    R, C, ng = 24, 256, 2
    W = torch.randn(R, C, dtype=torch.float64) * 0.02
    H = _H(C)
    T, S, _ = SO.gptq_bilanciato(W, H, 3, ng)
    Tr, Sr = _ref_gptq_bil(W, H, 3, ng)
    assert torch.equal(S, Sr)
    assert (T == Tr).float().mean() > 0.9999
    assert torch.equal(S, SO.fp16(S))
    p = torch.tensor([9.0, 3.0, 1.0], dtype=torch.float64)
    # ogni scala e' fp16(3^(K-1-k) D) arrotondata da sola (come bil_gptq.py): rapporto 9:3:1 entro mezzo ulp fp16
    assert torch.allclose(S, S[2:3] * p.view(3, 1, 1), rtol=1e-3, atol=0)
    assert T.abs().max() <= 1


def test_bil_meglio_di_rtn_e_jarq_in_serie_non_peggiora():
    R, C, ng = 32, 256, 2
    W = torch.randn(R, C, dtype=torch.float64) * 0.02
    H = _H(C, 3)
    T, S, _ = SO.gptq_bilanciato(W, H, 3, ng)
    f_bil = SO.obiettivo(W, SO.ricostruisci(T, S), H)
    legame = torch.tensor([9.0, 3.0, 1.0], dtype=torch.float64)
    Tu, Su = SO.tied_init(W, torch.zeros_like(W), 3, ng, legame)
    f_rtn = SO.obiettivo(W, SO.ricostruisci(Tu, Su), H)
    assert f_bil.sum() < f_rtn.sum()
    T2, S2, cat = SO.jarq(W, H, T, S, [0, 1, 2], giri=3, nu2=1e-4, legame=legame)
    Ha = H + 1e-4 * torch.eye(C, dtype=torch.float64)
    f0 = SO.obiettivo(W, SO.ricostruisci(T, S), Ha)
    f2 = SO.obiettivo(W, SO.ricostruisci(T2, S2.double()), Ha)
    assert bool((f2 <= f0 * (1 + 1e-12)).all())


def test_K4_griglia_81_livelli():
    R, C, ng = 16, 256, 1
    W = torch.randn(R, C, dtype=torch.float64) * 0.02
    H = _H(C, 5)
    T4, S4, _ = SO.gptq_bilanciato(W, H, 4, ng)
    T3, S3, _ = SO.gptq_bilanciato(W, H, 3, ng)
    assert T4.shape == (4, R, C)
    e4 = SO.obiettivo(W, SO.ricostruisci(T4, S4), H).sum()
    e3 = SO.obiettivo(W, SO.ricostruisci(T3, S3), H).sum()
    assert e4 < 0.5 * e3                                              # un trit in piu' = passo 3x piu' fine
