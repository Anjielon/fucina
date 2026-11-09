"""Garanzie dei LAVORI verificate contro oracoli INDIPENDENTI (9/10, PROVA_LEVE_VS_PAPER.md).

Ogni test confronta la nostra implementazione con un riferimento scritto da un'altra strada matematica, non con una
copia dello stesso algoritmo:
  * GPTQ bilanciato (BIL) == OBQ con H^-1 aggiornata a mano (Frantar & Alistarh 2022, eq. 2-3; GPTQ 2210.17323 §3
    dimostra che l'aggiornamento di Cholesky e' la stessa cosa) -> stessi codici e stesse scale, al bit;
  * perdita GPTQ: tr(E H E^T) = sum_j ((w~_j - q_j) / U_jj)^2 (OBS, identita' esatta con la H smorzata);
  * JARQ eq. 6 / Alg. 1 riga 6: le scale congiunte == minimi quadrati di numpy sulla matrice di disegno esplicita;
  * JARQ Corollario 1: l'obiettivo (con la penalita' nu dei pesi) non sale mai sopra quello dell'ospite;
  * FOEM (2507.11017): la nostra forma «codice» == fasterquant del repository ufficiale (ramo ours_v2, trascritto
    riga per riga) con lo stesso quantizzatore.
uso: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python -m pytest -q test_garanzie_paper.py
"""
import os
import sys

os.environ.setdefault("LEVE_DT", "float64")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import solutori as SO  # noqa: E402

torch.set_num_threads(2)


def _H(C, seed=1, n=None):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n or 4 * C, C, generator=g, dtype=torch.float64)
    X = X @ (torch.eye(C, dtype=torch.float64) + 0.6 * torch.randn(C, C, generator=g, dtype=torch.float64) / C ** 0.5)
    H = X.T @ X
    return H / torch.diagonal(H).mean()


def _W(R, C, seed=2):
    g = torch.Generator().manual_seed(seed)
    return 0.05 * torch.randn(R, C, generator=g, dtype=torch.float64)


def _obq_riferimento(W, H, K, ng, damp=0.01):
    """OBQ in ordine naturale con H^-1 ESPLICITA e rimozione della colonna quantizzata (eq. 3 di OBQ), senza Cholesky.
    Scala del gruppo scelta come BIL all'inizio del gruppo, sui pesi gia' compensati."""
    R, C = W.shape
    G = C // ng
    tab = SO.tabella_trit(K)
    Hd = H + damp * torch.diagonal(H).mean() * torch.eye(C, dtype=torch.float64)
    Hinv = torch.linalg.inv(Hd)
    Wk = W.clone()
    T = torch.zeros(K, R, C, dtype=torch.int8)
    S = torch.zeros(K, R, ng, dtype=torch.float64)
    perdita = torch.zeros(R, dtype=torch.float64)
    for g in range(ng):
        a, b = g * G, (g + 1) * G
        D = SO.bil_scegli_D(Wk[:, a:b], K, tab)
        Sg = SO.bil_scale(D, K)
        S[:, :, g] = Sg.T
        Ls, perm = SO.bil_livelli(Sg, tab)
        for j in range(a, b):
            q, idx = SO.bil_vicino(Wk[:, j:j + 1], Ls)
            T[:, :, j] = tab[perm.gather(1, idx).squeeze(1)].T
            d = Hinv[j, j]
            err = (Wk[:, j] - q[:, 0])
            perdita += err ** 2 / d
            Wk[:, j:] -= (err / d).unsqueeze(1) * Hinv[j, j:].unsqueeze(0)       # delta = -(w-q)/[H^-1]jj * H^-1[j,:]
            Hinv = Hinv - torch.outer(Hinv[:, j], Hinv[j, :]) / d               # rimozione della colonna j (eq. 3)
    return T, S, perdita


@pytest.mark.parametrize("K", [2, 3, 4])
def test_gptq_bilanciato_uguale_obq_esplicito(K):
    R, C, ng = 6, 256, 2
    W, H = _W(R, C, 3 + K), _H(C, 5 + K)
    T, S, _ = SO.gptq_bilanciato(W, H, K, ng)
    T2, S2, _ = _obq_riferimento(W, H, K, ng)
    assert torch.equal(S, S2)
    assert torch.equal(T, T2)


def test_perdita_gptq_identita_obs():
    """tr((W-Q) Hd (W-Q)^T) = somma delle perdite OBS per colonna (esatta, con la H smorzata)."""
    R, C, ng, K = 5, 256, 2, 3
    W, H = _W(R, C, 11), _H(C, 13)
    T, S, perdita = _obq_riferimento(W, H, K, ng)
    T1, S1, _ = SO.gptq_bilanciato(W, H, K, ng)
    Q = SO.ricostruisci(T1, S1)
    Hd = H + 0.01 * torch.diagonal(H).mean() * torch.eye(C, dtype=torch.float64)
    E = W - Q
    assert torch.allclose(((E @ Hd) * E).sum(1), perdita, rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("legame", [None, (9.0, 3.0, 1.0)])
def test_scale_congiunte_uguale_lstsq(legame):
    R, C, ng, K = 3, 256, 2, 3
    W, H = _W(R, C, 21), _H(C, 23)
    g = torch.Generator().manual_seed(7)
    T = torch.randint(-1, 2, (K, R, C), generator=g).to(torch.int8)
    lg = None if legame is None else torch.tensor(legame, dtype=torch.float64)
    S = SO.scale_congiunte(W, H, T, torch.zeros_like(W), ng, legame=lg)
    G = C // ng
    Lh = np.linalg.cholesky(H.numpy())                                  # H = L L^T: ||e||_H = ||L^T e||_2
    for r in range(R):
        cols = []
        for k in range(K if lg is None else 1):
            for q in range(ng):
                v = np.zeros(C)
                if lg is None:
                    v[q * G:(q + 1) * G] = T[k, r, q * G:(q + 1) * G].numpy()
                else:
                    v[q * G:(q + 1) * G] = sum(legame[kk] * T[kk, r, q * G:(q + 1) * G].numpy() for kk in range(K))
                cols.append(Lh.T @ v)
        x = np.linalg.lstsq(np.stack(cols, 1), Lh.T @ W[r].numpy(), rcond=None)[0]
        if lg is None:
            ref = x.reshape(K, ng)
        else:
            ref = np.stack([legame[k] * x for k in range(K)])
        assert np.allclose(S[:, r].numpy(), ref, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_jarq_corollario1_non_sale_sopra_ospite(seed):
    R, C, ng, K = 8, 256, 2, 3
    W, H = _W(R, C, 30 + seed), _H(C, 40 + seed)
    leg = torch.tensor([9.0, 3.0, 1.0], dtype=torch.float64)
    T0, S0 = SO.tied_init(W, torch.zeros_like(W), K, ng, leg)
    nu2 = 1e-4
    T1, S1, cat = SO.jarq(W, H, T0, S0.double(), [0, 1, 2], giri=3, nu2=nu2, legame=leg)
    Ha = H + nu2 * torch.eye(C, dtype=torch.float64)
    f0 = SO.obiettivo(W, SO.ricostruisci(T0, S0.double()), Ha)
    f1 = SO.obiettivo(W, SO.ricostruisci(T1, S1.double()), Ha)
    assert bool((f1 <= f0 * (1 + 1e-12)).all())
    assert cat["finale"] <= cat["ospite"] * (1 + 1e-12)


# ── FOEM: fasterquant UFFICIALE (github Xingyu-Zheng/FOEM, LLM/weight-only/gptq.py, ours_v2 senza gptqv2), trascritto
#    riga per riga (righe 192-268), col quantizzatore sostituito dalla griglia bilanciata e find_params = bil_scegli_D.
def _foem_ufficiale(W, H, K, ng, beta, percdamp=0.01, blocksize=128):
    R, C = W.shape
    groupsize = C // ng
    tab = SO.tabella_trit(K)
    fp_weight = W.clone()
    W = W.clone()
    H = H.clone()
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(C)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    Q = torch.zeros_like(W)
    st = {}
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        fp_weight1 = fp_weight[:, i1:i2]
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            if (i1 + i) % groupsize == 0:                                   # find_params su W (non W1), come l'ufficiale
                D = SO.bil_scegli_D(W[:, (i1 + i):(i1 + i + groupsize)], K, tab)
                st["Ls"], st["perm"] = SO.bil_livelli(SO.bil_scale(D, K), tab)
            q = SO.bil_vicino(w.unsqueeze(1), st["Ls"])[0].flatten()
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] = W1[:, i:] - err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) \
                - (W1[:, i:] - fp_weight1[:, i:]) @ (Hinv1[i:, i:].t() @ Hinv1[i:, i:]) * beta
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        W[:, i2:] = W[:, i2:] - Err1.matmul(Hinv[i1:i2, i2:]) \
            - (W[:, i2:] - fp_weight[:, i2:]) @ (Hinv[i2:, i2:].t() @ Hinv[i2:, i2:]) * beta
    return Q


@pytest.mark.parametrize("beta", [0.0, 1e-3, 3e-2])
def test_foem_forma_codice_uguale_ufficiale(beta):
    R, C, ng, K = 6, 256, 2, 3
    W, H = _W(R, C, 50), _H(C, 51)
    Q = _foem_ufficiale(W, H, K, ng, beta)
    T1, S1, _ = SO.gptq_bilanciato(W, H, K, ng, foem_beta=beta, foem_forma="codice" if beta > 0 else "lavoro")
    assert torch.allclose(SO.ricostruisci(T1, S1), Q, rtol=0, atol=1e-12)
