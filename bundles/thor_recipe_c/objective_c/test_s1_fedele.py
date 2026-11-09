"""Garanzie di s1_fedele.py (11/10/2026), dati sintetici, CPU, fp64.
  * identita' esatta delle pieghe (norma F32 con s_eff; up->down nello SwiGLU): uscita invariata entro 1e-10;
  * H' = diag(s)^-1 H diag(s)^-1 == X'^T X' / N con X' = X diag(s)^-1, ed errore pesato invariante;
  * OSAQ (2605.04738): regola dell'energia (eq. 8) minimale; forma chiusa (eq. 12-14) == lstsq sul disegno esplicito
    dell'eq. 11 (oracolo indipendente); gradiente dell'eq. 11 nullo; dL = sum_i b_i^T diag(lambda_K) b_i;
  * AWQ (2306.00978, auto_scale.py): ratio 0 -> s = 1 (perdita = RTN puro), scale normalizzate max·min = 1, scelta <= RTN;
  * Q = RTN del formato coincide col quantizzatore di gptq_trits (GPTQ con H diagonale non propaga errore);
  * segni (2609.21450): ruotare W·D coi segni S == ruotare W coi segni S·D (piega esatta);
  * post_attention_norm NULLA con gli esperti fermi: R diag(d) R^T e' diagonale solo se d e' costante su ogni blocco
    da 1024; con d = 2^k costante a blocchi GPTQ da' gli STESSI codici e scale x 2^k (stesso errore nella base originale).
uso: HIP_VISIBLE_DEVICES= python3 -m pytest -q test_s1_fedele.py
"""
import os, sys
os.environ.setdefault("LEVE_DT", "float64")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import s1_fedele as S1

JF, SO, CM = S1.JF, S1.SO, S1.CM
F64 = torch.float64
torch.set_num_threads(4)


def _g(seed):
    return torch.Generator().manual_seed(seed)


def test_identita_norma_f32():
    g = _g(1)
    W = torch.randn(96, 2048, generator=g, dtype=F64) * 0.02
    gam = (1 + 0.1 * torch.randn(2048, generator=g, dtype=F64)).float().numpy()
    s = torch.rand(2048, generator=g, dtype=F64) * 3 + 0.2
    gnew = (torch.from_numpy(gam).double() / s).float().numpy()
    s_eff = torch.from_numpy(gam).double() / torch.from_numpy(gnew).double()
    x = torch.randn(32, 2048, generator=g, dtype=F64)
    y0 = (x * torch.from_numpy(gam).double()) @ W.T
    y1 = (x * torch.from_numpy(gnew).double()) @ (W * s_eff).T
    assert float((y0 - y1).abs().max() / y0.abs().max()) < 1e-10


def test_identita_swiglu():
    g = _g(2)
    Wg, Wu = torch.randn(64, 256, generator=g, dtype=F64), torch.randn(64, 256, generator=g, dtype=F64)
    Wd = torch.randn(256, 64, generator=g, dtype=F64)
    s = torch.rand(64, generator=g, dtype=F64) * 4 + 0.1
    x = torch.randn(16, 256, generator=g, dtype=F64)
    f = lambda gg, u, d: (torch.nn.functional.silu(x @ gg.T) * (x @ u.T)) @ d.T
    y0, y1 = f(Wg, Wu, Wd), f(Wg, Wu / s.unsqueeze(1), Wd * s)
    assert float((y0 - y1).abs().max() / y0.abs().max()) < 1e-10


def test_H_trasformata():
    g = _g(3)
    X = torch.randn(500, 64, generator=g, dtype=F64) * (torch.rand(64, generator=g, dtype=F64) * 5)
    s = torch.rand(64, generator=g, dtype=F64) + 0.3
    H = X.T @ X / 500
    Hp = (X / s).T @ (X / s) / 500
    assert torch.allclose(Hp, H / torch.outer(s, s), rtol=1e-12, atol=1e-14)
    E = torch.randn(7, 64, generator=g, dtype=F64)
    assert abs(S1.trHE(E * s, Hp) - S1.trHE(E, H)) / S1.trHE(E, H) < 1e-12


def _osaq_dati(seed=4, M=12, C=48):
    g = _g(seed)
    Q, _ = torch.linalg.qr(torch.randn(C, C, generator=g, dtype=F64))
    lam = torch.logspace(-7, 1, C, dtype=F64)
    H = Q @ torch.diag(lam) @ Q.T
    W = torch.randn(M, C, generator=g, dtype=F64) * 0.05
    W[0, 3] = 0.6                                                  # un outlier
    return W, H


def test_osaq_forma_chiusa_contro_oracolo():
    W, H = _osaq_dati()
    gamma, tau, mu1, mu2 = 1e-4, 0.1, 1e-3, 1e-3
    dW, info, (Nn, s, v, B) = S1.osaq(W, H, gamma, tau, mu1, mu2)
    lam = torch.linalg.eigvalsh(H).abs().sort().values
    cum = torch.cumsum(lam, 0)
    K = info["K"]
    assert cum[K - 1] >= gamma * lam.sum() and (K == 1 or cum[K - 2] < gamma * lam.sum())   # eq. 8 minimale
    C = W.shape[1]
    for i in range(W.shape[0]):
        Phi = torch.cat([(s[i].sqrt().unsqueeze(1) * Nn.T), mu1 ** 0.5 * torch.eye(K, dtype=F64),
                         mu2 ** 0.5 * v.unsqueeze(0)], 0)
        tgt = torch.cat([-(s[i].sqrt() * W[i]), torch.zeros(K + 1, dtype=F64)])
        b = torch.from_numpy(np.linalg.lstsq(Phi.numpy(), tgt.numpy(), rcond=None)[0])
        assert torch.allclose(B[i], b, rtol=1e-9, atol=1e-13), i
        r = W[i] + B[i] @ Nn                                        # gradiente eq. 11 = 0
        grad = Nn @ (s[i] * r) + mu1 * B[i] + mu2 * (B[i] @ v) * v
        assert float(grad.abs().max()) < 1e-12
    assert abs(info["dL_rel"] - info["dL_rel_da_beta"]) <= 1e-10 * max(info["dL_rel"], 1e-30)


def test_awq_ratio0_e_normalizzazione():
    g = _g(5)
    W = torch.randn(40, 2048, generator=g, dtype=F64) * 0.03
    X = torch.randn(300, 2048, generator=g, dtype=F64) * (torch.rand(2048, generator=g, dtype=F64) * 3 + 0.05)
    H = X.T @ X / 300
    xm = torch.diagonal(H).sqrt()
    sc, r, st = S1.awq_cerca([W], H, xm, True)
    assert abs(st[0] - S1.trHE(S1.Q_formato(W, True) - W, H)) < 1e-12 * st[0]
    assert abs(float(sc.max() * sc.min()) - 1) < 1e-12
    assert min(st) <= st[0]


def test_rtn_uguale_quantizzatore_gptq():
    g = _g(6)
    W = torch.randn(10, 512, generator=g, dtype=F64) * 0.05
    T, S = JF.gptq_trits(W, torch.eye(512, dtype=F64), 2)        # H diagonale: nessuna propagazione
    assert torch.allclose(SO.ricostruisci(T, S).float(), S1.rtn_trits(W.float().double(), 256).float())


def test_segni_piega_esatta():
    g = _g(7)
    W = torch.randn(8, 2048, generator=g, dtype=F64)
    S = S1.segni_motore()
    D = (torch.randint(0, 2, (1024,), generator=g) * 2 - 1).double()
    Dt = D.repeat(2)
    assert torch.equal(S1.ruota_s(W * Dt, S), S1.ruota_s(W, S * D))


def test_post_attention_nulla_con_esperti_fermi():
    g = _g(8)
    S = S1.segni_motore()
    R = S1.ruota_s(torch.eye(2048, dtype=F64), S).T                 # colonne: R tale che W' = W R^T
    d = torch.rand(2048, generator=g, dtype=F64) + 0.5
    M = R.T @ torch.diag(d) @ R                                      # base memorizzata
    off = (M - torch.diag(torch.diagonal(M))).abs().max()
    assert off > 1e-3                                                # d generica: mescola le colonne (codici cambiano)
    db = torch.cat([torch.full((1024,), 2.0, dtype=F64), torch.full((1024,), 0.5, dtype=F64)])
    Mb = R.T @ torch.diag(db) @ R
    assert (Mb - torch.diag(torch.diagonal(Mb))).abs().max() < 1e-12   # costante a blocchi: diagonale
    # con 2^k a blocchi il problema di quantizzazione e' lo STESSO: (T, S) <-> (T, S·2^k) e' una biiezione fra le
    # soluzioni ammissibili (prodotto fp16 per potenza di 2 esatto) che conserva l'obiettivo pesato al bit -> leva nulla
    W = torch.randn(16, 2048, generator=g, dtype=F64) * 0.02
    X = torch.randn(400, 2048, generator=g, dtype=F64)
    H = X.T @ X / 400
    T = torch.randint(-1, 2, (1, 16, 2048), generator=g).to(torch.int8)
    Sc = SO.fp16(torch.rand(1, 16, 16, generator=g, dtype=F64) * 0.05)
    cb = torch.cat([torch.full((8,), 2.0, dtype=F64), torch.full((8,), 0.5, dtype=F64)])
    e1 = S1.trHE(S1.deruota_s(SO.ricostruisci(T, Sc), S) - W, H)
    Hs = H / torch.outer(db, db)
    e2 = S1.trHE(S1.deruota_s(SO.ricostruisci(T, SO.fp16(Sc * cb)), S) - W * db, Hs)
    assert abs(e1 - e2) <= 1e-12 * e1
    q1 = S1.Q_formato(W, True)
    q2 = S1.Q_formato(W * db, True) / db                             # RTN: soluzione corrispondente, al bit
    assert float((q1 - q2).abs().max()) < 1e-15
