#!/usr/bin/env python3
"""S1: prove CPU pure (nessun modello, nessuna GPU). Uscita 0 = tutte verdi; stampa una riga per prova."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import s1_comune as SC  # noqa: E402
import s1_cattura as CA  # noqa: E402
import s1_gradienti as GR  # noqa: E402

torch.set_num_threads(4)
ESITI = []


def prova(nome):
    def dec(f):
        try:
            f()
            ESITI.append((nome, True, ""))
        except Exception as e:  # noqa: BLE001 - si riporta l'esito di ogni prova
            ESITI.append((nome, False, f"{type(e).__name__}: {e}"))
        return f
    return dec


@prova("triangolo impacchetta/spacchetta (numpy e torch)")
def _():
    A = np.random.default_rng(0).normal(size=(7, 7))
    S = A + A.T
    assert np.allclose(SC.spacchetta(SC.impacchetta(S), 7), S)
    T = torch.from_numpy(S)
    assert np.allclose(SC.impacchetta(T).numpy(), SC.impacchetta(S))
    assert SC.n_da_tri(28) == 7


@prova("restringimento verso la H di strato")
def _():
    He, Hs = np.eye(3) * 2, np.eye(3)
    assert np.allclose(SC.ristretta(He, 1000, Hs, 0), He)
    assert np.allclose(SC.ristretta(He, 0, Hs, 8000), Hs)
    assert np.allclose(SC.ristretta(He, 8000, Hs, 8000), 1.5 * np.eye(3))


@prova("hash dei 13-grammi = confronto esatto")
def _():
    r = np.random.default_rng(1)
    a = r.integers(0, 50, 200)
    b = np.concatenate([r.integers(0, 50, 30), a[57:57 + 13], r.integers(0, 50, 30)])
    ha, hb = SC.hash_ngrammi(a), SC.hash_ngrammi(b)
    comuni = set(ha.tolist()) & set(hb.tolist())
    wa = {tuple(x) for x in np.lib.stride_tricks.sliding_window_view(a, 13)}
    wb = {tuple(x) for x in np.lib.stride_tricks.sliding_window_view(b, 13)}
    assert len(comuni) == len(wa & wb) >= 1


def omega(z):
    p = torch.softmax(z, -1)
    return torch.diag(p) - torch.outer(p, p)


@prova("Teorema 1: KL(p_T||p_S) = int (1 - tau) dz^T Omega dz (quadratura)")
def _():
    g = torch.Generator().manual_seed(3)
    zT, zS = torch.randn(6, generator=g, dtype=torch.float64) * 2, torch.randn(6, generator=g, dtype=torch.float64)
    pT, pS = torch.softmax(zT, -1), torch.softmax(zS, -1)
    kl = float((pT * (pT.log() - pS.log())).sum())
    x, w = np.polynomial.legendre.leggauss(64)
    t = (x + 1) / 2
    dz = zS - zT
    q = sum(wi / 2 * (1 - ti) * float(dz @ omega(zT + ti * dz) @ dz) for ti, wi in zip(t, w))
    assert abs(q - kl) < 1e-10, (q, kl)


@prova("Fisher di cammino: E[xi xi^T] = int w Omega (Monte Carlo, tre pesi)")
def _():
    g = torch.Generator().manual_seed(5)
    V = 5
    zT = torch.randn(V, generator=g, dtype=torch.float64) * 2
    zS = torch.randn(V, generator=g, dtype=torch.float64)
    x, w = np.polynomial.legendre.leggauss(64)
    t = (x + 1) / 2
    pesi = {"diretta": lambda u: 1 - u, "inversa": lambda u: u, "centro": lambda u: 3 * u * (1 - u)}
    for modo, wf in pesi.items():
        ref = sum(wi / 2 * wf(ti) * omega(zT + ti * (zS - zT)) for ti, wi in zip(t, w))
        n = 400000
        xi = GR.xi_da_logit(zT.expand(n, V).clone(), zS.expand(n, V).clone(), modo, g)
        emp = xi.T @ xi / n
        err = float((emp - ref).abs().max() / ref.abs().max())
        assert err < 0.02, (modo, err)


@prova("accumulo per esperto (Gq, C, H del down, conteggi) = somme esplicite")
def _():
    r = torch.Generator().manual_seed(7)
    N, H, I, E, k = 50, 6, 3, 4, 2
    xq, xf = torch.randn(N, H, generator=r), torch.randn(N, H, generator=r)
    idx = torch.stack([torch.randperm(E, generator=r)[:k] for _ in range(N)])
    w = torch.rand(N, k, generator=r)
    w = w / w.sum(-1, keepdim=True)
    piatto = idx.reshape(-1)
    ordine = torch.argsort(piatto, stable=True)
    offs = torch.searchsorted(piatto[ordine], torch.arange(E + 1))
    a = torch.randn(N * k, I, generator=r)                   # ingresso del down per ogni assegnazione (ordinato)
    Wgu = torch.randn(E, 2 * I, H, generator=r)
    acc = CA.Accumulatori(0, "linear_attention", {"attn": H}, E, I, H, torch.device("cpu"), True)
    acc.moe(xq, xf, w, k, a, ordine, offs, Wgu)
    for e in range(E):
        G, C, Hd, n, ps = torch.zeros(H, H), torch.zeros(2 * I, H), torch.zeros(I, I), 0, 0.0
        for q in range(N * k):
            p = int(ordine[q])
            t, j = divmod(p, k)
            if int(idx[t, j]) != e:
                continue
            ww = float(w[t, j])
            G += ww * torch.outer(xq[t], xq[t])
            C += ww * torch.outer(Wgu[e] @ xf[t], xq[t])
            Hd += ww * torch.outer(a[q], a[q])
            n += 1
            ps += ww
        assert torch.allclose(acc.Gq[e], G, atol=1e-4) and torch.allclose(acc.C[e], C, atol=1e-4)
        assert torch.allclose(acc.Hd[e], Hd, atol=1e-4) and int(acc.conta[e]) == n and abs(float(acc.peso[e]) - ps) < 1e-6


@prova("salvataggio e lettura (densi fp32 a triangolo, esperti fp16 scalati, permutazione ssm_out)")
def _():
    r = torch.Generator().manual_seed(9)
    H, I, E, n4 = 6, 3, 4, 8
    dims = {"attn": H, "ffn": H, "att_out": n4, "down": I}
    acc = CA.Accumulatori(0, "linear_attention", dims, E, I, H, torch.device("cpu"), True)
    cp, cq = {}, {}
    for p, n in dims.items():
        cq[p], cp[p] = torch.randn(20, n, generator=r), torch.randn(20, n, generator=r)
    for s in SC.SORGENTI:
        acc.densi(s, cp, cq)
    for e in range(E):
        X = torch.randn(10, H, generator=r)
        acc.Gq[e] = X.T @ X
        acc.C[e] = torch.randn(2 * I, H, generator=r)
        Y = torch.randn(10, I, generator=r)
        acc.Hd[e] = Y.T @ Y
        acc.conta[e] = 10
        acc.peso[e] = 3.0
    perm = np.random.default_rng(2).permutation(n4)
    with tempfile.TemporaryDirectory() as d:
        acc.salva(Path(d) / "A" / "strato_00", perm, {"strato": 0, "tipo": "linear_attention"})
        lt = SC.LettoreS1(d)
        Hq = lt.H_denso("A", 0, "ffn")
        ref = (cq["ffn"].double().T @ cq["ffn"].double()) / 20
        assert np.allclose(Hq, ref.numpy(), rtol=1e-5, atol=1e-6)
        Ho = lt.H_denso("A", 0, "att_out", ("M",))
        refo = ((cq["att_out"].double().T @ cq["att_out"].double()) / 20).numpy()[perm][:, perm]
        assert np.allclose(Ho, refo, rtol=1e-5, atol=1e-6)
        ex = lt.esperti("A", 0)
        G0 = SC.spacchetta(np.asarray(ex["Gq"][1], np.float64) * ex["Gq_scala"][1], H)
        assert np.allclose(G0, acc.Gq[1].double().numpy(), rtol=2e-3, atol=2e-3 * float(acc.Gq[1].abs().max()))
        He = lt.H_esperto("A", 0, 1, k=0.0)
        assert np.allclose(He, G0 / 3.0)
        C1 = lt.C_esperto("A", 0, 2)
        assert np.allclose(C1, acc.C[2].double().numpy() / 3.0, atol=2e-3 * float(acc.C[2].abs().max()))
        assert (Path(d) / "A" / "strato_00" / "FATTO").exists()


@prova("toppa di FamigliaEsperti.apply: cattura l'ingresso del down e si toglie lasciando la classe intatta")
def _():
    class FE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, d, fam, piano):
            return x * 2

        @staticmethod
        def backward(ctx, g):
            return g * 2, None, None, None

    class Smod:
        FamigliaEsperti = FE

    class Lin(torch.nn.Linear):
        pass

    m = torch.nn.Module()
    m.input_layernorm, m.post_attention_layernorm = torch.nn.Identity(), torch.nn.Identity()
    m.linear_attn = torch.nn.Module()
    m.linear_attn.out_proj = Lin(3, 3)
    m.mlp = torch.nn.Module()
    m.mlp.shared_expert = torch.nn.Module()
    m.mlp.shared_expert.down_proj = Lin(3, 3)
    fam_d, fam_g = object(), object()
    prima = FE.apply
    g = CA.Ganci(m, False, moe=True, S=Smod, fam_down=fam_d)
    x = torch.ones(2, 3)
    FE.apply(x, None, fam_g, "p1")
    assert "a_ord" not in g.d
    y = FE.apply(x, None, fam_d, "p2")
    assert torch.equal(y, x * 2) and torch.equal(g.d["a_ord"], x) and g.d["piano"] == "p2"
    g.chiudi()
    assert "apply" not in FE.__dict__ and FE.apply == prima


@prova("raccoglitore dei gradienti: modi kl/cam, solo densi di primo piano")
def _():
    rac = GR.Raccoglitore()

    class F:
        def __init__(s, nome, E):
            s.nome, s.E, s.strato = nome, E, 3
    assert rac.vuole(F("attn_qkv", 1)) and not rac.vuole(F("attn_qkv2", 1)) and not rac.vuole(F("gate", 256))
    g, x = torch.randn(5, 4), torch.randn(5, 3)
    rac.aggiungi(3, "attn_qkv", g, x)
    rac.modo = "cam"
    rac.aggiungi(3, "attn_qkv", 2 * g, x)
    k = (3, "attn_qkv")
    assert torch.allclose(rac.G[k], g.T @ x) and torch.allclose(rac.B[k], g.T @ g)
    assert torch.allclose(rac.Bc[k], 4 * g.T @ g) and rac.n[k] == 5 and rac.nc[k] == 5


@prova("router ricalcolato = matematica di MoEStudente (top-k e rinormalizzazione)")
def _():
    class Mlp:
        router_w32 = None
        top_k = 2
    m = Mlp()
    m.router_w = torch.randn(6, 4)
    x = torch.randn(9, 4)
    idx, w = CA.router(m, x)
    p = torch.softmax(x @ m.router_w.T, -1)
    i2 = torch.topk(p, 2, -1).indices
    assert torch.equal(idx, i2) and torch.allclose(w.sum(-1), torch.ones(9))


if __name__ == "__main__":
    for n, ok, msg in ESITI:
        print(f"{'OK' if ok else 'KO'}  {n}{'' if ok else '  -> ' + msg}")
    sys.exit(0 if all(ok for _, ok, _ in ESITI) else 1)
