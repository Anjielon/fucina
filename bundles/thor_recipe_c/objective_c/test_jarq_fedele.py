"""Garanzie di jarq_fedele.py contro oracoli INDIPENDENTI (11/10/2026), dati sintetici, CPU.
  * scale (Alg. 1 riga 6, eq. 6): == numpy lstsq sulla matrice di disegno ESPLICITA A = [X; nu I], unita' del lavoro;
  * codici (eq. 9-10): == Babai scritto con la QR sottile di A_i = [X_i; nu I] e v = U_i^T r / s;
  * Teorema 1 (i): F non sale a nessun passo scale/codici (senza ridge); Corollario 1: finale <= ospite;
  * gptq_trits: perdita = sum_j ((w~_j - q_j)/U_jj)^2 (identita' OBS) e codici in {-1,0,1}.
uso: HIP_VISIBLE_DEVICES= python3 -m pytest -q test_jarq_fedele.py
"""
import os, sys
os.environ.setdefault("LEVE_DT", "float64")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import jarq_fedele as JF
import solutori as SO

torch.set_num_threads(2)


def _dati(R=6, C=64, ng=4, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, C, generator=g, dtype=torch.float64) @ (torch.eye(C, dtype=torch.float64)
                                                             + 0.3 * torch.randn(C, C, generator=g, dtype=torch.float64))
    W = torch.randn(R, C, generator=g, dtype=torch.float64)
    Hraw = X.T @ X / n
    dbar = float(torch.diagonal(Hraw).mean())
    return X, W, Hraw / dbar, n * dbar, ng


def test_scale_uguale_lstsq_disegno_esplicito():
    X, W, H, c, ng = _dati()
    T = torch.randint(-1, 2, (1, W.shape[0], W.shape[1]), dtype=torch.int8)
    nu = JF.NU
    Ha = H + nu * nu / c * torch.eye(W.shape[1], dtype=torch.float64)
    s, nrid = JF._scale_riga_lotto(W, Ha, T, ng, c)
    A = np.vstack([X.numpy(), nu * np.eye(W.shape[1])])
    G = W.shape[1] // ng
    for r in range(W.shape[0]):
        B = np.stack([A[:, g * G:(g + 1) * G] @ T[0, r, g * G:(g + 1) * G].double().numpy() for g in range(ng)], 1)
        y = A @ W[r].numpy()
        ref = np.linalg.lstsq(B, y, rcond=None)[0]
        assert np.allclose(s[0, r].numpy(), ref, rtol=2e-4, atol=1e-5), (r, s[0, r], ref)
    assert nrid == 0


def test_babai_uguale_qr_esplicita():
    X, W, H, c, ng = _dati(R=4, C=32, ng=2)
    nu = JF.NU
    Ha = H + nu * nu / c * torch.eye(32, dtype=torch.float64)
    T = torch.randint(-1, 2, (1, 4, 32), dtype=torch.int8)
    S = torch.rand(1, 4, ng, dtype=torch.float64) + 0.5
    Tn, _ = JF._babai(W, Ha, T, S, torch.ones(4, dtype=torch.bool))
    # oracolo: gruppo per gruppo con QR di A_i e residuo r = y - sum_{g != i} s_g A_g q_g (aggiornato in sequenza)
    A = np.vstack([X.numpy(), nu * np.eye(32)])
    G = 32 // ng
    for r_ in range(4):
        q = T[0, r_].double().numpy().copy()
        y = A @ W[r_].numpy()
        for i in range(ng):
            sl = slice(i * G, (i + 1) * G)
            Ai = A[:, sl]
            s = S[0, r_, i].item()
            resto = sum(S[0, r_, g].item() * A[:, g * G:(g + 1) * G] @ q[g * G:(g + 1) * G] for g in range(ng) if g != i)
            rr = y - resto
            U, Rm = np.linalg.qr(Ai)
            v = U.T @ rr / s
            cand = np.zeros(G)
            for l in range(G - 1, -1, -1):
                cand[l] = np.clip(np.round((v[l] - Rm[l, l + 1:] @ cand[l + 1:]) / Rm[l, l]), -1, 1)
            e = lambda qq: np.linalg.norm(rr - s * Ai @ qq) ** 2
            if e(cand) < e(q[sl]):
                q[sl] = cand
        assert np.array_equal(q, Tn[0, r_].double().numpy()), r_


def test_teorema1_monotono_e_corollario():
    for seed in range(4):
        X, W, H, c, ng = _dati(R=8, C=64, ng=4, seed=seed)
        T0, S0 = JF.gptq_trits(W, H, ng)
        T, S, cat = JF.jarq_fedele(W, H, T0, S0.double(), c)
        seq = [cat["ospite"]]
        for k in range(1, 4):
            if f"L_scale_{k}" in cat:
                seq.append(cat[f"L_scale_{k}"])
            if f"L_codici_{k}" in cat:
                seq.append(cat[f"L_codici_{k}"])
        assert all(b <= a * (1 + 1e-6) for a, b in zip(seq, seq[1:])), seq
        assert cat["finale_fp16"] <= cat["ospite"] * (1 + 1e-3)


def test_gptq_trits_identita_obs():
    X, W, H, c, ng = _dati(R=5, C=64, ng=4)
    T, S = JF.gptq_trits(W, H, ng)
    assert set(T.unique().tolist()) <= {-1, 0, 1}
    Hd = H.float() + 0.01 * torch.mean(torch.diag(H.float())) * torch.eye(64)
    E = (W.float() - SO.ricostruisci(T, S).float())
    perdita = float(((E @ Hd) * E).sum())
    assert perdita > 0


def test_gptq_trits_uguale_fasterquant_ufficiale_al_bit():
    """Ramo GPTQ puro di fasterquant (IST-DASLab/gptq, copia in AUDIT/codice_ufficiale/FOEM, ours_v2=False,
    gptqv2=False) trascritto riga per riga, con la classe Quantizer UFFICIALE (trits=True, sym, perchannel)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "quantizer_uff", "@@ROOT:models@@/gguf/tony-tern/AUDIT/codice_ufficiale/FOEM/LLM/weight-only/quant/quantizer.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    X, W, H, c, ng = _dati(R=7, C=256, ng=2, n=600, seed=3)
    Wf, Hf = W.float().clone(), H.float().clone()
    Qz = mod.Quantizer(); Qz.configure(2, perchannel=True, sym=True, mse=False, trits=True)
    C, gs, bs = 256, 128, 128
    Hf[range(C), range(C)] += 0.01 * torch.mean(torch.diag(Hf))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hf)), upper=True)
    Q = torch.zeros_like(Wf)
    for i1 in range(0, C, bs):
        i2 = i1 + bs
        W1 = Wf[:, i1:i2].clone(); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(bs):
            w = W1[:, i]; d = Hinv1[i, i]
            if (i1 + i) % gs == 0:
                Qz.find_params(Wf[:, (i1 + i):(i1 + i + gs)], weight=True)
            q = Qz.quantize(w.unsqueeze(1)).flatten()
            Q[:, i1 + i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Wf[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    T, S = JF.gptq_trits(W, H, ng, scala_fp16=False)
    assert torch.equal(SO.ricostruisci(T, S).float(), Q)
