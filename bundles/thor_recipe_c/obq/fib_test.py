"""Test puri del solutore (CPU, secondi). uso: python3 fib_test.py
 1. lam = 0  ==  GPTQ a scale fisse scritto a parte (riferimento indipendente, colonna per colonna), sia il
    riferimento sia il solutore veloce a blocchi;
 2. con un compito: obq_fibrato_rif == l'implementazione di V1 (ricerche/MATEMATICA_FUSA_1013/script/v1_fibrato.py), copiata qui al
    carattere come riferimento: stessi trit al bit;
 2b. solutore veloce (base originale, a blocchi) == riferimento a meno dell'arrotondamento: >= 99,9 % dei trit
    uguali e KF entro lo 0,5 %; anche con blocco 16 (piu' confini di blocco);
 3. in campione la metrica fusa M scende rispetto a lam = 0 (sistema sintetico con B a basso rango, come lo strato 39);
 4. andata e ritorno della codifica TQ1_0R_1024 / TQ1_0 (se rd_ternary e' raggiungibile).
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fib_solutore as FS  # noqa: E402

F64 = torch.float64
torch.manual_seed(0)


def gptq_scale_fisse(W, H, D, damp=0.01):
    R, C = W.shape
    G = C // D.shape[1]
    Hd = H + damp * torch.diagonal(H).mean() * torch.eye(C, dtype=F64)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    Wc = W.clone()
    T = torch.zeros(R, C, dtype=torch.int8)
    for j in range(C):
        Dj = D[:, j // G]
        t = torch.clamp(torch.round(Wc[:, j] / Dj), -1, 1)
        T[:, j] = t.to(torch.int8)
        e = (Wc[:, j] - Dj * t) / U[j, j]
        Wc[:, j + 1:] -= e.unsqueeze(1) * U[j, j + 1:].unsqueeze(0)
    return T


def v1_riferimento(W, H, D, metriche, damp=0.01, NTOP=8):
    """v1_fibrato.obq_fibrato, al carattere (salvo le variabili globali passate come argomenti)."""
    R, C = W.shape
    G = C // D.shape[1]
    Bmix = sum(l * Bi for l, Ai, Bi in metriche)
    mu_, V_ = torch.linalg.eigh(Bmix); V = V_.flip(1).contiguous()
    coef = [torch.einsum("rk,rs,sk->k", V, Bi, V) for l, Ai, Bi in metriche]
    bins = [[k] for k in range(NTOP)] + [list(range(NTOP, R))]
    U = []
    for ks in bins:
        Hk = H.clone()
        for (l, Ai, Bi), cf in zip(metriche, coef):
            Hk = Hk + l * float(cf[ks].clamp_min(0).mean()) * Ai
        Hk = Hk + damp * torch.diagonal(Hk).mean() * torch.eye(C, dtype=F64)
        U.append(torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hk)), upper=True))
    kbin = torch.empty(R, dtype=torch.long)
    for b, ks in enumerate(bins): kbin[ks] = b
    Yt = V.T @ W
    T = torch.zeros(R, C, dtype=torch.int8)
    V8 = V[:, :NTOP]
    for j in range(C):
        dj = torch.stack([U[b][j, j] for b in range(len(U))])[kbin]
        lamj = 1.0 / dj ** 2
        y = V @ Yt[:, j]
        Dj = D[:, j // G]
        t = torch.clamp(torch.round(y / Dj), -1, 1)
        delta = float(lamj[NTOP:].mean())
        l8 = lamj[:NTOP] - delta
        r = y - Dj * t
        z = V8.T @ r
        Bii = delta + (V8 ** 2 * l8).sum(1)
        for _ in range(4 * NTOP + 16):
            Br = delta * r + V8 @ (l8 * z)
            best = None
            for dt in (-1.0, 1.0):
                tn = t + dt
                df = -2 * dt * Dj * Br + (dt * Dj) ** 2 * Bii
                df = torch.where((tn >= -1) & (tn <= 1), df, torch.full_like(df, math.inf))
                v, i = df.min(0)
                if best is None or v < best[0]: best = (float(v), int(i), dt)
            if best[0] >= -1e-15: break
            _, i, dt = best
            t[i] += dt; r[i] -= dt * Dj[i]; z -= dt * Dj[i] * V8[i]
        T[:, j] = t.to(torch.int8)
        et = (V.T @ (y - Dj * t)) / dj
        for b, ks in enumerate(bins):
            Yt[ks, j + 1:] -= et[ks].unsqueeze(1) * U[b][j, j + 1:].unsqueeze(0)
    return T


def sistema(R=48, C=64, ng=4, rango_B=2):
    X = torch.randn(4 * C, C, dtype=F64) @ (torch.eye(C, dtype=F64) + 0.3 * torch.randn(C, C, dtype=F64))
    H = FS.psd(X.T @ X)
    Xa = torch.randn(4 * C, C, dtype=F64) * torch.linspace(0.2, 2, C, dtype=F64)
    A = FS.psd(Xa.T @ Xa)
    Gb = torch.randn(R, rango_B, dtype=F64)
    B = FS.psd(Gb @ Gb.T + 0.02 * torch.eye(R, dtype=F64))
    W = torch.randn(R, C, dtype=F64) * 0.05
    D = FS.ricostruisci(torch.ones(R, C, dtype=torch.int8), torch.ones(R, ng, dtype=F64))[:, ::C // ng].clone()
    D = (W.abs().reshape(R, ng, C // ng).amax(-1) * 0.7).half().double()
    return W, H, A, B, D


def main():
    ok = True
    W, H, A, B, D = sistema()
    T0 = FS.obq_fibrato_rif(W, H, D, [])
    Tr = gptq_scale_fisse(W, H, D)
    T0v = FS.obq_fibrato(W, H, D, [], blocco=16)
    e1 = torch.equal(T0, Tr)
    a1 = float((T0v == Tr).double().mean())
    print(f"1. lam=0: riferimento == GPTQ a scale fisse: {e1}; veloce (blocco 16) uguale su {a1*100:.3f} % dei trit")
    ok &= e1 and a1 >= 0.999
    for lam in (0.3, 1.0):
        T1 = FS.obq_fibrato_rif(W, H, D, [(lam, A, B)])
        Tv = v1_riferimento(W, H, D, [(lam, A, B)])
        e2 = torch.equal(T1, Tv)
        print(f"2. lam={lam}: riferimento == V1 al bit: {e2} (trit diversi da lam=0: {float((T1 != T0).double().mean())*100:.1f} %)")
        ok &= e2
        for bl in (128, 16):
            Tf = FS.obq_fibrato(W, H, D, [(lam, A, B)], blocco=bl)
            ag = float((Tf == T1).double().mean())
            kr = FS.relKF(W - FS.ricostruisci(T1, D), W, A, B); kf = FS.relKF(W - FS.ricostruisci(Tf, D), W, A, B)
            e2b = ag >= 0.999 and abs(kf / kr - 1) < 0.005
            print(f"2b. lam={lam} veloce blocco {bl}: trit uguali al riferimento {ag*100:.3f} %, KF {kr:.5f} vs {kf:.5f}: {e2b}")
            ok &= e2b
        M0 = FS.relKF(W - FS.ricostruisci(T0, D), W, A, B) ** 2 * lam + FS.rel(W - FS.ricostruisci(T0, D), W, H) ** 2
        M1 = FS.relKF(W - FS.ricostruisci(T1, D), W, A, B) ** 2 * lam + FS.rel(W - FS.ricostruisci(T1, D), W, H) ** 2
        k0 = FS.relKF(W - FS.ricostruisci(T0, D), W, A, B); k1 = FS.relKF(W - FS.ricostruisci(T1, D), W, A, B)
        e3 = k1 < k0
        print(f"3. lam={lam}: KF {k0:.4f} -> {k1:.4f}; metrica fusa (proxy) {M0:.4f} -> {M1:.4f}: KF scende {e3}")
        ok &= e3
    try:
        sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
        import comune as CM
        for tid, C, G in ((45, 2048, 128), (34, 512, 256)):
            R = 16
            t = torch.randint(-1, 2, (R, C), dtype=torch.int8).numpy()
            s = (torch.rand(R, C // G) * 0.01 + 1e-3).half().numpy()
            b = CM.gp.encode_rows(t, s, tid)
            t2, s2 = CM.gp.decode_rows(b, tid, C)
            e4 = (t2 == t).all() and (s2.view("uint16") == s.view("uint16")).all()
            print(f"4. codifica tipo {tid}: andata e ritorno {bool(e4)}")
            ok &= bool(e4)
    except Exception as ex:          # noqa: BLE001 — il test 4 dipende da un albero esterno: lo dichiara
        print(f"4. codifica: non eseguito ({ex})")
        ok = False
    print("TEST OK" if ok else "TEST FALLITI")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
