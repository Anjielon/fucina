"""Obiettivo C / C0T (9/10): prove NUMERICHE fp64 delle trasformazioni usate per rendere ternari i ~91 MB rimasti.
Esatte (errore ~1e-15): T1 piega RMSNorm nei consumatori; T2 piega della norma GATED (ssm_norm) in ssm_out;
T3 piega di g_q*g_k sul lato k per le dimensioni NON ruotate (IMROPE: coppie (i, i+32) sulle prime 64);
T4 piega in base RUOTATA (TQ1_0R_1024): ruota(deruota(Q)*g) == Q @ R diag(g) R^T; T5 testa: output*diag(g_out) con
shared_head_norm' = g_sh/g_out; T6 conv1d trasposta (dequant-on-load 'trasposta').
NON esatte (si misura l'errore): T3b coppie ruotate con g_q diverso nella coppia; T7 hnorm dell'MTP dopo la piega di
output_norm (rapporto di rms dipendente dal token)."""
import sys
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py"); sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
import numpy as np
import torch
torch.set_num_threads(6)
rng = np.random.default_rng(1)
EPS = 1e-6


def rms(x):
    return np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS)


def t1():
    x = rng.standard_normal((64, 2048)); g = rng.uniform(.2, 3, 2048); W = rng.standard_normal((300, 2048))
    a = (x / rms(x) * g) @ W.T; b = (x / rms(x)) @ (W * g).T
    return np.abs(a - b).max() / np.abs(a).max()


def t2():
    o = rng.standard_normal((64, 32, 128)); z = rng.standard_normal((64, 32, 128)); g = rng.uniform(.5, 1, 128)
    W = rng.standard_normal((2048, 4096)); silu = z / (1 + np.exp(-z))
    a = ((o / rms(o) * g) * silu).reshape(64, -1) @ W.T
    b = ((o / rms(o)) * silu).reshape(64, -1) @ (W * np.tile(g, 32)).T
    return np.abs(a - b).max() / np.abs(a).max()


def rope(x, th):
    y = x.copy(); c, s = np.cos(th), np.sin(th)
    y[..., :32] = x[..., :32] * c - x[..., 32:64] * s; y[..., 32:64] = x[..., :32] * s + x[..., 32:64] * c
    return y


def t3(pari_nelle_coppie):
    q = rng.standard_normal((500, 256)); k = rng.standard_normal((500, 256)); tm = rng.uniform(0, 50, (500, 1))
    tn = rng.uniform(0, 50, (500, 1)); fr = 1e4 ** (-np.arange(32) / 32)
    gq = rng.uniform(.6, 1.9, 256); gk = rng.uniform(.6, 2.1, 256)
    if pari_nelle_coppie:
        gq[32:64] = gq[:32]
    a = np.sum(rope(q * gq, tm * fr) * rope(k * gk, tn * fr), -1)
    c = gq * gk; gk2 = c.copy()
    if pari_nelle_coppie:
        gk2[:64] = gk[:64]; gq2 = np.ones(256); gq2[:64] = gq[:64]    # q tiene solo il guadagno per COPPIA
        gk2[:64] *= 1.0
        # q: guadagno costante nella coppia commuta con la rotazione -> si sposta su k
        gq2 = np.ones(256); gk2[:64] = gq[:64] * gk[:64]
    else:
        gq2 = np.ones(256)
    b = np.sum(rope(q * gq2, tm * fr) * rope(k * gk2, tn * fr), -1)
    return np.abs(a - b).max() / np.abs(a).max()


def t4():
    import comune as CM
    Q = torch.from_numpy(rng.standard_normal((40, 2048)))
    g = torch.from_numpy(rng.uniform(.2, 3, 2048))
    a = CM.ruota(CM.deruota(Q) * g)
    M = CM.ruota_H(torch.diag(g))
    b = Q @ M
    return float((a - b).abs().max() / a.abs().max())


def t5():
    h = rng.standard_normal((64, 2048)); go = rng.uniform(.7, 3.5, 2048); gs = rng.uniform(.5, 2, 2048)
    W = rng.standard_normal((500, 2048))
    a = (h / rms(h) * go) @ W.T; a2 = (h / rms(h) * gs) @ W.T
    b = (h / rms(h)) @ (W * go).T; b2 = (h / rms(h) * (gs / go)) @ (W * go).T
    return max(np.abs(a - b).max() / np.abs(a).max(), np.abs(a2 - b2).max() / np.abs(a2).max())


def t6():
    w = rng.standard_normal((8192, 4))           # memoria ggml [4, 8192]: riga = canale, 4 tap contigui
    st = w.T.copy()                              # memorizzato [8192 canali contigui] x 4 tap = ggml [8192, 4]
    a, b = st.shape[1], st.shape[0]              # ne_src0 = 8192, ne_src1 = 4
    tmp = st.reshape(-1); dst = np.empty(a * b)
    for j in range(b):
        for i in range(a):
            dst[i * b + j] = tmp[j * a + i]
    return np.abs(dst.reshape(8192, 4) - w).max()


def t7():
    """hnorm dopo la piega: errore relativo del vettore normalizzato con il rapporto rms costante stimato."""
    H = np.load("@@ROOT:models@@/gguf/tony-tern/hessiane-output/H.npy").astype(np.float64)
    lam, U = np.linalg.eigh(0.5 * (H + H.T)); lam = np.clip(lam, 0, None)
    sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
    from gguf import GGUFReader
    r = GGUFReader("@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
    T = {t.name: t for t in r.tensors}
    go = np.array(T["output_norm.weight"].data, np.float64); gh = np.array(T["blk.40.nextn.hnorm.weight"].data, np.float64)
    h = rng.standard_normal((4000, 2048)) @ (U * np.sqrt(lam)).T          # h = ingresso della testa (con g_out)
    print("T7 scala h: rms medio", float(np.mean(np.sqrt(np.mean(h * h, -1)))))
    rms0 = lambda x: np.sqrt(np.mean(x * x, -1, keepdims=True))         # noqa: E731 (eps trascurabile: |h|~g_out)
    a = h / rms0(h) * gh
    hp = h / go
    rr = rms0(hp) / rms0(h); rho = float(np.mean(rr))
    print("T7 rapporto rms per token: media", rho, "cv", float(np.std(rr) / rho))
    b = hp / rms0(hp) * (gh * go * rho)
    return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2))), rho


if __name__ == "__main__":
    r = {"T1_piega_rmsnorm": t1(), "T2_piega_norma_gated": t2(), "T3_qk_coppie_pari_esatto": t3(True),
         "T3b_qk_coppie_diverse_errore": t3(False), "T4_piega_base_ruotata": t4(), "T5_testa_e_shared_head": t5(),
         "T6_conv_trasposta": t6()}
    e7, rho = t7(); r["T7_hnorm_mtp_errore_rel"] = e7; r["T7_rho"] = rho
    for k, v in r.items():
        print(f"{k:32s} {v:.3e}")
    assert r["T1_piega_rmsnorm"] < 1e-12 and r["T2_piega_norma_gated"] < 1e-12 and r["T3_qk_coppie_pari_esatto"] < 1e-12
    assert r["T4_piega_base_ruotata"] < 1e-12 and r["T5_testa_e_shared_head"] < 1e-12 and r["T6_conv_trasposta"] == 0
    assert r["T3b_qk_coppie_diverse_errore"] > 1e-3      # la non esattezza e' reale, non un artefatto
    import json; json.dump(r, open("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/C0T/analisi/test_matematica.json", "w"), indent=1)
    print("OK")
