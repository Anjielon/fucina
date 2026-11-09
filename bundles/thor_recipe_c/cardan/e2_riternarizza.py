#!/usr/bin/env python3
"""e2_riternarizza.py — ri-ternarizzazione CONSAPEVOLE DEI DATI delle righe lettera nel formato
TQ1_0R_1024 (tipo ggml 45), per e2_scrivi_gguf.py.

Perche': la ri-ternarizzazione RTN (quantize_rows del codec) di W_edit = W + piccola modifica
cancella quasi tutta la modifica (rank1 su R2B3m: 99,7% persa; con amplificazione alpha
l'errore sui logit cala solo del ~20%). Le 4 righe lettera pero' servono SOLO ai 4 logit
lettera, quindi l'unica cosa che conta e' E_h[((W_q - W_edit) h)^2].

Formato (per blocco di 1024): W = S (.) F(d (.) t), F = FWHT-1024 normalizzata (1/32, simmetrica,
involutiva), t in {-1,0,+1}, d = 8 scale fp16 (una ogni 128). Quindi W.h = x.g con
x = d (.) t e g = F(S (.) h): nel dominio ruotato l'obiettivo e' la forma quadratica
(x - x_e)^T C' (x - x_e), C' = E[g g^T] (+ ridge, che senza dati la riduce a Frobenius).
Si minimizza a coordinate sui trit (esatto per coordinata) alternato col rifit LS delle 8*nb
scale (arrotondate a fp16 come nel file). Partenza = trit/scale ORIGINALI della riga
(=> mai peggio di "nessuna modifica" nella metrica) e, in alternativa, RTN(W_edit): si tiene la
migliore. Il risultato e' ancora UN piano ternario + scale fp16 nello stesso layout di byte.
"""
from __future__ import annotations

import numpy as np

from e2_bnp_edit import CODEC_DIR
import sys
if CODEC_DIR not in sys.path:
    sys.path.insert(0, CODEC_DIR)
from tq1_0r_1024_codec import (BLOCK_BYTES, QK, S1024, _pack_quarter, _unpack_quarter,  # noqa: E402
                               fwht1024_f32, quantize_rows)


def unpack_row(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """riga (nb*224,) uint8 -> t (nb*1024,) in {-1,0,1} float64, d (nb*8,) float64 (gia' fp16)."""
    b = raw.reshape(-1, BLOCK_BYTES)
    nb = len(b)
    t = np.zeros((nb, QK))
    for q in range(4):
        t[:, q * 256:(q + 1) * 256] = _unpack_quarter(b[:, q * 48:(q + 1) * 48], b[:, 192 + q * 4:192 + (q + 1) * 4])
    d = b[:, 208:224].copy().view(np.float16).astype(np.float64).reshape(nb * 8)
    return t.reshape(-1), d


def pack_row(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    nb = len(t) // QK
    T = (np.rint(t).astype(np.int64) + 1).reshape(nb, 4, 256)
    assert T.min() >= 0 and T.max() <= 2
    out = np.zeros((nb, BLOCK_BYTES), dtype=np.uint8)
    for q in range(4):
        qs, qh = _pack_quarter(T[:, q, :])
        out[:, q * 48:(q + 1) * 48] = qs
        out[:, 192 + q * 4:192 + (q + 1) * 4] = qh
    out[:, 208:224] = d.astype(np.float16).reshape(nb, 8).view(np.uint8)
    return out.reshape(-1)


def rotate(V: np.ndarray) -> np.ndarray:
    """(N, nb*1024) -> F(S (.) v) per blocco (= x per una riga di pesi, = g per un h)."""
    N, C = V.shape
    nb = C // QK
    B = np.ascontiguousarray(V, dtype=np.float32).reshape(N * nb, QK) * S1024[None, :]
    return fwht1024_f32(B).reshape(N, C).astype(np.float64)


def rotated_cov(H: np.ndarray | None, C: int, ridge: float = 1e-3) -> np.ndarray:
    """C' = E[g g^T] + ridge * tr/C * I. Senza h: identita' (Frobenius)."""
    if H is None or len(H) == 0:
        return np.eye(C)
    G = rotate(H)
    M = G.T @ G / len(G)
    return M + ridge * np.trace(M) / C * np.eye(C)


def objective(x: np.ndarray, xe: np.ndarray, Cp: np.ndarray) -> float:
    e = x - xe
    return float(e @ Cp @ e)


def _refit_d(t: np.ndarray, xe: np.ndarray, Cp: np.ndarray, d_old: np.ndarray) -> np.ndarray:
    ng = len(d_old)
    Mg = np.zeros((len(t), ng))
    Mg[np.arange(len(t)), np.arange(len(t)) // 128] = t        # x = Mg d
    A = Mg.T @ Cp @ Mg
    b = Mg.T @ Cp @ xe
    live = np.diag(A) > 0
    d = d_old.copy()
    if live.any():
        Al = A[np.ix_(live, live)] + 1e-12 * np.trace(A) * np.eye(live.sum())
        d[live] = np.linalg.solve(Al, b[live])
    neg = d < 0                                                  # il segno sta nei trit: d >= 0
    return np.abs(d).astype(np.float16).astype(np.float64), neg


def cd_row(t0: np.ndarray, d0: np.ndarray, xe: np.ndarray, Cp: np.ndarray, rounds: int = 4,
           sweeps: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    t, d = t0.copy(), d0.copy()
    diag = np.diag(Cp)
    grp = np.arange(len(t)) // 128
    best = (t.copy(), d.copy(), objective(d[grp] * t, xe, Cp))
    for _ in range(rounds):
        for _ in range(sweeps):
            x = d[grp] * t
            r = Cp @ (x - xe)
            moved = 0
            for j in range(len(t)):
                dj = d[grp[j]]
                if dj == 0:
                    continue
                cur = t[j]
                cand = np.array([-1.0, 0.0, 1.0]) - cur
                gain = 2 * dj * cand * r[j] + dj * dj * cand * cand * diag[j]
                k = int(np.argmin(gain))
                if gain[k] < -1e-18 and cand[k] != 0:
                    t[j] += cand[k]
                    r += Cp[:, j] * (dj * cand[k])
                    moved += 1
            if not moved:
                break
        d_new, neg = _refit_d(t, xe, Cp, d)
        if neg.any():                                             # scala negativa: gira i trit del gruppo
            for g in np.nonzero(neg)[0]:
                t[grp == g] *= -1
        obj = objective(d_new[grp] * t, xe, Cp)
        if obj < best[2]:
            best = (t.copy(), d_new.copy(), obj)
            d = d_new
        else:
            break
    return best


def riternarizza(raw0: np.ndarray, W_edit: np.ndarray, H: np.ndarray | None,
                 rounds: int = 4, sweeps: int = 3, Cp: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """raw0 (R, nb*224) righe originali, W_edit (R, C) float -> (R, nb*224) nuove righe + info.
    Se W_edit coincide con la dequantizzazione dell'originale, restituisce i byte originali."""
    R, C = W_edit.shape
    if Cp is None:
        Cp = rotated_cov(H, C)
    Xe = rotate(W_edit)
    out = raw0.copy()
    info = []
    grp = np.arange(C) // 128
    for i in range(R):
        t0, d0 = unpack_row(raw0[i])
        o0 = objective(d0[grp] * t0, Xe[i], Cp)
        rtn = quantize_rows(W_edit[i:i + 1].astype(np.float32)).reshape(-1)
        t1, d1 = unpack_row(rtn)
        o1 = objective(d1[grp] * t1, Xe[i], Cp)
        if o0 <= 1e-10 * max(float(Xe[i] @ Cp @ Xe[i]), 1e-30):  # modifica nulla: byte originali
            info.append({"obj_original": o0, "obj_rtn": o1, "obj_final": o0, "start": "original"})
            continue
        cands = [cd_row(t0, d0, Xe[i], Cp, rounds, sweeps) + ("original",),
                 cd_row(t1, d1, Xe[i], Cp, rounds, sweeps) + ("rtn",)]
        t, d, o, start = min(cands, key=lambda c: c[2])
        if o >= o0:
            info.append({"obj_original": o0, "obj_rtn": o1, "obj_final": o0, "start": "original(no gain)"})
            continue
        out[i] = pack_row(t, d)
        info.append({"obj_original": o0, "obj_rtn": o1, "obj_final": o, "start": start,
                     "trits_changed": int((t != t0).sum()), "residual_frac": o / o0})
    return out, info


def deq_rows(raw: np.ndarray) -> np.ndarray:
    from tq1_0r_1024_codec import dequantize_blocks
    return dequantize_blocks(raw.reshape(-1, BLOCK_BYTES)).reshape(len(raw), -1).astype(np.float32)


def riternarizza_piani(raws: list[np.ndarray], W_edit: np.ndarray, H: np.ndarray | None,
                       passes: int = 2, rounds: int = 4, sweeps: int = 3,
                       ridge: float = 0.05) -> tuple[list[np.ndarray], dict]:
    """Testa Thor a P piani (logit = somma dei piani): ogni piano resta UN piano TQ1_0R_1024.
    Discesa a blocchi: per p = P-1..0 (dal piano a grana piu' fine) si ri-ternarizza il piano p
    verso W_edit - somma degli altri, nella metrica dei logit; l'obiettivo totale non cresce mai."""
    R, C = W_edit.shape
    Cp = rotated_cov(H, C, ridge)     # ridge: tiene W vicina a W_edit anche fuori dagli h di train
    cur = [r.copy() for r in raws]
    Wp = [deq_rows(r) for r in cur]
    grp_obj = lambda: float(sum(((rotate((sum(Wp) - W_edit)[i:i + 1])[0]) @ Cp
                                 @ rotate((sum(Wp) - W_edit)[i:i + 1])[0]) for i in range(R)))
    info = {"obj_start": grp_obj(), "steps": []}
    for _ in range(passes):
        for p in reversed(range(len(cur))):
            others = sum(Wp[q] for q in range(len(cur)) if q != p) if len(cur) > 1 else 0.0
            new, inf = riternarizza(cur[p], W_edit - others, None, rounds, sweeps, Cp=Cp)
            cur[p] = new
            Wp[p] = deq_rows(new)
            info["steps"].append({"plane": p, "rows": inf, "obj_total": grp_obj()})
    info["obj_end"] = grp_obj()
    info["residual_frac"] = info["obj_end"] / max(info["obj_start"], 1e-30)
    return cur, info


if __name__ == "__main__":
    # autotest: pack/unpack bit-esatto e guadagno su un problema sintetico
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((3, 2048)) * 0.02).astype(np.float32)
    raw = quantize_rows(W).reshape(3, -1)
    for i in range(3):
        t, d = unpack_row(raw[i])
        assert np.array_equal(pack_row(t, d), raw[i]), "pack/unpack non bit-esatto"
    from tq1_0r_1024_codec import dequantize_blocks
    Wq = dequantize_blocks(raw.reshape(-1, BLOCK_BYTES)).reshape(3, 2048)
    same, inf0 = riternarizza(raw, Wq, None)
    assert np.array_equal(same, raw), "modifica nulla deve restituire i byte originali"
    H = (rng.standard_normal((500, 2048)) * np.linspace(0.1, 3, 2048)).astype(np.float32)
    b = rng.standard_normal(2048).astype(np.float32); b /= np.linalg.norm(b)
    We = Wq - np.outer(Wq @ b, b)
    new, inf = riternarizza(raw, We, H)
    Wn = dequantize_blocks(new.reshape(-1, BLOCK_BYTES)).reshape(3, 2048)
    Wr = dequantize_blocks(quantize_rows(We).reshape(-1, BLOCK_BYTES)).reshape(3, 2048)
    e = lambda A: float(np.mean(((A - We) @ H.T) ** 2))
    print(f"pack ok, nulla ok; errore logit: originale {e(Wq):.3e}  RTN {e(Wr):.3e}  CD {e(Wn):.3e}")
    print(inf)
