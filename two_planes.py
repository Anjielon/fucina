"""JOINT two-plane ternary quantization (PTQTP) + GPTQ.

    W  ≈  d1 ⊙ T1  +  d2 ⊙ T2        with T1, T2 ∈ {-1, 0, +1}

3.38 bits per weight. Measured on a real tensor of a 397B model:

    one plane, naive            43.54%
    two planes, SEQUENTIAL      20.40%   (first plane, then its residual)
    two planes, JOINT           17.26%   <- this module

The joint solution is not the sequential one refined: the two planes are
optimized together, alternating the closed-form scales with the sign search.

Companion caveat, learned the hard way: the plane-1 of a joint solution is
co-adapted to its partner. If only one plane will be stored, quantize it
DEDICATED instead — using the joint plane-1 alone costs about 10 points.
"""
from __future__ import annotations
import numpy as np

BLOCK = 256


# ─────────────────────────────────────────────────────────────────────────
def _joint_scales(W: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                     lam: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Given the two ternary grids, the optimal PER-ROW pair of scales.

    Minimizza ‖W − d1·T1 − d2·T2‖². Derivando: sistema 2×2
        [ΣT1²   ΣT1T2] [d1]   [ΣWT1]
        [ΣT1T2  ΣT2² ] [d2] = [ΣWT2]
    risolto in forma chiusa (niente inversione di matrice).
    """
    a11 = (T1 * T1).sum(-1) + lam
    a22 = (T2 * T2).sum(-1) + lam
    a12 = (T1 * T2).sum(-1)
    b1 = (W * T1).sum(-1)
    b2 = (W * T2).sum(-1)
    det = a11 * a22 - a12 * a12
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    d1 = (b1 * a22 - b2 * a12) / det
    d2 = (b2 * a11 - b1 * a12) / det
    return d1[..., None], d2[..., None]


def _assign_nine(W: np.ndarray, d1: np.ndarray, d2: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Assign each weight to the best of the NINE (t1,t2) ∈ {-1,0,1}² pairs.

    Forza bruta su 9 combinazioni: e' esatto e costa 9 sottrazioni.
    """
    migliore = None
    err_min = None
    for t1 in (-1.0, 0.0, 1.0):
        for t2 in (-1.0, 0.0, 1.0):
            e = (W - (d1 * t1 + d2 * t2)) ** 2
            if err_min is None:
                err_min = e
                migliore = np.full(W.shape, t1 * 3 + t2 + 4, dtype=np.int8)
            else:
                m = e < err_min
                err_min = np.where(m, e, err_min)
                migliore = np.where(m, np.int8(t1 * 3 + t2 + 4), migliore)
    T1 = (migliore // 3 - 1).astype(np.float32)
    T2 = (migliore % 3 - 1).astype(np.float32)
    return T1, T2


def joint_planes(W: np.ndarray, rounds: int = 8) -> tuple:
    """W: (n_righe, 256) → (d1, T1, d2, T2). Ottimizzazione alternata.

    ⚠️ THE INITIALIZATION MATTERS. Starting from T2 = 0 makes the 2×2 system
    singular and d2 stays at zero forever — the method silently degenerates
    into a single plane. So we start from the SEQUENTIAL solution (plane one
    on the signal, plane two on the residual) and only then alternate, which
    guarantees the iterations can only improve.
    """
    from ternario_ottimo import scala_e_segni
    W = np.asarray(W, np.float32)
    d1, q1 = scala_e_segni(W)
    T1 = q1.astype(np.float32)
    res = W - d1 * T1
    d2, q2 = scala_e_segni(res)
    T2 = q2.astype(np.float32)
    for _ in range(rounds):
        d1, d2 = _joint_scales(W, T1, T2)
        T1n, T2n = _assign_nine(W, d1, d2)
        if np.array_equal(T1n, T1) and np.array_equal(T2n, T2):
            break
        T1, T2 = T1n, T2n
    d1, d2 = _joint_scales(W, T1, T2)
    return d1, T1.astype(np.int8), d2, T2.astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────
def prepare_hessian(H: np.ndarray, smorzamento: float = 0.01) -> np.ndarray:
    """From H = XXᵀ to the triangular factor GPTQ uses to propagate error.

    ⚠️ Damping is not cosmetic: H is near-singular (a few thousand samples
    for 4096 dimensions) and without the diagonal term the Cholesky either
    fails or explodes. One percent of the mean trace is the standard choice.
    """
    H = np.array(H, dtype=np.float64, copy=True)
    n = H.shape[0]
    morti = np.diag(H) == 0
    H[morti, morti] = 1.0
    H += np.eye(n) * (smorzamento * np.mean(np.diag(H)))
    L = np.linalg.cholesky(H)
    Li = np.linalg.inv(L)
    Hinv = Li.T @ Li                     # H^-1
    # ⛔ GPTQ needs the UPPER factor U with Uᵀ U = H⁻¹. numpy returns the
    #    lower one (H⁻¹ = L Lᵀ), so U = Lᵀ and nothing else. The index-reversal
    #    trick written here first is a DIFFERENT factorization: GPTQ then ran
    #    and improved nothing at all (17.43% with it and without it). A wrong
    #    factor does not crash — it silently disables the method.
    return np.linalg.cholesky(Hinv).T.copy()


def gptq_two_planes(W: np.ndarray, Hchol: np.ndarray) -> tuple:
    """GPTQ over 256-weight blocks with two jointly optimized planes.

    W: (n_out, n_in) · Hchol: da prepare_hessian(H), (n_in, n_in)
    → (d1, T1, d2, T2) with d shaped (n_out, n_blocks, 1)
    """
    W = np.array(W, dtype=np.float32, copy=True)
    n_out, n_in = W.shape
    assert n_in % BLOCK == 0, f"{n_in} is not a multiple of {BLOCK}"
    nb = n_in // BLOCK
    D1 = np.zeros((n_out, nb, 1), np.float32); D2 = np.zeros_like(D1)
    Q1 = np.zeros((n_out, n_in), np.int8);     Q2 = np.zeros_like(Q1)

    for b in range(nb):
        i0, i1 = b * BLOCK, (b + 1) * BLOCK
        Wb = W[:, i0:i1]
        # scales fixed on the block BEFORE the sweep (TQ1_0 stores one per block)
        d1, t1, d2, t2 = joint_planes(Wb)
        D1[:, b], D2[:, b] = d1, d2
        Wb = Wb.copy()
        err = np.zeros_like(Wb)
        for j in range(BLOCK):
            w = Wb[:, j]
            # miglior valore fra i nove, a scale fisse
            best = None; emin = None
            for a in (-1.0, 0.0, 1.0):
                for c in (-1.0, 0.0, 1.0):
                    v = d1[:, 0] * a + d2[:, 0] * c
                    e = (w - v) ** 2
                    if emin is None:
                        emin = e; best = np.full(n_out, a * 3 + c + 4, np.int8)
                    else:
                        m = e < emin
                        emin = np.where(m, e, emin)
                        best = np.where(m, np.int8(a * 3 + c + 4), best)
            a = (best // 3 - 1); c = (best % 3 - 1)
            Q1[:, i0 + j] = a; Q2[:, i0 + j] = c
            deq = d1[:, 0] * a + d2[:, 0] * c
            e = (w - deq) / Hchol[i0 + j, i0 + j]
            err[:, j] = e
            if j + 1 < BLOCK:   # propagate the error forward, inside the block
                Wb[:, j+1:] -= np.outer(e, Hchol[i0 + j, i0 + j + 1:i1])
        # and propagate across the following blocks as well
        if i1 < n_in:
            W[:, i1:] -= err @ Hchol[i0:i1, i1:]
    return D1, Q1, D2, Q2


def relative_error(W, D1, Q1, D2, Q2, H=None) -> float:
    n_out, n_in = W.shape
    nb = n_in // BLOCK
    ric = (D1 * Q1.reshape(n_out, nb, BLOCK) + D2 * Q2.reshape(n_out, nb, BLOCK)).reshape(n_out, n_in)
    d = W - ric
    if H is None:
        return float(np.linalg.norm(d) / max(np.linalg.norm(W), 1e-12))
    return float(np.sqrt(np.trace(d @ H @ d.T) / max(np.trace(W @ H @ W.T), 1e-30)))
