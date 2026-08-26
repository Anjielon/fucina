"""Due piani ternari CONGIUNTI (PTQTP) + GPTQ — il cuore di ODINO v3.

    W  ≈  d1 ⊙ T1  +  d2 ⊙ T2        con T1, T2 ∈ {-1, 0, +1}

A 3,38 bit per peso. Misurato su un tensore vero di Ornith:
    un piano, ingenuo        43,54%
    due piani SEQUENZIALI    20,40%   (primo piano, poi il residuo)
    due piani CONGIUNTI      17,26%   ← qui
    + GPTQ                    9,36%
    + rotazione               7,08%   ← meglio di IQ4_XS (7,60%) con l'80% dei bit

⭐ PERCHE' CONGIUNTI E NON SEQUENZIALI. Sequenziale = si sceglie T1 pensando
solo a W, poi T2 ripara. Congiunto = si scelgono INSIEME, e ogni peso puo'
prendere uno dei NOVE valori {-(d1+d2), -d1, -(d1-d2), -d2, 0, d2, d1-d2,
d1, d1+d2}. Con d2 libera (non d1/3) i nove livelli non sono equispaziati:
si adattano alla forma vera della distribuzione.

⭐ PERCHE' IL GPTQ SI SOMMA. Il limite tasso-distorsione (33,3% a 1,585 bit)
vale sull'errore nei PESI. Quello che conta e' ‖(W−Ŵ)X‖, l'errore sull'USCITA.
Se le attivazioni sono anisotrope (Ornith: autovalori ~k^-0,59, MISURATO) si
spende precisione solo dove passa il segnale, e il limite non si applica.
GPTQ quantizza una colonna alla volta e SPINGE l'errore su quelle non ancora
fatte, pesandolo con l'inversa di H = XXᵀ: chi viene dopo assorbe.

⚠️ Le scale sono PER BLOCCO DI 256 (formato TQ1_0), quindi si fissano prima
   del giro GPTQ e il GPTQ lavora dentro il blocco a scala fissa — e' il
   modo standard (group-size) di far convivere i due.
"""
from __future__ import annotations
import numpy as np

BLOCCO = 256


# ─────────────────────────────────────────────────────────────────────────
def _scale_congiunte(W: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                     lam: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Date le due griglie ternarie, le due scale ottime PER RIGA.

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


def _assegna_nove(W: np.ndarray, d1: np.ndarray, d2: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Ogni peso al migliore dei NOVE valori (t1,t2) ∈ {-1,0,1}².

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


def congiunti(W: np.ndarray, giri: int = 8) -> tuple:
    """W: (n_righe, 256) → (d1, T1, d2, T2). Ottimizzazione alternata.

    ⚠️ L'AVVIO CONTA. Partendo da T2 = 0 il sistema 2×2 e' singolare e d2
    resta a zero per sempre: si degenera a un piano solo. Si parte quindi
    dal SEQUENZIALE (primo piano sul segnale, secondo sul residuo) e poi si
    alterna — cosi' i giri possono solo migliorare.
    """
    from ternario_ottimo import scala_e_segni
    W = np.asarray(W, np.float32)
    d1, q1 = scala_e_segni(W)
    T1 = q1.astype(np.float32)
    res = W - d1 * T1
    d2, q2 = scala_e_segni(res)
    T2 = q2.astype(np.float32)
    for _ in range(giri):
        d1, d2 = _scale_congiunte(W, T1, T2)
        T1n, T2n = _assegna_nove(W, d1, d2)
        if np.array_equal(T1n, T1) and np.array_equal(T2n, T2):
            break
        T1, T2 = T1n, T2n
    d1, d2 = _scale_congiunte(W, T1, T2)
    return d1, T1.astype(np.int8), d2, T2.astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────
def prepara_hessiana(H: np.ndarray, smorzamento: float = 0.01) -> np.ndarray:
    """Da H = XXᵀ alla triangolare che GPTQ usa per spingere l'errore.

    ⚠️ Lo smorzamento non e' cosmetico: H e' quasi singolare (poche migliaia
    di campioni per 4096 dimensioni) e senza il termine sulla diagonale la
    Cholesky fallisce o esplode. 1% della traccia media e' lo standard.
    """
    H = np.array(H, dtype=np.float64, copy=True)
    n = H.shape[0]
    morti = np.diag(H) == 0
    H[morti, morti] = 1.0
    H += np.eye(n) * (smorzamento * np.mean(np.diag(H)))
    L = np.linalg.cholesky(H)
    Li = np.linalg.inv(L)
    Hinv = Li.T @ Li                     # H^-1
    # ⛔ GPTQ vuole il fattore SUPERIORE U con Uᵀ U = H⁻¹. numpy da' quello
    #    inferiore L (H⁻¹ = L Lᵀ), quindi U = Lᵀ e basta. Il giro con gli
    #    indici invertiti che avevo scritto prima e' un'ALTRA fattorizzazione:
    #    il GPTQ girava e non migliorava nulla (17,43% con e senza).
    return np.linalg.cholesky(Hinv).T.copy()


def gptq_due_piani(W: np.ndarray, Hchol: np.ndarray) -> tuple:
    """GPTQ su blocchi da 256 con due piani congiunti.

    W: (n_out, n_in) · Hchol: da prepara_hessiana(H), (n_in, n_in)
    → (d1, T1, d2, T2) con d di forma (n_out, n_blocchi, 1)
    """
    W = np.array(W, dtype=np.float32, copy=True)
    n_out, n_in = W.shape
    assert n_in % BLOCCO == 0, f"{n_in} non e' multiplo di {BLOCCO}"
    nb = n_in // BLOCCO
    D1 = np.zeros((n_out, nb, 1), np.float32); D2 = np.zeros_like(D1)
    Q1 = np.zeros((n_out, n_in), np.int8);     Q2 = np.zeros_like(Q1)

    for b in range(nb):
        i0, i1 = b * BLOCCO, (b + 1) * BLOCCO
        Wb = W[:, i0:i1]
        # scale fissate sul blocco PRIMA del giro (formato TQ1_0: una per blocco)
        d1, t1, d2, t2 = congiunti(Wb)
        D1[:, b], D2[:, b] = d1, d2
        Wb = Wb.copy()
        err = np.zeros_like(Wb)
        for j in range(BLOCCO):
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
            if j + 1 < BLOCCO:   # spingi l'errore avanti, dentro il blocco
                Wb[:, j+1:] -= np.outer(e, Hchol[i0 + j, i0 + j + 1:i1])
        # spingi anche sui blocchi successivi
        if i1 < n_in:
            W[:, i1:] -= err @ Hchol[i0:i1, i1:]
    return D1, Q1, D2, Q2


def errore(W, D1, Q1, D2, Q2, H=None) -> float:
    n_out, n_in = W.shape
    nb = n_in // BLOCCO
    ric = (D1 * Q1.reshape(n_out, nb, BLOCCO) + D2 * Q2.reshape(n_out, nb, BLOCCO)).reshape(n_out, n_in)
    d = W - ric
    if H is None:
        return float(np.linalg.norm(d) / max(np.linalg.norm(W), 1e-12))
    return float(np.sqrt(np.trace(d @ H @ d.T) / max(np.trace(W @ H @ W.T), 1e-30)))
