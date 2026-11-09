"""
Walsh-Hadamard veloce (FWHT), stessa ricorrenza del butterfly GLSL in
~/tony-forgia/rot256_shader/tq1_derot.comp — cosi' la verifica di correttezza
copre anche la CONVENZIONE DI SEGNO dichiarata come debito in quello script
(30/8): "lo shader usa (b-a) nel ramo alto: verificare, non fidarsi a mente".

Convenzione dello shader, per una coppia (i basso, j=i+len alto):
    nuovo[i] = x[i] + x[j]
    nuovo[j] = x[i] - x[j]      (letto nello shader come b-a con a=x[j], b=x[i])
Questa e' ESATTAMENTE la forma Sylvester classica H_{2n} = [[H_n, H_n],[H_n,-H_n]].
Normalizzazione 1/sqrt(256) applicata UNA volta, dopo le 8 passate.
"""
import numpy as np


def fwht_shader_convention(x: np.ndarray) -> np.ndarray:
    """Replica ESATTA, passata per passata con barriere simulate, del kernel
    GLSL: 8 stadi su un blocco di 256, poi normalizzazione 1/16."""
    assert x.shape[-1] == 256
    v = x.astype(np.float64).copy()
    n = 256
    length = 1
    while length < n:
        # vettorizzato: per ogni coppia (i, i+length) con i che ha il bit
        # `length` a zero, aggiorna entrambi SIMULTANEAMENTE (come le barriere
        # del compute shader: tutti leggono i valori vecchi prima che uno scriva)
        idx = np.arange(n)
        low_mask = (idx & length) == 0
        pair = idx ^ length
        a = v[..., idx]
        b = v[..., pair]
        new = np.where(low_mask, a + b, b - a)
        v = new
        length <<= 1
    return (v / 16.0).astype(np.float32)  # 1/sqrt(256) = 1/16


def fwht_matrix(n: int = 256) -> np.ndarray:
    """Costruzione esplicita della matrice di Hadamard normalizzata (Sylvester),
    per un controllo incrociato indipendente dal butterfly."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x = rng.standard_normal(256).astype(np.float32)

    # 1) involuzione: U(U(x)) == x
    y = fwht_shader_convention(x)
    z = fwht_shader_convention(y)
    err_inv = np.abs(z - x).max()
    print(f"[1] involuzione butterfly:           scarto max = {err_inv:.3e}  "
          f"({'OK' if err_inv < 1e-4 else 'FALLITO'})")

    # 2) il butterfly == la matrice esplicita?
    H = fwht_matrix(256)
    y_mat = (H @ x.astype(np.float64)).astype(np.float32)
    err_mat = np.abs(y - y_mat).max()
    print(f"[2] butterfly == matrice densa:      scarto max = {err_mat:.3e}  "
          f"({'OK' if err_mat < 1e-3 else 'FALLITO'})")

    # 3) norma conservata (trasformazione ortogonale)
    n0 = np.linalg.norm(x.astype(np.float64))
    n1 = np.linalg.norm(y.astype(np.float64))
    print(f"[3] norma conservata:                |x|={n0:.6f}  |Ux|={n1:.6f}  "
          f"delta={abs(n0-n1):.3e}  ({'OK' if abs(n0-n1) < 1e-3 else 'FALLITO'})")

    # 4) matrice ortogonale (H^2 = I)
    err_orth = np.abs(H @ H - np.eye(256)).max()
    print(f"[4] H^2 == I (matrice esplicita):    scarto max = {err_orth:.3e}  "
          f"({'OK' if err_orth < 1e-9 else 'FALLITO'})")
