"""
Porta Python, bit-esatta, di quantize_row_tq1_0_ref / dequantize_row_tq1_0
(ggml/src/ggml-quants.c, fork @@ROOT:home@@/build-llamacpp-tq1). Serve per
generare blocchi TQ1_0 veri (stesso impacchettamento in base 3) da dare in
pasto al kernel Vulkan, e per avere un riferimento CPU indipendente.

QK_K = 256. Un blocco = 48 byte qs (5 valori/byte, basi 3) + 4 byte qh
(4 valori/byte) + 1 float16 d = 53 byte, 256 pesi.
"""
import numpy as np

QK_K = 256
POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.int64)


def quantize_block_tq1_0(w: np.ndarray) -> bytes:
    """w: 256 float32 -> 53 byte (qs[48] + qh[4] + d fp16 = 54 byte), stesso algoritmo
    di quantize_row_tq1_0_ref (arrotondamento a 3 livelli, impacchettamento
    in base 3 a 5 cifre/byte con ceiling division)."""
    assert w.shape == (QK_K,)
    amax = float(np.max(np.abs(w))) if w.size else 0.0
    d = amax
    idd = (1.0 / d) if d != 0.0 else 0.0

    qs = np.zeros(48, dtype=np.uint8)
    qh = np.zeros(4, dtype=np.uint8)

    x = w  # puntatore logico, avanza come nel C
    pos = 0  # indice in x

    def trit(val):
        xi = int(np.round(val * idd)) + 1  # -1,0,1 -> 0,1,2
        return max(0, min(2, xi))

    # primo gruppo: 32 byte, blocchi da 5*32=160
    j = 0
    for m in range(32):
        q = 0
        for n in range(5):
            xi = trit(x[pos + m + n * 32])
            q = q * 3 + xi
        q = (q * 256 + (243 - 1)) // 243
        qs[j + m] = q & 0xFF
    pos += 5 * 32

    # secondo gruppo: 16 byte, blocchi da 5*16=80
    j = 32
    for m in range(16):
        q = 0
        for n in range(5):
            xi = trit(x[pos + m + n * 16])
            q = q * 3 + xi
        q = (q * 256 + (243 - 1)) // 243
        qs[j + m] = q & 0xFF
    pos += 5 * 16

    # qh: 4 byte, 4 valori/byte (shift extra per il quinto trit assente)
    for j2 in range(4):
        q = 0
        for m in range(4):
            xi = trit(x[j2 + m * 4 + pos])
            q = q * 3 + xi
        q *= 3
        q = (q * 256 + (243 - 1)) // 243
        qh[j2] = q & 0xFF
    pos += 4 * 4
    assert pos == 256

    d16 = np.float16(d).tobytes()
    return qs.tobytes() + qh.tobytes() + d16


def dequantize_block_tq1_0(block: bytes) -> np.ndarray:
    """53 byte -> 256 float32, stesso algoritmo di dequantize_row_tq1_0."""
    qs = np.frombuffer(block[0:48], dtype=np.uint8)
    qh = np.frombuffer(block[48:52], dtype=np.uint8)
    d = float(np.frombuffer(block[52:54], dtype=np.float16)[0])

    out = np.zeros(256, dtype=np.float32)
    pow3 = [1, 3, 9, 27, 81]
    e = 0
    # primi 32 byte -> 160 valori (n major, m minor, come il C: for n in 0..5: for m in 0..32)
    for n in range(5):
        for m in range(32):
            q = (int(qs[m]) * pow3[n]) & 0xFF
            xi = (q * 3) >> 8
            out[e] = float(xi - 1) * d
            e += 1
    # successivi 16 byte -> 80 valori
    for n in range(5):
        for m in range(16):
            q = (int(qs[32 + m]) * pow3[n]) & 0xFF
            xi = (q * 3) >> 8
            out[e] = float(xi - 1) * d
            e += 1
    # qh -> 16 valori
    for n in range(4):
        for j in range(4):
            q = (int(qh[j]) * pow3[n]) & 0xFF
            xi = (q * 3) >> 8
            out[e] = float(xi - 1) * d
            e += 1
    assert e == 256
    return out


def tq1_val_index_map():
    """Ricostruisce, per ogni indice e in [0,256), (t, byte_kind, byte_idx)
    come lo fa tq1_0_val() nel GLSL — usato per verificare che la mappa
    Python coincida esattamente con lo shader."""
    m = []
    for e in range(256):
        if e < 160:
            t = e // 32
            kind, idx = "qs", e % 32
        elif e < 240:
            e2 = e - 160
            t = e2 // 16
            kind, idx = "qs", 32 + (e2 % 16)
        else:
            e3 = e - 240
            t = e3 // 4
            kind, idx = "qh", e3 % 4
        m.append((t, kind, idx))
    return m


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w = rng.standard_normal(256).astype(np.float32) * 0.05
    blk = quantize_block_tq1_0(w)
    assert len(blk) == 54, len(blk)
    w_hat = dequantize_block_tq1_0(blk)
    err = np.abs(w - w_hat)
    print(f"encode/decode roundtrip: max|w|={np.max(np.abs(w)):.4f}  "
          f"MAE={err.mean():.5f}  MSE={np.mean(err**2):.6f}")
    # confronto indice-a-indice fra ordine dequant "n,m" e ordine GLSL "e"
    m1 = tq1_val_index_map()
    print("prime 5 mappe (t, kind, idx):", m1[:5], "...", m1[159:162], "...", m1[253:256])
