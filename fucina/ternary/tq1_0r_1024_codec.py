"""
numpy codec of the GGML type TQ1_0R_1024 (Thor-v9, 2026-09-20):
1024-block Hadamard rotation with fixed sign S1024, 8 fp16 scales (one
every 128 weights), 4 quarters packed in base 3 like block_tq1_0.

Replicates OP BY OP the C reference in ggml-quants.c (the k1024 llama.cpp
build tree): same float32 FWHT (1/32 scale first, 10 stages, same butterfly
sign convention), same ternarization (id = 1/amax NOT rounded to fp16,
product x*id in float32, half-away-from-zero rounding like lroundf), same
packing. Thus the Python forge and the C quantization produce the SAME bytes,
and the C dequantization and this one produce the SAME floats (check (a) of
the plan: bit-exact round trip — see verifica_cpu_1024.py).

Block layout (224 bytes = 1.75 bpw):
  qs[0:192]   4 quarters x 48 bytes, 5 trits/byte (elements q*256 + 0..239)
  qh[192:208] 4 quarters x 4 bytes,  4 trits/byte (elements q*256 + 240..255)
  d[208:224]  8 x fp16, d[g] for elements g*128 .. g*128+127
"""
from __future__ import annotations
import numpy as np

from fucina.ternary.gen_sign_1024 import generate as generate_s1024

QK = 1024
BLOCK_BYTES = 224
S1024 = generate_s1024().astype(np.float32)   # +-1, (1024,), same source as the C code and the shaders
assert S1024.shape == (QK,)

_IDX = np.arange(QK)


def fwht1024_f32(V: np.ndarray) -> np.ndarray:
    """V: (..., 1024) -> (1/32) H_1024 V, in float32, op by op like
    ggml_tq1_0r_1024_fwht1024: scale first (power of 2, exact), then 10
    stages; pair (i low, j=i^len high): new[i]=v[i]+v[j], new[j]=v[i]-v[j]."""
    v = np.ascontiguousarray(V, dtype=np.float32) * np.float32(0.03125)
    lead = v.shape[:-1]
    for s in range(10):
        ln = 1 << s
        # view (..., 1024/(2*len), 2, len): [...,0,:] = low elements (i), [...,1,:] = high (i|len)
        v4 = v.reshape(*lead, 1024 // (2 * ln), 2, ln)
        a = v4[..., 0, :]
        b = v4[..., 1, :]
        out = np.empty_like(v4)
        np.add(a, b, out=out[..., 0, :])        # new[i]     = v[i] + v[i|len]
        np.subtract(a, b, out=out[..., 1, :])   # new[i|len] = v[i] - v[i|len]
        v = out.reshape(*lead, 1024)
    return v


def _lroundf(v: np.ndarray) -> np.ndarray:
    """lroundf: half away from zero. In float64 the sum |v|+0.5 is exact for
    every float32 |v|<=1 (24 mantissa bits versus 53), so the result is
    exactly that of lroundf on the float32 value."""
    v64 = v.astype(np.float64)
    return (np.sign(v64) * np.floor(np.abs(v64) + 0.5)).astype(np.int64)


def _pack_quarter(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T: (N,256) trits in {0,1,2} -> (qs (N,48) uint8, qh (N,4) uint8),
    same scheme as quantize_row_tq1_0_ref (n-major: digit n weighs 3^(4-n))."""
    N = T.shape[0]
    qs = np.zeros((N, 48), dtype=np.uint8)
    # bytes 0..31: elements m + n*32
    q = np.zeros((N, 32), dtype=np.int64)
    for n in range(5):
        q = q * 3 + T[:, n * 32:(n + 1) * 32]
    qs[:, 0:32] = ((q * 256 + 242) // 243) & 0xFF
    # bytes 32..47: elements 160 + m + n*16
    q = np.zeros((N, 16), dtype=np.int64)
    for n in range(5):
        q = q * 3 + T[:, 160 + n * 16:160 + (n + 1) * 16]
    qs[:, 32:48] = ((q * 256 + 242) // 243) & 0xFF
    # qh: elements 240 + j + m*4, then a one-trit shift (q *= 3)
    q = np.zeros((N, 4), dtype=np.int64)
    for m in range(4):
        q = q * 3 + T[:, 240 + m * 4:240 + (m + 1) * 4]
    q = q * 3
    qh = (((q * 256 + 242) // 243) & 0xFF).astype(np.uint8)
    return qs, qh


def _unpack_quarter(qs: np.ndarray, qh: np.ndarray) -> np.ndarray:
    """(N,48),(N,4) uint8 -> (N,256) values {-1,0,+1} float32, natural order
    (same loops as dequantize_row_tq1_0, 8-bit truncation included)."""
    N = qs.shape[0]
    out = np.zeros((N, 256), dtype=np.float32)
    pow3 = np.array([1, 3, 9, 27, 81], dtype=np.int64)
    b = qs[:, 0:32].astype(np.int64)
    for n in range(5):
        xi = (((b * pow3[n]) & 255) * 3) >> 8
        out[:, n * 32:(n + 1) * 32] = xi - 1
    b = qs[:, 32:48].astype(np.int64)
    for n in range(5):
        xi = (((b * pow3[n]) & 255) * 3) >> 8
        out[:, 160 + n * 16:160 + (n + 1) * 16] = xi - 1
    b = qh.astype(np.int64)
    for n in range(4):
        xi = (((b * pow3[n]) & 255) * 3) >> 8
        out[:, 240 + n * 4:240 + (n + 1) * 4] = xi - 1
    return out


GRID_LS = (0.55, 0.66, 0.76, 0.86, 1.00, 1.20)


def scale_ls(G: np.ndarray) -> np.ndarray:
    """LEAST-SQUARES scale per group (G: (N,8,128) float32 -> d (N,8) float32).
    Pure RTN, no Hessian: the same grid as the single-plane scale of the GPU ternary
    forge (used for the Thor experts). For each threshold m: live = |w| > m*mean,
    d = mean(|w| live) is the optimal scale for that support; the d that maximizes
    sum^2/k is kept (= minimizes ||w - d t||^2 up to ||w||^2, which is common).
    ⛔ NOT amax: with amax (the TQ1_0 quantizer, designed for weights that are ALREADY
    ternary) the RTN ternarization of a dense tensor loses 59-62% of the energy
    (measured 2026-09-20 on 3 real tensors of Ornith-1.5) and the model collapses
    (Thor-vBonsai-completo: garbage on GPU and on CPU); with LS, 18.7%.
    Arithmetic in float64 on float32 values (EXACT sums: order-independent)
    -> identical, bit for bit, to the C reference quantize_row_tq1_0r_1024_ref."""
    ab = np.abs(G).astype(np.float64)
    sum_all = ab.sum(-1)                                   # exact in double
    mean = sum_all / 128.0
    best_score = np.full(mean.shape, -1.0)
    best_d = np.zeros(mean.shape)
    for m in GRID_LS:
        live = ab > (m * mean)[..., None]
        k = live.sum(-1).astype(np.float64)
        sum_ = np.where(live, ab, 0.0).sum(-1)
        score = np.where(k > 0, sum_ * sum_ / np.maximum(k, 1.0), 0.0)
        ok = score > best_score                              # strictly greater: on ties the earlier m wins, as in C
        best_score = np.where(ok, score, best_score)
        best_d = np.where(ok, np.where(k > 0, sum_ / np.maximum(k, 1.0), 0.0), best_d)
    return best_d.astype(np.float32)


def quantize_rotated_blocks(Wrot: np.ndarray) -> np.ndarray:
    """Wrot: (N,1024) float32 ALREADY rotated (S then H). Returns (N,224) uint8.
    Same arithmetic as quantize_row_tq1_0r_1024_ref after the rotation."""
    N = Wrot.shape[0]
    G = np.ascontiguousarray(Wrot, dtype=np.float32).reshape(N, 8, 128)
    amax = scale_ls(G)                                                 # (N,8) — historical name: it is the scale d
    safe = np.where(amax == 0, np.float32(1), amax).astype(np.float32)
    idd = np.where(amax != 0, np.float32(1.0) / safe, np.float32(0)).astype(np.float32)  # 1.0f/d in float32
    Xn = (G * idd[:, :, None]).astype(np.float32).reshape(N, 4, 256)  # x*id in float32
    T = np.clip(_lroundf(Xn) + 1, 0, 2)
    out = np.zeros((N, BLOCK_BYTES), dtype=np.uint8)
    for q in range(4):
        qs, qh = _pack_quarter(T[:, q, :])
        out[:, q * 48:(q + 1) * 48] = qs
        out[:, 192 + q * 4:192 + (q + 1) * 4] = qh
    out[:, 208:224] = amax.astype(np.float16).view(np.uint8).reshape(N, 16)
    return out


def quantize_rows(W: np.ndarray, rows_per_piece: int = 8192) -> np.ndarray:
    """W: (R, C) float32 with C a multiple of 1024 -> (R*C/1024, 224) uint8:
    sign S1024, FWHT-1024, RTN with 8 scales. The forge uses this one.
    In pieces of rows (default 8192 rows = 64 MiB at C=2048) to keep peak memory
    low: token_embd/output (248320 x 2048) in a single shot would cost
    ~2 GiB per temporary, with ~4 temporaries alive."""
    R, C = W.shape
    assert C % QK == 0, (R, C)
    nb = C // QK
    out = np.empty((R * nb, BLOCK_BYTES), dtype=np.uint8)
    for r0 in range(0, R, rows_per_piece):
        r1 = min(R, r0 + rows_per_piece)
        Wb = np.ascontiguousarray(W[r0:r1], dtype=np.float32).reshape((r1 - r0) * nb, QK)
        rot = fwht1024_f32(Wb * S1024[None, :])
        out[r0 * nb:r1 * nb] = quantize_rotated_blocks(rot)
    return out


def dequantize_blocks(packed: np.ndarray) -> np.ndarray:
    """(N,224) uint8 -> (N,1024) float32, op by op like dequantize_row_tq1_0r_1024:
    trit * d[g] (float32), FWHT-1024, then sign S."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    N = packed.shape[0]
    raw = np.zeros((N, QK), dtype=np.float32)
    for q in range(4):
        raw[:, q * 256:(q + 1) * 256] = _unpack_quarter(packed[:, q * 48:(q + 1) * 48], packed[:, 192 + q * 4:192 + (q + 1) * 4])
    d = packed[:, 208:224].copy().view(np.float16).astype(np.float32).reshape(N, 8)   # (N,8)
    raw = (raw.reshape(N, 8, 128) * d[:, :, None]).astype(np.float32).reshape(N, QK)
    v = fwht1024_f32(raw)
    return (v * S1024[None, :]).astype(np.float32)


if __name__ == "__main__":
    # self-checks: FWHT involution (up to float32 rounding),
    # and a consistent quantize->dequantize round trip (the reconstructed trits match).
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, QK)).astype(np.float32)
    y = fwht1024_f32(fwht1024_f32(x))
    print(f"FWHT-1024 f32 involution: max|H(H x)-x| = {np.abs(y - x).max():.3e}")
    W = (rng.standard_normal((4, 2048)) * 0.05).astype(np.float32)
    P = quantize_rows(W)
    assert P.shape == (8, BLOCK_BYTES)
    Wd = dequantize_blocks(P).reshape(4, 2048)
    P2 = quantize_rows(Wd)   # requantizing the reconstruction must give the same trits (scales aside: fp16)
    print(f"round-trip: MSE(W,Wd)={np.mean((W - Wd) ** 2):.3e}  stable trits={np.array_equal(P[:, :208], P2[:, :208])}")
