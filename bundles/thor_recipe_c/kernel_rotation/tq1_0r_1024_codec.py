"""
Codec numpy del tipo GGML TQ1_0R_1024 (MOGAVIS Thor-v9, 20/9/2026):
rotazione Hadamard a blocco 1024 con segno fisso S1024, 8 scale fp16 (una
ogni 128 pesi), 4 quarti impacchettati in base 3 come block_tq1_0.

Replica OP PER OP il riferimento C di ggml-quants.c (albero
@@ROOT:models@@/build/llamacpp-tq1-k1024): stessa FWHT in float32 (scala 1/32
prima, 10 stadi, stessa convenzione di segno del butterfly), stessa
ternarizzazione (id = 1/amax NON arrotondato a fp16, prodotto x*id in
float32, arrotondamento half-away-from-zero come lroundf), stesso packing.
Cosi' la forgia Python e la quantizzazione C danno gli STESSI byte, e la
dequantizzazione C e questa danno gli STESSI float (verifica (a) del piano:
round-trip bit-esatto — vedi verifica_cpu_1024.py).

Layout del blocco (224 byte = 1.75 bpw):
  qs[0:192]   4 quarti x 48 byte, 5 trit/byte (elementi q*256 + 0..239)
  qh[192:208] 4 quarti x 4 byte,  4 trit/byte (elementi q*256 + 240..255)
  d[208:224]  8 x fp16, d[g] per gli elementi g*128 .. g*128+127
"""
from __future__ import annotations
import numpy as np

from gen_sign_1024 import genera as genera_s1024

QK = 1024
BLOCK_BYTES = 224
S1024 = genera_s1024().astype(np.float32)   # +-1, (1024,), stessa fonte del C e degli shader
assert S1024.shape == (QK,)

_IDX = np.arange(QK)


def fwht1024_f32(V: np.ndarray) -> np.ndarray:
    """V: (..., 1024) -> (1/32) H_1024 V, in float32, op per op come
    ggml_tq1_0r_1024_fwht1024: scala prima (potenza di 2, esatta), poi 10
    stadi; coppia (i basso, j=i^len alto): nuovo[i]=v[i]+v[j], nuovo[j]=v[i]-v[j]."""
    v = np.ascontiguousarray(V, dtype=np.float32) * np.float32(0.03125)
    lead = v.shape[:-1]
    for s in range(10):
        ln = 1 << s
        # vista (..., 1024/(2*len), 2, len): [...,0,:] = elementi bassi (i), [...,1,:] = alti (i|len)
        v4 = v.reshape(*lead, 1024 // (2 * ln), 2, ln)
        a = v4[..., 0, :]
        b = v4[..., 1, :]
        out = np.empty_like(v4)
        np.add(a, b, out=out[..., 0, :])        # nuovo[i]     = v[i] + v[i|len]
        np.subtract(a, b, out=out[..., 1, :])   # nuovo[i|len] = v[i] - v[i|len]
        v = out.reshape(*lead, 1024)
    return v


def _lroundf(v: np.ndarray) -> np.ndarray:
    """lroundf: half away from zero. In float64 la somma |v|+0.5 e' esatta per
    ogni float32 |v|<=1 (24 bit di mantissa contro 53), quindi il risultato e'
    esattamente quello di lroundf sul valore float32."""
    v64 = v.astype(np.float64)
    return (np.sign(v64) * np.floor(np.abs(v64) + 0.5)).astype(np.int64)


def _pack_quarter(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T: (N,256) trit in {0,1,2} -> (qs (N,48) uint8, qh (N,4) uint8),
    stesso schema di quantize_row_tq1_0_ref (n-major: la cifra n pesa 3^(4-n))."""
    N = T.shape[0]
    qs = np.zeros((N, 48), dtype=np.uint8)
    # bytes 0..31: elementi m + n*32
    q = np.zeros((N, 32), dtype=np.int64)
    for n in range(5):
        q = q * 3 + T[:, n * 32:(n + 1) * 32]
    qs[:, 0:32] = ((q * 256 + 242) // 243) & 0xFF
    # bytes 32..47: elementi 160 + m + n*16
    q = np.zeros((N, 16), dtype=np.int64)
    for n in range(5):
        q = q * 3 + T[:, 160 + n * 16:160 + (n + 1) * 16]
    qs[:, 32:48] = ((q * 256 + 242) // 243) & 0xFF
    # qh: elementi 240 + j + m*4, poi shift di un trit (q *= 3)
    q = np.zeros((N, 4), dtype=np.int64)
    for m in range(4):
        q = q * 3 + T[:, 240 + m * 4:240 + (m + 1) * 4]
    q = q * 3
    qh = (((q * 256 + 242) // 243) & 0xFF).astype(np.uint8)
    return qs, qh


def _unpack_quarter(qs: np.ndarray, qh: np.ndarray) -> np.ndarray:
    """(N,48),(N,4) uint8 -> (N,256) valori {-1,0,+1} float32, ordine naturale
    (stessi loop di dequantize_row_tq1_0, troncamento a 8 bit incluso)."""
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


GRIGLIA_LS = (0.55, 0.66, 0.76, 0.86, 1.00, 1.20)


def scala_ls(G: np.ndarray) -> np.ndarray:
    """Scala ai MINIMI QUADRATI per gruppo (G: (N,8,128) float32 -> d (N,8) float32).
    RTN puro, nessuna Hessiana: la stessa griglia di odino-lab ternario_gpu._scala_un_piano
    (usata per gli esperti di Thor). Per ogni soglia m: vivi = |w| > m*media,
    d = media(|w| vivi) e' la scala ottima per quel supporto; si tiene la d che
    massimizza somma^2/k (= minimizza ||w - d t||^2 a meno di ||w||^2, comune).
    ⛔ NON amax: con amax (quantizzatore TQ1_0, pensato per pesi GIA' ternari)
    la ternarizzazione RTN di un tensore denso perde il 59-62% dell'energia
    (misurato 20/9 su 3 tensori veri di Ornith-1.5) e il modello collassa
    (Thor-vBonsai-completo, 14:32: spazzatura su GPU e su CPU); con la LS 18.7%.
    Aritmetica in float64 sui valori float32 (somme ESATTE: ordine-indipendenti)
    -> identica, bit per bit, al riferimento C quantize_row_tq1_0r_1024_ref."""
    ab = np.abs(G).astype(np.float64)
    somma_all = ab.sum(-1)                                   # esatta in double
    media = somma_all / 128.0
    best_score = np.full(media.shape, -1.0)
    best_d = np.zeros(media.shape)
    for m in GRIGLIA_LS:
        vivi = ab > (m * media)[..., None]
        k = vivi.sum(-1).astype(np.float64)
        somma = np.where(vivi, ab, 0.0).sum(-1)
        score = np.where(k > 0, somma * somma / np.maximum(k, 1.0), 0.0)
        ok = score > best_score                              # strettamente maggiore: a parita' vince la prima m, come in C
        best_score = np.where(ok, score, best_score)
        best_d = np.where(ok, np.where(k > 0, somma / np.maximum(k, 1.0), 0.0), best_d)
    return best_d.astype(np.float32)


def quantize_rotated_blocks(Wrot: np.ndarray) -> np.ndarray:
    """Wrot: (N,1024) float32 GIA' ruotati (S poi H). Ritorna (N,224) uint8.
    Stessa aritmetica di quantize_row_tq1_0r_1024_ref dopo la rotazione."""
    N = Wrot.shape[0]
    G = np.ascontiguousarray(Wrot, dtype=np.float32).reshape(N, 8, 128)
    amax = scala_ls(G)                                                 # (N,8) — nome storico: e' la scala d
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


def quantize_rows(W: np.ndarray, righe_per_pezzo: int = 8192) -> np.ndarray:
    """W: (R, C) float32 con C multiplo di 1024 -> (R*C/1024, 224) uint8:
    segno S1024, FWHT-1024, RTN a 8 scale. La forgia usa questa.
    A pezzi di righe (default 8192 righe = 64 MiB a C=2048) per tenere bassa
    la memoria di picco: token_embd/output (248320 x 2048) in un colpo solo
    costerebbero ~2 GiB per temporaneo, con ~4 temporanei vivi."""
    R, C = W.shape
    assert C % QK == 0, (R, C)
    nb = C // QK
    out = np.empty((R * nb, BLOCK_BYTES), dtype=np.uint8)
    for r0 in range(0, R, righe_per_pezzo):
        r1 = min(R, r0 + righe_per_pezzo)
        Wb = np.ascontiguousarray(W[r0:r1], dtype=np.float32).reshape((r1 - r0) * nb, QK)
        rot = fwht1024_f32(Wb * S1024[None, :])
        out[r0 * nb:r1 * nb] = quantize_rotated_blocks(rot)
    return out


def dequantize_blocks(packed: np.ndarray) -> np.ndarray:
    """(N,224) uint8 -> (N,1024) float32, op per op come dequantize_row_tq1_0r_1024:
    trit * d[g] (float32), FWHT-1024, poi segno S."""
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
    # autoverifiche: involuzione della FWHT (a meno dell'arrotondamento float32),
    # e round-trip quantizza->dequantizza coerente (i trit ricostruiti coincidono).
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, QK)).astype(np.float32)
    y = fwht1024_f32(fwht1024_f32(x))
    print(f"involuzione FWHT-1024 f32: max|H(H x)-x| = {np.abs(y - x).max():.3e}")
    W = (rng.standard_normal((4, 2048)) * 0.05).astype(np.float32)
    P = quantize_rows(W)
    assert P.shape == (8, BLOCK_BYTES)
    Wd = dequantize_blocks(P).reshape(4, 2048)
    P2 = quantize_rows(Wd)   # ri-quantizzare il ricostruito deve dare gli stessi trit (scale a parte: fp16)
    print(f"round-trip: MSE(W,Wd)={np.mean((W - Wd) ** 2):.3e}  trit stabili={np.array_equal(P[:, :208], P2[:, :208])}")
