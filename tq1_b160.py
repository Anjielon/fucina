#!/usr/bin/env python3
"""TQ1_B160 — reference encoder/decoder for block-160 ternary.

Layout per block (160 elements): 32 bytes of base-3 payload (5 trits per
byte, TQ1_0-style digit order) + one f16 scale = 34 bytes = 1.70 bpw.

Target: embedding tables whose row length is a multiple of 160 (the
Flash-Next n-gram table is exactly [160, N]). One block = one row = one
scale, which is the natural granularity for a lookup table.

The scale uses the same two-level optimal search as the forge — NOT abs-max,
which the stock TQ1_0 quantizer uses and which zeroes 87% of weights.
"""
from __future__ import annotations

import numpy as np

BLOCK = 160
PAYLOAD = 32          # 160 / 5 trits per byte
BYTES = PAYLOAD + 2   # + f16 scale

_SCALE_GRID = (0.55, 0.66, 0.76, 0.86, 1.00, 1.20)


def _optimal_scale(rows: np.ndarray) -> np.ndarray:
    """Per-row scale minimising ||W - d*round(clip(W/d))||^2 over a grid
    anchored to the mean absolute value (the forge's search, vectorised)."""
    base = np.abs(rows).mean(axis=1, keepdims=True) * 1.5
    best_d = np.full((rows.shape[0], 1), 1e-12)
    best_e = np.full((rows.shape[0], 1), np.inf)
    for g in _SCALE_GRID:
        d = np.maximum(base * g, 1e-12)
        q = np.clip(np.round(rows / d), -1, 1)
        e = np.square(rows - d * q).sum(axis=1, keepdims=True)
        take = e < best_e
        best_d = np.where(take, d, best_d)
        best_e = np.where(take, e, best_e)
    return best_d


def encode(W: np.ndarray, d_fixed: np.ndarray | None = None) -> np.ndarray:
    """W (n_rows, k*160) float → raw bytes (n_rows * k * 34,) uint8.

    NOT idempotent by design: the scale anchors to mean(|W|), which shifts
    after a decode (only d*{-1,0,1} values remain), so a re-encode may pick
    a different grid point. Quantization is applied once, from the source
    precision; callers that need a reproducible scale (the forge's own
    search, or a validation harness) pass `d_fixed` (n_blocks, 1)."""
    n, m = W.shape
    assert m % BLOCK == 0, f"row length {m} not a multiple of {BLOCK}"
    B = W.reshape(-1, BLOCK).astype(np.float32)
    d = d_fixed.reshape(-1, 1).astype(np.float32) if d_fixed is not None \
        else _optimal_scale(B)
    q = np.clip(np.round(B / d), -1, 1).astype(np.int8) + 1   # {0,1,2}
    # pack 5 trits per byte, digit t extracted as (q*pow3[t]*3)>>8 like TQ1_0:
    # byte = sum_t trit_t * 3^(4-t) * ... — use the same convention: the
    # decoder computes trit_t = ((byte * 3^t) & 255) * 3 >> 8, which holds
    # when byte = sum_{t=0..4} trit_t * 243 / 3^t / ... ; the simple exact
    # construction is byte = q0*81 + q1*27 + q2*9 + q3*3 + q4, then the
    # decoder uses pow3 = [1,3,9,27,81] with digit index reversed.
    Q = q.reshape(-1, PAYLOAD, 5)
    byte = (Q[..., 0] * 81 + Q[..., 1] * 27 + Q[..., 2] * 9
            + Q[..., 3] * 3 + Q[..., 4]).astype(np.uint8)
    out = np.empty((B.shape[0], BYTES), dtype=np.uint8)
    out[:, :PAYLOAD] = byte
    out[:, PAYLOAD:] = d.astype(np.float16).view(np.uint8)
    return out.reshape(-1)


def decode(raw: np.ndarray, n_rows: int, m: int) -> np.ndarray:
    """raw uint8 → W (n_rows, m) float32."""
    nb = n_rows * m // BLOCK
    R = raw.reshape(nb, BYTES)
    d = R[:, PAYLOAD:].copy().view(np.float16).astype(np.float32)
    byte = R[:, :PAYLOAD].astype(np.int32)
    trits = np.empty((nb, PAYLOAD, 5), dtype=np.int32)
    rest = byte
    for i, p in enumerate((81, 27, 9, 3, 1)):
        trits[..., i] = rest // p
        rest = rest % p
    q = trits.reshape(nb, BLOCK).astype(np.float32) - 1.0
    return (q * d).reshape(n_rows, m)


def self_test() -> None:
    rng = np.random.default_rng(7)
    W = rng.normal(0, 0.02, (64, 320)).astype(np.float32)
    raw = encode(W)
    assert raw.size == 64 * 2 * BYTES
    # 1. determinism: same input, same bytes
    assert np.array_equal(raw, encode(W)), "encode must be deterministic"
    # 2. layout, against a hand-built block: known trits, known scale
    q = rng.integers(-1, 2, (1, BLOCK)).astype(np.float32)
    d = np.array([[0.0173]], dtype=np.float32)
    d16 = d.astype(np.float16).astype(np.float32)   # what survives storage
    Wt = d16 * q
    raw_t = encode(Wt, d_fixed=d16)
    back_t = decode(raw_t, 1, BLOCK)
    exact = float(np.abs(back_t - Wt).max())
    print(f"hand-built block, fixed scale: max abs err {exact:.2e} (must be 0)")
    assert exact == 0.0
    # 3. fixed-scale round trip is a true fixed point
    raw_t2 = encode(decode(raw_t, 1, BLOCK), d_fixed=d16)
    assert np.array_equal(raw_t, raw_t2), "fixed-scale re-encode must be byte-identical"
    # 4. quality on gaussians
    back = decode(raw, 64, 320)
    rel = float(np.linalg.norm(back - W) / np.linalg.norm(W)) * 100
    print(f"gaussian relative error: {rel:.2f}% (expect ~44% for a single plane)")
    assert 35 < rel < 55
    print("self-test OK")


if __name__ == "__main__":
    self_test()
