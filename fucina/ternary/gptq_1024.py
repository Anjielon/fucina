"""GPTQ for the TQ1_0R_1024 type (1024-block Hadamard rotation + sign S1024,
LS scale every 128 weights) — 2026-09-20, for "Thor-vBonsai-GPTQ".

Extends the RTN codec (`tq1_0r_1024_codec`) with real GPTQ: the tensor's Hessian,
rotated CORRECTLY (H' = R H Rᵀ — R is not symmetric as in the k=256 case WITHOUT
sign, here the transpose matters), drives the sequential error correction across
the 128-weight groups, on top of the ALREADY rotated values.

⚠️ Verified BEFORE being used in a real forge: see `verifica_gptq_1024.py`
(orthogonality of R, consistency between the explicit matrix and the rotation
functions, GPTQ-vs-RTN on a real tensor).
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

from fucina.ternary.tq1_0r_1024_codec import S1024, QK, BLOCK_BYTES, fwht1024_f32, _pack_quarter  # noqa: E402

GROUP = 128  # SCALE granularity (Bonsai g128) — different from the rotation (1024)


# ───────────────────────────── rotation ─────────────────────────────
def _R_1024_np(V: np.ndarray) -> np.ndarray:
    """R v = FWHT_1024(S ⊙ v), in blocks of 1024 along the last axis.
    Exactly the same arithmetic as tq1_0r_1024_codec.quantize_rows BEFORE
    quantization (separated here so that GPTQ can be inserted in between)."""
    n = V.shape[-1]
    assert n % QK == 0
    Vb = V.reshape(*V.shape[:-1], n // QK, QK)
    out = fwht1024_f32(Vb * S1024)
    return out.reshape(V.shape)


def _RT_1024_np(V: np.ndarray) -> np.ndarray:
    """Rᵀ v = S ⊙ FWHT_1024(v) — same arithmetic as dequantize_blocks
    BEFORE repacking (the exact inverse of _R_1024_np, R orthogonal)."""
    n = V.shape[-1]
    assert n % QK == 0
    Vb = V.reshape(*V.shape[:-1], n // QK, QK)
    out = fwht1024_f32(Vb) * S1024
    return out.reshape(V.shape)


def R_1024_matrix() -> np.ndarray:
    """Explicit 1024x1024 matrix of R, built ONLY to rotate the Hessian
    (2048x2048 at most: two matmuls, negligible cost) — avoids trusting by
    hand which convention (R vs Rᵀ) to use in the reshape."""
    I = np.eye(QK, dtype=np.float32)
    # row i of _R_1024_np(I) = R @ e_i = COLUMN i of R -> transposing gives R
    col_of_R = _R_1024_np(I)
    return col_of_R.T.astype(np.float64).copy()


def rotate_hessian_1024(H: np.ndarray) -> np.ndarray:
    """H' = R H Rᵀ, R = block_diag(R_1024, R_1024, ...) over n = H.shape[0]."""
    n = H.shape[0]
    assert n % QK == 0
    nb = n // QK
    R1 = R_1024_matrix()
    Rfull = np.zeros((n, n), dtype=np.float64)
    for b in range(nb):
        Rfull[b * QK:(b + 1) * QK, b * QK:(b + 1) * QK] = R1
    return Rfull @ H.astype(np.float64) @ Rfull.T


def rotate_rows_1024(W: np.ndarray) -> np.ndarray:
    """Applies R to each row of W (input axis) — the same weight rotation
    that the RTN forge applies before quantizing."""
    return _R_1024_np(W.astype(np.float32))


# ───────────────────────────── GPTQ (torch/GPU) ─────────────────────────────
def prepare_hessian(H: np.ndarray, damping: float = 0.01) -> np.ndarray:
    """Identical to prepare_hessian of the two-plane and ODINO forges: UPPER
    Cholesky factor U with UᵀU = H⁻¹ (H already in the rotated basis, when needed)."""
    H = np.array(H, dtype=np.float64, copy=True)
    n = H.shape[0]
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    H += np.eye(n) * (damping * np.mean(np.diag(H)))
    L = np.linalg.cholesky(H)
    Li = np.linalg.inv(L)
    Hinv = Li.T @ Li
    return np.linalg.cholesky(Hinv).T.copy()


def _scale_one_group(B: torch.Tensor, grid=(0.55, 0.66, 0.76, 0.86, 1.00, 1.20)) -> torch.Tensor:
    """LS scale for the group (pure RTN) — same grid/formula as the single-plane
    scale of the GPU ternary forge and as tq1_0r_1024_codec.scale_ls."""
    ab = B.abs()
    mean = ab.mean(-1, keepdim=True)
    en = (B * B).sum(-1)
    best_e = torch.full_like(en, float("inf"))
    best_d = torch.zeros_like(en)
    for m in grid:
        live = ab > (m * mean)
        k = live.sum(-1).to(B.dtype)
        sum_ = torch.where(live, ab, torch.zeros_like(ab)).sum(-1)
        e = en - torch.where(k > 0, sum_ * sum_ / k.clamp(min=1), torch.zeros_like(en))
        ok = e < best_e
        best_e = torch.where(ok, e, best_e)
        best_d = torch.where(ok, torch.where(k > 0, sum_ / k.clamp(min=1), torch.zeros_like(k)), best_d)
    return best_d.unsqueeze(-1)


@torch.no_grad()
def gptq_1024(W: torch.Tensor, Hchol: torch.Tensor, device: str = "cuda", piece: int = 2_000_000):
    """Single-plane ternary GPTQ, one scale per group of 128 (GROUP), Hessian
    recomputed on the ALREADY compensated block — generalization of the GPU
    ternary forge's single-plane quantizer from BLOCK=256 to GROUP=128, same algorithm.

    W: (n_rows, n_in) ALREADY rotated by the caller. Hchol: (n_in, n_in) from
    prepare_hessian(H_rotated). Returns (D (n_rows, n_in/128, 1) f32,
    Q (n_rows, n_in) int8 in {-1,0,1}), on CPU.
    """
    n_rows, n_in = W.shape
    assert n_in % GROUP == 0
    ng = n_in // GROUP
    D = torch.empty(n_rows, ng, 1, dtype=torch.float32)
    Q = torch.empty(n_rows, n_in, dtype=torch.int8)
    H = Hchol.to(device)

    rows_per_piece = max(1, piece // ng)
    i0 = 0
    while i0 < n_rows:
        i1 = min(i0 + rows_per_piece, n_rows)
        try:
            Wc = W[i0:i1].to(device, torch.float32, non_blocking=True).clone()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows_per_piece = max(1, rows_per_piece // 2)
            continue
        rows = i1 - i0
        dloc = torch.empty(rows, ng, 1, device=device)
        qloc = torch.empty(rows, n_in, device=device)
        for g in range(ng):
            j0, j1 = g * GROUP, (g + 1) * GROUP
            block = Wc[:, j0:j1].clone()
            d = _scale_one_group(block)
            dloc[:, g] = d
            err = torch.empty_like(block)
            inv = 1.0 / d.clamp(min=1e-20)
            for j in range(GROUP):
                w = block[:, j]
                q = torch.clamp(torch.round(w * inv[:, 0]), -1, 1)
                qloc[:, j0 + j] = q
                e = (w - d[:, 0] * q) / H[j0 + j, j0 + j].clamp(min=1e-8)
                err[:, j] = e
                if j + 1 < GROUP:
                    block[:, j + 1:] -= e.unsqueeze(1) * H[j0 + j, j0 + j + 1:j1].unsqueeze(0)
            if j1 < n_in:
                Wc[:, j1:] -= err @ H[j0:j1, j1:]
        D[i0:i1] = dloc.cpu()
        Q[i0:i1] = qloc.to(torch.int8).cpu()
        del Wc, dloc, qloc
        i0 = i1
    return D, Q


def pack_from_d_q(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d: (N,8) float32 scales per group of 128 · q: (N,1024) int8 {-1,0,1}
    -> (N,224) uint8, SAME format as quantize_rotated_blocks (4 quarters of
    256, fp16 scales at the tail) but with the trits already decided (RTN or GPTQ)."""
    N = d.shape[0]
    assert d.shape == (N, 8) and q.shape == (N, QK)
    T = np.clip(q.astype(np.int64) + 1, 0, 2).reshape(N, 4, 256)
    out = np.zeros((N, BLOCK_BYTES), dtype=np.uint8)
    for qq in range(4):
        qs, qh = _pack_quarter(T[:, qq, :])
        out[:, qq * 48:(qq + 1) * 48] = qs
        out[:, 192 + qq * 4:192 + (qq + 1) * 4] = qh
    out[:, 208:224] = d.astype(np.float16).view(np.uint8).reshape(N, 16)
    return out
