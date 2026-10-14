"""Hessian rotation on GPU WITHOUT matmul: H' = R H R^T via two applications
of the FWHT (elementwise add/sub, no reduction) — deterministic by
construction. Verified (2026-09-22, night): CPU vs GPU max diff ~1e-14 (float64),
cross-process determinism confirmed. 4.4x faster than the CPU rotation
(numpy, dense matmul) on 256 experts."""
from __future__ import annotations
import sys
import numpy as np
import torch

from fucina.ternary.tq1_0r_1024_codec import S1024, QK  # noqa: E402


def fwht1024_torch(V: torch.Tensor) -> torch.Tensor:
    v = V.contiguous() * 0.03125
    lead = v.shape[:-1]
    for s in range(10):
        ln = 1 << s
        v4 = v.reshape(*lead, 1024 // (2 * ln), 2, ln)
        a = v4[..., 0, :]
        b = v4[..., 1, :]
        new_a = a + b
        new_b = a - b
        v = torch.stack([new_a, new_b], dim=-2).reshape(*lead, 1024)
    return v


def _R_1024_torch(V: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    n = V.shape[-1]
    Vb = V.reshape(*V.shape[:-1], n // QK, QK)
    out = fwht1024_torch(Vb * S)
    return out.reshape(V.shape)


def rotate_rows_1024_gpu(W_batch: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Rotation of the WEIGHTS (rows) — R applied directly, batched."""
    return _R_1024_torch(W_batch, S)


def rotate_hessian_batch_gpu(H_batch_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """H_batch_np: (E, n, n) -> H' = R H R^T for each, via two FWHTs, ZERO
    matmuls. float64 to match rotate_hessian_1024 exactly."""
    S = torch.from_numpy(S1024).to(device, dtype=torch.float64)
    H = torch.from_numpy(H_batch_np).to(device, dtype=torch.float64)
    Ht = H.transpose(-2, -1)
    step1 = _R_1024_torch(Ht, S).transpose(-2, -1)
    step2 = _R_1024_torch(step1, S)
    return step2.cpu().numpy()


def rotate_rows_batch_gpu(W_batch_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """W_batch_np: (E, rows, n) -> rotated rows, batched, float64."""
    S = torch.from_numpy(S1024).to(device, dtype=torch.float64)
    W = torch.from_numpy(W_batch_np).to(device, dtype=torch.float64)
    return _R_1024_torch(W, S).cpu().numpy()


def _RT_1024_torch(V: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """2026-09-23 — "everything on GPU". Unrotation (Rᵀ, the inverse —
    R is orthogonal) for the final assembly of the ternary weights, previously
    done on pure CPU/numpy (`_RT_1024_np` in `fucina.ternary.gptq_1024`, via numpy
    `fwht1024_f32`) on (256,1024,2048) twice per layer (gate+up). EXACTLY the
    same arithmetic as `_RT_1024_np` (FWHT then ⊙S, not ⊙S then FWHT as in
    the forward rotation) — only on GPU instead of CPU, reusing
    `fwht1024_torch`, already verified for the forward rotation."""
    n = V.shape[-1]
    Vb = V.reshape(*V.shape[:-1], n // QK, QK)
    out = fwht1024_torch(Vb) * S
    return out.reshape(V.shape)


def de_rotate_weights_batch_gpu(V_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """GPU replacement for `_RT_1024_np` (`fucina.ternary.gptq_1024`) for the final
    assembly of the unrotated ternary weights. V_np: (..., n) with n%1024==0,
    float32. Returns float32 (same output dtype as the numpy original)."""
    S = torch.from_numpy(S1024).to(device, dtype=torch.float32)
    V = torch.from_numpy(V_np.astype(np.float32)).to(device, dtype=torch.float32)
    return _RT_1024_torch(V, S).cpu().numpy()
