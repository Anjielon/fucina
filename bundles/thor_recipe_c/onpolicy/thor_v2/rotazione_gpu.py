"""Rotazione Hessiana su GPU SENZA matmul: H' = R H R^T via due applicazioni
della FWHT (add/sub elementwise, nessuna riduzione) — deterministica per
costruzione. Verificata (22/9 notte): CPU vs GPU max diff ~1e-14 (float64),
determinismo cross-processo confermato. 4.4x piu' veloce della rotazione
CPU (numpy, matmul denso) su 256 esperti."""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")
from tq1_0r_1024_codec import S1024, QK  # noqa: E402


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


def ruota_righe_1024_gpu(W_batch: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Rotazione dei PESI (righe) — R applicata direttamente, batchata."""
    return _R_1024_torch(W_batch, S)


def ruota_hessiana_batch_gpu(H_batch_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """H_batch_np: (E, n, n) -> H' = R H R^T per ognuno, via due FWHT, ZERO
    matmul. float64 per combaciare esattamente ruota_hessiana_1024."""
    S = torch.from_numpy(S1024).to(device, dtype=torch.float64)
    H = torch.from_numpy(H_batch_np).to(device, dtype=torch.float64)
    Ht = H.transpose(-2, -1)
    step1 = _R_1024_torch(Ht, S).transpose(-2, -1)
    step2 = _R_1024_torch(step1, S)
    return step2.cpu().numpy()


def ruota_righe_batch_gpu(W_batch_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """W_batch_np: (E, rows, n) -> righe ruotate, batchate, float64."""
    S = torch.from_numpy(S1024).to(device, dtype=torch.float64)
    W = torch.from_numpy(W_batch_np).to(device, dtype=torch.float64)
    return _R_1024_torch(W, S).cpu().numpy()


def _RT_1024_torch(V: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """⭐ 23/9 20:1x — Angelo: "tutto su GPU". De-rotazione (Rᵀ, l'inversa —
    R e' ortogonale) per l'assemblaggio finale dei pesi ternari, oggi fatta
    su CPU/numpy pura (`_RT_1024_np` in gptq_1024_lib.py, via `fwht1024_f32`
    numpy) su (256,1024,2048) due volte per strato (gate+up). STESSA
    identica aritmetica di `_RT_1024_np` (FWHT poi ⊙S, non ⊙S poi FWHT come
    la rotazione diretta) — solo su GPU invece che CPU, riusando
    `fwht1024_torch` gia' verificato per la rotazione diretta."""
    n = V.shape[-1]
    Vb = V.reshape(*V.shape[:-1], n // QK, QK)
    out = fwht1024_torch(Vb) * S
    return out.reshape(V.shape)


def de_ruota_pesi_batch_gpu(V_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Sostituto GPU di `_RT_1024_np` (gptq_1024_lib.py) per l'assemblaggio
    finale dei pesi ternari de-ruotati. V_np: (..., n) con n%1024==0,
    float32. Ritorna float32 (stesso dtype d'uscita dell'originale numpy)."""
    S = torch.from_numpy(S1024).to(device, dtype=torch.float32)
    V = torch.from_numpy(V_np.astype(np.float32)).to(device, dtype=torch.float32)
    return _RT_1024_torch(V, S).cpu().numpy()
