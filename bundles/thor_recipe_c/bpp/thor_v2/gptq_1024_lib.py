"""GPTQ per il tipo TQ1_0R_1024 (rotazione Hadamard blocco 1024 + segno S1024,
scala LS ogni 128 pesi) — 20/9/2026, per "Thor-vBonsai-GPTQ".

Estende il codec RTN (`tq1_0r_1024_codec.py`) con GPTQ vero: la Hessiana del
tensore, ruotata CORRETTAMENTE (H' = R H Rᵀ — R non e' simmetrica come nel
caso k=256 SENZA segno, qui la trasposta conta), guida la correzione
sequenziale dell'errore fra i gruppi da 128, sopra i valori GIA' ruotati.

⚠️ Verificata PRIMA di essere usata in una forgia vera: vedi
`verifica_gptq_1024.py` nello stesso albero (ortogonalita' di R, coerenza fra
matrice esplicita e funzioni di rotazione, GPTQ-vs-RTN su un tensore vero).
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernel-rotazione"))
from tq1_0r_1024_codec import S1024, QK, BLOCK_BYTES, fwht1024_f32, _pack_quarter  # noqa: E402

GROUP = 128  # granularita' della SCALA (Bonsai g128) — diversa dalla rotazione (1024)


# ───────────────────────────── rotazione ─────────────────────────────
def _R_1024_np(V: np.ndarray) -> np.ndarray:
    """R v = FWHT_1024(S ⊙ v), per blocchi di 1024 sull'ultimo asse.
    Stessa aritmetica esatta di tq1_0r_1024_codec.quantize_rows PRIMA della
    quantizzazione (qui separata per poter inserire GPTQ in mezzo)."""
    n = V.shape[-1]
    assert n % QK == 0
    Vb = V.reshape(*V.shape[:-1], n // QK, QK)
    out = fwht1024_f32(Vb * S1024)
    return out.reshape(V.shape)


def _RT_1024_np(V: np.ndarray) -> np.ndarray:
    """Rᵀ v = S ⊙ FWHT_1024(v) — stessa aritmetica di dequantize_blocks
    PRIMA di ri-impacchettare (l'inversa esatta di _R_1024_np, R ortogonale)."""
    n = V.shape[-1]
    assert n % QK == 0
    Vb = V.reshape(*V.shape[:-1], n // QK, QK)
    out = fwht1024_f32(Vb) * S1024
    return out.reshape(V.shape)


def R_1024_matrice() -> np.ndarray:
    """Matrice 1024x1024 esplicita di R, costruita SOLO per ruotare la
    Hessiana (2048x2048 al piu': due matmul, costo irrisorio) — evita di
    fidarsi a mano di quale convenzione (R vs Rᵀ) usare nel reshape."""
    I = np.eye(QK, dtype=np.float32)
    # riga i di _R_1024_np(I) = R @ e_i = COLONNA i di R -> trasponendo si ottiene R
    col_di_R = _R_1024_np(I)
    return col_di_R.T.astype(np.float64).copy()


def ruota_hessiana_1024(H: np.ndarray) -> np.ndarray:
    """H' = R H Rᵀ, R = block_diag(R_1024, R_1024, ...) su n = H.shape[0]."""
    n = H.shape[0]
    assert n % QK == 0
    nb = n // QK
    R1 = R_1024_matrice()
    Rfull = np.zeros((n, n), dtype=np.float64)
    for b in range(nb):
        Rfull[b * QK:(b + 1) * QK, b * QK:(b + 1) * QK] = R1
    return Rfull @ H.astype(np.float64) @ Rfull.T


def ruota_righe_1024(W: np.ndarray) -> np.ndarray:
    """Applica R a ciascuna riga di W (asse ingresso) — stessa rotazione del
    peso che la forgia RTN applica prima di quantizzare."""
    return _R_1024_np(W.astype(np.float32))


# ───────────────────────────── GPTQ (torch/GPU) ─────────────────────────────
def prepara_hessiana(H: np.ndarray, smorzamento: float = 0.01) -> np.ndarray:
    """Identica a due_piani.prepara_hessiana / odino: fattore Cholesky
    SUPERIORE U con UᵀU = H⁻¹ (H gia' nella base ruotata, quando serve)."""
    H = np.array(H, dtype=np.float64, copy=True)
    n = H.shape[0]
    morti = np.diag(H) == 0
    H[morti, morti] = 1.0
    H += np.eye(n) * (smorzamento * np.mean(np.diag(H)))
    L = np.linalg.cholesky(H)
    Li = np.linalg.inv(L)
    Hinv = Li.T @ Li
    return np.linalg.cholesky(Hinv).T.copy()


def _scala_un_gruppo(B: torch.Tensor, griglia=(0.55, 0.66, 0.76, 0.86, 1.00, 1.20)) -> torch.Tensor:
    """Scala LS per il gruppo (RTN puro) — stessa griglia/formula di
    ternario_gpu._scala_un_piano e di tq1_0r_1024_codec.scala_ls."""
    ab = B.abs()
    media = ab.mean(-1, keepdim=True)
    en = (B * B).sum(-1)
    best_e = torch.full_like(en, float("inf"))
    best_d = torch.zeros_like(en)
    for m in griglia:
        vivi = ab > (m * media)
        k = vivi.sum(-1).to(B.dtype)
        somma = torch.where(vivi, ab, torch.zeros_like(ab)).sum(-1)
        e = en - torch.where(k > 0, somma * somma / k.clamp(min=1), torch.zeros_like(en))
        ok = e < best_e
        best_e = torch.where(ok, e, best_e)
        best_d = torch.where(ok, torch.where(k > 0, somma / k.clamp(min=1), torch.zeros_like(k)), best_d)
    return best_d.unsqueeze(-1)


@torch.no_grad()
def gptq_1024(W: torch.Tensor, Hchol: torch.Tensor, device: str = "cuda", pezzo: int = 2_000_000):
    """GPTQ a UN piano ternario, scala per gruppo di 128 (GROUP), Hessiana
    ricalcolata sul blocco GIA' compensato — generalizzazione di
    ternario_gpu.quantizza_un_piano da BLOCCO=256 a GROUP=128, stesso algoritmo.

    W: (n_righe, n_in) GIA' ruotata dal chiamante. Hchol: (n_in, n_in) da
    prepara_hessiana(H_ruotata). Ritorna (D (n_righe, n_in/128, 1) f32,
    Q (n_righe, n_in) int8 in {-1,0,1}), su CPU.
    """
    n_righe, n_in = W.shape
    assert n_in % GROUP == 0
    ng = n_in // GROUP
    D = torch.empty(n_righe, ng, 1, dtype=torch.float32)
    Q = torch.empty(n_righe, n_in, dtype=torch.int8)
    H = Hchol.to(device)

    righe_per_pezzo = max(1, pezzo // ng)
    i0 = 0
    while i0 < n_righe:
        i1 = min(i0 + righe_per_pezzo, n_righe)
        try:
            Wc = W[i0:i1].to(device, torch.float32, non_blocking=True).clone()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            righe_per_pezzo = max(1, righe_per_pezzo // 2)
            continue
        righe = i1 - i0
        dloc = torch.empty(righe, ng, 1, device=device)
        qloc = torch.empty(righe, n_in, device=device)
        for g in range(ng):
            j0, j1 = g * GROUP, (g + 1) * GROUP
            blocco = Wc[:, j0:j1].clone()
            d = _scala_un_gruppo(blocco)
            dloc[:, g] = d
            err = torch.empty_like(blocco)
            inv = 1.0 / d.clamp(min=1e-20)
            for j in range(GROUP):
                w = blocco[:, j]
                q = torch.clamp(torch.round(w * inv[:, 0]), -1, 1)
                qloc[:, j0 + j] = q
                e = (w - d[:, 0] * q) / H[j0 + j, j0 + j].clamp(min=1e-8)
                err[:, j] = e
                if j + 1 < GROUP:
                    blocco[:, j + 1:] -= e.unsqueeze(1) * H[j0 + j, j0 + j + 1:j1].unsqueeze(0)
            if j1 < n_in:
                Wc[:, j1:] -= err @ H[j0:j1, j1:]
        D[i0:i1] = dloc.cpu()
        Q[i0:i1] = qloc.to(torch.int8).cpu()
        del Wc, dloc, qloc
        i0 = i1
    return D, Q


def pack_da_d_q(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d: (N,8) float32 scala per gruppo di 128 · q: (N,1024) int8 {-1,0,1}
    -> (N,224) uint8, STESSO formato di quantize_rotated_blocks (4 quarti da
    256, scala fp16 in coda) ma coi trit gia' decisi (RTN o GPTQ)."""
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
