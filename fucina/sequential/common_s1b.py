#!/usr/bin/env python3
"""S1b (home-hardware plan for target C, 2026-10-13) — common parts: bases, reading/writing ONLY the scales inside
the GGUF blocks, closed-form scale system, objective, rotation.

The S1b arm does NOT change a single trit: the trit bytes (qs/qh) of every block stay bit-exact with the host; only
the fp16 scales change (TQ1_0R_1024: 8 fp16 at bytes 208:224 of every 224 B block, one per 128 weights in the rotated
basis; TQ1_0: 1 fp16 at bytes 52:54 of every 54 B block, one per 256 weights, no rotation). Size, types, offsets and
header are identical by construction; speed is identical (same graph, same types).

Bases (conventions of rd_ternary.gguf_piani, verified in its tests against ggml-quants.c):
  TQ1_0R_1024  stored basis w' = H(S w) per 1024 block  =>  W' = W R^T, R = H1024 diag(S) orthogonal; the effective
               weight in the original basis is Q = Q' R; the error Hessian in the stored basis is R H R^T (rotate_H).
  TQ1_0        no rotation.
The S1 statistics are in the NON-rotated HF basis, which for the inputs coincides with the GGUF order (the only
column permutation is ssm_out, which S1 already saves in GGUF order).

Per-row objective (K=1, scale s_g per group, fixed codes t):
  J(s) = s^T M s - 2 s^T b,  M[g,g'] = t_g Hs[g,g'] t_g'^T,  b[g] = t_g . Cs_g
  symmetric  (output error on the C stream, = recipe of (c) with a different H):  Cs = W Hq
  asymmetric (GPTAQ 2504.02692 / QEP 2504.09629, target = output of the TEACHER): Cs = W Mx, Mx = E[x_f x_q^T]
              E||W x_f - Q x_q||^2 = const - 2 tr(Q (W Mx)^T) + tr(Q Hq Q^T)
The closed-form per-row minimum is the AGA of PT2-LLM (2510.03267) made SYMMETRIC (no offset: BitNet convention) and
asymmetric in the GPTAQ sense; for the experts it is the scale step of JARQ (2609.38599 eq. 6) on the per-expert H.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from fucina.extdeps import ENV_RD_TERNARY, add_path  # noqa: E402
add_path(ENV_RD_TERNARY)   # directory holding the rd_ternary package (GGUF plane reader/writer); see fucina.extdeps

from rd_ternary import gguf_piani as gp  # noqa: E402
from rd_ternary.formato import fwht  # noqa: E402

F64 = torch.float64
T_TQ1_0, T_TQ1_0R_1024 = 34, 45
# type_ -> (weights per block, bytes per block, scale bytes in the block [a, b), scales per block, group, rotated)
FORMAT = {T_TQ1_0: (256, 54, (52, 54), 1, 256, False),
           T_TQ1_0R_1024: (1024, 224, (208, 224), 8, 128, True)}
FP16_MAX = 65504.0


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def write_json(p, d) -> None:
    p = Path(p)
    t = Path(str(p) + ".tmp")
    t.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    os.replace(t, p)


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


# ───────────────────────────── rotation (copy of the refine package's common.py, same conventions) ─────────────────────────────
def _s1024(dt, dev=None):
    return torch.from_numpy(gp.sign1024().astype(np.float32)).to(dtype=dt, device=dev)


def rotate(W: torch.Tensor) -> torch.Tensor:
    """rows in the original basis -> stored basis: w' = H(S w) per 1024 block (W' = W R^T)."""
    C = W.shape[-1]
    x = (W.reshape(*W.shape[:-1], C // 1024, 1024) * _s1024(W.dtype, W.device)).reshape(W.shape)
    return fwht(x, 1024)


def unrotate(Wp: torch.Tensor) -> torch.Tensor:
    """stored basis -> original: w = S * H(w') (Q = Q' R)."""
    y = fwht(Wp, 1024)
    C = Wp.shape[-1]
    return (y.reshape(*Wp.shape[:-1], C // 1024, 1024) * _s1024(Wp.dtype, Wp.device)).reshape(Wp.shape)


def rotate_H(H: torch.Tensor) -> torch.Tensor:
    """R H R^T."""
    Hr = rotate(H)
    return rotate(Hr.T.contiguous()).T.contiguous()


# ───────────────────────────── blocks: trits and scales ─────────────────────────────
def decode(raw_rows: np.ndarray, tid: int, in_: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """raw rows [R, row_bytes] -> (T int8 [R, C] stored basis, S f64 [R, ng] fp16 values)."""
    t, d = gp.decode_rows(np.ascontiguousarray(raw_rows), tid, in_)
    return torch.from_numpy(t.astype(np.int8)), torch.from_numpy(d.astype(np.float64))


def replace_scale(raw_rows: np.ndarray, S16: np.ndarray, tid: int) -> np.ndarray:
    """Copy of the raw rows with ONLY the scales replaced (trit bytes bit-exact)."""
    blk, bb, (a, b), nsb, _, _ = FORMAT[tid]
    R = raw_rows.shape[0]
    out = np.array(raw_rows, dtype=np.uint8, copy=True).reshape(-1, bb)
    s = np.ascontiguousarray(np.asarray(S16, dtype="<f2")).reshape(-1, nsb)
    assert s.shape[0] == out.shape[0], (s.shape, out.shape)
    out[:, a:b] = s.view(np.uint8).reshape(-1, b - a)
    return out.reshape(R, -1)


def mask_scale(tid: int, nbytes: int) -> np.ndarray:
    """bool [nbytes]: True on the scale bytes."""
    _, bb, (a, b), _, _, _ = FORMAT[tid]
    m = np.zeros(bb, dtype=bool)
    m[a:b] = True
    return np.tile(m, nbytes // bb)


def expand(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    R, C = T.shape
    ng = S.shape[-1]
    return (T.to(F64).reshape(R, ng, C // ng) * S.to(F64).unsqueeze(-1)).reshape(R, C)


def q_original(T, S, rotated: bool, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """effective weight in the ORIGINAL input basis (after the optional per-column scale c folded upstream)."""
    Q = expand(T, S)
    if rotated:
        Q = unrotate(Q)
    return Q if c is None else Q * c.to(F64).unsqueeze(0)


# ───────────────────────────── scale system (K=1) ─────────────────────────────
def bases_stored(Hq: torch.Tensor, Clin: torch.Tensor, rotated: bool, c: Optional[torch.Tensor] = None):
    """(Hq, Clin) in the original basis -> stored basis, with the optional column scale c: the effective weight is
    Q' R diag(c), hence Hs = R D Hq D R^T and Cs = (Clin D) R^T."""
    if c is not None:
        c = c.to(F64)
        Hq = Hq * c.unsqueeze(0) * c.unsqueeze(1)
        Clin = Clin * c.unsqueeze(0)
    if rotated:
        return rotate_H(Hq), rotate(Clin)
    return Hq, Clin


def system(T: torch.Tensor, Hs: torch.Tensor, Cs: torch.Tensor, ng: int, batch: int = 256):
    """M [R, ng, ng], b [R, ng] (see the module docstring). T int8 [R, C] in the stored basis."""
    R, C = T.shape
    G = C // ng
    M = torch.empty(R, ng, ng, dtype=F64)
    b = torch.empty(R, ng, dtype=F64)
    Hs = Hs.to(F64)
    for a in range(0, R, batch):
        z = min(R, a + batch)
        r = z - a
        A3 = T[a:z].to(F64).reshape(r, ng, G)
        BH = torch.empty(r, ng, C, dtype=F64)
        for g in range(ng):
            BH[:, g, :] = A3[:, g, :] @ Hs[g * G:(g + 1) * G, :]
        M[a:z] = (BH.reshape(r, ng, ng, G) * A3.unsqueeze(1)).sum(-1)
        b[a:z] = (A3 * Cs[a:z].to(F64).reshape(r, ng, G)).sum(-1)
    return M, b


def solve(M: torch.Tensor, b: torch.Tensor, S0: torch.Tensor, ridge: float = 1e-8) -> torch.Tensor:
    """argmin_s s^T M s - 2 s^T b per row (not rounded). Groups without codes (zero diagonal): host scale."""
    dia = torch.diagonal(M, dim1=1, dim2=2)
    empty = dia <= 0
    Mm = M + torch.diag_embed(empty.to(F64))
    sc = dia.clamp_min(0).mean(1, keepdim=True).clamp_min(1e-300)
    sol = None
    try:
        L = torch.linalg.cholesky(Mm + ridge * torch.diag_embed(sc.expand_as(dia)))
        sol = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        pass
    if sol is None or not torch.isfinite(sol).all():
        sol = torch.linalg.lstsq(Mm, b.unsqueeze(-1)).solution.squeeze(-1)
    return torch.where(empty, S0.to(F64), sol)


def J(M: torch.Tensor, b: torch.Tensor, S: torch.Tensor) -> float:
    """objective (up to a constant) summed over the rows."""
    S = S.to(F64)
    return float((torch.einsum("rg,rgh,rh->r", S, M, S) - 2 * (S * b).sum(1)).sum())


def scale_ok(S: torch.Tensor, S0: torch.Tensor) -> bool:
    """new scales are writable: finite, within fp16 range, and no nonzero host scale has become zero."""
    return bool(torch.isfinite(S).all() and (S.abs() <= FP16_MAX).all() and not ((S0 != 0) & (S == 0)).any())


# ───────────────────────────── per-column scale (step c, closed form) ─────────────────────────────
def step_columns(Qo: torch.Tensor, Hq: torch.Tensor, Clin: torch.Tensor, P: Optional[torch.Tensor] = None,
                  ridge: float = 1e-9) -> torch.Tensor:
    """Given the effective weight WITHOUT column scale Qo [R, C] (original basis), c = argmin J(Qo diag(c)):
    (Hq ⊙ Qo^T Qo) c = rowsum(Qo ⊙ Clin)  (ARB-LLM_RC 2410.03129 eq. 13; H-weighted = declared Thor variant;
    with Clin = W Mx it becomes asymmetric). P [C, m] (optional) ties the columns: c = P c_m (GQA attn_v -> attn_output)."""
    A = Hq * (Qo.T @ Qo)
    bc = (Qo * Clin).sum(0)
    if P is not None:
        A = P.T @ A @ P
        bc = P.T @ bc
    A = 0.5 * (A + A.T)
    A = A + ridge * torch.diagonal(A).clamp_min(0).mean() * torch.eye(A.shape[0], dtype=F64)
    try:
        cm = torch.linalg.solve(A, bc)
    except RuntimeError:
        cm = torch.linalg.lstsq(A, bc.unsqueeze(-1)).solution.squeeze(-1)
    return P @ cm if P is not None else cm


def tie_gqa(n_head: int, n_kv: int, head_dim: int) -> torch.Tensor:
    """P [n_head*head_dim, n_kv*head_dim]: column j of attn_output (q head h, dim d) <- row of attn_v (kv head
    h // (n_head/n_kv), dim d). Same as repeat_kv in transformers (kv heads repeated in contiguous blocks)."""
    rep = n_head // n_kv
    P = torch.zeros(n_head * head_dim, n_kv * head_dim, dtype=F64)
    j = torch.arange(n_head * head_dim)
    P[j, (j // head_dim // rep) * head_dim + j % head_dim] = 1.0
    return P


def restricted(H_e: torch.Tensor, n_e: float, H_s: torch.Tensor, k: float) -> torch.Tensor:
    """shrinkage towards the layer statistic (declared Thor variant, same as common_s1.restricted)."""
    return (n_e * H_e + k * H_s) / (n_e + k)
