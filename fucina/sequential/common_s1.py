#!/usr/bin/env python3
"""S1 (home-hardware plan for target C, 2026-10-13) — common parts: paths, on-disk formats, readers for steps S1b-S4.

Formats (everything in the NON-rotated HF basis, except the gradients, which are in the STORED basis as in
gradienti_densi.py):
  dense tensors, per layer L and half h (A builds, B acts as guard), input point p:
    Hq_<src>.npy   upper triangle fp32 of sum_t x_q x_q^T / n_src   (x_q = stream of C, src in M|Kv|Kw)
    Hf_M.npy       same on the bf16 teacher stream, source M only (check against the manifest H)
    Mx.npy         full fp32 matrix sum_t x_f x_q^T / n (all sources): cross term of GPTAQ/QEP
    sq_<src>.npy / sf.npy  vector sums (means for the bias correction)
    ssm_out: rows/columns already in GGUF order (permutation of mappa_completa_gguf_hf_perm, like the manifest H)
  points -> families: attn -> attn_qkv attn_gate ssm_alpha ssm_beta | attn_q attn_k attn_v
                      ffn  -> ffn_gate_shexp ffn_up_shexp (and the layer H of the gate/up experts)
                      att_out -> ssm_out | attn_output        down -> ffn_down_shexp
  experts (weighted by the router of the C stream, w = renormalized top-8 probabilities, MoEQuant AGQ 2505.03804):
    esperti_Gq.npy   fp16 (E, n_tri)  triangle of sum w x_q x_q^T, divided by its scale (max |diag|): esperti_Gq_scala.npy
    esperti_C.npy    fp16 (E, 2I, H)  W_ref[gate;up] @ sum w x_f x_q^T  (asymmetric target of GPTAQ 2504.02692 / C of
                     SchurQuant), divided by esperti_C_scala.npy (max |.| per expert)
    esperti_Hdown.npy fp16 (E, n_tri(I)) sum w a a^T with a = silu(g) * u of the C stream (the TRUE input of down)
    esperti_conta.npy (E,) int64 tokens · esperti_peso.npy (E,) fp64 sum w · Hdown_strato.npy fp32 triangle
  Layer H for the shrinkage: sum_e sum_{t in e} w x x^T = sum_t x x^T (the w of one token sum to 1) = Hq of the
  ffn point summed over the sources: no extra file needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

POINTS = ("attn", "ffn", "att_out", "down")
FAMILIES_POINT = {"attn": ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "attn_q", "attn_k", "attn_v"),
                  "ffn": ("ffn_gate_shexp", "ffn_up_shexp"), "att_out": ("ssm_out", "attn_output"),
                  "down": ("ffn_down_shexp",)}
SOURCES = ("M", "Kv", "Kw")
NGRAM = 13


def point_of(fam: str) -> str:
    for p, fs in FAMILIES_POINT.items():
        if fam in fs:
            return p
    raise KeyError(fam)


# ── triangles (symmetric matrices in half the space) ──
def triu_indices(n: int):
    return np.triu_indices(n)


def pack(M):
    """(…, n, n) symmetric -> (…, n(n+1)/2) row-major upper triangle (torch or numpy)."""
    n = M.shape[-1]
    try:
        import torch
        if isinstance(M, torch.Tensor):
            i, j = torch.triu_indices(n, n, device=M.device)
            return M[..., i, j]
    except ImportError:
        pass
    i, j = np.triu_indices(n)
    return M[..., i, j]


def unpack(v: np.ndarray, n: int, dtype=np.float64) -> np.ndarray:
    M = np.zeros((n, n), dtype=dtype)
    i, j = np.triu_indices(n)
    M[i, j] = v
    M[j, i] = v
    return M


def n_from_tri(m: int) -> int:
    n = int((np.sqrt(8 * m + 1) - 1) / 2)
    assert n * (n + 1) // 2 == m, m
    return n


# ── shrinkage towards the layer H (declared Thor variant: Ledoit-Wolf / Bayesian, instead of the
#    oversampling of EAQuant) ──
def restricted(H_e: np.ndarray, n_e: float, H_layer: np.ndarray, k: float) -> np.ndarray:
    """H_e(k) = (n_e H_e + k H_layer) / (n_e + k), with H_e and H_layer ALREADY normalized (mean per token/weight)."""
    return (n_e * H_e + k * H_layer) / (n_e + k)


# ── readers for the later steps ──
class S1Reader:
    """Read-only access to the S1 outputs (root = capture folder, cattura/)."""

    def __init__(self, root):
        self.r = Path(root)

    def dir(self, meta: str, L: int) -> Path:
        return self.r / meta / f"strato_{L:02d}"

    def info(self, meta: str, L: int) -> dict:
        return json.load(open(self.dir(meta, L) / "info.json"))

    def H_dense(self, meta: str, L: int, point: str, sources=SOURCES, stream: str = "q") -> np.ndarray:
        """Normalized H (mean per token) of the point, summing the requested sources weighted by their token counts."""
        inf = self.info(meta, L)
        tot, n = None, 0
        for s in sources:
            f = self.dir(meta, L) / f"{point}_H{stream}_{s}.npy"
            if not f.exists():
                continue
            ns = inf["token"][s]
            v = np.load(f).astype(np.float64) * ns
            tot = v if tot is None else tot + v
            n += ns
        assert tot is not None, (meta, L, point, sources)
        return unpack(tot / n, n_from_tri(tot.shape[0]))

    def M_cross(self, meta: str, L: int, point: str) -> np.ndarray:
        return np.load(self.dir(meta, L) / f"{point}_Mx.npy").astype(np.float64)

    def experts(self, meta: str, L: int) -> dict:
        d = self.dir(meta, L)
        return {"count": np.load(d / "esperti_conta.npy"), "weight": np.load(d / "esperti_peso.npy"),
                "Gq": np.load(d / "esperti_Gq.npy", mmap_mode="r"), "Gq_scale": np.load(d / "esperti_Gq_scala.npy"),
                "C": np.load(d / "esperti_C.npy", mmap_mode="r"), "C_scale": np.load(d / "esperti_C_scala.npy"),
                "Hdown": np.load(d / "esperti_Hdown.npy", mmap_mode="r"),
                "Hdown_scale": np.load(d / "esperti_Hdown_scala.npy")}

    def H_expert(self, meta: str, L: int, e: int, k: float = 8000.0, ex: Optional[dict] = None) -> np.ndarray:
        """Gq_e/weight_e shrunk towards the layer H (ffn point) with k (equivalent tokens), non-rotated HF basis."""
        ex = ex or self.experts(meta, L)
        n = int(ex["count"][e])
        Hs = self.H_dense(meta, L, "ffn")
        if n == 0:
            return Hs
        He = unpack(np.asarray(ex["Gq"][e], np.float64) * float(ex["Gq_scale"][e]), Hs.shape[0]) / float(ex["weight"][e])
        return restricted(He, n, Hs, k)

    def C_expert(self, meta: str, L: int, e: int, ex: Optional[dict] = None) -> np.ndarray:
        """W_ref[gate;up]_e @ M_e / weight_e (2I, H)."""
        ex = ex or self.experts(meta, L)
        return np.asarray(ex["C"][e], np.float64) * float(ex["C_scale"][e]) / max(float(ex["weight"][e]), 1e-30)


def write_json(p: Path, d: dict) -> None:
    t = Path(str(p) + ".tmp")
    t.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    os.replace(t, p)


# ── n-grams for contamination checks (64-bit polynomial hash, exact verification on hits) ──
_P = np.uint64(1000003)


def hash_ngrams(ids: np.ndarray, n: int = NGRAM) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    m = ids.shape[-1] - n + 1
    if m <= 0:
        return np.zeros(ids.shape[:-1] + (0,), np.uint64)
    with np.errstate(over="ignore"):
        h = np.zeros(ids.shape[:-1] + (m,), np.uint64)
        for j in range(n):
            h = h * _P + ids[..., j:j + m] + np.uint64(1)
    return h
