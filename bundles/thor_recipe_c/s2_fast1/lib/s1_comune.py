#!/usr/bin/env python3
"""S1 (piano casalingo di C, 13/10/2026) — parti comuni: percorsi, formati su disco, lettura per i passi S1b-S4.

Formati (tutto nella base HF NON ruotata, salvo i gradienti che sono nella base MEMORIZZATA come gradienti_densi.py):
  densi, per strato L e meta' h (A costruisce, B fa da guardia), punto d'ingresso p:
    Hq_<src>.npy   triangolo superiore fp32 di sum_t x_q x_q^T / n_src   (x_q = flusso di C, src in M|Kv|Kw)
    Hf_M.npy       idem sul flusso del maestro bf16, solo sorgente M (controllo contro la H del manifesto)
    Mx.npy         matrice piena fp32 sum_t x_f x_q^T / n (tutte le sorgenti): termine incrociato di GPTAQ/QEP
    sq_<src>.npy / sf.npy  somme dei vettori (medie per la correzione del bias)
    ssm_out: righe/colonne gia' nell'ordine GGUF (permutazione di mappa_completa_gguf_hf_perm, come la H del manifesto)
  punti -> famiglie:  attn -> attn_qkv attn_gate ssm_alpha ssm_beta | attn_q attn_k attn_v
                      ffn  -> ffn_gate_shexp ffn_up_shexp (e la H di strato degli esperti gate/up)
                      att_out -> ssm_out | attn_output        down -> ffn_down_shexp
  esperti (pesati dal router del flusso di C, w = probabilita' top-8 rinormalizzate, MoEQuant AGQ 2505.03804):
    esperti_Gq.npy   fp16 (E, n_tri)  triangolo di sum w x_q x_q^T, diviso per la sua scala (max |diag|): esperti_Gq_scala.npy
    esperti_C.npy    fp16 (E, 2I, H)  W_ref[gate;up] @ sum w x_f x_q^T  (bersaglio asimmetrico GPTAQ 2504.02692 / C di
                     SchurQuant), diviso per esperti_C_scala.npy (max |.| per esperto)
    esperti_Hdown.npy fp16 (E, n_tri(I)) sum w a a^T con a = silu(g) * u del flusso di C (ingresso VERO del down)
    esperti_conta.npy (E,) int64 token · esperti_peso.npy (E,) fp64 sum w · Hdown_strato.npy fp32 triangolo
  H di strato per il restringimento: sum_e sum_{t in e} w x x^T = sum_t x x^T (le w di un token sommano a 1) = Hq del
  punto ffn sommato sulle sorgenti: nessun file in piu'.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PUNTI = ("attn", "ffn", "att_out", "down")
FAMIGLIE_PUNTO = {"attn": ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "attn_q", "attn_k", "attn_v"),
                  "ffn": ("ffn_gate_shexp", "ffn_up_shexp"), "att_out": ("ssm_out", "attn_output"),
                  "down": ("ffn_down_shexp",)}
SORGENTI = ("M", "Kv", "Kw")
NGRAM = 13


def punto_di(fam: str) -> str:
    for p, fs in FAMIGLIE_PUNTO.items():
        if fam in fs:
            return p
    raise KeyError(fam)


# ── triangoli (matrici simmetriche a meta' spazio) ──
def triu_indici(n: int):
    return np.triu_indices(n)


def impacchetta(M):
    """(…, n, n) simmetrica -> (…, n(n+1)/2) triangolo superiore per righe (torch o numpy)."""
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


def spacchetta(v: np.ndarray, n: int, dtype=np.float64) -> np.ndarray:
    M = np.zeros((n, n), dtype=dtype)
    i, j = np.triu_indices(n)
    M[i, j] = v
    M[j, i] = v
    return M


def n_da_tri(m: int) -> int:
    n = int((np.sqrt(8 * m + 1) - 1) / 2)
    assert n * (n + 1) // 2 == m, m
    return n


# ── restringimento verso la H di strato (variante Thor dichiarata: Ledoit-Wolf / bayesiano, al posto del
#    sovracampionamento di EAQuant) ──
def ristretta(H_e: np.ndarray, n_e: float, H_strato: np.ndarray, k: float) -> np.ndarray:
    """H_e(k) = (n_e H_e + k H_strato) / (n_e + k), con H_e e H_strato GIA' normalizzate (media per token/peso)."""
    return (n_e * H_e + k * H_strato) / (n_e + k)


# ── lettura per i passi successivi ──
class LettoreS1:
    """Accesso in sola lettura alle uscite di S1 (radice = cartella cattura/)."""

    def __init__(self, radice):
        self.r = Path(radice)

    def dir(self, meta: str, L: int) -> Path:
        return self.r / meta / f"strato_{L:02d}"

    def info(self, meta: str, L: int) -> dict:
        return json.load(open(self.dir(meta, L) / "info.json"))

    def H_denso(self, meta: str, L: int, punto: str, sorgenti=SORGENTI, flusso: str = "q") -> np.ndarray:
        """H normalizzata (media per token) del punto, sommando le sorgenti richieste pesate coi loro token."""
        inf = self.info(meta, L)
        tot, n = None, 0
        for s in sorgenti:
            f = self.dir(meta, L) / f"{punto}_H{flusso}_{s}.npy"
            if not f.exists():
                continue
            ns = inf["token"][s]
            v = np.load(f).astype(np.float64) * ns
            tot = v if tot is None else tot + v
            n += ns
        assert tot is not None, (meta, L, punto, sorgenti)
        return spacchetta(tot / n, n_da_tri(tot.shape[0]))

    def M_incrociata(self, meta: str, L: int, punto: str) -> np.ndarray:
        return np.load(self.dir(meta, L) / f"{punto}_Mx.npy").astype(np.float64)

    def esperti(self, meta: str, L: int) -> dict:
        d = self.dir(meta, L)
        return {"conta": np.load(d / "esperti_conta.npy"), "peso": np.load(d / "esperti_peso.npy"),
                "Gq": np.load(d / "esperti_Gq.npy", mmap_mode="r"), "Gq_scala": np.load(d / "esperti_Gq_scala.npy"),
                "C": np.load(d / "esperti_C.npy", mmap_mode="r"), "C_scala": np.load(d / "esperti_C_scala.npy"),
                "Hdown": np.load(d / "esperti_Hdown.npy", mmap_mode="r"),
                "Hdown_scala": np.load(d / "esperti_Hdown_scala.npy")}

    def H_esperto(self, meta: str, L: int, e: int, k: float = 8000.0, ex: Optional[dict] = None) -> np.ndarray:
        """Gq_e/peso_e ristretta verso la H di strato (punto ffn) con k (token equivalenti), base HF non ruotata."""
        ex = ex or self.esperti(meta, L)
        n = int(ex["conta"][e])
        Hs = self.H_denso(meta, L, "ffn")
        if n == 0:
            return Hs
        He = spacchetta(np.asarray(ex["Gq"][e], np.float64) * float(ex["Gq_scala"][e]), Hs.shape[0]) / float(ex["peso"][e])
        return ristretta(He, n, Hs, k)

    def C_esperto(self, meta: str, L: int, e: int, ex: Optional[dict] = None) -> np.ndarray:
        """W_ref[gate;up]_e @ M_e / peso_e (2I, H)."""
        ex = ex or self.esperti(meta, L)
        return np.asarray(ex["C"][e], np.float64) * float(ex["C_scala"][e]) / max(float(ex["peso"][e]), 1e-30)


def scrivi_json(p: Path, d: dict) -> None:
    t = Path(str(p) + ".tmp")
    t.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    os.replace(t, p)


# ── n-grammi per la contaminazione (hash polinomiale a 64 bit, verifica esatta sui colpi) ──
_P = np.uint64(1000003)


def hash_ngrammi(ids: np.ndarray, n: int = NGRAM) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    m = ids.shape[-1] - n + 1
    if m <= 0:
        return np.zeros(ids.shape[:-1] + (0,), np.uint64)
    with np.errstate(over="ignore"):
        h = np.zeros(ids.shape[:-1] + (m,), np.uint64)
        for j in range(n):
            h = h * _P + ids[..., j:j + m] + np.uint64(1)
    return h
