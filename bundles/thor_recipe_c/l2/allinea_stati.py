#!/usr/bin/env python3
"""allinea_stati.py — Lever 2 of SINTESI_1025/R5 (25/10): intermediate-state alignment between the teacher and the student
on the SAME rows of the paper-loss corpus, added to the joint CARDAN loss of cardan_congiunto_scale.py.

Paper (faithful arm): CKA-QAD, arXiv 2606.05682 —
    L = L_out + sg(L_out / (L_CKA + eps)) * L_CKA,
    L_CKA = (1/|Lambda|) sum_{l in Lambda} (1 - CKA(H_S^l, H_T^l)),
    CKA(X, Y) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F),   Xc = X centred over the tokens (linear CKA).
  Here L_out = paper loss + w4 * KL4 (+ wk * full KL): the whole output-side loss of the step; Lambda = all cached layer
  outputs by default (the paper: all layers, no subsampling).
Thor arm (declared, Prova 2 on A, thor_v2/LEVA4_KD_INTERMEDIA.md): relative MSE ||H_S - H_T||_F^2 / ||H_T||_F^2 per layer,
  fixed weight (--allinea-modo fisso --allinea-peso 0.5).

H^l is the residual stream at the OUTPUT of layer l (llama.cpp tensor "l_out-<l>"; student: input of layer l+1, or xL for
the last layer). The residual stream of the student is NOT rotated (only the expert/dense GEMM inputs are), so it lives
in the teacher's basis; CKA is invariant to orthogonal transforms and isotropic scale anyway.

Teacher states: NOT in the existing teacher caches (they hold only h_nextn = final hidden after output_norm). They come
from a cached teacher pass made by cattura_stati_maestro.py (llama.cpp e2e_dump_logit --l-out on the q8 teacher, the
same tool and model as the paper-loss cache). An on-the-fly teacher forward is not possible here: the teacher exists only
as a 37.8 GB Q8_0 GGUF served by llama.cpp (no torch implementation of Q8_0), and its bf16 original is ~70 GB.

Gradient injection: the alignment gradient dL/dH^l is added to the gradient flowing into the output of layer l during the
student's own backward (Studente.indietro recomputes layer by layer; a tensor hook on each recomputed layer output adds
the extra term). Cost: one addition per aligned layer, no extra forward/backward. Works in the three backward paths of
e2e_studente (full, truncated, E2E_SENZA_RICALCOLO held graph): in the held graph the hook is registered at forward time.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
VERSIONE = "allinea_stati 1 (25/10/2026)"


def _lettore_dump():
    """e2e_dump_logit_leggi of the thor_v2 tree. NO sys.path change at import: inside the trainer the tree is already on
    the path (cardan_kd_paper puts REC there); a stand-alone caller gets thor_v2 APPENDED (never ahead of the others)."""
    try:
        import e2e_dump_logit_leggi as M
    except ImportError:
        sys.path.append(str(QUI / "thor_v2"))
        import e2e_dump_logit_leggi as M
    return M


# ───────────────────────────────────────── teacher-state cache ─────────────────────────────────────────
def righe_in_ordine(train_rows: list, seme: int) -> list:
    """The order in which cardan_congiunto_scale.py visits the paper rows: random.Random(seme).shuffle on a copy of the
    train rows, FIRST call on a fresh generator (= `rng.shuffle(ord_w)` in the trainer)."""
    o = list(train_rows)
    random.Random(seme).shuffle(o)
    return o


def _bf16_torch(a: np.ndarray) -> torch.Tensor:
    """bf16 bits (uint16 memmap) -> torch float32 (exact)."""
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype != np.uint16:
        raise TypeError(f"expected bf16 bits as uint16, got {a.dtype}")
    return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).float()


class SorgenteStati:
    """Reader of a cattura_stati_maestro.py cache. Layout:
        piano_stati.json               version, source teacher cache, model/tool sha, layers, n_token, blocks
        <lato>/blocco_NNNN/blocco.json SEAL (exists <=> block complete and verified): rows, sha256 of ids
        <lato>/blocco_NNNN/ids.i32     ids given to the teacher (n_rows x n_token, int32)
        <lato>/blocco_NNNN/dump/       e2e_dump_logit output (manifesto.json + seq_KKKK/l_out-<l>.bf16 [n_token, 2048])
    `stati(lato, i, strati, device)` -> (float32 tensor [len(strati), n_token, H], ids int64 numpy)."""

    def __init__(self, cartella):
        Uscita = _lettore_dump().Uscita
        self.dir = Path(cartella)
        self.piano = json.loads((self.dir / "piano_stati.json").read_text())
        self.strati = [int(x) for x in self.piano["strati"]]
        self.n_token = int(self.piano["n_token"])
        self._idx: dict = {}
        self._usc: dict = {}
        for b in self.piano["blocchi"]:
            d = self.dir / b["nome"]
            p = d / "blocco.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            ids = np.fromfile(d / "ids.i32", dtype=np.int32).reshape(len(s["righe"]), self.n_token)
            if hashlib.sha256(ids.tobytes()).hexdigest() != s["sha256_ids"]:
                raise RuntimeError(f"{d}: ids.i32 does not match its seal")
            self._usc[b["nome"]] = Uscita(d / "dump")
            for k, i in enumerate(s["righe"]):
                self._idx[(b["lato"], int(i))] = (b["nome"], k, ids[k])

    def disponibile(self, lato: str, i: int) -> bool:
        return (lato, int(i)) in self._idx

    def righe(self, lato: str) -> list:
        return sorted(i for (lt, i) in self._idx if lt == lato)

    def stati(self, lato: str, i: int, strati, device) -> tuple:
        nome, k, ids = self._idx[(lato, int(i))]
        u = self._usc[nome]
        out = []
        for L in strati:
            if L not in self.strati:
                raise KeyError(f"layer {L} not in the cache (layers {self.strati})")
            a = u.tensore(k, f"l_out-{L}")
            if tuple(a.shape) != (self.n_token, self.piano["n_embd"]):
                raise ValueError(f"{nome} seq {k} l_out-{L}: shape {tuple(a.shape)}")
            out.append(_bf16_torch(a))
        return torch.stack(out).to(device), ids.astype(np.int64)


# ───────────────────────────────────────── losses ─────────────────────────────────────────
def cka_lineare(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear CKA of two feature matrices X [n, d1], Y [n, d2] over the same n tokens (Kornblith et al. 2019; the CKA of
    CKA-QAD 2606.05682): ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F). Gram form when n <= d (identical value:
    ||Xc^T Yc||_F^2 = <Xc Xc^T, Yc Yc^T>, ||Xc^T Xc||_F = ||Xc Xc^T||_F), feature form otherwise. Differentiable."""
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    if X.shape[0] <= max(X.shape[1], Y.shape[1]):
        Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
        num = (Kx * Ky).sum()
        return num / (Kx.norm() * Ky.norm())
    A = Xc.T @ Yc
    return A.pow(2).sum() / ((Xc.T @ Xc).norm() * (Yc.T @ Yc).norm())


def mse_relativa(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Thor arm (R5 #2 variant, Prova 2 on A): ||X - Y||_F^2 / ||Y||_F^2."""
    return (X - Y).pow(2).sum() / Y.pow(2).sum()


def _acc(x: torch.Tensor) -> torch.Tensor:
    """Accumulation dtype of the alignment: float32, float64 kept as is (CPU reference tests)."""
    return x if x.dtype == torch.float64 else x.float()


METRICHE = {"cka": lambda X, Y: 1.0 - cka_lineare(X, Y), "mse_rel": mse_relativa}


def gradienti_allineamento(S: dict, T: dict, metrica: str, maschera: Optional[torch.Tensor] = None) -> tuple:
    """S, T: {layer: [n, d]} student (any dtype, detached here) and teacher states. Returns (L_al, {layer: dL_al/dS_l
    [n, d] float32}, {layer: per-layer loss}) with L_al = mean over the layers of METRICHE[metrica]. Each layer is
    differentiated on its own (autograd on a detached float32 (or float64) leaf: memory of one layer at a time); rows outside
    `maschera` (bool [n]) get a zero gradient and do not enter the statistic."""
    f = METRICHE[metrica]
    n_l = len(S)
    grad, per, tot = {}, {}, 0.0
    for L in sorted(S):
        x = _acc(S[L].detach())
        y = T[L].detach().to(device=x.device, dtype=x.dtype)
        leaf = x.clone().requires_grad_(True)
        with torch.enable_grad():
            xs, ys = (leaf, y) if maschera is None else (leaf[maschera], y[maschera])
            v = f(xs, ys)
            (v / n_l).backward()
        per[L] = float(v.detach())
        tot += per[L] / n_l
        grad[L] = leaf.grad.detach()
    return tot, grad, per


@torch.no_grad()
def valore_allineamento(S: dict, T: dict, metrica: str, maschera: Optional[torch.Tensor] = None) -> float:
    """L_al only (held-out metric): mean over the layers of METRICHE[metrica], no gradient."""
    f = METRICHE[metrica]
    v = []
    for L in sorted(S):
        x = _acc(S[L])
        y = T[L].to(device=x.device, dtype=x.dtype)
        if maschera is not None:
            x, y = x[maschera], y[maschera]
        v.append(float(f(x, y)))
    return sum(v) / len(v)


def peso_effettivo(peso: float, modo: str, L_out: float, L_al: float, eps: float) -> float:
    """CKA-QAD loss-ratio weight sg(L_out / (L_al + eps)) times --allinea-peso (rapporto), or the plain weight (fisso)."""
    if modo == "rapporto":
        return float(peso) * float(L_out) / (float(L_al) + float(eps))
    if modo == "fisso":
        return float(peso)
    raise ValueError(f"--allinea-modo {modo!r}: rapporto | fisso")


def maschera_inizi(inizi, n: int, device) -> Optional[torch.Tensor]:
    """Thor option --allinea-salta-inizi: drop the first token of every segment (attention-sink tokens with massive
    activations dominate a centred covariance). inizi (1, T) 0/1 or None (row mode: only position 0)."""
    if inizi is None:
        m = torch.ones(n, dtype=torch.bool)
        m[0] = False
    else:
        m = torch.as_tensor(inizi).reshape(-1)[:n].to(torch.long) == 0
    return m.to(device)


# ───────────────────────────────────────── gradient injection into the student backward ─────────────────────────────
class Iniettore:
    """Wraps st._strato (instance attribute: the Studente methods call self._strato). For the layers in `strati`:
      - registra = True and a forward call (instr None): records the output prefix y[0, :n] (detached float32 copy, float64 kept);
      - attivo = True and the output requires grad: registers a hook that adds self.extra[layer] to the gradient of
        rows [0, :n] of that output. Recomputed outputs (indietro) get the hook when they are recreated; outputs of the
        held graph (E2E_SENZA_RICALCOLO) get it at forward time, and read self.extra when the backward reaches them.
    Off (attivo False, registra False) it is a pass-through: same calls, same tensors."""

    def __init__(self, st, strati, n: int):
        self.st = st
        self.strati = set(int(x) for x in strati)
        self.n = int(n)
        self._orig = st._strato
        self.registra = False
        self.attivo = False
        self.uscite: dict = {}
        self.extra: dict = {}
        self.colpi = 0                                   # hooks that actually added a term (checked by the trainer)
        st._strato = self._strato

    def _strato(self, L, x, ctx, instr=None):
        y = self._orig(L, x, ctx, instr=instr)
        if L in self.strati:
            if self.registra and instr is None:
                self.uscite[L] = _acc(y[0, : self.n].detach()).clone()
            if self.attivo and y.requires_grad:
                y.register_hook(functools.partial(self._aggiungi, L))
        return y

    def _aggiungi(self, L, g):
        e = self.extra.get(L)
        if e is None:
            return None
        g2 = g.clone()
        g2[0, : e.shape[0]] += e.to(g2.dtype)
        self.colpi += 1
        return g2

    def togli(self) -> None:
        self.st._strato = self._orig
