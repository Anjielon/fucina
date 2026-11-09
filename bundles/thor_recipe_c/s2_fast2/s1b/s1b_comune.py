#!/usr/bin/env python3
"""S1b (piano casalingo di C, 13/10/2026) — parti comuni: basi, lettura/scrittura delle SOLE scale nei blocchi del GGUF,
sistema delle scale in forma chiusa, obiettivo, rotazione.

Il braccio S1b NON cambia un trit: i byte dei trit (qs/qh) di ogni blocco restano quelli dell'ospite al bit, cambiano solo
le scale fp16 (TQ1_0R_1024: 8 fp16 ai byte 208:224 di ogni blocco da 224 B, una ogni 128 pesi nella base ruotata;
TQ1_0: 1 fp16 ai byte 52:54 di ogni blocco da 54 B, una ogni 256 pesi, nessuna rotazione). Taglia, tipi, offset e
intestazione identici per costruzione; velocita' identica (stesso grafo, stessi tipi).

Basi (convenzioni di rd_ternary.gguf_piani, verificate nei suoi test contro ggml-quants.c):
  TQ1_0R_1024  base memorizzata w' = H(S w) per blocco 1024  =>  W' = W R^T, R = H1024 diag(S) ortogonale; l'effettivo
               nella base originale e' Q = Q' R; l'Hessiana dell'errore nella base memorizzata e' R H R^T (ruota_H).
  TQ1_0        nessuna rotazione.
Le statistiche di S1 sono nella base HF NON ruotata, che per gli ingressi coincide con l'ordine GGUF (unica permutazione
di colonne: ssm_out, che S1 salva gia' nell'ordine GGUF).

Obiettivo per riga (K=1, scala s_g per gruppo, codici t fissi):
  J(s) = s^T M s - 2 s^T b,  M[g,g'] = t_g Hs[g,g'] t_g'^T,  b[g] = t_g . Cs_g
  simmetrico  (errore d'uscita sul flusso di C, = ricetta di (c) con altra H):  Cs = W Hq
  asimmetrico (GPTAQ 2504.02692 / QEP 2504.09629, bersaglio = uscita del MAESTRO): Cs = W Mx, Mx = E[x_f x_q^T]
              E||W x_f - Q x_q||^2 = cost - 2 tr(Q (W Mx)^T) + tr(Q Hq Q^T)
Il minimo in forma chiusa per riga e' AGA di PT2-LLM (2510.03267) resa SIMMETRICA (niente offset: convenzione BitNet) e
asimmetrica nel senso di GPTAQ; per gli esperti e' il passo delle scale di JARQ (2609.38599 eq. 6) sulla H per esperto.
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

QUI = Path(__file__).resolve().parent
_LIB = QUI.parent / "lib"                 # albero congelato: <codice>/lib/{rd_ternary,sorgente.py,s1_comune.py}
for p in (str(_LIB),):
    if p not in sys.path:
        sys.path.insert(0, p)

from rd_ternary import gguf_piani as gp  # noqa: E402
from rd_ternary.formato import fwht  # noqa: E402

F64 = torch.float64
T_TQ1_0, T_TQ1_0R_1024 = 34, 45
# tipo -> (pesi per blocco, byte per blocco, byte delle scale nel blocco [a, b), scale per blocco, gruppo, ruotato)
FORMATO = {T_TQ1_0: (256, 54, (52, 54), 1, 256, False),
           T_TQ1_0R_1024: (1024, 224, (208, 224), 8, 128, True)}
FP16_MAX = 65504.0


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def scrivi_json(p, d) -> None:
    p = Path(p)
    t = Path(str(p) + ".tmp")
    t.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    os.replace(t, p)


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(x.dtype)


# ───────────────────────────── rotazione (copia di leve/raffina/comune.py, stesse convenzioni) ─────────────────────────────
def _s1024(dt, dev=None):
    return torch.from_numpy(gp.sign1024().astype(np.float32)).to(dtype=dt, device=dev)


def ruota(W: torch.Tensor) -> torch.Tensor:
    """righe nella base originale -> base memorizzata: w' = H(S w) per blocco 1024 (W' = W R^T)."""
    C = W.shape[-1]
    x = (W.reshape(*W.shape[:-1], C // 1024, 1024) * _s1024(W.dtype, W.device)).reshape(W.shape)
    return fwht(x, 1024)


def deruota(Wp: torch.Tensor) -> torch.Tensor:
    """base memorizzata -> originale: w = S * H(w') (Q = Q' R)."""
    y = fwht(Wp, 1024)
    C = Wp.shape[-1]
    return (y.reshape(*Wp.shape[:-1], C // 1024, 1024) * _s1024(Wp.dtype, Wp.device)).reshape(Wp.shape)


def ruota_H(H: torch.Tensor) -> torch.Tensor:
    """R H R^T."""
    Hr = ruota(H)
    return ruota(Hr.T.contiguous()).T.contiguous()


# ───────────────────────────── blocchi: trit e scale ─────────────────────────────
def decodifica(raw_rows: np.ndarray, tid: int, in_: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """righe grezze [R, byte_riga] -> (T int8 [R, C] base memorizzata, S f64 [R, ng] valori fp16)."""
    t, d = gp.decode_rows(np.ascontiguousarray(raw_rows), tid, in_)
    return torch.from_numpy(t.astype(np.int8)), torch.from_numpy(d.astype(np.float64))


def sostituisci_scale(raw_rows: np.ndarray, S16: np.ndarray, tid: int) -> np.ndarray:
    """Copia delle righe grezze con le SOLE scale sostituite (byte dei trit identici al bit)."""
    blk, bb, (a, b), nsb, _, _ = FORMATO[tid]
    R = raw_rows.shape[0]
    out = np.array(raw_rows, dtype=np.uint8, copy=True).reshape(-1, bb)
    s = np.ascontiguousarray(np.asarray(S16, dtype="<f2")).reshape(-1, nsb)
    assert s.shape[0] == out.shape[0], (s.shape, out.shape)
    out[:, a:b] = s.view(np.uint8).reshape(-1, b - a)
    return out.reshape(R, -1)


def maschera_scale(tid: int, nbytes: int) -> np.ndarray:
    """bool [nbytes]: True sui byte delle scale."""
    _, bb, (a, b), _, _, _ = FORMATO[tid]
    m = np.zeros(bb, dtype=bool)
    m[a:b] = True
    return np.tile(m, nbytes // bb)


def espandi(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    R, C = T.shape
    ng = S.shape[-1]
    return (T.to(F64).reshape(R, ng, C // ng) * S.to(F64).unsqueeze(-1)).reshape(R, C)


def q_originale(T, S, ruotato: bool, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """peso effettivo nella base ORIGINALE dell'ingresso (dopo un'eventuale scala per colonna c piegata a monte)."""
    Q = espandi(T, S)
    if ruotato:
        Q = deruota(Q)
    return Q if c is None else Q * c.to(F64).unsqueeze(0)


# ───────────────────────────── sistema delle scale (K=1) ─────────────────────────────
def basi_memorizzate(Hq: torch.Tensor, Clin: torch.Tensor, ruotato: bool, c: Optional[torch.Tensor] = None):
    """(Hq, Clin) nella base originale -> base memorizzata, con l'eventuale colonna c: l'effettivo e' Q' R diag(c), quindi
    Hs = R D Hq D R^T e Cs = (Clin D) R^T."""
    if c is not None:
        c = c.to(F64)
        Hq = Hq * c.unsqueeze(0) * c.unsqueeze(1)
        Clin = Clin * c.unsqueeze(0)
    if ruotato:
        return ruota_H(Hq), ruota(Clin)
    return Hq, Clin


def sistema(T: torch.Tensor, Hs: torch.Tensor, Cs: torch.Tensor, ng: int, lotto: int = 256):
    """M [R, ng, ng], b [R, ng] (vedi in testa). T int8 [R, C] nella base memorizzata."""
    R, C = T.shape
    G = C // ng
    M = torch.empty(R, ng, ng, dtype=F64)
    b = torch.empty(R, ng, dtype=F64)
    Hs = Hs.to(F64)
    for a in range(0, R, lotto):
        z = min(R, a + lotto)
        r = z - a
        A3 = T[a:z].to(F64).reshape(r, ng, G)
        BH = torch.empty(r, ng, C, dtype=F64)
        for g in range(ng):
            BH[:, g, :] = A3[:, g, :] @ Hs[g * G:(g + 1) * G, :]
        M[a:z] = (BH.reshape(r, ng, ng, G) * A3.unsqueeze(1)).sum(-1)
        b[a:z] = (A3 * Cs[a:z].to(F64).reshape(r, ng, G)).sum(-1)
    return M, b


def risolvi(M: torch.Tensor, b: torch.Tensor, S0: torch.Tensor, ridge: float = 1e-8) -> torch.Tensor:
    """argmin_s s^T M s - 2 s^T b per riga (non arrotondato). Gruppi senza codici (diagonale 0): scala dell'ospite."""
    dia = torch.diagonal(M, dim1=1, dim2=2)
    vuoti = dia <= 0
    Mm = M + torch.diag_embed(vuoti.to(F64))
    sc = dia.clamp_min(0).mean(1, keepdim=True).clamp_min(1e-300)
    sol = None
    try:
        L = torch.linalg.cholesky(Mm + ridge * torch.diag_embed(sc.expand_as(dia)))
        sol = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        pass
    if sol is None or not torch.isfinite(sol).all():
        sol = torch.linalg.lstsq(Mm, b.unsqueeze(-1)).solution.squeeze(-1)
    return torch.where(vuoti, S0.to(F64), sol)


def J(M: torch.Tensor, b: torch.Tensor, S: torch.Tensor) -> float:
    """obiettivo (a meno della costante) sommato sulle righe."""
    S = S.to(F64)
    return float((torch.einsum("rg,rgh,rh->r", S, M, S) - 2 * (S * b).sum(1)).sum())


def scala_ok(S: torch.Tensor, S0: torch.Tensor) -> bool:
    """scale nuove scrivibili: finite, dentro fp16, e nessuna scala non nulla dell'ospite diventata zero."""
    return bool(torch.isfinite(S).all() and (S.abs() <= FP16_MAX).all() and not ((S0 != 0) & (S == 0)).any())


# ───────────────────────────── scala per colonna (passo c, forma chiusa) ─────────────────────────────
def passo_colonne(Qo: torch.Tensor, Hq: torch.Tensor, Clin: torch.Tensor, P: Optional[torch.Tensor] = None,
                  ridge: float = 1e-9) -> torch.Tensor:
    """Dato l'effettivo SENZA colonna Qo [R, C] (base originale), c = argmin J(Qo diag(c)):
    (Hq ⊙ Qo^T Qo) c = rowsum(Qo ⊙ Clin)  (ARB-LLM_RC 2410.03129 eq. 13, pesata H = variante Thor dichiarata;
    con Clin = W Mx diventa asimmetrica). P [C, m] (facoltativa) lega le colonne: c = P c_m (GQA attn_v -> attn_output)."""
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


def legame_gqa(n_head: int, n_kv: int, head_dim: int) -> torch.Tensor:
    """P [n_head*head_dim, n_kv*head_dim]: colonna j di attn_output (testa q h, dim d) <- riga di attn_v (testa kv
    h // (n_head/n_kv), dim d). Come repeat_kv di transformers (teste kv ripetute in blocchi contigui)."""
    rep = n_head // n_kv
    P = torch.zeros(n_head * head_dim, n_kv * head_dim, dtype=F64)
    j = torch.arange(n_head * head_dim)
    P[j, (j // head_dim // rep) * head_dim + j % head_dim] = 1.0
    return P


def ristretta(H_e: torch.Tensor, n_e: float, H_s: torch.Tensor, k: float) -> torch.Tensor:
    """restringimento verso la statistica di strato (variante Thor dichiarata, come s1_comune.ristretta)."""
    return (n_e * H_e + k * H_s) / (n_e + k)
