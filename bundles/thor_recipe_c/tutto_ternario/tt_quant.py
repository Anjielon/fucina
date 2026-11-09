"""Ternarizzazione senza addestramento dei tensori densi di Thor-T (28/9), SOLO CPU.

Formato: TQ1_0R_1024 (segno S1024 + FWHT-1024 sull'asse d'ingresso, trit {-1,0,1}, una scala fp16 ogni 128) per ogni
tensore con ingresso multiplo di 1024; TQ1_0 (blocco 256, una scala fp16, niente rotazione: e' il formato di
ffn_down_exps, l'unico ternario a 256 servito dal banco del Dell) per ingresso 512 (ffn_down_shexp).

Criterio:
 * con Hessiana gia' catturata (H = X^T X dell'ingresso vero del tensore, sul bf16): nella base ruotata
   H' = R H R^T (R NON simmetrica col segno), smorzata 1 %: GPTQ (gptq_1024_lib, scala di gruppo = ottimo ESATTO dei
   minimi quadrati ternari invece della griglia a 6 soglie) -> coordinate descent esatta (trit ottimo per posizione
   con la forma quadratica completa + scale congiunte pesate, gptq_1024_coorddesc/refine) -> scale arrotondate a fp16
   -> un ultimo giro di trit con le scale fp16 VERE (quelle scritte nel file).
 * senza Hessiana: per ogni gruppo l'ottimo ESATTO di min ||w - d t||^2 (supporto = i k valori piu' grandi in modulo,
   d = media dei loro moduli, k che massimizza S_k^2/k), poi d a fp16 e t = arrotondamento di w/d16 (ottimo dato d16),
   due alternanze.
Ogni tensore esce con la sua misura: errore relativo non pesato, pesato-H (se c'e' H), round-trip dai byte.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QUI)
torch.set_num_threads(int(os.getenv("TT_THREADS", "4")))

import gptq_1024_lib as G  # noqa: E402
from tq1_0r_1024_codec import QK, BLOCK_BYTES, S1024, fwht1024_f32, dequantize_blocks, _pack_quarter  # noqa: E402
from gptq_1024_coorddesc import raffina_trit_sweep_batch_blocchi  # noqa: E402
from gptq_1024_refine import raffina_scale_joint  # noqa: E402

GROUP = G.GROUP                      # 128
SMORZAMENTO = 0.01
N_ROUND = int(os.getenv("TT_ROUND", "4"))
RIGHE_PEZZO = int(os.getenv("TT_RIGHE_PEZZO", "16384"))
DEVICE = os.getenv("TT_DEVICE", "cpu")   # 28/9 20:2x: "cuda" (ROCm) quando la GPU e' libera; aritmetica identica (GPTQ fp32, coordinate fp64)
TQ1_0_BYTES = 54                     # block_tq1_0: qs[48] qh[4] d(fp16)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ───────────────────────── ottimo ESATTO ternario di un gruppo (non pesato) ─────────────────────────
@torch.no_grad()
def scala_esatta(B: torch.Tensor) -> torch.Tensor:
    """B (righe, g) -> d (righe, 1): argmin_{d, t in {-1,0,1}^g} ||b - d t||^2. Per supporto fissato l'ottimo e'
    t = segno, d = media dei moduli sul supporto; il supporto ottimo di taglia k sono i k moduli maggiori; si sceglie
    k che massimizza S_k^2 / k. Esatto (non una griglia). Sostituisce gptq_1024_lib._scala_un_gruppo."""
    ab = B.abs().to(torch.float64)
    s, _ = torch.sort(ab, dim=-1, descending=True)
    cs = torch.cumsum(s, dim=-1)
    k = torch.arange(1, B.shape[-1] + 1, dtype=torch.float64, device=B.device)
    score = cs * cs / k
    i = torch.argmax(score, dim=-1, keepdim=True)
    d = torch.gather(cs, -1, i) / (i + 1).to(torch.float64)
    return d.to(B.dtype)


_GRIGLIA = G._scala_un_gruppo          # griglia a 6 soglie (ricetta 20/9, validata sul modello intero)
RICETTA = os.getenv("TT_RICETTA", "20set")   # "20set" (A1/A2, al bit del 20/9) | "ext" (A1-ext: scala esatta + giri trit)
assert RICETTA in ("20set", "ext"), RICETTA


def _lround(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def ternario_esatto(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X (N, g) float32 -> (d16 (N,) float32 esattamente rappresentabile in fp16, t (N, g) int8)."""
    d = scala_esatta(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).numpy()[:, 0].astype(np.float64)
    X64 = X.astype(np.float64)
    for _ in range(2):
        d16 = d.astype(np.float16).astype(np.float64)
        inv = np.where(d16 != 0, 1.0 / np.where(d16 != 0, d16, 1.0), 0.0)
        t = np.clip(_lround(X64 * inv[:, None]), -1, 1)
        n = np.abs(t).sum(1)
        d = np.where(n > 0, (np.abs(X64) * np.abs(t)).sum(1) / np.maximum(n, 1), 0.0)   # LS dato t (t = segno)
    d16 = d.astype(np.float16).astype(np.float64)
    inv = np.where(d16 != 0, 1.0 / np.where(d16 != 0, d16, 1.0), 0.0)
    t = np.clip(_lround(X64 * inv[:, None]), -1, 1).astype(np.int8)
    return d16.astype(np.float32), t


# ───────────────────────── TQ1_0R_1024 ─────────────────────────
def ruota(W: np.ndarray) -> np.ndarray:
    return G.ruota_righe_1024(W)


def smorza(Hrot: np.ndarray) -> np.ndarray:
    H = np.array(Hrot, dtype=np.float64, copy=True)
    morti = np.diag(H) == 0
    H[morti, morti] = 1.0
    H += np.eye(H.shape[0]) * (SMORZAMENTO * np.mean(np.diag(H)))
    return H


def _loss_rel(Wr: torch.Tensor, Q: torch.Tensor, D: torch.Tensor, Hd: torch.Tensor, gruppo: int = GROUP):
    """(sum_r e_r H e_r, sum_r w_r H w_r) con e = Wr - D*Q (mai matrici righe x righe)."""
    E = Wr.double() - Q.double() * D.double().repeat_interleave(gruppo, dim=1)
    return float(((E @ Hd) * E).sum()), float(((Wr.double() @ Hd) * Wr.double()).sum())


@torch.no_grad()
def _pesato(Wt: np.ndarray, Hb: np.ndarray, gruppo: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nucleo pesato su W gia' nella base giusta (ruotata per TQ1_0R_1024, originale per TQ1_0) e H nella stessa base.
    RICETTA 20set = forgia_thor_gptq_1024_refined.py del 20/9 AL BIT: Hchol = prepara_hessiana(H) (smorzamento 1 %),
    gptq_1024 con la griglia a 6 soglie, poi raffina_scale_joint con la H NON smorzata (fp32 -> fp64 dentro), scale fp16.
    RICETTA ext = scala esatta dentro GPTQ, N_ROUND giri (trit + scale congiunte) sulla H smorzata, scale fp16, un
    ultimo giro di trit con le scale fp16. Ritorna (D16 (R, ng) float32 fp16-esatte, Q (R, C) int8, catena)."""
    import gptq_1024_refine as RF
    R, C = Wt.shape
    vecchi = (G.GROUP, RF.GROUP)
    G.GROUP = RF.GROUP = gruppo
    G._scala_un_gruppo = _GRIGLIA if RICETTA == "20set" else scala_esatta
    try:
        Hc = torch.from_numpy(G.prepara_hessiana(Hb, SMORZAMENTO).astype(np.float32))
        Hd = torch.from_numpy(smorza(Hb)).to(DEVICE)          # obiettivo riportato (smorzato), per entrambe
        Wr = torch.from_numpy(np.ascontiguousarray(Wt, dtype=np.float32))
        D, Q = G.gptq_1024(Wr, Hc, device=DEVICE)             # su CPU in uscita: (R, ng, 1) f32, (R, C) int8
        D = D[:, :, 0]
        Wd_ = Wr.to(DEVICE)
        catena = {}
        num, den = _loss_rel(Wd_, Q.to(DEVICE), D.to(DEVICE), Hd, gruppo); catena["gptq"] = (num / den) ** 0.5
        if RICETTA == "20set":
            Hrot_t = torch.from_numpy(Hb.astype(np.float32))    # la H VERA, come Hrot_t del 20/9
            D = raffina_scale_joint(Wr, Q.float(), Hrot_t, device=DEVICE)
            catena["scale_congiunte"] = (_loss_rel(Wd_, Q.to(DEVICE), D.to(DEVICE), Hd, gruppo)[0] / den) ** 0.5
            D16 = D.to(torch.float16).to(torch.float32)
            catena["finale"] = (_loss_rel(Wd_, Q.to(DEVICE), D16.to(DEVICE), Hd, gruppo)[0] / den) ** 0.5
        else:
            Hd32 = Hd.float(); Qd = Q.to(DEVICE); Dd = D.to(DEVICE)
            for k in range(N_ROUND):
                Qd = raffina_trit_sweep_batch_blocchi(Wd_[None], Qd[None], Dd[None], Hd[None], device=DEVICE)[0].to(DEVICE)
                Dd = raffina_scale_joint(Wd_, Qd.float(), Hd32, device=DEVICE).to(DEVICE)
                catena[f"giro_{k+1}"] = (_loss_rel(Wd_, Qd, Dd, Hd, gruppo)[0] / den) ** 0.5
            D16d = Dd.to(torch.float16).to(torch.float32)
            catena["scale_fp16"] = (_loss_rel(Wd_, Qd, D16d, Hd, gruppo)[0] / den) ** 0.5
            Qd = raffina_trit_sweep_batch_blocchi(Wd_[None], Qd[None], D16d[None], Hd[None], device=DEVICE)[0].to(DEVICE)
            catena["finale"] = (_loss_rel(Wd_, Qd, D16d, Hd, gruppo)[0] / den) ** 0.5
            D16, Q = D16d.cpu(), Qd.cpu()
        return D16.numpy().astype(np.float32), Q.numpy().astype(np.int8), catena
    finally:
        G.GROUP, RF.GROUP = vecchi


def _err_pesato(E: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    Hh = torch.from_numpy(H.astype(np.float64)).to(DEVICE)
    Et = torch.from_numpy(E.astype(np.float64)).to(DEVICE); Wt = torch.from_numpy(W.astype(np.float64)).to(DEVICE)
    return float((((Et @ Hh) * Et).sum() / ((Wt @ Hh) * Wt).sum()) ** 0.5)


@torch.no_grad()
def quantizza_1024(W: np.ndarray, H: np.ndarray | None, nome: str = "") -> tuple[np.ndarray, dict]:
    """W (R, C) float32, C % 1024 == 0, ordine GGUF. Ritorna (blocchi (R*C/1024, 224) uint8, misure)."""
    R, C = W.shape
    assert C % QK == 0, (nome, W.shape)
    nb, ng = C // QK, C // GROUP
    t0 = time.time()
    Wrot = ruota(W)                                                   # (R, C) float32, base ruotata (S poi FWHT)
    mis: dict = {"formato": "TQ1_0R_1024", "righe": R, "colonne": C, "pesato": H is not None,
                 "ricetta": (RICETTA if H is not None else "ottimo_esatto_non_pesato"), "device": DEVICE}
    if H is None:
        d16, t = ternario_esatto(Wrot.reshape(R * ng, GROUP))
        D_all, Q_all = d16.reshape(R, ng), t.reshape(R, C)
    else:
        Hrot = G.ruota_hessiana_1024(H.astype(np.float64))           # H' = R H R^T (float64)
        D_all, Q_all, mis["catena_pesata_rel"] = _pesato(Wrot, Hrot, GROUP)
    packed = G.pack_da_d_q(D_all.reshape(R * nb, 8), Q_all.reshape(R * nb, QK))
    # round-trip DAI BYTE (codec = riferimento C, verificato bit-esatto il 20/9): i byte ridanno trit x scala fp16
    Wd = dequantize_blocks(packed).reshape(R, C)
    rec = (Q_all.astype(np.float32).reshape(R, ng, GROUP) * D_all[:, :, None]).reshape(R * nb, QK)
    rec = (fwht1024_f32(rec) * S1024[None, :]).reshape(R, C)
    mis["roundtrip_max_abs"] = float(np.abs(Wd - rec).max())
    E = (Wd - W).astype(np.float64)
    mis["err_rel"] = float(np.linalg.norm(E) / np.linalg.norm(W.astype(np.float64)))
    if H is not None:
        mis["err_rel_pesato_H"] = _err_pesato(E, W, H)
        d16, t = ternario_esatto(Wrot.reshape(R * ng, GROUP))       # riferimento: ottimo esatto NON pesato
        rec0 = (t.astype(np.float32).reshape(R, ng, GROUP) * d16.reshape(R, ng, 1)).reshape(R * nb, QK)
        E0 = ((fwht1024_f32(rec0) * S1024[None, :]).reshape(R, C) - W).astype(np.float64)
        mis["rif_nonpesato_err_rel"] = float(np.linalg.norm(E0) / np.linalg.norm(W.astype(np.float64)))
        mis["rif_nonpesato_err_rel_pesato_H"] = _err_pesato(E0, W, H)
    mis["frazione_zeri"] = float((Q_all == 0).mean())
    mis["secondi"] = round(time.time() - t0, 1)
    return packed, mis


# ───────────────────────── TQ1_0 (blocco 256, niente rotazione: ingresso 512) ─────────────────────────
def quantizza_tq1_0(W: np.ndarray, H: np.ndarray | None = None, nome: str = "") -> tuple[np.ndarray, dict]:
    import gguf
    R, C = W.shape
    assert C % 256 == 0, (nome, W.shape)
    t0 = time.time()
    mis = {"formato": "TQ1_0", "righe": R, "colonne": C, "pesato": H is not None,
           "ricetta": (RICETTA + "_g256_senza_rotazione" if H is not None else "ottimo_esatto_non_pesato"), "device": DEVICE}
    if H is None:
        d16, t = ternario_esatto(np.ascontiguousarray(W, dtype=np.float32).reshape(R * C // 256, 256))
    else:
        D, Q, mis["catena_pesata_rel"] = _pesato(W, H.astype(np.float64), 256)
        d16, t = D.reshape(-1), Q.reshape(R * C // 256, 256)
    qs, qh = _pack_quarter((t.astype(np.int64) + 1))
    packed = np.concatenate([qs, qh, d16.astype(np.float16).view(np.uint8).reshape(-1, 2)], axis=1)
    assert packed.shape[1] == TQ1_0_BYTES
    # round-trip con il dequantizzatore INDIPENDENTE di gguf-py (TQ1_0 standard di llama.cpp)
    Wd = gguf.quants.dequantize(packed.reshape(R, -1), gguf.GGMLQuantizationType.TQ1_0).reshape(R, C)
    rec = (t.astype(np.float32) * d16[:, None]).reshape(R, C)
    E = (Wd - W).astype(np.float64)
    mis.update({"roundtrip_max_abs": float(np.abs(Wd - rec).max()),
                "err_rel": float(np.linalg.norm(E) / np.linalg.norm(W.astype(np.float64))),
                "frazione_zeri": float((t == 0).mean())})
    if H is not None:
        mis["err_rel_pesato_H"] = _err_pesato(E, W, H)
        d0, t0_ = ternario_esatto(np.ascontiguousarray(W, dtype=np.float32).reshape(R * C // 256, 256))
        E0 = ((t0_.astype(np.float32) * d0[:, None]).reshape(R, C) - W).astype(np.float64)
        mis["rif_nonpesato_err_rel"] = float(np.linalg.norm(E0) / np.linalg.norm(W.astype(np.float64)))
        mis["rif_nonpesato_err_rel_pesato_H"] = _err_pesato(E0, W, H)
    mis["secondi"] = round(time.time() - t0, 1)
    return packed, mis
