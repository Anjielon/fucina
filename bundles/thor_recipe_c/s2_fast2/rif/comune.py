"""Leve di RAFFINAMENTO del quantizzatore su R2B3m — parte comune (solo CPU, nessuna GPU).

Gruppo «raffinamento» (7/10/2026): #1 Tied Trit-Planes, #6 JARQ, #8 FastKron, #11 SchurQuant, #12 QTEA, #15 AWSRC,
#17 G2PTQ, #21 ICBQ, applicate come TRASFORMAZIONI di R2B3m che restano caricabili dal motore attuale
(stessi nomi, tipi, forme e byte per tensore: il GGUF nuovo e' R2B3m con alcuni tensori riscritti AL POSTO).

Convenzioni (identiche a rd_ternary.gguf_piani, che legge il file):
  * TQ1_0R_1024: base memorizzata w' = H(S w) per blocco 1024 (S segni fissi, H Walsh-Hadamard normalizzata),
    W' = sum_k s_k T_k con una scala fp16 ogni 128; dequant w = S * H(w').  ⇒  W' = W Rᵀ con R = H·diag(S)
    (R ortogonale), e l'Hessiana dell'errore d'uscita nella base memorizzata e' H' = R H Rᵀ.
  * TQ1_0 (ffn_down_shexp, ingresso 512): niente rotazione, scala ogni 256.
Bersaglio dei densi = quello con cui sono nati i piani 2-3 di R2B3m (tutto_ternario/tt_quant, ricetta 20set):
bf16 pieno + Delta W LoRA di EB3 p20 (`sorgente.SorgenteHF.peso_bersaglio`), Hessiana = quella scritta nel manifesto
`tensori_R2B3/manifesto.json` per lo stesso tensore (corpus piccolo del 20/9, H = XᵀX del bf16).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

RD = "@@ROOT:models@@/build/wt-r022b/thor_v2/rd_ternary"
TT = "@@ROOT:depot@@/tutto_ternario/codice"
for p in (RD, TT):
    if p not in sys.path:
        sys.path.insert(0, p)

from rd_ternary import gguf_piani as gp  # noqa: E402
from rd_ternary.formato import fwht  # noqa: E402

R2B3M = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"
SHA_R2B3M = "56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43"
# 9/10: OSPITE = GGUF da cui si leggono i piani (default R2B3m). Con PV12P/COMBO A le 68 famiglie dense che hanno il
# QUARTO piano (nome4.weight) si leggono e si riscrivono a 4 piani: rifare i piani 1-3 lasciando il 4° di PV12P
# (adattato al residuo dei piani AVIDI di R2B3m) aggiunge rumore — e' cio' che e' successo in COMBO A.
OSPITE = os.getenv("LEVE_OSPITE", R2B3M)
MANIF_R2B3 = "@@ROOT:depot@@/tutto_ternario/tensori_R2B3/manifesto.json"
OUT_BASE = Path(os.getenv("LEVE_OUT", "@@ROOT:models@@/gguf/tony-tern/LEVE_RAFF"))

torch.set_num_threads(int(os.getenv("LEVE_THREADS", "8")))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ─────────────────────────────────────────── rotazione ───────────────────────────────────────────
def ruota(W: torch.Tensor) -> torch.Tensor:
    """Righe nella base originale -> base memorizzata: w' = H(S w) per blocco 1024."""
    s = torch.from_numpy(gp.sign1024().astype(np.float32)).to(W.dtype)
    C = W.shape[-1]
    x = (W.reshape(*W.shape[:-1], C // 1024, 1024) * s).reshape(W.shape)
    return fwht(x, 1024)


def deruota(Wp: torch.Tensor) -> torch.Tensor:
    s = torch.from_numpy(gp.sign1024().astype(np.float32)).to(Wp.dtype)
    y = fwht(Wp, 1024)
    C = Wp.shape[-1]
    return (y.reshape(*Wp.shape[:-1], C // 1024, 1024) * s).reshape(Wp.shape)


def ruota_H(H: torch.Tensor) -> torch.Tensor:
    """H' = R H Rᵀ (R = H1024·diag(S) a blocchi): ruota righe e colonne."""
    Hr = ruota(H)                 # R applicata dal lato delle colonne: H Rᵀ (ogni riga ruotata)
    return ruota(Hr.T.contiguous()).T.contiguous()


# ─────────────────────────────────────────── tensori ─────────────────────────────────────────────
class Denso:
    """Un tensore denso multi-piano di un GGUF (piani, scale, tipo) + bersaglio e Hessiana nella base memorizzata."""

    def __init__(self, src: "gp.GGUFPlaneSource", nome: str):
        self.nome = nome                                    # es. blk.3.attn_qkv.weight
        L = int(nome.split(".")[1]) if nome.startswith("blk.") else -1
        base = nome.split(".")[-2] if L >= 0 else nome[:-len(".weight")]
        self.key = gp.TensorKey(L, -1, base) if hasattr(gp, "TensorKey") else None
        from rd_ternary.formato import TensorKey
        self.key = TensorKey(L, -1, base)
        e = src.entry(self.key)
        self.tid = e.fmt.type_id
        self.rotated = e.fmt.rotated
        self.group = e.fmt.group
        self.out, self.in_ = e.out, e.in_
        self.nomi_piani = [t.name for t in e.tensors]
        tt = src.load(self.key)
        self.T = tt.planes.clone()                          # int8 [K, R, C]
        self.S = tt.scales.float().clone()                  # fp32 (valori fp16) [K, R, ng]
        self.K = self.T.shape[0]

    def ricostruisci(self, T=None, S=None) -> torch.Tensor:
        T = self.T if T is None else T
        S = self.S if S is None else S
        K, R, C = T.shape
        ng = S.shape[-1]
        w = torch.zeros(R, C, dtype=torch.float64)
        for k in range(K):
            w += (T[k].double().reshape(R, ng, C // ng) * S[k].double().unsqueeze(-1)).reshape(R, C)
        return w

    def byte_piani(self, T, S) -> List[np.ndarray]:
        out = []
        for k in range(T.shape[0]):
            s16 = S[k].to(torch.float16).numpy()
            out.append(gp.encode_rows(T[k].numpy().astype(np.int8), s16, self.tid))
        return out


_SORG = None
_MANIF = None


def bersaglio_e_H(nome: str, rotated: bool, smorza_rel: float = 0.0) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
    """(W*' [R,C] f64, H' [C,C] f64 o None, provenienza) nella base memorizzata del tensore."""
    global _SORG, _MANIF
    if _SORG is None:
        from sorgente import SorgenteHF
        _SORG = SorgenteHF()
        _MANIF = json.load(open(MANIF_R2B3))
    W, prov = _SORG.peso_bersaglio(nome)
    W = torch.from_numpy(W).double()
    k3 = nome.replace(".weight", "3.weight")
    hinfo = _MANIF.get(k3, {}).get("hessiana")
    H = None
    if hinfo:
        H = torch.from_numpy(np.load(hinfo["file"])).double()
        prov["hessiana"] = hinfo
    if rotated:
        W = ruota(W)
        if H is not None:
            H = ruota_H(H)
    if H is not None:
        H = 0.5 * (H + H.T)
        d = torch.diagonal(H)
        morti = d <= 0
        if morti.any():
            H[morti, morti] = 1.0
        prov["diag_media_grezza"] = float(d.clamp_min(1e-30).mean())   # 9/10: per riportare il beta di FOEM
        H = H / d.clamp_min(1e-30).mean()                   # scala adimensionale: mean(diag) = 1
        if smorza_rel > 0:
            H = H + smorza_rel * torch.eye(H.shape[0], dtype=H.dtype)
    return W, H, prov


def err_pesato(W: torch.Tensor, What: torch.Tensor, H: Optional[torch.Tensor]) -> Tuple[float, float]:
    """(errore relativo pesato-H, errore relativo di Frobenius)."""
    E = (W - What).double()
    fro = float(E.norm() / W.double().norm())
    if H is None:
        return fro, fro
    num = float(((E @ H) * E).sum())
    den = float(((W.double() @ H) * W.double()).sum())
    return (num / den) ** 0.5, fro


def elenco_densi(src) -> List[str]:
    """I 250 densi multi-piano dei blocchi 0-39 con Hessiana nel manifesto R2B3 (niente embedding/testa/MTP)."""
    man = json.load(open(MANIF_R2B3))
    nomi = []
    for k3 in sorted(man):
        nome = k3.replace("3.weight", ".weight")
        L = int(nome.split(".")[1])
        if L < 40:
            nomi.append(nome)
    return nomi


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
