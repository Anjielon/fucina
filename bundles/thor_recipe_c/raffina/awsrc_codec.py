#!/usr/bin/env python3
"""#15 AWSRC (arXiv 2608.23144) — codec a semi deterministici sul residuo degli ESPERTI di R2B3m, solo CPU.

Il motore attuale (llama.cpp Vulkan, tipi TQ1_0/TQ1_0R_1024) NON sa rigenerare basi da un seme: il «sidecar» di AWSRC
non e' caricabile. Qui si implementa il codec fedele (AWSRC-U: C=128, P=2, b=4, S=2; basi = colonne di Hadamard
normalizzate con segni e permutazione dal seme; LS pesato con a_k = E[x_k^2]; coefficienti RTN a 4 bit con una scala
potenza di 2 per tessera; record 7 byte; guadagno per byte rho_t) e lo si CONFRONTA, a parita' di byte e sullo stesso
obiettivo pesato, con l'unica forma che il motore carica: un 2° piano ternario sugli esperti (TQ1_0R_1024, 224 B/1024).

uso: awsrc_codec.py --strato 10 --famiglia gate|up [--esperti 0-255]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comune as CM  # noqa: E402

GQ = "@@ROOT:models@@/gguf/tony-tern/pod-schur-esperti"


def hadamard(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def basi(C=128, P=2, S=2, seme0=0):
    out = []
    Hn = hadamard(C)
    for s in range(S):
        g = torch.Generator().manual_seed(seme0 + s)
        segni = torch.randint(0, 2, (C,), generator=g).double() * 2 - 1
        perm = torch.randperm(C, generator=g)
        col = torch.randperm(C, generator=g)[:P]
        out.append((segni.unsqueeze(1) * Hn[perm][:, col]))           # [C, P]
    return torch.stack(out)                                            # [S, C, P]


def awsrc_u(R, a, C=128, P=2, b=4, S=2, eps=1e-8):
    """R [N, Cin] residuo, a [Cin] pesi d'attivazione. Ritorna (Rhat, guadagno per tessera, errore prima/dopo)."""
    N, Cin = R.shape
    nt = Cin // C
    Rt = R.reshape(N, nt, C)
    at = a.reshape(1, nt, C)
    B = basi(C, P, S)                                                  # [S, C, P]
    best_err = torch.full((N, nt), float("inf"), dtype=torch.float64)
    best_rec = torch.zeros_like(Rt)
    qmax = 2 ** (b - 1) - 1
    for s in range(S):
        Bs = B[s]
        M = torch.einsum("tc,cp,cq->tpq", at[0], Bs, Bs) + eps * torch.eye(P, dtype=torch.float64)   # [nt, P, P]
        rhs = torch.einsum("ntc,tc,cp->ntp", Rt, at[0], Bs)
        alpha = torch.linalg.solve(M.unsqueeze(0).expand(N, -1, -1, -1), rhs.unsqueeze(-1)).squeeze(-1)  # [N, nt, P]
        amax = alpha.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        esp = torch.ceil(torch.log2(amax / qmax))                      # scala potenza di 2 per tessera
        sc = torch.pow(2.0, esp)
        aq = torch.clamp(torch.round(alpha / sc), -qmax - 1, qmax) * sc
        rec = torch.einsum("ntp,cp->ntc", aq, Bs)
        err = ((Rt - rec) ** 2 * at).sum(-1)
        m = err < best_err
        best_err = torch.where(m, err, best_err)
        best_rec = torch.where(m.unsqueeze(-1), rec, best_rec)
    err0 = (Rt ** 2 * at).sum(-1)
    gain = err0 - best_err                                             # rho_t * L_t
    return best_rec.reshape(N, Cin), gain, err0, best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strato", type=int, default=10)
    ap.add_argument("--famiglia", default="gate")
    ap.add_argument("--esperti", default="0-63")
    a = ap.parse_args()
    e0, e1 = (int(x) for x in a.esperti.split("-"))
    L = a.strato
    sys.path.insert(0, CM.TT)
    from sorgente import SorgenteHF
    src = CM.gp.GGUFPlaneSource(CM.R2B3M, layers=[L])
    sh = SorgenteHF()
    Gq = np.load(f"{GQ}/Gq_{L:02d}.npy", mmap_mode="r")
    righe = []
    tot = {"err0": 0.0, "awsrc": 0.0, "piano2": 0.0, "n": 0}
    guadagni = []
    # pesi bf16 degli esperti (fusi gate_up in HF): via la mappa della sorgente
    t0 = time.time()
    Wall = esperti_bf16(sh, L, a.famiglia)                            # [256, 512, 2048] ordine GGUF
    for e in range(e0, e1 + 1):
        key = CM.gp.TensorKey(L, e, a.famiglia) if hasattr(CM.gp, "TensorKey") else None
        from rd_ternary.formato import TensorKey
        tt = src.load(TensorKey(L, e, a.famiglia))
        W0 = torch.from_numpy(CM.gp.reconstruct(tt, original_basis=True)).double()   # [512, 2048]
        W = torch.from_numpy(Wall[e]).double()
        R = W - W0
        off = 0 if a.famiglia == "gate" else 1
        aw = torch.from_numpy(np.diagonal(Gq[e]).astype(np.float64)).clamp_min(0)
        aw = aw / aw.mean().clamp_min(1e-30)
        Rhat, gain, err0, err1 = awsrc_u(R, aw)
        guadagni.append(gain.reshape(-1))
        # 2° piano ternario (caricabile): ottimo ternario esatto per gruppo di 128 sul residuo RUOTATO, pesato nella base
        # originale con gli stessi a_k (stessa misura)
        Rr = CM.ruota(R)
        Xg = Rr.reshape(-1, 128)
        ab = Xg.abs()
        so, _ = torch.sort(ab, 1, descending=True)
        cs = torch.cumsum(so, 1)
        k = torch.arange(1, 129, dtype=torch.float64)
        i = torch.argmax(cs * cs / k, 1, keepdim=True)
        d = (torch.gather(cs, 1, i) / (i + 1)).to(torch.float16).double()
        t = torch.clamp(torch.round(Xg / d.clamp_min(1e-30)), -1, 1)
        P2 = CM.deruota((t * d).reshape(R.shape))
        errp2 = (((R - P2) ** 2) * aw).sum()
        tot["err0"] += float(err0.sum()); tot["awsrc"] += float(err1.sum()); tot["piano2"] += float(errp2); tot["n"] += 1
    nw = tot["n"] * 512 * 2048
    byte_awsrc = tot["n"] * 512 * 16 * 7                              # una tessera = 128 pesi, record 7 B (tutte le tessere)
    byte_p2 = nw * 224 / 1024
    res = {"strato": L, "famiglia": a.famiglia, "esperti": a.esperti,
           "errore_pesato_residuo_rel_awsrc": tot["awsrc"] / tot["err0"], "errore_pesato_residuo_rel_piano2": tot["piano2"] / tot["err0"],
           "bpw_awsrc_tutte_le_tessere": 8 * byte_awsrc / nw, "bpw_piano2": 8 * byte_p2 / nw,
           "guadagno_per_byte_awsrc": (tot["err0"] - tot["awsrc"]) / byte_awsrc,
           "guadagno_per_byte_piano2": (tot["err0"] - tot["piano2"]) / byte_p2,
           "guadagno_per_byte_awsrc_miglior_10pct_tessere": float(torch.sort(torch.cat(guadagni), descending=True)[0][: max(1, len(torch.cat(guadagni)) // 10)].sum()) / (byte_awsrc / 10),
           "secondi": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1))
    out = CM.OUT_BASE / "awsrc"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "codec.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


def esperti_bf16(sh, L, fam):
    """[256, 512, 2048] float32 nell'ordine del GGUF per gate|up (fetta del gate_up_proj fuso di HF)."""
    nome = f"blk.{L}.ffn_{fam}_exps.weight"
    v = sh.voce(nome)
    u16, _ = sh.leggi_hf(v["hf"])
    W = (u16.astype(np.uint32) << 16).view(np.float32)
    fetta = v.get("fetta")
    if fetta == "righe[:512]":
        W = W[:, :512, :]
    elif fetta == "righe[512:]":
        W = W[:, 512:, :]
    assert v.get("rel") == "identita", v
    return np.ascontiguousarray(W)


if __name__ == "__main__":
    main()
