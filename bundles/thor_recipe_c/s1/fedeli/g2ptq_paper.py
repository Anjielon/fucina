#!/usr/bin/env python3
"""G²PTQ FEDELE al lavoro (arXiv 2609.31009) e al codice ufficiale (github.com/G2PTQ/G2PTQ, `gptq_utils/g2ptq_utils.py`,
script `scripts/g2ptq.sh`), nucleo per UN modulo lineare. 10/10/2026 sera.

L'audit (ricerche/AUDIT_ADERENZA_PAPER.md) aveva trovato tre scostamenti nella nostra leva `g2ptq_p23`: α fisso a 1
invece della ricerca per blocco su [0, 3]; curvatura dalla Fisher EMPIRICA della KL invece della Fisher con etichette
CAMPIONATE; gradiente misurato una volta sul modello gia' quantizzato invece che blocco per blocco sul prefisso
quantizzato. Qui il metodo del codice ufficiale, pezzo per pezzo:

  * salienza (Hessiana a gruppi di righe): per ogni token il gradiente rispetto all'USCITA del modulo di una perdita
    con bersaglio CAMPIONATO — MSE contro uscita + rumore N(0, d) (`compute_mse_grad_hessian`) per i blocchi interni,
    NLL con etichette campionate dal maestro top-256 (`compute_kl_grad_hessian`) per l'ultimo blocco; al quadrato,
    media per gruppo di righe (num_groups = 4) → s[t, g] (`SaliencyCache.grad_hook`);
  * H_g = media sui lotti di Σ_t s[t, g] x_t x_tᵀ / n_token (`G2PTQ.add_batch`, stessa normalizzazione);
  * gradiente del peso della perdita VERA del blocco (MSE uscita del blocco vs uscita del blocco in precisione piena,
    o KL top-256 sull'ultimo), media per token (`GradientCache.cache_gradient`), con il blocco in precisione piena sul
    flusso del PREFISSO QUANTIZZATO e rinfrescato blocco per blocco;
  * passo di compensazione col gradiente e regione di fiducia: β per riga dall'Eq. 56 (righe 460-470 del file), Z =
    (g Hinvᵀ)·β, GHinv = Z Hinv, poi il ciclo per colonna (`_run_torch_compensation`) e l'aggiornamento fra blocchi
    (`_run_compensation`);
  * quantizzatore: simmetrico per canale con ricerca del clipping MSE (norma 2,4, griglia 50, restringimento 0,5),
    act_order con gruppi statici, smorzamento 1 % con incremento automatico 0,0015;
  * α per blocco: griglia [0, 3] passo 0,1, arresto al primo peggioramento (margine = passo), perdita del blocco su
    64 campioni (Alg. 3) — nel motore sequenziale, non qui.

Variante THOR (dichiarata): il quantizzatore per colonna e' il formato ternario a K piani di Thor (scale per piano
e per gruppo di 128 colonne, livelli Σ_k a_k t_k), scale stimate all'inizio (gruppi statici, come act_order) col
min-max ternario a piani avidi e ricerca del restringimento MSE come il clipping del codice ufficiale."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch

F32 = torch.float32


# ───────────────────────────────────── salienza e Hessiana a gruppi ─────────────────────────────────────
def salienza_da_grad_uscita(gy: torch.Tensor, num_groups: int = 4) -> torch.Tensor:
    """`SaliencyCache.grad_hook`: gy [..., d_out] → s [n_token, num_groups] = media sul gruppo di gy²."""
    gy = gy.reshape(-1, gy.shape[-1]).float()
    n, d = gy.shape
    return gy.pow(2).view(n, num_groups, d // num_groups).mean(-1)


class HessianaGruppi:
    """`G2PTQ.add_batch` per un solo modulo: H[g] = media sui lotti di (Σ_t s[t,g] x x ᵀ)/n_token."""

    def __init__(self, din: int, num_groups: int = 4, device="cpu"):
        self.H = torch.zeros(num_groups, din, din, dtype=F32, device=device)
        self.act_square = torch.zeros(din, dtype=F32, device=device)
        self.index = 0
        self.total_tokens = 0

    @torch.no_grad()
    def aggiungi(self, x: torch.Tensor, s: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        self.H *= self.index / (self.index + 1)
        self.index += 1
        self.act_square *= self.total_tokens / (self.total_tokens + n)
        self.total_tokens += n
        sw = torch.einsum("nj,ng->njg", x, s.float())
        blk = torch.einsum("ni,njg->gij", x, sw)
        self.H.add_(blk, alpha=1.0 / (n * self.index))
        self.act_square.add_((x ** 2).sum(0), alpha=1.0 / self.total_tokens)


class GradienteMedio:
    """`GradientCache.cache_gradient`: media corrente di grad/n_token sui lotti."""

    def __init__(self):
        self.g = None
        self.index = 0

    def aggiungi(self, grad: torch.Tensor, n_tokens: int) -> None:
        c = 0 if self.g is None else self.g
        c = c * (self.index / (self.index + 1))
        self.index += 1
        self.g = c + grad.float() / (n_tokens * self.index)


# ───────────────────────────────────── quantizzatore uniforme simmetrico con clipping MSE ─────────────────────────────────────
def trova_parametri_sym(x: torch.Tensor, bits: int, mse: bool = True, norm: float = 2.4, grid: int = 50,
                        maxshrink: float = 0.5) -> torch.Tensor:
    """`WeightQuantizer.find_params` (sym, perchannel): x [righe, colonne del gruppo] → scala per riga.
    maxq = 2^(b-1) − 1, minq = −(maxq+1), bit_range = maxq − minq; scala = max(|xmin|, xmax)/(bit_range/2)."""
    maxq = 2 ** (bits - 1) - 1
    bit_range = 2 * maxq + 1
    xmin = torch.minimum(x.amin(-1, keepdim=True), torch.zeros_like(x[:, :1]))
    xmax = torch.maximum(x.amax(-1, keepdim=True), torch.zeros_like(x[:, :1]))
    scale = torch.maximum(xmin.abs(), xmax) / (bit_range / 2)
    if mse:
        best = torch.full(x.shape[:-1], float("inf"), dtype=x.dtype, device=x.device)
        for i in range(int(maxshrink * grid)):
            for j in range(int(maxshrink * grid)):
                xmin1 = (1 - i / grid) * xmin
                xmax1 = (1 - j / grid) * xmax
                s1 = torch.maximum(xmin1.abs(), xmax1) / (bit_range / 2)
                q = torch.clamp(torch.round(x / s1), -(maxq + 1), maxq) * s1
                err = (q - x).abs().pow(norm).sum(-1)
                meglio = err < best
                scale = torch.where(meglio.unsqueeze(-1), s1, scale)
                best = torch.where(meglio, err, best)
    return scale.squeeze(-1)


def quantizza_sym(w: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    maxq = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(w / scale), -(maxq + 1), maxq) * scale


# ───────────────────────────────────── quantizzatore THOR (ternario a K piani) ─────────────────────────────────────
def _tabella(K: int, device) -> torch.Tensor:
    import itertools
    return torch.tensor(list(itertools.product((-1.0, 0.0, 1.0), repeat=K)), dtype=F32, device=device)


def scale_ternarie_mse(x: torch.Tensor, K: int, grid: int = 50, maxshrink: float = 0.5,
                       norm: float = 2.4) -> torch.Tensor:
    """Scale [K, righe] per il gruppo x [righe, g]: piani avidi (scala del piano = 2/3 del max del residuo) moltiplicati
    per un fattore di restringimento comune cercato sulla stessa griglia del clipping MSE ufficiale."""
    tab = _tabella(K, x.device)
    best = torch.full(x.shape[:-1], float("inf"), dtype=x.dtype, device=x.device)
    out = torch.zeros(K, x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(int(maxshrink * grid)):
        f = 1 - i / grid
        res = x.clone()
        sc = torch.zeros(K, x.shape[0], dtype=x.dtype, device=x.device)
        for k in range(K):
            a = (res.abs().amax(-1) * (2.0 / 3.0) * f).clamp_min(1e-12)
            c = torch.clamp(torch.round(res / a.unsqueeze(1)), -1, 1)
            sc[k] = a
            res = res - a.unsqueeze(1) * c
        q = livello_vicino(x, sc, tab)
        err = (q - x).abs().pow(norm).sum(-1)
        meglio = err < best
        out = torch.where(meglio.unsqueeze(0), sc, out)
        best = torch.where(meglio, err, best)
    return out.to(torch.float16).to(x.dtype)


def livello_vicino(w: torch.Tensor, sc: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    """w [righe, n] (o [righe]), sc [K, righe]: il livello Σ_k a_k t_k piu' vicino per ogni elemento."""
    V = sc.T @ tab.T                                              # [righe, 3^K]
    if w.dim() == 1:
        k = (w.unsqueeze(1) - V).abs().argmin(1)
        return V.gather(1, k.unsqueeze(1)).squeeze(1)
    k = (w.unsqueeze(-1) - V.unsqueeze(1)).abs().argmin(-1)      # [righe, n]
    return V.gather(1, k)


def codici_da_livelli(q: torch.Tensor, sc: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    """Ricava i trit [K, righe, n] del livello q (esatto: q e' uno dei valori di V)."""
    V = sc.T @ tab.T
    k = (q.unsqueeze(-1) - V.unsqueeze(1)).abs().argmin(-1)
    return tab[k].permute(2, 0, 1).to(torch.int8)


# ───────────────────────────────────── fasterquant (un modulo, num_groups gruppi di righe) ─────────────────────────────────────
def fasterquant(W: torch.Tensor, H: torch.Tensor, grad: Optional[torch.Tensor], alpha: float,
                act_square: torch.Tensor, quant: str = "sym", bits: int = 2, K: int = 3, groupsize: int = -1,
                blocksize: int = 128, percdamp: float = 0.01, actorder: bool = True, w_clip: bool = True,
                enable_gradient_update: bool = True) -> Dict[str, torch.Tensor]:
    """Porta di `G2PTQ.fasterquant` (percorso torch) per UN modulo W [rows, cols] con H [num_groups, cols, cols].
    quant="sym": quantizzatore del lavoro (bits, per canale se groupsize = -1, clipping MSE se w_clip).
    quant="thor": formato ternario a K piani, scale per gruppo di `groupsize` colonne (gruppi statici)."""
    rows, cols = W.shape
    ng = H.shape[0]
    rps = rows // ng
    W = W.float().clone()
    Wsub = W.view(ng, rps, cols).clone()
    gsub = (torch.zeros_like(Wsub) if grad is None else grad.float().view(ng, rps, cols).clone())
    Hs = H.float().clone()
    for b in range(ng):
        if bool((Hs[b] == 0).all()):
            Hs[b] = torch.eye(cols, dtype=F32, device=W.device)
            gsub[b].zero_()
        dead = torch.diag(Hs[b]) == 0
        Hs[b, dead, dead] = 1
        Wsub[b][:, dead] = 0
    # parametri del quantizzatore (gruppi statici: calcolati su W prima della compensazione, ordine ORIGINALE)
    gsz = cols if groupsize == -1 else groupsize
    ngr = cols // gsz
    tab = _tabella(K, W.device) if quant == "thor" else None
    Wflat = Wsub.reshape(rows, cols)
    if quant == "sym":
        scale = torch.stack([trova_parametri_sym(Wflat[:, i * gsz:(i + 1) * gsz], bits, mse=w_clip)
                             for i in range(ngr)], 1)                     # [rows, ngr]
    else:
        scale = torch.stack([scale_ternarie_mse(Wflat[:, i * gsz:(i + 1) * gsz], K) for i in range(ngr)], -1)
        # [K, rows, ngr]

    # indice di gruppo (ordine ORIGINALE) per ogni colonna
    gidx = torch.arange(cols, device=W.device) // gsz

    def qcols(w: torch.Tensor, j_orig: torch.Tensor) -> torch.Tensor:
        """w [ng, rps, n] colonne (indici originali j_orig [n]) per tutti i gruppi di righe insieme → livelli."""
        gi = gidx[j_orig]                                            # [n]
        if quant == "sym":
            sc = scale.view(ng, rps, ngr)[:, :, gi]                  # [ng, rps, n]
            maxq = 2 ** (bits - 1) - 1
            return torch.clamp(torch.round(w / sc), -(maxq + 1), maxq) * sc
        sc = scale.view(K, ng, rps, ngr)[:, :, :, gi]                 # [K, ng, rps, n]
        V = torch.einsum("kbrn,ak->brna", sc, tab)                   # [ng, rps, n, 3^K]
        k = (w.unsqueeze(-1) - V).abs().argmin(-1)
        return V.gather(-1, k.unsqueeze(-1)).squeeze(-1)

    # Q_sub per β (fake-quant RTN di W coi parametri, Eq. 56 usa l'errore RTN)
    if actorder:
        perm = torch.argsort(act_square, descending=True)
        invperm = torch.argsort(perm)
    else:
        perm = torch.arange(cols, device=W.device)
        invperm = perm
    Qrtn = torch.cat([qcols(Wsub[:, :, j0:j0 + 256], torch.arange(j0, min(j0 + 256, cols), device=W.device))
                      for j0 in range(0, cols, 256)], -1)
    Hinv_init = torch.empty_like(Hs)
    Hinv = torch.empty_like(Hs)
    for b in range(ng):
        Wsub[b] = Wsub[b][:, perm]
        Hs[b] = Hs[b][perm][:, perm]
        gsub[b] = gsub[b][:, perm]
        dp = percdamp
        while 1 > dp > 0:
            try:
                damp = dp * torch.mean(torch.diag(Hs[b]))
                Ht = Hs[b].clone()
                Ht[range(cols), range(cols)] += damp
                Lc = torch.linalg.cholesky(Ht)
                Hinv_init[b] = torch.cholesky_inverse(Lc)
                Hinv[b] = torch.linalg.cholesky(Hinv_init[b], upper=True)
                if bool(torch.isnan(Hinv[b]).any()):
                    raise RuntimeError("nan")
                break
            except RuntimeError:
                dp += 0.0015
        if not (0 < dp < 1):
            raise ValueError("smorzamento fuori da (0,1)")
    Qrtn = Qrtn[:, :, perm]
    if enable_gradient_update and alpha > 0:
        eps = torch.finfo(torch.float32).tiny
        GH = torch.bmm(gsub, Hinv_init)
        qerr = Qrtn - Wsub
        dH = torch.diagonal(Hinv_init, dim1=1, dim2=2).unsqueeze(1)
        loss_gptq = ((qerr * GH + qerr ** 2 / 2) / dH).abs()
        c = (gsub * GH).sum(dim=2, keepdim=True) - (GH ** 2) / dH
        c = c.clamp(min=2 * alpha * loss_gptq).clamp(min=eps)
        beta = (1 - torch.sqrt(torch.clamp(1 - (2 * alpha * loss_gptq) / c, min=0.0))).mean(-1)
    else:
        beta = torch.zeros(ng, rps, dtype=F32, device=W.device)
    Z = torch.bmm(gsub, Hinv.transpose(1, 2)) * beta.unsqueeze(2)
    GHinv = torch.bmm(Z, Hinv)
    Q = torch.zeros_like(Wsub)
    D = torch.arange(blocksize - 1, -1, -1, device=W.device).to(GHinv)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = Wsub[:, :, i1:i2].clone()
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        Z1 = Z[:, :, i1:i2]
        GH1 = GHinv[:, :, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        for c_ in range(i2 - i1):
            j_orig = perm[i1 + c_:i1 + c_ + 1]
            w = W1[:, :, c_:c_ + 1]                                   # [ng, rps, 1]
            q = qcols(w, j_orig).squeeze(-1)                          # [ng, rps]
            ce = (w.squeeze(-1) - q - GH1[:, :, c_]) / Hinv1[:, c_, c_].unsqueeze(1)
            W1[:, :, c_] = q
            Err1[:, :, c_] = ce
            if c_ + 1 < i2 - i1:
                hrow = Hinv1[:, c_, c_ + 1:]                            # [ng, n_rest]
                W1[:, :, c_ + 1:] = W1[:, :, c_ + 1:] - GH1[:, :, c_ + 1:] - ce.unsqueeze(2) * hrow.unsqueeze(1)
                GH1[:, :, c_ + 1:] = GH1[:, :, c_ + 1:] - Z1[:, :, c_].unsqueeze(2) * hrow.unsqueeze(1)
        Q[:, :, i1:i2] = W1
        if i2 == cols:
            continue
        upd = (i2 - i1) * GHinv[:, :, i2:]
        upd = upd - torch.bmm(Z1 * D[: i2 - i1].reshape(1, 1, -1), Hinv[:, i1:i2, i2:])
        Wsub[:, :, i2:] = Wsub[:, :, i2:] - upd - torch.bmm(Err1, Hinv[:, i1:i2, i2:])
        GHinv[:, :, i2:] = GHinv[:, :, i2:] - torch.bmm(Z[:, :, i1:i2], Hinv[:, i1:i2, i2:])
    Q = Q[:, :, invperm].reshape(rows, cols)
    out = {"Q": Q, "beta": beta.reshape(rows)}
    if quant == "thor":
        Ttr = torch.zeros(K, rows, cols, dtype=torch.int8, device=W.device)
        for gi in range(ngr):
            sl = slice(gi * gsz, (gi + 1) * gsz)
            Ttr[:, :, sl] = codici_da_livelli(Q[:, sl], scale[:, :, gi], tab)
        out.update({"T": Ttr, "S": scale})
    else:
        out["scale"] = scale
    return out
