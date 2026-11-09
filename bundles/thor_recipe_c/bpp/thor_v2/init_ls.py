#!/usr/bin/env python3
"""Inizializzazione dei codici ternari ai MINIMI QUADRATI (audit B3, 24/9).

Prima: `TernarioDiffBatch.__init__` partiva da s0 = media|W| per gruppo e
delta = 0.5 fisso — e, nella pipeline globale, W era la v20 ri-quantizzata
(errore sui pesi contro il teacher 0.65, peggio della v20 stessa, 0.48).
Ora: W = pesi ORIGINALI bf16 (il chiamante li passa), scala e soglia per
gruppo scelte con la STESSA griglia LS di `kernel-rotazione/tq1_0r_1024_codec
.scala_ls` (usata per gli esperti di Thor, identica al riferimento C), e i
parametri del quantizzatore differenziabile impostati in modo che il codice
DURO iniziale coincida esattamente con l'arrotondamento LS:

    q_LS = sign(w)·[|w| > t],  d = media(|w| sopra t),  t = m*·media|w|
    log_s = log d
    z0    = atanh(clip(w/d)) / beta_0                (monotona in |w|)
    delta = atanh(t/d) / beta_0   per gruppo  ⇒  |z0| > delta  ⟺  |w| > t

(t/d < 1 sempre, perche' d e' la media dei |w| SOPRA t.) Funziona su un
TernarioDiffBatch/Fuso gia' costruito, sovrascrivendone i parametri.
"""
from __future__ import annotations

import torch

GRIGLIA_LS = (0.55, 0.66, 0.76, 0.86, 1.00, 1.20)   # = tq1_0r_1024_codec.GRIGLIA_LS


def scala_soglia_ls(Wg: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Wg: (..., group) -> (d, t) di forma (..., 1): scala LS e soglia assoluta.
    Stesse regole di scala_ls del codec: a parita' di punteggio vince la prima m."""
    ab = Wg.abs().to(torch.float64)
    media = ab.mean(-1, keepdim=True)
    miglior_punteggio = torch.full_like(media, -1.0)
    miglior_d = torch.zeros_like(media)
    miglior_t = torch.zeros_like(media)
    for m in GRIGLIA_LS:
        soglia = m * media
        vivi = ab > soglia
        k = vivi.sum(-1, keepdim=True).to(torch.float64)
        somma = torch.where(vivi, ab, torch.zeros_like(ab)).sum(-1, keepdim=True)
        punteggio = torch.where(k > 0, somma * somma / k.clamp_min(1.0), torch.zeros_like(k))
        ok = punteggio > miglior_punteggio
        miglior_punteggio = torch.where(ok, punteggio, miglior_punteggio)
        miglior_d = torch.where(ok, torch.where(k > 0, somma / k.clamp_min(1.0), torch.zeros_like(k)), miglior_d)
        miglior_t = torch.where(ok, soglia, miglior_t)
    return miglior_d.to(torch.float32), miglior_t.to(torch.float32)


@torch.no_grad()
def reinit_ls_(ternari, W: torch.Tensor, blocco_esperti: int = 16, eps: float = 1e-8) -> dict:
    """Sovrascrive IN-PLACE log_s, z, delta_raw di `ternari` (TernarioDiffBatch[Fuso],
    costruito su un tensore della stessa forma di W) con l'init LS da W.
    W: (E, n_out, n_in), nello STESSO spazio in cui `ternari` quantizza (per
    gate/up: gia' RUOTATO). Lavora a blocchi di esperti per contenere la memoria.
    Ritorna diagnostica: frazione di codici duri != codice LS (atteso ~0)."""
    E, n_out, n_in = W.shape
    assert tuple(ternari.shape) == (E, n_out, n_in), f"forma {tuple(ternari.shape)} != {tuple(W.shape)}"
    g = ternari.group
    ng = n_in // g
    beta_0 = float(getattr(ternari, "beta_0", 1.0))
    clip = float(getattr(ternari, "clip", 0.95))
    delta_max = float(ternari.delta_max)
    disaccordi, totale, gruppi_morti, gruppi = 0, 0, 0, 0
    for e0 in range(0, E, blocco_esperti):
        e1 = min(E, e0 + blocco_esperti)
        Wg = W[e0:e1].detach().to(ternari.z.device, torch.float32).reshape(e1 - e0, n_out, ng, g)
        d, t = scala_soglia_ls(Wg)
        # ⛔ 24/9 17:3x — gruppi MORTI nell'ORIGINALE: allo strato 0 di Thor il
        # 36,4% delle righe gate+up ha |w|max < 1e-20 (valori ~1e-37, non zero
        # esatto). Scala ~1e-37 → clamp a eps → codice 0: e' la scelta giusta
        # (il peso e' nullo), ma va escluso dal confronto col codice LS.
        vivo = d > eps
        gruppi_morti += (~vivo).sum().item()
        gruppi += vivo.numel()
        d = d.clamp_min(eps)
        z0 = torch.atanh((Wg / d).clamp(-clip, clip)) / beta_0
        # soglia in unita' di z; resta sotto atanh(clip) e sotto delta_max
        delta = torch.atanh((t / d).clamp(0.0, clip * (1 - 1e-4))) / beta_0
        delta = delta.clamp(1e-6, delta_max * (1 - 1e-6))
        p = delta / delta_max
        ternari.log_s[e0:e1].copy_(d.log())
        ternari.z[e0:e1].copy_(z0)
        ternari.delta_raw[e0:e1].copy_(torch.log(p) - torch.log1p(-p))
        # verifica: codice duro dei nuovi parametri == codice LS
        delta_eff = ternari._delta_norm()[e0:e1]
        hard = torch.where(z0 > delta_eff, 1, torch.where(z0 < -delta_eff, -1, 0))
        q_ls = torch.where(Wg.abs() > t, torch.sign(Wg), torch.zeros_like(Wg)).to(hard.dtype)
        vivo_el = vivo.expand_as(hard)
        disaccordi += ((hard != q_ls) & vivo_el).sum().item()
        totale += vivo_el.sum().item()
    return {"frazione_disaccordo_ls": disaccordi / max(totale, 1),
            "frazione_gruppi_morti": gruppi_morti / max(gruppi, 1)}
