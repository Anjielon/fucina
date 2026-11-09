"""Prodotto raggruppato degli esperti come UN solo bmm batched (24/9 20:0x).

Perche': la radiografia della forgia (torch.profiler, 5 passi) ha mostrato che su
ROCm/gfx1151 `F.grouped_mm` esegue un CICLO di `mm` piccoli, uno per esperto:
4168 chiamate per passo, ~21% del tempo GPU. Qui i token (gia' ordinati per esperto,
come li prepara transformers) si impilano in un tensore (E, C, in) con C = token del
piu' carico, si fa UN bmm (E, C, in) @ (E, in, out) e si rileggono le righe utili.
Il riempimento spreca qualche FLOP (C >= media), ma un kernel grande su questa GPU
rende molto piu' di centinaia di piccoli.

Contratto identico a `transformers.integrations.moe._grouped_mm`:
    input (S, in) ordinato per esperto · weight (E, in, out) · offs (E,) somme cumulate
    -> (S, out). Differenziabile (index_put / bmm / indicizzazione).
Righe oltre offs[-1] (sentinelle expert-parallel): restituite a zero; qui non esistono.

Installazione: `installa()` sostituisce il nome nel modulo di transformers, che
`_grouped_linear` risolve a ogni chiamata. Verifica: `test_grouped_mm_bmm.py`.
"""
from __future__ import annotations

import torch


def grouped_mm_bmm(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    E = weight.shape[0]
    S = input.shape[0]
    offs64 = offs.to(torch.int64)
    conteggi = torch.diff(offs64, prepend=offs64.new_zeros(1))          # token per esperto
    utili = int(offs64[-1].item())                                     # righe coperte dagli esperti
    capienza = int(conteggi.max().item())                              # C: una sola sincronizzazione
    out = input.new_zeros(S, weight.shape[2])
    if utili == 0 or capienza == 0:
        return out
    esperto = torch.repeat_interleave(torch.arange(E, device=input.device), conteggi, output_size=utili)
    posizione = torch.arange(utili, device=input.device) - (offs64 - conteggi)[esperto]
    impilato = input.new_zeros(E, capienza, input.shape[1])
    impilato = impilato.index_put((esperto, posizione), input[:utili])
    prodotto = torch.bmm(impilato, weight.to(input.dtype))            # (E, C, out)
    righe = prodotto[esperto, posizione]                               # (utili, out)
    if utili == S:
        return righe
    return torch.cat([righe, out[utili:]], dim=0)


_ORIGINALE = None


def installa() -> None:
    """Sostituisce `_grouped_mm` di transformers con la versione a un solo bmm."""
    global _ORIGINALE
    import transformers.integrations.moe as moe
    if _ORIGINALE is None:
        _ORIGINALE = moe._grouped_mm
    moe._grouped_mm = grouped_mm_bmm


def disinstalla() -> None:
    import transformers.integrations.moe as moe
    if _ORIGINALE is not None:
        moe._grouped_mm = _ORIGINALE
