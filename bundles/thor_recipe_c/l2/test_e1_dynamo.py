#!/usr/bin/env python3
"""test_e1_dynamo.py — R012 (5/10): il profilo di E1 si fermava su `torch._dynamo hit config.recompile_limit (8)` in
gdn_veloce_corsa_v5._prep. Causa: nel modo segmenti (`Studente._miscela_segmenti`) il GDN e' chiamato PROMPT PER
PROMPT, e `_prep` e' compilata con dynamic=False: ogni N = ceil(len/64) nuovo = una compilazione nuova. La lunghezza
del lotto non c'entra (un lotto a lunghezza fissa ha comunque prompt di lunghezze diverse dentro).

Prova su CPU (inductor CPU, nessuna GPU): 12 lunghezze di segmento (N = 1..12, come i prompt di items_teacher, max
~700 token) attraverso la funzione VERA della corsa (chunk_gated_delta_rule_veloce, versione 5, preparazione
compilata) con il limite che mette e1_recupero_lettera (--recompile-limit 64):
  1. nessun avviso «recompile_limit» (cioe' ogni forma resta compilata, niente ripiego silenzioso in eager)
  2. uscita e gradienti == versione 3 (stessa matematica senza torch.compile), a meno dell'arrotondamento
  3. controprova: col limite di default (8) l'avviso compare davvero (la prova sa vedere il difetto)
"""
from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI / "recupero_7dabd6f" / "thor_v2"))
from strumenti import gdn_veloce_corsa_v5 as G   # noqa: E402

H, DK, DV = 2, 32, 32


def _dati(T: int, seme: int):
    g = torch.Generator().manual_seed(seme)
    q = torch.randn(1, T, H, DK, generator=g)
    k = torch.randn(1, T, H, DK, generator=g)
    v = torch.randn(1, T, H, DV, generator=g)
    gg = -torch.rand(1, T, H, generator=g)
    beta = torch.rand(1, T, H, generator=g)
    return [x.requires_grad_(True) for x in (q, k, v, gg, beta)]


def _giro(lunghezze, versione: int):
    out = []
    for i, T in enumerate(lunghezze):
        x = _dati(T, 1000 + i)
        o, _ = G.chunk_gated_delta_rule_veloce(*x, chunk_size=64, use_qk_l2norm_in_kernel=True, _versione=versione)
        o.pow(2).mean().backward()
        out.append((o.detach(), [t.grad.detach() for t in x]))
    return out


def _cattura_avvisi(fn):
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    lg = logging.getLogger("torch._dynamo")
    lg.addHandler(h)
    try:
        r = fn()
    finally:
        lg.removeHandler(h)
    return r, buf.getvalue()


def main() -> int:
    import torch._dynamo
    lunghezze = [64 * n - 7 for n in range(1, 13)]        # N = 1..12, come i prompt veri
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 64
    G._PREP_COMPILATA = None
    ris5, avvisi = _cattura_avvisi(lambda: _giro(lunghezze, 5))
    assert "recompile_limit" not in avvisi, avvisi[-500:]
    assert not G._COMPILAZIONE_FALLITA, "torch.compile della preparazione fallito: la prova non vale"
    ris3 = _giro(lunghezze, 3)
    peggio = 0.0
    for (o5, g5), (o3, g3) in zip(ris5, ris3):
        peggio = max(peggio, float((o5 - o3).abs().max()), *[float((a - b).abs().max()) for a, b in zip(g5, g3)])
    assert peggio < 1e-4, peggio
    print(f"ok limite 64: 12 lunghezze di segmento, nessun avviso recompile_limit; v5 == v3 (scarto max {peggio:.2e})")

    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 8
    G._PREP_COMPILATA = None
    _, avvisi8 = _cattura_avvisi(lambda: _giro(lunghezze, 5))
    assert "recompile_limit" in avvisi8, "controprova: col limite 8 l'avviso doveva comparire"
    print("ok controprova: col limite 8 l'avviso recompile_limit compare (difetto di R012 riprodotto)")
    print("TUTTE LE 2 PROVE OK")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")
    sys.exit(main())
