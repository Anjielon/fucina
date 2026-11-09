#!/usr/bin/env python3
"""Invariante che avrebbe intercettato il bug del confine (24/9 15:1x,
PIANO.md): il confine calcolato A CATENA (finestra [0,1], poi [2,3] col seme
della prima) deve coincidere col confine calcolato IN UN COLPO su [0,1,2,3].
E il vecchio comportamento (niente seme) deve FALLIRE lo stesso confronto —
altrimenti il test non distinguerebbe il caso rotto da quello giusto.

Strati VERI (Qwen3_5MoeDecoderLayer) a dimensioni giocattolo, CPU, zero Thor."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from valida_finestra_8_12 import (  # noqa: E402
    _costruisci_config_giocattolo, _init_pesi_giocattolo, calcola_confine_finestra)


def main() -> bool:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    torch.manual_seed(0)
    n = 4
    cfg = _costruisci_config_giocattolo(n)
    cfg._attn_implementation = "sdpa"
    puliti, quant = {}, {}
    for L in range(n):
        p = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        _init_pesi_giocattolo(p, seed=100 + L)
        q = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        with torch.no_grad():
            g = torch.Generator().manual_seed(900 + L)
            for (_, a), (_, b) in zip(p.named_parameters(), q.named_parameters()):
                b.copy_(a + torch.randn(a.shape, generator=g) * 0.15)
        puliti[L], quant[L] = p.eval(), q.eval()

    # scala dell'embedding come nel modello vero (residuo che cresce coi layer)
    emb = torch.randn(cfg.vocab_size, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (2, 2, 6))

    sub = lambda d, ks: {k: d[k] for k in ks}  # noqa: E731

    h01, e01 = calcola_confine_finestra(sub(puliti, [0, 1]), sub(quant, [0, 1]), ids, cfg, emb)
    h_cat, e_cat = calcola_confine_finestra(sub(puliti, [2, 3]), sub(quant, [2, 3]), ids, cfg, emb,
                                            h0_ingresso_per_b=h01, e0_ingresso_per_b=e01)
    h_dir, e_dir = calcola_confine_finestra(puliti, quant, ids, cfg, emb)
    h_old, e_old = calcola_confine_finestra(sub(puliti, [2, 3]), sub(quant, [2, 3]), ids, cfg, emb)

    ok = True
    for b in h_dir:
        rel_t = ((h_cat[b] - h_dir[b]).norm() / h_dir[b].norm()).item()
        hq_cat, hq_dir = h_cat[b] + e_cat[b], h_dir[b] + e_dir[b]
        rel_q = ((hq_cat - hq_dir).norm() / hq_dir.norm()).item()
        rel_old = ((h_old[b] - h_dir[b]).norm() / h_dir[b].norm()).item()
        print(f"batch {b}: catena vs diretto  teacher={rel_t:.2e}  quant={rel_q:.2e}   "
              f"| vecchio (senza seme) vs diretto = {rel_old:.2e}")
        ok &= rel_t < 1e-6 and rel_q < 1e-5
        # il test deve saper riconoscere il caso rotto: a scala giocattolo
        # (pesi 0.02) gli strati spostano poco il residuo, quindi il vecchio
        # codice sbaglia di ~1e-3 — ordini di grandezza sopra il rumore del
        # percorso giusto (~1e-8). Su Thor vero lo scarto misurato e' 8.2.
        ok &= rel_old > 1e-4

    print("✅ INVARIANTE CONFINE A CATENA: PASSA (e sa fallire sul vecchio codice)" if ok
          else "⛔ INVARIANTE CONFINE A CATENA: FALLISCE")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
