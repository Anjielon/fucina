#!/usr/bin/env python3
"""Test del percorso HELD-OUT (audit B6) di ottimizza_finestra_JOINT, CPU,
strati Qwen3.5-MoE veri a scala giocattolo: 3 batch, 2 di addestramento + 1
held-out. Verifica: (1) la diagnostica held-out esiste con v20/init/fine
finiti; (2) la batch held-out non viene mai usata per un passo (la loss di
addestramento registrata ha solo le batch 0-1: controllato col conteggio dei
passi per batch tramite un avvolgitore di passo_ottimizzazione)."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import valida_finestra_8_12 as v  # noqa: E402


def main() -> bool:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    torch.manual_seed(0)
    n = 2
    cfg = v._costruisci_config_giocattolo(n)
    cfg._attn_implementation = "sdpa"
    puliti, quant = {}, {}
    for L in range(n):
        p = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        v._init_pesi_giocattolo(p, seed=100 + L)
        q = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        with torch.no_grad():
            g = torch.Generator().manual_seed(900 + L)
            for (_, a), (_, b) in zip(p.named_parameters(), q.named_parameters()):
                b.copy_(a + torch.randn(a.shape, generator=g) * 0.15)
        puliti[L], quant[L] = p.eval(), q.eval()
    emb = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
    ids = torch.randint(0, cfg.vocab_size, (3, 2, 6))

    viste = []
    originale = v.passo_ottimizzazione

    def spia(tf, qf, h0, opt, e0=None, **kw):
        viste.append(h0.data_ptr())
        return originale(tf, qf, h0, opt, e0, **kw)

    v.passo_ottimizzazione = spia
    try:
        diag = []
        v.ottimizza_finestra_JOINT(quant, ids, cfg, n_passi=20, strati_teacher=puliti, emb=emb,
                                   raccogli_diagnostica=diag, n_batch_addestramento=2)
    finally:
        v.passo_ottimizzazione = originale

    ho = diag[0].get("heldout")
    ok1 = ho is not None and all(math.isfinite(ho[k]) for k in ("v20", "init", "fine"))
    print(f"  (1) diagnostica held-out {ho}: {'✅' if ok1 else '⛔'}")
    ok2 = len(set(viste)) == 2   # solo i tensori h0 delle 2 batch di addestramento
    print(f"  (2) batch distinte usate per i passi: {len(set(viste))} (attese 2): {'✅' if ok2 else '⛔'}")
    esito = ok1 and ok2
    print("✅ PERCORSO HELD-OUT CORRETTO" if esito else "⛔ PERCORSO HELD-OUT DIFETTOSO")
    return esito


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
