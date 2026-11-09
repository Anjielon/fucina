"""Invariante del ramo torch da_gguf (audit B9, 24/9/2026 18:40).

Il ramo "quantizzato" di torch deve caricare dal GGUF gli STESSI numeri che
llama.cpp usa. Il 24/9 `linear_attn.A_log` arrivava col segno invertito (err_rel
2,000, coseno -1,000) e 30 strati su 40 calcolavano un altro modello. Questo test
confronta peso per peso, dopo la mappa inversa GGUF->HF, contro il bf16 originale:
- ogni tensore salvato F32 nel GGUF deve combaciare (err_rel < 1e-3): e' una
  copia esatta a meno delle trasformazioni dichiarate (+1, -exp, permutazioni);
- ogni tensore, quantizzato o no, deve avere coseno (float64) > 0,8: una
  permutazione o un segno sbagliati danno ~0 o -1, la quantizzazione ternaria
  (la piu' aggressiva) sta sopra 0,85.
Uno strato lineare (0) e uno ad attenzione piena (3). Solo CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from gguf import GGUFReader  # noqa: E402
from safetensors import safe_open  # noqa: E402

from carica_strato_gguf import MAPPA, pesi_strato_da_gguf  # noqa: E402

SRC = Path("@@ROOT:models@@/gguf/thor-src-hf")
GGUF = "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf"


def _tipi_f32(L: int) -> set[str]:
    """Nomi HF dei tensori che il GGUF salva in F32 (copia esatta attesa)."""
    tipo = "full" if L in {3, 7, 11, 15, 19, 23, 27, 31, 35, 39} else "linear"
    tens = {t.name: t for t in GGUFReader(GGUF).tensors}
    return {v["hf"] for fam, v in MAPPA[tipo].items()
            if f"blk.{L}.{fam}" in tens and tens[f"blk.{L}.{fam}"].tensor_type.name == "F32"}


def verifica_strato(L: int) -> list[str]:
    idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    pref = f"model.language_model.layers.{L}."
    esatti = _tipi_f32(L)
    guasti = []
    for nome, t in pesi_strato_da_gguf(GGUF, L).items():
        with safe_open(str(SRC / idx[pref + nome]), "pt") as sf:
            h = sf.get_tensor(pref + nome).double()
        g = t.double()
        assert tuple(g.shape) == tuple(h.shape), f"{nome}: forma {tuple(g.shape)} contro {tuple(h.shape)}"
        rel = ((g - h).norm() / h.norm()).item()
        cos = (torch.dot(g.flatten(), h.flatten()) / (g.norm() * h.norm())).item()
        if nome in esatti and rel > 1e-3:
            guasti.append(f"strato {L} {nome}: F32 nel GGUF ma err_rel {rel:.4g} (atteso copia esatta)")
        if cos < 0.8:
            guasti.append(f"strato {L} {nome}: coseno {cos:.4f} < 0,8 (segno o permutazione?)")
    return guasti


if __name__ == "__main__":
    tutti = verifica_strato(0) + verifica_strato(3)
    for g in tutti:
        print("⛔", g)
    print("OK: ramo torch da_gguf fedele peso per peso" if not tutti else f"FALLITO: {len(tutti)} guasti")
    sys.exit(1 if tutti else 0)
