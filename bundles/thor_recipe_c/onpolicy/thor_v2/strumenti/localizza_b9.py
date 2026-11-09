"""B9 — quale tensore carica male il ramo torch da_gguf? Confronto peso per peso
(GGUF dequantizzato + mappa inversa verso HF) contro il bf16 originale."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carica_strato_gguf import pesi_strato_da_gguf
from safetensors import safe_open
SRC = Path("@@ROOT:models@@/gguf/thor-src-hf")
GGUF = "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf"
idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
for L in [int(x) for x in sys.argv[1:]] or [0, 3]:
    q = pesi_strato_da_gguf(GGUF, L)
    pref = f"model.language_model.layers.{L}."
    print(f"--- strato {L}: {len(q)} tensori dal GGUF")
    for nome, t in sorted(q.items()):
        k = pref + nome
        if k not in idx:
            print(f"  {nome:45s} NON IN HF"); continue
        with safe_open(str(SRC / idx[k]), "pt") as sf:
            h = sf.get_tensor(k).float()
        if tuple(h.shape) != tuple(t.shape):
            print(f"  {nome:45s} FORMA diversa gguf {tuple(t.shape)} hf {tuple(h.shape)}"); continue
        rel = ((t - h).norm() / h.norm()).item()
        cos = torch.nn.functional.cosine_similarity(t.flatten(), h.flatten(), dim=0).item()
        flag = "  <<<<" if rel > 0.3 else ""
        print(f"  {nome:45s} {str(tuple(h.shape)):22s} err_rel {rel:8.4f} cos {cos:7.4f}{flag}")
