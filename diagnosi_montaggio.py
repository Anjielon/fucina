"""Quale tensore del blocco rimontato è sbagliato? Confronto con il MAESTRO.
Per ogni tensore: correlazione fra la versione dal GGUF (dequantizzata,
riorientata) e quella bf16 del NAS. Un tensore montato male ha correlazione ~0."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py"); sys.path.insert(0, ".")
from gguf import GGUFReader
from gguf import quants as GQ
from safetensors import safe_open
src = Path("norm_tweak.py").read_text()
ns = {"__file__": str(Path("norm_tweak.py").resolve()), "__name__": "diag"}
exec(compile(src, "norm_tweak.py", "exec"), ns)
MAPPA, dequant_strato = ns["MAPPA"], ns["dequant_strato"]

from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration
from accelerate import init_empty_weights
NAS = Path("/mnt/nas/CACHEDEV1_DATA/modelli-fp/Ornith-1.5-397B")
cfg = AutoConfig.from_pretrained(str(NAS))
with init_empty_weights(): m = Qwen3_5MoeForConditionalGeneration(cfg)
lm = m.model.language_model
L = 0
forme = {n: tuple(p.shape) for n, p in lm.layers[L].named_parameters()}
forme.update({n: tuple(b.shape) for n, b in lm.layers[L].named_buffers()})

r = GGUFReader("/mnt/models/gguf/odino-v31/ODINO-397B-v31.gguf")
tens = {t.name: t for t in r.tensors}
sd = dequant_strato(tens, L, forme, device="cpu")

idx = json.load(open(NAS/"model.safetensors.index.json"))["weight_map"]
pref = f"model.language_model.layers.{L}."
print(f"{'tensore':38s} {'corr':>7s} {'scala mia/vera':>16s}")
for hf_nome, mio in sorted(sd.items()):
    k = pref + hf_nome
    if k not in idx: print(f"{hf_nome:38s}   (non nel NAS)"); continue
    with safe_open(str(NAS/idx[k]), "pt") as f:
        vero = f.get_slice(k)[:] if mio.ndim < 3 else f.get_slice(k)[0:2]
    v = vero.float().numpy().ravel()[:400000]
    mm = mio[:2].float().numpy().ravel()[:400000] if mio.ndim == 3 else mio.float().numpy().ravel()[:400000]
    n = min(len(v), len(mm))
    c = float(np.corrcoef(v[:n], mm[:n])[0,1]) if n > 10 else float("nan")
    rap = (np.abs(mm[:n]).mean() + 1e-12) / (np.abs(v[:n]).mean() + 1e-12)
    seg = "✅" if c > 0.9 else ("⛔" if c < 0.3 else "⚠️")
    print(f"{seg} {hf_nome:36s} {c:7.3f} {rap:16.3f}")
