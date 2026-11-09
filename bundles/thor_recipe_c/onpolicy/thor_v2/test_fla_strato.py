"""Collaudo di fla (kernel Triton della regola delta) contro il ripiego PyTorch di transformers,
sullo STRATO VERO di Thor (24/9 22:2x). Perche': le finestre con due strati ad attenzione lineare
costano 1,89 s/passo contro 1,24 di quelle con uno ad attenzione piena; fla non si caricava
(mancava einops) e transformers usava in silenzio il ripiego a pezzetti.

Si lancia CON PYTHONPATH=thor_v2/strumenti/pylib_fla (cosi' transformers risolve fla).
Confronto su strato 0 (lineare), bf16, ingresso reale (embedding di token veri), 8 x 1024 token:
uscita dello strato e gradiente rispetto all'ingresso, errore relativo in norma, e tempi
avanti+indietro (media di 5 dopo 2 di riscaldamento). Criterio: errore <= quello del bf16
(ordine 1e-2) e nessun NaN."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M  # noqa: E402

fn_fla = M.torch_chunk_gated_delta_rule
fn_torch = getattr(fn_fla, "__wrapped__", None)
impl = [c.cell_contents for c in (fn_fla.__closure__ or ()) if callable(c.cell_contents)]
print("risolto da transformers:", [getattr(x, "__module__", "?") for x in impl])
assert any("fla" in getattr(x, "__module__", "") for x in impl), "fla non attiva: lanciare con PYTHONPATH=.../pylib_fla"
assert fn_torch is not None, "manca __wrapped__ del ripiego"

import valida_finestra_8_12 as V  # noqa: E402

cfg, emb, puliti, _q = V.carica_finestra_corrente([0])
strato = puliti[0]
for p in strato.parameters():
    p.requires_grad_(False)
ids, _ = V.carica_batch_calibrazione()
x_ids = ids[0].to("cuda")                                    # (8, 1024)
h0 = torch.nn.functional.embedding(x_ids, emb).to(torch.bfloat16)
from transformers.masking_utils import create_causal_mask  # noqa: E402
rot = M.Qwen3_5MoeTextRotaryEmbedding(cfg).to("cuda")
pos = torch.arange(h0.shape[1], device="cuda").unsqueeze(0).expand(h0.shape[0], -1)
pe = rot(h0, pos)
mask = create_causal_mask(config=cfg, inputs_embeds=h0, attention_mask=None, past_key_values=None, position_ids=pos)
g_out = torch.randn_like(h0)


def corsa(fn):
    M.torch_chunk_gated_delta_rule = fn
    h = h0.clone().requires_grad_(True)
    o = strato(hidden_states=h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
    o = o[0] if isinstance(o, tuple) else o
    (o.float() * g_out.float()).sum().backward()
    return o.detach().float(), h.grad.float()


def tempo(fn, n=5):
    for _ in range(2):
        corsa(fn)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        corsa(fn)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


o_t, g_t = corsa(fn_torch)
o_f, g_f = corsa(fn_fla)
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"uscita strato: fla vs torch errore relativo {rel(o_f, o_t):.3e} · NaN fla {torch.isnan(o_f).any().item()}")
print(f"gradiente ingresso: fla vs torch errore relativo {rel(g_f, g_t):.3e} · NaN fla {torch.isnan(g_f).any().item()}")
tt, tf = tempo(fn_torch), tempo(fn_fla)
print(f"tempo avanti+indietro strato 0 (8x1024): torch {tt*1e3:.1f} ms · fla {tf*1e3:.1f} ms · {tt/tf:.2f}x")
M.torch_chunk_gated_delta_rule = fn_fla

# --- riferimento fp32 (ripiego PyTorch in float32): chi dei due bf16 ci sta piu' vicino? ---
strato32 = strato.float()
h032, pe32, go32 = h0.float(), tuple(t.float() for t in pe), g_out.float()


def corsa32():
    M.torch_chunk_gated_delta_rule = fn_torch
    h = h032.clone().requires_grad_(True)
    o = strato32(hidden_states=h, position_embeddings=pe32, attention_mask=mask, position_ids=pos)
    o = o[0] if isinstance(o, tuple) else o
    (o * go32).sum().backward()
    return o.detach(), h.grad


o_r, g_r = corsa32()
print(f"contro fp32 — uscita: torch-bf16 {rel(o_t, o_r):.3e} · fla-bf16 {rel(o_f, o_r):.3e}")
print(f"contro fp32 — gradiente: torch-bf16 {rel(g_t, g_r):.3e} · fla-bf16 {rel(g_f, g_r):.3e}")
M.torch_chunk_gated_delta_rule = fn_fla
