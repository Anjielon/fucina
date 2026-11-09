#!/usr/bin/env python3
"""fla.chunk_gated_delta_rule su gfx1151: dove si rompe? Dimensioni crescenti, bf16 e fp32, solo avanti poi avanti+indietro.
Ambiente SEPARATO (PYTHONPATH con einops in strumenti/ambiente_prova_fla): il venv di EB2 resta senza einops."""
import sys, time, torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f
casi = [(1, 256, 4, 64), (1, 1024, 8, 128), (2, 4096, 32, 128)]
dtipi = [torch.bfloat16, torch.float32] if len(sys.argv) < 2 else [getattr(torch, sys.argv[1])]
for dt in dtipi:
    for (B, T, H, D) in casi:
        g = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(B, T, H, D, generator=g).to("cuda", dt).requires_grad_(True) for _ in range(3))
        gg = (-torch.nn.functional.softplus(torch.randn(B, T, H, generator=g))).to("cuda", dt).requires_grad_(True)
        be = torch.sigmoid(torch.randn(B, T, H, generator=g)).to("cuda", dt).requires_grad_(True)
        print(f"{dt} {(B, T, H, D)} avanti...", flush=True)
        o, _ = f(q, k, v, g=gg, beta=be, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize(); print("  avanti ok", tuple(o.shape), o.dtype, flush=True)
        o.float().sum().backward(); torch.cuda.synchronize(); print("  indietro ok", flush=True)
