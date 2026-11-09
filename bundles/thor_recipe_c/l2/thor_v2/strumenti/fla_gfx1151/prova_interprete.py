#!/usr/bin/env python3
"""Collaudo SENZA GPU della ricetta gfx1151 (27/9/2026): kernel Triton nell'interprete (TRITON_INTERPRET=1, numpy su
CPU), dimensioni piccole. Controlla la LOGICA delle patch (prodotto spezzato con reshape/permute, blocchi da 16,
prepare_wy_repr_bwd in torch), non la compilazione per gfx1151 (quella la vede compila_offline.py).
  1. prepare_wy_repr_bwd_torch contro il kernel originale di fla (interpretato), stessi ingressi;
  2. fla intera (ricetta gfx1151) contro il ripiego di transformers in fp64, avanti e 5 gradienti.
Uso: env TRITON_INTERPRET=1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 python3 prova_interprete.py
"""
import os
import sys
from pathlib import Path

assert os.environ.get("TRITON_INTERPRET") == "1"
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402

sys.modules["fla"] = None
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402

RIPIEGO = M.torch_chunk_gated_delta_rule


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def main() -> int:
    B, T, H, D = 1, 128, 2, 128
    torch.manual_seed(0)
    fla_gdn = G.configura()                                  # sostituzioni comprese
    import fla.ops.gated_delta_rule.wy_fast as W
    # 1. prepare_wy_repr_bwd: torch contro kernel originale (interpretato)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D), dim=-1)
    v, dw, du = (torch.randn(B, T, H, D) for _ in range(3))
    beta = torch.sigmoid(torch.randn(B, T, H))
    g = torch.cumsum(-torch.rand(B, T, H), 1).reshape(B, T // 64, 64, H)
    g = (g - g[:, :, :1]).reshape(B, T, H) / 0.6931471805599453   # cumsum per blocco, in base 2 come fla
    A = (torch.tril(torch.randn(B, T // 64, H, 64, 64) * 0.1, -1) + torch.eye(64))     # [B, NC, H, i, j]
    A = A.permute(0, 1, 3, 2, 4).reshape(B, T, H, 64).contiguous()                       # [B, T, H, j]
    ref = W.prepare_wy_repr_bwd(k=k, v=v, beta=beta, A=A, dw=dw, du=du, g=g)
    mio = G.prepare_wy_repr_bwd_torch(k=k, v=v, beta=beta, A=A, dw=dw, du=du, g=g)
    for n, a, b in zip(("dk", "dv", "db", "dg"), mio, ref):
        print(f"prepare_wy {n}: torch~kernel rel {rel(a, b):.2e}", flush=True)
    # 2. fla intera contro ripiego fp64
    q, k, v, do = (torch.randn(B, T, H, D, dtype=torch.float64) for _ in range(4))
    gg = -torch.nn.functional.softplus(torch.randn(B, T, H, dtype=torch.float64))
    be = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float64))

    def passo(f, dt):
        xs = [t.to(dt).clone().requires_grad_(True) for t in (q, k, v, gg, be)]
        o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True)
        o.backward(do.to(dt))
        return [o.detach()] + [x.grad.detach() for x in xs]
    uf = passo(fla_gdn, torch.float32)
    u64 = passo(RIPIEGO, torch.float64)
    ut = passo(RIPIEGO, torch.float32)
    for n, a, b, c in zip(("avanti", "dq", "dk", "dv", "dg", "dbeta"), uf, ut, u64):
        print(f"{n:7s} fla~fp64 {rel(a, c):.2e} · ripiego32~fp64 {rel(b, c):.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
