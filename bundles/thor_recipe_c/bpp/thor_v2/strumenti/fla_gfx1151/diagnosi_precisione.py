#!/usr/bin/env python3
"""Dove perde precisione fla fp32 su gfx1151? Errore di OGNI stadio dell'avanti contro lo stesso stadio in fp64 torch,
calcolato dagli ingressi fp32 dello stadio (errore proprio dello stadio, non ereditato) — 27/9/2026.
Stadi: l2norm, cumsum (base 2), A = (I + tril(beta K K^T Gamma))^-1, w/u, h/v_new, o. Lanciare con lancia_prova.sh."""
import math
import sys
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402

LN2 = math.log(2)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def blocchi(x, BT=64):              # [B, T, H, ...] -> [B, H, NC, BT, ...] fp64
    B, T, H = x.shape[:3]
    return x.double().reshape(B, T // BT, BT, H, *x.shape[3:]).permute(0, 3, 1, 2, *range(4, x.dim() + 1))


def _ripiego():
    """Il ripiego PyTorch di transformers (la funzione originale sotto il decoratore)."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as MM
    f = MM.torch_chunk_gated_delta_rule
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    assert f.__code__.co_filename.endswith("modeling_qwen3_5_moe.py") and "l2norm" in f.__code__.co_names, f
    return f


def main() -> int:
    G.configura()
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    B, T, H, D = 2, 4096, 32, 128
    for forza in (1.0, 8.0):
        gen = torch.Generator().manual_seed(1)
        q, k, v = (torch.randn(B, T, H, D, generator=gen).cuda() for _ in range(3))
        g = (-forza * torch.nn.functional.softplus(torch.randn(B, T, H, generator=gen))).cuda()
        beta = torch.sigmoid(torch.randn(B, T, H, generator=gen)).cuda()
        print(f"--- forza_g {forza}", flush=True)
        qn, _ = l2norm_fwd(q)
        kn, _ = l2norm_fwd(k)
        q64 = q.double() / (q.double().pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        k64 = k.double() / (k.double().pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        print(f"  l2norm q {rel(qn, q64):.2e} k {rel(kn, k64):.2e}", flush=True)
        g2 = chunk_local_cumsum(g, chunk_size=64, scale=1 / LN2)
        g2_64 = blocchi(g).cumsum(-1) / LN2                             # [B, H, NC, BT]
        g2c = blocchi(g2)
        print(f"  cumsum base 2 {rel(g2c, g2_64):.2e} · differenze entro blocco "
              f"{rel(g2c[..., :, None] - g2c[..., None, :], g2_64[..., :, None] - g2_64[..., None, :]):.2e}", flush=True)
        w, u, A = chunk_gated_delta_rule_fwd_intra(k=kn, v=v, g=g2, beta=beta, chunk_size=64)
        # riferimento fp64 dagli ingressi fp32 dello stadio (kn, g2, beta, v)
        Kc, Vc, bc, gc = blocchi(kn), blocchi(v), blocchi(beta), blocchi(g2)
        Gam = torch.exp2(gc[..., :, None] - gc[..., None, :])
        strict = torch.tril(torch.ones(64, 64, dtype=torch.bool, device="cuda"), -1)
        M = torch.where(strict, bc[..., :, None] * (Kc @ Kc.transpose(-1, -2)) * Gam, 0.0)
        I = torch.eye(64, dtype=torch.float64, device="cuda")
        A64 = torch.linalg.solve_triangular(I + M, I.expand_as(M), upper=False)
        Ac = blocchi(A)
        print(f"  A (kkt+inversa) {rel(Ac, A64):.2e}", flush=True)
        w64 = Ac @ (Kc * (bc * torch.exp2(gc))[..., None])
        u64 = Ac @ (Vc * bc[..., None])
        print(f"  w {rel(blocchi(w), w64):.2e} u {rel(blocchi(u), u64):.2e}", flush=True)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=kn, w=w, u=u, g=g2, initial_state=None, output_final_state=False,
                                                   chunk_size=64)
        Wc, Uc = blocchi(w), blocchi(u)
        S = torch.zeros(B, H, D, D, dtype=torch.float64, device="cuda")
        hs, vn = [], []
        NC = T // 64
        for c in range(NC):
            hs.append(S)
            vnc = Uc[:, :, c] - Wc[:, :, c] @ S
            vn.append(vnc)
            gl = gc[:, :, c, -1]
            S = S * torch.exp2(gl)[..., None, None] + Kc[:, :, c].transpose(-1, -2) @ (
                vnc * torch.exp2(gl[..., None] - gc[:, :, c])[..., None])
        h64 = torch.stack(hs, 1).permute(0, 1, 2, 3, 4)               # [B, NC, H, K, V]
        vn64 = torch.stack(vn, 2)                                      # [B, H, NC, BT, V]
        print(f"  h {rel(h, h64):.2e} v_new {rel(blocchi(v_new), vn64):.2e}", flush=True)
        o = chunk_fwd_o(q=qn, k=kn, v=v_new, h=h, g=g2, scale=D ** -0.5, chunk_size=64)
        Qc, VNc = blocchi(qn), blocchi(v_new)
        hc = h.double().permute(0, 2, 1, 3, 4)                         # [B, H, NC, K, V]
        incl = torch.tril(torch.ones(64, 64, dtype=torch.bool, device="cuda"))
        o64 = (D ** -0.5) * ((Qc * torch.exp2(gc)[..., None]) @ hc
                             + torch.where(incl, (Qc @ Kc.transpose(-1, -2)) * Gam, 0.0) @ VNc)
        print(f"  o {rel(blocchi(o), o64):.2e}", flush=True)
        # catena INTERA in fp64 dagli ingressi originali, contro il ripiego fp64 e contro fla fp32 end-to-end
        del M, A64, Gam, hs, vn, h64, vn64, w64, u64
        torch.cuda.empty_cache()
        Qc, Kc, Vc, bc = blocchi(q64), blocchi(k64), blocchi(v), blocchi(beta)
        gcn = blocchi(g).cumsum(-1)                                     # base e
        Gam = torch.exp(gcn[..., :, None] - gcn[..., None, :])
        M = torch.where(strict, bc[..., :, None] * (Kc @ Kc.transpose(-1, -2)) * Gam, 0.0)
        Ac = torch.linalg.solve_triangular(I + M, I.expand_as(M), upper=False)
        Wc = Ac @ (Kc * (bc * torch.exp(gcn))[..., None])
        Uc = Ac @ (Vc * bc[..., None])
        S = torch.zeros(B, H, D, D, dtype=torch.float64, device="cuda")
        oo = []
        for c in range(NC):
            vnc = Uc[:, :, c] - Wc[:, :, c] @ S
            oo.append((D ** -0.5) * ((Qc[:, :, c] * torch.exp(gcn[:, :, c])[..., None]) @ S
                                     + torch.where(incl, (Qc[:, :, c] @ Kc[:, :, c].transpose(-1, -2))
                                                   * Gam[:, :, c], 0.0) @ vnc))
            gl = gcn[:, :, c, -1]
            S = S * torch.exp(gl)[..., None, None] + Kc[:, :, c].transpose(-1, -2) @ (
                vnc * torch.exp(gl[..., None] - gcn[:, :, c])[..., None])
        catena = torch.stack(oo, 2)                                     # [B, H, NC, BT, V]
        del M, Ac, Wc, Uc, Gam
        torch.cuda.empty_cache()
        with torch.no_grad():
            r64, _ = _ripiego()(q.double(), k.double(), v.double(), g=g.double(), beta=beta.double(),
                                use_qk_l2norm_in_kernel=True)
            r32, _ = _ripiego()(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        with torch.no_grad():
            of, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        print(f"  catena fp64 ~ ripiego fp64 {rel(blocchi(r64), catena):.2e} · fla32 ~ catena {rel(blocchi(of), catena):.2e}"
              f" · ripiego32 ~ catena {rel(blocchi(r32), catena):.2e}", flush=True)
        del q, k, v, catena
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
