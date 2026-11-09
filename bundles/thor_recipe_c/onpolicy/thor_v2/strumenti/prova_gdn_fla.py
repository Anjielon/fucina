#!/usr/bin/env python3
"""GDN di Thor-T: kernel Triton di flash-linear-attention (fla.chunk_gated_delta_rule) contro il ripiego PyTorch di
transformers (torch_chunk_gated_delta_rule), in fp32 come la ricetta B dello studente (26/9/2026 sera).
Prima di cambiare kernel sotto un allenamento: stessa matematica? (avanti e gradienti di q, k, v, g, beta, errore relativo)
e quanto piu' veloce? Dimensioni vere (config di Thor): 32 teste valore, dk = dv = 128, micro-lotto 2 x 4096, modo riga.
Lanciare come unita' systemd --user (la shell ha HSA_OVERRIDE_GFX_VERSION sbagliato)."""
import sys
import time

import torch

sys.modules["fla"] = None                     # 1) il ripiego PyTorch: transformers lo sceglie se fla non si importa
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
torch_gdn = M.torch_chunk_gated_delta_rule
del sys.modules["fla"]
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdn  # noqa: E402  2) il kernel Triton


def ingressi(seed: int, B=2, T=4096, H=32, D=128, dev="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, T, H, D, generator=g)
    k = torch.randn(B, T, H, D, generator=g)
    v = torch.randn(B, T, H, D, generator=g)
    gg = -torch.nn.functional.softplus(torch.randn(B, T, H, generator=g))     # log-decadimento <= 0
    beta = torch.sigmoid(torch.randn(B, T, H, generator=g))
    do = torch.randn(B, T, H, D, generator=g)
    return [t.to(dev, torch.float32) for t in (q, k, v, gg, beta, do)]


def corri(f, q, k, v, gg, beta, do):
    xs = [t.clone().requires_grad_(True) for t in (q, k, v, gg, beta)]
    torch.cuda.synchronize()
    t0 = time.time()
    o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    t1 = time.time()
    o.backward(do)
    torch.cuda.synchronize()
    t2 = time.time()
    return o.detach().float(), [x.grad.detach().float() for x in xs], t1 - t0, t2 - t1


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30)), float(((a - b).abs().max()) / b.abs().max().clamp_min(1e-30))


def main() -> int:
    for seed in (1, 2):
        dati = ingressi(seed)
        corri(fla_gdn, *dati)                   # compilazione Triton fuori dai tempi
        ot, gt, at, it = corri(torch_gdn, *dati)
        of, gf, af, ifl = corri(fla_gdn, *dati)
        print(f"seme {seed}: dtype uscita fla {of.dtype}")
        print(f"  avanti   rel {rel(of, ot)[0]:.2e} (max {rel(of, ot)[1]:.2e})   tempo torch {at:.3f}s fla {af:.3f}s")
        for n, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), gf, gt):
            r, m = rel(a, b)
            print(f"  {n:6s} rel {r:.2e} (max {m:.2e})")
        print(f"  indietro tempo torch {it:.3f}s fla {ifl:.3f}s  -> totale x{(at + it) / (af + ifl):.1f}")
        # riferimento di rumore: torch in fp64 contro torch fp32
        d64 = [t.double() for t in dati]
        o64, g64, _, _ = corri(torch_gdn, *d64)
        print(f"  rumore fp32 (torch fp32 vs fp64): avanti {rel(ot, o64.float())[0]:.2e} · fla vs fp64 "
              f"{rel(of, o64.float())[0]:.2e} · dq torch {rel(gt[0], g64[0].float())[0]:.2e} fla {rel(gf[0], g64[0].float())[0]:.2e}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
