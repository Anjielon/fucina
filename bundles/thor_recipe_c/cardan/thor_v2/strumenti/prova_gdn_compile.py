#!/usr/bin/env python3
"""GDN di Thor-T: torch.compile del ripiego PyTorch di transformers (torch_chunk_gated_delta_rule) contro lo stesso
ripiego non compilato, fp32, dimensioni vere (2 x 4096, 32 teste, dk = dv = 128). Stessa matematica attesa (le stesse
operazioni, fuse): si misura l'errore relativo verso fp64 in avanti e nei 5 gradienti, e il tempo avanti+indietro.
Lanciare come unita' systemd --user. fla NON e' importabile nel venv (einops assente): qui si prova solo torch."""
import sys
import time

import torch
torch.set_num_threads(4)

sys.modules["fla"] = None
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402

base = M.torch_chunk_gated_delta_rule
MODO = sys.argv[1] if len(sys.argv) > 1 else "default"
comp = torch.compile(base, mode=None if MODO == "default" else MODO, dynamic=False)


def ingressi(seed, B=2, T=4096, H=32, D=128, dt=torch.float32):
    g = torch.Generator().manual_seed(seed)
    t = [torch.randn(B, T, H, D, generator=g) for _ in range(3)]
    gg = -torch.nn.functional.softplus(torch.randn(B, T, H, generator=g))
    be = torch.sigmoid(torch.randn(B, T, H, generator=g))
    do = torch.randn(B, T, H, D, generator=g)
    return [x.to("cuda", dt) for x in (*t, gg, be, do)]


def corri(f, q, k, v, gg, be, do, ripeti=1):
    tt = []
    sinc = torch.cuda.synchronize if q.is_cuda else (lambda: None)
    for _ in range(ripeti):
        xs = [x.clone().requires_grad_(True) for x in (q, k, v, gg, be)]
        sinc(); t0 = time.time()
        o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True)
        o.backward(do)
        sinc(); tt.append(time.time() - t0)
    return o.detach().double(), [x.grad.detach().double() for x in xs], min(tt)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def main() -> int:
    for seed in (1, 2):
        d32 = ingressi(seed)
        d64 = [x.double().cpu() for x in d32]          # riferimento fp64 su CPU: l'fp64 su RDNA e' lentissimo
        o64, g64, _ = corri(base, *d64)
        o64, g64 = o64.cuda(), [x.cuda() for x in g64]
        ob, gb, tb = corri(base, *d32, ripeti=2)
        t0 = time.time()
        corri(comp, *d32)
        tc0 = time.time() - t0
        oc, gc, tc = corri(comp, *d32, ripeti=3)
        print(f"seme {seed} ({MODO}): tempo avanti+indietro torch {tb:.3f}s, compilato {tc:.3f}s (x{tb / tc:.2f}); "
              f"prima compilazione {tc0:.0f}s", flush=True)
        print(f"  vs fp64 avanti: torch {rel(ob, o64):.2e} compilato {rel(oc, o64):.2e}", flush=True)
        for n, a, b, r in zip(("dq", "dk", "dv", "dg", "dbeta"), gb, gc, g64):
            print(f"  vs fp64 {n:6s}: torch {rel(a, r):.2e} compilato {rel(b, r):.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
