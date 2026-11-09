#!/usr/bin/env python3
"""Da dove viene l'errore 1-2e-6 di fla (ricetta gfx1151) contro 1e-7 del ripiego fp32? Ablazione (27/9/2026).
Varianti, ognuna contro il ripiego in fp64, avanti + 5 gradienti:
  base   ricetta gfx1151 com'e'
  L      l2norm di q/k in torch (la stessa formula del ripiego, con autograd) invece dei kernel l2norm di fla
  C      cumsum del gate per blocco in torch fp64 (poi fp32) invece di chunk_local_cumsum di fla (avanti e indietro)
  LC     entrambe
Lanciare con lancia_prova.sh (unita' systemd --user)."""
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402
import prova_equivalenza as P  # noqa: E402  (blocca fla per transformers e lega il ripiego)


def cumsum_torch(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False,
                 output_dtype=torch.float, chunk_indices=None, **kw):
    assert cu_seqlens is None and not head_first and g.dim() == 3
    B, T, H = g.shape
    x = g.double().reshape(B, T // chunk_size, chunk_size, H)
    y = x.flip(2).cumsum(2).flip(2) if reverse else x.cumsum(2)
    if scale is not None:
        y = y * scale
    return y.reshape(B, T, H).to(output_dtype or g.dtype)


def main() -> int:
    fla_gdn = G.configura()
    import fla.ops.gated_delta_rule.chunk as C
    originale = C.chunk_local_cumsum
    M = P.M

    def con_l2_torch(q, k, v, g=None, beta=None, use_qk_l2norm_in_kernel=True, **kw):
        return fla_gdn(M.l2norm(q, dim=-1, eps=1e-6), M.l2norm(k, dim=-1, eps=1e-6), v, g=g, beta=beta,
                       use_qk_l2norm_in_kernel=False, **kw)

    varianti = {"base": (fla_gdn, False), "L": (con_l2_torch, False), "C": (fla_gdn, True), "LC": (con_l2_torch, True)}
    for seme, forza in ((1, 1.0), (2, 8.0)):
        dati = P.ingressi(seme, 2, 4096, 32, 128, forza)
        u64, _, _ = P.passo(P.RIPIEGO, [t.double() for t in dati])
        ut, _, _ = P.passo(P.RIPIEGO, dati)
        rif = [P.rel(y, z) for y, z in zip(ut, u64)]
        print(f"seme {seme} forza {forza} · ripiego fp32~fp64: " + " ".join(f"{n} {r:.2e}" for n, r in zip(P.NOMI, rif)),
              flush=True)
        for nome, (f, cum) in varianti.items():
            C.chunk_local_cumsum = cumsum_torch if cum else originale
            t0 = time.perf_counter()
            uf, _, _ = P.passo(f, dati)
            rap = [P.rel(x, z) / r for x, z, r in zip(uf, u64, rif)]
            print(f"  {nome:4s} ({time.perf_counter() - t0:.2f}s) rapporti: "
                  + " ".join(f"{n} {r:.2f}" for n, r in zip(P.NOMI, rap)) + f" · peggiore {max(rap):.2f}", flush=True)
        C.chunk_local_cumsum = originale
        del dati, u64, ut
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
