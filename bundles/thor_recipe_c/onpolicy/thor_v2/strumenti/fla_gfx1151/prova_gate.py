#!/usr/bin/env python3
"""Dopo prova_stadi (tutti gli stadi in fp64 e l'errore resta 1.1e-6): l'errore nasce nel GATE cumulato g' = cumsum(g)/ln2.
Varianti (stadi intra+h+o in fp64 per isolare il gate; poi gli stadi fla veri):
  K32   cumsum fla (fp32, costante RCP_LN2=1.4426950216) -> come oggi
  C_c   cumsum fp64 con la costante di fla, uscita fp64
  C_e   cumsum fp64 con 1/ln2 esatto, uscita fp64
  C_e32 cumsum fp64 con 1/ln2 esatto, uscita arrotondata a fp32"""
import math
import sys
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402
import prova_equivalenza as P  # noqa: E402
import prova_stadi as S  # noqa: E402


def fai_cumsum(costante, dtype_uscita):
    def f(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False, output_dtype=torch.float,
          chunk_indices=None, **kw):
        B, T, H = g.shape
        x = g.double().reshape(B, T // chunk_size, chunk_size, H)
        y = x.flip(2).cumsum(2).flip(2) if reverse else x.cumsum(2)
        if scale is not None:
            y = y * (costante if costante is not None else scale)
        return y.reshape(B, T, H).to(dtype_uscita)
    return f


def main() -> int:
    fla_gdn = G.configura()
    import fla.ops.gated_delta_rule.chunk as C
    orig = {"chunk_gated_delta_rule_fwd_intra": C.chunk_gated_delta_rule_fwd_intra,
            "chunk_gated_delta_rule_fwd_h": C.chunk_gated_delta_rule_fwd_h, "chunk_fwd_o": C.chunk_fwd_o,
            "chunk_local_cumsum": C.chunk_local_cumsum}
    varianti = {"K32": None, "C_c": fai_cumsum(1.4426950216, torch.float64),
                "C_e": fai_cumsum(1 / math.log(2), torch.float64), "C_e32": fai_cumsum(1 / math.log(2), torch.float32)}
    for seme, forza in ((1, 1.0), (2, 8.0)):
        q, k, v, gg, be, do = P.ingressi(seme, 2, 4096, 32, 128, forza)
        with torch.no_grad():
            o64_, _ = P.RIPIEGO(q.double(), k.double(), v.double(), g=gg.double(), beta=be.double(),
                                use_qk_l2norm_in_kernel=True)
            o32r, _ = P.RIPIEGO(q, k, v, g=gg, beta=be, use_qk_l2norm_in_kernel=True)
            er = P.rel(o32r, o64_)
            print(f"seme {seme} forza {forza}: ripiego fp32~fp64 {er:.2e}", flush=True)
            for stadi64 in (True, False):
                for nome, cum in varianti.items():
                    if not stadi64 and nome in ("C_c", "C_e"):
                        continue                          # i kernel fla vogliono g in fp32
                    C.chunk_local_cumsum = cum or orig["chunk_local_cumsum"]
                    C.chunk_gated_delta_rule_fwd_intra = S.intra64 if stadi64 else orig["chunk_gated_delta_rule_fwd_intra"]
                    C.chunk_gated_delta_rule_fwd_h = S.h64 if stadi64 else orig["chunk_gated_delta_rule_fwd_h"]
                    C.chunk_fwd_o = S.o64 if stadi64 else orig["chunk_fwd_o"]
                    of, _ = fla_gdn(q, k, v, g=gg, beta=be, use_qk_l2norm_in_kernel=True)
                    e = P.rel(of, o64_)
                    print(f"  stadi {'fp64' if stadi64 else 'fla '} gate {nome:6s}: fla~fp64 {e:.2e} (rapporto {e / er:.2f})",
                          flush=True)
            for n, f in orig.items():
                setattr(C, n, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
