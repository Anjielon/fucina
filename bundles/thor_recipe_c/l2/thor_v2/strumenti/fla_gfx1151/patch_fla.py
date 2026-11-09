#!/usr/bin/env python3
"""Patch della COPIA di fla per gfx1151 (27/9/2026). Ogni patch ha il suo perche' misurato; i default restano quelli
di fla (senza le variabili d'ambiente la copia si comporta come l'originale)."""
import sys
from pathlib import Path

_E = "__import__('os').environ.get"
PATCH: list[tuple[str, str, str, int, str]] = [   # (file, vecchio, nuovo, occorrenze attese, perche')
    ("ops/gated_delta_rule/chunk_fwd.py",
     "    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')",
     f"    SOLVE_TRIL_DOT_PRECISION = tl.constexpr({_E}('TRITON_F32_DEFAULT') or 'ieee')", 1,
     "l'inversa a blocchi di kkt_solve fissa input_precision='ieee' e cosi' scavalca TRITON_F32_DEFAULT"),
    ("ops/gated_delta_rule/wy_fast.py",
     "    BK = 64\n    BV = 64\n",
     f"    BK = int({_E}('FLA_GFX1151_BLOCCO_WU', 64))\n    BV = BK\n", 1,
     "recompute_w_u_fwd: blocchi 64x64 cablati; in fp32 (prodotto FMA) fanno spill su gfx1151 (1100 B a 16 warp)"),
    ("ops/gated_delta_rule/wy_fast.py",
     "    CONST_TILING = 64 if check_shared_mem() else 32\n",
     f"    CONST_TILING = int({_E}('FLA_GFX1151_BLOCCO_WY', 64 if check_shared_mem() else 32))\n", 1,
     "prepare_wy_repr_bwd: blocco 32 fa spill in fp32 su gfx1151 (1152 B a 16 warp)"),
    ("ops/common/chunk_o.py",
     "        CONST_TILING = 32\n",
     f"        CONST_TILING = int({_E}('FLA_GFX1151_BLOCCO_O', 32))\n", 3,
     "chunk_bwd_dv / dv_local / dqkwg: blocco 32 fa spill in fp32 su gfx1151 (dqkwg 984 B a 16 warp)"),
    ("ops/common/chunk_o.py",
     "    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)\n",
     f"    BV = int({_E}('FLA_GFX1151_BLOCCO_O_V', min(max(triton.next_power_of_2(V), 16), CONST_TILING)))\n", 3,
     "BV separato da BK in dv / dv_local / dqkwg (dqkwg rifa' dS=do v^T per ogni blocco K: BK grande conviene)"),
]


# Prodotto spezzato lungo la contrazione: su gfx1151 (RDNA3.5, niente WMMA fp32) tl.dot fp32 diventa FMA e Triton
# carica gli operandi interi nei registri (righe x K per thread): con K=64 servono 384+ VGPR -> spill in scratch
# (misura 27/9: dhu 1452 B a 16 warp -> 0 B spezzato; prepare_wy_repr_bwd resta 268 B anche spezzato: va in torch,
# vedi gdn_fla_gfx1151.prepare_wy_repr_bwd_torch). Qui K si spezza in fette da 16 con un prodotto 3D
# (lotto = fetta) e una somma: stessa aritmetica fp32 (ordine di somma diverso), operandi per thread /4.
HELPER = """

@triton.jit
def _dot_spezzato(a, b):
    M: tl.constexpr = a.shape[0]
    KK: tl.constexpr = a.shape[1]
    N: tl.constexpr = b.shape[1]
    if KK == 16:
        return tl.dot(a, b)
    else:
        a3 = tl.permute(tl.reshape(a, (M, KK // 16, 16)), (1, 0, 2))
        b3 = tl.reshape(b, (KK // 16, 16, N))
        return tl.sum(tl.dot(a3, b3), axis=0)
"""
REGIONI = [   # (file, funzione kernel, ancora dopo cui inserire HELPER)
    ("ops/common/chunk_delta_h.py", "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64", "\nNUM_WARPS = "),
]


def spezza(radice: Path) -> None:
    import os
    if os.environ.get("FLA_GFX1151_SENZA_SPEZZATO") == "1":
        return
    for rel, funzione, ancora in REGIONI:
        f = radice / rel
        t = f.read_text()
        if "def _dot_spezzato(" not in t:
            i = t.index(ancora)
            t = t[:i] + HELPER + t[i:]
        a = t.index(f"def {funzione}(")
        b = min(x for x in (t.find("\n@", a), t.find("\ndef ", a + 5)) if x > 0)
        corpo = t[a:b]
        nuovo = corpo.replace("tl.dot(", "_dot_spezzato(")
        if nuovo != corpo:
            print(f"patch {rel}: {funzione} usa _dot_spezzato ({corpo.count('tl.dot(')} prodotti)")
        f.write_text(t[:a] + nuovo + t[b:])


def main() -> int:
    radice = Path(sys.argv[1])
    spezza(radice)
    for rel, vecchio, nuovo, quante, perche in PATCH:
        f = radice / rel
        testo = f.read_text()
        if testo.count(nuovo) == quante:
            continue
        assert testo.count(vecchio) == quante, f"{rel}: {testo.count(vecchio)} occorrenze, attese {quante}"
        f.write_text(testo.replace(vecchio, nuovo))
        print(f"patch {rel} (x{quante}): {perche}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
