#!/usr/bin/env python3
"""Sintonia SICURA delle configurazioni fla su gfx1151 (27/9/2026): discesa per coordinate, un kernel alla volta,
candidati misurati a passi ALTERNATI con la migliore corrente (la GPU e' condivisa con la corsa EB2).
Sicura perche' la guardia scratch (gdn_fla_gfx1151.installa_guardia_scratch) rifiuta di CARICARE ogni candidato con
spill: nessun kernel a rischio page fault viene mai lanciato; il candidato rifiutato si scarta.
Stampa la tabella migliore (JSON) da copiare in CONFIG_GFX1151 / AMBIENTE_GFX1151, poi va riverificata con
prova_equivalenza.py. Lanciare con lancia_prova.sh (PROVA=sintonia.py)."""
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402

# (gruppo, [candidati]); un candidato = {"tab": {kernel: config}, "env": {variabile: valore}}
def C(kernel, nw, **kw):
    return {"tab": {kernel: {**kw, "num_warps": nw, "num_stages": 1}}, "env": {}}


CANDIDATI = [
    ("dqkwg+dv_local (blocchi wrapper)", [
        {"tab": {}, "env": {"FLA_GFX1151_BLOCCO_O": str(bk), "FLA_GFX1151_BLOCCO_O_V": str(bv)}}
        for bk, bv in ((32, 16), (64, 16), (16, 32), (32, 32))]),
    ("dqkwg warp", [C("chunk_bwd_kernel_dqkwg", nw) for nw in (8, 32)]),
    ("dhu", [C("chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64", nw, BV=bv) for bv, nw in ((32, 16), (16, 8), (32, 8))]),
    ("fwd_o", [C("chunk_fwd_kernel_o", nw, BK=bk, BV=bv)
               for bk, bv, nw in ((32, 16, 16), (16, 32, 16), (32, 32, 16), (64, 16, 16), (16, 16, 8), (32, 32, 8))]),
    ("fwd_h", [C("chunk_gated_delta_rule_fwd_kernel_h_blockdim64", nw, BV=bv) for bv, nw in ((32, 16), (16, 8), (16, 32))]),
    ("recompute_w_u", [{"tab": {"recompute_w_u_fwd_kernel": {"num_warps": nw, "num_stages": 1}},
                        "env": {"FLA_GFX1151_BLOCCO_WU": str(t)}} for t, nw in ((32, 16), (16, 8), (32, 32))]),
    ("dv_local warp", [C("chunk_bwd_kernel_dv_local", nw) for nw in (8, 32)]),
    ("kkt_solve", [C("chunk_gated_delta_rule_fwd_kkt_solve_kernel", nw, BK=bk) for bk, nw in ((32, 16), (16, 8), (64, 16))]),
]


def applica(cfg):
    for k in [k for k in os.environ if k.startswith("FLA_GFX1151_BLOCCO")]:   # niente residui del candidato prima
        del os.environ[k]
    for k, v in {**G.AMBIENTE_GFX1151, **cfg["env"]}.items():
        os.environ[k] = v
    G.fissa_config(tabella=cfg["tab"], verbose=False)


def passo(f, dati):
    q, k, v, gg, beta, do = dati
    xs = [t.clone().requires_grad_(True) for t in (q, k, v, gg, beta)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], use_qk_l2norm_in_kernel=True)
    o.backward(do)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> int:
    R = int(os.environ.get("SINTONIA_R", "5"))
    fla_gdn = G.configura()
    B, T, H, D = 2, 4096, 32, 128
    gen = torch.Generator().manual_seed(7)
    dati = [torch.randn(B, T, H, D, generator=gen) for _ in range(3)]
    dati += [-torch.nn.functional.softplus(torch.randn(B, T, H, generator=gen)),
             torch.sigmoid(torch.randn(B, T, H, generator=gen)), torch.randn(B, T, H, D, generator=gen)]
    dati = [t.cuda() for t in dati]
    migliore = {"tab": copy.deepcopy(G.CONFIG_GFX1151), "env": {}}
    applica(migliore)
    passo(fla_gdn, dati)
    base = statistics.median(passo(fla_gdn, dati) for _ in range(R))
    print(f"partenza: {base * 1e3:.1f} ms avanti+indietro (mediana di {R})", flush=True)
    for gruppo, candidati in CANDIDATI:
        for cand in candidati:
            prova = {"tab": {**migliore["tab"], **cand["tab"]}, "env": {**migliore["env"], **cand["env"]}}
            descr = json.dumps({**cand["tab"], **cand["env"]})
            try:
                applica(prova)
                passo(fla_gdn, dati)                           # compila (la guardia rifiuta gli spill qui)
                tm, tc = [], []
                for _ in range(R):
                    applica(migliore)
                    tm.append(passo(fla_gdn, dati))
                    applica(prova)
                    tc.append(passo(fla_gdn, dati))
            except G.ScratchVietato as e:
                print(f"  [{gruppo}] {descr}: RIFIUTATO dalla guardia ({e})", flush=True)
                applica(migliore)
                continue
            m, c = statistics.median(tm), statistics.median(tc)
            meglio = c < 0.95 * m
            print(f"  [{gruppo}] {descr}: {c * 1e3:.1f} ms contro {m * 1e3:.1f} ms {'-> PRESO' if meglio else ''}",
                  flush=True)
            if meglio:
                migliore = prova
            applica(migliore)
    finale = statistics.median(passo(fla_gdn, dati) for _ in range(R))
    print(f"fine: {finale * 1e3:.1f} ms (partenza {base * 1e3:.1f} ms)", flush=True)
    print("MIGLIORE " + json.dumps(migliore), flush=True)
    (QUI / "esiti").mkdir(exist_ok=True)
    (QUI / "esiti" / "sintonia_migliore.json").write_text(json.dumps(migliore, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
