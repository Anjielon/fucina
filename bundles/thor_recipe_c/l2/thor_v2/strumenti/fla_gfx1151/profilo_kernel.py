#!/usr/bin/env python3
"""Tempo per kernel (torch.profiler, tempo GPU) di fla (ricetta gfx1151) e del ripiego, avanti+indietro alle dimensioni
vere, e tempi a passi ALTERNATI fla/ripiego (la GPU e' condivisa con la corsa: alternare rende il confronto equo).
Uso: lancia_prova.sh con PROVA=profilo_kernel.py [--tabella-json t.json] [--ripetizioni N]"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402

sys.modules["fla"] = None
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402

RIPIEGO = M.torch_chunk_gated_delta_rule


def passo(f, dati):
    q, k, v, gg, beta, do = dati
    xs = [t.clone().requires_grad_(True) for t in (q, k, v, gg, beta)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    o.backward(do)
    torch.cuda.synchronize()
    return t1 - t0, time.perf_counter() - t1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabella-json", default="")
    ap.add_argument("--ripetizioni", type=int, default=9)
    ap.add_argument("--senza-profilo", action="store_true")
    a = ap.parse_args()
    tab = json.loads(Path(a.tabella_json).read_text()) if a.tabella_json else None
    fla_gdn = G.configura(tabella=tab)
    B, T, H, D = 2, 4096, 32, 128
    gen = torch.Generator().manual_seed(3)
    dati = [torch.randn(B, T, H, D, generator=gen) for _ in range(3)]
    dati += [-torch.nn.functional.softplus(torch.randn(B, T, H, generator=gen)),
             torch.sigmoid(torch.randn(B, T, H, generator=gen)), torch.randn(B, T, H, D, generator=gen)]
    dati = [t.cuda() for t in dati]
    passo(fla_gdn, dati)                                   # compilazione
    passo(RIPIEGO, dati)
    tf, tr = [], []
    for _ in range(a.ripetizioni):                         # alternati
        tf.append(passo(fla_gdn, dati))
        tr.append(passo(RIPIEGO, dati))
    med = lambda xs, i: statistics.median(x[i] for x in xs)  # noqa: E731
    fa, fi, ra, ri = med(tf, 0), med(tf, 1), med(tr, 0), med(tr, 1)
    print(f"mediana su {a.ripetizioni} passi alternati: fla avanti {fa:.3f}s indietro {fi:.3f}s totale {fa + fi:.3f}s · "
          f"ripiego avanti {ra:.3f}s indietro {ri:.3f}s totale {ra + ri:.3f}s · x{(ra + ri) / (fa + fi):.2f}", flush=True)
    print(f"  min: fla {min(x[0] + x[1] for x in tf):.3f}s · ripiego {min(x[0] + x[1] for x in tr):.3f}s", flush=True)
    if a.senza_profilo:
        return 0
    from torch.profiler import ProfilerActivity, profile
    for nome, f in (("fla", fla_gdn), ("ripiego", RIPIEGO)):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            passo(f, dati)
        tab_k = prof.key_averages().table(sort_by="device_time_total", row_limit=14, max_name_column_width=60)
        print(f"=== {nome}\n{tab_k}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
