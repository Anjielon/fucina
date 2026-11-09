#!/usr/bin/env python3
"""fla (copia pkg/, config fissate) contro il ripiego PyTorch di transformers, alle dimensioni vere di Thor-T (27/9/2026).

Criterio (dal mandato): errore verso fp64 del candidato non peggiore di 2x quello del ripiego fp32, in avanti e in
TUTTI i gradienti (dq, dk, dv, dg, dbeta). Riferimento fp64 = il sorgente del ripiego con float32 -> float64
(G.ripiego_fp64): il ripiego chiamato con tensori fp64 converte tutto a float32 dentro e NON e' un riferimento
(errore che strumenti/prova_gdn_fla.py commette).
Due regimi di ingresso: g = -softplus(N(0,1)) (decadimento dolce) e g = -8*softplus(N(0,1)) (decadimento forte,
exp che scende sotto 1e-30 dentro il blocco: e' dove exp2 di fla e exp del ripiego possono divergere).

Lanciare con lancia_prova.sh (unita' systemd --user, un solo processo GPU di prova, cache Triton separata).
  --config fissi      regola di ripiego di gdn_fla_gfx1151._scegli (nessun autotune)
  --config tabella    CONFIG_GFX1151 (+ ripiego per i kernel assenti)
  --config autotune   spazio originale di fla (ATTENZIONE: su gfx1151 fp32 e' qui che nasceva il page fault)
  --diag              stampa ogni lancio di kernel (con AMD_SERIALIZE_KERNEL=3 il fault cade dopo l'ultimo stampato)
"""
from __future__ import annotations

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

sys.modules["fla"] = None                        # transformers deve legare il RIPIEGO, come nella corsa
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402

RIPIEGO = M.torch_chunk_gated_delta_rule
RIPIEGO64 = G.ripiego_fp64(M)          # fp64 VERO (il ripiego chiamato in fp64 calcola in fp32: vedi G.ripiego_fp64)
NOMI = ("avanti", "dq", "dk", "dv", "dg", "dbeta")


def ingressi(seme: int, B: int, T: int, H: int, D: int, forza_g: float, dt=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seme)
    q = torch.randn(B, T, H, D, generator=g)
    k = torch.randn(B, T, H, D, generator=g)
    v = torch.randn(B, T, H, D, generator=g)
    gg = -forza_g * torch.nn.functional.softplus(torch.randn(B, T, H, generator=g))
    beta = torch.sigmoid(torch.randn(B, T, H, generator=g))
    do = torch.randn(B, T, H, D, generator=g)
    return [t.to("cuda", dt) for t in (q, k, v, gg, beta, do)]


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
    t2 = time.perf_counter()
    return [o.detach()] + [x.grad.detach() for x in xs], t1 - t0, t2 - t1


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def tempi(f, dati, rip):
    av, ind = [], []
    for _ in range(rip):
        _, a, i = passo(f, dati)
        av.append(a)
        ind.append(i)
    return statistics.median(av), statistics.median(ind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gfx1151", choices=("gfx1151", "fissi", "autotune"))
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--dims", default="2,4096,32,128")
    ap.add_argument("--semi", default="1:1,2:8", help="seme:forza_g, separati da virgola")
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--senza-fp64", action="store_true")
    ap.add_argument("--esito", default="")
    ap.add_argument("--tabella-json", default="", help="config per kernel (sovrascrive CONFIG_GFX1151)")
    ap.add_argument("--senza-guardia", action="store_true", help="lancia anche kernel con scratch (rischio fault)")
    a = ap.parse_args()
    B, T, H, D = (int(x) for x in a.dims.split(","))

    tab = json.loads(Path(a.tabella_json).read_text()) if a.tabella_json else None
    if a.config == "gfx1151":
        fla_gdn = G.configura(tabella=tab)              # ricetta intera (guardia scratch compresa)
    else:
        fla_gdn = G.carica_fla()
        if a.config == "fissi":
            G.fissa_config(tabella=tab or {}, verbose=False)
        if not a.senza_guardia:
            G.installa_guardia_scratch()
    import fla
    print(f"fla {fla.__version__} da {fla.__file__} · impronta copia {(G.PKG / 'IMPRONTA').read_text().strip()}",
          flush=True)
    print(f"triton {__import__('triton').__version__} · torch {torch.__version__} · "
          f"{torch.cuda.get_device_name(0)} · dims {(B, T, H, D)}", flush=True)
    import os
    print(f"TRITON_F32_DEFAULT={os.environ.get('TRITON_F32_DEFAULT')} · cache {os.environ.get('TRITON_CACHE_DIR')}",
          flush=True)
    print("ambiente: " + " ".join(f"{k}={os.environ.get(k)}" for k in G.AMBIENTE_GFX1151), flush=True)
    if a.diag:
        import triton
        def entra(md):
            print(f"    lancio {md.get()['name']}", flush=True)
        triton.knobs.runtime.launch_enter_hook.add(entra)

    esiti = {"config": a.config, "dims": [B, T, H, D], "casi": []}
    for voce in a.semi.split(","):
        seme, forza = int(voce.split(":")[0]), float(voce.split(":")[1])
        dati = ingressi(seme, B, T, H, D, forza)
        t0 = time.perf_counter()
        try:
            uf, af, if_ = passo(fla_gdn, dati)             # primo passo: compilazione (+ autotune) dentro
        except G.ScratchVietato as e:
            print(f"GUARDIA SCRATCH: {e}", flush=True)
            for n, r in G.RISORSE.items():
                print(f"  risorse {n}: {r}", flush=True)
            return 4
        print(f"seme {seme} forza_g {forza}: primo passo fla {time.perf_counter() - t0:.1f}s "
              f"(avanti {af:.1f}s, indietro {if_:.1f}s)", flush=True)
        if a.diag:
            triton.knobs.runtime.launch_enter_hook.remove(entra)
        ok_finito = all(bool(torch.isfinite(t).all()) for t in uf)
        uf2, _, _ = passo(fla_gdn, dati)                   # determinismo: stesso ingresso, stessa uscita?
        det = max(float((x - y).abs().max()) for x, y in zip(uf, uf2))
        taf, tif = tempi(fla_gdn, dati, a.ripetizioni)
        ut, _, _ = passo(RIPIEGO, dati)
        tat, tit = tempi(RIPIEGO, dati, a.ripetizioni)
        caso = {"seme": seme, "forza_g": forza, "finito": ok_finito, "determinismo_maxabs": det,
                "tempo_fla": [taf, tif], "tempo_ripiego": [tat, tit]}
        print(f"  finito {ok_finito} · determinismo (maxabs fra due passi) {det:.1e}", flush=True)
        print(f"  tempo avanti/indietro: fla {taf:.3f}/{tif:.3f}s · ripiego {tat:.3f}/{tit:.3f}s · "
              f"x{(tat + tit) / (taf + tif):.2f}", flush=True)
        if not a.senza_fp64:
            u64, _, _ = passo(RIPIEGO64, [t.double() for t in dati])
            peggio = 0.0
            for n, x, y, z in zip(NOMI, uf, ut, u64):
                ef, et, eft = rel(x, z), rel(y, z), rel(x, y)
                caso[n] = {"fla_vs_fp64": ef, "ripiego_vs_fp64": et, "fla_vs_ripiego": eft, "rapporto": ef / et}
                peggio = max(peggio, ef / et)
                print(f"  {n:7s} fla~fp64 {ef:.2e} · ripiego~fp64 {et:.2e} · rapporto {ef / et:.2f} · "
                      f"fla~ripiego {eft:.2e}", flush=True)
            caso["rapporto_peggiore"] = peggio
            caso["nel_rumore_fp32"] = bool(peggio <= 2.0 and ok_finito)
            print(f"  ESITO: rapporto peggiore {peggio:.2f} -> {'NEL RUMORE fp32' if peggio <= 2 else 'FUORI'}",
                  flush=True)
        esiti["casi"].append(caso)
        del dati, uf, uf2, ut
        torch.cuda.empty_cache()
    esiti["risorse"] = G.RISORSE
    for n, r in G.RISORSE.items():
        print(f"  risorse {n}: {r}", flush=True)
    if a.esito:
        Path(a.esito).parent.mkdir(parents=True, exist_ok=True)
        Path(a.esito).write_text(json.dumps(esiti, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
