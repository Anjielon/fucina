"""Allocazione R-D sui piani VERI di un GGUF ternario (CPU o GPU: CUDA e ROCm/HIP).

Sorgenti:
  piani   : GGUFPlaneSource (il GGUF ternario, mmap sola lettura)
  teacher : GGUFTeacherSource (GGUF bf16 della base originale, anche a frammenti)
  Fisher  : ImatrixFisherSource = PROXY diagonale dall'imatrix (H_I = diag, H_O = I/out)
            oppure --fisher-dir con i file fisher_L*.safetensors di KFACCollector (GPU).

Esempio (dry run su 2 strati, 16 esperti per tipo):
  nice -n 19 python3 -m rd_ternary.dry_run_gguf --layers 0 3 --max-experts 16 \
      --out piano_dry.json
Tutto il modello: --layers all --max-experts 256 (vedi README del risultato).

Su GPU (ROCm/HIP o CUDA, stesso codice; --device auto la sceglie da sola):
  python3 -m rd_ternary.dry_run_gguf --layers all --max-experts 256 --out piano_gpu.json
Divisa fra processi: --layers 0-13 / 14-27 / 28-40 con --out piano_parte{0,1,2}.json,
poi  python3 -m rd_ternary.unisci piano_parte*.json --expect-layers 0-40 --out piano.json
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import torch

from .allocazione import allocate
from .dispositivo import backend_name, batch_memory_budget, resolve_device, synchronize
from .gguf_piani import GGUFPlaneSource, GGUFTeacherSource, ImatrixFisherSource
from .menu import MenuConfig, build_menus
from .piano import export_plan, materialize
from .sorgenti import SafetensorsFisherSource
from .unisci import parse_layers

GGUF = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"
TEACHER = ["@@ROOT:models@@/gguf/thor-src/Ornith-1.5-35B-A3B-bf16/Ornith-1.5-35B-A3B-bf16-00001-of-00002.gguf",
           "@@ROOT:models@@/gguf/thor-src/Ornith-1.5-35B-A3B-bf16/Ornith-1.5-35B-A3B-bf16-00002-of-00002.gguf"]
IMATRIX = "@@ROOT:models@@/gguf/thor-rivale/Ornith-1.5-35B-A3B-imatrix.gguf"


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--teacher", nargs="+", default=TEACHER)
    ap.add_argument("--imatrix", default=IMATRIX)
    ap.add_argument("--fisher-dir", default=None, help="fisher_L*.safetensors (K-FAC vera) al posto dell'imatrix")
    ap.add_argument("--fisher-fallback", choices=("uniform", "error"), default="uniform",
                    help="voci assenti nell'imatrix (es. blk.40 MTP): ripiego uniforme (registrato) o errore")
    ap.add_argument("--fallback-fisher-dir", default=None,
                    help="fisher_L*.safetensors da usare PRIMA del ripiego uniforme per le voci assenti")
    ap.add_argument("--layers", nargs="+", default=["0", "3"], help="indici di strato, intervalli (0-40) o 'all'")
    ap.add_argument("--bases", nargs="*", default=None, help="solo questi tensori GGUF (es. attn_q ffn_gate_exps)")
    ap.add_argument("--max-experts", type=int, default=16, help="esperti per tipo e strato (256 = tutti)")
    ap.add_argument("--densities", nargs="*", type=float, default=[0.10, 0.25])
    ap.add_argument("--budget-rel", type=float, default=0.0, help="budget = byte attuali * (1 + x)")
    ap.add_argument("--device", default="auto", help="auto (GPU se c'e', CUDA o ROCm/HIP) | cpu | cuda | cuda:N")
    ap.add_argument("--batch", type=int, default=None, help="tensori per lotto (default: 8 su CPU, 256 su GPU)")
    ap.add_argument("--max-batch-weights", type=int, default=None,
                    help="tetto out*in*B per lotto (default: 8M su CPU, nessuno su GPU)")
    ap.add_argument("--mem-frac", type=float, default=0.5,
                    help="frazione della memoria libera del dispositivo per un lotto (stima per_tensor_bytes)")
    ap.add_argument("--no-fast-diag", action="store_true",
                    help="disattiva i percorsi per fattori diagonali (solo per confronto col riferimento)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None, help="piano JSON")
    ap.add_argument("--verify", action="store_true", help="rimaterializza e verifica byte/distorsione")
    ap.add_argument("--loadable-only", action="store_true",
                    help="allocazione solo fra le opzioni che il motore carica (motore.py)")
    ap.add_argument("--motore", choices=("oggi", "piani-variabili", "scala-g64"), default="oggi")
    ap.add_argument("--g64", action="store_true",
                    help="6/10: opzioni 'g64' e '+1g64' (scala ogni 64 pesi; caricabili solo col motore scala-g64)")
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    dev = resolve_device(a.device)
    on_gpu = dev.type == "cuda"
    batch = a.batch if a.batch is not None else (256 if on_gpu else 8)
    mbw = a.max_batch_weights if a.max_batch_weights is not None else (None if on_gpu else 8 * 1024 * 1024)
    mbb = batch_memory_budget(dev, a.mem_frac) if on_gpu else None
    t0 = time.time()
    layers = None if a.layers == ["all"] else parse_layers(a.layers)
    ps = GGUFPlaneSource(a.gguf, layers=layers, bases=a.bases, max_experts=a.max_experts)
    ts = GGUFTeacherSource(a.teacher)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.fisher_dir:
        fs = SafetensorsFisherSource(a.fisher_dir)
    else:
        fb = SafetensorsFisherSource(a.fallback_fisher_dir) if a.fallback_fisher_dir else None
        fs = ImatrixFisherSource(a.imatrix, out_of=lambda k: ps.info(k).out, in_of=lambda k: ps.info(k).in_,
                                 fallback=a.fisher_fallback, fallback_source=fb, log=log)
    keys = ps.keys()
    t_open = time.time() - t0
    print(f"[apertura] {len(keys)} tensori in {t_open:.1f}s  rss={peak_rss_mb():.0f} MB  "
          f"dispositivo={dev} ({backend_name(dev)}) lotto={batch} tetto_pesi={mbw} "
          f"tetto_byte={mbb if mbb is None else f'{mbb / 2**30:.1f} GiB'}", flush=True)

    cfg = MenuConfig(densities=tuple(a.densities), batch=batch, device=str(dev),
                     max_batch_weights=mbw, max_batch_bytes=mbb, fast_diag=not a.no_fast_diag, allow_g64=a.g64)
    t1 = time.time()
    menus = build_menus(ps, ts, fs, cfg, keys,
                        log=lambda m: print(f"  [{time.time() - t1:7.1f}s rss={peak_rss_mb():.0f}MB] {m}", flush=True))
    synchronize(dev)
    t_menu = time.time() - t1
    base = sum(m.get("cur").bytes for m in menus)
    budget = int(base * (1.0 + a.budget_rel))
    t2 = time.time()
    extra = {}
    if a.loadable_only:
        from . import motore
        from .gguf_grezzo import read_gguf
        from .scrivi_piano_gguf import n_main_layers
        src = read_gguf(a.gguf)
        n_exp = int(src.kv.get(f"{src.kv.get('general.architecture')}.expert_count", 256) or 256)
        alloc, cons = motore.solve_loadable(menus, budget, a.motore, n_main_layers(src), n_exp)
        extra["vincoli_motore"] = {"motore": a.motore, "opzioni_tolte": cons.dropped_options,
                                   "gruppi_esperti": cons.groups, "gruppi_fermi": cons.frozen_groups}
    else:
        alloc = allocate(menus, budget)
    t_alloc = time.time() - t2
    missing = dict(getattr(fs, "missing", {}))
    plan = export_plan(menus, alloc, cfg, a.out, note=f"dry run GGUF {a.gguf} layers={a.layers}",
                       extra={"missing_fisher": missing, "device": str(dev), **extra})
    counts: dict = {}
    for t in plan["tensors"]:
        counts[t["option"]] = counts.get(t["option"], 0) + 1
    rep = {"n_tensors": len(menus), "base_bytes": base, "budget": budget, "total_bytes": alloc.total_bytes,
           "pred_distortion_base": plan["pred_distortion_base"], "pred_distortion": alloc.total_distortion,
           "lambda": alloc.lam, "options": counts, "t_open_s": round(t_open, 2), "t_menu_s": round(t_menu, 2),
           "t_alloc_s": round(t_alloc, 4), "peak_rss_mb": round(peak_rss_mb(), 1), "device": str(dev),
           "missing_fisher": missing}
    if a.verify:
        t3 = time.time()
        r = materialize(plan, ps, ts, fs, device=str(dev), batch=batch, max_batch_weights=mbw,
                        max_batch_bytes=mbb)
        rep.update({"verify_bytes_mismatch": len(r["bytes_mismatch"]), "verify_distortion_ok": r["distortion_ok"],
                    "verify_encoder_mismatch": len(r["encoder_mismatch"]),
                    "verify_max_rel_err": r["max_rel_err_distortion"], "t_verify_s": round(time.time() - t3, 2),
                    "peak_rss_mb": round(peak_rss_mb(), 1)})
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
