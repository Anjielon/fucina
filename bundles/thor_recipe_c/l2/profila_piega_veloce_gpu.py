#!/usr/bin/env python3
"""profila_piega_veloce_gpu.py (22/10) — where the GPU time of ONE expert family goes (torch.profiler), lotti vs veloce.
Usage (GPU window only, via finestra_gpu.py): python profila_piega_veloce_gpu.py --motore lotti|veloce --out DIR
Writes DIR/<motore>_kernel.txt (top kernels by device time) and DIR/<motore>_profilo.json (totals)."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardan_piega_lotti as PL  # noqa: E402
import cardan_piega_veloce as PV  # noqa: E402

W_ = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_CONGIUNTO_S7T550_1020"
BASE = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/S7_CONGIUNTO_T550_1020/Thor-T-C-S7-T550-g10.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motore", choices=("lotti", "veloce"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strato", type=int, default=3)
    ap.add_argument("--esperti", type=int, default=32)
    a = ap.parse_args()
    os.environ["LEVE_DT"] = "float32"
    out = Path(a.out); (out / a.motore / "tensori").mkdir(parents=True, exist_ok=True)
    d = torch.load(f"{W_}/allena/tappe/cardan_350.pt", map_location="cpu", weights_only=False)
    stato = d["stato"].get(a.strato, d["stato"].get(str(a.strato)))
    mod = PL if a.motore == "lotti" else PV
    dev = torch.device("cuda")
    torch.ones(1, device=dev).sum().item()
    with open(out / a.motore / "registro.jsonl", "w") as reg, ThreadPoolExecutor(PL.PREP_FILI) as pool:
        t0 = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            mod.piega_strato(a.strato, stato, BASE, dev, torch.float32, 16, out / a.motore / "tensori", reg, pool,
                             esperti=a.esperti)
            torch.cuda.synchronize()
        parete = time.time() - t0
    ka = pr.key_averages()
    (out / f"{a.motore}_kernel.txt").write_text(ka.table(sort_by="self_cuda_time_total", row_limit=40))
    dev_us = sum(e.self_device_time_total for e in ka)
    n_kern = sum(e.count for e in ka if e.device_type == torch.autograd.DeviceType.CUDA)
    tot = {"motore": a.motore, "strato": a.strato, "esperti": a.esperti, "parete_s": round(parete, 2),
           "gpu_s": round(dev_us / 1e6, 2), "kernel_lanciati": n_kern}
    (out / f"{a.motore}_profilo.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
