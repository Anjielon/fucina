#!/usr/bin/env python3
"""Compila OFFLINE (nessun processo GPU) tutti i kernel del GDN di fla per gfx1151, alle dimensioni vere, e ne legge
registri, scratch (spill), LDS e numero di istruzioni WMMA dal .amdgcn (27/9/2026).

Perche': sulla GPU condivisa con la corsa ogni prova con un kernel che fa spill rischia il page fault (misurato:
chunk_gated_delta_rule_fwd_kernel_h_blockdim64 fp32 con 1588 B di scratch per corsia). Qui si scelgono SENZA GPU
le configurazioni a scratch zero; la GPU vede solo quelle.

Come: driver Triton finto (bersaglio gfx1151, wave32), JITFunction.run forzato a warmup=True (compila, non lancia),
tensori su CPU, autotune ridotto a una config per kernel (tabella JSON). Si percorre la vera funzione di fla in
avanti e all'indietro, quindi si compilano esattamente i kernel e le specializzazioni della corsa.
Uso: env HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 python3 compila_offline.py [--tabella t.json] [--f32 ieee]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

QUI = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabella", default="")
    ap.add_argument("--f32", default="ieee")
    ap.add_argument("--dims", default="2,4096,32,128")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="tc_offline_")
    os.environ["FLA_CACHE_MODE"] = "disabled"
    if a.f32 != "ieee":
        os.environ["TRITON_F32_DEFAULT"] = a.f32

    import triton
    import importlib
    D = importlib.import_module("triton.runtime.driver")
    from triton.backends.compiler import GPUTarget

    class Utils:
        def get_device_properties(self, i):
            return {"max_shared_mem": 65536, "multiprocessor_count": 20, "warpSize": 32}

    class DriverFinto:
        backend = "hip"
        utils = Utils()

        def get_current_target(self):
            return GPUTarget("hip", "gfx1151", 32)

        def get_current_device(self):
            return 0

        def get_current_stream(self, device=None):
            return 0

        def get_active_torch_device(self):
            import torch
            return torch.device("cpu")

        def get_device_interface(self):
            import torch
            return torch.cuda

    D.driver.set_active(DriverFinto())
    compilati: dict[str, dict] = {}
    orig_run = triton.runtime.jit.JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        k = orig_run(self, *args, grid=grid, warmup=True, **kwargs)
        if hasattr(k, "result"):
            k = k.result()
        asm = k.asm["amdgcn"]

        def campo(c):
            m = re.search(rf"\.{c}:\s+(\d+)", asm)
            return int(m.group(1)) if m else -1
        compilati.setdefault(k.name, {"num_warps": k.metadata.num_warps, "num_stages": k.metadata.num_stages,
                                      "vgpr": campo("vgpr_count"), "scratch": campo("private_segment_fixed_size"),
                                      "lds": k.metadata.shared, "wmma": len(re.findall(r"v_wmma", asm))})
        return k

    triton.runtime.jit.JITFunction.run = run

    import torch
    sys.path.insert(0, str(QUI))
    import gdn_fla_gfx1151 as G
    for k_, v_ in G.AMBIENTE_GFX1151.items():
        os.environ.setdefault(k_, v_)
    fla_gdn = G.carica_fla()
    tab = json.loads(Path(a.tabella).read_text()) if a.tabella else G.CONFIG_GFX1151
    G.fissa_config(tabella=tab, verbose=False)
    G.installa_sostituzioni()
    B, T, H, Dk = (int(x) for x in a.dims.split(","))
    q, k, v, do = (torch.randn(B, T, H, Dk) for _ in range(4))
    gg = -torch.nn.functional.softplus(torch.randn(B, T, H))
    be = torch.sigmoid(torch.randn(B, T, H))
    xs = [t.requires_grad_(True) for t in (q, k, v, gg, be)]
    o, _ = fla_gdn(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], use_qk_l2norm_in_kernel=True)
    o.backward(do)
    for n, r in sorted(compilati.items()):
        segno = "OK " if r["scratch"] == 0 else "SPILL"
        print(f"{segno} {n:52s} nw={r['num_warps']:2d} ns={r['num_stages']} vgpr={r['vgpr']:3d} "
              f"scratch={r['scratch']:5d} lds={r['lds']:6d} wmma={r['wmma']}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(compilati, indent=1))
    return 0 if all(r["scratch"] == 0 for r in compilati.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
