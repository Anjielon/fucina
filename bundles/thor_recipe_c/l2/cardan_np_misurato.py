#!/usr/bin/env python3
"""Number of expert-fold workers from MEASURED peaks (D3 closure, 22/10). Stdlib only, no GPU context.

The v10 chain sized workers as (MemAvailable - reserve) / 3 GiB, an assumption: one production lotti worker measured
8.7 GB peak RSS (one full layer, 22/10). This helper replaces the divisor with the measured profile:

  profile JSON: {"rss_anon_byte": peak anonymous RSS per worker, "rss_byte": peak total RSS (fallback),
                 "vram_byte": peak VRAM per worker, "ginocchio": workers at the measured throughput knee,
                 "fonte": "..."}
  NP = max(1, min(ginocchio, max_np, (MemAvailable - riserva) // rss, (VRAM free - riserva_vram) // vram))

Usage (prints NP on stdout): python3 cardan_np_misurato.py PROFILE.json [--riserva-gib 8] [--max-np 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

GIB = 1 << 30


def mem_available(path="/proc/meminfo"):
    for line in open(path):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def vram_libera(drm="/sys/class/drm"):
    best = None
    for tot in glob.glob(f"{drm}/card*/device/mem_info_vram_total"):
        used = tot.replace("_total", "_used")
        try:
            free = int(open(tot).read()) - int(open(used).read())
        except (OSError, ValueError):
            continue
        best = free if best is None else max(best, free)
    return best


def numero(profilo, mem, vram, riserva=8 * GIB, riserva_vram=4 * GIB, max_np=3):
    rss = profilo.get("rss_anon_byte") or profilo["rss_byte"]
    if type(rss) is not int or rss <= 0:
        raise ValueError("profile without a positive measured RSS peak")
    limiti = [int(profilo.get("ginocchio", max_np)), max_np, (mem - riserva) // rss]
    if vram is not None and profilo.get("vram_byte"):
        limiti.append((vram - riserva_vram) // int(profilo["vram_byte"]))
    return max(1, min(limiti))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("profilo")
    ap.add_argument("--riserva-gib", type=float, default=8)
    ap.add_argument("--max-np", type=int, default=3)
    a = ap.parse_args(argv)
    p = json.load(open(a.profilo))
    n = numero(p, mem_available(), vram_libera(), int(a.riserva_gib * GIB), max_np=a.max_np)
    print(n)
    print(f"NP={n} from measured profile {a.profilo}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
