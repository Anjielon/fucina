"""GPU hardware-queue admission for the forge (AMD KFD; no-op elsewhere).

Why it exists (21/10/2026): the GPU (gfx1151, 1 SDMA engine x 6 queues, 8 compute
queues) hung for 1.5 h because 39 processes held /dev/kfd at once: the training run
plus its 33 torch-inductor compile workers (each opens the device and takes one SDMA
queue) plus 5 fold workers.  The kernel logged «No more SDMA queue to allocate» and
the device never recovered (reset failed, -110): only a reboot freed it.

What it does: reads the queue capacity from the KFD topology and the queues every
process holds right now (/sys/class/kfd/kfd/proc/<pid>/queues/<q>/type), and turns
them into a number of workers that can be admitted without
  - exceeding the SDMA queues (hard limit: an allocation failure wedges the device);
  - oversubscribing the compute queues beyond what the caller allows (soft limit:
    the scheduler time-slices, so every process slows down; the caller decides how
    much oversubscription is acceptable after measuring throughput).
VRAM is not handled here: pianificatore_gpu.Budget already reserves it.

KFD queue types (include/uapi/linux/kfd_ioctl.h): 0 compute, 1 SDMA, 2 SDMA xGMI.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

KFD = Path("/sys/class/kfd/kfd")
COMPUTE, SDMA, SDMA_XGMI = 0, 1, 2
_READ = object()  # "read it from the system" (None already means "no KFD")


@dataclasses.dataclass(frozen=True)
class Capacity:
    compute: int  # hardware compute queues (num_cp_queues)
    sdma: int     # num_sdma_engines * num_sdma_queues_per_engine (+ xGMI engines)


@dataclasses.dataclass(frozen=True)
class Usage:
    compute: int
    sdma: int
    per_pid: dict


def _props(node: Path) -> dict:
    out = {}
    for line in (node / "properties").read_text().splitlines():
        k, _, v = line.partition(" ")
        if v.strip().lstrip("-").isdigit():
            out[k] = int(v)
    return out


def capacity(kfd: Path = KFD) -> Capacity | None:
    """Queue capacity of the first GPU node; None when there is no KFD (CUDA/CPU hosts)."""
    nodes = kfd / "topology" / "nodes"
    if not nodes.is_dir():
        return None
    for node in sorted(nodes.iterdir()):
        p = _props(node)
        if p.get("simd_count", 0) > 0:
            per = p.get("num_sdma_queues_per_engine", 0)
            sdma = (p.get("num_sdma_engines", 0) + p.get("num_sdma_xgmi_engines", 0)) * per
            return Capacity(compute=p.get("num_cp_queues", 0), sdma=sdma)
    return None


def usage(kfd: Path = KFD) -> Usage:
    """Queues held right now, summed over every process that opened the device."""
    comp = sdma = 0
    per_pid = {}
    proc = kfd / "proc"
    for d in (proc.iterdir() if proc.is_dir() else ()):
        c = s = 0
        for q in (d / "queues").glob("*/type"):
            try:
                t = int(q.read_text().strip())
            except (OSError, ValueError):
                continue  # process exited while we were reading
            if t == COMPUTE:
                c += 1
            elif t in (SDMA, SDMA_XGMI):
                s += 1
        per_pid[d.name] = {"compute": c, "sdma": s}
        comp += c
        sdma += s
    return Usage(compute=comp, sdma=sdma, per_pid=per_pid)


def admissible(per_worker_compute: int, per_worker_sdma: int, *, sdma_reserve: int = 1,
               compute_oversubscription: float = 1.0, cap=_READ, used=_READ) -> int | None:
    """How many NEW workers fit.  None = no queue limit known (not an AMD KFD host).

    per_worker_*: queues one worker takes, measured (a fold with HSA_ENABLE_SDMA=0 takes 2 compute, 0 SDMA;
    a torch process with SDMA on takes 2 compute + 1 SDMA).  sdma_reserve keeps queues free for the
    processes that will start later (a resumed run, a copy).  compute_oversubscription = 1.0 never
    exceeds the hardware compute queues; >1 only after a throughput measurement proved it pays."""
    cap = capacity() if cap is _READ else cap
    if cap is None:
        return None
    used = usage() if used is _READ else used
    limits = []
    if per_worker_sdma > 0:
        limits.append((cap.sdma - sdma_reserve - used.sdma) // per_worker_sdma)
    elif used.sdma > cap.sdma - sdma_reserve:
        return 0  # the device is already at the SDMA edge: admit nothing, whatever the worker needs
    if per_worker_compute > 0:
        limits.append(int(cap.compute * compute_oversubscription - used.compute) // per_worker_compute)
    return max(0, min(limits)) if limits else 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-worker-compute", type=int, default=2)
    ap.add_argument("--per-worker-sdma", type=int, default=0)
    ap.add_argument("--sdma-reserve", type=int, default=1)
    ap.add_argument("--compute-oversubscription", type=float, default=1.0)
    ap.add_argument("--json", action="store_true", help="print capacity, usage and verdict as JSON")
    a = ap.parse_args(argv)
    cap, used = capacity(), usage()
    n = admissible(a.per_worker_compute, a.per_worker_sdma, sdma_reserve=a.sdma_reserve,
                   compute_oversubscription=a.compute_oversubscription, cap=cap, used=used)
    if a.json:
        print(json.dumps({"capacity": dataclasses.asdict(cap) if cap else None,
                          "usage": dataclasses.asdict(used), "admissible": n}))
    else:
        print("unlimited" if n is None else n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
