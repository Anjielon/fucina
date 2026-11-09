#!/usr/bin/env python3
"""Short GPU proofs for the closure of the 21/10 GPU-forge defects (22/10).

One mode per process, run serially. Admission BEFORE torch is imported: at most 2 KFD processes on the machine
(rule of 22/10), MemAvailable >= 8 GiB, swap-in < 20 MB/s; otherwise wait and retry (no threshold is ever raised).
No rocSOLVER call on the GPU, no extra threads with streams, single process, nice 19.

Modes (all read real data; all writes go to --out, a NEW directory):
  d4   Hessian normalisation on the GPU (guarded variant and legacy variant) vs the CPU original, real Gq of layer 16.
  d1   GPU blocked Cholesky (algebra_gpu, no rocSOLVER) + Freivalds guard on the real Ha blocks of layer 16; injected
       wrong factor / false "not SPD" must be detected and redone on CPU LAPACK fp64; host->device transfer check.
  d12  expert correction C[e] @ D^T on the GPU (fp64) vs CPU (fp64), real checkpoint; then the full fold target W.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

KFD = Path("/sys/class/kfd/kfd/proc")
OC = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C"
COD = f"{OC}/CARDAN_C0B/codice"


def host_ok():
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    def pswpin():
        return int(next(l.split()[1] for l in open("/proc/vmstat") if l.startswith("pswpin ")))
    a = pswpin(); time.sleep(2); b = pswpin()
    si = (b - a) * 4096 / 2
    return mem["MemAvailable"] / 1048576, si


def ammissione(max_kfd=2, attesa_max_s=6 * 3600):
    t0 = time.time()
    while True:
        n = len(list(KFD.iterdir()))
        mem, si = host_ok()
        if n <= max_kfd and mem >= 8 and si < 20e6:
            return {"kfd_prima": n, "mem_avail_gib": round(mem, 2), "swapin_Bps": si,
                    "pid_kfd": sorted(p.name for p in KFD.iterdir())}
        if time.time() - t0 > attesa_max_s:
            raise SystemExit(f"AMMISSIONE NEGATA dopo {attesa_max_s} s: kfd={n} mem={mem:.1f} si={si:.0f}")
        print(f"[attesa] kfd={n} mem={mem:.1f} GiB swap-in={si / 1e6:.1f} MB/s", flush=True)
        time.sleep(60)


def scrivi(out, nome, d):
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{nome}.tmp"
    tmp.write_text(json.dumps(d, indent=1, default=str))
    os.replace(tmp, out / nome)
    print(json.dumps(d, default=str)[:3000], flush=True)


def _vram_proprio():
    p = KFD / str(os.getpid())
    try:
        return max(int(f.read_text()) for f in p.glob("vram_*"))
    except (OSError, ValueError):
        return None


def d4(a, torch, np):
    sys.path.insert(0, COD)
    import cardan_piega_gpu as P
    import cardan_guardia_gpu as G
    import cardan_verifica_algebra_gpu as V
    dev = torch.device("cuda")
    gq = np.load(f"@@ROOT:models@@/gguf/tony-tern/pod-schur-esperti/Gq_{a.strato:02d}.npy", mmap_mode="r")
    res = {"strato": a.strato, "esperti": a.esperti, "n": int(gq.shape[-1])}
    diff = {"guardia": 0, "verifica_algebra": 0, "legacy_cardan_piega_gpu": 0, "trasferimento_ricopiati": 0}
    elementi = 0
    V.installa(P, normalizza=True, algebra=False, trasferimento=False)
    nuova = P._normalizza_gpu
    import importlib
    P2 = importlib.reload(P)                      # legacy variant (antagonist D4: 1 ulp on the diagonal mean)
    legacy = P2._normalizza_gpu
    t = {"cpu": 0.0, "gpu": 0.0}
    for e in range(a.esperti):
        g = np.asarray(gq[e], np.float64)
        t0 = time.time(); ref = P._normalizza(torch.from_numpy(g.copy())); t["cpu"] += time.time() - t0
        Hd = V.a_device_verificato(g, dev)
        torch.cuda.synchronize(); t0 = time.time()
        x1 = G.normalizza(Hd.clone()); torch.cuda.synchronize(); t["gpu"] += time.time() - t0
        x2 = nuova(Hd.clone())
        x3 = legacy(Hd.clone())
        r = ref.contiguous().view(torch.int64)
        for k, x in (("guardia", x1), ("verifica_algebra", x2), ("legacy_cardan_piega_gpu", x3)):
            diff[k] += int((x.cpu().contiguous().view(torch.int64) != r).sum())
        elementi += ref.numel()
    diff["trasferimento_ricopiati"] = V.CONTATORI["h_ricopiati"]
    res.update(elementi=elementi, elementi_diversi=diff, secondi=t, trasferimenti=V.CONTATORI["h_trasferimenti"])
    return res


def d1(a, torch, np):
    sys.path.insert(0, COD)
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import cardan_piega_gpu as P
    import cardan_verifica_algebra_gpu as V
    import comune as CM
    sys.path.insert(0, f"{OC}/ROCSOLVER_RIPARA_1020")
    import algebra_gpu as AG
    dev = torch.device("cuda")
    gq = np.load(f"@@ROOT:models@@/gguf/tony-tern/pod-schur-esperti/Gq_{a.strato:02d}.npy", mmap_mode="r")
    blocchi, ref = [], []
    for e in range(a.esperti):
        H = P._normalizza(torch.from_numpy(np.asarray(gq[e], np.float64)))
        Ha = CM.ruota_H(H) + 1e-4 * torch.eye(H.shape[0], dtype=torch.float64)
        for s in range(0, Ha.shape[0], 128):                   # Hgg = diagonal group blocks (fmt.group = 128)
            blocchi.append(Ha[s:s + 128, s:s + 128])
    M = torch.stack(blocchi)                                    # [E*16, 128, 128] fp64
    Lc, ic = torch.linalg.cholesky_ex(M)                        # CPU LAPACK fp64 = the original
    Md = M.to(dev)
    torch.cuda.synchronize(); t0 = time.time()
    Lg, ig = AG.chol_ex(Md)
    torch.cuda.synchronize(); t_gpu = time.time() - t0
    L1, i1 = V.controlla_fattori(Md, Lg, ig)
    base = dict(V.CONTATORI)
    # injected faults: a wrong-but-finite factor on 3 matrices, a false "not SPD" on 2 others
    Lbad, ibad = Lg.clone(), ig.clone()
    Lbad[[5, 77, 300]] *= 1.0001
    ibad[[11, 401]] = 1
    for k in V.CONTATORI:
        V.CONTATORI[k] = 0
    L2, i2 = V.controlla_fattori(Md, Lbad, ibad)
    iniettati = dict(V.CONTATORI)
    err_lapack = float((L1.cpu() - Lc).abs().max())
    err_iniettati = float((L2.cpu() - Lc).abs().max())
    # transfer: exact int64 bit-sum of the device copy vs the host array
    g = np.asarray(gq[0], np.float64)
    Hd = V.a_device_verificato(g, dev)
    return {"strato": a.strato, "esperti": a.esperti, "matrici": int(M.shape[0]), "ordine": 128,
            "gpu_info_non_zero": int((ig != 0).sum()), "lapack_info_non_zero": int((ic != 0).sum()),
            "controllo_senza_guasti": base, "max_abs_L_vs_lapack": err_lapack,
            "controllo_con_guasti_iniettati": iniettati, "max_abs_L_dopo_rifacimento_vs_lapack": err_iniettati,
            "secondi_chol_gpu": t_gpu, "trasferimento_ok": bool(V.somma_bit(Hd) == V.somma_bit(g))}


def d12(a, torch, np):
    dev = torch.device("cuda")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    st = ck["stato"].get(a.strato, ck["stato"].get(str(a.strato)))
    out = {"checkpoint": a.checkpoint, "strato": a.strato}
    for f in ("gate", "up"):
        D, C = st[f"D_{f}"].double(), st[f"C_{f}"].double()
        Dg = D.to(dev)
        diversi = tot = 0
        maxabs = 0.0
        t = {"cpu": 0.0, "gpu": 0.0}
        for e in range(C.shape[0]):
            t0 = time.time(); w_cpu = C[e] @ D.T; t["cpu"] += time.time() - t0
            torch.cuda.synchronize(); t0 = time.time()
            w_gpu = (C[e].to(dev) @ Dg.T); torch.cuda.synchronize(); t["gpu"] += time.time() - t0
            w_gpu = w_gpu.cpu()
            diversi += int((w_gpu.view(torch.int64) != w_cpu.contiguous().view(torch.int64)).sum())
            tot += w_cpu.numel()
            maxabs = max(maxabs, float((w_gpu - w_cpu).abs().max()))
        out[f] = {"esperti": int(C.shape[0]), "elementi": tot, "elementi_diversi": diversi,
                  "max_abs": maxabs, "secondi": t, "rango": int(D.shape[1]) if D.dim() == 2 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["d1", "d4", "d12"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--strato", type=int, default=16)
    ap.add_argument("--esperti", type=int, default=32)
    ap.add_argument("--checkpoint", default=f"{OC}/CARDAN_CONGIUNTO_S7T550_1020/allena/tappe/cardan_300.pt")
    a = ap.parse_args()
    out = Path(a.out)
    amm = ammissione()
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    torch.ones(1, device="cuda"); torch.cuda.synchronize()
    amm["kfd_durante"] = len(list(KFD.iterdir()))
    res = {"d1": d1, "d4": d4, "d12": d12}[a.modo](a, torch, np)
    torch.cuda.synchronize()
    res.update(modo=a.modo, ammissione=amm, wall_s=round(time.time() - t0, 2), vram_proprio_byte=_vram_proprio(),
               picco_torch_byte=torch.cuda.max_memory_allocated(), torch=torch.__version__,
               ora=time.strftime("%F %T"))
    scrivi(out, f"{a.modo}.json", res)


if __name__ == "__main__":
    main()
