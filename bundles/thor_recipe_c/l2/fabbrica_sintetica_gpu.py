"""Synthetic but EXACT GPU work unit for physical scheduler proofs on any CUDA/HIP device (D19, 22/10).

Each unit multiplies integer-valued fp64 matrices (entries in [-8, 8]) `ripetizioni` times, reducing modulo 997
between products. Every partial sum stays below 2**53, so the result is bit-exact on ANY device and in ANY
reduction order: the SHA-256 of the result is a device-independent check of correct execution.
The unit records which device/process ran it and when, so concurrency (several processes per GPU, several GPUs)
is visible in the published results, not inferred.

factory spec for pianificatore_gpu: "fabbrica_sintetica_gpu:crea"; config: {"n": 4096, "ripetizioni": 10, "fili": 1}
"""
from __future__ import annotations

import hashlib
import json
import os
import time


def riferimento(n, ripetizioni, seme, dev="cpu"):
    import torch
    g = torch.Generator().manual_seed(int(seme))
    A = torch.randint(-8, 9, (n, n), generator=g, dtype=torch.int64).double().to(dev)
    B = torch.randint(-8, 9, (n, n), generator=g, dtype=torch.int64).double().to(dev)
    C = A @ B
    for _ in range(ripetizioni):
        C = torch.remainder(C, 997) @ B
    if dev != "cpu":
        torch.cuda.synchronize(dev)
    return hashlib.sha256(C.cpu().numpy().tobytes()).hexdigest()


def crea(config, device):
    import torch
    from pianificatore_gpu import tipo_device
    dev = torch.device(f"cuda:{int(device)}")
    torch.cuda.set_device(dev)

    def lavora(dati, out):
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        sha = riferimento(config["n"], config["ripetizioni"], dati["seme"], dev)
        t1 = time.time()
        (out / "risultato.json").write_text(json.dumps({
            "seme": dati["seme"], "sha256": sha, "device": int(device), "pid": os.getpid(),
            "nome_gpu": torch.cuda.get_device_name(dev), "inizio": t0, "fine": t1}))
        return {"picco_gpu_byte": int(torch.cuda.max_memory_reserved(dev)), "device_tipo": tipo_device(torch, int(device))}
    return lavora
