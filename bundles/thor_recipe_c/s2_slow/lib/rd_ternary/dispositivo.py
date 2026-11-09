"""Scelta del dispositivo e del tetto di memoria per lotto, senza dipendenze CUDA-only.

Su ROCm/HIP torch espone la GPU come dispositivo "cuda" (torch.version.hip valorizzato):
lo stesso codice gira su AMD (gfx1151, 96 GiB UMA), NVIDIA e CPU. Niente estensioni
compilate, niente triton: solo operatori torch standard.
"""
from __future__ import annotations

from typing import Optional

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"dispositivo {spec} richiesto ma torch non vede GPU "
                           f"(torch {torch.__version__}, hip={getattr(torch.version, 'hip', None)})")
    return dev


def backend_name(dev: torch.device) -> str:
    if dev.type != "cuda":
        return "cpu"
    return f"rocm/hip {torch.version.hip}" if getattr(torch.version, "hip", None) else f"cuda {torch.version.cuda}"


def _mem_available_cpu() -> Optional[int]:
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) * 1024
    except OSError:
        return None
    return None


def batch_memory_budget(dev: torch.device, frac: float = 0.5) -> Optional[int]:
    """Byte da dedicare a UN lotto: frazione della memoria libera del dispositivo
    (GPU: mem_get_info; CPU: MemAvailable). None se non misurabile."""
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free * frac)
    avail = _mem_available_cpu()
    return int(avail * frac) if avail else None


def synchronize(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
