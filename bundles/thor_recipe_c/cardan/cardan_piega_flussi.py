"""Ingresso storico multi-thread: GPU disabilitata per i blocchi del 21/10; solo CPU."""
from __future__ import annotations

import argparse
import copy
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))


class RegistroLock:
    def __init__(self, f):
        self.f, self.lock = f, threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)

    def flush(self):
        with self.lock:
            self.f.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=16)
    ap.add_argument("--esperti", type=int, default=None)
    ap.add_argument("--flussi", type=int, default=4)
    a = ap.parse_args(argv)
    if a.device.split(":", 1)[0] != "cpu":
        ap.error("AVVISO: piega GPU multi-thread/stream VIETATA (paletti 21/10 17:25). "
                 "Usare cardan_piega_gpu_atomica.py con pianificatore; "
                 "multi-flusso solo nel modello esperti_lotti.py, un thread e memoria pinned.")
    import torch
    import cardan_piega as P
    dev = torch.device(a.device)
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    os.environ.setdefault("LEVE_DT", "float64" if dt == torch.float64 else "float32")
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    lo, _, hi = a.strati.partition("-")
    strati = [L for L in range(int(lo), int(hi or lo) + 1) if L in d["stato"] or str(L) in d["stato"]]
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    todo = [L for L in strati
            if not all((td / f"blk.{L}.ffn_{f}_exps.weight.npy").exists() for f in ("gate", "up"))]
    # import preliminare nel thread principale: evita corse sull'import di comune
    sys.path.insert(0, P.RAFF)
    import comune  # noqa: F401
    t0 = time.time()
    with open(out / "registro.jsonl", "a") as f:
        reg = RegistroLock(f)

        def lavora(L):
            stato = copy.deepcopy(d["stato"].get(L, d["stato"].get(str(L))))
            P.piega_strato(L, stato, a.gguf, dev, dt, a.lotto, td, reg, esperti=a.esperti)

        with ThreadPoolExecutor(max_workers=max(1, a.flussi)) as ex:
            for fut in [ex.submit(lavora, L) for L in todo]:
                fut.result()
    P.log("FINE piega_flussi", out, f"flussi={a.flussi} parete={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
