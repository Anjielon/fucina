"""Costruisce UNA VOLTA la cache su disco dei 40 strati GGUF dequantizzati (24/9 20:1x,
mandato forgia < 2 ore): ~27 s/strato di numpy tolti da ogni forgia futura.
Un processo, un thread, priorita' minima: gira accanto alla GPU senza disturbarla."""
import os
import sys
import time

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("THOR_CACHE_STRATI", "@@ROOT:models@@/gguf/tony-tern/cache_strati")
from cattura_schur_esperti import GGUF, _pesi_gguf_con_cache  # noqa: E402

for L in range(40):
    t0 = time.time()
    sd = _pesi_gguf_con_cache(GGUF, L)
    print(f"[{time.strftime('%H:%M:%S')}] strato {L:2d}: {len(sd)} tensori in {time.time() - t0:.1f} s", flush=True)
    del sd
print("CACHE COMPLETA", flush=True)
