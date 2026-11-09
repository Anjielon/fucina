#!/usr/bin/env python3
"""cardan_congiunto_veloce_gpu.py — cardan_congiunto.py ORIGINALE (stessa main, stessa perdita, stessi argomenti,
anche --riprendi) con le leve del passo attivabili una per una da variabili d'ambiente:

  FORGIA_SEGMENTI_LOTTO=1     proiezioni del modo segmenti in un solo lancio per riga (cardan_segmenti_gpu.py)
  FORGIA_SEGMENTI_LOTTO=riempito  in piu' conv + GDN su lotti di segmenti con lo stesso numero di chunk
  FORGIA_RICOMPILA_LIMITE=N   torch._dynamo recompile_limit = N (default di torch 8: oltre, _prep di gdn_veloce
                              ricade in eager per le lunghezze di segmento nuove)
  FORGIA_CACHE_MMAP=1         cache dequant (E2E_CACHE_DEQUANT) letta per mmap, SHA verificato (cardan_cache_mmap_gpu)
  FORGIA_CACHE_COMPILA=DIR    cache persistente di Triton/Inductor (non /tmp): niente ricompilazioni a ogni avvio

Nessuna variabile = cardan_congiunto.py identico. Ogni leva e' collaudata in profila_strati_gpu.py / sui primi passi
(FUCINA_GPU_INVENTARIO_1021.md).
"""
from __future__ import annotations

import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # priorita' 0 (21/10): ogni worker inductor apre /dev/kfd e prende una coda SDMA
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))


def prepara(log=print) -> dict:
    attive = {}
    d = os.environ.get("FORGIA_CACHE_COMPILA", "").strip()
    if d:
        p = Path(d).resolve()
        if p == Path("/tmp") or Path("/tmp") in p.parents:
            raise ValueError("FORGIA_CACHE_COMPILA sotto /tmp non ammessa")
        (p / "triton").mkdir(parents=True, exist_ok=True); (p / "inductor").mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(p / "triton")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(p / "inductor")
        attive["cache_compila"] = str(p)
    import torch
    n = os.environ.get("FORGIA_RICOMPILA_LIMITE", "").strip()
    if n:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = int(n)
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                torch._dynamo.config.accumulated_recompile_limit, 4 * int(n))
        attive["ricompila_limite"] = int(n)
    if os.environ.get("FORGIA_CACHE_MMAP") == "1":
        # output.weight dequantizzata letta per mmap (pagine del file, recuperabili senza swap) invece di np.load
        # in memoria anonima; chiave, versione e SHA dell'array verificati come l'originale.
        if not os.environ.get("E2E_CACHE_DEQUANT"):
            raise ValueError("FORGIA_CACHE_MMAP richiede E2E_CACHE_DEQUANT gia' popolata")
        import cardan_kd_paper  # noqa: F401
        import e2e_cache_dequant as CD
        from cardan_cache_mmap_gpu import lettore
        CD.somma_piani = lettore(CD)
        attive["cache_mmap"] = True
    sl = os.environ.get("FORGIA_SEGMENTI_LOTTO", "").strip()
    if sl in ("1", "lotto", "riempito"):
        import cardan_kd_paper  # noqa: F401  percorsi (REC / E2E_KERNEL_NUOVI) come il trainer
        import e2e_studente as S
        import cardan_segmenti_gpu as SG
        SG.imposta(S, "riempito" if sl == "riempito" else "lotto")
        attive["segmenti"] = "riempito" if sl == "riempito" else "lotto"
    elif sl:
        raise ValueError(f"FORGIA_SEGMENTI_LOTTO sconosciuto: {sl}")
    log(f"leve passo GPU attive: {attive or 'nessuna'}")
    return attive


if __name__ == "__main__":
    prepara()
    import cardan_congiunto as CG
    sys.exit(CG.main())
