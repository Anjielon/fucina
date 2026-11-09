#!/usr/bin/env python3
"""Sonda: quanto l'impacchettamento A BLOCCHI in un THREAD dello stesso processo ferma il thread principale (GIL).

La misura del progettista (+0..5 % sul passo) veniva da un PROCESSO separato: niente GIL in comune. Con
THOR_EXPORT_INCREMENTALE=1 il pacchetto si fa in un thread DENTRO la forgia, e il thread principale — quello
che lancia i kernel sulla GPU — deve riprendere il GIL fra un'operazione numpy e l'altra. Qui: il thread
principale gira un ciclo di lavoro Python minuto (come il lancio dei kernel) e si registrano gli intervalli fra
un giro e l'altro, prima SENZA e poi CON un thread che fa esattamente il lavoro del pacchetto (carica i codici
veri di uno strato, impacchetta gate/up/down a blocchi di 32 con le funzioni vere, salva .npy + sha256).

    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 taskset -c 30,31 nice -n 19 \\
        python3 strumenti/sonda_gil_impacchettamento.py --codici @@ROOT:models@@/gguf/tony-tern/v2-conf6h-B1 --strato 1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "@@ROOT:forge@@/thor_v2")


def ciclo(durata: float, stop: "threading.Event | None" = None) -> list:
    """Il thread principale: giri di ~50 µs di Python puro; ritorna gli intervalli fra un giro e l'altro (s)."""
    gap, t_prec, fine = [], time.perf_counter(), time.perf_counter() + durata
    while True:
        x = 0
        for i in range(300):
            x += i * i
        t = time.perf_counter()
        gap.append(t - t_prec)
        t_prec = t
        if t > fine or (stop is not None and stop.is_set()):
            return gap


def riassunto(g: list) -> dict:
    s = sorted(g)
    return {"giri": len(g), "p50_us": s[len(s) // 2] * 1e6, "p99_us": s[int(len(s) * 0.99)] * 1e6,
            "max_ms": s[-1] * 1e3, "oltre_5ms": sum(1 for x in g if x > 5e-3),
            "tempo_in_intervalli_oltre_1ms_s": sum(x for x in g if x > 1e-3)}


def impacchetta(codici: Path, L: int, scratch: Path, esito: dict) -> None:
    import numpy as np
    from esporta_schur_esperti import impacchetta_famiglia
    from pack_tq1_0_standard import pack_tq1_0_da_d_q
    t0 = time.time()
    b = 32
    for fam, xD, xQ in (("gate", "Dg", "Qg"), ("up", "Du", "Qu"), ("down", "Dd", "Qd")):
        D, Q = np.load(codici / f"{xD}_{L:02d}.npy"), np.load(codici / f"{xQ}_{L:02d}.npy")
        if fam == "down":
            a = np.concatenate([pack_tq1_0_da_d_q(D[e:e + b], Q[e:e + b]) for e in range(0, Q.shape[0], b)],
                               axis=0).reshape(D.shape[0], D.shape[1], -1)
        else:
            a = np.concatenate([impacchetta_famiglia(D[e:e + b], Q[e:e + b]) for e in range(0, Q.shape[0], b)], axis=0)
        with open(os.devnull, "wb") as f:      # le stesse write() di np.save, senza file nuovi (niente /tmp)
            np.save(f, a)
        esito[fam] = hashlib.sha256(np.ascontiguousarray(a).view(np.uint8).reshape(-1)).hexdigest()[:16]
        del D, Q, a
    esito["secondi"] = time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codici", type=Path, required=True)
    ap.add_argument("--strato", type=int, default=1)
    ap.add_argument("--scratch", type=Path, default=None, help="(non usato: si scrive su /dev/null)")
    ap.add_argument("--processo", action="store_true", help="controprova: il lavoro in un processo separato")
    a = ap.parse_args()
    import numpy  # noqa: F401 — import fuori dalla misura
    import esporta_schur_esperti  # noqa: F401
    import pack_tq1_0_standard  # noqa: F401
    print(f"intervallo di cambio del GIL: {sys.getswitchinterval() * 1e3:.1f} ms", flush=True)
    senza = riassunto(ciclo(6.0))
    print("SENZA thread:", senza, flush=True)
    esito: dict = {}
    fatto = threading.Event()
    if a.processo:                          # controprova: lo STESSO lavoro in un PROCESSO (niente GIL in comune)
        import multiprocessing as mp
        coda = mp.get_context("fork").Queue()

        def nel_processo(q):
            e: dict = {}
            impacchetta(a.codici, a.strato, a.scratch, e)
            q.put(e)
        pr = mp.get_context("fork").Process(target=nel_processo, args=(coda,))

        def attendi_processo():
            esito.update(coda.get())
            pr.join()
            fatto.set()
        t0 = time.perf_counter()
        pr.start()
        threading.Thread(target=attendi_processo).start()
    else:
        def lavoro():
            try:
                impacchetta(a.codici, a.strato, a.scratch, esito)
            finally:
                fatto.set()
        th = threading.Thread(target=lavoro, name="impacchetta")
        t0 = time.perf_counter()
        th.start()
    g = ciclo(600.0, fatto)
    durata = time.perf_counter() - t0
    con = riassunto(g)
    con["durata_s"] = durata
    print(f"CON {'PROCESSO' if a.processo else 'thread'} (strato intero a blocchi di 32):", con, flush=True)
    print("pacchetto:", esito, flush=True)
    giri_attesi = senza["giri"] / 6.0 * durata
    print(f"giri del thread principale: {con['giri']} contro {giri_attesi:.0f} attesi senza thread "
          f"({100 * (1 - con['giri'] / giri_attesi):.1f} % in meno) · intervalli oltre 5 ms: {con['oltre_5ms']} "
          f"(senza: {senza['oltre_5ms']}) · intervallo massimo {con['max_ms']:.1f} ms (senza {senza['max_ms']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
