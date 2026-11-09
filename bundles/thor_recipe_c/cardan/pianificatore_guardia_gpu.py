#!/usr/bin/env python3
"""Esecutore additivo D7/D8: drain su guardia, figli legati al coordinatore."""
from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import errno
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback
import uuid

import pianificatore_gpu as PG


class ErroreLavoratore(RuntimeError):
    pass


def lega_al_padre(pid):
    """Solo i nuovi figli: SIGTERM dal kernel se muore il loro coordinatore."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        codice = ctypes.get_errno()
        raise OSError(codice, os.strerror(codice))
    if os.getppid() != pid:
        raise ProcessLookupError(errno.ESRCH, "coordinatore gia' terminato")


def _lavoratore(factory, config, device, root, firma, ingresso, uscita, fili, padre):
    lega_al_padre(padre)
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.environ.update(OMP_NUM_THREADS=str(fili), MKL_NUM_THREADS=str(fili),
                      LEVE_THREADS=str(fili), PYTHONDONTWRITEBYTECODE="1",
                      TORCHINDUCTOR_COMPILE_THREADS="1")  # (d) 21/10: ogni worker inductor prende una coda SDMA
    pid = os.getpid()
    with open(Path(root) / f"lavoratore-{device}-{pid}.log", "x", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = None
            try:
                if config.get("pianificazione", {}).get("gpu_reale"):
                    import cardan_code_gpu as CQ
                    if (CQ.KFD / "topology").is_dir():
                        from cardan_picchi_code_gpu import PicchiCode
                        code = PicchiCode(device)
                        code.avvia()
                for p, sha in config.get("codici", {}).items():
                    if PG.sha_file(p) != sha:
                        raise RuntimeError(f"codice cambiato prima dell'avvio: {p}")
                handler = PG.risolvi_factory(factory)(config, device)
                uscita.send(("pronto", pid, None))
                while os.getppid() == padre:
                    if not ingresso.poll(1):
                        continue
                    u = ingresso.recv()
                    if u is None:
                        return
                    for meta in config.get("ingressi", []):
                        if PG.identita_file(meta["path"]) != meta:
                            raise RuntimeError(f"input cambiato prima dell'unita': {meta['path']}")
                    for p, sha in config.get("codici", {}).items():
                        if PG.sha_file(p) != sha:
                            raise RuntimeError(f"codice cambiato prima dell'unita': {p}")
                    stage = Path(root) / f".lavoro-{u.chiave}-{uuid.uuid4().hex}"
                    stage.mkdir()
                    t0 = time.monotonic()
                    metriche = handler(u.dati, stage)
                    misure = {"secondi": time.monotonic() - t0,
                               "picco_rss_byte": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                               **metriche}
                    if code is not None:
                        misure.update(code.misure())
                    PG.pubblica_unita(stage, Path(root) / "unita" / u.chiave, firma, u, misure)
                    uscita.send(("fatta", pid, {"chiave": u.chiave, **misure}))
            except BaseException:
                errore = traceback.format_exc()
                print(errore, flush=True)
                uscita.send(("errore", pid, errore))
                raise
            finally:
                if code is not None and code.thread.is_alive():
                    code.chiudi()


def _esegui_prima_obiettivo(unita, factory, config, root, firma, posti, budget):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if len({u.chiave for u in unita}) != len(unita):
        raise ValueError("unita' duplicate")
    (root / "unita").mkdir(exist_ok=True)
    piano = {"firma": firma, "unita": [dataclasses.asdict(u) for u in unita],
             "factory": factory, "config": config}
    meta = root / "piano.json"
    if meta.exists():
        if json.loads(meta.read_text()) != piano:
            raise RuntimeError("piano/config/input diversi; scegliere una nuova --out")
    else:
        if any((root / "unita").iterdir()):
            raise RuntimeError("unita' senza piano: non le adotto")
        PG.scrivi_json(meta, piano)
    pendenti = [u for u in unita if not PG.completa(root / "unita" / u.chiave, firma, u)]
    if not pendenti:
        return
    if shutil.disk_usage(root).free < budget.disco * len(pendenti) + PG.GIB:
        raise RuntimeError("spazio insufficiente per risultati + 1 GiB")
    if sum(posti.values()) < 1:
        raise ValueError("nessun posto per lavoratori")
    ctx = mp.get_context("spawn")
    risultati = {}
    figli, code, attivi, ritirati = {}, {}, {}, set()
    errore = None
    indice = 0

    def blocca(e):
        nonlocal errore
        if errore is None:
            errore = e
            print(json.dumps({"guardia": "DRAIN", "errore": str(e),
                              "in_volo": [u.chiave for u in attivi.values()],
                              "azione": "nessuna nuova assegnazione; finire le unita' attive"}),
                  flush=True)

    def guardia():
        PG.guardia_corse()
        PG.guardia_host(*PG.misura_host())

    try:
        for dev, numero in posti.items():
            for _ in range(min(numero, len(pendenti) - len(figli))):
                guardia()
                ingresso, q = ctx.Pipe(duplex=False)
                risultato, uscita = ctx.Pipe(duplex=False)
                p = ctx.Process(target=_lavoratore, args=(
                    factory, config, dev, str(root), firma, ingresso, uscita, budget.fili, os.getpid()))
                p.start()
                ingresso.close()
                uscita.close()
                figli[p.pid], code[p.pid] = p, q
                risultati[p.pid] = risultato
                while True:
                    if risultato.poll(.2):
                        try:
                            stato, pid, msg = risultato.recv()
                        except EOFError as e:
                            raise ErroreLavoratore(f"lavoratore {p.pid}: pipe chiusa all'avvio") from e
                        if stato != "pronto" or pid != p.pid:
                            raise ErroreLavoratore(f"avvio lavoratore {p.pid}: {msg}")
                        break
                    if p.exitcode is not None:
                        raise ErroreLavoratore(f"lavoratore {p.pid} morto all'avvio: {p.exitcode}")
        while True:
            if errore is None:
                for pid in figli:
                    if pid in attivi or pid in ritirati or indice == len(pendenti):
                        continue
                    try:
                        guardia()
                    except RuntimeError as e:
                        blocca(e)
                        break
                    attivi[pid] = pendenti[indice]
                    indice += 1
                    try:
                        code[pid].send(attivi[pid])
                    except BrokenPipeError:
                        blocca(ErroreLavoratore(f"lavoratore {pid}: pipe chiusa durante assegnazione"))
                        del attivi[pid]
                        ritirati.add(pid)
                        break
            if not attivi:
                break
            pronti = wait([risultati[pid] for pid in attivi], timeout=.2)
            if not pronti:
                for pid in list(attivi):
                    if figli[pid].exitcode is not None:
                        blocca(ErroreLavoratore(
                            f"lavoratore {pid} morto: {figli[pid].exitcode}; unita' {attivi[pid].chiave}"))
                        del attivi[pid]
                        ritirati.add(pid)
                continue
            conn = pronti[0]
            try:
                stato, pid, msg = conn.recv()
            except EOFError:
                pid = next(pid for pid in attivi if risultati[pid] is conn)
                blocca(ErroreLavoratore(f"lavoratore {pid}: pipe chiusa; unita' {attivi[pid].chiave}"))
                del attivi[pid]
                ritirati.add(pid)
                continue
            if stato == "errore":
                blocca(ErroreLavoratore(f"lavoratore {pid}: {msg}"))
                attivi.pop(pid, None)
                ritirati.add(pid)
                continue
            if stato != "fatta" or pid not in attivi or msg["chiave"] != attivi[pid].chiave:
                raise RuntimeError("risultato senza assegnazione coerente")
            u = attivi.pop(pid)
            if not PG.completa(root / "unita" / u.chiave, firma, u):
                raise RuntimeError(f"worker dichiara completamento assente: {u.chiave}")
            print(json.dumps({"pid": pid, **msg}), flush=True)
            if PG.picco_oltre_budget(msg, budget):
                blocca(RuntimeError("picco oltre budget: profilare di nuovo, risultati conservati"))
        if errore is not None:
            raise errore
    finally:
        # Nel drain attivi e' vuoto. Interruzioni esterne fermano solo questi figli.
        for pid, p in figli.items():
            if p.is_alive():
                try:
                    code[pid].send(None)
                except BrokenPipeError:
                    print(json.dumps({"guardia": "FIGLIO_GIA_TERMINATO", "pid": pid}), flush=True)
        for p in figli.values():
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                raise RuntimeError(f"figlio {p.pid} non terminato dopo SIGTERM")
        for q in code.values():
            q.close()
        for conn in risultati.values():
            conn.close()


def esegui(unita, factory, config, root, firma, posti, budget):
    from pianificatore_misurato_gpu import esegui as misurato
    return misurato(unita, factory, config, root, firma, posti, budget)
