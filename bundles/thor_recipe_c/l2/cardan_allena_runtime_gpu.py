"""Adattamento opt-in dei trainer originali; nessuna modifica alla loss.

Non implementa DDP: replicare il trainer o dividerlo per strati cambierebbe
l'obiettivo. Cache mmap e anticipo CPU sono leve separate da collaudare.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import pianificatore_gpu as PG


def esegui(modulo, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seme", type=int, default=20261007)
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--device", default="cuda")
    noti, _ = ap.parse_known_args(argv)
    if os.environ.get("FORGIA_ALLENA_GPU") != "1":
        raise RuntimeError("attivazione richiesta: FORGIA_ALLENA_GPU=1")
    PG.guardia_corse()
    PG.guardia_host(*PG.misura_host())
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    if "--riprendi" in argv:
        raise ValueError("ripresa del nuovo runtime non ancora certificata")
    out = Path(noti.uscita)
    if out.exists():
        raise FileExistsError("il runtime nuovo richiede --uscita nuova")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 24 * PG.GIB:
        raise RuntimeError("meno di 24 GiB disponibili per checkpoint/cache/log")
    out.mkdir()
    temporanei = out / "_tmp"
    temporanei.mkdir()
    for nome, path in (("TMPDIR", temporanei), ("TRITON_CACHE_DIR", temporanei / "triton"),
                       ("TORCHINDUCTOR_CACHE_DIR", temporanei / "inductor")):
        valore = Path(os.environ.get(nome, str(path))).resolve()
        if valore == Path("/tmp") or Path("/tmp") in valore.parents:
            raise ValueError(f"{nome} sotto /tmp non ammesso")
        valore.mkdir(parents=True, exist_ok=True)
        os.environ[nome] = str(valore)
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # priorita' 0 (21/10): ogni worker inductor prende una coda SDMA
    # Importare il trainer seleziona lo stesso albero E2E_KERNEL_NUOVI.
    trainer = importlib.import_module(modulo)
    import torch
    if torch.device(noti.device).type != "cuda":
        raise ValueError("questo runtime richiede --device cuda[:N]")
    torch.cuda.set_device(torch.device(noti.device))
    A = importlib.import_module("e2e_allena")
    diagnostica = out / "runtime_gpu.jsonl"
    originale = A.Caricatore
    oggetti = []
    rng = random.Random(noti.seme)
    sostituzioni = []
    # Il registro globale appartiene alle corse storiche; questa corsa ne ha uno proprio.
    registri = [trainer]
    if hasattr(trainer, "KDP"):
        registri.append(trainer.KDP)
    for obj in registri:
        if hasattr(obj, "CICLI"):
            sostituzioni.append((obj, "CICLI", obj.CICLI))
            obj.CICLI = str(out / "cicli.tsv")
    if os.environ.get("FORGIA_CACHE_MMAP") == "1":
        if not os.environ.get("E2E_CACHE_DEQUANT"):
            raise ValueError("FORGIA_CACHE_MMAP richiede E2E_CACHE_DEQUANT gia' popolata")
        CD = importlib.import_module("e2e_cache_dequant")
        from cardan_cache_mmap_gpu import lettore
        sostituzioni.append((CD, "somma_piani", CD.somma_piani))
        CD.somma_piani = lettore(CD)

    class Anticipo(originale):
        def __init__(self, src, device):
            super().__init__(src, device)
            self.ordine = src.righe_previste("train")[:]
            rng.shuffle(self.ordine)
            self.successivo = dict(zip(self.ordine, self.ordine[1:] + self.ordine[:1]))
            self.prefetch = os.environ.get("FORGIA_ANTICIPO") == "1"
            oggetti.append(self)

        def prendi(self, lato, righe):
            k = (lato, tuple(righe))
            # Al massimo una prenotazione speculativa; non accumulare il corpus.
            for kk in list(self._fut):
                if kk != k:
                    futuro = self._fut.pop(kk)
                    if not futuro.cancel():
                        futuro.result()
            t0 = time.monotonic()
            lotto = super().prendi(lato, righe)
            with open(diagnostica, "a") as f:
                f.write(json.dumps({"lato": lato, "righe": righe,
                                    "caricamento_e_copia_s": time.monotonic() - t0}) + "\n")
            if self.prefetch and lato == "train" and len(righe) == 1 and righe[0] in self.successivo:
                self.prenota("train", [self.successivo[righe[0]]])
            return lotto

    A.Caricatore = Anticipo
    try:
        return trainer.main(argv)
    finally:
        for car in oggetti:
            car.chiudi()
        A.Caricatore = originale
        for obj, nome, val in sostituzioni:
            setattr(obj, nome, val)
