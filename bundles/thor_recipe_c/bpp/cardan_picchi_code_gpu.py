"""Picchi KFD osservati nel lavoratore e discendenti, senza importare torch."""
from __future__ import annotations

import os
import threading

import cardan_code_gpu as CQ


class PicchiCode:
    def __init__(self, device, intervallo=0.1):
        if not 0 < intervallo <= 1:
            raise ValueError("intervallo di campionamento code fuori (0, 1]")
        self.gid, _ = CQ.gpu_id_di(device)
        self.pid = os.getpid()
        self.intervallo = intervallo
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.errore = None
        self.picchi = {"code_compute": 0, "code_sdma": 0, "code_processi_kfd": 0,
                       "code_campioni": 0}
        self.thread = threading.Thread(target=self._ciclo, name="misura-code-kfd", daemon=True)

    def campiona(self):
        _, processi = CQ.uso(pids=CQ._albero(self.pid))
        compute, sdma, n = 0, 0, 0
        for code in processi.values():
            if any(g != self.gid and any(v.values()) for g, v in code.items()):
                raise RuntimeError("lavoratore usa code su un device diverso da quello assegnato")
            if self.gid in code:
                compute += code[self.gid]["compute"]
                sdma += code[self.gid]["sdma"]
                n += 1
        with self.lock:
            for nome, valore in (("code_compute", compute), ("code_sdma", sdma), ("code_processi_kfd", n)):
                self.picchi[nome] = max(self.picchi[nome], valore)
            self.picchi["code_campioni"] += 1

    def _ciclo(self):
        try:
            while not self.stop.wait(self.intervallo):
                self.campiona()
        except (OSError, ValueError, RuntimeError) as e:
            self.errore = e

    def avvia(self):
        self.campiona()
        self.thread.start()

    def misure(self):
        if self.errore is not None:
            raise RuntimeError("telemetria code del lavoratore fallita") from self.errore
        self.campiona()
        with self.lock:
            return {**self.picchi, "code_gpu_id": self.gid,
                    "code_intervallo_s": self.intervallo,
                    "code_metodo": "massimi campionati dal caricamento; lavoratore e discendenti"}

    def chiudi(self):
        self.stop.set()
        self.thread.join()
