"""Input persistenti opt-in dei densi; nessuna modifica numerica o cache disco.

Un mmap GGUF per worker, solo metadati/byte compressi condivisi dal kernel.
Le Hessiane normalizzate dal riferimento restano CPU; cache LRU limitata
in byte, copie private per impedire modifiche da parte del consumatore.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import pianificatore_gpu as PG


class InputDensi:
    def __init__(self, config, delta, manif, CM, CD):
        limite = config["cache_h_byte"]
        if type(limite) is not int or limite < 0:
            raise ValueError("cache_h_byte deve essere un intero >= 0")
        strati = config["strati"]
        nomi = [n for n in delta if int(n.split(".")[1]) in strati]
        if not nomi:
            raise ValueError("nessun delta nei --strati selezionati")
        self.config, self.manif, self.CM, self.CD = config, manif, CM, CD
        self.nomi = set(nomi)
        self.basi = sorted({n.split(".")[-2] for n in nomi})
        self.sorgente = None
        self.identita_gguf = PG.identita_file(config["gguf"])
        self.cache = OrderedDict()
        self.limite, self.byte_cache = limite, 0
        self.identita_h = {}
        self.misure = {
            "gguf_aperture": 0, "gguf_apertura_s": 0.0,
            "h_caricamenti": 0, "h_hit": 0, "h_caricamento_s": 0.0,
            "h_copia_s": 0.0, "decode_s": 0.0}

    def denso(self, nome):
        if nome not in self.nomi:
            raise ValueError(f"tensore fuori piano: {nome}")
        if PG.identita_file(self.config["gguf"]) != self.identita_gguf:
            raise RuntimeError("GGUF cambiato durante il riuso degli input")
        if self.sorgente is None:
            t0 = time.monotonic()
            self.sorgente = self.CM.gp.GGUFPlaneSource(
                self.config["gguf"], layers=self.config["strati"], bases=self.basi)
            self.misure["gguf_aperture"] += 1
            self.misure["gguf_apertura_s"] += time.monotonic() - t0
        t0 = time.monotonic()
        d = self.CM.Denso(self.sorgente, nome)
        self.misure["decode_s"] += time.monotonic() - t0
        if PG.identita_file(self.config["gguf"]) != self.identita_gguf:
            raise RuntimeError("GGUF cambiato durante la lettura del tensore")
        return d

    def hessiana(self, nome):
        if nome not in self.nomi:
            raise ValueError(f"tensore fuori piano: {nome}")
        info = self.manif.get(nome.replace(".weight", "3.weight"), {}).get("hessiana")
        if not info:
            raise KeyError(f"{nome}: no Hessian in the R2B3 manifest")
        path = str(Path(info["file"]).resolve(strict=True))
        identita = PG.identita_file(path)
        if path in self.identita_h and self.identita_h[path] != identita:
            raise RuntimeError(f"Hessiana cambiata durante il riuso: {path}")
        self.identita_h[path] = identita
        if path in self.cache:
            h = self.cache.pop(path)
            self.cache[path] = h
            self.misure["h_hit"] += 1
        else:
            t0 = time.monotonic()
            h = self.CD.hessiana(nome, self.manif)
            self.misure["h_caricamento_s"] += time.monotonic() - t0
            self.misure["h_caricamenti"] += 1
            if PG.identita_file(path) != identita:
                raise RuntimeError(f"Hessiana cambiata durante la lettura: {path}")
            byte = h.numel() * h.element_size()
            if byte <= self.limite:
                while self.cache and self.byte_cache + byte > self.limite:
                    _, vecchia = self.cache.popitem(last=False)
                    self.byte_cache -= vecchia.numel() * vecchia.element_size()
                self.cache[path] = h
                self.byte_cache += byte
            else:
                return h
        t0 = time.monotonic()
        copia = h.clone()
        self.misure["h_copia_s"] += time.monotonic() - t0
        return copia

    def statistiche(self):
        return {**self.misure, "cache_h_residente_byte": self.byte_cache,
                "cache_h_limite_byte": self.limite}
