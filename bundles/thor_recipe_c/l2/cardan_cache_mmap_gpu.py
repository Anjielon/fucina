"""Lettura della cache dequant esistente tramite mmap COW, SHA sempre verificato.

Non cambia chiave, dequantizzazione o byte. Cache mancante/corrotta = errore
esplicito: questo lettore non sovrascrive mai una cache di una corsa viva.
"""
from __future__ import annotations

import json
from pathlib import Path


def lettore(CD):
    def somma_piani(tens, nome_g, piani, cartella, calcola, log=print):
        import numpy as np
        k = CD.chiave(tens, piani)
        base = Path(cartella) / f"{nome_g}-{k[:24]}"
        pn = base.with_name(base.name + ".npy")
        pj = base.with_name(base.name + ".json")
        if not pn.is_file() or not pj.is_file():
            raise FileNotFoundError(f"cache mmap assente: {pn}; prepararla col percorso originale "
                                    "in una cartella nuova, a corse terminate")
        meta = json.loads(pj.read_text())
        if meta["chiave"] != k or meta["versione"] != CD.VERSIONE:
            raise ValueError(f"chiave/versione cache diversa: {pj}")
        a = np.load(pn, mmap_mode="c", allow_pickle=False)
        if not a.flags.c_contiguous:
            raise ValueError(f"cache non C-contigua: SHA richiederebbe una copia eager, rifiuto {pn}")
        if a.dtype.str != meta["dtype"] or list(a.shape) != meta["forma"]:
            raise ValueError(f"dtype/forma cache diversi: {pn}")
        if CD.sha_array(a) != meta["sha256_array"]:
            raise ValueError(f"SHA cache diverso: {pn}; non ricalcolo/sovrascrivo")
        log(f"cache dequant mmap COW: {nome_g}, {a.nbytes / 2**30:.2f} GiB, SHA al bit")
        return a
    return somma_piani
