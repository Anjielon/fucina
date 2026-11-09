#!/usr/bin/env python3
"""Una sola patch esperti+densi, streaming CPU, GGUF pubblicato atomicamente.

Il suffisso indica l'integrazione nella pipeline GPU, non una copia disco
spacciata per calcolo GPU. Verifica TUTTI i byte fuori dagli intervalli patchati.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import pianificatore_gpu as PG

AMMESSI = re.compile(
    r"blk\.(?:[0-9]|[12][0-9]|3[0-9])\."
    r"(?:ffn_(?:gate|up)_exps2?|attn_(?:q|k|v|output|qkv|gate)[2-4]?|ssm_out[2-4]?)\.weight")


def verifica_fuori(base, nuovo, intervalli, pezzo=1 << 20):
    """Confronto binario comprende header, padding e tutti i tensori non toccati."""
    n = Path(base).stat().st_size
    if Path(nuovo).stat().st_size != n:
        raise ValueError("taglia GGUF diversa")
    pos, controllati = 0, 0
    with open(base, "rb") as a, open(nuovo, "rb") as b:
        for inizio, fine in [*intervalli, (n, n)]:
            if inizio < pos or fine < inizio or fine > n:
                raise ValueError("intervalli sovrapposti o fuori file")
            a.seek(pos)
            b.seek(pos)
            while pos < inizio:
                q = min(pezzo, inizio - pos)
                x, y = a.read(q), b.read(q)
                if len(x) != q or len(y) != q or x != y:
                    raise ValueError(f"byte fuori intervento diverso/troncato: offset {pos}")
                controllati += q
                pos += q
            pos = fine
    return controllati


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tensori", required=True, action="append",
                    help="ripetibile: directory pubblicato/tensori di esperti e/o densi")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nome", default="Thor-T-C-GPU.gguf")
    a = ap.parse_args(argv)
    if Path(a.nome).name != a.nome or not a.nome.endswith(".gguf"):
        ap.error("--nome deve essere un basename .gguf")
    PG.guardia_corse()
    PG.guardia_host(*PG.misura_host())
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    import numpy as np
    sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
    from gguf import GGUFReader
    base = Path(a.base).resolve(strict=True)
    files = [p for d in a.tensori for p in sorted(Path(d).glob("*.npy"))]
    if not files or any(not list(Path(d).glob("*.npy")) for d in a.tensori):
        raise ValueError("directory tensori vuota")
    nomi = [p.name[:-4] for p in files]
    if len(set(nomi)) != len(nomi):
        raise ValueError("tensori duplicati/sovrapposti: non combinare due correzioni della stessa famiglia")
    identita = [PG.identita_file(p) for p in [base, *files]]
    unita = PG.Unita("gguf", {"nome": a.nome, "input": identita})
    firma = PG.firma_json({"unita": unita.dati, "codice": PG.sha_file(__file__)})
    out = Path(a.out_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with PG.lucchetto(out.parent / f".{out.name}.lock"):
        if PG.completa(out, firma, unita):
            print(f"GIA' VERIFICATO {out / a.nome}", flush=True)
            return 0
        if shutil.disk_usage(out.parent).free < base.stat().st_size + PG.GIB:
            raise RuntimeError("disco insufficiente per GGUF nuovo + 1 GiB")
        rd = GGUFReader(str(base), "r")
        tens = {t.name: t for t in rd.tensors}
        piano = []
        for nome, f in zip(nomi, files):
            if not AMMESSI.fullmatch(nome) or nome not in tens:
                raise ValueError(f"tensore fuori famiglia o inesistente: {nome}")
            t = tens[nome]
            arr = np.load(f, mmap_mode="r", allow_pickle=False)
            if arr.dtype != np.uint8 or arr.nbytes != t.n_bytes:
                raise ValueError(f"dtype/taglia diverso dall'ospite: {nome}")
            piano.append((int(t.data_offset), int(t.data_offset) + int(t.n_bytes), nome, arr))
        piano.sort(key=lambda p: p[0])
        stage = out.parent / f".{out.name}.parziale-{uuid.uuid4().hex}"
        stage.mkdir()
        g = stage / a.nome
        t0 = time.monotonic()
        shutil.copyfile(base, g)
        copia = time.monotonic() - t0
        prov = {"base": str(base), "tensori": {}}
        t0 = time.monotonic()
        with open(g, "r+b") as f:
            for inizio, fine, nome, arr in piano:
                f.seek(inizio)
                b = memoryview(np.ascontiguousarray(arr).reshape(-1))
                h = hashlib.sha256()
                for i in range(0, len(b), 1 << 20):
                    q = b[i:i + (1 << 20)]
                    if f.write(q) != len(q):
                        raise OSError(f"scrittura incompleta: {nome}")
                    h.update(q)
                prov["tensori"][nome] = {"sha256_byte": h.hexdigest(), "byte": fine - inizio}
            f.flush()
            os.fsync(f.fileno())
        scrittura = time.monotonic() - t0
        t0 = time.monotonic()
        fuori = verifica_fuori(base, g, [(p[0], p[1]) for p in piano])
        with open(g, "rb") as f:
            for inizio, fine, nome, _ in piano:
                f.seek(inizio)
                h = hashlib.sha256()
                restano = fine - inizio
                while restano:
                    b = f.read(min(restano, 1 << 20))
                    if not b:
                        raise EOFError(f"rilettura GGUF troncata: {nome}")
                    h.update(b)
                    restano -= len(b)
                if h.hexdigest() != prov["tensori"][nome]["sha256_byte"]:
                    raise ValueError(f"SHA patch diverso: {nome}")
        for meta in identita:
            if PG.identita_file(meta["path"]) != meta:
                raise RuntimeError(f"input modificato durante patch: {meta['path']}")
        verifica = time.monotonic() - t0
        prov.update(byte_file=g.stat().st_size, n_tensori=len(piano),
                    byte_fuori_verificati=fuori, copia_s=copia, scrittura_s=scrittura, verifica_s=verifica)
        PG.scrivi_json(stage / "provenienza.json", prov)
        PG.pubblica_unita(stage, out, firma, unita, {"copia_s": copia, "scrittura_s": scrittura,
                                                   "verifica_s": verifica})
        print(json.dumps(prov, indent=1), flush=True)
        print(f"GGUF VERIFICATO {out / a.nome}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
