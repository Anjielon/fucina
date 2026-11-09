#!/usr/bin/env python3
"""OBQ fibrato, scrittura: copia dell'OSPITE con i soli tensori accettati riscritti AL POSTO (stesso tipo, forma, offset,
byte: taglia identica al byte, velocita' identica per costruzione). Adattato da s23/s23_scrivi.py (stessa verifica).
Verifica fail-closed prima di rinominare: taglia = ospite; intestazione identica al byte; per ogni tensore stessi
nome/tipo/forma/offset/byte; non scelti identici al byte; scelti = byte voluti; trit in {-1,0,+1} e scale fp16 identiche
a quelle dell'ospite (la leva non tocca le scale); sha256; .verifica.json.
uso: fib_scrivi.py --ospite G --tensori DIR --uscita FILE.gguf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")


def sha(p, n=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if n is not None:
            h.update(f.read(n))
            return h.hexdigest()
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ospite", required=True)
    ap.add_argument("--tensori", required=True)
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args()
    from gguf import GGUFReader
    import comune as CM
    scelti = {f.name[:-4]: f for f in sorted(Path(a.tensori).glob("blk.*.npy"))}
    if not scelti:
        raise SystemExit("nessun tensore accettato: niente GGUF")
    rd = GGUFReader(a.ospite, "r")
    tens = {t.name: t for t in rd.tensors}
    piano = []
    for nome, f in scelti.items():
        t = tens[nome]
        nuovo = np.load(f)
        assert nuovo.dtype == np.uint8 and nuovo.nbytes == t.n_bytes, (nome, nuovo.dtype, nuovo.nbytes, t.n_bytes)
        piano.append((nome, int(t.data_offset), nuovo, int(t.tensor_type), int(t.shape[0])))
    out = Path(a.uscita)
    tmp = out.with_suffix(out.suffix + ".parziale")
    shutil.copyfile(a.ospite, tmp)
    with open(tmp, "r+b") as fh:
        for nome, off, nuovo, _, _ in piano:
            fh.seek(off)
            fh.write(np.ascontiguousarray(nuovo).tobytes())
    rd2 = GGUFReader(str(tmp), "r")
    t2 = {t.name: t for t in rd2.tensors}
    assert list(t2) == list(tens), "nomi/ordine dei tensori diversi"
    assert os.path.getsize(tmp) == os.path.getsize(a.ospite), "taglia diversa"
    o = min(int(t.data_offset) for t in rd.tensors)
    assert sha(tmp, o) == sha(a.ospite, o), "intestazione diversa"
    pn = {p[0]: p for p in piano}
    cambiati, fam = [], {}
    for n, x in tens.items():
        y = t2[n]
        assert (int(x.tensor_type), list(x.shape), int(x.data_offset), int(x.n_bytes)) == \
               (int(y.tensor_type), list(y.shape), int(y.data_offset), int(y.n_bytes)), n
        bx = np.asarray(x.data).reshape(-1).view(np.uint8)
        by = np.asarray(y.data).reshape(-1).view(np.uint8)
        if n not in pn:
            assert np.array_equal(bx, by), f"{n} cambiato ma non scelto"
            continue
        _, _, nuovo, tid, C = pn[n]
        assert np.array_equal(by, np.ascontiguousarray(nuovo).reshape(-1)), n
        rx = np.asarray(x.data); rx = rx.reshape(-1, rx.shape[-1])
        ry = np.asarray(y.data); ry = ry.reshape(-1, ry.shape[-1])
        tx, sx = CM.gp.decode_rows(np.ascontiguousarray(rx), tid, C)
        ty, sy = CM.gp.decode_rows(np.ascontiguousarray(ry), tid, C)
        assert np.isin(ty, (-1, 0, 1)).all(), n
        assert np.array_equal(sx.view(np.uint16), sy.view(np.uint16)), f"{n}: scale cambiate"
        if not np.array_equal(bx, by):
            cambiati.append(n)
            f = n.split(".")[2]
            fam[f] = fam.get(f, 0) + 1
    os.replace(tmp, out)
    s = sha(out)
    Path(str(out) + ".sha256").write_text(s + "\n")
    ver = {"ok": True, "ospite": a.ospite, "taglia": os.path.getsize(out), "sha256": s, "tensori_cambiati": len(cambiati),
           "per_famiglia": fam, "elenco": cambiati, "scale": "identiche all'ospite", "intestazione": "identica"}
    Path(str(out) + ".verifica.json").write_text(json.dumps(ver, indent=1) + "\n")
    print(f"ESITO OK {out} {ver['taglia']} B sha {s[:16]}; {len(cambiati)} tensori riscritti {fam}; scale, tipi, forme, "
          f"offset e intestazione identici all'ospite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
