"""GGUF nuovo = R2B3m con alcuni tensori riscritti AL POSTO (stessi nomi, tipi, forme, byte): copia del file e
sovrascrittura dei soli intervalli dei tensori cambiati. Il risultato e' caricabile dal motore attuale per costruzione
(header, tabella e allineamenti identici al byte). Si verifica rileggendo ogni tensore scritto.

uso: patch_gguf.py --dir CARTELLA_TENSORI --out FILE.gguf [--base R2B3m]
     CARTELLA_TENSORI contiene <nome_tensore>.npy (uint8 [righe, byte_riga]) per ogni tensore da sostituire.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comune import OSPITE, gp, log, sha256_file  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=OSPITE)
    a = ap.parse_args(argv)
    gguf = gp._import_gguf(None)
    rd = gguf.GGUFReader(a.base, "r")
    tens = {t.name: t for t in rd.tensors}
    files = sorted(Path(a.dir).glob("*.npy"))
    assert files, f"nessun tensore in {a.dir}"
    piano = []
    for f in files:
        nome = f.name[:-4]
        assert nome in tens, f"{nome} non e' nel GGUF base"
        t = tens[nome]
        arr = np.load(f, mmap_mode="r")
        assert arr.dtype == np.uint8 and arr.nbytes == t.n_bytes, (nome, arr.dtype, arr.nbytes, t.n_bytes)
        piano.append((nome, int(t.data_offset), arr))
    out = Path(a.out)
    tmp = out.with_suffix(out.suffix + ".parziale")
    log(f"copio {a.base} -> {tmp}")
    shutil.copyfile(a.base, tmp)
    prov = {"base": a.base, "tensori": {}}
    with open(tmp, "r+b") as fh:
        for nome, off, arr in piano:
            fh.seek(off)
            b = np.ascontiguousarray(arr).tobytes()
            fh.write(b)
            prov["tensori"][nome] = {"sha256_byte": hashlib.sha256(b).hexdigest(), "byte": len(b)}
    # rilettura
    rd2 = gguf.GGUFReader(str(tmp), "r")
    t2 = {t.name: t for t in rd2.tensors}
    for nome, off, arr in piano:
        got = np.asarray(t2[nome].data).reshape(-1).view(np.uint8)
        assert hashlib.sha256(got.tobytes()).hexdigest() == prov["tensori"][nome]["sha256_byte"], nome
    assert os.path.getsize(tmp) == os.path.getsize(a.base)
    os.replace(tmp, out)
    prov["byte_file"] = os.path.getsize(out)
    prov["n_tensori"] = len(piano)
    Path(str(out) + ".provenienza.json").write_text(json.dumps(prov, indent=1))
    log(f"scritto {out} ({prov['byte_file']} B, {len(piano)} tensori sostituiti, riletti al byte)")


if __name__ == "__main__":
    main()
