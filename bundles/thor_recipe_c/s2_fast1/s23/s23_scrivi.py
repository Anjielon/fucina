#!/usr/bin/env python3
"""S2+S3 tappa CPU: scrittura di un braccio = copia della PARTENZA con i tensori scelti riscritti AL POSTO (stesso tipo,
stessa forma, stessi byte: taglia identica al byte, tg identica per costruzione).

Bracci: s2 = solo i DENSI della passata (C-S2) · s3 = densi + esperti (C-S3). Nota dichiarata: nella passata unica i densi
dello strato L+1 sono stati risolti sul flusso con gli esperti degli strati <= L GIA' nuovi (piano §1.2: «la stessa passata,
con due GGUF in uscita»): C-S2 isola i densi, ma non e' la passata che si otterrebbe senza S3.
Verifica (fail-closed, prima di rinominare): taglia = partenza al byte; intestazione identica al byte; per ogni tensore stessi
nome/tipo/forma/offset/byte; non scelti identici; scelti = byte voluti; trit di ogni tensore riscritto in {-1,0,+1} e scale
fp16 finite (decodifica completa); elenco; sha256.
uso: s23_scrivi.py --partenza G --strati DIR --braccio s2|s3 --uscita FILE.gguf
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

QUI = Path(__file__).resolve().parent
for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b")):
    if p not in sys.path:
        sys.path.insert(0, p)
import s1b_comune as K  # noqa: E402

ESPERTI = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def sha(p, n=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if n is not None:
            h.update(f.read(n))
            return h.hexdigest()
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def raccogli(d: Path, braccio: str, n_strati: int):
    out = {}
    fatti = sorted(d.glob("strato_*"))
    if len([s for s in fatti if (s / "FATTO").exists()]) != n_strati:
        raise SystemExit(f"strati finiti {len([s for s in fatti if (s / 'FATTO').exists()])} != {n_strati}")
    for st in fatti:
        for f in (st / "tensori").glob("blk.*.npy"):
            nome = f.name[:-4]
            if braccio == "s2" and nome.split(".")[2] in ESPERTI:
                continue
            out[nome] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partenza", required=True)
    ap.add_argument("--strati", required=True)
    ap.add_argument("--n-strati", type=int, default=40)
    ap.add_argument("--braccio", choices=("s2", "s3"), required=True)
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args()
    scelti = raccogli(Path(a.strati), a.braccio, a.n_strati)
    if not scelti:
        raise SystemExit("nessun tensore da cambiare")
    gguf = K.gp._import_gguf(None)
    rd = gguf.GGUFReader(a.partenza, "r")
    tens = {t.name: t for t in rd.tensors}
    piano = []
    for nome, f in sorted(scelti.items()):
        t = tens[nome]
        tid = int(t.tensor_type)
        assert tid in K.FORMATO, (nome, tid)
        nuovo = np.load(f)
        raw = np.asarray(t.data)
        assert nuovo.dtype == np.uint8 and nuovo.nbytes == t.n_bytes, (nome, nuovo.dtype, nuovo.nbytes, t.n_bytes)
        assert nuovo.reshape(-1).size == raw.reshape(-1).view(np.uint8).size, nome
        piano.append((nome, int(t.data_offset), nuovo, tid, int(t.shape[0])))
    out = Path(a.uscita)
    tmp = out.with_suffix(out.suffix + ".parziale")
    shutil.copyfile(a.partenza, tmp)
    with open(tmp, "r+b") as fh:
        for nome, off, nuovo, _, _ in piano:
            fh.seek(off)
            fh.write(np.ascontiguousarray(nuovo).tobytes())
    rd2 = gguf.GGUFReader(str(tmp), "r")
    t2 = {t.name: t for t in rd2.tensors}
    assert list(t2) == list(tens), "nomi/ordine dei tensori diversi"
    assert os.path.getsize(tmp) == os.path.getsize(a.partenza), "taglia diversa"
    o = min(int(t.data_offset) for t in rd.tensors)
    assert sha(tmp, o) == sha(a.partenza, o), "intestazione diversa"
    pn = {p[0]: p for p in piano}
    cambiati, tipi = [], {}
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
        rows = np.asarray(y.data)
        rows = rows.reshape(-1, rows.shape[-1])
        tr, sc = K.gp.decode_rows(np.ascontiguousarray(rows), tid, C)
        assert np.isin(tr, (-1, 0, 1)).all() and np.isfinite(sc.astype(np.float32)).all(), n
        tipi[tid] = tipi.get(tid, 0) + 1
        if not np.array_equal(bx, by):
            cambiati.append(n)
    os.replace(tmp, out)
    s = sha(out)
    Path(str(out) + ".sha256").write_text(s + "\n")
    fam = {}
    for n in cambiati:
        f = n.split(".")[2]
        fam[f] = fam.get(f, 0) + 1
    ver = {"ok": True, "braccio": a.braccio, "partenza": a.partenza, "taglia": os.path.getsize(out), "sha256": s,
           "tensori_cambiati": len(cambiati), "per_famiglia": fam, "per_tipo": tipi, "elenco": cambiati, "tipi_identici": True,
           "intestazione": "identica"}
    K.scrivi_json(str(out) + ".verifica.json", ver)
    print(f"braccio {a.braccio}: {out} {ver['taglia']} B sha {s[:16]}; {len(cambiati)} tensori riscritti {fam}; "
          f"tipi/forme/offset identici, intestazione identica")
    return 0


if __name__ == "__main__":
    sys.exit(main())
