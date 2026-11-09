#!/usr/bin/env python3
"""bpost_verifica_1026.py — 26/10 B+POST: byte check of a patched GGUF against the GGUF it was patched from.

  PRIMA and DOPO: same tensor names and order, types, shapes, offsets, n_bytes, file size;
  every tensor whose bytes changed must be ALLOWED (--ammessi REGEX, --ammessi-file list of names, --f32 = any F32
  tensor outside the MTP layer blk.40); changed tensors matching --codici REGEX (ternary TQ1_0 / TQ1_0R_1024) must have
  their CODE bytes identical (only the fp16 d fields may move); changed F32 tensors must be finite.
uso: bpost_verifica_1026.py PRIMA.gguf DOPO.gguf [--ammessi REGEX] [--ammessi-file NOMI.txt] [--f32] [--codici REGEX]
         [--json OUT.json] [--almeno N]
Exit 0 = ok, 1 = check failed. Last stdout line = one-line summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/BPP/codice/thor_v2")
import e2e_campi_scala as CAMPI  # noqa: E402  (inserts the k1024 gguf-py path)
from gguf import GGUFReader      # noqa: E402


def famiglia(n: str) -> str:
    m = re.match(r"blk\.\d+\.(.+?)(\.weight|\.bias)?$", n)
    return m.group(1) if m else n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prima"); ap.add_argument("dopo")
    ap.add_argument("--ammessi", default=None)
    ap.add_argument("--ammessi-file", default=None)
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--codici", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--almeno", type=int, default=0, help="minimum number of changed tensors")
    a = ap.parse_args(argv)
    if os.path.getsize(a.prima) != os.path.getsize(a.dopo):
        print(f"ERRORE taglia: {os.path.getsize(a.prima)} contro {os.path.getsize(a.dopo)}"); return 1
    ta = {t.name: t for t in GGUFReader(a.prima, "r").tensors}
    tb = {t.name: t for t in GGUFReader(a.dopo, "r").tensors}
    if list(ta) != list(tb):
        print("ERRORE: nomi/ordine dei tensori diversi"); return 1
    nomi_ok = set()
    if a.ammessi_file:
        nomi_ok = {r.strip() for r in open(a.ammessi_file) if r.strip()}
    rx = re.compile(a.ammessi) if a.ammessi else None
    rc = re.compile(a.codici) if a.codici else None
    rep = {"prima": a.prima, "dopo": a.dopo, "tensori": len(ta), "cambiati": 0, "per_famiglia": collections.Counter(),
           "fuori": [], "codici_diversi": [], "f32_non_finiti": [], "intestazione_diversa": [], "codici_controllati": 0}
    for n, x in ta.items():
        y = tb[n]
        if (x.tensor_type, list(x.shape), int(x.data_offset), int(x.n_bytes)) != \
                (y.tensor_type, list(y.shape), int(y.data_offset), int(y.n_bytes)):
            rep["intestazione_diversa"].append(n); continue
        bx = np.asarray(x.data).reshape(-1).view(np.uint8)
        by = np.asarray(y.data).reshape(-1).view(np.uint8)
        if np.array_equal(bx, by):
            continue
        rep["cambiati"] += 1; rep["per_famiglia"][famiglia(n)] += 1
        tipo = x.tensor_type.name
        ok = (n in nomi_ok) or (rx is not None and rx.fullmatch(n) is not None) \
            or (a.f32 and tipo == "F32" and not n.startswith("blk.40."))
        if not ok:
            rep["fuori"].append(n)
        if rc is not None and rc.fullmatch(n) and tipo in CAMPI.LAYOUT:
            m = CAMPI.maschera_codici(tipo)
            rep["codici_controllati"] += 1
            if not np.array_equal(CAMPI.blocchi(bx, tipo)[:, m], CAMPI.blocchi(by, tipo)[:, m]):
                rep["codici_diversi"].append(n)
        if tipo == "F32" and not np.isfinite(by.view("<f4")).all():
            rep["f32_non_finiti"].append(n)
    rep["per_famiglia"] = dict(sorted(rep["per_famiglia"].items()))
    esito = not (rep["fuori"] or rep["codici_diversi"] or rep["f32_non_finiti"] or rep["intestazione_diversa"]) \
        and rep["cambiati"] >= a.almeno
    rep["esito"] = esito
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    print(json.dumps(rep["per_famiglia"]))
    print(f"{'OK' if esito else 'ERRORE'}: tensori {rep['tensori']}; cambiati {rep['cambiati']}; fuori {rep['fuori'][:5]}"
          f"{'...' if len(rep['fuori']) > 5 else ''}; codici controllati {rep['codici_controllati']}, diversi "
          f"{rep['codici_diversi'][:5]}; F32 non finiti {rep['f32_non_finiti'][:5]}; intestazione diversa "
          f"{rep['intestazione_diversa'][:5]}")
    return 0 if esito else 1


if __name__ == "__main__":
    sys.exit(main())
