"""RAM RESIDENTE di un piano col motore TQZ1 «al volo» (R033, thor-llama ramo cloud/tq1z-al-volo).

Col motore al volo i tensori TQ1_0 / TQ1_0R_1024 restano in memoria come blob TQZ1 (lo stesso del file GGUF-Z) piu' un
indice dei segni: per tensore  pad8(blob) + 8 + 4*ceil(nblocchi/E)  byte, E = 16384 trit / blocco (16 per TQ1_0R_1024,
64 per TQ1_0). La regola di Angelo conta la RAM OCCUPATA a modello caricato (< 9 GiB), non il file: questo modulo la
calcola, ESATTA, senza scrivere nessun file e senza GPU.

  ram_gguf_z(path)                -> RAM al volo di un GGUF-Z (dall'indice thorz.index: istantaneo)
  ram_gguf(path)                  -> RAM al volo che avrebbe il GGUF-Z di un GGUF qualsiasi (decodifica i trit: CPU)
  blob_bytes(raw, tipo)           -> byte del blob TQZ1 di un tensore (stessa scelta di scale_w del codificatore)
  rapporti_ram(report)            -> RAM/grezzo per gruppo da un pack_*.json (per stimare un piano non ancora scritto)
  menus_in_ram(menus, fn)         -> copia dei menu con i byte delle opzioni in byte di RAM (per lo zaino)

  python3 -m rd_ternary.ram_tq1z --gguf piano.gguf [--tetto 9663676416]     # quanto occupa al volo, quanto resta
  python3 -m rd_ternary.ram_tq1z --gguf-z piano.z.gguf

Prova: tests/test_ram_tq1z.py (formula == codificatore C++/Python, al byte, su tensori sintetici e su z9 se c'e').
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from typing import Callable, Dict, Optional, Sequence

import numpy as np

T_TQ1_0, T_TQ1_0R_1024 = 34, 45
GEOM = {T_TQ1_0: (256, 54), T_TQ1_0R_1024: (1024, 224)}     # (trit per blocco, byte per blocco)
CHUNK_BLOCKS = 4096
TRIT_PER_INDICE = 16384
TETTO_9GIB = 9 * 2**30


def gruppo_indice(tipo: int, trit_per_indice: int = TRIT_PER_INDICE) -> int:
    blk = GEOM[tipo][0]
    E = max(1, trit_per_indice // blk)
    while CHUNK_BLOCKS % E:
        E -= 1
    return E


def ram_da_blob(tipo: int, nblocchi: int, zbytes: int, trit_per_indice: int = TRIT_PER_INDICE) -> int:
    """Byte in memoria di un tensore al volo (= ggml_tq1z_mem_size del motore)."""
    E = gruppo_indice(tipo, trit_per_indice)
    return ((zbytes + 7) // 8) * 8 + 8 + 4 * ((nblocchi + E - 1) // E)


def _lut():
    b = np.arange(256, dtype=np.uint32)
    dig = [(((b * 3 ** t) & 255) * 3) >> 8 for t in range(5)]
    z5 = sum((d == 1).astype(np.uint8) for d in dig)
    z4 = sum((d == 1).astype(np.uint8) for d in dig[:4])
    return z5.astype(np.uint8), z4.astype(np.uint8)


_Z5, _Z4 = _lut()


def nnz_per_blocco(raw: np.ndarray, tipo: int) -> np.ndarray:
    """raw uint8 [n*bb] -> non-zeri per blocco (int64 [n])."""
    blk, bb = GEOM[tipo]
    x = np.asarray(raw, dtype=np.uint8).reshape(-1, bb)
    if tipo == T_TQ1_0:
        zeri = _Z5[x[:, :48]].sum(1, dtype=np.int64) + _Z4[x[:, 48:52]].sum(1, dtype=np.int64)
    else:
        zeri = _Z5[x[:, :192]].sum(1, dtype=np.int64) + _Z4[x[:, 192:208]].sum(1, dtype=np.int64)
    return blk - zeri


def blob_bytes(raw: np.ndarray, tipo: int, chunk_blocks: int = CHUNK_BLOCKS) -> int:
    """Byte ESATTI del blob TQZ1 (header + bitmap + segni per chunk + scale + tabella chunk + eccezioni)."""
    blk, bb = GEOM[tipo]
    x = np.asarray(raw, dtype=np.uint8).reshape(-1, bb)
    n = x.shape[0]
    nch = (n + chunk_blocks - 1) // chunk_blocks
    nz = nnz_per_blocco(x, tipo)
    segni = 0
    for c in range(nch):
        segni += (int(nz[c * chunk_blocks:(c + 1) * chunk_blocks].sum()) + 7) // 8
    if tipo == T_TQ1_0:
        scale, n_exc = 2 * n, 0
    else:
        sc = np.ascontiguousarray(x[:, 208:224]).view("<u2").astype(np.int32)
        ad = np.abs(sc[:, 1:] - sc[:, :1])
        best = None
        for w in range(2, 17):
            ne = int((ad > (1 << (w - 1)) - 1).sum())
            cost = n * (16 + 7 * w) / 8.0 + 10.0 * ne
            if best is None or cost < best[0]:
                best = (cost, w, ne)
        _, w, n_exc = best
        scale = 0
        for c in range(nch):
            m = min(n, (c + 1) * chunk_blocks) - c * chunk_blocks
            scale += (m * (16 + 7 * w) + 7) // 8
    return 128 + n * blk // 8 + segni + scale + 16 * nch + 10 * n_exc


def _gguf(path):
    sys_path_ok = True
    try:
        from gguf import GGUFReader            # gguf-py di thor-llama nel PYTHONPATH
    except ImportError:                        # pragma: no cover
        sys_path_ok = False
    if not sys_path_ok:
        raise SystemExit("serve gguf-py nel PYTHONPATH (es. PYTHONPATH=@@ROOT:models@@/build/thor-z/gguf-py)")
    return GGUFReader(path)


def ram_gguf_z(path: str, trit_per_indice: int = TRIT_PER_INDICE) -> Dict:
    """RAM al volo di un GGUF-Z: tensori non ternari + (blob + indice) dei ternari, dall'indice thorz.index."""
    r = _gguf(path)
    f = r.fields["thorz.index"]
    righe = bytes(f.parts[f.data[0]]).decode().splitlines()[1:]
    idx = {l.split("\t")[0]: l.split("\t") for l in righe if l}
    altri = ram = grezzi = blob = 0
    for t in r.tensors:
        if t.name in idx:
            _, tipo, _, n_raw, z = idx[t.name]
            tipo, n_raw, z = int(tipo), int(n_raw), int(z)
            ram += ram_da_blob(tipo, n_raw // GEOM[tipo][1], z, trit_per_indice)
            grezzi += n_raw
            blob += z
        else:
            altri += int(t.n_bytes)
    return {"file": path, "ram_al_volo": altri + ram, "ram_espanso": altri + grezzi, "non_ternari": altri,
            "ternari_ram": ram, "ternari_blob": blob, "ternari_grezzi": grezzi}


def ram_gguf(path: str, trit_per_indice: int = TRIT_PER_INDICE, log=None) -> Dict:
    """RAM al volo che avrebbe questo GGUF una volta impacchettato (GGUF-Z) e caricato col motore al volo."""
    r = _gguf(path)
    altri = ram = grezzi = blob = 0
    per_tipo: Dict[int, Dict[str, int]] = {}
    for i, t in enumerate(r.tensors):
        tipo = int(t.tensor_type)
        if tipo in GEOM:
            raw = np.asarray(t.data, dtype=np.uint8).reshape(-1)
            z = blob_bytes(raw, tipo)
            m = ram_da_blob(tipo, raw.size // GEOM[tipo][1], z, trit_per_indice)
            ram += m
            grezzi += raw.size
            blob += z
            d = per_tipo.setdefault(tipo, {"grezzi": 0, "ram": 0})
            d["grezzi"] += raw.size
            d["ram"] += m
        else:
            altri += int(t.n_bytes)
        if log and i % 200 == 0:
            log(f"  {i}/{len(r.tensors)} tensori")
    return {"file": path, "ram_al_volo": altri + ram, "ram_espanso": altri + grezzi, "non_ternari": altri,
            "ternari_ram": ram, "ternari_blob": blob, "ternari_grezzi": grezzi,
            "rapporto_per_tipo": {str(k): v["ram"] / v["grezzi"] for k, v in per_tipo.items()}}


def rapporti_ram(report_path: str) -> Dict[str, float]:
    """Da un pack_*.json (tq1z_gguf pack): RAM/grezzo per gruppo (indice compreso), es. {'densi/p1': 0.9337, ...}."""
    rep = json.load(open(report_path))
    out = {}
    for g, v in rep["per_gruppo"].items():
        tipo = int(v["tipo"])
        blk, bb = GEOM[tipo]
        nb = int(v["blocchi"])
        E = gruppo_indice(tipo)
        indice = 8 * int(v["tensori"]) + 4 * (nb + E - 1) // E   # + pad8 (<8 B per tensore)
        out[g] = (int(v["bytes_z"]) + indice + 4 * int(v["tensori"])) / float(v["bytes_raw"])
    return out


def menus_in_ram(menus: Sequence, rapporto: Callable[[object], float]) -> list:
    """Copia dei menu con option.bytes in byte di RAM al volo: bytes_ram = ceil(bytes * rapporto(menu)).
    Lo zaino (allocazione/unisci) lavora poi in byte di RAM: il tetto da passargli e' la RAM, non il file."""
    out = []
    for m in menus:
        r = float(rapporto(m))
        mm = copy.copy(m)
        mm.options = [copy.copy(o) for o in m.options]
        for o in mm.options:
            o.bytes = int(math.ceil(o.bytes * r))
        out.append(mm)
    return out


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gguf", help="GGUF qualsiasi: RAM che avrebbe al volo (decodifica i trit, solo CPU)")
    g.add_argument("--gguf-z", help="GGUF-Z: RAM al volo esatta dall'indice")
    g.add_argument("--report", help="pack_*.json: rapporti RAM/grezzo per gruppo")
    ap.add_argument("--tetto", type=int, default=TETTO_9GIB, help="tetto di RAM (default 9 GiB)")
    ap.add_argument("--trit-per-indice", type=int, default=TRIT_PER_INDICE, help="THOR_TQ1Z_GRUPPO del motore")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.report:
        res = {"rapporti_ram": rapporti_ram(a.report)}
    else:
        res = ram_gguf_z(a.gguf_z, a.trit_per_indice) if a.gguf_z else \
            ram_gguf(a.gguf, a.trit_per_indice, log=lambda s: print(s, file=sys.stderr, flush=True))
        res["tetto"] = a.tetto
        res["margine_al_volo"] = a.tetto - res["ram_al_volo"]
        res["margine_espanso"] = a.tetto - res["ram_espanso"]
        res["entro_tetto_al_volo"] = res["ram_al_volo"] <= a.tetto
        res["GiB"] = {k: round(res[k] / 2**30, 4) for k in ("ram_al_volo", "ram_espanso", "margine_al_volo")}
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s)
    return res


if __name__ == "__main__":
    main()
