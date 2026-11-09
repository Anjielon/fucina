#!/usr/bin/env python3
"""e1_scrivi_residui_gguf.py — GGUF NUOVO = originale AL BYTE + tensori ternari IN PIU' (residui di rango basso E1).

Stile di e2_scrivi_gguf.py: l'originale non si apre MAI in scrittura; copia nel kernel (copy_file_range) in
<out>.parziale, rilettura e confronto, fsync, rename atomico, <out>.modifica.json accanto.

Come cambia il file (GGUF v2/v3, little-endian):
  intestazione  magic, versione                                   identici
                n_tensori, n_kv                                   + N nuovi tensori (+1 KV se --meta)
                coppie KV                                         byte originali, poi la KV nuova (stringa JSON)
                info tensori                                      byte originali (offset INVARIATI: sono relativi
                                                                  all'inizio dei dati), poi le info dei nuovi
                imbottitura fino all'allineamento (general.alignment, default 32)
  dati          regione dati originale copiata AL BYTE, poi i nuovi tensori, ognuno allineato
Con zero tensori nuovi e niente --meta l'intestazione ricostruita e' identica e il file ha la STESSA sha256
(modifica nulla: prova in test_residui_ternari.py, e `--nulla` la verifica sul file vero).

Ingressi: --npz residui_tq1.npz (byte TQ1_0 per tensore, da residui_ternari.tensori_gguf) + --info residui.json
({"tensori": {nome: {"type": 34, "ne": [ne0, ne1]}}}). Tipi ammessi per i nuovi tensori: TQ1_0 (34), F32 (0), F16 (1).

Uso:
  e1_scrivi_residui_gguf.py --src ST.gguf --out OUT.gguf --nulla                       # collaudo: sha identica
  e1_scrivi_residui_gguf.py --src ST.gguf --out OUT.gguf --npz E1/residui_tq1.npz --info E1/residui.json --meta
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ggml_type -> (valori per blocco, byte per blocco) per i tipi che si possono AGGIUNGERE
TIPI_AGGIUNGIBILI = {0: (1, 4), 1: (1, 2), 34: (256, 54)}
_SCALARI = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_STRING, GGUF_ARRAY, GGUF_UINT32 = 8, 9, 4
KV_META = "thor.lettera.residui"


class Lettore:
    def __init__(self, f):
        self.f = f

    def leggi(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("GGUF troncato")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.leggi(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.leggi(8))[0]

    def stringa(self) -> str:
        return self.leggi(self.u64()).decode("utf-8")

    def valore(self, t: int):
        if t in _SCALARI:
            b = self.leggi(_SCALARI[t])
            return struct.unpack("<" + {1: "B", 2: "H", 4: "I", 8: "Q"}[len(b)], b)[0] if t in (0, 2, 4, 10, 7) \
                else b
        if t == GGUF_STRING:
            return self.stringa()
        if t == GGUF_ARRAY:
            ti, n = self.u32(), self.u64()
            if ti in _SCALARI:
                self.leggi(_SCALARI[ti] * n)
                return None
            for _ in range(n):
                self.valore(ti)
            return None
        raise ValueError(f"tipo KV sconosciuto {t}")


def indice(path: str) -> dict:
    """Mappa dei byte dell'intestazione: versione, conteggi, campate KV e info tensori, allineamento, tensori, inizio dati."""
    with open(path, "rb") as f:
        r = Lettore(f)
        if r.leggi(4) != b"GGUF":
            raise ValueError(f"{path}: non e' un GGUF")
        ver = r.u32()
        if ver not in (2, 3):
            raise ValueError(f"{path}: versione GGUF {ver} non gestita")
        nt, nkv = r.u64(), r.u64()
        kv0 = f.tell()
        allin = 32
        chiavi = []
        for _ in range(nkv):
            k = r.stringa()
            t = r.u32()
            v = r.valore(t)
            chiavi.append(k)
            if k == "general.alignment" and t == GGUF_UINT32:
                allin = int(v)
        kv1 = f.tell()
        tens = []
        for _ in range(nt):
            nome = r.stringa()
            nd = r.u32()
            ne = [r.u64() for _ in range(nd)]
            ty = r.u32()
            off = r.u64()
            tens.append({"nome": nome, "ne": ne, "type": ty, "offset": off})
        ti1 = f.tell()
    dati = (ti1 + allin - 1) // allin * allin
    return {"versione": ver, "n_tensori": nt, "n_kv": nkv, "kv": (kv0, kv1), "info": (kv1, ti1), "allineamento": allin,
            "tensori": tens, "chiavi": chiavi, "inizio_dati": dati, "taglia": os.path.getsize(path)}


def _stringa(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def info_tensore(nome: str, ne: list, ty: int, off: int) -> bytes:
    return _stringa(nome) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", int(x)) for x in ne) + \
        struct.pack("<I", ty) + struct.pack("<Q", off)


def kv_stringa(k: str, v: str) -> bytes:
    return _stringa(k) + struct.pack("<I", GGUF_STRING) + _stringa(v)


def nbyte(ty: int, ne: list) -> int:
    if ty not in TIPI_AGGIUNGIBILI:
        raise ValueError(f"tipo {ty} non ammesso per un tensore nuovo ({sorted(TIPI_AGGIUNGIBILI)})")
    qk, bb = TIPI_AGGIUNGIBILI[ty]
    if ne[0] % qk:
        raise ValueError(f"ne0 {ne[0]} non multiplo di {qk} per il tipo {ty}")
    return int(np.prod(ne)) // qk * bb


def _allinea(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def copia_campata(fi, fo, n: int, off_i: int, off_o: int) -> None:
    fatto = 0
    while fatto < n:
        k = os.copy_file_range(fi.fileno(), fo.fileno(), min(1 << 30, n - fatto), off_i + fatto, off_o + fatto)
        if k <= 0:
            raise OSError("copy_file_range: 0 byte")
        fatto += k


def sha256_campata(path: str, a: int, b: int, bs: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(a)
        n = b - a
        while n > 0:
            x = f.read(min(bs, n))
            if not x:
                break
            h.update(x)
            n -= len(x)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    return sha256_campata(path, 0, os.path.getsize(path))


def scrivi(src: str, out: str, nuovi: list, meta: str | None = None, min_libero_gb: float = 0.0) -> dict:
    """nuovi = [(nome, ggml_type, ne, byte np.uint8)]. Ritorna il rapporto (anche scritto in <out>.modifica.json)."""
    src, out = os.path.realpath(src), os.path.realpath(out)
    if src == out:
        raise SystemExit("--out coincide con --src: l'originale non si tocca")
    if os.path.exists(out):
        raise SystemExit(f"{out} esiste gia': non sovrascrivo")
    t0 = time.time()
    ix = indice(src)
    st = os.statvfs(os.path.dirname(out))
    libero = st.f_bavail * st.f_frsize / 2**30
    extra = sum(int(np.asarray(b).nbytes) for *_, b in nuovi) / 2**30
    if libero < ix["taglia"] / 2**30 + extra + min_libero_gb:
        raise SystemExit(f"spazio: {libero:.1f} GiB liberi, servono {ix['taglia'] / 2**30 + extra + min_libero_gb:.1f}")
    esistenti = {t["nome"] for t in ix["tensori"]}
    al = ix["allineamento"]
    D = ix["taglia"] - ix["inizio_dati"]
    pos = _allinea(D, al)
    info_n, blob = [], []
    for nome, ty, ne, b in nuovi:
        if nome in esistenti:
            raise SystemExit(f"{nome}: esiste gia' nel GGUF")
        b = np.ascontiguousarray(b, dtype=np.uint8)
        if b.nbytes != nbyte(ty, ne):
            raise SystemExit(f"{nome}: {b.nbytes} byte, attesi {nbyte(ty, ne)} per tipo {ty} ne {ne}")
        info_n.append(info_tensore(nome, ne, ty, pos))
        blob.append((pos, b))
        esistenti.add(nome)
        pos = _allinea(pos + b.nbytes, al)
    if meta is not None and KV_META in ix["chiavi"]:
        raise SystemExit(f"KV {KV_META} gia' presente")
    with open(src, "rb") as fi:
        testa = fi.read(ix["info"][1])
    kv_raw = testa[ix["kv"][0]:ix["kv"][1]]
    ti_raw = testa[ix["info"][0]:ix["info"][1]]
    nkv = ix["n_kv"] + (1 if meta is not None else 0)
    hdr = testa[:8] + struct.pack("<QQ", ix["n_tensori"] + len(nuovi), nkv) + kv_raw + \
        (kv_stringa(KV_META, meta) if meta is not None else b"") + ti_raw + b"".join(info_n)
    hdr += b"\x00" * (_allinea(len(hdr), al) - len(hdr))
    with open(src, "rb") as fi:
        orig_pad = fi.read(ix["inizio_dati"])
    if not nuovi and meta is None and hdr != orig_pad:
        raise SystemExit("modifica nulla: intestazione ricostruita DIVERSA dall'originale (parser da correggere)")
    tmp = out + ".parziale"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(hdr)
        fo.flush()
        copia_campata(fi, fo, D, ix["inizio_dati"], len(hdr))
        fo.seek(len(hdr) + D)
        for off, b in blob:
            fo.write(b"\x00" * (len(hdr) + off - fo.tell()))
            fo.write(b.tobytes())
        fo.flush()
        os.fsync(fo.fileno())
    # rilettura: indice coerente, tensori originali identici (info e byte), nuovi al loro posto
    ox = indice(tmp)
    if ox["tensori"][:ix["n_tensori"]] != ix["tensori"] or ox["allineamento"] != al:
        raise SystemExit("rilettura: info dei tensori originali cambiate (.parziale lasciato)")
    if sha256_campata(tmp, ox["inizio_dati"], ox["inizio_dati"] + D) != sha256_campata(src, ix["inizio_dati"],
                                                                                         ix["taglia"]):
        raise SystemExit("rilettura: regione dati originale diversa (.parziale lasciato)")
    with open(tmp, "rb") as f:
        for (nome, ty, ne, _), (off, b), t in zip(nuovi, blob, ox["tensori"][ix["n_tensori"]:]):
            f.seek(ox["inizio_dati"] + off)
            if t["nome"] != nome or t["offset"] != off or f.read(b.nbytes) != b.tobytes():
                raise SystemExit(f"rilettura {nome}: non combacia (.parziale lasciato)")
    os.replace(tmp, out)
    rep = {"src": src, "out": out, "tensori_aggiunti": [{"nome": n, "type": ty, "ne": list(map(int, ne)),
                                                          "byte": int(np.asarray(b).nbytes)} for n, ty, ne, b in nuovi],
           "kv_aggiunta": KV_META if meta is not None else None, "byte_intestazione": [ix["inizio_dati"], len(hdr)],
           "regione_dati_originale_byte": D, "secondi": round(time.time() - t0, 1)}
    return rep


def carica_nuovi(npz: str, info: str) -> list:
    z = np.load(npz)
    inf = json.loads(Path(info).read_text())["tensori"]
    if set(z.files) != set(inf):
        raise SystemExit(f"npz e info non combaciano: {sorted(set(z.files) ^ set(inf))[:5]}")
    return [(n, int(inf[n]["type"]), [int(x) for x in inf[n]["ne"]], z[n]) for n in sorted(inf)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz")
    ap.add_argument("--info")
    ap.add_argument("--nulla", action="store_true", help="nessun tensore nuovo: verifica sha256 identica")
    ap.add_argument("--meta", action="store_true", help=f"aggiunge la KV {KV_META} (JSON di --info)")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--skip-sha", action="store_true")
    a = ap.parse_args(argv)
    if a.nulla == bool(a.npz):
        raise SystemExit("serve --nulla oppure --npz + --info")
    nuovi = [] if a.nulla else carica_nuovi(a.npz, a.info)
    meta = Path(a.info).read_text() if (a.meta and a.info) else None
    rep = scrivi(a.src, a.out, nuovi, meta, a.min_free_gb)
    if not a.skip_sha:
        rep["sha256_src"], rep["sha256_out"] = sha256_file(a.src), sha256_file(a.out)
        if a.nulla and rep["sha256_src"] != rep["sha256_out"]:
            raise SystemExit("modifica nulla con sha256 DIVERSA")
    Path(a.out + ".modifica.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "tensori_aggiunti"} | {"n_aggiunti": len(nuovi)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
