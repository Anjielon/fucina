"""Lettura/scrittura GGUF a livello di BYTE (v2/v3), senza dipendere da gguf-py.

Serve a scrivi_piano_gguf: la sezione dei metadati (KV) si copia COSI' COM'E' (byte per
byte), le informazioni dei tensori si ricostruiscono e i dati si dispongono con lo
stesso allineamento del file di partenza. Se la lista dei tensori non cambia, il file
riscritto e' identico al sorgente (verificato dai test: sha256 uguale).

Layout GGUF:
  magic 'GGUF' | version u32 | n_tensors u64 | n_kv u64 | KV... | tensor info... |
  padding all'allineamento | dati (ogni tensore a un offset multiplo dell'allineamento)
  tensor info = nome (u64 len + byte) | n_dims u32 | ne[i] u64 | tipo u32 | offset u64
"""
from __future__ import annotations

import hashlib
import mmap
import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# (pesi per blocco, byte per blocco) dei tipi ggml che servono qui (ggml.h / gguf-py del ramo)
TYPE_SIZE: Dict[int, Tuple[int, int]] = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 36),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
    43: (256, 54),   # TQ1_0R (Hadamard 256)
    45: (1024, 224), # TQ1_0R_1024
    46: (256, 60),   # TQ1_0_G64       (ramo thor-llama cloud/scala-g64, 6/10)
    47: (1024, 240), # TQ1_0R_1024_G64 (idem)
}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def tensor_nbytes(dims: Sequence[int], type_id: int) -> int:
    if type_id not in TYPE_SIZE:
        raise ValueError(f"tipo ggml {type_id} sconosciuto (aggiungerlo a TYPE_SIZE)")
    blk, bb = TYPE_SIZE[type_id]
    n = 1
    for d in dims:
        n *= int(d)
    if int(dims[0]) % blk:
        raise ValueError(f"ne0={dims[0]} non multiplo del blocco {blk} (tipo {type_id})")
    return n // blk * bb


def align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass
class TensorInfo:
    name: str
    dims: Tuple[int, ...]          # ordine ggml (ne0 = dimensione piu' interna)
    type_id: int
    offset: int                    # relativo all'inizio della sezione dati
    nbytes: int


@dataclass
class GGUFFile:
    path: str
    version: int
    n_kv: int
    kv_raw: bytes                  # byte della sezione KV, copiati tali e quali
    kv: Dict[str, object]          # KV decodificati (scalari e stringhe; array -> lista)
    tensors: List[TensorInfo]
    alignment: int
    data_start: int
    file_size: int

    def by_name(self) -> Dict[str, TensorInfo]:
        return {t.name: t for t in self.tensors}


class _Rd:
    def __init__(self, mm, off: int = 0):
        self.mm, self.off = mm, off

    def take(self, n: int) -> bytes:
        b = self.mm[self.off: self.off + n]
        if len(b) != n:
            raise ValueError("GGUF troncato")
        self.off += n
        return bytes(b)

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8")

    def value(self, vt: int):
        if vt in _SCALAR:
            fmt = _SCALAR[vt]
            return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]
        if vt == 8:
            return self.string()
        if vt == 9:   # array: si attraversa sempre; si tiene la lista solo se corta (altrimenti la lunghezza)
            et, n = self.u32(), self.u64()
            if et in _SCALAR:
                sz = struct.calcsize(_SCALAR[et])
                raw = self.take(sz * n)
                return np.frombuffer(raw, dtype=np.dtype(_SCALAR[et])).tolist() if n <= 64 else n
            vals = [self.value(et) for _ in range(n)]
            return vals if n <= 64 else n
        raise ValueError(f"tipo KV {vt} sconosciuto")


def read_gguf(path: str) -> GGUFFile:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            r = _Rd(mm)
            if r.take(4) != GGUF_MAGIC:
                raise ValueError(f"{path}: non e' un GGUF")
            version = r.u32()
            if version not in (2, 3):
                raise ValueError(f"{path}: versione GGUF {version} non supportata")
            n_t, n_kv = r.u64(), r.u64()
            kv_start = r.off
            kv: Dict[str, object] = {}
            for _ in range(n_kv):
                k = r.string()
                vt = r.u32()
                kv[k] = r.value(vt)
            kv_raw = bytes(mm[kv_start: r.off])
            alignment = int(kv.get("general.alignment", DEFAULT_ALIGNMENT))
            tensors = []
            for _ in range(n_t):
                name = r.string()
                nd = r.u32()
                dims = tuple(r.u64() for _ in range(nd))
                tt = r.u32()
                off = r.u64()
                tensors.append(TensorInfo(name, dims, tt, off, tensor_nbytes(dims, tt)))
            data_start = align(r.off, alignment)
        finally:
            mm.close()
    return GGUFFile(path, version, n_kv, kv_raw, kv, tensors, alignment, data_start, size)


# --------------------------------------------------------------------------- scrittura
def _enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key: str, value: str) -> bytes:
    """Una coppia KV di tipo stringa, pronta da accodare alla sezione KV."""
    return _enc_str(key) + struct.pack("<I", 8) + _enc_str(value)


@dataclass
class OutTensor:
    """Tensore in uscita: dati copiati dal sorgente (src_offset assoluto) o forniti (data)."""
    name: str
    dims: Tuple[int, ...]
    type_id: int
    src_offset: Optional[int] = None      # offset ASSOLUTO nel file sorgente
    data: Optional[np.ndarray] = None     # uint8 (o qualunque array: si scrivono i suoi byte)
    nbytes: int = field(default=0)

    def __post_init__(self):
        n = tensor_nbytes(self.dims, self.type_id)
        if self.data is not None:
            got = np.ascontiguousarray(self.data).nbytes
            if got != n:
                raise ValueError(f"{self.name}: {got} byte forniti, attesi {n}")
        elif self.src_offset is None:
            raise ValueError(f"{self.name}: serve src_offset oppure data")
        self.nbytes = n


def layout(tensors: Sequence[OutTensor], alignment: int) -> List[int]:
    offs, off = [], 0
    for t in tensors:
        off = align(off, alignment)
        offs.append(off)
        off += t.nbytes
    return offs


def header_bytes(src: GGUFFile, tensors: Sequence[OutTensor], extra_kv: Sequence[bytes] = ()) -> bytes:
    offs = layout(tensors, src.alignment)
    h = bytearray()
    h += GGUF_MAGIC + struct.pack("<I", src.version) + struct.pack("<QQ", len(tensors), src.n_kv + len(extra_kv))
    h += src.kv_raw
    for kvb in extra_kv:
        h += kvb
    for t, o in zip(tensors, offs):
        h += _enc_str(t.name) + struct.pack("<I", len(t.dims))
        for d in t.dims:
            h += struct.pack("<Q", int(d))
        h += struct.pack("<I", t.type_id) + struct.pack("<Q", o)
    h += b"\0" * (align(len(h), src.alignment) - len(h))
    return bytes(h)


def predicted_size(src: GGUFFile, tensors: Sequence[OutTensor], extra_kv: Sequence[bytes] = ()) -> int:
    offs = layout(tensors, src.alignment)
    end = (offs[-1] + tensors[-1].nbytes) if tensors else 0
    tail = _tail_padding(src)
    return len(header_bytes(src, tensors, extra_kv)) + (align(end, src.alignment) if tail else end)


def _tail_padding(src: GGUFFile) -> bool:
    """Il sorgente ha il padding anche dopo l'ultimo tensore? (lo si riproduce uguale)."""
    if not src.tensors:
        return False
    last = max(src.tensors, key=lambda t: t.offset)
    end = src.data_start + last.offset + last.nbytes
    return src.file_size > end


def write_gguf(src: GGUFFile, tensors: Sequence[OutTensor], out_path: str,
               extra_kv: Sequence[bytes] = (), chunk: int = 64 << 20) -> Dict:
    """Scrive il GGUF; ritorna {'bytes', 'sha256'}. I dati copiati si leggono dal sorgente a pezzi."""
    hdr = header_bytes(src, tensors, extra_kv)
    offs = layout(tensors, src.alignment)
    h = hashlib.sha256()
    tmp = out_path + ".parziale"
    with open(src.path, "rb") as fs, open(tmp, "wb") as fo:
        def w(b):
            fo.write(b)
            h.update(b)
        w(hdr)
        pos = 0
        for t, o in zip(tensors, offs):
            if o > pos:
                w(b"\0" * (o - pos))
            if t.data is not None:
                w(np.ascontiguousarray(t.data).tobytes())
            else:
                fs.seek(t.src_offset)
                left = t.nbytes
                while left:
                    b = fs.read(min(chunk, left))
                    if not b:
                        raise IOError(f"{t.name}: sorgente troncato")
                    w(b)
                    left -= len(b)
            pos = o + t.nbytes
        if _tail_padding(src):
            w(b"\0" * (align(pos, src.alignment) - pos))
        n = fo.tell()
    os.replace(tmp, out_path)
    return {"bytes": n, "sha256": h.hexdigest()}


def sha256_file(path: str, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_all(src: GGUFFile) -> List[OutTensor]:
    """Lista dei tensori del sorgente, tutti 'copiati' (piano nullo)."""
    return [OutTensor(t.name, t.dims, t.type_id, src_offset=src.data_start + t.offset) for t in src.tensors]


def read_tensor(src: GGUFFile, t: TensorInfo) -> np.ndarray:
    """Byte grezzi di un tensore (uint8, copia in memoria)."""
    with open(src.path, "rb") as f:
        f.seek(src.data_start + t.offset)
        return np.frombuffer(f.read(t.nbytes), dtype=np.uint8).copy()
