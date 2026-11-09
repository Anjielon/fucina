"""Tensor-level identity of GGUF files, with no dependency beyond the standard library.

Two GGUF files can hold byte-identical tensors and still differ as files: writers embed metadata (names, notes,
sometimes the absolute paths of their inputs). The replay therefore checks the chain at two levels:

* ``file`` — sha256 of the whole file (exact identity, needs identical metadata);
* ``tensors`` — sha256 of every tensor's name, type, shape and data bytes, and one digest over all of them
  (:func:`tensor_digest`). This is the identity that matters for the model's arithmetic.

The reader parses the GGUF header (versions 2 and 3) and hashes each tensor's data region. Sizes come from the
block table below; a type that is not in the table is sized by the distance to the next tensor (padding included),
which is still a stable identity for files written by the same writer.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# ggml type id -> (name, elements per block, bytes per block)
TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    45: ("TQ1_0R_1024", 1024, 224),
    46: ("TQ1_0R_1024_X2", 1024, 448),
    47: ("TQ1_0R_1024_X3", 1024, 672),
    48: ("TQ1_0R_1024_X4", 1024, 896),
}

# GGUF metadata value types -> struct format (fixed-size scalars)
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


@dataclass
class TensorInfo:
    name: str
    shape: tuple
    type_id: int
    offset: int          # relative to the start of the data section
    nbytes: int = 0

    @property
    def type_name(self) -> str:
        return TYPES.get(self.type_id, (f"type{self.type_id}", 0, 0))[0]


class GGUFError(ValueError):
    pass


def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    data = f.read(n)
    if len(data) != n:
        raise GGUFError("truncated string")
    return data.decode("utf-8", errors="replace")


def _value(f: BinaryIO, vtype: int):
    if vtype in _SCALAR:
        return _read(f, _SCALAR[vtype])
    if vtype == _STRING:
        return _string(f)
    if vtype == _ARRAY:
        itype = _read(f, "<I")
        n = _read(f, "<Q")
        return [_value(f, itype) for _ in range(n)]
    raise GGUFError(f"unknown metadata type {vtype}")


def read_header(path: str | Path) -> tuple[dict, list[TensorInfo], int]:
    """(metadata, tensors with data sizes, absolute offset of the data section)."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError(f"{path}: not a GGUF file")
        version = _read(f, "<I")
        if version not in (2, 3):
            raise GGUFError(f"{path}: GGUF version {version} not supported")
        n_tensors = _read(f, "<Q")
        n_kv = _read(f, "<Q")
        meta = {}
        for _ in range(n_kv):
            key = _string(f)
            meta[key] = _value(f, _read(f, "<I"))
        tensors = []
        for _ in range(n_tensors):
            name = _string(f)
            n_dims = _read(f, "<I")
            shape = tuple(_read(f, "<Q") for _ in range(n_dims))
            tensors.append(TensorInfo(name, shape, _read(f, "<I"), _read(f, "<Q")))
        align = int(meta.get("general.alignment", DEFAULT_ALIGNMENT))
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
    order = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(order):
        known = TYPES.get(t.type_id)
        if known:
            n = 1
            for d in t.shape:
                n *= d
            t.nbytes = n // known[1] * known[2]
        else:
            end = order[i + 1].offset if i + 1 < len(order) else size - data_start
            t.nbytes = end - t.offset
        if data_start + t.offset + t.nbytes > size:
            raise GGUFError(f"{path}: tensor {t.name} extends past the end of the file")
    return meta, tensors, data_start


def tensor_hashes(path: str | Path, chunk: int = 1 << 24) -> dict[str, str]:
    """sha256 of every tensor: name, type, shape and data bytes."""
    meta, tensors, start = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for t in sorted(tensors, key=lambda t: t.offset):
            h = hashlib.sha256(f"{t.name}|{t.type_id}|{','.join(map(str, t.shape))}|".encode())
            f.seek(start + t.offset)
            left = t.nbytes
            while left:
                b = f.read(min(chunk, left))
                if not b:
                    raise GGUFError(f"{path}: short read in {t.name}")
                h.update(b)
                left -= len(b)
            out[t.name] = h.hexdigest()
    return out


def tensor_digest(hashes: dict[str, str]) -> str:
    """One sha256 over all tensor hashes, in name order (independent of the order in the file)."""
    h = hashlib.sha256()
    for name in sorted(hashes):
        h.update(f"{name}\t{hashes[name]}\n".encode())
    return h.hexdigest()


def compare(a: dict[str, str], b: dict[str, str]) -> dict:
    """Tensors only in a / only in b / with different content."""
    return {"only_in_first": sorted(set(a) - set(b)), "only_in_second": sorted(set(b) - set(a)),
            "different": sorted(k for k in set(a) & set(b) if a[k] != b[k])}


def file_sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def main(argv: Optional[list] = None) -> int:
    import argparse
    import json
    p = argparse.ArgumentParser(description="tensor-level sha256 of GGUF files")
    p.add_argument("gguf", nargs="+")
    p.add_argument("--per-tensor", action="store_true", help="print every tensor hash (JSON)")
    a = p.parse_args(argv)
    for g in a.gguf:
        hs = tensor_hashes(g)
        if a.per_tensor:
            print(json.dumps({"file": g, "tensors": hs}, indent=1))
        else:
            print(f"{tensor_digest(hs)}  {len(hs)} tensors  {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
