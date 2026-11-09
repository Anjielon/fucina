"""Unpack a packed multi-plane GGUF back into separate plane tensors (exact inverse of ``pack_planes.py``).

The released root of Thor-T is the *packed* file: every family with K ternary planes is one tensor of type
``TQ1_0R_1024_X{K}`` / ``TQ1_0_X{K}`` whose block b holds the K planes' blocks b back to back. That file loads on the
release engine. The research code of the recorded chain reads the *separate-plane* layout (``blk.N.attn_qkv.weight``,
``blk.N.attn_qkv2.weight``, ``blk.N.attn_qkv3.weight``, all of the base type). This tool rebuilds it:

* plane 1 keeps the base name, planes 2..K are named ``<stem>2.weight`` .. ``<stem>K.weight`` (the inverse of the
  pack rule);
* tensor order: the order recorded in ``--order`` (one name per line, the recorded root), otherwise base tensors in
  file order followed by the extra planes;
* metadata is copied unchanged, in the same order (``pack_planes`` copies it the same way).

``--hash-only`` writes nothing: it streams the output through sha256 and also prints the tensor digest, so the result
can be compared with the recorded root (file sha256 56cb5bcd..., or tensor by tensor with ``--compare``) without
needing 8.8 GB of free disk. Needs the ``gguf`` package of the release engine (``--gguf-py`` or installed).

    python -m fucina.replay.unpack_planes PACKED.gguf OUT.gguf [--order ORDER.txt] [--gguf-py DIR]
    python -m fucina.replay.unpack_planes PACKED.gguf --hash-only [--order ORDER.txt] [--compare ORIGINAL.gguf]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from . import gguf_hash

STEM = re.compile(r"^(?P<stem>.+?)(?P<suffix>\.weight)$")


class HashSink:
    """A write-only file object that hashes what is written (GGUFWriter writes sequentially)."""

    def __init__(self):
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, b) -> int:
        b = bytes(b) if not isinstance(b, (bytes, bytearray)) else b
        self.h.update(b)
        self.n += len(b)
        return len(b)

    def tell(self) -> int:
        return self.n

    def flush(self):
        pass

    def close(self):
        pass


class Raw:
    """Tensor bytes handed to GGUFWriter: written with ``write`` (works for files and for the hashing sink)."""

    def __init__(self, a):
        self.a = a
        self.nbytes = a.nbytes

    def tofile(self, f) -> None:
        import numpy as np
        f.write(memoryview(np.ascontiguousarray(self.a)).cast("B"))


def unpacked_tensors(reader, gguf):
    """[(name, type, data uint8 array (rows, row_bytes) or original array)] of the separate-plane layout."""
    QT = gguf.GGMLQuantizationType
    base_of = {}
    for k in (2, 3, 4):
        for b in ("TQ1_0R_1024", "TQ1_0"):
            name = f"{b}_X{k}"
            if hasattr(QT, name):
                base_of[getattr(QT, name)] = (getattr(QT, b), k)
    out = []
    for t in reader.tensors:
        if t.tensor_type not in base_of:
            out.append((t.name, t.tensor_type, t.data, None))
            continue
        base, k = base_of[t.tensor_type]
        _, bsz = gguf.GGML_QUANT_SIZES[base]
        m = STEM.match(t.name)
        if not m:
            raise ValueError(f"{t.name}: packed tensor without a .weight name")
        rows = t.data.shape[0] if t.data.ndim > 1 else 1
        blocks = t.data.reshape(-1, k, bsz)
        for p in range(k):
            name = t.name if p == 0 else f"{m.group('stem')}{p + 1}{m.group('suffix')}"
            out.append((name, base, blocks[:, p, :].reshape(rows, -1), t))
    return out


def unpack(src: Path, dst, order: Path | None = None) -> dict:
    import gguf
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    items = unpacked_tensors(reader, gguf)
    by_name = {n: (n, ty, d, src_t) for n, ty, d, src_t in items}
    if order:
        names = [x.strip() for x in Path(order).read_text().splitlines() if x.strip()]
        if sorted(names) != sorted(by_name):
            raise ValueError("the order file does not list exactly the unpacked tensors")
    else:
        names = [n for n, _, _, s in items if s is None or n == s.name] + \
                [n for n, _, _, s in items if s is not None and n != s.name]
    sink = HashSink() if dst is None else None
    writer = gguf.GGUFWriter(None if sink else dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)
    for n in names:
        _, ty, d, _ = by_name[n]
        writer.add_tensor_info(n, d.shape, d.dtype, d.nbytes, ty)
    if sink:
        writer.fout = [sink]
        writer.path = None
        writer.state = gguf.gguf_writer.WriterState.EMPTY
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    tensor_hashes = {}
    for n in names:
        _, ty, d, _ = by_name[n]
        writer.write_tensor_data(Raw(d), tensor_endianess=reader.endianess)
        shape = tuple(reversed(_ggml_shape(d, ty, gguf)))
        h = hashlib.sha256(f"{n}|{int(ty)}|{','.join(map(str, shape))}|".encode())
        h.update(d.tobytes())
        tensor_hashes[n] = h.hexdigest()
    writer.close()
    res = {"tensors": len(names), "tensor_digest": gguf_hash.tensor_digest(tensor_hashes), "hashes": tensor_hashes}
    if sink:
        res["file_sha256"] = sink.h.hexdigest()
        res["bytes"] = sink.n
    return res


def _ggml_shape(d, ty, gguf) -> tuple:
    """ggml shape (innermost last, as stored in numpy rows) of a raw block array: elements per row x rows."""
    bs, bsz = gguf.GGML_QUANT_SIZES[ty]
    if d.dtype.itemsize == 1 and (bs, bsz) != (1, 1):
        cols = d.shape[-1] // bsz * bs
        return (*d.shape[:-1], cols)
    return tuple(d.shape)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("packed", type=Path)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--order", type=Path)
    ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--compare", type=Path, help="original separate-plane GGUF: compare tensor by tensor")
    ap.add_argument("--gguf-py", type=Path, help="gguf-py directory of the release engine")
    a = ap.parse_args(argv)
    if a.gguf_py:
        sys.path.insert(0, str(a.gguf_py))
    if not a.hash_only and not a.out:
        ap.error("give OUT or --hash-only")
    res = unpack(a.packed, None if a.hash_only else a.out, a.order)
    print(f"{res['tensors']} tensors, tensor digest {res['tensor_digest']}"
          + (f", file sha256 {res['file_sha256']} ({res['bytes']} bytes)" if "file_sha256" in res else ""))
    if a.compare:
        ref = gguf_hash.tensor_hashes(a.compare)
        c = gguf_hash.compare(ref, res["hashes"])
        bad = sum(len(v) for v in c.values())
        print(f"compare with {a.compare.name}: {len(ref)} tensors, {bad} differences {c if bad else ''}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
