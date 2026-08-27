#!/usr/bin/env python3
"""STRIP TENSORS — remove tensors a model no longer needs, and the metadata that
declares them.

The case this exists for: a forge run that predates a fix can leave a file
carrying an *orphaned* second plane — corrections optimized against a first
plane the file no longer contains. Such a plane is not merely useless, it is
harmful when an engine enables it, and it occupies VRAM whether or not the
graph reads it, because tensors are loaded before the graph is built.

Removing them gives back the bytes at exactly zero quality cost, and removes
the possibility that a future engine switches the broken correction back on.

    strip_tensors.py in.gguf out.gguf --match _exps2 --drop-key expert_count2
    strip_tensors.py in.gguf out.gguf --match _exps2 --dry-run

Every surviving tensor is copied byte-for-byte; nothing is re-quantized. The
source is opened read-only and never modified.

⚠️ Removing a tensor an engine still expects makes the model fail to load. Drop
the metadata key that declares the optional family in the same pass (here
`<arch>.expert_count2`), so the engine stops looking for it.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
from gguf import GGUFReader, GGUFWriter


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def plan_strip(src: str, match: str) -> tuple[list[tuple[str, int]], int, int]:
    """(name, bytes) of the tensors that would be removed, plus kept/removed totals."""
    removed, kept_bytes, removed_bytes = [], 0, 0
    for t in GGUFReader(src).tensors:
        if match in t.name:
            removed.append((t.name, int(t.n_bytes)))
            removed_bytes += int(t.n_bytes)
        else:
            kept_bytes += int(t.n_bytes)
    return removed, kept_bytes, removed_bytes


def strip(src: str, dst: str, match: str, drop_keys: list[str]) -> None:
    reader = GGUFReader(src)
    arch = next((str(bytes(f.parts[-1]), "utf8")
                 for f in reader.fields.values() if f.name.endswith(".architecture")), None)

    w = GGUFWriter(path=None, arch=arch)
    dropped_keys = []
    for f in reader.fields.values():
        if f.name.startswith(("GGUF.", "split.")):
            continue
        if any(k in f.name for k in drop_keys):
            dropped_keys.append(f.name)
            continue
        try:
            w.add_key_value(f.name, f.contents(), f.types[0],
                            sub_type=f.types[-1] if len(f.types) > 1 else None)
        except Exception:
            pass
    if dropped_keys:
        log(f"metadata keys dropped: {', '.join(dropped_keys)}")

    keep = [t for t in reader.tensors if match not in t.name]
    log(f"keeping {len(keep)} tensors, removing {len(reader.tensors) - len(keep)}")

    for t in keep:                                  # declare every tensor first
        d = np.asarray(t.data)
        w.add_tensor_info(t.name, list(d.shape), d.dtype, int(d.nbytes),
                          raw_dtype=t.tensor_type)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(Path(dst))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    for i, t in enumerate(keep):                    # then the data, in order
        w.write_tensor_data(np.asarray(t.data))
        if i % 200 == 0:
            log(f"  {i}/{len(keep)} tensors written")
    w.close()
    log(f"done -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--match", required=True, help="substring of the tensor names to remove")
    ap.add_argument("--drop-key", action="append", default=[],
                    help="substring of a metadata key to remove (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    removed, kept_bytes, removed_bytes = plan_strip(a.src, a.match)
    print(f"{len(removed)} tensors match '{a.match}'")
    for name, n in removed[:5]:
        print(f"  {name:44s} {n/2**20:9.1f} MiB")
    if len(removed) > 5:
        print(f"  ... and {len(removed)-5} more")
    print(f"kept {kept_bytes/2**30:.2f} GiB · removed {removed_bytes/2**30:.2f} GiB")
    if not a.dry_run:
        strip(a.src, a.dst, a.match, a.drop_key)
