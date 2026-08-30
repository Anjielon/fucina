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

The second case this exists for: a correction that helps on some layers and
harms on others. `--keep-layers` removes the matched tensors *except* on the
named layers, which lets a measured layer profile be baked into the file:

    strip_tensors.py in.gguf out.gguf --match _exps2 --keep-layers 30-39

⚠️ With `--keep-layers`, do **not** drop the declaring metadata key — the
surviving layers still need it. A model where only some layers carry the
optional family is well-formed as long as the engine treats a missing tensor as
"this layer has no correction" rather than as an error; verify that on a small
model before spending a large run on it.

Every surviving tensor is copied byte-for-byte; nothing is re-quantized. The
source is opened read-only and never modified.

⚠️ Removing a tensor an engine still expects makes the model fail to load. Drop
the metadata key that declares the optional family in the same pass (here
`<arch>.expert_count2`), so the engine stops looking for it.
"""
from __future__ import annotations
import argparse
import re
import sys
import os
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/build-llamacpp/gguf-py")))
from gguf import GGUFReader, GGUFWriter


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def parse_layers(spec: str) -> set[int]:
    """"30-39", "0,5,7", "30-39,44" -> the set of layer indices."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def layer_of(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def should_remove(name: str, match, keep_layers: set[int] | None) -> bool:
    matches = [match] if isinstance(match, str) else list(match)
    if not any(m in name for m in matches):
        return False
    if not keep_layers:
        return True
    layer = layer_of(name)
    # a matched tensor outside any layer block is removed: keeping it would
    # leave a global tensor declaring a family most layers no longer have
    return layer is None or layer not in keep_layers


def plan_strip(src: str, match: str,
               keep_layers: set[int] | None = None
               ) -> tuple[list[tuple[str, int]], int, int]:
    """(name, bytes) of the tensors that would be removed, plus kept/removed totals."""
    removed, kept_bytes, removed_bytes = [], 0, 0
    for t in GGUFReader(src).tensors:
        if should_remove(t.name, match, keep_layers):
            removed.append((t.name, int(t.n_bytes)))
            removed_bytes += int(t.n_bytes)
        else:
            kept_bytes += int(t.n_bytes)
    return removed, kept_bytes, removed_bytes


def strip(src: str, dst: str, match: str, drop_keys: list[str],
          keep_layers: set[int] | None = None) -> None:
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

    keep = [t for t in reader.tensors
            if not should_remove(t.name, match, keep_layers)]
    log(f"keeping {len(keep)} tensors, removing {len(reader.tensors) - len(keep)}")
    if keep_layers:
        matches = [match] if isinstance(match, str) else list(match)
        survivors = sorted({layer_of(t.name) for t in keep
                            if any(m in t.name for m in matches)} - {None})
        log(f"{matches} survive on layers: {survivors}")

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
    ap.add_argument("--match", required=True, action="append",
                    help="substring of the tensor names to remove (repeatable — "
                         "one pass, one output, however many families)")
    ap.add_argument("--drop-key", action="append", default=[],
                    help="substring of a metadata key to remove (repeatable)")
    ap.add_argument("--keep-layers", default=None,
                    help="layers on which matched tensors SURVIVE, e.g. 30-39 "
                         "or 0,5,30-39. Without this, every match is removed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    keep_layers = parse_layers(a.keep_layers) if a.keep_layers else None
    if keep_layers and a.drop_key:
        sys.exit("--keep-layers with --drop-key would remove the metadata key "
                 "that the surviving layers still need. Refusing.")

    removed, kept_bytes, removed_bytes = plan_strip(a.src, a.match, keep_layers)
    print(f"{len(removed)} tensors match '{a.match}'"
          + (f" outside layers {a.keep_layers}" if keep_layers else ""))
    for name, n in removed[:5]:
        print(f"  {name:44s} {n/2**20:9.1f} MiB")
    if len(removed) > 5:
        print(f"  ... and {len(removed)-5} more")
    print(f"kept {kept_bytes/2**30:.2f} GiB · removed {removed_bytes/2**30:.2f} GiB")
    if not a.dry_run:
        strip(a.src, a.dst, a.match, a.drop_key, keep_layers)
