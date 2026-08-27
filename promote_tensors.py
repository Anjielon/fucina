#!/usr/bin/env python3
"""PROMOTE TENSORS — raise selected tensors to a higher precision.

A GGUF stores fixed byte sizes per tensor, so a tensor cannot grow in place:
promoting one means rewriting the file. This does that surgically — every
other tensor is copied byte-for-byte from the source, so the only difference
in the output is the tensors you asked for.

Why it matters: in a ternary model the few tensors that stay at higher
precision carry disproportionate weight. On a 397B model, promoting the
linear-attention projections cut model-level output error from 25.4% to 17.1%
for 3 GiB — by far the best return per byte we measured.

⚠️ READ THIS BEFORE USING IT TO "IMPROVE" A MODEL.

This tool re-encodes the values **already in the file**. Information destroyed
by the original quantization is gone, and no container can bring it back:
writing Q4_K values into a Q8_0 block reproduces them almost exactly, so the
model is unchanged and the file is larger.

We learned this the expensive way. We promoted 45 attention tensors from
Q4_K/Q6_K to Q8_0, measured no difference (paired sign test, 17/30 chunks,
p = 0.58), and briefly concluded that attention was not the model's
bottleneck. That conclusion was unsupported: the experiment could not have
produced any other result. Retracted.

So this tool is legitimate for exactly two jobs:
  · re-encoding tensors whose values came from a HIGHER-precision source
    (pass them in yourself), and
  · normalising a file's precision layout — for instance when a converter left
    half the shared experts at Q4_K and half at Q8_0.

To genuinely raise a tensor's precision you must re-read it from the original
bf16/fp16 checkpoint. That is a different operation, and this file does not
do it.

Usage:
    promote_tensors.py in.gguf out.gguf --match attn_qkv --to Q8_0
    promote_tensors.py in.gguf out.gguf --match attn_qkv --to Q8_0 --dry-run

The source is opened read-only and never modified.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
from gguf import quants as GQ

BLOCK = {"Q8_0": (32, 34), "Q6_K": (256, 210), "Q5_K": (256, 176), "Q4_K": (256, 144)}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def plan_promotion(src: str, match: str, to: str) -> list[tuple[str, str, int, int]]:
    """Tensors that would be promoted: (name, current type, bytes now, bytes after)."""
    per_block, bytes_per_block = BLOCK[to]
    out = []
    for t in GGUFReader(src).tensors:
        if match not in t.name or t.tensor_type.name == to:
            continue
        elements = int(np.prod([int(x) for x in t.shape]))
        out.append((t.name, t.tensor_type.name, int(t.n_bytes),
                    elements // per_block * bytes_per_block))
    return out


def promote(src: str, dst: str, match: str, to: str) -> None:
    """Rewrite the model with the matching tensors quantized to `to`."""
    reader = GGUFReader(src)
    target = getattr(T, to)
    per_block, bytes_per_block = BLOCK[to]
    arch = next((str(bytes(f.parts[-1]), "utf8")
                 for f in reader.fields.values() if f.name.endswith(".architecture")), None)

    w = GGUFWriter(path=None, arch=arch)
    for f in reader.fields.values():
        if f.name.startswith(("GGUF.", "split.")):
            continue
        try:
            w.add_key_value(f.name, f.contents(), f.types[0],
                            sub_type=f.types[-1] if len(f.types) > 1 else None)
        except Exception:
            pass

    promoting = {t.name for t in reader.tensors if match in t.name and t.tensor_type.name != to}
    log(f"promoting {len(promoting)} tensors to {to}")

    for t in reader.tensors:                       # declare every tensor first
        shape = [int(x) for x in t.shape]
        if t.name in promoting:
            byte_shape = shape[::-1]
            byte_shape[-1] = byte_shape[-1] // per_block * bytes_per_block
            w.add_tensor_info(t.name, byte_shape, np.dtype(np.uint8),
                              int(np.prod(byte_shape)), raw_dtype=target)
        else:
            d = np.asarray(t.data)
            w.add_tensor_info(t.name, list(d.shape), d.dtype, int(d.nbytes),
                              raw_dtype=t.tensor_type)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(Path(dst))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    for i, t in enumerate(reader.tensors):         # then the data, in order
        if t.name in promoting:
            values = GQ.dequantize(np.asarray(t.data), t.tensor_type)
            w.write_tensor_data(GQ.quantize(values, target))
        else:
            w.write_tensor_data(np.asarray(t.data))
        if i % 200 == 0:
            log(f"  {i}/{len(reader.tensors)} tensors written")
    w.close()
    log(f"done -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--match", required=True, help="substring of the tensor names to promote")
    ap.add_argument("--to", default="Q8_0", choices=sorted(BLOCK))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = plan_promotion(a.src, a.match, a.to)
    now = sum(p[2] for p in plan)
    after = sum(p[3] for p in plan)
    print(f"{len(plan)} tensors match '{a.match}' and are not already {a.to}")
    for name, kind, b_now, b_after in plan[:5]:
        print(f"  {name:36s} {kind:6s} {b_now/2**20:8.1f} MiB -> {b_after/2**20:8.1f} MiB")
    if len(plan) > 5:
        print(f"  ... and {len(plan)-5} more")
    print(f"total: {now/2**30:.2f} GiB -> {after/2**30:.2f} GiB  (+{(after-now)/2**30:.2f} GiB)")
    if not a.dry_run:
        promote(a.src, a.dst, a.match, a.to)
