#!/usr/bin/env python3
"""dequant_cache — on-disk cache of the student's dequantized tables (embedding and head), 2026-10-01, optimized
tt5 branch (TT5 audit, fix 9). NEW file.

## Why
`Student._sum_planes` dequantizes token_embd and output (Q5_K + 2/3 ternary planes in R2B3m) with single-threaded
numpy: 11 min 47 s at TT5 startup on 2026-10-01, 21 min on 2026-09-30 [M, registry]. Embedding and head are NOT
trained: at every start/resume the result is the same array. Here it is written once (fp32 .npy) and read back.

## Bit-exact
- The key is the sha256 of the RAW BYTES of all the table's planes (name, type, shape, GGUF data) + the sha256 of
  `dequant_gguf_torch.py` (the dequantizing code) + a version of this recipe (sum of the planes in the engine's order,
  fp32). A different GGUF, one extra plane or a changed dequantizer = a different key = recomputation.
- The .npy file preserves dtype, shape and bytes; next to it a .json with the sha256 of the array bytes. On reading
  the sha is re-checked: if it does not match (corrupted file) it is recomputed and rewritten. Atomic write (.partial,
  fsync, rename).
- Off (E2E_CACHE_DEQUANT variable missing or empty) = the usual `_sum_planes`, identical.

Test: `e2e_test_cache_dequant.py` (sha256 of the array from the cache == the computed one, byte for byte, for
token_embd and output of R2B3m).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

VERSION = "sum_planes_fp32_v1"
_HERE = Path(__file__).resolve().parent


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for piece in iter(lambda: f.read(1 << 24), b""):
            h.update(piece)
    return h.hexdigest()


def sha_array(a: np.ndarray) -> str:
    """sha256 of the array's dtype, shape and bytes (C order)."""
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{tuple(a.shape)}|".encode())
    c = np.ascontiguousarray(a)
    mv = memoryview(c.reshape(-1).view(np.uint8))
    for i in range(0, len(mv), 1 << 26):
        h.update(mv[i:i + (1 << 26)])
    return h.hexdigest()


def key(tens: dict, planes: list) -> str:
    """sha256 of the raw bytes of the planes (in summation order) + dequantizer code + version."""
    h = hashlib.sha256()
    h.update(VERSION.encode())
    h.update(_sha_file(_HERE.parent / "ternary" / "gguf_dequant.py").encode())
    for name in planes:
        t = tens[name]
        raw = np.ascontiguousarray(np.asarray(t.data))
        h.update(f"|{name}|{t.tensor_type.name}|{tuple(int(x) for x in raw.shape)}|{raw.dtype.str}|".encode())
        mv = memoryview(raw.reshape(-1).view(np.uint8))
        for i in range(0, len(mv), 1 << 26):
            h.update(mv[i:i + (1 << 26)])
    return h.hexdigest()


def sum_planes(tens: dict, name_g: str, planes: list, folder: Path, compute: Callable[[], np.ndarray],
                log: Callable = print) -> np.ndarray:
    """The array from `compute()` (the usual dequantization), taken from the cache if present and intact."""
    folder.mkdir(parents=True, exist_ok=True)
    k = key(tens, planes)
    # 2026-10-01: explicit names (name_g contains a dot: with_suffix would have cut "weight-<key>")
    p_npy, p_js = folder / f"{name_g}-{k[:24]}.npy", folder / f"{name_g}-{k[:24]}.json"
    if p_npy.exists() and p_js.exists():
        try:
            meta = json.loads(p_js.read_text())
            if meta.get("key") == k:
                a = np.load(p_npy, allow_pickle=False)
                if sha_array(a) == meta.get("sha256_array"):
                    log(f"cache dequant: {name_g} read from {p_npy.name} ({a.nbytes / 2 ** 30:.2f} GiB, sha bit-exact)")
                    return a
                log(f"cache dequant: {p_npy.name} with sha different from the registry: recomputing")
        except (OSError, ValueError) as e:
            log(f"cache dequant: {p_npy.name} unreadable ({e!r}): recomputing")
    a = compute()
    sha = sha_array(a)
    tmp = p_npy.with_name(p_npy.name + ".partial")
    with open(tmp, "wb") as f:
        np.save(f, a, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p_npy)
    tmp_j = p_js.with_name(p_js.name + ".partial")
    tmp_j.write_text(json.dumps({"key": k, "name": name_g, "planes": list(planes), "dtype": a.dtype.str,
                                 "shape": list(a.shape), "sha256_array": sha, "version": VERSION}, indent=1))
    os.replace(tmp_j, p_js)
    log(f"cache dequant: {name_g} written to {p_npy.name} ({a.nbytes / 2 ** 30:.2f} GiB)")
    return a
