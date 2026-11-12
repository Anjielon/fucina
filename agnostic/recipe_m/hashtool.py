#!/usr/bin/env python3
"""Content hashing and verification for the objective-C D2 recipe (stdlib only).

Commands
  file   <path>                         print the sha256 of one file
  tree   <dir> [--out MANIFEST]         per-file sha256 manifest of a directory (sorted, relative paths)
  cmp    <manifest_a> <manifest_b>      compare two manifests; exit 0 only if identical
  check  <path> <recorded>              <recorded> is a 64-hex sha256 (file) or a manifest path (directory);
                                        exit 0 = identical, 1 = different, 2 = missing
  verify-gguf-tensors <gguf> <provenance.json>
                                        compare the per-tensor 'sha256_byte' entries of a *.provenienza.json
                                        written by patch_gguf.py with the tensors of <gguf>

A cache keyed by (absolute path, size, mtime_ns) avoids re-hashing unchanged multi-GiB files:
$RECIPE_HASH_CACHE (default ~/.cache/fucina/sha256_cache.json).
Reads are done in 16 MiB blocks; the caller is expected to run under `nice -n 19 ionice -c3`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

BLOCK = 16 << 20
CACHE = Path(os.environ.get("RECIPE_HASH_CACHE", Path.home() / ".cache/fucina/sha256_cache.json"))
EXCLUDE = {"__pycache__"}


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(c))
    os.replace(tmp, CACHE)


def sha_file(p: Path, cache: dict | None = None) -> str:
    p = p.resolve()
    st = p.stat()
    key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    if cache is not None and key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(BLOCK)
            if not b:
                break
            h.update(b)
    d = h.hexdigest()
    if cache is not None:
        cache[key] = d
    return d


def manifest(d: Path, cache: dict | None = None) -> list[str]:
    rows = []
    for root, dirs, files in os.walk(d, followlinks=True):
        dirs[:] = sorted(x for x in dirs if x not in EXCLUDE)
        for n in sorted(files):
            p = Path(root) / n
            rows.append(f"{sha_file(p, cache)}  {p.relative_to(d)}")
    return sorted(rows, key=lambda r: r[66:])


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    cache = _load_cache()
    try:
        if cmd == "file":
            print(sha_file(Path(args[0]), cache))
            return 0
        if cmd == "tree":
            rows = manifest(Path(args[0]), cache)
            text = "\n".join(rows) + "\n"
            if "--out" in args:
                Path(args[args.index("--out") + 1]).write_text(text)
            else:
                sys.stdout.write(text)
            return 0
        if cmd == "cmp":
            a, b = (Path(x).read_text().splitlines() for x in args[:2])
            if a == b:
                print(f"IDENTICAL ({len(a)} files)")
                return 0
            sa, sb = set(a), set(b)
            print(f"DIFFERENT: {len(sa - sb)} only/changed in A, {len(sb - sa)} only/changed in B")
            for r in sorted(sa ^ sb)[:20]:
                print("  ", r)
            return 1
        if cmd == "check":
            p, rec = Path(args[0]), args[1]
            if not p.exists():
                print(f"MISSING {p}")
                return 2
            if p.is_dir():
                got = manifest(p, cache)
                want = Path(rec).read_text().splitlines()
                ok = got == want
                print(f"{'MATCH' if ok else 'DIFFER'} {p} ({len(got)} files vs {len(want)} recorded)")
                return 0 if ok else 1
            got = sha_file(p, cache)
            ok = got == rec.strip()
            print(f"{'MATCH' if ok else 'DIFFER'} {p} sha256 {got}" + ("" if ok else f" (recorded {rec.strip()})"))
            return 0 if ok else 1
        if cmd == "verify-gguf-tensors":
            sys.path.insert(0, os.environ.get("GGUF_PY", "/mnt/models/build/thor-rexppiani2/gguf-py"))
            import numpy as np  # noqa: E402  (only this sub-command needs numpy/gguf)
            from gguf import GGUFReader  # noqa: E402
            prov = json.loads(Path(args[1]).read_text())
            rec = prov.get("tensori", prov)
            t = {x.name: x for x in GGUFReader(args[0]).tensors}
            bad = [n for n, e in rec.items() if isinstance(e, dict) and "sha256_byte" in e and
                   hashlib.sha256(np.asarray(t[n].data).view(np.uint8).tobytes()).hexdigest() != e["sha256_byte"]]
            n = sum(1 for e in rec.values() if isinstance(e, dict) and "sha256_byte" in e)
            print(f"{'MATCH' if not bad else 'DIFFER'}: {n - len(bad)}/{n} recorded tensors identical" +
                  (f"; first differing {bad[:5]}" if bad else ""))
            return 0 if not bad else 1
        print(f"unknown command {cmd}")
        return 2
    finally:
        _save_cache(cache)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
