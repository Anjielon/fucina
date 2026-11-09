"""Check the external inputs of a replay recipe against their recorded hashes.

Usage: python -m fucina.replay.check_inputs RECIPE [--set NAME=PATH ...] [--out FILE.json]
Exit status 0 when every external input is present and matches; the report lists every input.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .gguf_hash import file_sha256
from .recipe import load, values_for
from .runner import configured_roots


def check(recipe, values: dict) -> dict:
    out = {}
    base = recipe.source.parent
    for k, i in recipe.inputs.items():
        if i.kind == "derived":
            continue
        p = Path(values[f"path.{k}"])
        if not p.exists():
            out[k] = {"ok": False, "detail": f"missing {p}"}
        elif i.sha256:
            got = file_sha256(p)
            out[k] = {"ok": got == i.sha256, "detail": got}
        elif i.manifest:
            bad = []
            for line in (base / i.manifest).read_text().splitlines():
                if line.strip():
                    h, rel = line.split(None, 1)
                    f = p / rel.strip()
                    if not f.exists() or file_sha256(f) != h:
                        bad.append(rel.strip())
            out[k] = {"ok": not bad, "detail": f"{len(bad)} files differ" + (f": {bad[:5]}" if bad else "")}
        else:
            out[k] = {"ok": True, "detail": "present (no recorded hash)"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recipe")
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    r = load(a.recipe)
    res = check(r, values_for(r, configured_roots(r, a.sets)))
    for k, v in res.items():
        print(f"{'ok ' if v['ok'] else 'BAD'} {k:<22} {v['detail']}")
    ok = all(v["ok"] for v in res.values())
    if a.out and ok:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"when": time.strftime("%FT%T"), "inputs": res}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
