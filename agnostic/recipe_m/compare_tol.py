#!/usr/bin/env python3
"""Tolerance comparisons for the non-bitwise steps of the objective-C D2 recipe.

  npy-dir  <new_dir> <recorded_dir> [--glob '*.npy'] [--rtol 1e-5]
      Arrays with identical bytes count as identical; for the others the relative Frobenius error
      ||new - rec|| / ||rec|| is computed (packed uint8 trit blocks: share of identical bytes instead).
      Exit 0 if every array is identical or within rtol.
  pt <new.pt> <recorded.pt> [--rtol 1e-5]
      Same on every floating tensor found (recursively) in two torch checkpoints.
  registry <new_registro.log> <recorded_registro.log> [--keys kl_held,kl4_held] [--rtol 0.01]
      Held-out metrics at the evaluation steps of a training registry (JSON lines after the timestamp).
  mmlu <new_log> <recorded_log> [--max-delta 1.0] [--min-p 0.05]
      MMLU-2000 'Final result' and an exact McNemar test on the paired MCLP lines
      (tony-forgia/thor_v2/strumenti/mcnemar_mclp.py). Exit 0 if |delta| <= max-delta and p >= min-p.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

MCN = "/home/angelo/tony-forgia/thor_v2/strumenti/mcnemar_mclp.py"


def _rel(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / (nb if nb > 0 else 1.0)


def cmd_npy_dir(a) -> int:
    import numpy as np
    new, rec = Path(a.new), Path(a.rec)
    names = sorted(p.relative_to(rec).as_posix() for p in rec.rglob("*") if p.is_file()
                   and fnmatch.fnmatch(p.name, a.glob))
    same = within = worse = missing = 0
    worst = (0.0, "")
    for n in names:
        q = new / n
        if not q.exists():
            missing += 1
            continue
        x, y = np.load(q, allow_pickle=False), np.load(rec / n, allow_pickle=False)
        if x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes():
            same += 1
            continue
        if x.dtype == np.uint8:                     # packed ternary blocks: share of differing bytes
            e = float((x != y).mean()) if x.shape == y.shape else 1.0
        else:
            e = _rel(x, y) if x.shape == y.shape else math.inf
        worst = max(worst, (e, n))
        if e <= a.rtol:
            within += 1
        else:
            worse += 1
    print(json.dumps({"files": len(names), "identical": same, "within_tol": within, "outside_tol": worse,
                      "missing": missing, "worst": {"error": worst[0], "file": worst[1]}, "rtol": a.rtol}))
    return 0 if (worse == 0 and missing == 0) else 1


def _floats(obj, prefix=""):
    import torch
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _floats(v, f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _floats(v, f"{prefix}[{i}]")


def cmd_pt(a) -> int:
    import torch
    x = dict(_floats(torch.load(a.new, map_location="cpu", weights_only=False)))
    y = dict(_floats(torch.load(a.rec, map_location="cpu", weights_only=False)))
    keys = sorted(set(x) | set(y))
    bad, worst = [], (0.0, "")
    for k in keys:
        if k not in x or k not in y or x[k].shape != y[k].shape:
            bad.append(k)
            continue
        e = _rel(x[k].double().numpy(), y[k].double().numpy())
        worst = max(worst, (e, k))
        if e > a.rtol:
            bad.append(k)
    print(json.dumps({"tensors": len(keys), "outside_tol": len(bad), "worst": {"error": worst[0], "key": worst[1]},
                      "rtol": a.rtol, "first_bad": bad[:5]}))
    return 0 if not bad else 1


def _registry(p: Path) -> dict:
    out = {}
    for line in p.read_text(errors="replace").splitlines():
        m = re.search(r"\]\s*(\{.*\})\s*$", line)
        if not m:
            continue
        try:
            d = json.loads(m.group(1))
        except ValueError:
            continue
        if "passo" in d:
            out[d["passo"]] = d
    return out


def cmd_registry(a) -> int:
    x, y = _registry(Path(a.new)), _registry(Path(a.rec))
    keys = a.keys.split(",")
    rows, bad = [], 0
    for s in sorted(set(x) & set(y)):
        for k in keys:
            if k in x[s] and k in y[s]:
                e = abs(x[s][k] - y[s][k]) / max(abs(y[s][k]), 1e-12)
                rows.append({"step": s, "key": k, "new": x[s][k], "recorded": y[s][k], "rel": e})
                bad += e > a.rtol
    print(json.dumps({"compared": len(rows), "outside_tol": bad, "rtol": a.rtol, "rows": rows[-12:]}))
    return 0 if rows and not bad else 1


def _acc(p: Path):
    m = re.findall(r"Final result: ([0-9.]+)", p.read_text(errors="replace"))
    return float(m[-1]) if m else None


def cmd_mmlu(a) -> int:
    an, ar = _acc(Path(a.new)), _acc(Path(a.rec))
    js = Path(a.new).with_suffix(".mcnemar_vs_recorded.json")
    subprocess.run([sys.executable, MCN, a.rec, a.new, "--json", str(js)], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)
    p = None
    if js.exists():
        p = json.loads(js.read_text()).get("p_mcnemar_esatto")
    ok = an is not None and ar is not None and abs(an - ar) <= a.max_delta and (p is None or p >= a.min_p)
    print(json.dumps({"new": an, "recorded": ar, "delta": None if an is None or ar is None else round(an - ar, 4),
                      "p_mcnemar_exact": p, "max_delta": a.max_delta, "min_p": a.min_p, "pass": ok}))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("npy-dir"); s.add_argument("new"); s.add_argument("rec")
    s.add_argument("--glob", default="*.npy"); s.add_argument("--rtol", type=float, default=1e-5)
    s = sp.add_parser("pt"); s.add_argument("new"); s.add_argument("rec"); s.add_argument("--rtol", type=float, default=1e-5)
    s = sp.add_parser("registry"); s.add_argument("new"); s.add_argument("rec")
    s.add_argument("--keys", default="kl_held,kl4_held"); s.add_argument("--rtol", type=float, default=0.01)
    s = sp.add_parser("mmlu"); s.add_argument("new"); s.add_argument("rec")
    s.add_argument("--max-delta", type=float, default=1.0); s.add_argument("--min-p", type=float, default=0.05)
    a = ap.parse_args()
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
    return {"npy-dir": cmd_npy_dir, "pt": cmd_pt, "registry": cmd_registry, "mmlu": cmd_mmlu}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
