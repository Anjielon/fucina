#!/usr/bin/env python3
"""S2+S3 CPU stage: writing one arm = copy of the START GGUF with the chosen tensors rewritten IN PLACE (same type,
same shape, same byte count: size identical to the byte, tg identical by construction).

Arms: s2 = only the DENSE tensors of the sweep (C-S2) · s3 = dense + experts (C-S3). Declared caveat: in the single
sweep the dense tensors of layer L+1 were solved on the stream with the experts of layers <= L ALREADY new (plan §1.2:
«the same sweep, with two GGUF outputs»): C-S2 isolates the dense tensors, but it is not the sweep one would obtain
without S3.
Verification (fail-closed, before renaming): size = start to the byte; header identical to the byte; for every tensor
same name/type/shape/offset/byte count; unchosen tensors identical; chosen = intended bytes; trits of every rewritten
tensor in {-1,0,+1} and finite fp16 scales (full decode); list; sha256.
usage: s23_scrivi.py --start G --layers DIR --arm s2|s3 --output FILE.gguf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from fucina.sequential import common_s1b as K  # noqa: E402

EXPERTS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def sha(p, n=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if n is not None:
            h.update(f.read(n))
            return h.hexdigest()
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def collect(d: Path, arm: str, n_layers: int):
    out = {}
    done = sorted(d.glob("strato_*"))
    if len([s for s in done if (s / "DONE").exists()]) != n_layers:
        raise SystemExit(f"finished layers {len([s for s in done if (s / 'DONE').exists()])} != {n_layers}")
    for st in done:
        for f in (st / "tensors").glob("blk.*.npy"):
            name = f.name[:-4]
            if arm == "s2" and name.split(".")[2] in EXPERTS:
                continue
            out[name] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--n-layers", type=int, default=40)
    ap.add_argument("--arm", choices=("s2", "s3"), required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    chosen = collect(Path(a.layers), a.arm, a.n_layers)
    if not chosen:
        raise SystemExit("no tensor to change")
    gguf = K.gp._import_gguf(None)
    rd = gguf.GGUFReader(a.start, "r")
    tens = {t.name: t for t in rd.tensors}
    plane = []
    for name, f in sorted(chosen.items()):
        t = tens[name]
        tid = int(t.tensor_type)
        assert tid in K.FORMAT, (name, tid)
        new = np.load(f)
        raw = np.asarray(t.data)
        assert new.dtype == np.uint8 and new.nbytes == t.n_bytes, (name, new.dtype, new.nbytes, t.n_bytes)
        assert new.reshape(-1).size == raw.reshape(-1).view(np.uint8).size, name
        plane.append((name, int(t.data_offset), new, tid, int(t.shape[0])))
    out = Path(a.output)
    tmp = out.with_suffix(out.suffix + ".partial")
    shutil.copyfile(a.start, tmp)
    with open(tmp, "r+b") as fh:
        for name, off, new, _, _ in plane:
            fh.seek(off)
            fh.write(np.ascontiguousarray(new).tobytes())
    rd2 = gguf.GGUFReader(str(tmp), "r")
    t2 = {t.name: t for t in rd2.tensors}
    assert list(t2) == list(tens), "tensor names/order differ"
    assert os.path.getsize(tmp) == os.path.getsize(a.start), "size differs"
    o = min(int(t.data_offset) for t in rd.tensors)
    assert sha(tmp, o) == sha(a.start, o), "header differs"
    pn = {p[0]: p for p in plane}
    changed, types = [], {}
    for n, x in tens.items():
        y = t2[n]
        assert (int(x.tensor_type), list(x.shape), int(x.data_offset), int(x.n_bytes)) == \
               (int(y.tensor_type), list(y.shape), int(y.data_offset), int(y.n_bytes)), n
        bx = np.asarray(x.data).reshape(-1).view(np.uint8)
        by = np.asarray(y.data).reshape(-1).view(np.uint8)
        if n not in pn:
            assert np.array_equal(bx, by), f"{n} changed but not chosen"
            continue
        _, _, new, tid, C = pn[n]
        assert np.array_equal(by, np.ascontiguousarray(new).reshape(-1)), n
        rows = np.asarray(y.data)
        rows = rows.reshape(-1, rows.shape[-1])
        tr, sc = K.gp.decode_rows(np.ascontiguousarray(rows), tid, C)
        assert np.isin(tr, (-1, 0, 1)).all() and np.isfinite(sc.astype(np.float32)).all(), n
        types[tid] = types.get(tid, 0) + 1
        if not np.array_equal(bx, by):
            changed.append(n)
    os.replace(tmp, out)
    s = sha(out)
    Path(str(out) + ".sha256").write_text(s + "\n")
    fam = {}
    for n in changed:
        f = n.split(".")[2]
        fam[f] = fam.get(f, 0) + 1
    ver = {"ok": True, "arm": a.arm, "start": a.start, "size": os.path.getsize(out), "sha256": s,
           "tensors_changed": len(changed), "per_family": fam, "per_type": types, "list_": changed, "types_identical": True,
           "header": "identical"}
    K.write_json(str(out) + ".verifica.json", ver)
    print(f"arm {a.arm}: {out} {ver['size']} B sha {s[:16]}; {len(changed)} tensors rewritten {fam}; "
          f"types/shapes/offsets identical, header identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
