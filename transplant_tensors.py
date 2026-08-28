#!/usr/bin/env python3
"""TRANSPLANT TENSORS — replace selected tensors with the DONOR's copy.

The missing third tool of the family:
  - promote_tensors.py re-encodes values already in the file (useless for
    quality — its own docstring explains the retraction);
  - gguf_surgeon.py patches same-size bytes in place;
  - THIS one rewrites the file taking --match tensors byte-verbatim from a
    SECOND file (the donor, presumably higher precision), everything else
    from the source. Optionally --drop tensors that stop making sense after
    the transplant (e.g. a second plane optimized as the residual of the
    ternary tensor being replaced — adding it on top of the donor's Q4
    would double-count).

Usage:
  transplant_tensors.py SRC.gguf DONOR.gguf DST.gguf \
      --match ffn_down_exps.weight [--match ...] [--drop ffn_down_exps2] \
      [--drop-key qwen35moe.expert_count2]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/angelo/build-llamacpp-tq1/gguf-py")
from gguf import GGUFReader, GGUFWriter

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("donor"); ap.add_argument("dst")
    ap.add_argument("--match", action="append", required=True,
                    help="substring: these tensors come from the DONOR")
    ap.add_argument("--drop", action="append", default=[],
                    help="substring: these tensors are omitted entirely")
    ap.add_argument("--drop-key", action="append", default=[],
                    help="KV keys to omit from the header")
    A = ap.parse_args()

    rs = GGUFReader(A.src); rd = GGUFReader(A.donor)
    ts = {t.name: t for t in rs.tensors}; td = {t.name: t for t in rd.tensors}
    arch = None
    for f in rs.fields.values():
        if f.name == "general.architecture":
            arch = str(f.contents())
    w = GGUFWriter(path=None, arch=arch)
    for f in rs.fields.values():
        if f.name.startswith(("GGUF.", "split.")) or f.name in A.drop_key:
            continue
        try:
            w.add_key_value(f.name, f.contents(), f.types[0],
                            sub_type=f.types[-1] if len(f.types) > 1 else None)
        except Exception:
            pass

    plan = []
    n_don = n_drop = 0
    for name in sorted(ts):
        if any(s in name for s in A.drop):
            n_drop += 1
            continue
        if any(s in name for s in A.match):
            assert name in td, f"{name}: assente nel donatore"
            t = td[name]; n_don += 1
        else:
            t = ts[name]
        d = np.asarray(t.data)
        w.add_tensor_info(name, list(d.shape), d.dtype, int(d.nbytes),
                          raw_dtype=t.tensor_type)
        plan.append((name, t))
    log(f"{len(plan)} tensori · {n_don} dal donatore · {n_drop} eliminati")
    assert n_don > 0, "nessun tensore combacia con --match"

    w.open_output_file(Path(A.dst))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for i, (name, t) in enumerate(plan):
        w.write_tensor_data(np.asarray(t.data))
        if i % 200 == 0:
            log(f"  {i}/{len(plan)}")
    w.close()
    log(f"🏁 scritto {A.dst}")


if __name__ == "__main__":
    main()
