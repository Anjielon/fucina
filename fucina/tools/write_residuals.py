"""CLI adapter of the "CARDAN GGUF" step: the same call to ``e1_scrivi_residui_gguf.scrivi`` (the external residual
writer), with source and destination as arguments instead of constants.

Usage: python -m fucina.tools.write_residuals --writer-dir DIR --src X.gguf --dir CARDAN_rN --out Y.gguf [--layers 40]"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(prog="fucina.tools.write_residuals")
    ap.add_argument("--writer-dir", required=True, help="directory containing e1_scrivi_residui_gguf.py")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=40)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.writer_dir)
    import e1_scrivi_residui_gguf as writer
    info = json.load(open(f"{a.dir}/info.json"))["tensori"]
    new = []
    for layer in range(a.layers):
        z = np.load(f"{a.dir}/strato_{layer:02d}.npz")
        for n in sorted(z.files):
            new.append((n, int(info[n]["type"]), info[n]["ne"], z[n].reshape(-1)))
    rep = writer.scrivi(a.src, a.out, new, None, 10.0)
    json.dump(rep, open(a.out + ".modifica.json", "w"), indent=1)
    print("wrote", len(new), "tensors in", rep["secondi"], "s")


if __name__ == "__main__":
    main()
