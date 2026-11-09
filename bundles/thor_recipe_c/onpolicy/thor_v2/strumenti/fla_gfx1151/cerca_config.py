#!/usr/bin/env python3
"""Cerca OFFLINE (compila_offline.py, nessuna GPU) le configurazioni a scratch zero dei kernel GDN di fla su gfx1151."""
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
# blocchi regolabili da Config (gli altri kernel ricevono BK/BV dal wrapper: si regola solo num_warps)
BLOCCHI = {
    "chunk_fwd_kernel_o": ("BK", "BV"),
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ("BV",),
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": ("BV",),
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": ("BK",),
}
ALTRI = ["chunk_bwd_kernel_dqkwg", "chunk_bwd_kernel_dv_local", "prepare_wy_repr_bwd_kernel", "recompute_w_u_fwd_kernel"]


def main() -> int:
    warps = [int(x) for x in sys.argv[1].split(",")]
    blocchi = [int(x) for x in sys.argv[2].split(",")]
    uscita = Path(sys.argv[3])
    esiti = json.loads(uscita.read_text()) if uscita.exists() else {}
    env = {**os.environ, "HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
    for nw, b in itertools.product(warps, blocchi):
        chiave = f"nw{nw}_b{b}"
        if chiave in esiti:
            continue
        tab = {n: {**{p: b for p in ps}, "num_warps": nw, "num_stages": 1} for n, ps in BLOCCHI.items()}
        tab.update({n: {"num_warps": nw, "num_stages": 1} for n in ALTRI})
        ft = QUI / "esiti" / f"tab_{chiave}.json"
        fj = QUI / "esiti" / f"ris_{chiave}.json"
        ft.parent.mkdir(exist_ok=True)
        ft.write_text(json.dumps(tab))
        r = subprocess.run(["nice", sys.executable, str(QUI / "compila_offline.py"), "--tabella", str(ft), "--json",
                            str(fj)], env=env, capture_output=True, text=True, timeout=3600)
        esiti[chiave] = json.loads(fj.read_text()) if fj.exists() else {"errore": r.stderr[-800:]}
        uscita.write_text(json.dumps(esiti, indent=1))
        print(chiave, "fatto" if fj.exists() else "ERRORE " + r.stderr[-300:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
