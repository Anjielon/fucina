#!/usr/bin/env python3
"""dθ del primo micro-lotto di `e2e_allena.py misura` (file di _salva_dtheta) fra varianti del GDN, come il cancello G1:
tra le scale vive (d0 != 0) col gradiente di riferimento non nullo, quota di dθ nulli o di segno opposto (soglia 1e-3),
errore relativo mediano/p99 e della norma. Solo CPU. Uso: confronta_dtheta.py CARTELLA [riferimento] (27/9/2026)"""
import sys
from pathlib import Path

import torch


def confronta(fa: Path, fb: Path) -> dict:
    a, b = torch.load(fa, map_location="cpu"), torch.load(fb, map_location="cpu")
    assert a["chiavi"] == b["chiavi"]
    vive_nz = nulli = rotti = 0
    rel, na, nd = [], 0.0, 0.0
    for ga, gb, v in zip(a["grad"], b["grad"], b["vive"]):
        ga, gb = ga.double(), gb.double()
        m = v & (gb != 0)
        vive_nz += int(m.sum())
        nulli += int((m & (ga == 0)).sum())
        rotti += int((m & (ga != 0) & (torch.sign(ga) != torch.sign(gb))).sum())
        rel.append(((ga - gb).abs() / gb.abs())[m].float())
        nd += float((ga - gb).pow(2).sum())
        na += float(gb.pow(2).sum())
    r = torch.cat(rel)
    r = r[torch.randperm(r.numel())[:5_000_000]]
    q = torch.quantile(r, torch.tensor([0.5, 0.99]))
    return {"vive_nz": vive_nz, "nulli": nulli, "segno_opposto": rotti, "quota": (nulli + rotti) / max(vive_nz, 1),
            "rel_mediano": float(q[0]), "rel_p99": float(q[1]), "rel_norma": (nd / na) ** 0.5}


def main() -> int:
    d = Path(sys.argv[1])
    rif = sys.argv[2] if len(sys.argv) > 2 else "ripiego"
    for strati in ("0", "0-1"):
        fb = d / f"dtheta_{rif}_strati{strati}.pt"
        for fa in sorted(d.glob(f"dtheta_*_strati{strati}.pt")):
            if fa == fb:
                continue
            e = confronta(fa, fb)
            print(f"strati {strati} {fa.name.split('_')[1]:9s} vs {rif}: nulli {e['nulli']} segno opposto "
                  f"{e['segno_opposto']} su {e['vive_nz']} -> quota {e['quota']:.2e} "
                  f"({'OK' if e['quota'] <= 1e-3 else 'OLTRE'} 1e-3) · rel mediano {e['rel_mediano']:.2e} "
                  f"p99 {e['rel_p99']:.2e} · norma {e['rel_norma']:.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
