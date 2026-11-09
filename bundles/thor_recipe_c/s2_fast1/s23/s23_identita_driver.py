#!/usr/bin/env python3
"""VELOCE 14/10 — driver del TEST DI IDENTITA' (s23_identita.py lo lancia due volte: codice congelato della corsa e codice
veloce). Su CPU, con statistiche REALI di uno strato (cattura di S1: H per sorgente e Mx, meta' A e B; fattori di Kronecker
di S1; W del maestro bf16; T0/S0 della partenza GGUF), esegue risolvi_denso sui casi dati e salva scelte, trit, scale,
registro (senza i tempi). Nessuna GPU, nessuna scrittura fuori da --out.
Uso: python s23_identita_driver.py --codice DIR --out FILE.pt --dt float64|float32 --casi L:fam:modo[:monte],...
  modo = fisso (DRQ combo fissa, come dallo strato 2) | tarato (griglia DRQ completa, come gli strati 0-1 e 3, 7)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))   # 14/10: mai piu' di 2 fili (la corsa usa la CPU)

OC = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C")
S1 = OC / "S1_FLUSSO_C_1013"
PARTENZA = OC / "PTQ2" / "Thor-T-C-p_bil1jarqdrq.gguf"
PUNTO = {"attn_qkv": "attn", "attn_gate": "attn", "ssm_alpha": "attn", "ssm_beta": "attn", "attn_q": "attn",
         "attn_k": "attn", "attn_v": "attn", "ssm_out": "att_out", "attn_output": "att_out", "ffn_gate_shexp": "ffn",
         "ffn_up_shexp": "ffn", "ffn_down_shexp": "down"}
OPZ = {"alfe": [0.0, 0.5, 1.0], "w": 1.0, "damp_qep": 0.01, "drq": True,
       "k1": ("ssm_out", "attn_output", "attn_v", "ffn_down_shexp"), "giri_k1": 2, "k2": True, "giri_k2": 2,
       "limite_c": 8.0, "drq_modo": "paper+norm", "drq_strati_tar": 2, "energia_min": 0.9, "energia_max": 1.1,
       "guardia": "sim", "peso_kw": 3.5, "energia_relativa": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", default="float64")
    ap.add_argument("--casi", required=True)
    ap.add_argument("--k1-banda", type=int, default=1, help="(solo codice veloce) K1 a banda + ricalcolo esatto entro delta")
    ap.add_argument("--k1-delta", type=float, default=0.002, help="-1e9 = mai ricalcolo (taratura dello scarto)")
    a = ap.parse_args()
    cod = Path(a.codice)
    for p in (cod / "s23", cod / "lib", cod / "s1b", cod / "rif"):
        sys.path.insert(0, str(p))
    import s23_algebra as AL
    import s23_risolvi as RS
    import s1b_comune as K
    import s1_comune as SC
    from sorgente import SorgenteHF, bf16_a_f32
    torch.manual_seed(0)
    dt = getattr(torch, a.dt)
    dev = torch.device("cpu")
    gguf = K.gp._import_gguf(None)
    tens = {t.name: t for t in gguf.GGUFReader(str(PARTENZA), "r").tensors}
    sh = SorgenteHF()
    P_gqa = AL.legame_gqa(16, 2, 256, dev, dt)
    gradc, gradk = S1 / "gradienti" / "cal", S1 / "gradienti" / "ctl"

    def acc(L, punto, h):
        d = S1 / "cattura" / h / f"strato_{L:02d}"
        info = json.loads((d / "info.json").read_text())
        n = info["dims"][punto]
        Mx = torch.from_numpy(np.load(d / f"{punto}_Mx.npy").astype(np.float64))
        out = {}
        for s, nt in info["token"].items():
            if not nt:
                continue
            H = torch.from_numpy(SC.spacchetta(np.load(d / f"{punto}_Hq_{s}.npy"), n))
            out[s] = (H * nt, Mx * nt, int(nt))
        return out

    ris = {}
    scelti = {}
    for caso in a.casi.split(","):
        L, fam, modo, *resto = caso.split(":")
        L = int(L)
        nome = f"blk.{L}.{fam}.weight"
        t = tens[nome]
        tid = int(t.tensor_type)
        C, R = int(t.shape[0]), int(t.shape[1])
        Tn, Sn = K.gp.decode_rows(np.ascontiguousarray(np.asarray(t.data).reshape(R, -1)), tid, C)
        T0, S0 = torch.from_numpy(Tn.astype(np.int8)), torch.from_numpy(Sn.astype(np.float64))
        u16, _ = sh.u16_ordine_gguf(nome)
        W = torch.from_numpy(bf16_a_f32(u16)).to(dev, dt)
        accA, accB = acc(L, PUNTO[fam], "A"), acc(L, PUNTO[fam], "B")
        kron = RS.Kron(gradc, gradk, nome, "B", dev, dt) if fam in OPZ["k1"] else None
        monte = scelti.get(resto[0]) if resto else None
        if modo == "tarato":
            tar = RS.taratura_famiglia([], fam, 2, OPZ["drq_modo"])
        else:
            tar = {"tarato": False, "combo": "norm:a0.1:b1", "modo": OPZ["drq_modo"], "strati_tarati": [0, 1]}
        opz = {**OPZ, "drq_taratura": tar, "k1_banda": bool(a.k1_banda), "k1_banda_delta": a.k1_delta}
        r = RS.risolvi_denso(nome, fam, tid, W, accA, accB, T0, S0, opz, kron, P_gqa, monte)
        scelti[fam] = (r["T"], r["S"])
        reg = r["registro"]

        def senza_tempi(x):
            if isinstance(x, dict):
                return {k: senza_tempi(v) for k, v in x.items() if k not in ("secondi", "s")}
            if isinstance(x, list):
                return [senza_tempi(v) for v in x]
            return x
        ris[caso] = {"T": r["T"].clone(), "S": r["S"].clone(),
                     "S_monte": None if r["S_monte"] is None else r["S_monte"].clone(), "scelto": r["scelto"],
                     "registro": json.dumps(senza_tempi(reg), sort_keys=True, default=str),
                     "secondi": reg["secondi"], "kron": bool(kron is not None and kron.ok),
                     "JB": {k: v["JB"] for k, v in reg["candidati"].items()}, "k1_banda": reg.get("k1_banda")}
        print(f"{caso}: scelto {r['scelto']} kron {ris[caso]['kron']} {reg['secondi']} s "
              + " ".join(f"{k}:{v['dJB_rel']:+.5f}" for k, v in reg["candidati"].items()), flush=True)
    torch.save(ris, a.out)


if __name__ == "__main__":
    main()
