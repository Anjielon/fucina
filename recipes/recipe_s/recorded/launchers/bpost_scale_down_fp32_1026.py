#!/usr/bin/env python3
"""bpost_scale_down_fp32_1026.py — 26/10 B+POST: the routed-expert ffn_down_exps with the down scales trained by
cardan_congiunto_scale.py --scale-down-lr, written in the STUDENT'S arithmetic, trits UNCHANGED (only the d fields).

    scale = fp16(d0_fp32 * torch.exp(theta_fp32))      on the training device (cuda)

= StratoEsperti._scala (e2e_studente.py: f.d0 fp32 of the exact fp16 values, theta fp32, ArrotondaFP16 = RNE).
scale_down_esporta.py and cardan_piega_down.py compute the product in fp64 instead: in rare ties the fp16 differs from the
one the student used. scale_<P>.pt ({L: theta (E, R, NG) fp32}) carries no sha256 of the student's scales (unlike
scale_gateup_<P>.pt), so exactness rests on using the SAME function as the gate/up exporter
(scale_gateup_esporta.esporta_tensore, device cuda), whose output matched the student's sha256 on 80/80 gate/up tensors
at B+ T350 (128,976,163 scales changed). Diagnostics per tensor (not gates): fp16 scales that differ from the fp64 path
and from the CPU fp32 path. Checks (gates): codes identical at the bit, d fields == the new scales at the bit, finite,
written file re-read equal (e2e_campi_scala.verifica_patch + round trip).

uso: bpost_scale_down_fp32_1026.py OSPITE.gguf scale_<P>.pt DIR [--device cuda|cpu] [--strati 0-39] [--senza-confronto]
  OSPITE = the trainer's --gguf (theta is relative to ITS down scales). Output DIR/blk.L.ffn_down_exps.weight.npy (uint8,
  host shape) for raffina/patch_gguf.py, and DIR/scale_down_fp32.json. Second planes (ffn_down_exps2) are frozen in the
  student and are not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

COD = Path("/mnt/models/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/BPP/codice")
sys.path.insert(0, str(COD))
import scale_gateup_esporta as SG  # noqa: E402  (thor_v2/e2e_campi_scala + k1024 gguf-py)
from gguf import GGUFReader        # noqa: E402

CAMPI = SG.CAMPI


def log(*m) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)


def scale_fp64(d0_fp16: np.ndarray, theta: torch.Tensor) -> np.ndarray:
    """The fp64 path of scale_down_esporta.py (diagnostic only)."""
    d0 = torch.from_numpy(np.ascontiguousarray(d0_fp16)).double()
    return (d0 * torch.exp(theta.double())).to(torch.float16).contiguous().numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ospite"); ap.add_argument("pacco"); ap.add_argument("dir")
    ap.add_argument("--device", default="cuda", help="device of the fp32 product (cuda = the training device)")
    ap.add_argument("--strati", default=None)
    ap.add_argument("--senza-confronto", action="store_true", help="skip the fp64 / CPU diagnostics")
    a = ap.parse_args(argv)
    th_all = torch.load(a.pacco, map_location="cpu")
    if not isinstance(th_all, dict) or not th_all:
        log(f"ERRORE: {a.pacco} non e' un pacco di theta down ({{strato: theta}})"); return 2
    strati = sorted(int(L) for L in th_all) if a.strati is None else SG.intervallo(a.strati)
    tens = {t.name: t for t in GGUFReader(a.ospite, "r").tensors}
    td = Path(a.dir); td.mkdir(parents=True, exist_ok=True)
    man = {"ospite": str(a.ospite), "pacco": str(a.pacco),
           "pacco_sha256": hashlib.sha256(Path(a.pacco).read_bytes()).hexdigest(), "dispositivo": a.device,
           "formula": "scale = fp16(d0_fp32 * exp(theta_fp32)) = StratoEsperti._scala (scale_gateup_esporta.scale_fp16)",
           "tensori": {}}
    tot = {"scale": 0, "cambiate": 0, "diverse_fp64": 0, "diverse_cpu": 0}
    for L in strati:
        th = th_all.get(L, th_all.get(str(L)))
        if th is None:
            log(f"ERRORE: strato {L} assente nel pacco"); return 2
        nome = f"blk.{L}.ffn_down_exps.weight"
        if nome not in tens:
            log(f"ERRORE: {nome} non e' in {a.ospite}"); return 2
        t = tens[nome]
        tipo = t.tensor_type.name
        raw = np.asarray(t.data)
        if raw.ndim != 3 or raw.dtype != np.uint8:
            log(f"ERRORE: {nome}: atteso uint8 (E, R, byte), trovato {raw.dtype} {raw.shape}"); return 2
        E, R = int(raw.shape[0]), int(raw.shape[1])
        d0 = CAMPI.leggi_scale(raw, tipo, E, R)
        th = th.detach().to(torch.float32)
        if th.numel() != d0.size:
            log(f"ERRORE: {nome}: theta {tuple(th.shape)} contro scale {d0.shape}"); return 2
        th = th.reshape(d0.shape)
        if not torch.isfinite(th).all():
            log(f"ERRORE: {nome}: theta non finite"); return 2
        nuovo, d, ver = SG.esporta_tensore(raw, tipo, th, a.device)
        rec = {**ver, "tipo": tipo, "forma": list(raw.shape), "sha256_scale_fp16": hashlib.sha256(d.tobytes()).hexdigest(),
               "theta_rms": float(th.double().pow(2).mean().sqrt()), "theta_max": float(th.abs().max())}
        if not a.senza_confronto:
            u = d.view(np.uint16)
            rec["diverse_da_fp64"] = int((u != scale_fp64(d0, th).view(np.uint16)).sum())
            rec["diverse_da_cpu_fp32"] = int((u != SG.scale_fp16(d0, th, "cpu").view(np.uint16)).sum()) \
                if a.device != "cpu" else 0
            tot["diverse_fp64"] += rec["diverse_da_fp64"]; tot["diverse_cpu"] += rec["diverse_da_cpu_fp32"]
        p = td / f"{nome}.npy"
        with open(str(p) + ".tmp", "wb") as fh:
            np.save(fh, nuovo)
        os.replace(str(p) + ".tmp", p)
        back = np.load(p, mmap_mode="r")
        if not (back.dtype == np.uint8 and back.shape == raw.shape and back.nbytes == int(t.n_bytes)
                and np.array_equal(back, nuovo)):
            log(f"ERRORE: {p}: rilettura diversa da quanto scritto"); return 4
        man["tensori"][nome] = rec
        tot["scale"] += int(d.size); tot["cambiate"] += int(ver["scale_cambiate"])
        log(f"{nome} {tipo}: theta rms {rec['theta_rms']:.3e} max {rec['theta_max']:.3e}, scale cambiate "
            f"{ver['scale_cambiate']}/{d.size}, codici identici {ver['codici_identici']}"
            + ("" if a.senza_confronto else f", diverse da fp64 {rec['diverse_da_fp64']}, da cpu fp32 "
               f"{rec['diverse_da_cpu_fp32']}"))
        if f"blk.{L}.ffn_down_exps2.weight" in tens:
            log(f"nota: blk.{L}.ffn_down_exps2 (secondo piano) lasciato com'e' (congelato nello studente)")
    man["totali"] = tot
    (td / "scale_down_fp32.json").write_text(json.dumps(man, indent=1))
    log(f"scritti {len(man['tensori'])} tensori in {td} (trit invariati): scale cambiate {tot['cambiate']}/{tot['scale']}"
        + ("" if a.senza_confronto else f", diverse dal percorso fp64 {tot['diverse_fp64']}, dal cpu fp32 "
           f"{tot['diverse_cpu']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
