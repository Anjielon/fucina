#!/usr/bin/env python3
"""scale_gateup_esporta.py — 25/10 SINTESI_1025 L3: writes the routed-expert ffn_gate_exps / ffn_up_exps tensors with the
fp16 group scales trained by cardan_congiunto_scale.py --scale-gateup-lr, trits UNCHANGED (only the d fields of every
block are rewritten), as <dir>/blk.L.ffn_{gate,up}_exps.weight.npy (uint8, host shapes) for raffina/patch_gguf.py.
Modelled on scale_down_esporta.py; the byte patch is e2e_campi_scala.scrivi_scale + verifica_patch (codes identical at the
bit, d fields == the new fp16 scales at the bit, finite).

    scale = fp16(d0 * exp(theta))        d0 = the fp16 scales of the OSPITE tensor, theta fp32, product in fp32
This is the arithmetic of StratoEsperti._scala (e2e_studente.py): d0 fp32 (exact fp16 values) * torch.exp(theta fp32),
rounded to fp16 RNE (scale_down_esporta.py / cardan_piega_down.py compute it in fp64 instead: a fp32 product can round to
a different fp16 in rare ties). The pack written by the trainer also carries the sha256 of the fp16 scales the student
used on the TRAINING device; each tensor is checked against it (a mismatch means CPU and GPU exp differ by an ulp at a
tie: rerun with --device cuda, the training device, or accept explicitly with --senza-sha).

uso: scale_gateup_esporta.py OSPITE.gguf scale_gateup_<passo>.pt DIR_TENSORI [--device cpu|cuda]
         [--controlla-base BASE.gguf] [--strati 0-39] [--senza-sha]
  OSPITE   the GGUF the tensors are written into (e.g. the one with dense + down folded): its gate/up must be the gate/up
           the run trained on; --controlla-base BASE (the trainer's --gguf) checks it byte for byte.
  The second planes (ffn_{gate,up}_exps2, hot experts) are frozen in the student and are not touched here.
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

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI / "thor_v2"))
import e2e_campi_scala as CAMPI  # noqa: E402  (inserts the k1024 gguf-py path: TQ1_0R_1024 is a known type there)
from gguf import GGUFReader      # noqa: E402

FAM_GU = ("gate", "up")


def log(*m) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)


def intervallo(s: str) -> list:
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p.strip():
            out.append(int(p))
    return out


def scale_fp16(d0_fp16: np.ndarray, theta: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """fp16(d0 * exp(theta)) in fp32 on `device` = StratoEsperti._scala; d0 (E, R, NG) float16, theta same shape."""
    d0 = torch.from_numpy(np.ascontiguousarray(d0_fp16)).to(device=device, dtype=torch.float32)
    th = theta.to(device=device, dtype=torch.float32)
    return (d0 * torch.exp(th)).to(torch.float16).cpu().contiguous().numpy()


def esporta_tensore(raw: np.ndarray, tipo: str, theta: torch.Tensor, device: str = "cpu") -> tuple:
    """raw (E, R, byte_per_row) uint8 of ONE ternary expert tensor -> (new bytes, fp16 scales, verification dict).
    Only the d fields change (e2e_campi_scala.scrivi_scale); verifica_patch re-reads the written bytes."""
    E, R = int(raw.shape[0]), int(raw.shape[1])
    d0 = CAMPI.leggi_scale(raw, tipo, E, R)                      # (E, R, NG) float16, bit exact
    if tuple(theta.shape) != tuple(d0.shape):
        raise ValueError(f"theta {tuple(theta.shape)} != scales {tuple(d0.shape)}")
    d = scale_fp16(d0, theta, device)
    nuovo = CAMPI.scrivi_scale(raw, tipo, d)
    ver = CAMPI.verifica_patch(raw, nuovo, tipo, d)
    return nuovo, d, ver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ospite"); ap.add_argument("pacco"); ap.add_argument("dir")
    ap.add_argument("--device", default="cpu", help="device of the fp32 product (cuda = the training device: at the bit)")
    ap.add_argument("--controlla-base", default=None, help="the trainer's --gguf: OSPITE gate/up must equal it byte for byte")
    ap.add_argument("--strati", default=None, help="layers to write (default: all layers of the pack)")
    ap.add_argument("--senza-sha", action="store_true", help="do not stop on a sha256 mismatch with the trainer's scales")
    a = ap.parse_args(argv)
    pk = torch.load(a.pacco, map_location="cpu", weights_only=False)
    if not isinstance(pk, dict) or "theta" not in pk:
        log(f"ERRORE: {a.pacco} non e' un pacco scale_gateup (manca 'theta')"); return 2
    th_all, sha_all = pk["theta"], pk.get("sha256_fp16", {})
    strati = sorted(int(L) for L in th_all) if a.strati is None else intervallo(a.strati)
    rd = GGUFReader(a.ospite, "r")
    tens = {t.name: t for t in rd.tensors}
    tb = None
    if a.controlla_base:
        tb = {t.name: t for t in GGUFReader(a.controlla_base, "r").tensors}
    td = Path(a.dir); td.mkdir(parents=True, exist_ok=True)
    man = {"ospite": str(a.ospite), "pacco": str(a.pacco), "pacco_sha256": hashlib.sha256(Path(a.pacco).read_bytes()).hexdigest(),
           "meta": pk.get("meta"), "dispositivo_allenamento": pk.get("dispositivo"), "dispositivo_qui": a.device,
           "formula": pk.get("formula"), "tensori": {}}
    n_sha_diversi = 0
    for L in strati:
        th_L = th_all.get(L, th_all.get(str(L)))
        if th_L is None:
            log(f"ERRORE: strato {L} assente nel pacco"); return 2
        for f in FAM_GU:
            nome = f"blk.{L}.ffn_{f}_exps.weight"
            if nome not in tens:
                log(f"ERRORE: {nome} non e' in {a.ospite}"); return 2
            t = tens[nome]
            tipo = t.tensor_type.name
            raw = np.asarray(t.data)                                  # memmap (E, R, byte_per_row)
            if raw.ndim != 3 or raw.dtype != np.uint8:
                log(f"ERRORE: {nome}: atteso uint8 (E, R, byte), trovato {raw.dtype} {raw.shape}"); return 2
            if tb is not None:
                if nome not in tb or not np.array_equal(np.asarray(tb[nome].data), raw):
                    log(f"ERRORE: {nome} di {a.ospite} diverso da quello della base di allenamento {a.controlla_base}: "
                        f"theta e' relativo alle scale della base"); return 3
            nuovo, d, ver = esporta_tensore(raw, tipo, th_L[f], a.device)
            sha = hashlib.sha256(d.tobytes()).hexdigest()
            atteso = sha_all.get(L, sha_all.get(str(L), {})).get(f)
            if atteso is not None and atteso != sha:
                n_sha_diversi += 1
                log(f"{'ATTENZIONE' if a.senza_sha else 'ERRORE'}: {nome}: scale fp16 diverse da quelle dello studente "
                    f"(sha {sha[:12]} contro {atteso[:12]}, dispositivo allenamento {pk.get('dispositivo')}, qui {a.device})")
                if not a.senza_sha:
                    return 3
            p = td / f"{nome}.npy"
            with open(str(p) + ".tmp", "wb") as fh:
                np.save(fh, nuovo)
            os.replace(str(p) + ".tmp", p)
            back = np.load(p, mmap_mode="r")                          # round trip of the written file
            if not (back.dtype == np.uint8 and back.shape == raw.shape and back.nbytes == int(t.n_bytes)
                    and np.array_equal(back, nuovo)):
                log(f"ERRORE: {p}: rilettura diversa da quanto scritto"); return 4
            thf = th_L[f].double()
            man["tensori"][nome] = {**ver, "tipo": tipo, "forma": list(raw.shape), "sha256_scale_fp16": sha,
                                    "sha_come_studente": None if atteso is None else atteso == sha,
                                    "theta_rms": float(thf.pow(2).mean().sqrt()), "theta_max": float(thf.abs().max())}
            log(f"{nome} {tipo}: theta rms {man['tensori'][nome]['theta_rms']:.3e} max {man['tensori'][nome]['theta_max']:.3e}, "
                f"scale cambiate {ver['scale_cambiate']}/{d.size}, codici identici {ver['codici_identici']}, "
                f"sha = studente {man['tensori'][nome]['sha_come_studente']}")
        for f in FAM_GU:
            if f"blk.{L}.ffn_{f}_exps2.weight" in tens:
                log(f"nota: blk.{L}.ffn_{f}_exps2 (secondo piano) lasciato com'e' (congelato nello studente)")
    man["sha_diversi"] = n_sha_diversi
    (td / "scale_gateup_esporta.json").write_text(json.dumps(man, indent=1))
    log(f"scritti {len(man['tensori'])} tensori in {td} (trit invariati); poi: patch_gguf.py --dir {td} --base {a.ospite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
