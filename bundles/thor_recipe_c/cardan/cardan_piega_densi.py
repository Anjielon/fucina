#!/usr/bin/env python3
"""cardan_piega_densi.py — FOLD of the distilled DENSE correction ("CARDAN of the dense", 10/10) into the EXISTING
ternary planes and fp16 group scales of the dense tensors (attention q/k/v/o, GDN qkv/gate/ssm_out, shared expert
gate/up/down), so the GGUF keeps the host's tensor names, types, shapes and bytes: same engine, same speed.

The correction comes from rilq_thor.py --continuo (residui_delta.pt): dW = diag(gamma) B A of rank r, trained with the
exact full-vocabulary KL against the teacher IN THE TERNARY FLOW of the host, stored in the FILE order of the GGUF
tensor (non-rotated). For each dense tensor of the host (any number of planes K = 2/3/4, read from the host itself):
    target  W*' = sum_k s_k T_k (host, ALL planes)  +  R dW   (R = stored-basis rotation of TQ1_0R; identity for TQ1_0)
    fold    JARQ with ALL planes free (cardan_piega.jarq_lotto, the batched twin of raffina/solutori.jarq, B = 1):
            joint least-squares of every fp16 scale of the row + Babai nearest-plane proposals on the composite
            alphabet, accepted row by row only if they lower the error; rows that end worse go back to the host
    metric  H' = input Hessian of that tensor from the R2B3 manifest (the same H the raffina levers use), rotated
Output: <out>/tensori/<plane name>.npy (uint8, host shapes) for raffina/patch_gguf.py, <out>/registro.jsonl with the
captured fraction 1 - |W* - fold|_H^2 / |R dW|_H^2 per tensor. Algebra: factorizations on CPU fp64 (cardan_piega,
never rocSOLVER Cholesky on gfx1151).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
RAFF = "@@ROOT:models@@/build/leve/raffina"
MANIF_R2B3 = "@@ROOT:depot@@/tutto_ternario/tensori_R2B3/manifesto.json"      # read only


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def hessiana(nome: str, manif: dict) -> torch.Tensor:
    """H [C, C] fp64 of the tensor input, normalised like comune.bersaglio_e_H (symmetric, dead diagonal -> 1,
    mean(diag) = 1). Raises if the manifest has none (no metric, no fold)."""
    info = manif.get(nome.replace(".weight", "3.weight"), {}).get("hessiana")
    if not info:
        raise KeyError(f"{nome}: no Hessian in the R2B3 manifest")
    H = torch.from_numpy(np.load(info["file"]).astype(np.float64))
    H = 0.5 * (H + H.T)
    d = torch.diagonal(H)
    morti = d <= 0
    if morti.any():
        H[morti, morti] = 1.0
    return H / d.clamp_min(1e-30).mean()


def nome_gguf(chiave: str) -> str:
    """'blk.L.attn_qkv' (rilq_thor residue name) -> 'blk.L.attn_qkv.weight' (GGUF tensor of plane 1)."""
    return chiave if chiave.endswith(".weight") else chiave + ".weight"


def piega_uno(d, dW: torch.Tensor, H: torch.Tensor, CM, CP, dev, dt) -> tuple:
    """d = comune.Denso of the host tensor, dW [R, C] fp64 file order. Returns (T2 int8 [K,R,C], S2 fp32 [K,R,ng],
    record)."""
    assert tuple(dW.shape) == (d.out, d.in_), (d.nome, tuple(dW.shape), (d.out, d.in_))
    dWs = CM.ruota(dW) if d.rotated else dW
    Hs = CM.ruota_H(H) if d.rotated else H
    Q0 = d.ricostruisci()                                                     # f64 [R, C], stored basis
    W = (Q0 + dWs).to(dev, dt).unsqueeze(0)
    Hd = Hs.to(dev, dt).unsqueeze(0)
    T = d.T.unsqueeze(0).to(dev)
    S = d.S.to(dev, dt).unsqueeze(0)
    Ha = Hd + 1e-4 * torch.eye(Hd.shape[-1], dtype=dt, device=dev)
    e_dw = float(CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, S), Ha).sum())
    T2, S2, cat = CP.jarq_lotto(W, Hd, T, S, giri=3, nu2=1e-4)
    e_f = float(CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum())
    T2, S2 = T2[0].cpu(), S2[0].cpu().float()
    assert torch.equal(S2, S2.to(torch.float16).float()), f"{d.nome}: scales not fp16"
    rec = {"nome": d.nome, "piani": d.K, "energia_correzione_H": e_dw, "residuo_dopo_piega_H": e_f,
           "catturata": 1 - e_f / max(e_dw, 1e-300), "righe_tornate_ospite": cat.get("righe_tornate_ospite"),
           "norma_rel_dW": float(dW.norm() / Q0.norm().clamp_min(1e-30))}
    return T2, S2, rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True, help="residui_delta.pt (rilq_thor.py --continuo)")
    ap.add_argument("--gguf", required=True, help="host GGUF (the one the correction was trained on)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    os.environ["LEVE_OSPITE"] = a.gguf                                     # comune reads the host's planes
    sys.path.insert(0, RAFF)
    sys.path.insert(0, str(QUI))
    import comune as CM                                                    # noqa: E402
    import cardan_piega as CP                                              # noqa: E402  (algebra off rocSOLVER)
    dev = torch.device(a.device)
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    dd = torch.load(a.delta, map_location="cpu", weights_only=False)
    delta = {nome_gguf(k): v.double() for k, v in dd["delta_file"].items()}
    manif = json.load(open(MANIF_R2B3))
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    reg_p = out / "registro.jsonl"
    fatti = {json.loads(r)["nome"] for r in reg_p.read_text().splitlines() if r.strip()} if reg_p.exists() else set()
    strati = sorted({int(n.split(".")[1]) for n in delta})
    log(f"host {a.gguf}, {len(delta)} dense corrections (rank {dd.get('rango')}, step {dd.get('passo')}), "
        f"layers {strati[0]}-{strati[-1]}, already done {len(fatti)}, device {dev}")
    t_tot = time.time()
    with open(reg_p, "a") as reg:
        for L in strati:
            nomi = [n for n in sorted(delta) if int(n.split(".")[1]) == L and n not in fatti]
            if not nomi:
                continue
            src = CM.gp.GGUFPlaneSource(a.gguf, layers=[L])
            for nome in nomi:
                t0 = time.time()
                d = CM.Denso(src, nome)
                T2, S2, rec = piega_uno(d, delta[nome], hessiana(nome, manif), CM, CP, dev, dt)
                byte = d.byte_piani(T2, S2)
                for k, (nm, bb) in enumerate(zip(d.nomi_piani, byte)):
                    tp, sp = CM.gp.decode_rows(bb, d.tid, d.in_)                   # bit-exact round trip
                    assert np.array_equal(tp, T2[k].numpy().astype(np.int8)) and np.array_equal(
                        sp.view(np.uint16), S2[k].to(torch.float16).numpy().view(np.uint16)), (nome, k)
                    np.save(td / f"{nm}.npy", bb)
                rec["secondi"] = round(time.time() - t0, 1)
                reg.write(json.dumps(rec) + "\n"); reg.flush()
                log(f"{nome} K={rec['piani']}: |dW|/|W| {rec['norma_rel_dW']:.4f}, captured "
                    f"{100 * rec['catturata']:.1f} % ({rec['secondi']} s)")
    log(f"FINE piega densi in {time.time() - t_tot:.0f} s -> {td}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
