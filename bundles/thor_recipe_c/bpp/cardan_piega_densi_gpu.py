#!/usr/bin/env python3
"""cardan_piega_densi_gpu.py (20/10, Angelo: "versione tutta GPU, scalabile"): copia di cardan_piega_densi.py con
(1) --dtype float64 sulla GPU (default: float64 anche su GPU, cosi' la precisione resta quella del percorso CPU fp64);
(2) --parte i/N: N processi si dividono i tensori (ognuno i suoi, registro proprio registro.<i>.jsonl), su una o piu' GPU
(--device cuda:k); (3) scrittura atomica dei .npy (file temporaneo + rinomina). Fattorizzazioni: CPU fp64 (cardan_piega).

Originale: cardan_piega_densi.py — FOLD of the distilled DENSE correction ("CARDAN of the dense", 10/10) into the EXISTING
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
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
import torch
import cardan_ripresa_densi_gpu as RD

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


def piega_gruppo(ds, dWs_, Hs_, CM, CP, dev, dt) -> list:
    """21/10 (Angelo: densi «stravelocissimi» sulla GPU): piega INSIEME B tensori della stessa forma (R, C, K, rotazione)
    con UNA chiamata jarq_lotto a lotto B (le funzioni di cardan_piega sono gia' a lotti: ogni elemento del lotto e'
    indipendente). Stessa matematica di piega_uno, tensore per tensore. Ritorna [(T2, S2, rec)]."""
    Ws, Hs, Ts, Ss, Q0n = [], [], [], [], []
    for d, dW, H in zip(ds, dWs_, Hs_):
        assert tuple(dW.shape) == (d.out, d.in_), (d.nome, tuple(dW.shape), (d.out, d.in_))
        dWs = CM.ruota(dW) if d.rotated else dW
        Hr = CM.ruota_H(H) if d.rotated else H
        Q0 = d.ricostruisci()
        Ws.append(Q0 + dWs); Hs.append(Hr); Ts.append(d.T); Ss.append(d.S)
        Q0n.append(float(dW.norm() / Q0.norm().clamp_min(1e-30)))
    W = torch.stack(Ws).to(dev, dt)
    Hd = torch.stack(Hs).to(dev, dt)
    T = torch.stack(Ts).to(dev)
    S = torch.stack(Ss).to(dev, dt)
    Ha = Hd + 1e-4 * torch.eye(Hd.shape[-1], dtype=dt, device=dev)
    e_dw = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, S), Ha).sum(-1)
    T2, S2, cat = CP.jarq_lotto(W, Hd, T, S, giri=3, nu2=1e-4)
    e_f = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum(-1)
    out = []
    for b, d in enumerate(ds):
        t2, s2 = T2[b].cpu(), S2[b].cpu().float()
        assert torch.equal(s2, s2.to(torch.float16).float()), f"{d.nome}: scales not fp16"
        ed, ef = float(e_dw[b]), float(e_f[b])
        out.append((t2, s2, {"nome": d.nome, "piani": d.K, "energia_correzione_H": ed, "residuo_dopo_piega_H": ef,
                             "catturata": 1 - ef / max(ed, 1e-300), "righe_tornate_ospite_lotto": cat.get("righe_tornate_ospite"),
                             "norma_rel_dW": Q0n[b], "lotto": len(ds)}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True, help="residui_delta.pt (rilq_thor.py --continuo)")
    ap.add_argument("--gguf", required=True, help="host GGUF (the one the correction was trained on)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--lotto", type=int, default=0, help="0 = un tensore alla volta (come l'originale); B>0 = piega a lotti di B tensori della stessa forma")
    ap.add_argument("--parte", default="0/1", help="i/N: questo processo piega solo i tensori di posto i modulo N")
    a = ap.parse_args(argv)
    # D15 (22/10): FORGIA_DENSI_DTYPE overrides --dtype for a whole chain without editing the orchestrator scripts.
    # Unset = previous behaviour. float64 on the GPU stays within CK_TOLL_BYTE=1e-5 of the CPU fp64 fold (measured).
    if os.environ.get("FORGIA_DENSI_DTYPE"):
        if os.environ["FORGIA_DENSI_DTYPE"] not in ("float64", "float32"):
            raise SystemExit("FORGIA_DENSI_DTYPE must be float64 or float32")
        a.dtype = os.environ["FORGIA_DENSI_DTYPE"]
    pi, pn = RD.parte(a.parte)
    with RD.Partizione(a.out, pi, pn):
        return esegui(a, pi, pn)


def esegui(a, pi, pn):
    os.environ["LEVE_OSPITE"] = a.gguf                                     # comune reads the host's planes
    sys.path.insert(0, RAFF)
    sys.path.insert(0, str(QUI))
    import comune as CM                                                    # noqa: E402
    import cardan_piega_gpu as CP                                          # noqa: E402  (PIEGA_ALGEBRA=gpu_sicura: GPU senza rocSOLVER)
    dev = torch.device(a.device)
    dt = torch.float64 if (dev.type == "cpu" or a.dtype == "float64") else torch.float32
    dd = torch.load(a.delta, map_location="cpu", weights_only=False)
    delta = {nome_gguf(k): v.double() for k, v in dd["delta_file"].items()}
    manif = json.load(open(MANIF_R2B3))
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    reg_p = out / (f"registro.{pi}.jsonl" if pn > 1 else "registro.jsonl")
    tutti = sorted(delta)
    mio = {n for j, n in enumerate(tutti) if j % pn == pi}
    strati = sorted({int(n.split(".")[1]) for n in delta})
    src = CM.gp.GGUFPlaneSource(a.gguf, layers=strati)
    prov = RD.Provenienza(src, delta, manif, RD.ricetta(dev, dt, (piega_uno, piega_gruppo, hessiana, nome_gguf),
                                                         (CP.__file__,)))   # D36
    fatti = RD.riprendi(out, mio, src, CM, log, prov)
    log(f"host {a.gguf}, {len(delta)} dense corrections (rank {dd.get('rango')}, step {dd.get('passo')}), "
        f"layers {strati[0]}-{strati[-1]}, already done {len(fatti)}, device {dev}")
    t_tot = time.time()
    if a.lotto > 0:
        return main_lotti(a, delta, manif, td, reg_p, fatti, mio, CM, CP, dev, dt, t_tot, src, prov)
    with RD.Registro(reg_p) as reg:
        for L in strati:
            nomi = [n for n in sorted(delta) if int(n.split(".")[1]) == L and n not in fatti and n in mio]
            if not nomi:
                continue
            for nome in nomi:
                t0 = time.time()
                d = CM.Denso(src, nome)
                T2, S2, rec = piega_uno(d, delta[nome], hessiana(nome, manif), CM, CP, dev, dt)
                byte = d.byte_piani(T2, S2)
                artefatti = {}
                for k, (nm, bb) in enumerate(zip(d.nomi_piani, byte)):
                    tp, sp = CM.gp.decode_rows(bb, d.tid, d.in_)                   # bit-exact round trip
                    assert np.array_equal(tp, T2[k].numpy().astype(np.int8)) and np.array_equal(
                        sp.view(np.uint16), S2[k].to(torch.float16).numpy().view(np.uint16)), (nome, k)
                    nome_file, scheda = RD.salva_piano(td, nm, bb)
                    artefatti[nome_file] = scheda
                rec["secondi"] = round(time.time() - t0, 1)
                RD.commit(reg, rec, artefatti, prov.di(nome))
                log(f"{nome} K={rec['piani']}: |dW|/|W| {rec['norma_rel_dW']:.4f}, captured "
                    f"{100 * rec['catturata']:.1f} % ({rec['secondi']} s)")
    log(f"FINE piega densi in {time.time() - t_tot:.0f} s -> {td}")
    return 0


def main_lotti(a, delta, manif, td, reg_p, fatti, mio, CM, CP, dev, dt, t_tot, src=None, prov=None) -> int:
    """Tutti i tensori da fare, raggruppati per (R, C, K, rotazione), piegati a lotti di a.lotto."""
    gruppi = {}
    if src is None:
        src = CM.gp.GGUFPlaneSource(a.gguf, layers=sorted({int(n.split(".")[1]) for n in delta}))
    for L in sorted({int(n.split(".")[1]) for n in delta}):
        nomi = [n for n in sorted(delta) if int(n.split(".")[1]) == L and n not in fatti and n in mio]
        if not nomi:
            continue
        for nome in nomi:
            d = CM.Denso(src, nome)
            gruppi.setdefault((d.out, d.in_, d.K, d.rotated), []).append(d)
    log(f"lotti: {sum(len(v) for v in gruppi.values())} tensori in {len(gruppi)} forme, lotto {a.lotto}")
    with RD.Registro(reg_p) as reg:
        for forma, ds in gruppi.items():
            for i in range(0, len(ds), a.lotto):
                parte = ds[i:i + a.lotto]
                t0 = time.time()
                ris = piega_gruppo(parte, [delta[d.nome] for d in parte], [hessiana(d.nome, manif) for d in parte],
                                   CM, CP, dev, dt)
                sec = round((time.time() - t0) / len(parte), 2)
                for d, (T2, S2, rec) in zip(parte, ris):
                    byte = d.byte_piani(T2, S2)
                    artefatti = {}
                    for k, (nm, bb) in enumerate(zip(d.nomi_piani, byte)):
                        tp, sp = CM.gp.decode_rows(bb, d.tid, d.in_)
                        assert np.array_equal(tp, T2[k].numpy().astype(np.int8)) and np.array_equal(
                            sp.view(np.uint16), S2[k].to(torch.float16).numpy().view(np.uint16)), (d.nome, k)
                        nome_file, scheda = RD.salva_piano(td, nm, bb)
                        artefatti[nome_file] = scheda
                    rec["secondi"] = sec
                    RD.commit(reg, rec, artefatti, prov.di(d.nome) if prov is not None else None)
                log(f"forma {forma}: {len(parte)} tensori in {time.time() - t0:.1f} s ({sec} s a tensore)")
    log(f"FINE piega densi a lotti in {time.time() - t_tot:.0f} s -> {td}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
