#!/usr/bin/env python3
"""Collaudo seriale su dati veri; nessuna chiamata GPU a rocSOLVER.

nice -n 19 env LEVE_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python test_piega_gpu.py --reali --out DIR

Default prudente: 4 righe complete, tutte le colonne e tutti i piani, esperto 0
di gate/up negli strati 3 e 32; tre famiglie dense (k, gate GDN, ssm_out).
--righe 0 verifica tutte le righe dei medesimi tensori (molto piu' oneroso).
I .npy nella directory di collaudo contengono SOLO le righe verificate:
non sono patch GGUF e non devono essere usati come tali.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import warnings

import numpy as np
import torch

import cardan_piega as CP
import cardan_piega_densi as PD
from cardan_algebra_gpu import AlgebraGPU, jarq_protetto, percorso

BASE = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C")
PT = BASE / "S5_SU_S7T550_1020/kd/allena/cardan_migliore.pt"
GGUF = BASE / "S7_CONGIUNTO_T550_1020/Thor-T-C-S7-T550-g10.gguf"
DELTA = BASE / "DENSI2_SU_S7T550_1020/thor/allena/residui_delta.pt"


@contextmanager
def vieta_rocsolver():
    funzioni = [(torch.linalg, n) for n in ("cholesky", "cholesky_ex", "solve")]
    funzioni.append((torch, "cholesky_solve"))
    originali = [(modulo, nome, getattr(modulo, nome)) for modulo, nome in funzioni]

    def guardia(f, nome):
        def chiamata(*args, **kwargs):
            if any(isinstance(t, torch.Tensor) and t.device.type == "cuda"
                   for t in (*args, *kwargs.values())):
                raise AssertionError(f"Vietata primitiva rocSOLVER sulla GPU: {nome}")
            return f(*args, **kwargs)
        return chiamata

    try:
        for modulo, nome, f in originali:
            setattr(modulo, nome, guardia(f, nome))
        yield
    finally:
        for modulo, nome, f in originali:
            setattr(modulo, nome, f)


def test_contesto_cpu():
    prima, flag = CP.chol_ex, CP.ALGEBRA_CPU
    a = AlgebraGPU()
    M = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    with percorso(CP, True, a):
        assert CP.chol_ex == a.chol_ex
        assert torch.equal(CP.chol(M), torch.linalg.cholesky(M))
        B = torch.ones(2, 1, dtype=torch.float64)
        assert torch.allclose(CP.risolvi(M, B), torch.linalg.solve(M, B))
    assert CP.chol_ex is prima and CP.ALGEBRA_CPU == flag
    try:
        with percorso(CP, True, a):
            raise ValueError("prova ripristino")
    except ValueError:
        pass
    assert CP.chol_ex is prima


def test_ripiego_jarq():
    originale = CP.jarq_lotto
    chiamate = []

    def simulato(W, H, T, S, **kwargs):
        chiamate.append(W.device.type)
        if len(chiamate) == 1:
            return T, S * float("nan"), {"ospite": 1.}
        return T, S, {"ospite": 1.}

    W = torch.ones(1, 2, 2, dtype=torch.float64)
    H = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    T = torch.ones(1, 1, 2, 2, dtype=torch.int8)
    S = torch.ones(1, 1, 2, 1, dtype=torch.float64)
    CP.jarq_lotto = simulato
    try:
        with warnings.catch_warnings(record=True) as avvisi:
            _, s, _ = jarq_protetto(CP, W, H, T, S)
        assert len(chiamate) == 2 and avvisi and torch.equal(s, S)
        try:
            jarq_protetto(CP, W * float("inf"), H, T, S)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Ingresso non finito accettato")
    finally:
        CP.jarq_lotto = originale


def test_wrapper_cpu():
    import cardan_piega_blocchi_gpu as PG
    import cardan_piega_densi_gpu as DG

    class Denso:
        nome, out, in_, K, rotated = "blk.0.prova.weight", 2, 8, 1, False
        T = (torch.arange(16).reshape(1, 2, 8) % 3 - 1).to(torch.int8)
        S = torch.full((1, 2, 2), .125, dtype=torch.float64)

        def ricostruisci(self):
            return CP.ricostruisci_lotto(self.T.unsqueeze(0), self.S.unsqueeze(0))[0]

    d = Denso()
    H = torch.eye(8, dtype=torch.float64)
    dw = d.ricostruisci() * .05
    rif = PD.piega_uno(d, dw, H, None, CP, torch.device("cpu"), torch.float64)
    got = DG.piega_uno(d, dw, H, None, CP, torch.device("cpu"), torch.float64)
    assert torch.equal(rif[0], got[0]) and torch.equal(rif[1], got[1]) and rif[2] == got[2]
    originale, chol_prima = CP.jarq_lotto, CP.chol_ex
    chiamate = []

    def strato(L, stato, gguf, dev, dt, lotto, *args, **kwargs):
        assert L == 3 and dt == torch.float64 and lotto == 1
        assert CP.jarq_lotto is not originale
        assert CP.chol_ex is not chol_prima
        W = (d.ricostruisci() + dw).unsqueeze(0)
        r = CP.jarq_lotto(W, H.unsqueeze(0), d.T.unsqueeze(0), d.S.unsqueeze(0))
        assert torch.equal(r[0][0], rif[0]) and torch.equal(r[1][0].float(), rif[1])
        chiamate.append(L)

    with tempfile.TemporaryDirectory(prefix="piega-gpu-cpu-") as out:
        with patch.dict(os.environ, {"PIEGA_GPU": "1", "PIEGA_GPU_BLOCCO": "7"}), \
                patch.object(torch, "load", return_value={"stato": {3: {}}}), \
                patch.object(CP, "piega_strato", side_effect=strato):
            PG.main(["--pt", "finto.pt", "--gguf", "finto.gguf", "--out", out,
                     "--strati", "3", "--device", "cpu"])
    assert chiamate == [3] and CP.jarq_lotto is originale and CP.chol_ex is chol_prima


@dataclass
class Caso:
    nome: str
    T: torch.Tensor
    S: torch.Tensor
    W: torch.Tensor
    H: torch.Tensor
    tid: int
    righe_totali: int
    preparazione: float


def guardie_reali(H, device="cuda"):
    """Lotto 4-D, fallimento SPD e NaN iniettati soltanto su una copia piccola."""
    a = AlgebraGPU(blocco=7, pausa=.002)
    M = H[:20, :20].clone().expand(2, 2, 20, 20).contiguous()
    gpu = M.to(device)
    l, info = a._chol_blocchi(gpu)
    assert not bool(info.any())
    lc = torch.linalg.cholesky(M)
    assert torch.allclose(l.cpu(), lc, rtol=1e-11, atol=1e-12)
    B = H[:20, :3].clone().expand(2, 2, 20, 3).contiguous()
    x = a._risolvi_triangolari(B.to(device), l)
    assert torch.allclose(x.cpu(), torch.cholesky_solve(B, lc), rtol=1e-10, atol=1e-11)
    gpu[1, 0, 0, 0] = -1.
    gpu[1, 1, 0, 0] = float("nan")
    with warnings.catch_warnings(record=True) as avvisi:
        _, info = a._chol_blocchi(gpu)
    assert int(info[1, 0]) != 0 and int(info[1, 1]) != 0
    assert a.ripieghi.get("cholesky") == 2 and avvisi
    trsm = torch.linalg.solve_triangular
    chiamate = []

    def inietta(*args, **kwargs):
        y = trsm(*args, **kwargs)
        chiamate.append(True)
        if len(chiamate) == 1:
            y[0, 0, 0, 0] = float("nan")
        return y

    torch.linalg.solve_triangular = inietta
    try:
        with warnings.catch_warnings(record=True) as avvisi:
            x = a._risolvi_triangolari(B.to(device), l)
        assert torch.allclose(x.cpu(), torch.cholesky_solve(B, lc), rtol=1e-10, atol=1e-11)
        assert a.ripieghi.get("sistema triangolare") == 1 and avvisi
    finally:
        torch.linalg.solve_triangular = trsm
    print(f"Guardie {device} su H vera: batch 4-D, blocco finale, non-SPD, NaN e ripiego TRSM OK",
          flush=True)


def confronta(caso, out, CM, pausa, algebra):
    risultati = {}
    rec = {"nome": caso.nome, "righe": caso.T.shape[-2], "righe_totali": caso.righe_totali,
           "colonne": caso.T.shape[-1], "piani": caso.T.shape[0],
           "preparazione_cpu_s": caso.preparazione}
    modi = [("cpu_fp64", "cpu", torch.float64, False),
            ("gpu_fp64", "cuda", torch.float64, True)]
    if "_exps" in caso.nome:
        modi.append(("originale_gpu_fp32", "cuda", torch.float32, False))
    for modo, dev, dt, abilitato in modi:
        t0 = time.monotonic()
        with percorso(CP, abilitato, algebra):
            W, H = caso.W.to(dev, dt).unsqueeze(0), caso.H.to(dev, dt).unsqueeze(0)
            T, S = caso.T.to(dev).unsqueeze(0), caso.S.to(dev, dt).unsqueeze(0)
            t, s, cat = jarq_protetto(CP, W, H, T, S, giri=3, nu2=1e-4)
            t, s = t[0].cpu(), s[0].cpu().half()
        rec[f"{modo}_s"] = time.monotonic() - t0
        cartella = out / modo
        cartella.mkdir(exist_ok=True)
        bb = []
        for k in range(t.shape[0]):
            b = CM.gp.encode_rows(t[k].numpy(), s[k].numpy(), caso.tid)
            tp, sp = CM.gp.decode_rows(b, caso.tid, caso.T.shape[-1])
            assert np.array_equal(tp, t[k].numpy())
            assert np.array_equal(sp.view(np.uint16), s[k].numpy().view(np.uint16))
            np.save(cartella / f"{caso.nome}.p{k}.npy", b)
            bb.append(b)
        q = CP.ricostruisci_lotto(t.unsqueeze(0), s.double().unsqueeze(0))[0]
        ha = caso.H + 1e-4 * torch.eye(caso.H.shape[-1], dtype=torch.float64)
        e = float(((caso.W - q) @ ha * (caso.W - q)).sum())
        risultati[modo] = (t, s, q, bb, e)
        rec[f"{modo}_errore_H"] = e
        print(f"{caso.nome} {modo}: {rec[f'{modo}_s']:.3f} s; errore H {e:.12g}", flush=True)
        time.sleep(pausa)
    t, s, q, bb, e = risultati["gpu_fp64"]
    for modo in ("cpu_fp64", "originale_gpu_fp32"):
        if modo not in risultati:
            continue
        tc, sc, qc, bc, ec = risultati[modo]
        rec[f"contro_{modo}"] = {
            "npy_identici": all(np.array_equal(x, y) for x, y in zip(bb, bc)),
            "byte_diversi": sum(int(np.count_nonzero(x != y)) for x, y in zip(bb, bc)),
            "byte_totali": sum(x.size for x in bb),
            "trit_diversi": int((t != tc).sum()),
            "scale_diverse": int((s != sc).sum()),
            "max_delta_scale": float((s.double() - sc.double()).abs().max()),
            "max_delta_peso": float((q - qc).abs().max()),
            "delta_rel_errore_H": (e - ec) / max(abs(ec), 1e-300)}
    rec["ripieghi_cumulativi"] = dict(algebra.ripieghi)
    with open(out / "risultati.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def reali(a):
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError("Avviare il collaudo con nice -n 19")
    torch.set_num_threads(1)
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    from rd_ternary.formato import TensorKey
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "ingressi.json", "w") as f:
        json.dump({"pt": str(a.pt), "gguf": str(a.gguf), "delta": str(a.delta),
                   "righe": a.righe, "torch": torch.__version__, "hip": torch.version.hip,
                   "gpu": torch.cuda.get_device_name(), "nice": 19}, f, indent=2)
    algebra = AlgebraGPU(blocco=a.blocco, pausa=a.pausa_blocco)
    # Priorita' 0 = minima priorita' HIP; non preempie i kernel gia' in corso.
    stream = torch.cuda.Stream(priority=0)
    records = []
    sondata = False

    def prova(caso):
        nonlocal sondata
        with torch.no_grad(), torch.cuda.stream(stream), vieta_rocsolver():
            if not sondata:
                guardie_reali(caso.H)
                sondata = True
            records.append(confronta(caso, out, CM, a.pausa, algebra))

    if a.sezione in ("tutti", "esperti"):
        dd = torch.load(a.pt, map_location="cpu", weights_only=False)["stato"]
        for L in (3, 32):
            src = CM.gp.GGUFPlaneSource(str(a.gguf), layers=[L],
                                        bases=["ffn_gate_exps", "ffn_up_exps"])
            hs = CP.Hessiane(L)
            stato = dd.get(L, dd.get(str(L)))
            for famiglia in ("gate", "up"):
                t0 = time.monotonic()
                key = TensorKey(L, 0, famiglia)
                tt = src.load(key)
                R = tt.planes.shape[-2]
                r = min(a.righe or R, R)
                T, S = tt.planes[:, :r], tt.scales[:, :r].double()
                H = CM.ruota_H(hs.di(0)[0])
                Q = CP.ricostruisci_lotto(T.unsqueeze(0), S.unsqueeze(0))[0]
                W = Q + stato[f"C_{famiglia}"][0, :r].double() @ stato[f"D_{famiglia}"].double().T
                prova(Caso(f"blk.{L}.ffn_{famiglia}_exps.e0", T, S, W, H,
                           src.entry(key).fmt.type_id, R, time.monotonic() - t0))
        del dd
    if a.sezione in ("tutti", "densi"):
        delta = torch.load(a.delta, map_location="cpu", weights_only=False)["delta_file"]
        with open(PD.MANIF_R2B3) as f:
            manif = json.load(f)
        for nome in a.nomi.split(","):
            t0 = time.monotonic()
            nome = PD.nome_gguf(nome)
            L = int(nome.split(".")[1])
            src = CM.gp.GGUFPlaneSource(str(a.gguf), layers=[L])
            d = CM.Denso(src, nome)
            R = d.out
            r = min(a.righe or R, R)
            dW = delta[nome.removesuffix(".weight")][:r].double()
            H = PD.hessiana(nome, manif)
            T, S = d.T[:, :r], d.S[:, :r].double()
            Q = d.ricostruisci(T, S)
            W = Q + (CM.ruota(dW) if d.rotated else dW)
            H = CM.ruota_H(H) if d.rotated else H
            prova(Caso(nome, T, S, W, H, d.tid, R, time.monotonic() - t0))
    with open(out / "riepilogo.json", "w") as f:
        json.dump(records, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reali", action="store_true")
    ap.add_argument("--guardie-cpu", action="store_true",
                    help="Usa una piccola H_ffn reale, senza inizializzare HIP")
    ap.add_argument("--out")
    ap.add_argument("--pt", type=Path, default=PT)
    ap.add_argument("--gguf", type=Path, default=GGUF)
    ap.add_argument("--delta", type=Path, default=DELTA)
    ap.add_argument("--righe", type=int, default=4)
    ap.add_argument("--sezione", choices=("tutti", "esperti", "densi"), default="tutti")
    ap.add_argument("--nomi", default="blk.3.attn_k,blk.0.attn_gate,blk.0.ssm_out")
    ap.add_argument("--blocco", type=int, default=32)
    ap.add_argument("--pausa", type=float, default=2.)
    ap.add_argument("--pausa-blocco", type=float, default=.002)
    a = ap.parse_args()
    if a.righe < 0:
        ap.error("--righe deve essere >= 0")
    test_contesto_cpu()
    test_ripiego_jarq()
    test_wrapper_cpu()
    print("Prove CPU e ripristino contesto: OK", flush=True)
    if a.guardie_cpu:
        H = torch.from_numpy(np.load(Path(CP.HFFN_DIR) / "H_ffn_03.npy").astype(np.float64))
        H = CP._normalizza(H)
        guardie_reali(H, "cpu")
    if a.reali:
        if not a.out:
            ap.error("--out richiesto per --reali")
        reali(a)


if __name__ == "__main__":
    main()
