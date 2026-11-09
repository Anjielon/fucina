#!/usr/bin/env python3
"""Banco GPU dei densi PIEGATI (1/10/2026, ramo densi-piegati): per ogni famiglia densa di R2B3m (forme VERE, 3 piani,
byte veri) e un micro-lotto di T token (default 8192, quello di TT5), tempo di

  avanti     (senza gradiente: l'avanti dello studente)
  av+ind     (avanti col grafo + indietro: il ricalcolo dello strato nell'indietro, dX e dL/dd dei 3 piani)
  av+ind V2  (lo stesso col motore V in modo 2 = verifica: il P dei passi con V, kernel `_k_dd_v` oggi)

nelle vie: «oggi» (3 FamigliaEsperti, motore E2E_MOTORE = triton2, ricetta --ricetta, default B3_ind32; opz. anche
B3_ind2) e «piegato» x2 (gemm auto = torch.mm out_dtype fp32, e gemm triton = `_k_mm16`) ed f32. Piu': costo della sola
costruzione del peso effettivo (con e senza cache dei trit), memoria (picco torch sopra la base), errori GPU contro la
via di oggi e contro i LIMITI deterministici dichiarati (su 512 token portati in CPU, `e2e_test_densi_piegati.controlla`
con u = 2^-23: l'accumulo WMMA di gfx1151 tronca).

Proiezione al passo: per micro-lotto un denso costa avanti + av+ind; per passo P 16 micro-lotti; conteggi per famiglia
da R2B3m (30 strati GDN, 10 d'attenzione piena, 40 esperti condivisi). Il passo V (sonde, dosi, curvatura) NON e' qui:
si stima a parte (PIEGATI.md §5).

Uso (GPU LIBERA: lo lancia `strumenti/cancello_densi_piegati.sh`, mai con TT5 viva):
    python3 e2e_banco_densi_piegati.py --uscita DIR [--token 8192] [--ripetizioni 5] [--ricette B3_ind32,B3_ind2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["E2E_DP_GPU"] = "1"                 # e2e_test_densi_piegati (limiti) importato senza toccare l'interprete
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_densi_piegati as DP  # noqa: E402
from gguf import GGUFReader  # noqa: E402

MOTORE = os.environ.get("E2E_MOTORE", "triton2")
CONTI_R2B3M = {"attn_qkv": 30, "attn_gate": 30, "ssm_out": 30, "ffn_gate_shexp": 40, "ffn_up_shexp": 40,
               "ffn_down_shexp": 40, "attn_q": 10, "attn_k": 10, "attn_v": 10, "attn_output": 10}


def _ms(fn, n: int) -> float:
    if not torch.cuda.is_available():            # prova di fumo su CPU: tempo a parete, una volta
        t0 = time.perf_counter()
        fn()
        return 1000 * (time.perf_counter() - t0)
    fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b))
    return sorted(t)[len(t) // 2]


def _picco(fn) -> float:
    if not torch.cuda.is_available():
        fn()
        return 0.0
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def _err(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def piani(tens: dict, L: int, base: str, device, righe=None) -> list:
    return [S.famiglia_da_tensore(tens[f"blk.{L}.{S.FAMIGLIE_DENSE[base]}{suf}.weight"], f"{base}{suf}", device=device,
                                  righe=righe) for suf in ("", S.SUFFISSO_P2, S.SUFFISSO_P3)]


def primo_strato(tens: dict, base: str) -> int:
    Ls = sorted(int(m.group(1)) for n in tens if (m := re.match(rf"blk\.(\d+)\.{S.FAMIGLIE_DENSE[base]}3\.weight$", n)))
    return Ls[0]


def oggi(xg, ds, fams, ric, mot, gy=None):
    piano = S.piano_unico(xg.shape[0], xg.device)
    if gy is None:
        with torch.no_grad():
            y = None
            for f, d in zip(fams, ds):
                yk = S.FamigliaEsperti.apply(xg, d, f, piano, ric, mot, None)
                y = yk if y is None else y + yk
            return y, None, None
    x = xg.detach().requires_grad_(True)
    dl = [d.detach().requires_grad_(True) for d in ds]
    y = None
    for f, d in zip(fams, dl):
        yk = S.FamigliaEsperti.apply(x, d, f, piano, ric, mot, None)
        y = yk if y is None else y + yk
    y.backward(gy)
    return y.detach(), x.grad, [d.grad for d in dl]


def piegato(xg, ds, fams, ric, mot, modo, gy=None):
    if gy is None:
        with torch.no_grad():
            return DP.DensoPiegato.apply(xg, tuple(fams), ric, mot, None, modo, *ds), None, None
    x = xg.detach().requires_grad_(True)
    dl = [d.detach().requires_grad_(True) for d in ds]
    y = DP.DensoPiegato.apply(x, tuple(fams), ric, mot, None, modo, *dl)
    y.backward(gy)
    return y.detach(), x.grad, [d.grad for d in dl]


def banco(gguf: Path, uscita: Path, T: int = 8192, ripetizioni: int = 5, ricette=("B3_ind32",), log=print,
          device: str = "cuda", famiglie=None, righe=None) -> dict:
    """`device` "cpu" + `famiglie`/`righe` (sottoinsieme): SOLO prova di fumo su CPU (TRITON_INTERPRET=1)."""
    dev = torch.device(device)
    tens = {t.name: t for t in GGUFReader(str(gguf)).tensors}
    import e2e_test_densi_piegati as TP
    TP.U = 2.0 ** -23                                   # accumulo WMMA troncato: u di troncamento
    out = {"gguf": str(gguf), "token": T, "motore": MOTORE, "ricette": list(ricette), "famiglie": {}}
    g = torch.Generator(device="cpu").manual_seed(2026)
    for base in (famiglie or CONTI_R2B3M):
        L = primo_strato(tens, base)
        fams = piani(tens, L, base, dev, righe)
        f0 = fams[0]
        ds = [(f.d0.cpu() * torch.exp(0.01 * torch.randn(f.d0.shape, generator=g))).half().float().to(dev) for f in fams]
        x = torch.randn(T, f0.C, generator=g).to(dev)
        xg = (S.ruota(x) if f0.ruotata else x).half().contiguous()
        gy = (torch.randn(T, f0.R, generator=g) * 1e-3).to(dev)
        r = {"strato": L, "R": f0.R, "C": f0.C, "gs": f0.gs, "tipo": f0.tipo, "ms": {}, "picco_MiB": {}, "errori": {}}
        mot = S.motore(MOTORE)
        stv = PV.StatoV()
        motv = PV.MotoreTritonV(stv, v2=getattr(mot, "v2", False)) if isinstance(mot, S.MotoreTriton) \
            else PV.MotoreTorchV(stv)
        stv.prepara(fams)
        for f in fams:
            stv.buffer(f).prop.fill_(1.0)                   # candidati presenti (colonna 0): verifica piena
        vie = {}
        for ric_n in ricette:
            ric = S.RICETTE[ric_n]
            vie[f"oggi_{ric_n}"] = (lambda gy_, m_, ric=ric: oggi(xg, ds, fams, ric, m_, gy_))
        ric0 = S.RICETTE[ricette[0]]
        for modo, gemm in (("x2", "triton"), ("f32", "f32")):
            def via(gy_, m_, modo=modo, gemm=gemm):
                DP.imposta(gemm=gemm)
                return piegato(xg, ds, fams, ric0, m_, modo, gy_)
            vie[f"piegato_{modo}_{gemm}"] = via
        for nome, fn in vie.items():
            try:
                stv.modo = 0
                r["ms"][f"{nome}|avanti"] = _ms(lambda: fn(None, mot), ripetizioni)
                r["ms"][f"{nome}|av+ind"] = _ms(lambda: fn(gy, mot), ripetizioni)
                r["picco_MiB"][nome] = round(_picco(lambda: fn(gy, mot)), 1)
                stv.modo = 2
                r["ms"][f"{nome}|av+ind V2"] = _ms(lambda: fn(gy, motv), ripetizioni)
                stv.modo = 0
            except Exception as ex:                          # una via che fallisce non ferma le altre
                r["ms"][f"{nome}|errore"] = repr(ex)[:300]
                log(f"banco {base} {nome}: {ex!r}")
        # costo della sola costruzione del peso effettivo
        DP.imposta(gemm="triton")
        for cache in (False, True):
            DP.imposta(cache_trit=cache)
            if cache:
                [DP.trit(f, mot) for f in fams]
            r["ms"][f"costruzione_peso_x2|cache_trit={int(cache)}"] = _ms(
                lambda: DP.peso_x2(DP.peso_effettivo_f32([DP.trit(f, mot) for f in fams], ds, f0.gs)), ripetizioni)
        DP.imposta(cache_trit=False)
        # errori sul dispositivo (T token) contro la via di oggi della prima ricetta
        o = oggi(xg, ds, fams, ric0, mot, gy)
        for modo, gemm in (("x2", "triton"), ("f32", "f32")):
            try:
                DP.imposta(gemm=gemm)
                p = piegato(xg, ds, fams, ric0, mot, modo, gy)
                r["errori"][f"{modo}_{gemm}"] = {"Y": _err(p[0], o[0]), "dX": _err(p[1], o[1]),
                                                 "dX_quota_diversi": float((p[1] != o[1]).float().mean()),
                                                 "dd": [_err(a, b) for a, b in zip(p[2], o[2])]}
            except Exception as ex:
                r["errori"][f"{modo}_{gemm}"] = {"errore": repr(ex)[:300]}
        del o
        # limiti deterministici su 512 token (CPU per fp64; aritmetica della GPU)
        n = 512
        fc = piani(tens, L, base, "cpu", righe)
        n = min(n, T)
        xs, gs_ = xg[:n].contiguous(), (gy[:n] * torch.logspace(-6, 2, n, device=dev)[:, None]).contiguous()
        o = [t.cpu() if torch.is_tensor(t) else [u.cpu() for u in t] for t in oggi(xs, ds, fams, ric0, mot, gs_)]
        for modo, gemm in (("x2", "triton"), ("f32", "f32")):
            try:
                DP.imposta(gemm=gemm)
                p = [t.cpu() if torch.is_tensor(t) else [u.cpu() for u in t]
                     for t in piegato(xs, ds, fams, ric0, mot, modo, gs_)]
                c = TP.controlla(xs.cpu(), [d.cpu() for d in ds], fc, gs_.cpu(), o, p, modo)
                r["errori"][f"limiti_{modo}_{gemm}"] = {"dentro": c["dentro_i_limiti"], "eccesso": c["eccesso_limite"],
                                                        "dd_vs_fp64_piegato": c["dd_piegato_vs_fp64"],
                                                        "dd_vs_fp64_oggi": c["dd_oggi_vs_fp64"],
                                                        "dX_vs_fp64_piegato": c["dX_piegato_vs_fp64"],
                                                        "dX_vs_fp64_oggi": c["dX_oggi_vs_fp64"]}
            except Exception as ex:
                r["errori"][f"limiti_{modo}_{gemm}"] = {"errore": repr(ex)[:300]}
        DP.imposta(gemm="triton")
        out["famiglie"][base] = r
        log(f"banco {base} (strato {L}, {f0.R}x{f0.C}): " + ", ".join(f"{k} {v:.2f}" for k, v in r["ms"].items()
                                                                     if isinstance(v, float)))
        del fams, ds, xg, gy, x
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        (uscita / "banco_densi_piegati.json").write_text(json.dumps(out, indent=1, default=str))
    out["proiezione"] = proiezione(out)
    (uscita / "banco_densi_piegati.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"banco densi piegati: proiezione {json.dumps(out['proiezione'])}")
    return out


def proiezione(out: dict, microlotti_P: int = 16) -> dict:
    """s/passo dei densi per via = somma_famiglie conteggio x (avanti + av+ind) x micro-lotti (P di G1, modo 0) e lo
    stesso col modo V2 (P dei passi con V). Il passo V e le valutazioni NON sono inclusi."""
    vie = sorted({k.split("|")[0] for r in out["famiglie"].values() for k in r["ms"] if "|" in k
                  and not k.startswith("costruzione")})
    res = {}
    for v in vie:
        tot0 = totv = 0.0
        ok = True
        for base, r in out["famiglie"].items():
            ms = r["ms"]
            if f"{v}|avanti" not in ms or f"{v}|av+ind" not in ms:
                ok = False
                break
            n = CONTI_R2B3M[base]
            tot0 += n * (ms[f"{v}|avanti"] + ms[f"{v}|av+ind"])
            totv += n * (ms[f"{v}|avanti"] + ms.get(f"{v}|av+ind V2", ms[f"{v}|av+ind"]))
        if ok:
            res[v] = {"s_per_passo_P_modo0": round(tot0 * microlotti_P / 1000, 1),
                      "s_per_passo_P_modoV2": round(totv * microlotti_P / 1000, 1)}
    rif = next((k for k in res if k.startswith("oggi_")), None)
    if rif:
        for v in res:
            res[v]["risparmio_s_modo0"] = round(res[rif]["s_per_passo_P_modo0"] - res[v]["s_per_passo_P_modo0"], 1)
            res[v]["risparmio_s_modoV2"] = round(res[rif]["s_per_passo_P_modoV2"] - res[v]["s_per_passo_P_modoV2"], 1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default="@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--token", type=int, default=8192)
    ap.add_argument("--ripetizioni", type=int, default=5)
    ap.add_argument("--ricette", default="B3_ind32")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("banco densi piegati: serve la GPU")
    usc = Path(a.uscita)
    usc.mkdir(parents=True, exist_ok=True)
    banco(Path(a.gguf), usc, a.token, a.ripetizioni, tuple(c for c in a.ricette.split(",") if c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
