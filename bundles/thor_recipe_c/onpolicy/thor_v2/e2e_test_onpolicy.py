#!/usr/bin/env python3
"""Prove SOLO CPU dell'OPD (e2e_onpolicy.py + aggancio in e2e_allena.py), 12/10/2026. Nessun file su /mnt scritto.

  O1  termine_opd (fuso, a pezzi) == riferimento autograd float64: gradiente dL/dhS per campionata / piena / eakld,
      con e senza PIVOT (+ ancore); valore di piena/eakld == riferimento
  O2  PIVOT-Q: al piu' fasi x m stati per completione, distanza >= sep, pesi in [0,5, 2]/media e media 1, determinismo,
      lo stato PRIMA di un picco isolato di discrepanza e' scelto (r_t)
  O3  verificatore GSM8K e vantaggi GRPO
  O4  impacchettamento: ogni completione un segmento, le posizioni predicono SOLO token generati, mai un inizio;
      leggi_onpolicy rifiuta corpus manomesso e posizione di confine
  O5  ciclo `allena` (studente finto, 4 passi) con --peso-token generati in ogni modo: finito, theta si muove, e la
      KL inversa sulle posizioni generate SCENDE (piena, eakld); registro con le voci OPD
  O6  ConfigAllena: generati senza cakld/gamma 1 rifiutato; opd_* senza generati rifiutato
  O7  interruttori spenti: ciclo identico AL BIT al modulo congelato di E2E-C1
Uso: env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 e2e_test_onpolicy.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import e2e_allena as A
import e2e_onpolicy as O
from e2e_test_allena import StudenteFinto, _ca_finta

ESITI = {}


def esito(nome, ok, **d):
    ESITI[nome] = {"ok": bool(ok), **d}
    print(f"{'OK ' if ok else 'KO '} {nome} {json.dumps(d, default=str)[:300]}")
    return ok


def _ls(h, W):
    return h @ W.T


def _lt(h, W, dt):
    return h.to(dt) @ W.T.to(dt)


def _pt(p, W):
    return p.to(W.dtype) @ W


def _dati(seme=1, P=96, H=12, V=300, n_seg=3, ancore=0):
    g = torch.Generator().manual_seed(seme)
    hS = torch.randn(P + 1, H, generator=g, dtype=torch.float64)
    hT = torch.randn(P + 1, H, generator=g, dtype=torch.float64)
    WS = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.8
    WT = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.8
    pos = torch.arange(P)
    seg = torch.repeat_interleave(torch.arange(n_seg), P // n_seg)
    anc = torch.zeros(P, dtype=torch.bool)
    if ancore:
        anc[-ancore:] = True
    adv = torch.randn(P, generator=g, dtype=torch.float64)
    y = torch.randint(0, V, (P,), generator=g)
    return hS, hT, WS, WT, {"pos": pos, "y": y, "adv": adv, "seg": seg, "anc": anc, "beta": 1.0}


def prova_o1():
    ok, peggiore = True, {}
    for modo in O.MODI_KL:
        for pivot in (None, dict(O.PIVOT_DEFAULT)):
            hS, hT, WS, WT, opd = _dati(ancore=10 if pivot else 0)
            opd.update(modo=modo, pivot=pivot, eakld_k=16)
            dh = torch.zeros_like(hS)
            info = {}
            v = O.termine_opd(hS, WS, hT, WT, opd, 17, dh, _ls, _lt, _pt, info)
            h2 = hS.clone().requires_grad_(True)
            ref = O.termine_opd_riferimento(h2, WS, hT, WT, opd, info["_pesi_opd"], info["_lambda_opd"])
            ref.backward()
            e = float((dh - h2.grad).abs().max() / h2.grad.abs().max())
            ev = float(abs(v - ref.detach()) / abs(ref.detach())) if modo != "campionata" else 0.0
            k = f"{modo}{'+pivot' if pivot else ''}"
            peggiore[k] = (e, ev)
            ok &= e < 2e-5 and ev < 2e-5 and bool(torch.isfinite(v))
    return esito("O1 gradiente fuso == autograd", ok, errori_rel=peggiore)


def prova_o2():
    cfg = dict(O.PIVOT_DEFAULT)
    rng = np.random.default_rng(3)
    q = rng.random(200) * 0.1
    q[57] = 50.0                                         # stato isolato a discrepanza enorme
    seg = np.repeat(np.arange(2), 100)
    w = O.scegli_vulnerabili(q, seg, cfg)
    w2 = O.scegli_vulnerabili(q, seg, cfg)
    ok = np.array_equal(w, w2)
    for s in range(2):
        sel = np.nonzero((seg == s) & (w > 0))[0]
        ok &= len(sel) <= cfg["fasi"] * cfg["m"] and abs(w[sel].mean() - 1) < 1e-9
        ok &= w[sel].max() / w[sel].min() <= 4.0 + 1e-9
        ok &= all(np.diff(sel) >= 1)
        for f in np.array_split(np.arange(100), 4):       # distanza minima dentro ogni fase
            ss = sorted(set(sel) & set(f + 100 * s))
            ok &= all(b - a >= cfg["sep"] for a, b in zip(ss, ss[1:]))
    # ranghi percentili (non grandezze, come il paper): 56 ha r massimo (lo stato 57 e' nel suo futuro scontato)
    ok &= w[56] > 0
    return esito("O2 PIVOT-Q scelta", ok, scelti=int((w > 0).sum()), quota=float((w > 0).mean()))


def prova_o3():
    ok = O.oro_gsm8k("blah\n#### 1,234") == 1234.0
    ok &= O.ricompensa("so 5+5\n#### 10", 10.0) == 1.0 and O.ricompensa("#### 11", 10.0) == 0.0
    ok &= O.ricompensa("the answer is 10.", 10.0) == 1.0 and O.ricompensa("nessun numero", 10.0) == 0.0
    ok &= O.ricompensa("#### 3\n...\n#### 10", 10.0) == 1.0
    a = O.vantaggi_grpo([1, 0, 0, 1])
    ok &= abs(sum(a)) < 1e-12 and abs(np.std(a) - 1) < 1e-12 and O.vantaggi_grpo([1, 1, 1, 1]) == [0.0] * 4
    return esito("O3 verificatore e GRPO", ok, adv=a)


def prova_o4():
    rng = np.random.default_rng(4)
    seqs = [{"prompt": rng.integers(5, 99, int(rng.integers(3, 20))).tolist(),
             "gen": rng.integers(100, 200, int(rng.integers(1, 40))).tolist(), "adv": float(k)} for k in range(30)]
    pk = O.impacchetta(seqs, 128, 2, righe_max=4)
    ok = len(pk["ids"]) == 4
    for r in range(4):
        cu = pk["cu"][r]
        inizi = np.zeros(128, np.int32)
        inizi[cu[:-1]] = 1
        t = pk["pos"][r]
        ok &= not inizi[t + 1].any() and (pk["ids"][r][t + 1] >= 100).all()   # predicono solo token generati
        for k in set(pk["seg"][r].tolist()):
            ok &= (pk["adv"][r][pk["seg"][r] == k] == float(k)).all()
            ok &= int((pk["seg"][r] == k).sum()) == len(seqs[k]["gen"])
    d = Path(tempfile.mkdtemp())
    try:
        w = O.scrivi_lato(d, "train", "x", pk, {})
        f = Path(w["corpus"])
        iz = lambda r: np.isin(np.arange(128), pk["cu"][r][:-1]).astype(np.int32)  # noqa: E731
        O.leggi_onpolicy(d / "onpolicy_train.pt", f, 128, 4, iz)
        rifiuti = 0
        b = bytearray(f.read_bytes())
        f2 = d / "manomesso.pt"
        f2.write_bytes(bytes(b[:-1]) + bytes([b[-1] ^ 1]))
        try:
            O.leggi_onpolicy(d / "onpolicy_train.pt", f2, 128, 4, iz)
        except ValueError:
            rifiuti += 1
        x = torch.load(d / "onpolicy_train.pt", weights_only=True)
        x["pos"][0, 1] = int(pk["cu"][0][1]) - 1             # logit che predice l'inizio del secondo segmento
        torch.save(x, d / "onpolicy_train.pt")
        try:
            O.leggi_onpolicy(d / "onpolicy_train.pt", f, 128, 4, iz)
        except ValueError:
            rifiuti += 1
        ok &= rifiuti == 2
    finally:
        shutil.rmtree(d)
    return esito("O4 impacchettamento e verifica", ok, righe=len(pk["ids"]), scartate=len(pk["scartate"]))


def _sorgente_op(n_train=8, n_held=4, T=32, V=512, seme=5, ancore=False):
    """Maestro = studente finto con theta* (come sorgente_finta di e2e_test_allena); tre «completioni» per riga, le
    ultime 2/3 di ogni terzo sono posizioni generate, vantaggi a caso (gruppi), qualche ancora se chiesto."""
    st = StudenteFinto(V)
    g = torch.Generator().manual_seed(seme)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
    righe, dati = {}, {}
    for lato, n in (("train", n_train), ("heldout", n_held)):
        rr, dd = [], {}
        for i in range(n):
            ids = torch.randint(0, V, (T,), generator=g).numpy().astype(np.int32)
            h = st.nascosto(torch.from_numpy(ids.astype(np.int64))[None]).to(torch.bfloat16).clone()
            rr.append((ids, h, "opd", None))
            t, a, s, an = [], [], [], []
            for k in range(3):
                o = k * (T // 3)
                tt = np.arange(o + 3, o + T // 3 - 1)
                t.append(tt)
                a.append(np.full(len(tt), float(torch.randn(1, generator=g)), np.float32))
                s.append(np.full(len(tt), 100 * i + k))
                an.append(np.full(len(tt), 1 if (ancore and k == 2 and i % 2 == 0) else 0, np.int8))
            dd[i] = tuple(np.concatenate(x) for x in (t, a, s, an))
        righe[lato], dati[lato] = rr, dd
    WS = torch.randn(V, 16, generator=g) * 0.7
    return O.SorgenteOnPolicy(A.SorgenteMemoria(righe, {}), dati), WS


def _kl_inv_generati(st, src, WS):
    tot, n = 0.0, 0
    for i in src.righe_previste("train"):
        ids = torch.from_numpy(np.asarray(src.ids("train", i), np.int64))[None]
        hS = st.nascosto(ids)
        t, _, _, an = src.generati("train", i)
        t = torch.as_tensor(t)[torch.as_tensor(an) == 0]
        lS = torch.log_softmax(hS[t] @ WS.T, -1)
        lT = torch.log_softmax(src.h("train", i)[t].float() @ WS.T, -1)
        tot += float((lS.exp() * (lS - lT)).sum())
        n += len(t)
    return tot / n


def prova_o5():
    ok, mis = True, {}
    for modo, pivot in (("campionata", ""), ("piena", ""), ("eakld", ""), ("piena", "H=4,m=2,sep=1"),
                        ("campionata", "H=4,m=2,sep=1")):
        src, WS = _sorgente_op(ancore=bool(pivot))
        ca = _ca_finta(perdita="cakld", gamma=1.0, peso_token="generati", opd_kl=modo, opd_pivot=pivot,
                       righe_per_passo=2, lr=3e-2)
        st = StudenteFinto()
        prima = _kl_inv_generati(st, src, WS)
        th0 = [p.detach().clone() for p in st.parametri()]
        d = Path(tempfile.mkdtemp())
        reg = []
        try:
            A.allena(st, src, d, ca, WS, WS, log=reg.append)
        finally:
            shutil.rmtree(d)
        dopo = _kl_inv_generati(st, src, WS)
        mosso = any(not torch.equal(a, b) for a, b in zip(th0, st.parametri()))
        fin = all(torch.isfinite(p).all() for p in st.parametri())
        k = f"{modo}{'+pivot' if pivot else ''}"
        mis[k] = (round(prima, 5), round(dopo, 5))
        ok &= mosso and fin and any("OPD" in str(r) for r in reg)
        if modo != "campionata":
            ok &= dopo < prima
    return esito("O5 ciclo allena con OPD", ok, kl_inversa_generati_prima_dopo=mis)


def prova_o6():
    rif = 0
    for kw in (dict(peso_token="generati", perdita="kl"), dict(peso_token="generati", perdita="cakld", gamma=0.5),
               dict(opd_kl="piena"), dict(peso_token="generati", perdita="cakld", gamma=1.0, opd_kl="boh"),
               dict(peso_token="generati", perdita="cakld", gamma=1.0, opd_pivot="Z=3")):
        try:
            _ca_finta(**kw)
        except ValueError:
            rif += 1
    ca = _ca_finta(peso_token="generati", perdita="cakld", gamma=1.0, opd_kl="eakld", opd_pivot="m=4")
    return esito("O6 configurazione", rif == 5 and A.peso_di(ca) == ("generati",), rifiutate=rif)


def prova_o7():
    """Interruttore spento = IDENTICO AL BIT al modulo congelato di E2E-C1 (e2e_allena.py.orig-e2ec1, stessa sha del
    tree di E2E-C1): ciclo kl, cakld e cakld + peso-token kl sullo studente finto, theta finali al bit."""
    import importlib.util
    from e2e_test_allena import sorgente_finta
    f = Path(__file__).with_name("e2e_allena.py.orig-e2ec1")
    import importlib.machinery
    sp = importlib.util.spec_from_loader("e2e_allena_orig", importlib.machinery.SourceFileLoader("e2e_allena_orig", str(f)))
    P = importlib.util.module_from_spec(sp)
    sys.modules["e2e_allena_orig"] = P
    sp.loader.exec_module(P)
    ok, dett = True, {}
    for nome, kw in (("kl", {}), ("cakld", dict(perdita="cakld", gamma=0.4)),
                     ("cakld_peso_kl", dict(perdita="cakld", gamma=0.4, peso_token="kl"))):
        ris = []
        for M in (A, P):
            src, WS = sorgente_finta()
            base = dict(lr=3e-2, righe_per_passo=2, righe_per_micro=1, righe_per_micro_valuta=2, heldout_ogni=2,
                        ripresa_ogni=2, pezzo_kl=24, attesa_ogni_s=0.01, **kw)
            st = StudenteFinto()
            d = Path(tempfile.mkdtemp())
            try:
                M.allena(st, src, d, M.ConfigAllena(**base), WS, WS, log=lambda *_: None)
            finally:
                shutil.rmtree(d)
            ris.append([p.detach().clone() for p in st.parametri()])
        uguale = all(torch.equal(x, y) for x, y in zip(*ris))
        dett[nome] = uguale
        ok &= uguale
    return esito("O7 spento == E2E-C1 al bit", ok, **dett)


if __name__ == "__main__":
    torch.set_num_threads(4)
    r = [prova_o1(), prova_o2(), prova_o3(), prova_o4(), prova_o5(), prova_o6(), prova_o7()]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps({"ok": all(r), "prove": ESITI}, indent=1, default=str))
    print("TUTTE VERDI" if all(r) else "QUALCHE PROVA ROSSA")
    sys.exit(0 if all(r) else 1)
