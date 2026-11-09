#!/usr/bin/env python3
"""Prove di `--peso-token` (27/9/2026, ramo kl-pesata, TECNICHE_E2E_PV.md §35): la perdita del micro-lotto (KL in
avanti o CAKLD) pesata per token, w FISSI (senza gradiente) a media 1. SOLO CPU. W1-W4 senza file su /mnt (studente
finto di e2e_test_allena); W5 legge il GGUF VERO di Thor-T (strato 0) e il corpus held-out, come A3/A4.

Uso:
    nice -n 15 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 E2E_ALLENA_PROVE=/tmp/prove_peso \\
        @@ROOT:venv@@/bin/python3 e2e_test_peso_token.py [W1 W2 W3 W4 W5]
    esito: $E2E_ALLENA_PROVE/peso_token/esito_test_peso_token[_<prove>].json

  W1 default AL BIT (peso_token "uniforme") contro il codice di master 5b6c51f: kl_pezzo e KLFusa (valore e gradiente,
     fp32 percorso vero e fp64, kl e cakld) e il CICLO intero (allena, studente finto, kl e cakld: theta, Adam, stato,
     file della corsa e istantanee del migliore al byte); campi nuovi della config = soli peso_* ai default
  W2 gradiente pesato fuso a pezzi (17 su 64 token) == autograd float64 di sum_t w_t l_t con w FISSI (calcolati da un
     riferimento INDIPENDENTE: bisezione per "kl", ordinamento Python per "topk"), per kl e cakld (gamma 0,3 e 1) e sei
     configurazioni dei pesi: errore <= 1e-10 su valore, gradiente, pesi; somma uniforme == KLFusa senza pesi AL BIT;
     percorso vero fp32 (maestro bf16, logit fp32) contro fp64: <= 1e-5
  W3 pesi: media 1 (|sum w / N - 1| <= 1e-13), tetto rispettato, struttura del riempimento a tetto (w/r costante sotto il
     tetto, c r >= M sopra), == bisezione <= 1e-12, alfa 0 -> tutti 1, invarianza per scala di l (potenza di 2: al bit),
     topk (k = ceil(f N) token, N/k, i piu' alti, pari merito stabile), ripieghi (tutti zero, troppo pochi positivi)
  W4 ciclo con i pesi accesi (studente finto): 3 passi + ripresa da disco + 3 == 6 filati AL BIT (kl + pesi kl, cakld +
     topk, kl + pesi kl col tetto attivo); passo 1 e held-out 0 == corsa uniforme AL BIT (kl_train uniforme), theta
     diverso dopo; media dei pesi 1 in ogni voce; accesi e spenti alla ripresa (decisione registrata); held-out e
     sonda del passo V identici al bit con e senza pesi; CLI (argparse vero di `main`) e config rifiutate
  W5 (strato 0 VERO, ricetta B = indietro fp16) scala K di G1 coi pesi: G1 per uniforme / kl / topk (K, dY, confronto
     fp32) e, alla K di G1 UNIFORME (i pesi accesi alla ripresa), dY col peso: niente trabocco, quota di dY azzerati
     <= 1e-4, primo passo pieno atteso (|g| >= 99 eps_G1) >= 99% delle scale vive
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_test_allena as TA  # noqa: E402

DIR = Path(os.getenv("E2E_ALLENA_PROVE", "@@ROOT:lavoro@@/e2e/prove_allena")) / "peso_token"
BASE = "5b6c51f"                          # master su cui nasce il ramo kl-pesata
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "base": BASE, "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
CONFIG_PESI = [("kl", 1.0, 5.0, 0.6), ("kl", 2.0, 1.5, 0.6), ("kl", 0.5, 5.0, 0.6), ("topk", 1.0, 5.0, 0.6),
               ("topk", 1.0, 5.0, 0.25), ("topk", 1.0, 5.0, 1.0)]


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:1800]}", flush=True)
    return bool(ok)


# ─────────────────────────────── riferimento INDIPENDENTE dei pesi ───────────────────────────────
def pesi_rif(l, peso: tuple) -> np.ndarray:
    """Stessa definizione di `A.pesi_token`, altro algoritmo: "kl" per bisezione su c (200 dimezzamenti in fp64) di
    sum_t min(c r_t, M) = N; "topk" con `sorted` di Python (chiave (-l, indice): pari merito all'indice minore)."""
    modo, alfa, M, f = peso
    l = np.asarray(l, dtype=np.float64)
    N = l.size
    if modo == "topk":
        k = min(N, max(1, math.ceil(f * N)))
        w = np.zeros(N)
        w[[i for _, i in sorted((-float(x), i) for i, x in enumerate(l))[:k]]] = N / k
        return w
    m = l.mean()
    if not (m > 0 and math.isfinite(m)):
        return np.ones(N)
    r = (np.maximum(l, 0.0) / m) ** alfa
    pos = r > 0
    if pos.sum() * M <= N:
        return np.ones(N)
    lo, hi = 0.0, M / r[pos].min()
    for _ in range(200):
        c = 0.5 * (lo + hi)
        if np.minimum(c * r, M).sum() < N:
            lo = c
        else:
            hi = c
    return np.minimum(0.5 * (lo + hi) * r, M)


def per_token(hS, WS, hT, WT, gamma):
    """Perdita per token (fp64, autograd, non a pezzi): KL(T||S) o CAKLD; piu' le due KL per token."""
    lS = torch.log_softmax(hS @ WS.T, -1)
    lT = torch.log_softmax(A._logit_maestro(hT, WT, hS.dtype), -1)
    kf = (lT.exp() * (lT - lS)).sum(-1)
    ki = (lS.exp() * (lS - lT)).sum(-1)
    return (kf if gamma is None else gamma * ki + (1.0 - gamma) * kf), kf, ki


def _ca(**kw):
    return TA._ca_finta(**kw)


# ─────────────────────────────────────────── W1 ───────────────────────────────────────────
def prova_w1() -> bool:
    t0 = time.time()
    vec = TA._modulo_prima(BASE)
    ok_p, det = True, {}
    g = torch.Generator().manual_seed(31)
    for dt in (torch.float32, torch.float64):
        P, H, V = 70, 32, 1000
        hS = (torch.randn(P, H, generator=g) * 1.5).to(dt)
        WS = (torch.randn(V, H, generator=g) * 0.5).to(dt)
        hT = (torch.randn(P, H, generator=g) * 1.5).to(torch.bfloat16 if dt == torch.float32 else dt)
        WT = (torch.randn(V, H, generator=g) * 0.5).to(dt)
        for gamma in (None, 0.3):
            n = A.kl_pezzo(hS, WS, hT, WT, True, gamma=gamma)
            o = vec.kl_pezzo(hS, WS, hT, WT, True, gamma=gamma)
            u = set(n) == set(o) and all(torch.equal(n[k], o[k]) for k in n)
            x1, x2 = hS.clone().requires_grad_(True), hS.clone().requires_grad_(True)
            a1 = (x1, WS, hT, WT, 23) if gamma is None else (x1, WS, hT, WT, 23, gamma)
            a2 = (x2, WS, hT, WT, 23) if gamma is None else (x2, WS, hT, WT, 23, gamma)
            v1, v2 = A.KLFusa.apply(*a1), vec.KLFusa.apply(*a2)
            s1 = v1 if gamma is None else v1[0]
            s2 = v2 if gamma is None else v2[0]
            (s1 * 2.0 ** 20 / 4096).backward()
            (s2 * 2.0 ** 20 / 4096).backward()
            uf = torch.equal(x1.grad, x2.grad) and (torch.equal(v1, v2) if gamma is None else
                                                    all(torch.equal(a, b) for a, b in zip(v1, v2)))
            det[f"{dt}_{gamma}"] = {"kl_pezzo": u, "KLFusa": uf}
            ok_p &= u and uf
    ok_n = A.peso_di(A.ConfigAllena()) is None and A.ConfigAllena().peso_token == "uniforme"
    src, WS = TA.sorgente_finta()
    cicli = {}
    for nome, kw in (("kl", {}), ("cakld", {"perdita": "cakld"}), ("kl_ema_beta2", {"ema_theta": 0.5, "beta2": 0.9})):
        ca = _ca(**kw)
        es = {}
        for lato, mod in (("nuovo", A), ("prima", vec)):
            d = DIR / f"w1_{nome}_{lato}"
            shutil.rmtree(d, ignore_errors=True)
            st = TA.StudenteFinto()
            stato = mod.allena(st, src, d, ca if mod is A else TA._config_per(mod, ca), WS, WS, log=muto)
            es[lato] = (st, stato, mod.carica_ripresa(d, "cpu"), d)
        (sn, zn, rn, dn), (sp, zp, rp, dp) = es["nuovo"], es["prima"]
        th = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
        so, sr = rn["opt"]["state"], rp["opt"]["state"]
        adam = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step")) \
            and rn["opt"]["param_groups"] == rp["opt"]["param_groups"] and set(rn) == set(rp)
        a_, b_ = TA._senza_tempi(zn), TA._senza_tempi(zp)
        nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
        a_.pop("versione", None)
        b_.pop("versione", None)
        fn, fp_ = TA._sha_cartella(dn), TA._sha_cartella(dp)
        ist = [k for k in fn if k.startswith("migliore_") and not k.endswith("meta.json")]
        file_ok = set(fn) == set(fp_) and len(ist) > 0 and all(fn[k] == fp_[k] for k in ist)
        c = {"theta": th, "adam": adam, "stato": a_ == b_ and set(zn) == set(zp), "file": file_ok,
             "campi_nuovi": nuovi == TA._CFG_INTEGRATA_DEFAULT,  # integrata: anche i campi degli altri rami, ai default
             "passi": zn["passo"],
             "senza_pesi": all("pesi" not in x for x in zn["storia"]) and "pesi_passo0" not in zn["g1"]}
        cicli[nome] = c
        ok_p &= all(v for k, v in c.items() if k != "passi") and zn["passo"] == 6
    return registra("W1_default_al_bit", ok_p and ok_n, base=BASE, pezzi=det, peso_di_none=ok_n, cicli=cicli,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── W2 ───────────────────────────────────────────
def prova_w2() -> bool:
    t0 = time.time()
    g = torch.Generator().manual_seed(32)
    P, H, V = 64, 24, 1000
    hS = (torch.randn(P, H, generator=g, dtype=torch.float64) * 1.3).requires_grad_(True)
    WS = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.5
    hT = torch.randn(P, H, generator=g, dtype=torch.float64) * 1.3
    WT = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.5
    ok, err = True, {}
    for gamma in (None, 0.3, 1.0):
        base = A.KLFusa.apply(hS.detach(), WS, hT, WT, 17) if gamma is None else \
            A.KLFusa.apply(hS.detach(), WS, hT, WT, 17, gamma)[0]
        for peso in CONFIG_PESI:
            dg = {}
            out = A.KLFusa.apply(hS, WS, hT, WT, 17, gamma, peso, dg)
            (out[0] * 3.0).backward()
            gf = hS.grad.clone()
            hS.grad = None
            l, kf, ki = per_token(hS, WS, hT, WT, gamma)
            w_rif = torch.from_numpy(pesi_rif(l.detach().numpy(), peso))
            ((w_rif * l).sum() * 3.0).backward()
            gr = hS.grad.clone()
            hS.grad = None
            lf = A.kl_pezzo(hS.detach(), WS, hT, WT, False, gamma=gamma)["kl" if gamma is None else "perdita"]
            w_nostro, _ = A.pesi_token(lf, peso)
            rif_v = float((w_rif * l.detach()).sum())
            e = {"valore": abs(float(out[0].detach()) - rif_v) / abs(rif_v),
                 "grad_rel": float((gf - gr).norm() / gr.norm()), "grad_max_abs": float((gf - gr).abs().max()),
                 "pesi_max_abs": float((w_nostro - w_rif).abs().max()),
                 "somma_uniforme": abs(float(out[1]) - float(l.detach().sum())) / float(l.detach().sum()),
                 "media_w_meno_1": abs(float(dg["somma_w"]) / P - 1.0)}
            if gamma is not None:
                e["kl_fwd"] = abs(float(out[2]) - float(kf.detach().sum())) / float(kf.detach().sum())
                e["kl_inv"] = abs(float(out[3]) - float(ki.detach().sum())) / float(ki.detach().sum())
            uni_al_bit = torch.equal(out[1], base)
            nome = f"{'kl' if gamma is None else f'cakld{gamma}'}_{peso[0]}_a{peso[1]:g}_M{peso[2]:g}_f{peso[3]:g}"
            err[nome] = {**e, "uniforme_al_bit": uni_al_bit, "max_w": float(dg["max_w"]),
                         "tagliati": int(dg["tagliati"])}
            ok &= all(v <= 1e-10 for v in e.values()) and uni_al_bit
    tetto_attivo = any(v["tagliati"] > 0 for k, v in err.items() if "_kl_" in k and "M1.5" in k)
    # percorso VERO in fp32 (logit del maestro bf16 -> fp32 via e2e_teacher), contro fp64 degli stessi valori
    g2 = torch.Generator().manual_seed(33)
    h32 = (torch.randn(80, 32, generator=g2) * 1.5).requires_grad_(True)
    W32 = torch.randn(2000, 32, generator=g2) * 0.5
    hT16 = (torch.randn(80, 32, generator=g2) * 1.5).to(torch.bfloat16)
    WT32 = torch.randn(2000, 32, generator=g2) * 0.5
    e32 = {}
    for gamma, peso in ((None, ("kl", 1.0, 5.0, 0.6)), (0.3, ("topk", 1.0, 5.0, 0.6))):
        out = A.KLFusa.apply(h32, W32, hT16, WT32, 23, gamma, peso, {})
        out[0].backward()
        g32 = h32.grad.clone()
        h32.grad = None
        h64 = h32.detach().double().requires_grad_(True)
        l, _, _ = per_token(h64, W32.double(), hT16.double(), WT32.double(), gamma)
        w = torch.from_numpy(pesi_rif(l.detach().numpy(), peso))
        (w * l).sum().backward()
        rv = float((w * l.detach()).sum())
        e32[f"{gamma}_{peso[0]}"] = {"valore": abs(float(out[0].detach()) - rv) / rv,
                                     "grad_rel": float((g32.double() - h64.grad).norm() / h64.grad.norm())}
    ok32 = all(v <= 1e-5 for d in e32.values() for v in d.values())
    return registra("W2_gradiente_pesato_contro_autograd_fp64", ok and ok32 and tetto_attivo, P=P, V=V, pezzo=17,
                    errori=err, peggiore={k: max(v[k] for v in err.values()) for k in
                                          ("valore", "grad_rel", "grad_max_abs", "pesi_max_abs", "media_w_meno_1")},
                    tetto_attivo=tetto_attivo, fp32_percorso_vero=e32, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── W3 ───────────────────────────────────────────
def _casi_l() -> dict:
    g = np.random.default_rng(34)
    casi = {"lognormale_s1": np.exp(g.normal(0, 1, 8192)) * 0.3, "lognormale_s3": np.exp(g.normal(0, 3, 8192)),
            "uniforme": g.uniform(0, 1, 5000), "zeri_30pc": np.where(g.uniform(size=4000) < 0.3, 0.0,
                                                                     g.exponential(1.0, 4000)),
            "pari_merito": np.round(g.uniform(0, 3, 3000)), "N1": np.array([0.7]), "N2": np.array([0.1, 2.0]),
            "N7": g.exponential(1.0, 7), "negativi_minuscoli": np.concatenate([g.exponential(1.0, 900),
                                                                              -1e-12 * g.uniform(size=100)]),
            "tutti_uguali": np.full(1000, 0.37)}
    return casi


def prova_w3() -> bool:
    t0 = time.time()
    ok, esiti = True, {}
    confs = [("kl", a, M, 0.6) for a in (0.0, 0.5, 1.0, 2.0) for M in (1.0, 1.5, 5.0, 1e6)] + \
            [("topk", 1.0, 5.0, f) for f in (0.01, 0.25, 0.6, 1.0)]
    peggiori = {"media": 0.0, "rif": 0.0, "struttura": 0.0}
    for nome, l_np in _casi_l().items():
        l = torch.from_numpy(l_np.astype(np.float64))
        N = l.numel()
        for peso in confs:
            w, info = A.pesi_token(l, peso)
            w2, _ = A.pesi_token(l.clone(), peso)
            modo, alfa, M, f = peso
            wr = pesi_rif(l_np, peso)
            media = abs(float(w.sum()) / N - 1.0)
            d_rif = float((np.abs(w.numpy() - wr) / np.maximum(1.0, np.abs(wr))).max())   # relativo sopra 1
            c = {"media": media <= 1e-13, "rif": d_rif <= 1e-12, "deterministico": torch.equal(w, w2),
                 "non_negativi": bool((w >= 0).all()), "info_somma": abs(float(info["somma_w"]) - float(w.sum())) == 0}
            if modo == "kl":
                c["tetto"] = float(w.max()) <= M
                if not int(info["ripiego"]):
                    r = (l.clamp_min(0) / l.mean()).pow(alfa)
                    sotto = (w < M) & (r > 0)
                    if int(sotto.sum()) > 1:
                        q = w[sotto] / r[sotto]
                        s = float((q.max() - q.min()) / q.mean())
                        peggiori["struttura"] = max(peggiori["struttura"], s)
                        c["struttura"] = s <= 1e-13 and bool((q.mean() * r[w >= M] >= M * (1 - 1e-13)).all())
                    if M >= 1e6 and alfa == 1.0:
                        q0 = l.clamp_min(0) / l.mean()
                        c["senza_tetto_l_su_media"] = float(((w - q0).abs() / q0.clamp_min(1.0)).max()) <= 1e-13
                if alfa == 0.0:
                    c["alfa0_tutti_uno"] = bool((w == 1).all())
                for lam in (2.0 ** -20, 2.0 ** 13):             # scala di l per potenza di 2 (come K): pesi al bit
                    c[f"scala_{lam:g}"] = torch.equal(A.pesi_token(l * lam, peso)[0], w)
                c["scala_3.7"] = float(((A.pesi_token(l * 3.7, peso)[0] - w).abs() / w.clamp_min(1.0)).max()) <= 1e-12
            else:
                k = min(N, max(1, math.ceil(f * N)))
                tenuti = w > 0
                c["topk"] = (int(tenuti.sum()) == k and bool((w[tenuti] == N / k).all()) and int(info["tagliati"]) == k
                             and (k == N or float(l[tenuti].min()) >= float(l[~tenuti].max())))
            peggiori["media"] = max(peggiori["media"], media)
            peggiori["rif"] = max(peggiori["rif"], d_rif)
            if not all(c.values()):
                esiti[f"{nome}_{peso}"] = c
            ok &= all(c.values())
    # ripieghi: tutti zero; meno di N/M token positivi
    rip = {}
    for nome, l_np, M in (("tutti_zero", np.zeros(100), 5.0), ("un_positivo_su_100", np.eye(1, 100)[0], 5.0),
                          ("dieci_su_100_M5", np.r_[np.ones(10), np.zeros(90)], 5.0)):
        w, info = A.pesi_token(torch.from_numpy(l_np), ("kl", 1.0, M, 0.6))
        rip[nome] = {"ripiego": int(info["ripiego"]), "tutti_uno": bool((w == 1).all())}
    ok_r = all(v["ripiego"] == 1 and v["tutti_uno"] for v in rip.values())
    # pari merito nel topk: stabile (indice minore)
    wt, _ = A.pesi_token(torch.tensor([1.0, 2.0, 2.0, 2.0, 0.5], dtype=torch.float64), ("topk", 1.0, 5.0, 0.4))
    ok_pm = wt.tolist() == [0.0, 2.5, 2.5, 0.0, 0.0]
    return registra("W3_pesi_media_1_tetto_topk", ok and ok_r and ok_pm, peggiori=peggiori, falliti=esiti,
                    ripieghi=rip, pari_merito_stabile=ok_pm, casi=list(_casi_l()), configurazioni=len(confs),
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── W4 ───────────────────────────────────────────
def _cli(argv: list):
    """ConfigAllena costruita dall'argparse VERO di `A.main` (fermato subito dopo `_config_da_arg`)."""
    class Preso(Exception):
        pass
    orig = A._config_da_arg

    def cattura(a):
        raise Preso(orig(a))
    A._config_da_arg = cattura
    try:
        A.main(["allena", "--corsa", "x", "--teacher", "/nessuno", "--device", "cpu"] + argv)
    except Preso as e:
        return e.args[0]
    finally:
        A._config_da_arg = orig
    raise RuntimeError("main non ha costruito la config")


def prova_w4() -> bool:
    t0 = time.time()
    src, WS = TA.sorgente_finta()
    varianti = {"kl_pesi_kl": {"peso_token": "kl"}, "cakld_topk": {"perdita": "cakld", "peso_token": "topk"},
                "kl_pesi_kl_tetto": {"peso_token": "kl", "peso_alfa": 2.0, "peso_max": 1.5}}
    d_u = DIR / "w4_uniforme"
    shutil.rmtree(d_u, ignore_errors=True)
    st_u = TA.StudenteFinto()
    uni = A.allena(st_u, src, d_u, _ca(), WS, WS, log=muto)
    d_uc = DIR / "w4_uniforme_cakld"
    shutil.rmtree(d_uc, ignore_errors=True)
    uni_c = A.allena(TA.StudenteFinto(), src, d_uc, _ca(perdita="cakld"), WS, WS, log=muto)
    ok, det = True, {}
    for nome, kw in varianti.items():
        ca = _ca(**kw)
        d_f, d_r = DIR / f"w4_{nome}_filata", DIR / f"w4_{nome}_ripresa"
        for d in (d_f, d_r):
            shutil.rmtree(d, ignore_errors=True)
        st_f = TA.StudenteFinto()
        reg_f = []
        filata = A.allena(st_f, src, d_f, ca, WS, WS, log=reg_f.append)
        A.allena(TA.StudenteFinto(), src, d_r, ca, WS, WS, ferma_a=3, log=muto)
        st_b = TA.StudenteFinto()
        ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=muto)
        th = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
        rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
        so, sr = rf["opt"]["state"], rr["opt"]["state"]
        adam = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step"))
        stato = TA._senza_tempi(filata) == TA._senza_tempi(ripresa)
        rif = uni if kw.get("perdita", "kl") == "kl" else uni_c
        M = ca.peso_max if ca.peso_token == "kl" else 1.0 / ca.peso_frazione + 1e-12
        voci = filata["storia"]
        pesi_ok = all("pesi" in x and abs(x["pesi"]["media_w"] - 1.0) <= 1e-12 and x["pesi"]["max_w"] <= M + 1e-12
                      and x["pesi"]["ripieghi"] == 0 for x in voci) and "pesi_passo0" in filata["g1"]
        passo1_uguale = voci[0]["kl_train"] == rif["storia"][0]["kl_train"] and \
            filata["storia_heldout"][0] == rif["storia_heldout"][0]
        diverso = not all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_u.parametri()))
        tetto = [x["pesi"]["tagliati_quota"] for x in voci]
        log_ok = any("pesi per token" in r for r in reg_f)
        c = {"theta": th, "adam": adam, "stato": stato, "pesi": pesi_ok, "passo1_e_heldout0_uniformi": passo1_uguale,
             "allenamento_diverso": diverso, "log": log_ok, "passi": filata["passo"] == 6}
        if nome == "kl_pesi_kl_tetto":
            c["tetto_attivo"] = max(tetto) > 0
        det[nome] = {**c, "pesi_passo2": voci[1]["pesi"], "heldout": [(x["passo"], round(x["metrica"], 6))
                                                                       for x in filata["storia_heldout"]],
                     "heldout_uniforme": [(x["passo"], round(x["metrica"], 6)) for x in rif["storia_heldout"]]}
        ok &= all(c.values())
    # accesi alla ripresa, poi spenti (decisioni registrate)
    d_a = DIR / "w4_accesi_alla_ripresa"
    shutil.rmtree(d_a, ignore_errors=True)
    A.allena(TA.StudenteFinto(), src, d_a, _ca(), WS, WS, ferma_a=2, log=muto)
    r1, r2 = [], []
    s1 = A.allena(TA.StudenteFinto(), src, d_a, _ca(peso_token="kl"), WS, WS, riprendi=True, ferma_a=4, log=r1.append)
    s2 = A.allena(TA.StudenteFinto(), src, d_a, _ca(), WS, WS, riprendi=True, log=r2.append)
    ha = ["pesi" in x for x in s2["storia"]]
    ok_a = (ha == [False, False, True, True, False, False] and any("peso_token uniforme -> kl" in x for x in r1)
            and any("peso_token kl -> uniforme" in x for x in r2) and s2["passo"] == 6
            and s1["config"]["peso_token"] == "kl" and s2["config"]["peso_token"] == "uniforme")
    # held-out e sonda del passo V: uniformi, identici al bit con e senza pesi (stesso theta)
    st = TA.StudenteFinto()
    with torch.no_grad():
        for p in st.parametri():
            p.add_(0.05)
    car = A.Caricatore(src, torch.device("cpu"))
    cu, cw = _ca(), _ca(peso_token="kl")
    ccu, ccw = _ca(perdita="cakld", gamma=0.3), _ca(perdita="cakld", gamma=0.3, peso_token="topk")
    ok_s = (A.kl_sonda(st, car, "heldout", [0, 1], cu, WS, WS) == A.kl_sonda(st, car, "heldout", [0, 1], cw, WS, WS)
            and A.kl_sonda(st, car, "train", [2], ccu, WS, WS) == A.kl_sonda(st, car, "train", [2], ccw, WS, WS)
            and A.valuta(st, car, [0, 1, 2, 3], cu, WS, WS) == A.valuta(st, car, [0, 1, 2, 3], cw, WS, WS)
            and A.valuta(st, car, [0, 1, 2, 3], ccu, WS, WS) == A.valuta(st, car, [0, 1, 2, 3], ccw, WS, WS))
    car.chiudi()
    # CLI vero e config rifiutate
    c0 = _cli([])
    c1 = _cli(["--peso-token", "kl", "--peso-alfa", "2", "--peso-max", "3"])
    c2 = _cli(["--peso-token", "topk", "--peso-frazione", "0.5", "--perdita", "cakld"])
    ok_c = ((c0.peso_token, c0.peso_alfa, c0.peso_max, c0.peso_frazione) == ("uniforme", 1.0, 5.0, 0.6)
            and (c1.peso_token, c1.peso_alfa, c1.peso_max) == ("kl", 2.0, 3.0)
            and (c2.peso_token, c2.peso_frazione, c2.perdita) == ("topk", 0.5, "cakld")
            and {"peso_token", "peso_alfa", "peso_max", "peso_frazione"} <= A._CAMPI_DI_DECISIONE)
    rifiuti = []
    for kw in ({"peso_token": "xx"}, {"peso_token": "kl", "peso_max": 0.5}, {"peso_token": "topk", "peso_frazione": 0.0},
               {"peso_token": "topk", "peso_frazione": 1.5}, {"peso_token": "kl", "peso_alfa": -1.0},
               {"peso_alfa": 2.0}, {"peso_token": "kl", "peso_frazione": 0.5}, {"peso_token": "topk", "peso_max": 3.0},
               {"peso_token": "kl", "peso_alfa": float("nan")}):
        try:
            _ca(**kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    ok_c &= all(rifiuti)
    try:                                     # argparse rifiuta un modo sconosciuto
        _cli(["--peso-token", "focal"])
        ok_c = False
    except SystemExit:
        pass
    return registra("W4_ciclo_ripresa_al_bit_coi_pesi", ok and ok_a and ok_s and ok_c, varianti=det,
                    accesi_e_spenti_alla_ripresa=ok_a, pesi_per_passo=ha, heldout_e_sonda_uniformi=ok_s, cli=ok_c,
                    rifiuti=rifiuti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── W5 ───────────────────────────────────────────
def _quota_piena(st, K: float, eps_g1: float) -> float:
    """Quota delle scale vive con g != 0 per cui il primo passo di Adam (eps_G1 in unita' vere) e' >= 0,99 lr:
    |g|/K >= 99 eps_G1 (|g| / (|g| + eps) >= 0,99)."""
    n = pieni = 0
    for (L, f), p in zip(st.chiavi(), st.parametri()):
        g = p.grad.double() / K
        v = (st.esperti(L).fam[f].d0 != 0) & (g != 0)
        n += int(v.sum())
        pieni += int((v & (g.abs() >= 99.0 * eps_g1)).sum())
    return pieni / max(n, 1)


def prova_w5() -> bool:
    t0 = time.time()
    st0 = TA.studente((0,))
    WS = TA.testa_ristretta(st0)
    th = TA.theta_stella(st0)
    src = TA.sorgente(st0, 8, 4, th)
    del st0
    righe = [0, 1]
    confs = {"uniforme": {}, "kl": {"peso_token": "kl"}, "topk": {"peso_token": "topk"}}
    g1s, ok = {}, True
    for nome, kw in confs.items():
        st = TA.studente((0,))
        car = A.Caricatore(src, st.device)
        g1 = A.cancello_g1(st, car, "train", righe, TA._ca(**kw), WS, WS, muto)
        car.chiudi()
        g1s[nome] = {"K": g1["K"], "escalation": g1["escalation"], "eps_g1": g1["eps_g1"], "q01": g1["q01"],
                     "dy": {f: {k: v[k] for k in ("quota_azzerati", "max_abs_dy")} for f, v in g1["dy"].items()},
                     "confronto_fp32": {k: g1.get("confronto_fp32", {}).get(k) for k in ("quota_rotti",
                                                                                     "err_rel_mediano")},
                     "pesi": g1.get("pesi_passo0"), "quota_piena": _quota_piena(st, g1["K"], g1["eps_g1"])}
        del st
    Ku, eu = g1s["uniforme"]["K"], g1s["uniforme"]["eps_g1"]
    # pesi accesi alla ripresa: stessa K ed eps di G1 UNIFORME
    fissa = {}
    for nome in ("kl", "topk"):
        st = TA.studente((0,))
        car = A.Caricatore(src, st.device)
        st.diagnostica_indietro(True)
        A.gradiente_passo(st, car, "train", righe, TA._ca(**confs[nome]), WS, WS, Ku)
        dy = A.aggrega_dy(st.diagnostica_indietro(False))
        ctl = st.controlla(tollera_trabocco=True)
        car.chiudi()
        max_u = max(v["max_abs_dy"] for v in g1s["uniforme"]["dy"].values())
        fissa[nome] = {"trabocco": ctl["trabocco_fp16"],
                       "dy": {f: {k: v[k] for k in ("quota_azzerati", "max_abs_dy")} for f, v in dy.items()},
                       "max_dy_su_uniforme": max(v["max_abs_dy"] for v in dy.values()) / max_u,
                       "margine_trabocco": 65504.0 / max(v["max_abs_dy"] for v in dy.values()),
                       "quota_piena": _quota_piena(st, Ku, eu)}
        del st
        ok &= (not fissa[nome]["trabocco"] and all(v["quota_azzerati"] <= 1e-4 for v in dy.values())
               and fissa[nome]["quota_piena"] >= 0.99)
    for nome, g in g1s.items():
        ok &= g["quota_piena"] >= 0.99 and all(v["quota_azzerati"] <= 1e-4 for v in g["dy"].values())
        if g["pesi"] is not None:
            ok &= abs(g["pesi"]["media_w"] - 1.0) <= 1e-12
    return registra("W5_scala_K_di_G1_coi_pesi_strato0_vero", ok, g1=g1s, pesi_accesi_con_K_uniforme=fissa,
                    K_uguali={n: g["K"] == Ku for n, g in g1s.items()}, secondi=round(time.time() - t0, 1))


PROVE = {"W1": prova_w1, "W2": prova_w2, "W3": prova_w3, "W4": prova_w4, "W5": prova_w5}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    TA.DIR = DIR                              # _modulo_prima e le cartelle di lavoro dello studente finto qui
    scelte = [a.upper() for a in sys.argv[1:]] or list(PROVE)
    ok, t0 = True, time.time()
    for k in scelte:
        t1 = time.time()
        try:
            ok &= PROVE[k]()
        except Exception as e:
            ok = False
            registra(f"{k}_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-3000:])
        print(f"   ({k}: {time.time() - t1:.1f} s)", flush=True)
    ESITO["tutto_verde"] = ok
    ESITO["secondi"] = round(time.time() - t0, 1)
    suff = "_" + "_".join(scelte) if sys.argv[1:] else ""
    (DIR / f"esito_test_peso_token{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
