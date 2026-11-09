#!/usr/bin/env python3
"""Prove del ramo trainer-mc (27/9/2026): righe a scelta multipla nel ciclo E2E (`e2e_allena.py`: `SorgenteConMC`,
`PianoRighe`, --righe-mc, --peso-token risposta / --beta-risposta, --gamma-lettere, `valuta_mc`). SOLO CPU, studente
finto di e2e_test_allena (`StudenteFinto`), maestro SINTETICO noto (lo studente con theta*), righe da 32 token,
vocabolario di 512 (le 4 lettere vere del tokenizzatore di Thor, 357 417 351 414, ci stanno). Nessun file letto da /mnt
salvo il codice del commit di partenza (git show); tutto scritto in $E2E_ALLENA_PROVE/mc (default su @@ROOT:models@@).

Uso:
    nice -n 19 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
        OPENBLAS_NUM_THREADS=4 E2E_ALLENA_PROVE=@@ROOT:models@@/gguf/tony-tern/trainer-mc-prove/dopo \\
        @@ROOT:venv@@/bin/python3 e2e_test_mc.py [M1 M2 ...]
    esito: $E2E_ALLENA_PROVE/mc/esito_test_mc[_<prove>].json

  M1 opzioni SPENTE identiche AL BIT al codice di partenza 3234682 (merge di kl-pesata, prima di trainer-mc): kl_pezzo,
     KLFusa (anche coi pesi kl/topk), cicli interi (kl, cakld, pesi kl, cakld+topk, EMA+beta2+held-out rapido: theta,
     Adam, stato, istantanee del migliore al byte; campi nuovi = soli righe_mc/beta_risposta/gamma_lettere ai default,
     nessuna chiave "mc"); sorgente composta con righe_mc = 0 e senza held-out mc == sorgente base al bit; `PianoRighe`
     classico == formula di prima (epoche con resto scartato)
  M2 composizione delle righe: R - M base + M mc a passo, mc nelle posizioni di `intercala`; righe base == flusso delle
     epoche nell'ordine di sempre, senza buchi ne' doppioni; righe mc a giri (giro 0 = ordine del teacher, poi
     permutazioni col seme); passi totali; accensione a meta' corsa = le righe base continuano da dove erano; nel CICLO
     le righe di ogni passo == il piano, segmenti e passi totali nello stato
  M3 pesi-risposta: w = 1 ovunque e 1 + lambda ESATTAMENTE alle posizioni b * T + t delle risposte (i logit in t
     predicono la lettera in t + 1), lambda = lambda_risposta, quota dei pesi alle risposte == beta sul passo; coef
     delle lettere = G (1 + lambda); gradiente fuso (a pezzi) == autograd float64 di sum w l + sum coef KL4 con w fissi
     (kl e cakld; risposta+lettere, solo lettere, solo risposta) <= 1e-10; micro-lotto senza risposte == uniforme al bit;
     voce "pesi" del registro (risposte, peso, quota beta)
  M4 KL4 = 0 (e gradiente del termine = 0, dh al bit come senza lettere) quando studente = maestro (fp64 e percorso fp32
     col maestro bf16); valuta_mc con studente = maestro: KL4 0 e misure dello studente == quelle del maestro
  M5 ripresa di una corsa SENZA mc (V-Bop, EMA, cakld, held-out rapido) con l'AGGIUNTA del teacher mc (righe_mc,
     risposta, lettere): accettata, al primo passo dopo la ripresa theta e Adam == ripresa al bit, stato V-Bop riletto al
     bit, EMA == ricorrenza da quella salvata, righe base dal cursore di sempre; poi ripresa a meta' in modo mc == corsa
     filata al bit (theta, Adam, stato, V-Bop); cambio del teacher mc (nuova firma, cursore mc da 0) e spegnimento (M = 0,
     base dal cursore) alla ripresa
  M6 held-out mc registrato a ogni misura (rapida e piena, e al passo 0): accuratezza, p(C | giusta non C), scelta C,
     KL4, margine, scelte == ricalcolo indipendente; una corsa con held-out mc == la stessa senza (theta, Adam, stato
     tolte le voci "mc", migliore e istantanee al byte): le misure mc NON entrano nella scelta del migliore
  M7 percorso su DISCO: cache del teacher finte nel formato vero (piano.json, blocco.json, ids.i32, h .bf16) per base,
     mc e held-out mc (sottocartella heldout/ del corpus con il suo risposte.pt): `apri_sorgente_mc` == sorgente in
     memoria (ciclo al bit); risposte.pt verificato (lettera non in t + 1, riga fuori, t fuori, doppioni, lettere
     ripetute, giusta fuori, chiavi mancanti), modello del teacher mc diverso, held-out non disgiunto; configurazioni
     rifiutate; CLI vera (argparse di `main`); righe_mc senza sorgente mc
  M8 (solo se chiesta: GGUF VERO di Thor-T, strato 0, righe vere del held-out E2E, come W5) pesi-risposta e lettere alla
     scala K della G1 uniforme: ricetta B (indietro fp16) max|dY| rispetto all'uniforme, trabocco, dY azzerati; ricetta
     B_ind32 (indietro fp32, quella di EB2) gradiente finito; valuta_mc sullo studente vero
"""
from __future__ import annotations

import copy
import dataclasses
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
os.environ.setdefault("E2E_ALLENA_PROVE", "@@ROOT:models@@/gguf/tony-tern/trainer-mc-prove")
import e2e_allena as A  # noqa: E402
import e2e_test_allena as TA  # noqa: E402

DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "mc"
BASE = "3234682"                          # merge di kl-pesata nel ramo trainer-mc: il codice PRIMA di trainer-mc
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "base": BASE, "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
V, H, T = 512, 16, 32
ID4 = (357, 417, 351, 414)                # « A» « B» « C» « D» nel tokenizzatore di Thor (verificati il 27/9)
GRIGLIA = (5, 13, 21, 29)                 # posizioni t possibili delle risposte (la lettera in t + 1 <= 30 < T)
OFF = A.OFFSET_MC


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:1800]}", flush=True)
    return bool(ok)


# ─────────────────────────────── dati: maestro sintetico e righe mc ───────────────────────────────
class Fixture:
    """Maestro = StudenteFinto con theta* (come `TA.sorgente_finta`); righe base (train "tr", held-out a/b/a/wiki) e
    righe mc: id casuali, 1-3 risposte per riga su GRIGLIA, la lettera giusta ID4[g] scritta in t + 1; h del maestro in
    bf16 (come la cache vera) o fp32 (`h32`, per le prove «studente = maestro»)."""

    def __init__(self, n_train: int = 24, n_held: int = 4, n_mc: int = 6, n_mc_held: int = 3, seme: int = 5,
                 h32: bool = False):
        g = torch.Generator().manual_seed(seme)
        self.h32 = h32
        self.maestro = TA.StudenteFinto(V)
        with torch.no_grad():
            for p in self.maestro.parametri():
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
        doms = ["a", "b", "a", "wiki"]
        righe = {}
        for lato, n in (("train", n_train), ("heldout", n_held)):
            rr = []
            for i in range(n):
                ids = torch.randint(0, V, (T,), generator=g).numpy().astype(np.int32)
                rr.append((ids, self.h(ids), doms[i % 4] if lato == "heldout" else "tr", None))
            righe[lato] = rr
        self.righe_base = righe
        self.base = A.SorgenteMemoria(righe, {})
        self.WS = torch.randn(V, H, generator=g) * 0.7
        self.mc, self.mc_risp = self.righe_mc(g, n_mc, "mc")
        self.held, self.held_risp = self.righe_mc(g, n_mc_held, "mch")
        self.g = g

    def h(self, ids: np.ndarray) -> torch.Tensor:
        x = self.maestro.nascosto(torch.from_numpy(ids.astype(np.int64))[None])
        return x.clone() if self.h32 else x.to(torch.bfloat16).clone()

    def righe_mc(self, g, n: int, dom: str) -> tuple:
        rr, pos, giu, fonte = [], [], [], []
        for i in range(n):
            ids = torch.randint(0, V, (T,), generator=g)
            k = 1 + int(torch.randint(0, 3, (1,), generator=g))
            for t in sorted(GRIGLIA[j] for j in torch.randperm(len(GRIGLIA), generator=g)[:k].tolist()):
                gg = int(torch.randint(0, 4, (1,), generator=g))
                ids[t + 1] = ID4[gg]
                pos.append((i, t))
                giu.append(gg)
                fonte.append(f"{dom}{i}")
            ids = ids.numpy().astype(np.int32)
            rr.append((ids, self.h(ids), dom, None))
        return rr, {"pos": torch.tensor(pos, dtype=torch.long), "giusta": torch.tensor(giu, dtype=torch.long),
                    "id_lettere": torch.tensor(ID4, dtype=torch.long), "fonte": fonte}

    def sorgente(self, train_mc: bool = True, held_mc: bool = True, mc=None, nome: str = "fixture") -> A.SorgenteConMC:
        d = DIR / nome
        d.mkdir(parents=True, exist_ok=True)
        m = h = None
        firma = {"teacher_mc": "memoria", "train": None, "heldout": None}
        if train_mc:
            righe, risp = mc if mc is not None else (self.mc, self.mc_risp)
            s = A.SorgenteMemoria({"train": righe}, {})
            p = d / "risposte_train.pt"
            torch.save(risp, p)
            r = A.leggi_risposte(p, lambda i: s.ids("train", i), s.righe_previste("train"), T)
            m = (s, r)
            firma["train"] = {"sha256": s.firma()["sha256"], "risposte_sha256": r.sha256}
        if held_mc:
            s = A.SorgenteMemoria({"train": self.held}, {})
            p = d / "risposte_held.pt"
            torch.save(self.held_risp, p)
            r = A.leggi_risposte(p, lambda i: s.ids("train", i), s.righe_previste("train"), T)
            h = (s, "train", r)
            firma["heldout"] = {"sha256": s.firma()["sha256"], "risposte_sha256": r.sha256}
        return A.SorgenteConMC(self.base, m, h, firma)


def _ca(**kw) -> A.ConfigAllena:
    return TA._ca_finta(**kw)


def _pulisci(*dd) -> None:
    for d in dd:
        if Path(d).exists():
            shutil.rmtree(d)


def _uguali_opt(a: dict, b: dict) -> bool:
    return (set(a["state"]) == set(b["state"]) and a["param_groups"] == b["param_groups"]
            and all(set(a["state"][i]) == set(b["state"][i]) and
                    all(torch.equal(a["state"][i][k], b["state"][i][k]) for k in a["state"][i]) for i in a["state"]))


def _senza_mc_voci(s: dict) -> tuple:
    """Stato JSON senza tempi e senza le voci "mc" del held-out (in posto); ritorna (stato, quante voci avevano "mc")."""
    s = TA._senza_tempi(s)
    n = 0
    for k in ("storia_heldout", "storia_heldout_rapido"):
        for x in s.get(k, []):
            if "mc" in x:
                n += 1
                x.pop("mc")
    return s, n


# ─────────────────────────────────────────── M1 ───────────────────────────────────────────
def _vecchie_righe(train: list, ca, t: int) -> tuple:
    """Il calcolo delle righe del passo del codice di prima (e2e_allena @ 3234682, ciclo `allena`)."""
    pe = len(train) // ca.righe_per_passo
    ep, pos = divmod(t, pe)
    return A.ordine_epoca(train, ep, ca.seme)[pos * ca.righe_per_passo:(pos + 1) * ca.righe_per_passo], ep


def prova_m1() -> bool:
    t0 = time.time()
    vec = TA._modulo_prima(BASE)
    ok, det = True, {}
    g = torch.Generator().manual_seed(41)
    for dt in (torch.float32, torch.float64):
        P, Hh, Vv = 70, 32, 1000
        hS = (torch.randn(P, Hh, generator=g) * 1.5).to(dt)
        WS = (torch.randn(Vv, Hh, generator=g) * 0.5).to(dt)
        hT = (torch.randn(P, Hh, generator=g) * 1.5).to(torch.bfloat16 if dt == torch.float32 else dt)
        WT = (torch.randn(Vv, Hh, generator=g) * 0.5).to(dt)
        b = torch.randint(0, Vv, (P,), generator=g)
        for gamma in (None, 0.3):
            n = A.kl_pezzo(hS, WS, hT, WT, True, b, gamma=gamma, top1=True)
            o = vec.kl_pezzo(hS, WS, hT, WT, True, b, gamma=gamma, top1=True)
            u = set(n) == set(o) and all(torch.equal(n[k], o[k]) for k in n)
            for peso in (None, ("kl", 1.0, 5.0, 0.6), ("kl", 2.0, 1.5, 0.6), ("topk", 1.0, 5.0, 0.25)):
                x1, x2 = hS.clone().requires_grad_(True), hS.clone().requires_grad_(True)
                if peso is None:
                    a1 = (x1, WS, hT, WT, 23) if gamma is None else (x1, WS, hT, WT, 23, gamma)
                    a2 = (x2, WS, hT, WT, 23) if gamma is None else (x2, WS, hT, WT, 23, gamma)
                else:
                    a1, a2 = (x1, WS, hT, WT, 23, gamma, peso, {}), (x2, WS, hT, WT, 23, gamma, peso, {})
                v1, v2 = A.KLFusa.apply(*a1), vec.KLFusa.apply(*a2)
                s1 = v1 if isinstance(v1, torch.Tensor) else v1[0]
                s2 = v2 if isinstance(v2, torch.Tensor) else v2[0]
                (s1 * 2.0 ** 20 / 4096).backward()
                (s2 * 2.0 ** 20 / 4096).backward()
                uv = torch.equal(v1, v2) if isinstance(v1, torch.Tensor) else \
                    (len(v1) == len(v2) and all(torch.equal(p_, q_) for p_, q_ in zip(v1, v2)))
                uf = uv and torch.equal(x1.grad, x2.grad)
                det[f"{dt}_{gamma}_{peso and peso[0]}{peso and peso[1]}"] = {"kl_pezzo": u, "KLFusa": uf}
                ok &= u and uf
    # cicli interi contro 3234682
    fx = Fixture()
    cicli = {}
    confs = {"kl": {}, "cakld": {"perdita": "cakld"}, "pesi_kl": {"peso_token": "kl"},
             "cakld_topk": {"perdita": "cakld", "peso_token": "topk"},
             "ema_beta2_rapide": {"ema_theta": 0.5, "beta2": 0.9, "heldout_ogni": 1, "heldout_righe_rapide": 2,
                                  "heldout_pieno_ogni": 3}}
    for nome, kw in confs.items():
        ca = _ca(**kw)
        es = {}
        for lato, mod in (("nuovo", A), ("prima", vec)):
            d = DIR / f"m1_{nome}_{lato}"
            _pulisci(d)
            st = TA.StudenteFinto()
            stato = mod.allena(st, fx.base, d, ca if mod is A else TA._config_per(mod, ca), fx.WS, fx.WS, log=muto)
            es[lato] = (st, stato, mod.carica_ripresa(d, "cpu"), d)
        (sn, zn, rn, dn), (sp, zp, rp, dp) = es["nuovo"], es["prima"]
        th = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
        adam = _uguali_opt(rn["opt"], rp["opt"]) and set(rn) == set(rp)
        a_, b_ = TA._senza_tempi(zn), TA._senza_tempi(zp)
        nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
        a_.pop("versione", None)
        b_.pop("versione", None)
        fn, fp_ = TA._sha_cartella(dn), TA._sha_cartella(dp)
        ist = [k for k in fn if k.startswith("migliore_") and not k.endswith("meta.json")]
        c = {"theta": th, "adam": adam, "stato": a_ == b_ and set(zn) == set(zp) and "mc" not in zn,
             "file": set(fn) == set(fp_) and len(ist) > 0 and all(fn[k] == fp_[k] for k in ist),
             # integrata: nuovi anche i campi dei rami uniti (non quelli di kl-pesata, gia' nel commit di partenza)
             "campi_nuovi": nuovi == {k: v for k, v in TA._CFG_INTEGRATA_DEFAULT.items() if k not in TA._CFG_PESO_DEFAULT},
             "passi": zn["passo"]}
        cicli[nome] = c
        ok &= all(v for k, v in c.items() if k != "passi") and zn["passo"] == 12
    # sorgente composta con righe_mc = 0 e senza held-out mc: trasparente al bit
    ca = _ca(heldout_ogni=1, heldout_righe_rapide=2, heldout_pieno_ogni=3)
    dA, dB = DIR / "m1_composta", DIR / "m1_base"
    _pulisci(dA, dB)
    stA, stB = TA.StudenteFinto(), TA.StudenteFinto()
    zA = A.allena(stA, fx.sorgente(held_mc=False), dA, ca, fx.WS, fx.WS, log=muto)
    zB = A.allena(stB, fx.base, dB, ca, fx.WS, fx.WS, log=muto)
    fa, fb = TA._sha_cartella(dA), TA._sha_cartella(dB)
    ist = [k for k in fa if k.startswith("migliore_") and not k.endswith("meta.json")]   # il resto ha le ore
    trasp = (all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
             and _uguali_opt(A.carica_ripresa(dA, "cpu")["opt"], A.carica_ripresa(dB, "cpu")["opt"])
             and TA._senza_tempi(zA) == TA._senza_tempi(zB) and set(fa) == set(fb) and len(ist) > 0
             and all(fa[k] == fb[k] for k in ist))
    # PianoRighe classico == formula di prima
    train = list(range(26))
    cp = _ca(righe_per_passo=4, epoche=3)
    pr = A.PianoRighe(train, cp)
    pe = 26 // 4
    piano_ok = pr.classico and pr.totale() == pe * 3
    for t in range(pe * 3):
        piano_ok &= (pr.righe(t) == _vecchie_righe(train, cp, t) and pr.fine_epoca(t) == ((t + 1) % pe == 0)
                     and pr.fine_prima_epoca(t) == (t + 1 == pe) and pr.base_consumate(t) == t * 4)
    ok_n = A.pesato_di(A.ConfigAllena()) is False and A.peso_di(A.ConfigAllena()) is None
    return registra("M1_spento_al_bit", ok and trasp and piano_ok and ok_n, base=BASE, pezzi=det, cicli=cicli,
                    composta_trasparente=trasp, piano_classico=piano_ok, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M2 ───────────────────────────────────────────
def prova_m2() -> bool:
    t0 = time.time()
    fx = Fixture(n_train=26, n_mc=5)
    src = fx.sorgente()
    train, mc = src.righe_previste("train"), src.righe_mc()
    ok, det = True, {}
    # (a) piano dal passo 0: R = 4, M = 1, due epoche
    ca = _ca(righe_per_passo=4, righe_mc=1, epoche=2)
    seg = [{"passo": 0, "cursore_base": 0, "cursore_mc": 0, "righe_mc": 1, "firma": 0}]
    pr = A.PianoRighe(train, ca, mc, seg)
    tot = pr.totale()
    flusso = A.ordine_epoca(train, 0, ca.seme) + A.ordine_epoca(train, 1, ca.seme)
    base_cat, mc_cat, posiz, epoche, fini = [], [], True, [], []
    for t in range(tot):
        r, ep = pr.righe(t)
        b = [x for x in r if x < OFF]
        m = [x for x in r if x >= OFF]
        posiz &= len(r) == 4 and len(b) == 3 and len(m) == 1 and [k for k, x in enumerate(r) if x >= OFF] == [3]
        base_cat += b
        mc_cat += m
        epoche.append(ep)
        fini.append(pr.fine_epoca(t))
    det["passi"] = tot
    c_a = {"totale": tot == (26 * 2) // 3, "posizioni": posiz, "base_nel_flusso": base_cat == flusso[:3 * tot],
           "epoca_0_intera_senza_doppioni": sorted(base_cat[:26]) == train,
           "giro_0_mc": mc_cat[:5] == mc, "giro_1_permutato": sorted(mc_cat[5:10]) == mc and mc_cat[5:10] != mc,
           "deterministico": [A.PianoRighe(train, ca, mc, seg).righe(t)[0] for t in range(tot)] ==
                             [pr.righe(t)[0] for t in range(tot)],
           "epoche": epoche == [(3 * t) // 26 for t in range(tot)],
           "fine_epoca": fini == [(3 * t + 3) // 26 > (3 * t) // 26 for t in range(tot)],
           "intercala": (A.intercala([0, 1, 2, 3, 4, 5], [OFF, OFF + 1]) == [0, 1, 2, OFF, 3, 4, 5, OFF + 1] and
                         [k for k, x in enumerate(A.intercala(list(range(24)), [OFF + j for j in range(8)]))
                          if x >= OFF] == [3, 7, 11, 15, 19, 23, 27, 31])}
    det["a"] = c_a
    ok &= all(c_a.values())
    # (b) accensione a meta' (passo 3, piano classico prima): le righe base continuano da dove erano
    cc = _ca(righe_per_passo=4)
    classico = A.PianoRighe(train, cc)
    cb = classico.cursori(3)[0]
    seg2 = [{"passo": 3, "cursore_base": cb, "cursore_mc": 0, "righe_mc": 1, "firma": 0}]
    pm = A.PianoRighe(train, dataclasses.replace(cc, righe_mc=1), mc, seg2)
    dopo = [x for t in range(3, pm.totale()) for x in pm.righe(t)[0] if x < OFF]
    prima = [x for t in range(0, 3) for x in classico.righe(t)[0]]
    c_b = {"cursore": cb == 12, "continua": prima + dopo == A.ordine_epoca(train, 0, cc.seme)[:len(prima) + len(dopo)],
           "totale": pm.totale() == 3 + (26 - 12) // 3, "consumate": pm.base_consumate(5) == 12 + 2 * 3}
    det["b"] = c_b
    ok &= all(c_b.values())
    # (c) nel CICLO: le righe di ogni passo == il piano (spia su gradiente_passo)
    visti: list = []
    orig = A.gradiente_passo

    def spia(st, car, lato, righe, *a, **k):
        if len(righe) == 4 and (not visti or visti[-1] != list(righe)):
            visti.append(list(righe))
        return orig(st, car, lato, righe, *a, **k)
    ca1 = _ca(righe_per_passo=4, righe_mc=1)
    d = DIR / "m2_ciclo"
    _pulisci(d)
    A.gradiente_passo = spia
    try:
        stato = A.allena(TA.StudenteFinto(), src, d, ca1, fx.WS, fx.WS, log=muto)
    finally:
        A.gradiente_passo = orig
    p1 = A.PianoRighe(train, ca1, mc, seg)
    c_c = {"righe": visti == [p1.righe(t)[0] for t in range(p1.totale())], "segmenti": stato["mc"]["segmenti"] == seg,
           "passi_totali": stato["passi_totali"] == p1.totale() == stato["passo"] == 26 // 3,
           "firma_base": stato["sorgente"] == json.loads(json.dumps(fx.base.firma()))}
    det["c"] = {**c_c, "passi_visti": len(visti)}
    ok &= all(c_c.values())
    return registra("M2_composizione_delle_righe", ok, **det, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M3 ───────────────────────────────────────────
def _rif_pesato(hS, WS, hT, WT, gamma, w, pos, id4, coef) -> torch.Tensor:
    """Riferimento float64 non a pezzi, con autograd: sum_t w_t l_t (l = KL(T||S) o CAKLD) + sum_a coef_a KL4_a."""
    lS = torch.log_softmax(hS @ WS.T, -1)
    lT = torch.log_softmax(hT @ WT.T, -1)
    kf = (lT.exp() * (lT - lS)).sum(-1)
    ki = (lS.exp() * (lS - lT)).sum(-1)
    l_ = kf if gamma is None else gamma * ki + (1.0 - gamma) * kf
    tot = (w * l_).sum()
    if pos is not None and pos.numel():
        l4S = torch.log_softmax(hS[pos] @ WS[id4].T, -1)
        l4T = torch.log_softmax(hT[pos] @ WT[id4].T, -1)
        tot = tot + (coef * (l4T.exp() * (l4T - l4S)).sum(-1)).sum()
    return tot


def prova_m3() -> bool:
    t0 = time.time()
    fx = Fixture()
    src = fx.sorgente()
    B, G = 0.1, 0.5
    ca = _ca(righe_per_passo=4, righe_mc=2, peso_token="risposta", beta_risposta=B, gamma_lettere=G)
    seg = [{"passo": 0, "cursore_base": 0, "cursore_mc": 0, "righe_mc": 2, "firma": 0}]
    pr = A.PianoRighe(src.righe_previste("train"), ca, src.righe_mc(), seg)
    ok, det = True, {}
    for t in range(3):
        righe = pr.righe(t)[0]
        attese = {k: src.risposte("train", i) for k, i in enumerate(righe)}
        a = sum(0 if r is None else r[0].numel() for r in attese.values())
        lam = A.lambda_risposta(B, 4 * T, a)
        catture: list = []
        orig = A.KLFusa.apply

        def spia(*x):
            catture.append(x)
            return orig(*x)
        car = A.Caricatore(src, torch.device("cpu"))
        A.KLFusa.apply = spia
        try:
            somme: dict = {}
            A.gradiente_passo(TA.StudenteFinto(), car, "train", righe, ca, fx.WS, fx.WS, 2.0 ** 10, somme=somme)
        finally:
            del A.KLFusa.apply
            car.chiudi()
        c = {"micro": len(catture) == 4}
        sw = sa = 0.0
        for k, x in enumerate(catture):
            peso, lettere = x[6], x[8]
            r = attese[k]
            att = torch.zeros(T, dtype=torch.bool)
            if r is not None:
                att[r[0]] = True
            w_att = torch.where(att, torch.tensor(1.0 + lam, dtype=torch.float64), torch.ones(T, dtype=torch.float64))
            c[f"pesi_{k}"] = (peso[0] == "risposta" and torch.equal(peso[1], w_att) and torch.equal(peso[2], att))
            pos_att = torch.zeros(0, dtype=torch.long) if r is None else r[0]
            c[f"lettere_{k}"] = (torch.equal(lettere[0], pos_att) and torch.equal(lettere[1], torch.tensor(ID4))
                                 and torch.equal(lettere[2], torch.full((pos_att.numel(),), G * (1.0 + lam),
                                                                        dtype=torch.float64)))
            sw += float(peso[1].sum())
            sa += float(peso[1][peso[2]].sum())
        rias = A.riassunto_pesi(somme)
        quota = sa / sw
        c.update({"lambda_positivo": lam > 0, "quota_beta": abs(quota - B) <= 1e-12 * B,
                  "formula": abs(a * (1 + lam) / (4 * T + a * lam) - B) <= 1e-12,
                  "registro": (rias["risposte"] == a and rias["lettere_n"] == a
                               and abs(rias["quota_peso_risposte"] - B) <= 1e-12 and
                               abs(rias["peso_risposta"] - 1 - lam) <= 1e-12 and 0 < rias["quota_massa_risposte"] < 1)})
        det[f"passo_{t}"] = {"risposte": a, "lambda": lam, "quota": quota, "registro": rias, "falliti":
                             [k for k, v in c.items() if not v]}
        ok &= all(c.values())
    # gradiente fuso contro autograd float64
    g = torch.Generator().manual_seed(43)
    P = 2 * T
    hS = torch.randn(P, H, generator=g, dtype=torch.float64) * 1.3
    WS = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.5
    hT = torch.randn(P, H, generator=g, dtype=torch.float64) * 1.3
    WT = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.5
    pos = torch.tensor([5, 13, T + 21, T + 29])
    marc = torch.zeros(P, dtype=torch.bool)
    marc[pos] = True
    lam = A.lambda_risposta(B, P, pos.numel())
    w = torch.where(marc, torch.tensor(1.0 + lam, dtype=torch.float64), torch.ones(P, dtype=torch.float64))
    id4 = torch.tensor(ID4)
    coef = torch.full((pos.numel(),), G * (1 + lam), dtype=torch.float64)
    err = {}
    for gamma in (None, 0.3):
        uniforme = A.KLFusa.apply(hS, WS, hT, WT, 7) if gamma is None else A.KLFusa.apply(hS, WS, hT, WT, 7, gamma)[0]
        for nome, peso, lett, w_rif, pos_rif in (
                ("risposta_lettere", ("risposta", w, marc), (pos, id4, coef), w, pos),
                ("solo_lettere", None, (pos, id4, coef / (1 + lam)), torch.ones(P, dtype=torch.float64), pos),
                ("solo_risposta", ("risposta", w, marc), None, w, None)):
            x = hS.clone().requires_grad_(True)
            dg: dict = {}
            out = A.KLFusa.apply(x, WS, hT, WT, 7, gamma, peso, dg, lett)
            (out[0] * 3.0).backward()
            gf = x.grad.clone()
            y = hS.clone().requires_grad_(True)
            cr = coef if nome == "risposta_lettere" else coef / (1 + lam)
            r = _rif_pesato(y, WS, hT, WT, gamma, w_rif, pos_rif, id4, cr)
            (r * 3.0).backward()
            e = {"valore": abs(float(out[0].detach()) - float(r.detach())) / abs(float(r)),
                 "grad_rel": float((gf - y.grad).norm() / y.grad.norm()), "grad_max": float((gf - y.grad).abs().max()),
                 "uniforme_al_bit": torch.equal(out[1], uniforme)}
            err[f"{nome}_{gamma}"] = e
            ok &= e["valore"] <= 1e-10 and e["grad_rel"] <= 1e-10 and e["uniforme_al_bit"]
            if lett is not None:
                ok &= int(dg["n_lettere"]) == pos.numel()
    # micro-lotto senza risposte: pesi tutti 1 == uniforme AL BIT (valore e gradiente)
    x1, x2 = hS.clone().requires_grad_(True), hS.clone().requires_grad_(True)
    zero = torch.zeros(P, dtype=torch.bool)
    v1 = A.KLFusa.apply(x1, WS, hT, WT, 7, None, ("risposta", torch.ones(P, dtype=torch.float64), zero), {})
    v2 = A.KLFusa.apply(x2, WS, hT, WT, 7)
    (v1[0] * 5.0).backward()
    (v2 * 5.0).backward()
    senza = torch.equal(x1.grad, x2.grad) and torch.equal(v1[1], v2)
    # nel CICLO: voce "pesi" a ogni passo, quota dei pesi == beta
    d = DIR / "m3_ciclo"
    _pulisci(d)
    stato = A.allena(TA.StudenteFinto(), src, d, ca, fx.WS, fx.WS, log=muto)
    voci = [x["pesi"] for x in stato["storia"]]
    ciclo = (len(voci) == stato["passo"] and "pesi_passo0" in stato["g1"] and
             all(abs(v["quota_peso_risposte"] - B) <= 1e-12 and v["risposte"] > 0 and v["lettere_n"] == v["risposte"]
                 for v in voci))
    return registra("M3_pesi_risposta_e_lettere", ok and senza and ciclo, **det, gradiente=err,
                    senza_risposte_uniforme_al_bit=senza, ciclo=ciclo, voce_passo_2=voci[1] if len(voci) > 1 else None,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M4 ───────────────────────────────────────────
def prova_m4() -> bool:
    t0 = time.time()
    g = torch.Generator().manual_seed(44)
    P = 3 * T
    pos = torch.tensor([2, 17, 40, 70, 90])
    id4 = torch.tensor(ID4)
    esiti = {}
    ok = True
    for nome, dt in (("fp64", torch.float64), ("fp32_maestro_bf16", torch.float32)):
        W = torch.randn(V, H, generator=g, dtype=torch.float64).to(dt)
        if dt == torch.float32:
            hT = (torch.randn(P, H, generator=g) * 1.4).to(torch.bfloat16)
            hS = hT.float()
        else:
            hT = torch.randn(P, H, generator=g, dtype=torch.float64) * 1.4
            hS = hT.clone()
        k = A.kl_lettere(hS, W, hT, W, pos, id4)
        coef = torch.full((pos.numel(),), 7.5, dtype=torch.float64)
        x1, x2 = hS.clone().requires_grad_(True), hS.clone().requires_grad_(True)
        dg: dict = {}
        o1 = A.KLFusa.apply(x1, W, hT, W, 11, None, None, dg, (pos, id4, coef))
        o2 = A.KLFusa.apply(x2, W, hT, W, 11)
        o1[0].backward()
        o2.backward()
        e = {"kl4_zero": bool((k["kl4"] == 0).all()), "p_uguali": torch.equal(k["pS"], k["pT"]),
             "termine_zero": float(dg["kl4"]) == 0.0 and float(dg["kl4_pesata"]) == 0.0,
             "dh_al_bit": torch.equal(x1.grad, x2.grad), "kl_piena_zero": float(o2) == 0.0}
        esiti[nome] = e
        ok &= all(e.values())
    # e se lo studente NON e' il maestro, KL4 > 0 (la prova sopra non e' banale)
    W = torch.randn(V, H, generator=g, dtype=torch.float64)
    hT = torch.randn(P, H, generator=g, dtype=torch.float64)
    k2 = A.kl_lettere(hT + 0.3 * torch.randn(P, H, generator=g, dtype=torch.float64), W, hT, W, pos, id4)
    diverso = bool((k2["kl4"] > 0).all())
    # valuta_mc con studente = maestro (h del maestro in fp32 nella sorgente: stessi prodotti al bit)
    fx = Fixture(h32=True)
    src = fx.sorgente(train_mc=False)
    car = A.Caricatore(src, torch.device("cpu"))
    vm = A.valuta_mc(fx.maestro, car, src.righe_previste(A.LATO_HELD_MC), _ca(righe_per_micro_valuta=1), fx.WS, fx.WS)
    car.chiudi()
    chiavi = ("accuratezza", "p_C_se_giusta_non_C", "scelta_C_se_giusta_non_C", "margine_medio", "scelte")
    stesso = vm["kl4_media"] == 0.0 and all(vm[k] == vm["maestro"][k] for k in chiavi) and vm["n"] > 0
    return registra("M4_kl4_zero_con_studente_uguale_al_maestro", ok and diverso and stesso, esiti=esiti,
                    kl4_con_studente_diverso=float(k2["kl4"].mean()), valuta_mc=vm, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M5 ───────────────────────────────────────────
def prova_m5() -> bool:
    import e2e_pv_codici as PV
    t0 = time.time()
    fx = Fixture(n_train=40, n_mc=5, n_mc_held=3)
    src = fx.sorgente()
    WS = fx.WS
    ca1 = _ca(righe_per_passo=4, pv_codici=True, pv_modo="vbop", pv_rho_adattivo=True, pv_rho=0.5, pv_rho_max=4.0,
              pv_micro_curvatura=1, ema_theta=0.9, perdita="cakld", gamma=0.552, heldout_ogni=1,
              heldout_righe_rapide=2, heldout_pieno_ogni=3, ripresa_ogni=1, tieni_riprese=20)
    ca2 = dataclasses.replace(ca1, righe_mc=1, peso_token="risposta", gamma_lettere=1.0)
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    installa, PVF, ultimo = TA._pv_finto_vbop()
    PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)
    orig_rip, orig_gp = PV.StatoV.ripristina_bop, A.gradiente_passo
    riletto: dict = {}
    primo: dict = {}

    def ripristina_bop(self, ist):
        orig_rip(self, ist)
        riletto.clear()
        riletto.update(copy.deepcopy(self.istantanea_bop()))

    def spia(st, car, lato, righe, *a, **k):
        r = k.get("rumore")
        if "theta" not in primo and r is not None:
            primo.update({"theta": [p.detach().clone() for p in st.parametri()],
                          "opt": copy.deepcopy(r.opt.state_dict()), "righe": list(righe)})
        return orig_gp(st, car, lato, righe, *a, **k)

    def corsa(d, s, ca, **kw):
        stu = TA._StudenteFintoV()
        reg: list = []
        z = A.allena(stu, s, d, ca, WS, WS, log=reg.append, **kw)
        return stu, z, ultimo["pv"], reg
    det = {}
    try:
        PV.StatoV.ripristina_bop = ripristina_bop
        d1 = DIR / "m5_senza_mc"
        dA, dB, dC, dO = (DIR / f"m5_{n}" for n in ("filata", "ripresa", "altro_teacher", "spenta"))
        _pulisci(d1, dA, dB, dC, dO)
        _, z1, pv1, _ = corsa(d1, fx.base, ca1, ferma_a=3)
        ist_bop3 = copy.deepcopy(pv1.stato.istantanea_bop())
        rip3 = torch.load(d1 / "ripresa_000003.pt", map_location="cpu", weights_only=False)
        shutil.copytree(d1, dA)
        shutil.copytree(d1, dB)
        # (1) aggiunta del teacher mc alla ripresa: stato conservato al bit
        A.gradiente_passo = spia
        try:
            riletto.clear()
            stA, zA, pvA, regA = corsa(dA, src, ca2, riprendi=True, ferma_a=7)
        finally:
            A.gradiente_passo = orig_gp
        bop_ok = bool(riletto) and riletto["passi_bop"] == ist_bop3["passi_bop"] and all(
            torch.equal(riletto["stato"][f][n], ist_bop3["stato"][f][n]) for f in ist_bop3["stato"]
            for n in ist_bop3["stato"][f])
        chiavi = [A._chiave(k) for k in stA.chiavi()]
        theta_ok = all(torch.equal(a, rip3["theta"][c]) for a, c in zip(primo["theta"], chiavi))
        adam_ok = _uguali_opt(primo["opt"], rip3["opt"])
        r4 = torch.load(dA / "ripresa_000004.pt", map_location="cpu", weights_only=False)
        ema_ok = all(torch.equal(r4["ema"][c], rip3["ema"][c].clone().lerp_(r4["theta"][c], 1.0 - 0.9)) for c in chiavi)
        seg_att = [{"passo": 3, "cursore_base": 12, "cursore_mc": 0, "righe_mc": 1, "firma": 0}]
        c1 = {"theta_al_bit": theta_ok, "adam_al_bit": adam_ok, "vbop_riletto_al_bit": bop_ok, "ema_ricorrenza": ema_ok,
              "righe_passo_4": primo["righe"] == [12, 13, 14, OFF],
              "segmenti": zA["mc"]["segmenti"] == seg_att, "firma_base": zA["sorgente"] == z1["sorgente"],
              "passi_totali": zA["passi_totali"] == 3 + (40 - 12) // 3, "passo": zA["passo"] == 7,
              "log_accese": any("righe mc ACCESE al passo 3" in x for x in regA),
              "log_decisione": any("ripresa con campi cambiati" in x and "righe_mc" in x for x in regA),
              "log_heldout_mc": any("held-out mc alla ripresa" in x for x in regA),
              # la misura mc della ripresa DOPO i codici rigiocati dal passo V (installazione del PV)
              "heldout_mc_dopo_il_pv": min(i for i, x in enumerate(regA) if "fase 2 PV accesa" in x) <
                                       min(i for i, x in enumerate(regA) if "held-out mc alla ripresa" in x),
              "pesi_dopo": all("pesi" in x and x["pesi"]["risposte"] > 0 for x in zA["storia"][3:]),
              "senza_pesi_prima": all("pesi" not in x for x in zA["storia"][:3]),
              "pv_avanza": zA["pv"]["passi_v"] > z1["pv"]["passi_v"]}
        det["aggiunta"] = c1
        # (2) ripresa a meta' in modo mc == filata AL BIT
        _, zB5, _, _ = corsa(dB, src, ca2, riprendi=True, ferma_a=5)
        shutil.copytree(dB, dC)
        shutil.copytree(dB, dO)
        stB, zB, pvB, _ = corsa(dB, src, ca2, riprendi=True, ferma_a=7)
        fin_bop = pvA.stato.istantanea_bop(), pvB.stato.istantanea_bop()
        c2 = {"theta": all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri())),
              "adam": _uguali_opt(A.carica_ripresa(dA, "cpu")["opt"], A.carica_ripresa(dB, "cpu")["opt"]),
              "stato": TA._senza_tempi(zA) == TA._senza_tempi(zB),
              "vbop": fin_bop[0]["passi_bop"] == fin_bop[1]["passi_bop"] and all(
                  torch.equal(fin_bop[0]["stato"][f][n], fin_bop[1]["stato"][f][n]) for f in fin_bop[0]["stato"]
                  for n in fin_bop[0]["stato"][f]),
              "a_meta": zB5["passo"] == 5}
        det["ripresa_in_mc"] = c2
        # (3) teacher mc DIVERSO alla ripresa (passo 5): nuova firma, cursore mc da 0
        g = torch.Generator().manual_seed(99)
        altro = fx.righe_mc(g, 4, "mc2")
        src2 = fx.sorgente(mc=altro, nome="fixture_altro")
        _, zC, _, regC = corsa(dC, src2, ca2, riprendi=True, ferma_a=6)
        c3 = {"firme": len(zC["mc"]["firme"]) == 2,
              "segmento": zC["mc"]["segmenti"][-1] == {"passo": 5, "cursore_base": 18, "cursore_mc": 0, "righe_mc": 1,
                                                        "firma": 1},
              "righe": zC["storia"][-1]["righe"] == [18, OFF], "log": any("teacher mc NUOVO" in x for x in regC)}
        det["altro_teacher"] = c3
        # (4) spenta alla ripresa (passo 5, sorgente base, config senza mc): base dal cursore, niente pesi
        _, zO, _, _ = corsa(dO, fx.base, ca1, riprendi=True, ferma_a=6)
        c4 = {"segmento": zO["mc"]["segmenti"][-1] == {"passo": 5, "cursore_base": 18, "cursore_mc": 2, "righe_mc": 0,
                                                        "firma": 0},
              "righe": zO["storia"][-1]["righe"] == [18, 21], "senza_pesi": "pesi" not in zO["storia"][-1],
              "passi_totali": zO["passi_totali"] == 5 + (40 - 18) // 4}
        det["spenta"] = c4
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig
        PV.StatoV.ripristina_bop = orig_rip
        A.gradiente_passo = orig_gp
    ok = all(all(c.values()) for c in det.values())
    return registra("M5_ripresa_con_aggiunta_del_teacher_mc", ok, **det, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M6 ───────────────────────────────────────────
def _rif_mc(st, fx: Fixture, WS) -> dict:
    """Ricalcolo INDIPENDENTE delle misure mc sul held-out (riga per riga, senza `kl_lettere` ne' `valuta_mc`)."""
    zs, zt, gg = [], [], []
    for i, (ids, hT, _, _) in enumerate(fx.held):
        sel = fx.held_risp["pos"][:, 0] == i
        t = fx.held_risp["pos"][sel, 1]
        hS = st.nascosto(torch.from_numpy(ids.astype(np.int64))[None]).float()
        W4 = WS[torch.tensor(ID4)].float()
        zs.append((hS[t] @ W4.T).double())
        zt.append((hT.float()[t] @ W4.T).double())
        gg.append(fx.held_risp["giusta"][sel])
    zS, zT, g = torch.cat(zs), torch.cat(zt), torch.cat(gg)
    lS, lT = torch.log_softmax(zS, -1), torch.log_softmax(zT, -1)
    out = {}
    for nome, z, lp in (("s", zS, lS), ("t", zT, lT)):
        p = lp.exp()
        sc = z.argmax(-1)
        altri = torch.stack([torch.cat([z[j, :g[j]], z[j, g[j] + 1:]]).max() for j in range(len(g))])
        nc = g != 2
        out[nome] = {"accuratezza": float((sc == g).double().mean()), "p_C_se_giusta_non_C": float(p[nc, 2].mean()),
                     "scelta_C_se_giusta_non_C": float((sc[nc] == 2).double().mean()),
                     "margine_medio": float((z[torch.arange(len(g)), g] - altri).mean()),
                     "scelte": torch.bincount(sc, minlength=4).tolist()}
    out["kl4_media"] = float((lT.exp() * (lT - lS)).sum(-1).mean())
    out["n"] = int(g.numel())
    return out


def _confronta_mc(v: dict, r: dict) -> bool:
    ok = v["n"] == r["n"] and abs(v["kl4_media"] - r["kl4_media"]) <= 1e-5 * max(1.0, abs(r["kl4_media"]))
    for chi, rr in ((v, r["s"]), (v["maestro"], r["t"])):
        ok &= chi["scelte"] == rr["scelte"] and chi["accuratezza"] == rr["accuratezza"] and \
            chi["scelta_C_se_giusta_non_C"] == rr["scelta_C_se_giusta_non_C"]
        for k in ("p_C_se_giusta_non_C", "margine_medio"):
            ok &= abs(chi[k] - rr[k]) <= 1e-5 * max(1.0, abs(rr[k]))
    return ok


def prova_m6() -> bool:
    t0 = time.time()
    fx = Fixture(n_train=24)
    ca = _ca(righe_per_passo=4, heldout_ogni=1, heldout_righe_rapide=2, heldout_pieno_ogni=3, ema_theta=0.5,
             righe_per_micro_valuta=1)             # una riga per micro-lotto: gli stessi prodotti del ricalcolo
    dR, dS = DIR / "m6_con_heldout_mc", DIR / "m6_senza"
    _pulisci(dR, dS)
    stR, stS = TA.StudenteFinto(), TA.StudenteFinto()
    regR: list = []
    zR = A.allena(stR, fx.sorgente(train_mc=False), dR, ca, fx.WS, fx.WS, log=regR.append)
    zS = A.allena(stS, fx.base, dS, ca, fx.WS, fx.WS, log=muto)
    sR, n_mc = _senza_mc_voci(zR)
    sS, n0 = _senza_mc_voci(zS)
    fR, fS = TA._sha_cartella(dR), TA._sha_cartella(dS)
    ist = [k for k in fR if k.startswith("migliore") and not k.endswith("meta.json")]
    n_att = int(fx.held_risp["giusta"].numel())
    voci = zR["storia_heldout"] + zR["storia_heldout_rapido"]
    c = {"theta": all(torch.equal(a, b) for a, b in zip(stR.parametri(), stS.parametri())),
         "adam": _uguali_opt(A.carica_ripresa(dR, "cpu")["opt"], A.carica_ripresa(dS, "cpu")["opt"]),
         "stato_senza_voci_mc": sR == sS, "migliore": zR["migliore"] == zS["migliore"],
         "istantanee": set(fR) == set(fS) and len(ist) > 0 and all(fR[k] == fS[k] for k in ist),
         "ogni_misura": n_mc == len(voci) == 1 + zR["passo"] and n0 == 0 and all(x["mc"]["n"] == n_att for x in voci),
         "rapide_e_piene": len(zR["storia_heldout_rapido"]) > 0 and len(zR["storia_heldout"]) > 1,
         "log": sum("held-out mc al passo" in x for x in regR) == len(voci)}
    # ricalcolo indipendente: passo 0 (theta = 0) e ultima misura piena (theta finale)
    rif0 = _rif_mc(TA.StudenteFinto(), fx, fx.WS)
    rif_f = _rif_mc(stR, fx, fx.WS)
    ultima = zR["storia_heldout"][-1]
    c["passo0_ricalcolo"] = _confronta_mc(zR["storia_heldout"][0]["mc"], rif0)
    c["finale_ricalcolo"] = ultima["passo"] == zR["passo"] and _confronta_mc(ultima["mc"], rif_f)
    c["si_muove"] = ultima["mc"]["kl4_media"] != zR["storia_heldout"][0]["mc"]["kl4_media"]
    return registra("M6_heldout_mc_solo_registro", all(c.values()), **c, passo0=zR["storia_heldout"][0]["mc"],
                    ultima=ultima["mc"], secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M7 ───────────────────────────────────────────
def _scrivi_corpus(f: Path, righe: list) -> None:
    """Corpus nel formato di e2e_corpus_4096 (quello che `e2e_teacher.righe_lato` legge): righe [(dominio, ids)] in
    ORDINE d'uso, una sequenza per riga (cu = [0, T])."""
    nomi: list = []
    dom: dict = {}
    ordine = []
    for d, ids in righe:
        if d not in dom:
            nomi.append(d)
            dom[d] = []
        ordine.append([nomi.index(d), len(dom[d])])
        dom[d].append(np.asarray(ids, np.int32))
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"l_seq": T, "nomi_domini": nomi, "ordine": torch.tensor(ordine, dtype=torch.long),
                "domini": {d: torch.from_numpy(np.stack(v)) for d, v in dom.items()},
                "cu": {d: torch.tensor([0, T] * len(v), dtype=torch.int32) for d, v in dom.items()},
                "cu_ptr": {d: torch.arange(0, 2 * len(v) + 1, 2, dtype=torch.long) for d, v in dom.items()}}, f)


def _scrivi_cache(dir_c: Path, lati: dict, modello_sha: str = "a" * 64, strumento_sha: str = "b" * 64) -> Path:
    """Cache del teacher nel formato di e2e_teacher (piano.json; <lato>/blocco_0000/{blocco.json, ids.i32, h/*.bf16}
    con intestazione 4 x int64 (H, T, 1, 1) + bit bf16 [T, H]). lati = {lato: (file del corpus, [(dom, ids, h bf16)])}."""
    if dir_c.exists():
        shutil.rmtree(dir_c)
    blocchi, corpus = [], {}
    for lato, (f, rr) in lati.items():
        corpus[lato] = {"file": str(f), "sha256": A.sha256_file(f), "L": T, "n_righe": len(rr), "segmenti_totali": len(rr)}
        nome = f"{lato}/blocco_0000"
        conta: dict = {}
        voci, sig = [], []
        d = dir_c / nome
        (d / "h").mkdir(parents=True)
        np.concatenate([np.asarray(ids, np.int32) for _, ids, _ in rr]).tofile(d / "ids.i32")
        for i, (dom, ids, h) in enumerate(rr):
            r = conta.get(dom, 0)
            conta[dom] = r + 1
            voci.append([dom, r, i])
            fh = d / "h" / f"r{i:04d}.bf16"
            with open(fh, "wb") as fo:
                np.array([H, T, 1, 1], np.int64).tofile(fo)
                h.reshape(T, H).contiguous().view(torch.int16).numpy().tofile(fo)
            sig.append({"indice": i, "dominio": dom, "riga": r, "file_h": f"h/r{i:04d}.bf16",
                        "sha256_h": A.sha256_file(fh), "segmenti_corpus": [0]})
        (d / "blocco.json").write_text(json.dumps({"nome": nome, "lato": lato, "L": T, "righe": sig}))
        blocchi.append({"nome": nome, "lato": lato, "L": T, "da": 0, "a": len(rr), "righe": voci})
    piano = {"versione": "prova", "corsa": dir_c.name, "modo": "riga", "salva_tipo": "bf16", "ub": 1, "variante": "prova",
             "lati": list(lati), "modello": {"percorso": "finto", "byte": 0, "sha256": modello_sha, "n_embd": H},
             "strumento": {"percorso": "finto", "sha256": strumento_sha}, "corpus": corpus, "blocchi": blocchi}
    (dir_c / "piano.json").write_text(json.dumps(piano))
    return dir_c


def _atteso_errore(fn, tipo=ValueError) -> bool:
    try:
        fn()
    except tipo:
        return True
    except Exception:
        return False
    return False


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


def prova_m7() -> bool:
    t0 = time.time()
    fx = Fixture(n_train=24, n_mc=5, n_mc_held=3)
    rad = DIR / "m7_disco"
    _pulisci(rad)
    cb, cm = rad / "corpus_base", rad / "mc_corpus"
    rb = fx.righe_base
    _scrivi_corpus(cb / "e2e_corpus_4096_prova.pt", [(d, ids) for ids, _, d, _ in rb["train"]])
    _scrivi_corpus(cb / "e2e_heldout_4096_prova.pt", [(d, ids) for ids, _, d, _ in rb["heldout"]])
    _scrivi_corpus(cm / "e2e_corpus_4096_prova.pt", [(d, ids) for ids, _, d, _ in fx.mc])
    torch.save(fx.mc_risp, cm / "risposte.pt")
    _scrivi_corpus(cm / "heldout" / "e2e_corpus_4096_prova.pt", [(d, ids) for ids, _, d, _ in fx.held])
    torch.save(fx.held_risp, cm / "heldout" / "risposte.pt")
    kb = _scrivi_cache(rad / "teacher_base", {
        "train": (cb / "e2e_corpus_4096_prova.pt", [(d, ids, h) for ids, h, d, _ in rb["train"]]),
        "heldout": (cb / "e2e_heldout_4096_prova.pt", [(d, ids, h) for ids, h, d, _ in rb["heldout"]])})
    km = _scrivi_cache(rad / "teacher_mc", {"train": (cm / "e2e_corpus_4096_prova.pt",
                                                      [(d, ids, h) for ids, h, d, _ in fx.mc])})
    kh = _scrivi_cache(rad / "teacher_mc_heldout", {"train": (cm / "heldout" / "e2e_corpus_4096_prova.pt",
                                                              [(d, ids, h) for ids, h, d, _ in fx.held])})
    reg: list = []
    base_d = A.SorgenteCache(kb)
    src_d = A.apri_sorgente_mc(base_d, km, kh, reg.append)
    src_m = fx.sorgente()
    c = {"firma": (src_d.firma_mc()["train"]["risposte_n"] == int(fx.mc_risp["giusta"].numel())
                   and src_d.firma_mc()["heldout"]["risposte_n"] == int(fx.held_risp["giusta"].numel())
                   and src_d.firma_mc()["teacher_mc_heldout"] == str(kh)),
         "risposte_uguali": all(torch.equal(src_d.risposte("train", i)[0], src_m.risposte("train", i)[0]) and
                                torch.equal(src_d.risposte("train", i)[1], src_m.risposte("train", i)[1])
                                for i in src_m.righe_mc()),
         "h_uguali": all(torch.equal(src_d.h("train", i).reshape(T, H), src_m.h("train", i).reshape(T, H))
                         for i in src_m.righe_mc()[:3])}
    ca = _ca(righe_per_passo=4, righe_mc=1, peso_token="risposta", gamma_lettere=0.7, heldout_ogni=1,
             heldout_righe_rapide=2, heldout_pieno_ogni=3)
    dD, dM = rad / "corsa_disco", rad / "corsa_memoria"
    stD, stM = TA.StudenteFinto(), TA.StudenteFinto()
    zD = A.allena(stD, src_d, dD, ca, fx.WS, fx.WS, log=muto)
    zM = A.allena(stM, src_m, dM, ca, fx.WS, fx.WS, log=muto)
    a_, b_ = TA._senza_tempi(zD), TA._senza_tempi(zM)
    for s_ in (a_, b_):
        s_.pop("sorgente")
        s_["mc"].pop("firme")
    c.update({"ciclo_theta": all(torch.equal(a, b) for a, b in zip(stD.parametri(), stM.parametri())),
              "ciclo_stato": a_ == b_, "ciclo_passi": zD["passo"] > 3,
              "firma_nello_stato": zD["mc"]["firme"][0]["train"]["teacher"] == str(km)})
    # held-out mc come corsa del teacher DENTRO --teacher-mc (DIR/heldout/piano.json): trovato senza --teacher-mc-heldout
    kmh = _scrivi_cache(rad / "teacher_mc_con_heldout", {"train": (cm / "e2e_corpus_4096_prova.pt",
                                                                   [(d, ids, h) for ids, h, d, _ in fx.mc])})
    _scrivi_cache(kmh / "heldout", {"train": (cm / "heldout" / "e2e_corpus_4096_prova.pt",
                                              [(d, ids, h) for ids, h, d, _ in fx.held])})
    src_d2 = A.apri_sorgente_mc(base_d, kmh, None, muto)
    c["heldout_dentro_teacher_mc"] = (src_d2.firma_mc()["teacher_mc_heldout"] == str(kmh / "heldout") and
                                      src_d2.righe_previste(A.LATO_HELD_MC) == src_m.righe_previste(A.LATO_HELD_MC))
    # risposte.pt difettosi -> ValueError con la ragione
    s_mem = A.SorgenteMemoria({"train": fx.mc}, {})
    ids_di = lambda i: s_mem.ids("train", i)  # noqa: E731
    righe = s_mem.righe_previste("train")
    difetti = {}
    base_r = fx.mc_risp

    def con(**kw):
        r = {k: (v.clone() if isinstance(v, torch.Tensor) else list(v)) for k, v in base_r.items()}
        for k, v in kw.items():
            if v is None:
                r.pop(k)
            else:
                r[k] = v
        p = rad / "risposte_difettoso.pt"
        torch.save(r, p)
        return lambda: A.leggi_risposte(p, ids_di, righe, T)
    g2 = base_r["giusta"].clone()
    g2[0] = (g2[0] + 1) % 4
    pr_ = base_r["pos"].clone()
    pr_[0, 0] = 99
    pt_ = base_r["pos"].clone()
    pt_[0, 1] = T - 1
    pd_ = base_r["pos"].clone()
    pd_[1] = pd_[0]
    g4 = base_r["giusta"].clone()
    g4[0] = 4
    difetti = {"lettera_non_in_t+1": con(giusta=g2), "riga_fuori": con(pos=pr_), "t_fuori": con(pos=pt_),
               "doppioni": con(pos=pd_), "lettere_ripetute": con(id_lettere=torch.tensor([357, 417, 351, 357])),
               "giusta_fuori": con(giusta=g4), "manca_fonte": con(fonte=None),
               "forme": con(giusta=base_r["giusta"][:-1])}
    rifiuti = {k: _atteso_errore(f) for k, f in difetti.items()}
    # modello del teacher mc diverso; held-out uguale a righe mc di addestramento
    km2 = _scrivi_cache(rad / "teacher_mc_altro_modello", {"train": (cm / "e2e_corpus_4096_prova.pt",
                                                                     [(d, ids, h) for ids, h, d, _ in fx.mc])},
                        modello_sha="c" * 64)
    rifiuti["modello_diverso"] = _atteso_errore(lambda: A.apri_sorgente_mc(base_d, km2, None, muto), RuntimeError)
    s_tr = A.SorgenteMemoria({"train": fx.mc}, {})
    r_tr = A.leggi_risposte(cm / "risposte.pt", lambda i: s_tr.ids("train", i), s_tr.righe_previste("train"), T)
    rifiuti["heldout_non_disgiunto"] = _atteso_errore(lambda: A.SorgenteConMC(fx.base, (s_tr, r_tr),
                                                                              (s_tr, "train", r_tr)))
    # configurazioni rifiutate
    for nome, kw in {"risposta_senza_righe_mc": {"peso_token": "risposta"},
                     "lettere_senza_righe_mc": {"gamma_lettere": 1.0},
                     "beta_senza_risposta": {"righe_mc": 1, "righe_per_passo": 4, "beta_risposta": 0.2},
                     "beta_fuori": {"righe_mc": 1, "righe_per_passo": 4, "peso_token": "risposta", "beta_risposta": 1.0},
                     "righe_mc_troppe": {"righe_mc": 4, "righe_per_passo": 4},
                     "gamma_negativo": {"righe_mc": 1, "righe_per_passo": 4, "gamma_lettere": -0.1},
                     "alfa_con_risposta": {"righe_mc": 1, "righe_per_passo": 4, "peso_token": "risposta",
                                           "peso_alfa": 2.0}}.items():
        rifiuti[f"config_{nome}"] = _atteso_errore(lambda kw=kw: _ca(**kw))
    rifiuti["righe_mc_senza_sorgente_mc"] = _atteso_errore(
        lambda: A.allena(TA.StudenteFinto(), fx.base, rad / "senza_mc", _ca(righe_per_passo=4, righe_mc=1), fx.WS,
                         fx.WS, log=muto))
    # CLI vera
    c0 = _cli([])
    c1 = _cli(["--teacher-mc", "/x", "--teacher-mc-heldout", "/y", "--righe-mc", "8", "--peso-token", "risposta",
               "--beta-risposta", "0.2", "--gamma-lettere", "1.5"])
    cli = ((c0.righe_mc, c0.beta_risposta, c0.gamma_lettere, c0.peso_token) == (0, 0.1, 0.0, "uniforme")
           and (c1.righe_mc, c1.beta_risposta, c1.gamma_lettere, c1.peso_token) == (8, 0.2, 1.5, "risposta")
           and {"righe_mc", "beta_risposta", "gamma_lettere"} <= A._CAMPI_DI_DECISIONE)
    ok = all(c.values()) and all(rifiuti.values()) and cli
    return registra("M7_disco_verifiche_e_rifiuti", ok, **c, rifiuti=rifiuti, cli=cli, log_apertura=reg[-1:],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M8 ───────────────────────────────────────────
def _dy(st, src, righe, ca, WS, K) -> dict:
    """Un passo di gradiente (theta.grad azzerato prima) con la diagnostica dell'indietro: max|dY|, dY azzerati in fp16,
    trabocco, theta.grad finito, e il gradiente (copia) per i confronti."""
    A._azzera(st)
    car = A.Caricatore(src, st.device)
    st.diagnostica_indietro(True)
    try:
        A.gradiente_passo(st, car, "train", righe, ca, WS, WS, K)
        dy = A.aggrega_dy(st.diagnostica_indietro(False))
        ctl = st.controlla(tollera_trabocco=True)
    finally:
        car.chiudi()
    g = A._copia_grad(st)
    return {"max_abs_dy": max(v["max_abs_dy"] for v in dy.values()),
            "quota_azzerati": max(v["quota_azzerati"] for v in dy.values()), "trabocco": bool(ctl["trabocco_fp16"]),
            "grad_finito": all(bool(torch.isfinite(x).all()) for x in g), "g": g}


def prova_m8() -> bool:
    """(Solo se chiesta: legge il GGUF VERO di Thor-T, strato 0, e righe vere del corpus held-out, come W5.) Pesi-risposta
    e lettere sulla scala K della G1 UNIFORME (il caso della ripresa di EB2, che ha fatto G1 senza pesi): passo di 4 righe
    da 128 token (3 base + 1 mc con 3 risposte), beta 0,1 (peso ~19) e 0,47 (peso ~150, quello atteso in EB2 con 8 righe
    mc da ~12 risposte su 131.072 token). Ricetta B (indietro fp16): max|dY| rispetto all'uniforme, trabocco, dY azzerati.
    Ricetta B_ind32 (indietro fp32, quella di EB2): gradiente finito, e contro l'indietro fp16 senza trabocco."""
    t0 = time.time()
    st0 = TA.studente((0,))
    WS = TA.testa_ristretta(st0)
    th = TA.theta_stella(st0)
    base = TA.sorgente(st0, 8, 4, th)
    T8 = TA.T_RIGA

    def righe_con_lettere(da: int, n: int, seme: int):
        g = torch.Generator().manual_seed(seme)
        rr, pos, giu = [], [], []
        for i, (dom, ids) in enumerate(TA.righe_vere(n, da)):
            ids = ids.copy()
            for t in (20, 60, 100):
                gg = int(torch.randint(0, 4, (1,), generator=g))
                ids[t + 1] = ID4[gg]
                pos.append((i, t))
                giu.append(gg)
            rr.append((dom, ids))
        hh = TA.maestro_sintetico(st0, rr, th)
        righe = [(ids, h, "mc", None) for (_, ids), h in zip(rr, hh)]
        risp = {"pos": torch.tensor(pos), "giusta": torch.tensor(giu), "id_lettere": torch.tensor(ID4),
                "fonte": [f"m8_{da}_{k}" for k in range(len(giu))]}
        return righe, risp
    mc_r, mc_p = righe_con_lettere(100, 2, 81)
    he_r, he_p = righe_con_lettere(150, 2, 82)
    d = DIR / "m8"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(mc_p, d / "risposte_train.pt")
    torch.save(he_p, d / "risposte_held.pt")
    s_mc, s_he = A.SorgenteMemoria({"train": mc_r}, {}), A.SorgenteMemoria({"train": he_r}, {})
    r_mc = A.leggi_risposte(d / "risposte_train.pt", lambda i: s_mc.ids("train", i), [0, 1], T8)
    r_he = A.leggi_risposte(d / "risposte_held.pt", lambda i: s_he.ids("train", i), [0, 1], T8)
    src = A.SorgenteConMC(base, (s_mc, r_mc), (s_he, "train", r_he), {"teacher_mc": "m8"})
    del st0
    righe = A.intercala([0, 1, 2], [OFF])
    esiti, ok = {}, True
    st = TA.studente((0,))
    car = A.Caricatore(src, st.device)
    g1 = A.cancello_g1(st, car, "train", righe, _ca(righe_per_passo=4, righe_mc=1), WS, WS, muto)
    car.chiudi()
    K = g1["K"]
    esiti["g1_uniforme"] = {"K": K, "escalation": g1["escalation"], "eps_g1": g1["eps_g1"]}
    uni = _dy(st, src, righe, _ca(righe_per_passo=4, righe_mc=1), WS, K)
    esiti["uniforme_fp16"] = {**{k: v for k, v in uni.items() if k != "g"}, "margine_trabocco": 65504.0 / uni["max_abs_dy"]}
    for beta in (0.1, 0.47):
        ca = _ca(righe_per_passo=4, righe_mc=1, peso_token="risposta", beta_risposta=beta, gamma_lettere=1.0)
        lam = A.lambda_risposta(beta, 4 * T8, 3)
        r16 = _dy(st, src, righe, ca, WS, K)
        e = {k: v for k, v in r16.items() if k != "g"}
        e.update({"peso": 1 + lam, "max_dy_su_uniforme": r16["max_abs_dy"] / uni["max_abs_dy"],
                  "margine_trabocco": 65504.0 / r16["max_abs_dy"] if r16["max_abs_dy"] > 0 else None})  # dY gia' x K
        esiti[f"fp16_beta{beta}"] = e
        with A.ricetta_temporanea(st, A.ricetta_indietro_fp32(st.ric)):
            r32 = _dy(st, src, righe, ca, WS, K)
        e32 = {"trabocco": r32["trabocco"], "grad_finito": r32["grad_finito"]}
        if not r16["trabocco"] and r16["grad_finito"]:
            vive = rotti = 0
            for (L, n), a, b in zip(st.chiavi(), r16["g"], r32["g"]):
                v = (st.esperti(L).fam[n].d0 != 0) & (b != 0)
                vive += int(v.sum())
                rotti += int((v & ((a == 0) | (torch.sign(a) != torch.sign(b)))).sum())
            e32["quota_rotti_fp16_contro_fp32"] = rotti / max(vive, 1)
        esiti[f"fp32_beta{beta}"] = e32
        ok &= r32["grad_finito"] and not r32["trabocco"]
    car = A.Caricatore(src, st.device)
    vm = A.valuta_mc(st, car, src.righe_previste(A.LATO_HELD_MC), _ca(), WS, WS)
    car.chiudi()
    ok &= vm["n"] == 6 and math.isfinite(vm["kl4_media"])
    return registra("M8_strato0_vero_scala_K_coi_pesi_risposta", ok, **esiti, valuta_mc=vm,
                    secondi=round(time.time() - t0, 1))


PROVE = {"M1": prova_m1, "M2": prova_m2, "M3": prova_m3, "M4": prova_m4, "M5": prova_m5, "M6": prova_m6,
         "M7": prova_m7, "M8": prova_m8}
LEGGERE = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]      # senza argomenti: M8 (GGUF vero) solo se chiesta


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    TA.DIR = DIR                              # _modulo_prima scrive qui la copia del codice di partenza
    scelte = [a.upper() for a in sys.argv[1:]] or LEGGERE
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
    (DIR / f"esito_test_mc{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
