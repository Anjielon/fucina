#!/usr/bin/env python3
"""Prove del ramo newton-theta (27/9/2026): `--ottimizzatore-theta sophia` (e2e_sophia.py), la curvatura delle scale
per gruppo nel passo V (e2e_pv_codici: BufferV.HT, `_k_ht`, `ht_torch`) e il suo innesto in `e2e_allena`. SOLO CPU
(la GPU e' di EB2): il kernel gira nell'interprete di Triton; N6 lo COMPILA per gfx1151 senza eseguirlo.

  N1  DEFAULT identico AL BIT al master f1fe3cb (studente finto): ciclo `allena` senza PV e con PV finto (V-Bop, come
      A29): theta, Adam, stato (tolti i campi nuovi ai default e il rumore, solo registro), chiavi della ripresa; il
      default crea lo stesso torch.optim.AdamW; StatoV senza curv_theta non ha HT e ha gli stessi byte.
  N2  curvatura delle scale: `_k_ht` (interprete) == `ht_torch` (u materializzato) == formula esplicita in float64
      HT[e,r,g] = sum_s dY^2 (t_{r,g} . X_{s,g})^2, famiglie TQ1_0 sintetica e VERE (R1024 strato 0 gate, TQ1_0 strato 3
      down, righe con d0 = 0), ingressi fp32 e fp16; nel modo 3 H e dL/dd restano AL BIT quelli senza HT; StatoV azzera
      HT e il conteggio a ogni passo (anche dentro una finestra di accumulo); byte = 21 B/gruppo con HT.
  N3  problema quadratico finto (4096 coordinate, curvature log-uniformi 1e-2..1e2, gradiente esatto): Sophia con la h
      vera arriva a L <= 1e-3 L0 e a 1e-6 in MENO passi di AdamW a ogni lr in {1e-3, 3e-3, 1e-2} (allo stesso lr 1e-2
      AdamW si ferma sopra 1e-6); un passo == formula a mano (m, h^ corretta, clip); h assente -> lr segno(m).
  N4  ciclo `allena` con studente finto e PV finto che riempie HT: 3 passi + ripresa + 3 == 6 filati AL BIT (theta,
      stato di Sophia: exp_avg, hess, n_hess, step; storia con "sophia"); curvatura aggiornata a ogni passo V;
      passaggio adamw -> sophia alla ripresa (m == exp_avg di Adam salvato, n_hess 0, primo passo = lr segno(m)) e
      sophia -> adamw (Adam da zero per theta); sophia senza pv_codici rifiutata; CLI; campi di decisione.
  N5  studente VERO minuscolo (GGUF sintetico di e2e_test_f32, 8 esperti TQ1_0) con PV vero (motore torch, curvatura
      campionata, HT da `ht_torch`) + Sophia + F32 (AdamW): 3 + ripresa + 3 == 6 AL BIT (theta, F32, Sophia, Adam F32),
      HT > 0 sui gruppi vivi, theta si muove, gruppi AdamW invariati di forma.
  N6  (in piu') `_k_ht` compilato per gfx1151 coi BLOCCHI_HT: 0 VGPR riversati (R1024 e TQ1_0, fp16 e fp32).

Uso:
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TRITON_INTERPRET=1 nice -n 15 \\
        @@ROOT:venv@@/bin/python3 e2e_test_sophia.py [N1 N2 ...]
    esito: $E2E_ALLENA_PROVE/esito_test_sophia[_<prove>].json
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_sophia as SO  # noqa: E402
import e2e_studente as S  # noqa: E402
import e2e_test_allena as TA  # noqa: E402

DIR = Path(os.getenv("E2E_ALLENA_PROVE", "@@ROOT:lavoro@@/e2e/prove_allena"))
BASE_MASTER = "f1fe3cb"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:1800]}", flush=True)
    return bool(ok)


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


# ─────────────────────────────────────────── N1 ───────────────────────────────────────────
def _confronta(zn: dict, zp: dict) -> tuple:
    a_, b_ = TA._senza_tempi(zn), TA._senza_tempi(zp)
    nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    TA._togli_rumore(a_)
    TA._togli_rumore(b_)
    return a_ == b_, nuovi


def prova_n1() -> bool:
    t0 = time.time()
    prima = TA._modulo_prima(BASE_MASTER)
    src, WS = TA.sorgente_finta(n_train=24)
    es = {}
    # (a) senza PV
    ca = TA._ca_finta()
    for nome, mod in (("nuovo", A), ("master", prima)):
        d = DIR / f"n1_{nome}"
        shutil.rmtree(d, ignore_errors=True)
        st = TA.StudenteFinto()
        z = mod.allena(st, src, d, ca if mod is A else TA._config_per(mod, ca), WS, WS, log=muto)
        es[nome] = (st, z, mod.carica_ripresa(d, "cpu"))
    (sn, zn, rn), (sp, zp, rp) = es["nuovo"], es["master"]
    th_a = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
    so, sr = rn["opt"]["state"], rp["opt"]["state"]
    adam_a = so.keys() == sr.keys() and all(torch.equal(so[i][k], sr[i][k]) for i in so
                                            for k in ("exp_avg", "exp_avg_sq", "step")) \
        and rn["opt"]["param_groups"] == rp["opt"]["param_groups"]
    stato_a, nuovi_a = _confronta(zn, zp)
    chiavi_a = set(rn) == set(rp) and set(zn) == set(zp)
    # (b) con PV finto, V-Bop + curvatura ogni 2 + rho adattivo (il percorso della corsa vera)
    ca_v = TA._ca_finta(righe_per_passo=4, pv_codici=True, pv_modo="vbop", pv_curvatura_ogni=2, pv_rho_adattivo=True,
                        pv_rho=0.5, pv_rho_max=4.0, pv_micro_curvatura=1)
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    installa, PVF, ultimo = TA._pv_finto_vbop()
    PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)
    try:
        esv = {}
        for nome, mod in (("nuovo", A), ("master", prima)):
            d = DIR / f"n1v_{nome}"
            shutil.rmtree(d, ignore_errors=True)
            st = TA._StudenteFintoV()
            z = mod.allena(st, src, d, ca_v if mod is A else TA._config_per(mod, ca_v), WS, WS, log=muto)
            esv[nome] = (st, z, mod.carica_ripresa(d, "cpu"), ultimo["pv"])
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig
    (vn, wn, qn, pn), (vp, wp, qp, pp) = esv["nuovo"], esv["master"]
    th_b = all(torch.equal(a, b) for a, b in zip(vn.parametri(), vp.parametri()))
    so, sr = qn["opt"]["state"], qp["opt"]["state"]
    adam_b = so.keys() == sr.keys() and all(torch.equal(so[i][k], sr[i][k]) for i in so
                                            for k in ("exp_avg", "exp_avg_sq", "step"))
    stato_b, nuovi_b = _confronta(wn, wp)
    ht_assente = all(b.HT is None for b in pn.stato.buffer_.values()) and not pn.stato.curv_theta
    byte_uguali = pn.stato.byte() == pp.stato.byte()
    tipo = type(A.nuovo_ottimizzatore(TA.StudenteFinto(), ca, 1e-8)) is torch.optim.AdamW
    ok = (th_a and adam_a and stato_a and chiavi_a and nuovi_a == TA._CFG_INTEGRATA_DEFAULT and th_b and adam_b and stato_b
          and nuovi_b == TA._CFG_INTEGRATA_DEFAULT and ht_assente and byte_uguali and tipo
          and len([x for x in wn["storia"] if x.get("pv")]) >= 4 and "sophia" not in json.dumps(wn["storia"]))
    return registra("N1_default_al_bit_con_master", ok, base=BASE_MASTER, senza_pv={"theta": th_a, "adam": adam_a,
                    "stato": stato_a, "chiavi": chiavi_a, "config_nuovi": nuovi_a},
                    con_pv_vbop={"theta": th_b, "adam": adam_b, "stato": stato_b, "config_nuovi": nuovi_b},
                    ht_assente=ht_assente, byte_statov_uguali=byte_uguali, adamw_di_torch=tipo,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── N2 ───────────────────────────────────────────
def _ht_esplicito(gy, x, fam, piano) -> torch.Tensor:
    """HT in float64 con un decoder dei trit INDIPENDENTE (e2e_campi_scala via trit_esperto e poi un giro numpy)."""
    out = torch.zeros(fam.E, fam.R, fam.NG, dtype=torch.float64)
    o = piano.offs.tolist()
    for e in range(fam.E):
        if o[e + 1] <= o[e]:
            continue
        t = np.asarray(S.trit_esperto(fam, e).numpy(), dtype=np.float64).reshape(fam.R, fam.NG, fam.gs)
        xe = x[o[e]:o[e + 1]].double().numpy().reshape(-1, fam.NG, fam.gs)
        g2 = gy[o[e]:o[e + 1]].double().numpy() ** 2
        for g in range(fam.NG):
            u = xe[:, g, :] @ t[:, g, :].T                     # (s, R)
            out[e, :, g] = torch.from_numpy((g2 * u * u).sum(0))
    return out


def prova_n2() -> bool:
    import e2e_test_pv_codici as TP
    torch.manual_seed(21)
    casi = [("tq1_sintetica", lambda: TP.famiglia_sintetica(E=3, R=48, C=512, seme=5))]
    tp = True
    try:
        casi += [("R1024_L0_gate_vera", lambda: TP.famiglia(0, "gate", esperti=slice(0, 3),
                                                             righe=TP._finestra_con_morti(0, "gate", slice(0, 3), 40))),
                 ("TQ1_0_L3_down_vera", lambda: TP.famiglia(3, "down", esperti=slice(10, 12), righe=slice(0, 40)))]
    except Exception:                                              # noqa: BLE001  (GGUF non leggibile: solo sintetica)
        tp = False
    dati, ok_tot = {}, True
    for nome, crea in casi:
        fam = crea()
        for dt in (torch.float32, torch.float16):
            E = fam.E
            b_k, b_t, b_0, b_64 = (PV.BufferV.crea(fam, curv_theta=True), PV.BufferV.crea(fam, curv_theta=True),
                                   PV.BufferV.crea(fam), PV.BufferV.crea(fam, curv_theta=True))
            atteso = torch.zeros(E, fam.R, fam.NG, dtype=torch.float64)
            dd_uguali = h_uguale = True
            for k in range(3):
                idx = torch.randint(0, E, (40, 2))
                if k == 1:
                    idx = torch.randint(1, E, (40, 2))              # esperto 0 senza token
                piano = S.Piano(idx, E)
                x = torch.randn(piano.S, fam.C).to(dt)
                gy = (torch.randn(piano.S, fam.R) * 1e-3).to(dt)
                modo = 1 if k == 0 else 3
                dk = PV.dd_v_triton(gy, x, fam, piano, PV.BLOCCHI_V, b_k, modo)
                d0 = PV.dd_v_triton(gy, x, fam, piano, PV.BLOCCHI_V, b_0, modo)
                PV.dd_v_torch(gy, x, fam, piano, torch.float32, b_t, modo)
                PV.dd_v_torch(gy.double(), x.double(), fam, piano, torch.float64, b_64, modo)
                dd_uguali &= torch.equal(dk, d0)
                if modo == 3:
                    atteso += _ht_esplicito(gy, x, fam, piano)
            h_uguale = torch.equal(b_k.H, b_0.H) and torch.equal(b_k.prop, b_0.prop) and torch.equal(b_k.cand, b_0.cand)
            err = {"kernel_contro_torch": rel(b_k.HT, b_t.HT), "kernel_contro_fp64": rel(b_k.HT.double(), atteso),
                   "torch64_contro_esplicito": rel(b_64.HT.double(), atteso)}
            nonzero = int((atteso > 0).sum())
            ok = (max(err.values()) < 1e-5 and dd_uguali and h_uguale and nonzero > 0
                  and bool((b_k.HT >= 0).all()) and float(b_0.byte()) / b_0.prop.numel() == 17
                  and float(b_k.byte()) / b_k.prop.numel() == 21)
            ok_tot &= ok
            dati[f"{nome}_{str(dt)[6:]}"] = {"ok": ok, "tipo": fam.tipo, "gs": fam.gs, "NG": fam.NG, "E": E,
                                             "gruppi_con_HT": nonzero, "errori": err, "dd_al_bit": dd_uguali,
                                             "H_prop_cand_al_bit": h_uguale}
            print(f"  N2 {nome} {dt}: {dati[f'{nome}_{str(dt)[6:]}']}", flush=True)
    # StatoV: HT e conteggio azzerati a ogni passo, anche dentro una finestra di accumulo; senza curv_theta niente HT
    fam = TP.famiglia_sintetica(E=2, R=16, C=256, seme=2)
    sv = PV.StatoV(2, passi_accumulo=2, curv_theta=True)
    sv.prepara([fam])
    b = sv.buffer(fam)
    sv.inizia_passo()
    sv.microlotto(0)
    sv.n_propose = 1
    sv.curvatura()
    b.HT.fill_(3.0)
    b.S.fill_(2.0)
    n1 = sv.n_curvatura_passo
    sv.inizia_passo()                                               # secondo passo della STESSA finestra
    azzera = bool((b.HT == 0).all()) and bool((b.S == 2.0).all()) and sv.n_curvatura_passo == 0 and n1 == 1
    sv0 = PV.StatoV(2)
    sv0.prepara([fam])
    senza = sv0.buffer(fam).HT is None and sv0.byte() == 17 * fam.E * fam.R * fam.NG
    ok = ok_tot and azzera and senza
    return registra("N2_curvatura_delle_scale", ok, casi=dati, famiglie_vere=tp, statov_azzera=azzera,
                    statov_senza_ht=senza)


# ─────────────────────────────────────────── N3 ───────────────────────────────────────────
def _quadratica(tipo: str, lr: float, gamma: float = 0.1, passi: int = 400) -> list:
    n = 4096
    g = torch.Generator().manual_seed(3)
    h = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(math.log(1e-2), math.log(1e2), generator=g))
    ts = torch.randn(n, generator=g, dtype=torch.float64) * 0.05
    th = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    if tipo == "adamw":
        opt = torch.optim.AdamW([th], lr=lr, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    else:
        opt = SO.SophiaTheta([th], lr=lr, beta1=0.9, gamma=gamma, beta_h=0.99, eps=1e-15)
    L = lambda x: float(0.5 * (h * (x - ts) ** 2).sum())  # noqa: E731
    L0, st = L(th.detach()), []
    for _ in range(passi):
        opt.zero_grad(set_to_none=True)
        th.grad = h * (th.detach() - ts)
        opt.step()
        if tipo == "sophia":
            opt.aggiorna_curvatura(th, h)                       # curvatura misurata dopo il passo, come nel ciclo
        st.append(L(th.detach()) / L0)
    return st


def _primo(st: list, soglia: float):
    return next((i + 1 for i, x in enumerate(st) if x <= soglia), None)


def prova_n3() -> bool:
    t0 = time.time()
    adam = {lr: _quadratica("adamw", lr) for lr in (1e-3, 3e-3, 1e-2)}
    so = _quadratica("sophia", 1e-2, 0.1)
    p_adam = {lr: [_primo(s, x) for x in (1e-3, 1e-6)] for lr, s in adam.items()}
    p_so = [_primo(so, x) for x in (1e-3, 1e-6)]
    migliore_adam = min(v[0] for v in p_adam.values() if v[0] is not None)
    meno_passi = p_so[0] is not None and p_so[0] < p_adam[1e-2][0] and p_so[0] < migliore_adam
    # a 1e-6: Sophia prima di OGNI lr di AdamW; AdamW allo stesso lr di Sophia (1e-2) si ferma sopra (oscilla a ~lr)
    piu_giu = p_so[1] is not None and all(v[1] is None or p_so[1] < v[1] for v in p_adam.values()) \
        and p_adam[1e-2][1] is None
    # un passo a mano: m, h^ = h / (1 - b2^n), clip; e h assente -> lr segno(m)
    th = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    op = SO.SophiaTheta([th], lr=0.1, beta1=0.9, gamma=2.0, beta_h=0.99, eps=1e-15)
    g1 = torch.tensor([1.0, -2.0, 0.0, 4.0], dtype=torch.float64)
    th.grad = g1.clone()
    op.step()
    segno = torch.equal(th.detach(), -0.1 * torch.sign(0.1 * g1))
    hv = torch.tensor([0.5, 3.0, 1.0, 0.01], dtype=torch.float64)
    op.aggiorna_curvatura(th, hv)
    prima = th.detach().clone()
    g2 = torch.tensor([0.5, 1.0, 2.0, -1.0], dtype=torch.float64)
    th.grad = g2.clone()
    op.step()
    m = 0.9 * (0.1 * g1) + 0.1 * g2
    hh = (0.01 * hv) / (1 - 0.99)
    atteso = prima - 0.1 * (m / (2.0 * hh).clamp_min(1e-15)).clamp(-1, 1)
    a_mano = torch.allclose(th.detach(), atteso, rtol=0, atol=1e-15)
    st = op.state[th]
    stato_ok = st["n_hess"] == 1 and float(st["step"]) == 2 and torch.allclose(st["exp_avg"], m, atol=1e-16)
    ok = meno_passi and piu_giu and segno and a_mano and stato_ok
    return registra("N3_quadratica_sophia_meno_passi", ok, passi_adamw_a_1e3_e_1e6=p_adam,
                    passi_sophia_a_1e3_e_1e6=p_so, sophia_lr=1e-2, sophia_gamma=0.1,
                    finale_adamw={k: v[-1] for k, v in adam.items()}, finale_sophia=so[-1],
                    passo_senza_h_segno=segno, passo_a_mano=a_mano, stato=stato_ok, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── N4 ───────────────────────────────────────────
def _pv_finto_sophia():
    """PV finto di A28 (StatoV VERO) che nella curvatura riempie anche HT (>= 0, dipende da theta e dal passo)."""
    installa, Base, ultimo = TA._pv_finto_accumulo()

    class PVF(Base):
        def curvatura(self):
            m = super().curvatura()
            for (L, n), f in self.fam.items():
                b = self.stato.buffer(f)
                if b.HT is not None:
                    b.HT.add_(3e3 * (0.3 + torch.sin(self.st.theta[n].detach() * 311.0)) ** 2)
            return m
    return installa, PVF, ultimo


def _stato_sophia(rip) -> dict:
    return rip["opt"]["sophia"]["state"]


def _uguali_sophia(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        torch.equal(a[i]["exp_avg"], b[i]["exp_avg"]) and torch.equal(a[i]["hess"], b[i]["hess"])
        and torch.equal(a[i]["step"], b[i]["step"]) and a[i]["n_hess"] == b[i]["n_hess"] for i in a)


def prova_n4() -> bool:
    t0 = time.time()
    src, WS = TA.sorgente_finta(n_train=24)
    cfg = dict(righe_per_passo=4, pv_codici=True, pv_rho=0.5, pv_micro_curvatura=1, ripresa_ogni=1)
    ca_s = TA._ca_finta(ottimizzatore_theta="sophia", sophia_gamma=300.0, **cfg)
    ca_a = TA._ca_finta(**cfg)
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    installa, PVF, ultimo = _pv_finto_sophia()
    PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)
    reg: list = []

    def corsa(d, ca, **kw):
        st = TA._StudenteFintoV()
        return st, A.allena(st, src, d, ca, WS, WS, log=reg.append, **kw)
    try:
        dA, dB, dC, dD = (DIR / f"n4_{x}" for x in ("intera", "ripresa", "adam_a_sophia", "sophia_a_adam"))
        for d in (dA, dB, dC, dD):
            shutil.rmtree(d, ignore_errors=True)
        stA, sA = corsa(dA, ca_s)
        corsa(dB, ca_s, ferma_a=3)
        stB, sB = corsa(dB, ca_s, riprendi=True)
        rA, rB = A.carica_ripresa(dA, "cpu"), A.carica_ripresa(dB, "cpu")
        th = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
        so = SO.e_sophia(rA["opt"]) and _uguali_sophia(_stato_sophia(rA), _stato_sophia(rB))
        storia = TA._senza_tempi(sA) == TA._senza_tempi(sB)
        vs = [x.get("sophia") for x in sA["storia"]]
        curv = [v.get("curvatura_micro") for v in vs]
        n_hess = {i: v["n_hess"] for i, v in _stato_sophia(rA).items()}
        voci_ok = all(v is not None and 0 <= v["non_clippati"] <= v["vivi"] for v in vs) \
            and vs[0]["non_clippati"] == 0 and any(0 < v["non_clippati"] < v["vivi"] for v in vs[1:]) \
            and curv[0] is None and all(c == 1 for c in curv[1:]) and set(n_hess.values()) == {len(vs) - 1}
        mossi = sum(int((p != 0).sum()) for p in stA.parametri())
        # adamw -> sophia alla ripresa: spie sul caricamento e sul primo passo di Sophia
        corsa(dC, ca_a, ferma_a=3)
        rC0 = A.carica_ripresa(dC, "cpu")
        spie: dict = {"caricato": None, "passi": []}
        orig_car, orig_step = A.carica_ottimizzatore_sophia, SO.SophiaTheta.step

        def car_spia(opt, sd, st, np_, lp, log):
            orig_car(opt, sd, st, np_, lp, log)
            spie["caricato"] = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.sophia.state[p].items()}
                                for p in st.parametri()]

        def step_spia(self, closure=None):
            ps = [p for g in self.param_groups for p in g["params"]]
            prima = [p.detach().clone() for p in ps]
            orig_step(self, closure)
            spie["passi"].append([(a, p.detach().clone(), self.state[p]["exp_avg"].clone(), self.state[p]["n_hess"],
                                   self.param_groups[0]["lr"]) for a, p in zip(prima, ps)])
        A.carica_ottimizzatore_sophia, SO.SophiaTheta.step = car_spia, step_spia
        reg.clear()
        try:
            corsa(dC, ca_s, riprendi=True, ferma_a=4)
        finally:
            A.carica_ottimizzatore_sophia, SO.SophiaTheta.step = orig_car, orig_step
        rC = A.carica_ripresa(dC, "cpu")
        c0 = spie["caricato"]
        m_ripreso = c0 is not None and all(
            torch.equal(c0[i]["exp_avg"], rC0["opt"]["state"][i]["exp_avg"]) and c0[i]["n_hess"] == 0
            and torch.equal(c0[i]["step"], rC0["opt"]["state"][i]["step"]) and not bool(c0[i]["hess"].any())
            for i in range(len(c0)))
        primo = spie["passi"][0] if spie["passi"] else []
        passo_segno = bool(primo) and all(n == 0 and torch.equal(dopo, prima.add(torch.sign(m), alpha=-lr))
                                          for prima, dopo, m, n, lr in primo) \
            and all(v["n_hess"] == 1 for v in _stato_sophia(rC).values())
        log_c = any("adamw -> sophia" in x for x in reg)
        # sophia -> adamw
        corsa(dD, ca_s, ferma_a=3)
        reg.clear()
        stD, sD = corsa(dD, ca_a, riprendi=True, ferma_a=4)
        rD = A.carica_ripresa(dD, "cpu")
        adam_nuovo = (not SO.e_sophia(rD["opt"])) and all(float(v["step"]) == 1 for v in rD["opt"]["state"].values())
        log_d = any("sophia -> adamw" in x for x in reg)
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig
    rifiuto = False
    try:
        TA._ca_finta(ottimizzatore_theta="sophia")
    except ValueError as e:
        rifiuto = "pv-codici" in str(e)
    rifiuti = []
    for kw in ({"ottimizzatore_theta": "lion"}, {"ottimizzatore_theta": "sophia", "pv_codici": True, "sophia_gamma": 0},
               {"ottimizzatore_theta": "sophia", "pv_codici": True, "sophia_beta2": 1.0}):
        try:
            TA._ca_finta(**kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    from e2e_test_f32 import _arg
    a = _arg(pv_codici=True)
    a.ottimizzatore_theta, a.sophia_gamma, a.sophia_beta2, a.sophia_eps = "sophia", 3.0, 0.9, 1e-12
    c = A._config_da_arg(a)
    cli = (c.ottimizzatore_theta, c.sophia_gamma, c.sophia_beta2, c.sophia_eps) == ("sophia", 3.0, 0.9, 1e-12) \
        and A._config_da_arg(_arg()).ottimizzatore_theta == "adamw" \
        and {"ottimizzatore_theta", "sophia_gamma", "sophia_beta2", "sophia_eps"} <= A._CAMPI_DI_DECISIONE
    ok = (th and so and storia and voci_ok and mossi > 0 and m_ripreso and passo_segno and log_c and adam_nuovo and log_d
          and rifiuto and all(rifiuti) and cli and sA["passo"] == 6)
    return registra("N4_ciclo_sophia_ripresa_al_bit", ok, theta_al_bit=th, stato_sophia_al_bit=so, storia_al_bit=storia,
                    voci_sophia=vs, n_hess=sorted(set(n_hess.values())), theta_mossi=mossi,
                    adamw_a_sophia={"m_ripreso": m_ripreso, "passo_e_curvatura": passo_segno, "log": log_c},
                    sophia_a_adamw={"adam_da_zero": adam_nuovo, "log": log_d}, sophia_senza_pv_rifiutata=rifiuto,
                    rifiuti=rifiuti, cli=cli, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── N5 ───────────────────────────────────────────
def prova_n5() -> bool:
    import e2e_test_f32 as TF
    t0 = time.time()
    src, WS = TF.sorgente_minuscola(n_train=16)
    fam = S.famiglie_f32("attn_norm,ffn_gate_inp,ssm_dt")
    ca = TF._ca(allena_f32=fam, righe_per_passo=4, pv_codici=True, pv_rho=0.05, pv_micro_curvatura=1,
                ottimizzatore_theta="sophia", sophia_gamma=1.0)
    d_f, d_r = DIR / "n5_filata", DIR / "n5_ripresa"
    for d in (d_f, d_r):
        shutil.rmtree(d, ignore_errors=True)
    catturati: list = []
    orig = A.aggiorna_curvatura_sophia

    def spia(st, opt, pv, n_micro):
        r = orig(st, opt, pv, n_micro)
        catturati.append({k: pv.stato.buffer(f).HT.clone() for k, f in pv.fam.items()})
        return r
    A.aggiorna_curvatura_sophia = spia
    try:
        st_f = TF.studente("A32", fam)
        filata = A.allena(st_f, src, d_f, ca, WS, WS, log=muto)
        A.allena(TF.studente("A32", fam), src, d_r, ca, WS, WS, ferma_a=2, log=muto)
        st_b = TF.studente("A32", fam)
        ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=muto)
    finally:
        A.aggiorna_curvatura_sophia = orig
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    th = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
    f32 = all(torch.equal(a, b) for a, b in zip(st_f.parametri_f32(), st_b.parametri_f32()))
    so = SO.e_sophia(rf["opt"]) and _uguali_sophia(_stato_sophia(rf), _stato_sophia(rr))
    sa, sb = rf["opt"]["adam"]["state"], rr["opt"]["adam"]["state"]
    adam = sa.keys() == sb.keys() and len(sa) == len(st_f.parametri_f32()) and all(
        torch.equal(sa[i][k], sb[i][k]) for i in sa for k in ("exp_avg", "exp_avg_sq", "step"))
    stato = TA._senza_tempi(filata) == TA._senza_tempi(ripresa) and filata["passo"] == ripresa["passo"]
    vivi = {k: (st_f.esperti(k[0]).fam[k[1]].d0 != 0) for k in st_f.chiavi()}
    ht_pos = bool(catturati) and all(bool((c[k][vivi[k]] > 0).float().mean() > 0.5) for c in catturati for k in c)
    mossi = sum(int((p != 0).sum()) for p in st_f.parametri())
    storia_sophia = [x.get("sophia") for x in filata["storia"]]
    ok = th and f32 and so and adam and stato and ht_pos and mossi > 0 and all(v is not None for v in storia_sophia)
    return registra("N5_studente_vero_minuscolo_pv_sophia_f32", ok, theta=th, f32=f32, sophia=so, adam_f32=adam,
                    stato=stato, ht_positiva_sui_vivi=ht_pos, misure_ht=len(catturati), theta_mossi=mossi,
                    voci_sophia=storia_sophia, passi=filata["passo"],
                    pv=[x.get("pv", {}).get("salti") for x in filata["storia"]], secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── N6 ───────────────────────────────────────────
_N6_FIGLIO = r"""
import sys, re, json
sys.path.insert(0, %r)
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import e2e_pv_codici as PV
tgt = GPUTarget("hip", "gfx1151", 32)
b = PV.BLOCCHI_HT
out = {}
for dt in ("fp16", "fp32"):
    sig = {"dy_ptr": "*" + dt, "sdy_s": "i32", "sdy_r": "i32", "x_ptr": "*" + dt, "sx_s": "i32", "sx_c": "i32",
           "pb": "*u8", "sb_e": "i32", "NBR": "i32", "offs_ptr": "*i64", "R": "i32", "C": "i32", "ht_ptr": "*fp32",
           "sv_e": "i32", "NG": "i32", "GS": "constexpr", "QK": "constexpr", "NB": "constexpr", "ROT": "constexpr",
           "BR": "constexpr", "BS": "constexpr", "GSB": "constexpr"}
    for nome, rot, gs, qk, nb in (("R1024", 1, 128, 1024, 224), ("TQ1_0", 0, 256, 256, 54)):
        c = {"GS": gs, "QK": qk, "NB": nb, "ROT": rot, "BR": b.BR_dd, "BS": b.BS_dd, "GSB": min(gs, PV.GSB_HT)}
        k = triton.compile(ASTSource(fn=PV._k_ht, signature=sig, constexprs=c), target=tgt,
                           options={"num_warps": b.num_warps})
        a = k.asm["amdgcn"]
        out[f"{nome}_{dt}"] = {"spill": int(re.findall(r"vgpr_spill_count:\s+(\d+)", a)[0]),
                               "vgpr": int(re.findall(r"\.vgpr_count:\s+(\d+)", a)[0])}
print(json.dumps(out))
""" % str(QUI)


def prova_n6() -> bool:
    env = {**os.environ, "TRITON_INTERPRET": "0", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
           "ROCR_VISIBLE_DEVICES": ""}                           # solo compilazione: nessun dispositivo visibile
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", _N6_FIGLIO], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return registra("N6_compilazione_k_ht_gfx1151", False, errore=r.stderr[-1500:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    ok = all(v["spill"] == 0 for v in out.values())
    return registra("N6_compilazione_k_ht_gfx1151", ok, blocchi={"BR": PV.BLOCCHI_HT.BR_dd, "BS": PV.BLOCCHI_HT.BS_dd,
                    "warp": PV.BLOCCHI_HT.num_warps, "GSB": PV.GSB_HT}, risultati=out,
                    secondi=round(time.time() - t0, 1), nota="compilazione anticipata, nessuna GPU usata")


PROVE = {"N1": prova_n1, "N2": prova_n2, "N3": prova_n3, "N4": prova_n4, "N5": prova_n5, "N6": prova_n6}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = [a.upper() for a in sys.argv[1:]] or list(PROVE)
    ok = True
    for n in scelte:
        try:
            ok &= PROVE[n]()
        except Exception as e:                                     # noqa: BLE001
            traceback.print_exc()
            ok &= registra(n, False, errore=f"{type(e).__name__}: {e}")
    suf = "" if not sys.argv[1:] else "_" + "_".join(scelte)
    (DIR / f"esito_test_sophia{suf}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"TUTTO {'VERDE' if ok else 'ROSSO'}: {sum(v['ok'] for v in ESITO['prove'].values())}/{len(ESITO['prove'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
