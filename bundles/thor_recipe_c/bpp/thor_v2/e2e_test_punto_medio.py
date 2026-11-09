#!/usr/bin/env python3
"""Prove del ramo `punto-medio` (29/9/2026): prezzi delle primitive al PUNTO MEDIO PROPRIO nel passo V
(--pv-punto-medio; e2e_pv_codici.PassoV._prezzi_punto_medio; arXiv 2609.09867 Prop. 2; matematica/punto_medio_pv.md).
SOLO CPU (la GPU e' di TT2), niente @@ROOT:lavoro@@: i file vanno in $E2E_PM_DIR.

  M1  spento = al bit, MODULO: sul giocattolo «pacco» (sotto) e sul giocattolo V2/V3 di e2e_test_v_interazione, la sonda
      a dosi (senza e con ricentro, PV e V-Bop) del modulo nuovo coi default == modulo di 7c07fd6: byte, registri, esito,
      diagnostica, stato V-Bop. Piu' `ConfigDosi` rifiuta primitive e pm_max sbagliati.
  M2  spento = al bit, CICLO: `allena` del ramo coi campi nuovi ai default == `allena` di 7c07fd6 (studente vero
      minuscolo di e2e_test_integrata, V-Bop + dosi + ricentro): theta, F32, LoRA, byte, ottimizzatore, stato; i campi
      nuovi nel config sono tutti ai loro default.
  M3  PROBLEMA QUADRATICO NOTO: sul giocattolo «pacco» (perdita quadratica nei pesi) il prezzo al punto medio di ogni
      primitiva e' ESATTO (= F(q0 + d_p) - F(q0) misurato con un avanti sulla sonda A, errore relativo < 1e-3) mentre il
      prezzo nello stato corrente (d^T grad F(q0)) sbaglia il SEGNO sulla primitiva «pacco».
  M4  GIOCATTOLO: con le dosi grossolane (1/4, 1) la sonda a dosi di oggi RIFIUTA tutto (la prima dose e' gia' dannosa,
      come in TT2 al passo V 2); con --pv-punto-medio si tengono i salti buoni e la perdita scende su dati DISGIUNTI (lotto
      C, mai usato ne' per i prezzi ne' per le dosi ne' per la conferma). Con due funzionali (obiettivo A = sola famiglia
      buona, vincolo B = perdita intera) la scelta e la conferma dell'obiettivo su C funzionano. Registro anche il caso a
      dosi fini (dove le dosi di oggi trovano un pezzo del pacco: il punto medio per famiglie NON e' un sovrainsieme).
  M5  `scegli_primitive` (un funzionale; obiettivo + vincolo) e `prezzo_punto_medio` su casi scritti a mano.
  M6  CICLO ACCESO: `allena` con --pv-sonda-dosi --pv-punto-medio sullo studente vero minuscolo: 3 passi, riga del
      registro «punto medio», voce nella storia, ripresa 2 + 1 == 3 AL BIT.

Uso (4 fili, nice 19, niente GPU):
    systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=10G nice -n 19 env CUDA_VISIBLE_DEVICES= \\
        HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
        E2E_PM_DIR=@@ROOT:home@@/claude-pod-lavoro/punto-medio-prove \\
        @@ROOT:venv@@/bin/python3 e2e_test_punto_medio.py [M1 M2 ...]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
_PM = os.environ.setdefault("E2E_PM_DIR", "@@ROOT:home@@/claude-pod-lavoro/punto-medio-prove")
os.environ.setdefault("E2E_VI_DIR", str(Path(_PM) / "vi"))
os.environ.setdefault("E2E_PV_DIR", str(Path(_PM) / "pv"))
os.environ.setdefault("E2E_ALLENA_PROVE", str(Path(_PM) / "allena"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_test_pv_codici as TP  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
DIR = Path(_PM)
BASE = os.environ.get("E2E_PM_BASE", "7c07fd6")          # densi-ternari = il codice congelato di TT2
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
RIC = TP.RIC


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"{'OK ' if ok else 'NO '} {nome}", flush=True)
    return bool(ok)


def _json(x) -> str:
    return json.dumps(x, sort_keys=True, default=lambda o: repr(o))


# ───────────────────────────────────── giocattolo «pacco» ─────────────────────────────────────
class Pacco:
    """Due famiglie TQ1_0 sintetiche (1 esperto), scale ~ 1 (theta = -log d0), ingressi nulli fuori dalle colonne attive.
      strato 0 = PACCO: R0 righe, UNA colonna attiva c* comune (come la colonna 1074 dell'attivazione massiccia di
               39.ffn_down_shexp in TT2, dove 601 dei primi 1000 salti cadono sulla stessa colonna); maestro = studente +
               rho s_r d_r su c*, perdita (sum_r s_r e_r)^2 (le righe si sommano come nella proiezione sulla testa).
               Un salto da solo porta l'errore comune da rho R a rho R - 1 (buono se rho R > 1/2); tutti insieme a
               (rho - 1) R: con rho < 1/2 molto peggio. Il criterio diagonale vede R salti ottimi.
      strato 1 = BUONI: R1 righe indipendenti, residuo 0,9 d su una colonna propria: ogni salto porta 0,9 -> 0,1.
    Perdita del passo = somma sui 16 micro-lotti da T token / n_tot; sonda = media per token su T token."""

    def __init__(self, M, R0: int = 64, R1: int = 32, rho_res: float = 0.1, T: int = 256, seme: int = 5):
        self.M, self.T, self.seme, self.rho_res = M, T, seme, rho_res
        f0 = TP.famiglia_sintetica(E=1, R=R0, C=1024, seme=seme)
        f1 = TP.famiglia_sintetica(E=1, R=R1, C=1024, seme=seme + 1)
        self.f0 = M.famiglia_scrivibile(S.DatiFamiglia(f0.nome, f0.tipo, f0.blocchi, f0.d0, 0))
        self.f1 = M.famiglia_scrivibile(S.DatiFamiglia(f1.nome, f1.tipo, f1.blocchi, f1.d0, 1))
        self.th = {0: torch.log(1.0 / self.f0.d0), 1: torch.log(1.0 / self.f1.d0)}
        self.d = {0: TP.scala(self.f0, self.th[0]).detach(), 1: TP.scala(self.f1, self.th[1]).detach()}
        g = torch.Generator().manual_seed(seme)
        gs = self.f0.gs
        # pacco
        T0 = TP.trit_famiglia(self.f0)[0]
        W0 = (T0.to(torch.float64).view(R0, self.f0.NG, gs) * self.d[0][0].to(torch.float64).unsqueeze(-1)).view(R0, -1)
        self.c = int(torch.randint(0, gs, (1,), generator=g))
        self.u = torch.zeros(R0, dtype=torch.float64)
        self.Wt0 = W0.clone()
        for r in range(R0):
            t = int(T0[r, self.c])
            s = -1 if t == 1 else (1 if t == -1 else (1 if bool(torch.rand(1, generator=g) < 0.5) else -1))
            self.u[r] = s
            self.Wt0[r, self.c] += rho_res * s * float(self.d[0][0, r, self.c // gs])
        # buoni
        T1 = TP.trit_famiglia(self.f1)[0]
        W1 = (T1.to(torch.float64).view(R1, self.f1.NG, gs) * self.d[1][0].to(torch.float64).unsqueeze(-1)).view(R1, -1)
        self.Wt1 = W1.clone()
        self.col1, usate = [], set()
        for r in range(R1):
            cand = [c for c in range(gs) if int(T1[r, c]) == 0 and c not in usate]
            c = cand[int(torch.randint(0, len(cand), (1,), generator=g))]
            usate.add(c)
            s = 1 if bool(torch.rand(1, generator=g) < 0.5) else -1
            self.Wt1[r, c] += 0.9 * s * float(self.d[1][0, r, c // gs])
            self.col1.append(c)
        self.k0, self.k1 = (0, self.f0.nome), (1, self.f1.nome)

    def famiglie(self) -> dict:
        return {self.k0: TP._copia_fam(self.f0), self.k1: TP._copia_fam(self.f1)}

    def lotto(self, passo: int, kk: int):
        g = torch.Generator().manual_seed(self.seme * 1000003 + passo * 1009 + kk)
        z = torch.randn(self.T, 1024, generator=g, dtype=torch.float64)
        x0, x1 = torch.zeros_like(z), torch.zeros_like(z)
        x0[:, self.c] = z[:, self.c]
        x1[:, self.col1] = z[:, self.col1]
        return x0.float(), x1.float(), S.Piano(torch.zeros(self.T, 1, dtype=torch.int64), 1)

    def parti(self, fams, x0, x1, piano, mot, th=None):
        th = th or {0: self.th[0], 1: self.th[1]}
        y0 = S.FamigliaEsperti.apply(x0, TP.scala(fams[self.k0], th[0]), fams[self.k0], piano, RIC, mot)
        y1 = S.FamigliaEsperti.apply(x1, TP.scala(fams[self.k1], th[1]), fams[self.k1], piano, RIC, mot)
        e0 = y0.to(torch.float64) - x0.to(torch.float64) @ self.Wt0.T
        e1 = y1.to(torch.float64) - x1.to(torch.float64) @ self.Wt1.T
        return ((e0 @ self.u) ** 2).sum(), (e1 ** 2).sum(), y0, y1

    def raccogli(self, pv, fams, passo: int = 0, n_micro: int = 16, n_curv: int = 2) -> int:
        mot = self.M.MotoreTorchV(pv.stato)
        pv.inizia_passo()
        n_tot = n_micro * self.T
        for kk in range(n_micro):
            pv.microlotto(kk)
            x0, x1, p = self.lotto(passo, kk)
            th = {i: self.th[i].clone().requires_grad_(True) for i in (0, 1)}
            l0, l1, _, _ = self.parti(fams, x0, x1, p, mot, th)
            ((l0 + l1) / n_tot).backward()
        if pv.stato.serve_curvatura():
            for kk in range(n_curv):
                pv.curvatura()
                x0, x1, p = self.lotto(passo, 1000 + kk)
                th = {i: self.th[i].clone().requires_grad_(True) for i in (0, 1)}
                _, _, y0, y1 = self.parti(fams, x0, x1, p, mot, th)
                gr = torch.Generator().manual_seed(self.seme * 7919 + passo * 31 + kk)
                eps = torch.randint(0, 2, (self.T, 1), generator=gr).to(torch.float64) * 2 - 1
                v0 = (eps * self.u[None, :] * math.sqrt(2.0 / n_tot)).to(y0.dtype)          # cov = 2 u u^T / n_tot
                v1 = ((torch.randint(0, 2, y1.shape, generator=gr).to(y1.dtype) * 2 - 1) * math.sqrt(2.0 / n_tot))
                torch.autograd.backward([y0, y1], [v0, v1])
        pv.fine_raccolta()
        return n_tot

    def misura(self, fams, lot, quale: str = "B") -> float:
        x0, x1, p = lot
        with torch.no_grad():
            l0, l1, _, _ = self.parti(fams, x0, x1, p, S.MotoreTorch())
        return float((l1 if quale == "A" else l0 + l1) / self.T)

    def gradiente(self, pv, fams, lot, n_tot: int, quale: str = "B"):
        mot = self.M.MotoreTorchV(pv.stato)

        def g():
            x0, x1, p = lot
            th = {i: self.th[i].clone().requires_grad_(True) for i in (0, 1)}
            l0, l1, _, _ = self.parti(fams, x0, x1, p, mot, th)
            ((l1 if quale == "A" else l0 + l1) / n_tot).backward()
            return True
        return g

    def conta(self, fams) -> dict:
        a = int((TP.trit_famiglia(fams[self.k0]) != TP.trit_famiglia(self.f0)).sum())
        b = int((TP.trit_famiglia(fams[self.k1]) != TP.trit_famiglia(self.f1)).sum())
        return {"pacco": a, "buoni": b}


def _passo_v(M, gio: Pacco, cfg, bop=None, ricentra_grad=False, pm=False, due=False, rho=1.0):
    """Un passo V completo sul giocattolo col modulo M. Ritorna (fams, pv, reg, diag, esito, sonde)."""
    fams = gio.famiglie()
    pv = M.PassoV(fams, M.ConfigV(accetta="guadagno"), bop=bop)
    n_tot = gio.raccogli(pv, fams)
    lA, lB, lC = gio.lotto(1, 0), gio.lotto(1, 1), gio.lotto(1, 2)
    ma, mb = (lambda: gio.misura(fams, lA)), (lambda: gio.misura(fams, lB))
    gr = gio.gradiente(pv, fams, lA, n_tot) if ricentra_grad else None
    scale = {gio.k0: gio.d[0], gio.k1: gio.d[1]}
    kw = {}
    if pm:
        gp = {"B": gio.gradiente(pv, fams, lA, n_tot)}
        fat = {"B": n_tot / gio.T}
        mo = None
        if due:
            lM1 = gio.lotto(1, 3)
            gp["A"] = gio.gradiente(pv, fams, lM1, n_tot, "A")
            fat["A"] = n_tot / gio.T
            mo = lambda: gio.misura(fams, lC, "A")  # noqa: E731
        kw = {"gradienti_pm": gp, "fattori_pm": fat, "misura_obiettivo": mo}
    reg, diag, es = pv.seleziona_e_prova_dosi(scale, rho, ma, mb, cfg, gr, **kw)
    return fams, pv, reg, diag, es, (lA, lB, lC)


# ─────────────────────────────────────────── M1 ───────────────────────────────────────────
def _stato_bop(pv) -> dict:
    return {f"{k}": {c: getattr(z, c).clone() for c in PV._CAMPI_BOP} for k, z in pv.stato.bop_.items()}


def prova_m1() -> bool:
    t0 = time.time()
    VEC = TP._pv_prima(BASE)
    casi = {"dosi": (lambda m: m.ConfigDosi(), None, False),
            "dosi_grosse": (lambda m: m.ConfigDosi(frazioni=(0.25, 1.0)), None, False),
            "ricentro": (lambda m: m.ConfigDosi(frazioni=(1 / 64, 1 / 16, 1.0), ricentra=True), None, True),
            "vbop_ricentro": (lambda m: m.ConfigDosi(ricentra=True), lambda m: m.ConfigBop(gamma=0.5), True)}
    uguali = {}
    for nome, (cfg, bop, ric) in casi.items():
        r = {}
        for tag, M in (("prima", VEC), ("nuovo", PV)):
            gio = Pacco(M)
            fams, pv, reg, diag, es, _ = _passo_v(M, gio, cfg(M), bop(M) if bop else None, ric)
            r[tag] = {"byte": {str(k): f.blocchi.clone() for k, f in fams.items()},
                      "reg": None if reg is None else reg.righe.copy(), "diag": _json(diag), "esito": _json(es),
                      "registri": [x.righe.copy() for x in pv.registri],
                      "bop": _stato_bop(pv) if bop else {}, "conta": gio.conta(fams)}
        a, b = r["prima"], r["nuovo"]
        uguali[nome] = {
            "byte": all(torch.equal(a["byte"][k], b["byte"][k]) for k in a["byte"]),
            "registro": (a["reg"] is None and b["reg"] is None) or (a["reg"] is not None and b["reg"] is not None
                                                                    and np.array_equal(a["reg"], b["reg"])),
            "registri": len(a["registri"]) == len(b["registri"]) and all(
                np.array_equal(x, y) for x, y in zip(a["registri"], b["registri"])),
            "diag": a["diag"] == b["diag"], "esito": a["esito"] == b["esito"],
            "bop": all(torch.equal(a["bop"][k][c], b["bop"][k][c]) for k in a["bop"] for c in a["bop"][k]),
            "conta": a["conta"] == b["conta"], "salti_tenuti": json.loads(b["esito"])["salti_tenuti"]}
    # V2/V3 di e2e_test_v_interazione (stesso modulo nuovo coi default): gia' verdi contro integrata; qui contro 7c07fd6
    rifiuti = []
    for kw in ({"pm_primitiva": "riga"}, {"pm_max": 0}):
        try:
            PV.ConfigDosi(**kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    try:                                          # punto medio acceso senza il gradiente B: rifiutato
        gio = Pacco(PV)
        _passo_v(PV, gio, PV.ConfigDosi(punto_medio=True))
        rifiuti.append(False)
    except ValueError:
        rifiuti.append(True)
    ok = all(all(v2 for k2, v2 in v.items() if k2 != "salti_tenuti") for v in uguali.values()) and all(rifiuti)
    return registra("M1_spento_al_bit_modulo", ok, uguali=uguali, rifiuti=rifiuti, base=BASE,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M2 ───────────────────────────────────────────
def prova_m2() -> bool:
    import e2e_allena as A
    import e2e_test_allena as TA
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    t0 = time.time()
    VEC = TA._modulo_prima(BASE)
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    ris = {}
    for nome, mod in (("prima", VEC), ("nuovo", A)):
        d = DIR / f"m2_{nome}"
        TI._pulisci(d)
        base = TI._ca_integrata(pv_sonda_dosi=True, pv_ricentra=True, pv_dosi_frazioni=(1 / 16, 1 / 4, 1.0))
        ca = mod.ConfigAllena(**{k: v for k, v in base.__dict__.items() if k in mod.ConfigAllena.__dataclass_fields__})
        st = TI._studente_int()
        z = mod.allena(st, src, d, ca, WS, WS, log=TI.muto)
        ris[nome] = (st, z, mod.carica_ripresa(d, "cpu"))
    (sp, zp, rp), (sn, zn, rn) = ris["prima"], ris["nuovo"]
    a_, b_ = TI._stato_confrontabile(zn), TI._stato_confrontabile(zp)
    nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    a_.pop("versione", None)
    b_.pop("versione", None)
    attesi = {"pv_punto_medio", "pv_pm_primitiva", "pv_pm_max", "pv_pm_filtra", "pv_pm_lettere",
              "pv_pm_golosa"}                     # 29/9 ramo pv-lettere-dosi
    u = {"theta": TI._uguali_lista(sn.parametri(), sp.parametri()),
         "f32": TI._uguali_lista(sn.parametri_f32(), sp.parametri_f32()),
         "lora": TI._uguali_lora_ga(sn, sp), "byte": TI._uguali_lista(TI._byte(sn), TI._byte(sp)),
         "opt": TI._uguali_stato_opt(rn["opt"], rp["opt"]),
         "stato": _json(a_) == _json(b_), "registri_sha": zn["pv"]["sha"] == zp["pv"]["sha"],
         "campi_nuovi": set(nuovi) == attesi and nuovi == {k: A._CONFIG_DEFAULT[k] for k in nuovi}}
    pv = [(x["pv"]["salti"], x["pv"]["tenuto"], x["pv"].get("salti_tenuti")) for x in zn["storia"] if x.get("pv")]
    return registra("M2_spento_al_bit_ciclo", all(u.values()), uguali=u, campi_nuovi=nuovi, pv=pv, base=BASE,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M3 ───────────────────────────────────────────
def prova_m3() -> bool:
    """Prezzo al punto medio == variazione esatta (problema quadratico), prezzo nello stato corrente col segno sbagliato."""
    t0 = time.time()
    gio = Pacco(PV)
    fams = gio.famiglie()
    pv = PV.PassoV(fams, PV.ConfigV(accetta="guadagno"))
    n_tot = gio.raccogli(pv, fams)
    lA = gio.lotto(1, 0)
    spia = {}
    orig = pv._prezzi_punto_medio

    def spione(chiavi, sel, righe, *a, **k):
        spia["righe"] = righe.copy()
        return orig(chiavi, sel, righe, *a, **k)
    pv._prezzi_punto_medio = spione
    cfg = PV.ConfigDosi(frazioni=(1.0,), punto_medio=True, pm_filtra=False)
    B0 = {k: f.blocchi.clone() for k, f in fams.items()}
    # misure esatte PRIMA (lo stato dei byte e' q0): per ogni primitiva (strato) F(q0 + d_p) - F(q0) sulla sonda A
    reg, diag, es = pv.seleziona_e_prova_dosi({gio.k0: gio.d[0], gio.k1: gio.d[1]}, 1.0,
                                              lambda: gio.misura(fams, lA), lambda: gio.misura(fams, gio.lotto(1, 1)),
                                              cfg, None, gradienti_pm={"B": gio.gradiente(pv, fams, lA, n_tot)},
                                              fattori_pm={"B": n_tot / gio.T})
    if reg is not None:
        PV.annulla(fams, reg)                    # torno a q0 per le misure esatte
    torna_q0 = all(torch.equal(B0[k], f.blocchi) for k, f in fams.items())
    pm = es["punto_medio"]
    righe = spia["righe"]
    F0 = gio.misura(fams, lA)
    per = {}
    for r in pm["per_primitiva"]:
        L = int(r["primitiva"].strip("()").split(",")[0])
        rg = PV.Registro(righe[righe["L"] == L].copy())
        PV.rigioca(fams, [rg])
        esatto = gio.misura(fams, lA) - F0
        PV.annulla(fams, rg)
        per[L] = {"salti": r["salti"], "esatto": esatto, "medio": r["s_B"], "corrente": r["corrente_B"],
                  "errore_rel_medio": abs(r["s_B"] - esatto) / abs(esatto),
                  "errore_rel_corrente": abs(r["corrente_B"] - esatto) / abs(esatto)}
    ok = (torna_q0 and set(per) == {0, 1}
          and all(p["errore_rel_medio"] < 1e-3 for p in per.values())
          and per[0]["esatto"] > 0 and per[0]["corrente"] < 0            # pacco: il corrente sbaglia il segno
          and per[1]["esatto"] < 0 and per[1]["medio"] < 0
          and pm["scelte"] == [str(gio.k1)])
    return registra("M3_quadratico_punto_medio_esatto_corrente_segno_sbagliato", ok, primitive=per,
                    scelte=pm["scelte"], indietro=pm["indietro"], torna_a_q0=torna_q0, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M4 ───────────────────────────────────────────
def prova_m4() -> bool:
    t0 = time.time()
    out = {}
    for nome, fr in (("grosse", (0.25, 1.0)), ("fini", PV.FRAZIONI_DOSI)):
        r = {}
        for modo in ("oggi", "punto_medio", "punto_medio_ricentro"):
            gio = Pacco(PV)
            cfg = PV.ConfigDosi(frazioni=fr, punto_medio=modo != "oggi", ricentra=modo.endswith("ricentro"))
            fams, pv, reg, diag, es, (lA, lB, lC) = _passo_v(PV, gio, cfg, pm=modo != "oggi",
                                                             ricentra_grad=modo.endswith("ricentro"))
            F_C = gio.misura(fams, lC)
            if reg is not None:
                PV.annulla(fams, reg)
            F_C0 = gio.misura(fams, lC)
            r[modo] = {"tenuto": es["tenuto"], "salti_scelti": es["salti"], "salti_tenuti": es["salti_tenuti"],
                       "dosi": [(x["salti"], round(x["sonda_a"], 4)) for x in es["dosi"]],
                       "A": (es["sonda_prima_a"], es["sonda_dopo_a"]), "B": (es["sonda_prima_b"], es["sonda_dopo_b"]),
                       "C_prima": F_C0, "C_dopo": F_C, "applicati": gio.conta(fams) if reg is None else None,
                       "pm": {k: es.get("punto_medio", {}).get(k) for k in ("scelte", "indietro", "previsto_B")}}
            if reg is not None:
                PV.rigioca(fams, [reg])
                r[modo]["applicati"] = gio.conta(fams)
        out[nome] = r
    # due funzionali: obiettivo A = sola famiglia buona (lotto M1 per i prezzi), vincolo B = perdita intera (sonda A),
    # conferma dell'obiettivo sul lotto C
    gio = Pacco(PV)
    fams, pv, reg, diag, es, (lA, lB, lC) = _passo_v(PV, gio, PV.ConfigDosi(frazioni=(0.25, 1.0), punto_medio=True),
                                                     pm=True, due=True)
    due = {"tenuto": es["tenuto"], "salti_tenuti": es["salti_tenuti"], "obiettivo": es["punto_medio"].get("obiettivo"),
           "scelte": es["punto_medio"]["scelte"], "indietro": es["punto_medio"]["indietro"],
           "applicati": gio.conta(fams)}
    g, pm = out["grosse"]["oggi"], out["grosse"]["punto_medio"]
    ok = (not g["tenuto"] and g["salti_tenuti"] == 0 and g["C_dopo"] == g["C_prima"]        # oggi: rifiuta tutto
          and pm["tenuto"] and pm["salti_tenuti"] > 0 and pm["applicati"]["pacco"] == 0
          and pm["C_dopo"] < pm["C_prima"]                                                  # scende su dati disgiunti
          and pm["B"][1] <= pm["B"][0]
          and due["tenuto"] and due["obiettivo"]["confermato"] and due["applicati"]["pacco"] == 0)
    return registra("M4_giocattolo_punto_medio_accetta_mosse_che_migliorano", ok, dosi=out, due_funzionali=due,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── M5 ───────────────────────────────────────────
def prova_m5() -> bool:
    t = lambda *x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    casi = {
        "uno": PV.scegli_primitive(t(-1.0, 2.0, -3.0, math.nan, 0.0)) == [2, 0],
        # sicure (s_A<0, s_B<=0) per s_A; poi costose per s_A/s_B finche' sum s_B <= 0
        "due": PV.scegli_primitive(t(-1.0, -5.0, -2.0, -4.0, 1.0), t(-3.0, 1.0, 0.0, 2.0, -9.0)) == [1 * 0 + 2, 0, 1],
        "due_nessuna": PV.scegli_primitive(t(-1.0), t(1.0)) == [],
        "trapezio": bool(torch.allclose(PV.prezzo_punto_medio(t(2.0, -1.0), t(1.0, 1.0), t(3.0, -3.0)), t(4.0, 1.0))),
    }
    # "due": sicure = {0 (s_A -1, s_B -3), 2 (s_A -2, s_B 0)} per s_A -> [2, 0]; margine 3; costose 1 (-5/1 = -5) e
    # 3 (-4/2 = -2): prima la 1 (s_B 1 <= 3, margine 2), poi la 3 (s_B 2 <= 2, margine 0) -> [2, 0, 1, 3]
    casi["due"] = PV.scegli_primitive(t(-1.0, -5.0, -2.0, -4.0, 1.0), t(-3.0, 1.0, 0.0, 2.0, -9.0)) == [2, 0, 1, 3]
    import e2e_allena as A
    rif = []
    for kw in ({"pv_codici": True, "pv_punto_medio": True},
               {"pv_codici": True, "pv_sonda_dosi": True, "pv_punto_medio": True, "pv_pm_max": 0},
               {"pv_codici": True, "pv_sonda_dosi": True, "pv_pm_lettere": True},
               {"pv_codici": True, "pv_sonda_dosi": True, "pv_punto_medio": True, "pv_pm_lettere": True}):
        try:
            A.ConfigAllena(**kw)
            rif.append(False)
        except ValueError:
            rif.append(True)
    casi["config_rifiutate"] = all(rif)
    return registra("M5_scelta_e_trapezio", all(casi.values()), casi=casi, rifiuti=rif)


# ─────────────────────────────────────────── M6 ───────────────────────────────────────────
def prova_m6() -> bool:
    import e2e_allena as A
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    kw = dict(pv_sonda_dosi=True, pv_ricentra=True, pv_dosi_frazioni=(1 / 16, 1 / 4, 1.0), pv_punto_medio=True,
              pv_pm_max=4)
    d = DIR / "m6_tutto"
    TI._pulisci(d)
    st = TI._studente_int()
    lg: list = []
    z = A.allena(st, src, d, TI._ca_integrata(**kw), WS, WS, log=lg.append)
    # ripresa: fermo al passo 1, poi ripreso fino in fondo: stesso modello AL BIT della corsa filata
    d2 = DIR / "m6_ripresa"
    TI._pulisci(d2)
    A.allena(TI._studente_int(), src, d2, TI._ca_integrata(**kw), WS, WS, ferma_a=1, log=TI.muto)
    st3 = TI._studente_int()
    z3 = A.allena(st3, src, d2, TI._ca_integrata(**kw), WS, WS, riprendi=True, log=TI.muto)
    # accesa alla ripresa (spenta fino al passo 1): decisione registrata, nessun rifiuto della configurazione
    d3 = DIR / "m6_accesa"
    TI._pulisci(d3)
    kw0 = {k: v for k, v in kw.items() if not k.startswith("pv_p")}
    A.allena(TI._studente_int(), src, d3, TI._ca_integrata(**kw0), WS, WS, ferma_a=1, log=TI.muto)
    lg3: list = []
    z4 = A.allena(TI._studente_int(), src, d3, TI._ca_integrata(**kw), WS, WS, riprendi=True, log=lg3.append)
    cambi = next((m for m in lg3 if str(m).startswith("ripresa con campi cambiati")), "")
    righe_pm = [x for x in lg if "punto medio" in str(x)]
    voci = [x["pv"].get("punto_medio") for x in z["storia"] if x.get("pv")]
    ripresa = (TI._uguali_lista(st3.parametri(), st.parametri()) and TI._uguali_lista(TI._byte(st3), TI._byte(st))
               and z3["pv"]["sha"] == z["pv"]["sha"] and z3["passo"] == z["passo"])
    accesa = "pv_punto_medio" in cambi and z4["passo"] == z["passo"] and any(
        (x.get("pv") or {}).get("punto_medio") for x in z4["storia"])
    ok = bool(righe_pm) and any(v is not None for v in voci) and ripresa and accesa
    return registra("M6_ciclo_acceso", ok, righe_registro=[str(x)[:600] for x in righe_pm][:4],
                    voci=[_json(v)[:1500] for v in voci], ripresa_al_bit=ripresa, accesa_alla_ripresa=accesa,
                    riga_cambi=cambi[:400], secondi=round(time.time() - t0, 1))


PROVE = {"M1": prova_m1, "M2": prova_m2, "M3": prova_m3, "M4": prova_m4, "M5": prova_m5, "M6": prova_m6}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    for v in ("E2E_PV_DIR", "E2E_VI_DIR", "E2E_ALLENA_PROVE"):
        Path(os.environ[v]).mkdir(parents=True, exist_ok=True)
    scelte = [a for a in sys.argv[1:] if a in PROVE] or list(PROVE)
    esito_p = DIR / "esito_test_punto_medio.json"
    if esito_p.exists():
        try:
            ESITO["prove"].update(json.loads(esito_p.read_text()).get("prove", {}))
        except Exception as ex:  # noqa: BLE001
            print(f"esito precedente illeggibile ({ex}): si riparte", flush=True)
    ok = True
    for nome in scelte:
        for k in [k for k in ESITO["prove"] if k.startswith(nome + "_")]:
            del ESITO["prove"][k]
        t0 = time.time()
        try:
            ok &= PROVE[nome]()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            ok &= registra(f"{nome}_eccezione", False, errore=repr(ex))
        print(f"  ({nome}: {time.time() - t0:.1f} s)", flush=True)
        esito_p.write_text(json.dumps(ESITO, indent=1, default=str))
    print("ESITO:", "TUTTO VERDE" if ok else "QUALCOSA NON TORNA", "->", esito_p, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
