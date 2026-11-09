#!/usr/bin/env python3
"""Prove del ramo `pv-lettere-dosi` (29/9/2026): DOSI GOLOSE decise dal funzionale obiettivo nel passo V col punto medio
(--pv-pm-golosa; e2e_pv_codici.ConfigDosi.pm_golosa, PassoV._dosi_golose, ordina_primitive). SOLO CPU.

Causa misurata in TT2 al passo V 4 (registro, riga 100-101): 3 primitive scelte -> `conteggi_dosi(3, frazioni)` = [1, 3]
(il prefisso di 2 primitive non e' mai provato) e la dose e' l'argmin della sonda GENERICA A (0.86483 contro 0.86532):
la decisione non porta il termine delle lettere (regola della fucina E73), e una primitiva cattiva in mezzo boccia
tutte quelle dopo.

  G1  spento = al bit, MODULO: col punto medio acceso (un funzionale; due funzionali; con ricentro; V-Bop) e pm_golosa
      ai default, il modulo del ramo == modulo di c0cb0ff (il codice congelato di TT2) su due giocattoli («pacco» di
      e2e_test_punto_medio e «mosaico» sotto): byte, registri, esito, diagnostica, stato V-Bop.
  G2  spento = al bit, CICLO: `allena` del ramo col punto medio acceso e pv_pm_golosa ai default == `allena` di c0cb0ff.
  G3  GIOCATTOLO «mosaico» (due funzionali, obiettivo = «lettere», vincolo = generica): oggi si tiene la sola prima
      primitiva; golosa tiene PIU' salti (la coppia che interagisce viene spezzata, la primitiva buona dopo entra, quella
      che rompe il vincolo generico resta fuori), l'obiettivo di CONFERMA (M2) e quello su un lotto mai visto (C) non
      peggiorano rispetto a oggi, la generica di conferma B resta <= partenza.
  G4  MUTAZIONI: tre rotture del rimedio (primitiva rifiutata non annullata; vincolo della sonda A tolto; tenuta sempre)
      fanno diventare rossa la G3.
  G5  `ordina_primitive` su casi scritti a mano, ConfigDosi/ConfigAllena rifiutano pm_golosa senza punto medio.
  G6  CICLO ACCESO: `allena` con --pv-pm-golosa sullo studente minuscolo: dosi golose in storia e registro, ripresa al bit.

Uso (<= 8 fili, niente GPU):
    nice -n 19 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
        E2E_PM_DIR=@@ROOT:home@@/claude-pod-lavoro/punto-medio-prove \\
        @@ROOT:venv@@/bin/python3 e2e_test_pm_golosa.py [G1 G2 ...]
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
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
import e2e_test_punto_medio as TM  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
DIR = Path(_PM) / "golosa"
BASE = os.environ.get("E2E_GOLOSA_BASE", "c0cb0ff")        # il codice congelato di TT2 (@@ROOT:depot@@/e2e/codice/TT2-c0cb0ff)
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
RIC = TP.RIC


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"{'OK ' if ok else 'NO '} {nome}", flush=True)
    return bool(ok)


def _json(x) -> str:
    return json.dumps(x, sort_keys=True, default=lambda o: repr(o))


def _modulo_da_testo(testo: str, nome: str):
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{nome}.py"
    p.write_text(testo)
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


def _pv_base():
    testo = subprocess.run(["git", "-C", str(QUI), "show", f"{BASE}:thor_v2/e2e_pv_codici.py"], check=True,
                           capture_output=True, text=True).stdout
    return _modulo_da_testo(testo, f"e2e_pv_codici_{BASE}")


# ───────────────────────────────────── giocattolo «mosaico» ─────────────────────────────────────
# Ogni famiglia (un esperto, uno strato proprio = una primitiva «famiglia») ha R righe che scrivono tutte su UNA colonna
# d'ingresso; l'errore per token della famiglia i e' z_t * sum_r s_r (w_r - wt_r) (s = verso fattibile del salto). Le
# famiglie di uno stesso TERMINE ricevono la stessa z e i loro errori si sommano prima del quadrato: e' l'interazione
# FRA primitive che il prezzo al punto medio (una primitiva alla volta) non vede. Due bersagli per famiglia: generico
# (sonde A/B) e obiettivo («lettere»: lotti M1, M2, C). Perdita di raccolta = generica + 2 obiettivo.
#   A1, A2  termine «coppia», 16 righe, bersaglio 0,5 D in entrambi: da sola ciascuna chiude il residuo, insieme lo
#           riaprono dall'altra parte (A2 dopo A1 fa danno misurato).
#   G       termine «g», 4 righe, bersaglio pieno in entrambi: buona, piccola (in ordine dopo la coppia).
#   K       termine «k», 20 righe: bersaglio pieno nell'obiettivo, NESSUNO nel generico (costosa: s_A < 0, s_B > 0);
#           il bilancio PREVISTO (che conta anche il guadagno di A2) la ammette, quello MISURATO no.
SPEC_MOSAICO = (("A1", "coppia", 16, 0.5, 0.5), ("A2", "coppia", 16, 0.5, 0.5), ("G", "g", 4, 1.0, 1.0),
                ("K", "k", 20, 0.0, 1.0))
PESO_OBJ = 2.0


class Mosaico:
    def __init__(self, M, T: int = 256, seme: int = 11, spec=SPEC_MOSAICO):
        self.M, self.T, self.seme, self.spec = M, T, seme, spec
        g = torch.Generator().manual_seed(seme)
        self.termini = sorted({s[1] for s in spec})
        self.fam0, self.k, self.th, self.d, self.c, self.u, self.Wg, self.Wo, self.term = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for i, (nome, term, R, rg, ro) in enumerate(spec):
            f = TP.famiglia_sintetica(E=1, R=R, C=1024, seme=seme + 17 * i)
            f = M.famiglia_scrivibile(S.DatiFamiglia(f.nome, f.tipo, f.blocchi, f.d0, i))
            self.fam0[nome], self.k[nome], self.term[nome] = f, (i, f.nome), term
            self.th[nome] = torch.log(1.0 / f.d0)
            self.d[nome] = TP.scala(f, self.th[nome]).detach()
            gs = f.gs
            T0 = TP.trit_famiglia(f)[0]
            W = (T0.to(torch.float64).view(R, f.NG, gs) * self.d[nome][0].to(torch.float64).unsqueeze(-1)).view(R, -1)
            c = int(torch.randint(0, gs, (1,), generator=g))
            u = torch.zeros(R, dtype=torch.float64)
            Wg, Wo = W.clone(), W.clone()
            for r in range(R):
                t = int(T0[r, c])
                s = -1 if t == 1 else (1 if t == -1 else (1 if bool(torch.rand(1, generator=g) < 0.5) else -1))
                u[r] = s
                dr = float(self.d[nome][0, r, c // gs])
                Wg[r, c] += rg * s * dr
                Wo[r, c] += ro * s * dr
            self.c[nome], self.u[nome], self.Wg[nome], self.Wo[nome] = c, u, Wg, Wo

    def famiglie(self) -> dict:
        return {self.k[n]: TP._copia_fam(self.fam0[n]) for n in self.fam0}

    def lotto(self, passo: int, kk: int):
        g = torch.Generator().manual_seed(self.seme * 1000003 + passo * 1009 + kk)
        z = {t: torch.randn(self.T, generator=g, dtype=torch.float64) for t in self.termini}
        xs = {}
        for n in self.fam0:
            x = torch.zeros(self.T, 1024, dtype=torch.float64)
            x[:, self.c[n]] = z[self.term[n]]
            xs[n] = x.float()
        return xs, S.Piano(torch.zeros(self.T, 1, dtype=torch.int64), 1)

    def uscite(self, fams, lot, mot, th=None):
        xs, piano = lot
        th = th or self.th
        return {n: S.FamigliaEsperti.apply(xs[n], TP.scala(fams[self.k[n]], th[n]), fams[self.k[n]], piano, RIC, mot)
                for n in self.fam0}

    def perdite(self, lot, ys) -> tuple:
        xs, _ = lot
        gen = {t: 0.0 for t in self.termini}
        obj = {t: 0.0 for t in self.termini}
        for n in self.fam0:
            y = ys[n].to(torch.float64)
            x = xs[n].to(torch.float64)
            gen[self.term[n]] = gen[self.term[n]] + (y - x @ self.Wg[n].T) @ self.u[n]
            obj[self.term[n]] = obj[self.term[n]] + (y - x @ self.Wo[n].T) @ self.u[n]
        return sum((v ** 2).sum() for v in gen.values()), sum((v ** 2).sum() for v in obj.values())

    def raccogli(self, pv, fams, passo: int = 0, n_micro: int = 16, n_curv: int = 2) -> int:
        mot = self.M.MotoreTorchV(pv.stato)
        pv.inizia_passo()
        n_tot = n_micro * self.T
        for kk in range(n_micro):
            pv.microlotto(kk)
            lot = self.lotto(passo, kk)
            th = {n: self.th[n].clone().requires_grad_(True) for n in self.fam0}
            lg, lo = self.perdite(lot, self.uscite(fams, lot, mot, th))
            ((lg + PESO_OBJ * lo) / n_tot).backward()
        if pv.stato.serve_curvatura():
            for kk in range(n_curv):
                pv.curvatura()
                lot = self.lotto(passo, 1000 + kk)
                th = {n: self.th[n].clone().requires_grad_(True) for n in self.fam0}
                ys = self.uscite(fams, lot, mot, th)
                gr = torch.Generator().manual_seed(self.seme * 7919 + passo * 31 + kk)
                eps = {t: torch.randint(0, 2, (self.T, 1), generator=gr).to(torch.float64) * 2 - 1 for t in self.termini}
                vs = [(eps[self.term[n]] * self.u[n][None, :] * math.sqrt(2.0 * (1 + PESO_OBJ) / n_tot)).to(ys[n].dtype)
                      for n in self.fam0]
                torch.autograd.backward([ys[n] for n in self.fam0], vs)
        pv.fine_raccolta()
        return n_tot

    def misura(self, fams, lot, quale: str = "gen") -> float:
        with torch.no_grad():
            lg, lo = self.perdite(lot, self.uscite(fams, lot, S.MotoreTorch()))
        return float((lg if quale == "gen" else lo) / self.T)

    def gradiente(self, pv, fams, lot, n_tot: int, quale: str = "gen"):
        mot = self.M.MotoreTorchV(pv.stato)

        def g():
            th = {n: self.th[n].clone().requires_grad_(True) for n in self.fam0}
            lg, lo = self.perdite(lot, self.uscite(fams, lot, mot, th))
            ((lg if quale == "gen" else lo) / n_tot).backward()
            return True
        return g

    def conta(self, fams) -> dict:
        return {n: int((TP.trit_famiglia(fams[self.k[n]]) != TP.trit_famiglia(self.fam0[n])).sum()) for n in self.fam0}


def passo_v_mosaico(M, cfg, gio: Mosaico = None, due: bool = True, ricentra_grad: bool = False, bop=None, rho=1.0,
                    obj_dosi: bool = True):
    """Un passo V col punto medio sul mosaico, come `_passo_v_dosi` di e2e_allena con --pv-pm-lettere: B (vincolo) =
    gradiente generico sulla sonda A, A (obiettivo) = gradiente obiettivo su M1, conferma dell'obiettivo su M2, dosi
    golose (se accese) decise dall'obiettivo su M1. Ritorna (gio, fams, pv, reg, diag, esito, lotti)."""
    gio = gio or Mosaico(M)
    fams = gio.famiglie()
    pv = M.PassoV(fams, M.ConfigV(accetta="guadagno"), bop=bop)
    n_tot = gio.raccogli(pv, fams)
    L = {k: gio.lotto(1, i) for i, k in enumerate(("A", "B", "C", "M1", "M2"))}
    ma, mb = (lambda: gio.misura(fams, L["A"])), (lambda: gio.misura(fams, L["B"]))
    gr = gio.gradiente(pv, fams, L["A"], n_tot) if ricentra_grad else None
    scale = {gio.k[n]: gio.d[n] for n in gio.fam0}
    gp = {"B": gio.gradiente(pv, fams, L["A"], n_tot)}
    fat = {"B": n_tot / gio.T}
    kw = {"gradienti_pm": gp, "fattori_pm": fat}
    if due:
        gp["A"] = gio.gradiente(pv, fams, L["M1"], n_tot, "obj")
        fat["A"] = n_tot / gio.T
        kw["misura_obiettivo"] = lambda: gio.misura(fams, L["M2"], "obj")
        if obj_dosi and getattr(cfg, "pm_golosa", False):
            kw["misura_obiettivo_dosi"] = lambda: gio.misura(fams, L["M1"], "obj")
    reg, diag, es = pv.seleziona_e_prova_dosi(scale, rho, ma, mb, cfg, gr, **kw)
    return gio, fams, pv, reg, diag, es, L


# ─────────────────────────────────────────── G1 ───────────────────────────────────────────
def prova_g1() -> bool:
    t0 = time.time()
    VEC = _pv_base()
    casi = {
        "pacco_uno": ("pacco", lambda m: m.ConfigDosi(frazioni=(0.25, 1.0), punto_medio=True), None, False, False),
        "pacco_due": ("pacco", lambda m: m.ConfigDosi(frazioni=(0.25, 1.0), punto_medio=True), None, False, True),
        "pacco_ricentro": ("pacco", lambda m: m.ConfigDosi(punto_medio=True, ricentra=True), None, True, False),
        "mosaico_due": ("mosaico", lambda m: m.ConfigDosi(punto_medio=True), None, False, True),
        "mosaico_frazioni_tt2": ("mosaico", lambda m: m.ConfigDosi(
            frazioni=(1 / 4096, 1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0), punto_medio=True, ricentra=True),
            None, True, True),
        "mosaico_vbop": ("mosaico", lambda m: m.ConfigDosi(punto_medio=True, ricentra=True),
                         lambda m: m.ConfigBop(gamma=0.5), True, True),
        "dosi_senza_pm": ("mosaico", lambda m: m.ConfigDosi(), None, False, False),
    }
    uguali = {}
    for nome, (gioc, cfg, bop, ric, due) in casi.items():
        r = {}
        for tag, M in (("prima", VEC), ("nuovo", PV)):
            if gioc == "pacco":
                g = TM.Pacco(M)
                fams, pv, reg, diag, es, _ = TM._passo_v(M, g, cfg(M), bop(M) if bop else None, ric, pm=True, due=due)
            elif cfg(M).punto_medio:
                g, fams, pv, reg, diag, es, _ = passo_v_mosaico(M, cfg(M), due=due, ricentra_grad=ric,
                                                               bop=bop(M) if bop else None)
            else:
                g = Mosaico(M)
                fams = g.famiglie()
                pv = M.PassoV(fams, M.ConfigV(accetta="guadagno"))
                g.raccogli(pv, fams)
                lA, lB = g.lotto(1, 0), g.lotto(1, 1)
                reg, diag, es = pv.seleziona_e_prova_dosi({g.k[n]: g.d[n] for n in g.fam0}, 1.0,
                                                          lambda: g.misura(fams, lA), lambda: g.misura(fams, lB),
                                                          cfg(M), None)
            r[tag] = {"byte": {str(k): f.blocchi.clone() for k, f in fams.items()},
                      "reg": None if reg is None else reg.righe.copy(), "diag": _json(diag), "esito": _json(es),
                      "registri": [x.righe.copy() for x in pv.registri],
                      "bop": TM._stato_bop(pv) if bop else {}}
        a, b = r["prima"], r["nuovo"]
        uguali[nome] = {
            "byte": all(torch.equal(a["byte"][k], b["byte"][k]) for k in a["byte"]),
            "registro": (a["reg"] is None and b["reg"] is None) or (a["reg"] is not None and b["reg"] is not None
                                                                    and np.array_equal(a["reg"], b["reg"])),
            "registri": len(a["registri"]) == len(b["registri"]) and all(
                np.array_equal(x, y) for x, y in zip(a["registri"], b["registri"])),
            "diag": a["diag"] == b["diag"], "esito": a["esito"] == b["esito"],
            "bop": all(torch.equal(a["bop"][k][c], b["bop"][k][c]) for k in a["bop"] for c in a["bop"][k]),
            "salti_tenuti": json.loads(b["esito"])["salti_tenuti"]}
    ok = all(all(v2 for k2, v2 in v.items() if k2 != "salti_tenuti") for v in uguali.values())
    return registra("G1_spento_al_bit_modulo", ok, uguali=uguali, base=BASE, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── G2 ───────────────────────────────────────────
def prova_g2() -> bool:
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
        d = DIR / f"g2_{nome}"
        TI._pulisci(d)
        base = TI._ca_integrata(pv_sonda_dosi=True, pv_ricentra=True, pv_dosi_frazioni=(1 / 16, 1 / 4, 1.0),
                                pv_punto_medio=True, pv_pm_max=4)
        ca = mod.ConfigAllena(**{k: v for k, v in base.__dict__.items() if k in mod.ConfigAllena.__dataclass_fields__})
        st = TI._studente_int()
        z = mod.allena(st, src, d, ca, WS, WS, log=TI.muto)
        ris[nome] = (st, z, mod.carica_ripresa(d, "cpu"))
    (sp, zp, rp), (sn, zn, rn) = ris["prima"], ris["nuovo"]
    a_, b_ = TI._stato_confrontabile(zn), TI._stato_confrontabile(zp)
    nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    a_.pop("versione", None)
    b_.pop("versione", None)
    u = {"theta": TI._uguali_lista(sn.parametri(), sp.parametri()),
         "f32": TI._uguali_lista(sn.parametri_f32(), sp.parametri_f32()),
         "lora": TI._uguali_lora_ga(sn, sp), "byte": TI._uguali_lista(TI._byte(sn), TI._byte(sp)),
         "opt": TI._uguali_stato_opt(rn["opt"], rp["opt"]),
         "stato": _json(a_) == _json(b_), "registri_sha": zn["pv"]["sha"] == zp["pv"]["sha"],
         "campi_nuovi": set(nuovi) == {"pv_pm_golosa"} and nuovi == {k: A._CONFIG_DEFAULT[k] for k in nuovi}}
    pv = [(x["pv"]["salti"], x["pv"]["tenuto"], x["pv"].get("salti_tenuti")) for x in zn["storia"] if x.get("pv")]
    return registra("G2_spento_al_bit_ciclo", all(u.values()), uguali=u, campi_nuovi=nuovi, pv=pv, base=BASE,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── G3 ───────────────────────────────────────────
FRAZ_TT2 = (1 / 4096, 1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0)


def esegui_g3(M) -> dict:
    """Oggi (pm_golosa spento, frazioni di TT2) contro golosa, stesso mosaico e stessi lotti, modulo M."""
    out = {}
    for modo in ("oggi", "golosa"):
        cfg = M.ConfigDosi(frazioni=FRAZ_TT2, punto_medio=True, **({"pm_golosa": True} if modo == "golosa" else {}))
        gio, fams, pv, reg, diag, es, L = passo_v_mosaico(M, cfg)
        o_M2, o_C, g_B, g_A = (gio.misura(fams, L["M2"], "obj"), gio.misura(fams, L["C"], "obj"),
                               gio.misura(fams, L["B"]), gio.misura(fams, L["A"]))
        if reg is not None:
            M.annulla(fams, reg)
        o_M2_0, o_C_0, g_B_0 = gio.misura(fams, L["M2"], "obj"), gio.misura(fams, L["C"], "obj"), gio.misura(fams, L["B"])
        if reg is not None:
            M.rigioca(fams, [reg])
        pm = es.get("punto_medio", {})
        out[modo] = {"tenuto": es["tenuto"], "salti_scelti": es["salti"], "salti_tenuti": es["salti_tenuti"],
                     "applicati": gio.conta(fams), "scelte": pm.get("scelte"), "scelte_previste": pm.get("scelte_previste"),
                     "dosi": [{k: x.get(k) for k in ("salti", "sonda_a", "obiettivo", "tenuta")} for x in es["dosi"]],
                     "obj_M2": (o_M2_0, o_M2), "obj_C": (o_C_0, o_C), "gen_B": (g_B_0, g_B), "gen_A": (es["sonda_prima_a"], g_A),
                     "conferma_obj": pm.get("obiettivo")}
    return out


def giudica_g3(out: dict) -> dict:
    o, g = out["oggi"], out["golosa"]
    return {
        # premesse del giocattolo (che il caso di TT2 sia riprodotto): oggi tiene la sola prima primitiva
        "premessa_oggi_solo_prima": o["tenuto"] and o["salti_tenuti"] > 0 and len(o["scelte"] or []) >= 3
                                    and sum(1 for v in o["applicati"].values() if v) == 1,
        "golosa_tenuto": bool(g["tenuto"]),
        "golosa_piu_salti": g["salti_tenuti"] > o["salti_tenuti"],
        "golosa_M2_non_peggio": g["obj_M2"][1] <= o["obj_M2"][1] and g["obj_M2"][1] < g["obj_M2"][0],
        "golosa_C_non_peggio": g["obj_C"][1] <= o["obj_C"][1] and g["obj_C"][1] < g["obj_C"][0],
        "golosa_B_non_sale": g["gen_B"][1] <= g["gen_B"][0],
        "golosa_A_vincolo": g["gen_A"][1] <= g["gen_A"][0],
        "golosa_senza_A2_e_K": g["applicati"]["K"] == 0 and min(g["applicati"]["A1"], g["applicati"]["A2"]) == 0,
    }


def prova_g3() -> bool:
    t0 = time.time()
    out = esegui_g3(PV)
    v = giudica_g3(out)
    return registra("G3_giocattolo_golosa_piu_salti_lettere_non_peggio", all(v.values()), verdetti=v, esiti=out,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── G4 ───────────────────────────────────────────
MUTAZIONI = {
    "rifiutata_non_annullata": ("                else:\n                    annulla(self.fam, inc)\n                pendente = None",
                                "                else:\n                    pass\n                pendente = None"),
    "senza_vincolo_A": ("tiene = bool(mo < cur_o) and (not separato or bool(ma <= a0))", "tiene = bool(mo < cur_o)"),
    "tenuta_sempre": ("tiene = bool(mo < cur_o) and (not separato or bool(ma <= a0))", "tiene = True"),
}


def prova_g4() -> bool:
    t0 = time.time()
    testo = (QUI / "e2e_pv_codici.py").read_text()
    ris = {}
    for nome, (da, a) in MUTAZIONI.items():
        if testo.count(da) != 1:
            ris[nome] = {"applicata": False}
            continue
        M = _modulo_da_testo(testo.replace(da, a), f"e2e_pv_codici_mut_{nome}")
        try:
            v = giudica_g3(esegui_g3(M))
            ris[nome] = {"applicata": True, "rossa": not all(v.values()), "verdetti": v}
        except Exception as ex:  # noqa: BLE001
            ris[nome] = {"applicata": True, "rossa": True, "eccezione": repr(ex)}
    ok = all(r.get("applicata") and r.get("rossa") for r in ris.values())
    return registra("G4_mutazioni_rendono_rossa_G3", ok, mutazioni=ris, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── G5 ───────────────────────────────────────────
def prova_g5() -> bool:
    t = lambda *x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    casi = {
        "uno_come_scegli": PV.ordina_primitive(t(-1.0, 2.0, -3.0, math.nan, 0.0)) == PV.scegli_primitive(
            t(-1.0, 2.0, -3.0, math.nan, 0.0)) == [2, 0],
        # sicure {0, 2} per s_A -> [2, 0]; costose 1 (-5/1), 3 (-4/2), 5 (-6/10): TUTTE, senza bilancio previsto
        "due_senza_bilancio": PV.ordina_primitive(t(-1.0, -5.0, -2.0, -4.0, 1.0, -6.0),
                                                  t(-3.0, 1.0, 0.0, 2.0, -9.0, 10.0)) == [2, 0, 1, 3, 5],
        "scegli_invariata": PV.scegli_primitive(t(-1.0, -5.0, -2.0, -4.0, 1.0, -6.0),
                                                t(-3.0, 1.0, 0.0, 2.0, -9.0, 10.0)) == [2, 0, 1, 3],
    }
    try:
        PV.ConfigDosi(pm_golosa=True)
        casi["configdosi_rifiuta"] = False
    except ValueError:
        casi["configdosi_rifiuta"] = True
    import e2e_allena as A
    try:
        A.ConfigAllena(pv_codici=True, pv_sonda_dosi=True, pv_pm_golosa=True)
        casi["configallena_rifiuta"] = False
    except ValueError:
        casi["configallena_rifiuta"] = True
    casi["default_spento"] = (PV.ConfigDosi().pm_golosa is False and A.ConfigAllena().pv_pm_golosa is False)
    return registra("G5_ordine_e_configurazioni", all(casi.values()), casi=casi)


# ─────────────────────────────────────────── G6 ───────────────────────────────────────────
def prova_g6() -> bool:
    """CICLO ACCESO: `allena` con --pv-punto-medio --pv-pm-golosa sullo studente vero minuscolo (senza righe mc:
    obiettivo = sonda A): gira, la storia ha le dosi golose, il registro ha la riga «dosi golose», ripresa 2 + 1 al bit."""
    import e2e_allena as A
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    kw = dict(pv_sonda_dosi=True, pv_ricentra=True, pv_dosi_frazioni=(1 / 16, 1 / 4, 1.0), pv_punto_medio=True,
              pv_pm_max=4, pv_pm_golosa=True)
    d = DIR / "g6_tutto"
    TI._pulisci(d)
    st = TI._studente_int()
    lg: list = []
    z = A.allena(st, src, d, TI._ca_integrata(**kw), WS, WS, log=lg.append)
    d2 = DIR / "g6_ripresa"
    TI._pulisci(d2)
    A.allena(TI._studente_int(), src, d2, TI._ca_integrata(**kw), WS, WS, ferma_a=1, log=TI.muto)
    st3 = TI._studente_int()
    z3 = A.allena(st3, src, d2, TI._ca_integrata(**kw), WS, WS, riprendi=True, log=TI.muto)
    voci = [x["pv"] for x in z["storia"] if x.get("pv")]
    golose = [v for v in voci if v.get("dosi") and all("tenuta" in x for x in v["dosi"])]
    righe = [str(m) for m in lg if "dosi golose" in str(m)]
    ripresa = (TI._uguali_lista(st3.parametri(), st.parametri()) and TI._uguali_lista(TI._byte(st3), TI._byte(st))
               and z3["pv"]["sha"] == z["pv"]["sha"] and z3["passo"] == z["passo"])
    ok = bool(golose) and bool(righe) and ripresa
    return registra("G6_ciclo_acceso_golosa", ok, righe_registro=[r[:500] for r in righe][:4],
                    voci=[_json({k: v.get(k) for k in ("salti", "salti_tenuti", "tenuto", "dosi")})[:1200] for v in voci],
                    ripresa_al_bit=ripresa, secondi=round(time.time() - t0, 1))


PROVE = {"G1": prova_g1, "G2": prova_g2, "G3": prova_g3, "G4": prova_g4, "G5": prova_g5, "G6": prova_g6}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    for v in ("E2E_PV_DIR", "E2E_VI_DIR", "E2E_ALLENA_PROVE"):
        Path(os.environ[v]).mkdir(parents=True, exist_ok=True)
    scelte = [a for a in sys.argv[1:] if a in PROVE] or list(PROVE)
    esito_p = DIR / "esito_test_pm_golosa.json"
    ok = True
    for nome in scelte:
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
    sys.exit(main())
