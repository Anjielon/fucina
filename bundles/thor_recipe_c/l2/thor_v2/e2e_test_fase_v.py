#!/usr/bin/env python3
"""Prove del ramo tt5-fase-v (2/10/2026, da d2958b5): cure V1-V3 della fase V e il cancello. SOLO CPU, niente GPU.

Le cure (FASE_V.md) promettono le STESSE decisioni con meno lavoro; queste prove lo verificano AL BIT su input registrati
(stessi semi, stessi lotti), e verificano che il risparmio ci sia davvero (contatori), e che una rottura voluta
(mutazione) le faccia diventare rosse.

  F1  MOSAICO (giocattolo di e2e_test_pm_golosa: 4 famiglie su 4 strati, due funzionali, punto medio + golosa +
      ricentro, con e senza V-Bop): spento contro acceso (V1 + V2 + V3) -> registro tenuto, esito (voci pigre allineate),
      diagnostica, stato V-Bop, byte, S dei buffer UGUALI AL BIT; l'acceso salta davvero dL/dd (V1), misura meno lettere
      M1 (V3), prezza le primitive in ordine di strato decrescente (V2).
  F2  MUTAZIONI del mosaico che devono essere ROSSE: (a) dL/dd saltato anche sulle famiglie CON salti; (b) golosa che
      salta l'obiettivo quando la sonda A e' DENTRO il vincolo.
  F3  STRATI VERI di R1 (0-3, byte veri, 8 esperti, densi col secondo piano, motore torch) col riuso acceso: la sequenza
      dei gradienti della fase V (g0 + un g1 per primitiva + ricentro) con V1 spento/acceso -> S e misure uguali al bit;
      ricette B3_ind32, B3_ind2 (indietro diviso, e2e_indietro_hilo) e B3_ind2 coi densi piegati x2 (e2e_densi_piegati).
      Mutazione (a) rossa anche qui.
  F4  CICLO `allena` (studente minuscolo di e2e_test_integrata, V-Bop + dosi + punto medio + golosa + ricentro + riuso):
      spento contro --pv-dd-solo-salti --pv-pm-ordine-strati --pv-golosa-pigra --pv-tempi-v -> theta, F32, LoRA, byte,
      stato di Adam/Sophia, storia (tolti orologio e contatori), sha dei registri UGUALI.
  F5  CANCELLO in modo corsa (E2E_CANCELLO_V) nel ciclo F4: verde -> la corsa e' identica a quella spenta; con la
      mutazione (a) -> ROSSO, stato dello spento rimesso, cure spente: la corsa resta identica a quella spenta.

Uso (mai la GPU; al massimo 4 fili):
    systemd-run --user --scope --quiet -p CPUQuota=400% -p MemoryMax=8G nice -n 19 ionice -c3 \\
      env OMP_NUM_THREADS=4 @@ROOT:venv@@/bin/python3 e2e_test_fase_v.py [F1 F2 F3 F4 F5]
    esito: $E2E_FASE_V_DIR/esito_test_fase_v.json (default @@ROOT:depot@@/e2e/prove_fase_v)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
_D = os.environ.setdefault("E2E_FASE_V_DIR", "@@ROOT:depot@@/e2e/prove_fase_v")
os.environ.setdefault("E2E_PM_DIR", str(Path(_D) / "pm"))
os.environ.setdefault("E2E_VI_DIR", str(Path(_D) / "vi"))
os.environ.setdefault("E2E_PV_DIR", str(Path(_D) / "pv"))
os.environ.setdefault("E2E_ALLENA_PROVE", str(Path(_D) / "allena"))
os.environ.setdefault("E2E_LORA_DIR", str(Path(_D) / "lora"))
os.environ.setdefault("E2E_CURE_DIR", "@@ROOT:lavoro@@/tt3_cure_prove")     # giocattolo R1 gia' costruito (sola lettura)
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_riuso_prove as RP  # noqa: E402

DIR = Path(_D)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'OK' if ok else 'ROSSO'} {json.dumps(dati, ensure_ascii=False, default=str)[:2500]}", flush=True)
    return bool(ok)


def _j(x) -> str:
    return json.dumps(x, sort_keys=True, default=repr)


@contextmanager
def _patch(obj, nome: str, nuovo):
    vecchio = getattr(obj, nome)
    setattr(obj, nome, nuovo)
    try:
        yield
    finally:
        setattr(obj, nome, vecchio)


def _mut_dd_tutti(mot, fam) -> bool:
    """Mutazione (a): dL/dd saltato anche sulle famiglie CON salti preparati (e quindi niente S): deve essere rossa."""
    st = getattr(mot, "stato", None)
    return bool(st is not None and getattr(st, "salta_dd_superflui", False) and getattr(st, "modo", 0) == 2
                and getattr(st, "famiglie_ricentro", None) is not None)


# ═══════════════════════════════════ F1 / F2: mosaico ═══════════════════════════════════
def _mosaico_passo(cfg, dd: bool, bop=None, conta=None):
    """`e2e_test_pm_golosa.passo_v_mosaico` con lo StatoV in --pv-dd-solo-salti (`dd`), contatori delle misure delle
    lettere M1 e ordine dei rigiochi; ritorna le grandezze da confrontare."""
    import e2e_test_pm_golosa as TG
    gio = TG.Mosaico(PV)
    fams = gio.famiglie()
    pv = PV.PassoV(fams, PV.ConfigV(accetta="guadagno"), bop=bop)
    n_tot = gio.raccogli(pv, fams)
    pv.stato.salta_dd_superflui = dd
    L = {k: gio.lotto(1, i) for i, k in enumerate(("A", "B", "C", "M1", "M2"))}
    conta = conta if conta is not None else {}
    conta.setdefault("lettere_M1", 0)
    conta.setdefault("strati_rigiocati", [])

    def m1():
        conta["lettere_M1"] += 1
        return gio.misura(fams, L["M1"], "obj")
    ma, mb = (lambda: gio.misura(fams, L["A"])), (lambda: gio.misura(fams, L["B"]))
    gp = {"B": gio.gradiente(pv, fams, L["A"], n_tot), "A": gio.gradiente(pv, fams, L["M1"], n_tot, "obj")}
    kw = {"gradienti_pm": gp, "fattori_pm": {"B": n_tot / gio.T, "A": n_tot / gio.T},
          "misura_obiettivo": lambda: gio.misura(fams, L["M2"], "obj"), "misura_obiettivo_dosi": m1}
    scale = {gio.k[n]: gio.d[n] for n in gio.fam0}
    rig0 = PV.rigioca

    def rig(famiglie, registri):
        for r in registri:
            if len(r):
                conta["strati_rigiocati"].append(int(r.righe["L"].min()))
        return rig0(famiglie, registri)
    PV.CONTI_DD["saltati"] = 0
    with _patch(PV, "rigioca", rig):
        reg, diag, es = pv.seleziona_e_prova_dosi(scale, 1.0, ma, mb, cfg, gio.gradiente(pv, fams, L["A"], n_tot), **kw)
    conta["dd_saltati"] = PV.CONTI_DD["saltati"]
    S_ = {f"{k[0]}.{k[1]}": b.S.clone() for k, b in pv.stato.buffer_.items()}
    bop_ = {f"{k[0]}.{k[1]}": {n: getattr(z, n).clone() for n in ("cand", "m", "r", "H", "eta", "sm", "sr", "sH")}
            for k, z in pv.stato.bop_.items()}
    return {"reg": b"" if reg is None else reg.righe.tobytes(), "diag": diag, "esito": es, "S": S_, "bop": bop_,
            "byte": {f"{k[0]}.{k[1]}": f.blocchi.clone() for k, f in fams.items()}, "conta": conta}


def _uguali_mosaico(a, b) -> dict:
    import e2e_cancello_fase_v as CV
    return {"registro": a["reg"] == b["reg"], "diag": _j(a["diag"]) == _j(b["diag"]),
            "esito": _j(CV.allinea_pigra(a["esito"], b["esito"])) == _j(b["esito"]),
            "S": a["S"].keys() == b["S"].keys() and all(torch.equal(a["S"][k], b["S"][k]) for k in a["S"]),
            "bop": a["bop"].keys() == b["bop"].keys() and all(
                torch.equal(a["bop"][k][n], b["bop"][k][n]) for k in a["bop"] for n in a["bop"][k]),
            "byte": all(torch.equal(a["byte"][k], b["byte"][k]) for k in a["byte"])}


def _cfg(on: bool, **kw):
    return PV.ConfigDosi(punto_medio=True, pm_golosa=True, ricentra=True, pm_ordine_strati=on, golosa_pigra=on, **kw)


def prova_f1() -> bool:
    t0 = time.time()
    out, ok = {}, True
    for nome, bop in (("senza_vbop", None), ("vbop", PV.ConfigBop(gamma=0.5))):
        a = _mosaico_passo(_cfg(False), dd=False, bop=bop)
        b = _mosaico_passo(_cfg(True), dd=True, bop=bop)
        u = _uguali_mosaico(a, b)
        lavoro = {"dd_saltati": (a["conta"]["dd_saltati"], b["conta"]["dd_saltati"]),
                  "lettere_M1": (a["conta"]["lettere_M1"], b["conta"]["lettere_M1"]),
                  "strati_rigiocati_pm": (a["conta"]["strati_rigiocati"], b["conta"]["strati_rigiocati"]),
                  "salti_tenuti": a["esito"].get("salti_tenuti"), "dosi_tenute": [x.get("tenuta") for x in
                                                                                    a["esito"]["dosi"]]}
        risparmio = (b["conta"]["dd_saltati"] > 0 and a["conta"]["dd_saltati"] == 0
                     and b["conta"]["lettere_M1"] < a["conta"]["lettere_M1"])
        out[nome] = {"uguali": u, "lavoro": lavoro, "risparmio": risparmio}
        ok = ok and all(u.values()) and risparmio
    return registra("F1_mosaico_acceso_uguale_spento", ok, **out, secondi=round(time.time() - t0, 1))


def prova_f2() -> bool:
    t0 = time.time()
    a = _mosaico_passo(_cfg(False), dd=False)
    with _patch(PV, "dd_superfluo", _mut_dd_tutti):
        b = _mosaico_passo(_cfg(False), dd=True)
    ua = _uguali_mosaico(a, b)
    vecchia = PV.PassoV._dosi_golose

    def golosa_rotta(self, ordine, conti, righe_o, cum, n, misura_a, misura_b, misura_obj, esito, pigra=False):
        """Mutazione (b): l'obiettivo si «salta» (primitiva rifiutata) quando la sonda A e' DENTRO il vincolo
        (condizione rovesciata rispetto a V3)."""
        a0, ult = esito["sonda_prima_a"], {}

        def ma_():
            ult["a"] = misura_a()
            return ult["a"]

        def mo_():
            if "a" in ult and ult["a"] <= a0:
                return float("inf")
            return misura_obj()
        return vecchia(self, ordine, conti, righe_o, cum, n, ma_, misura_b, mo_, esito, pigra=False)
    with _patch(PV.PassoV, "_dosi_golose", golosa_rotta):
        c = _mosaico_passo(_cfg(True), dd=False)
    ub = _uguali_mosaico(a, c)
    ok = not all(ua.values()) and not all(ub.values())
    return registra("F2_mutazioni_rosse", ok, mutazione_dd=ua, mutazione_golosa=ub, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ F3: strati veri di R1 ═══════════════════════════════════
def _sequenza_r1(gguf, ricetta: str, piegati: str, dd: bool, mut=None) -> dict:
    import e2e_densi_piegati as DP
    DP.imposta(modo=piegati)
    st = S.Studente(gguf=gguf, strati=[0, 1, 2, 3], ricetta=ricetta, motore_nome="torch", device="cpu",
                    esperti=slice(0, 8), log=lambda *_: None, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            g = torch.Generator().manual_seed(1000 * L + sum(ord(c) for c in n))
            p.copy_(0.01 * torch.randn(p.shape, generator=g))
    for L in st.strati:
        for f in st.esperti(L).fam.values():
            PV.famiglia_scrivibile(f)
    stato = PV.StatoV()
    fams = PV.installa(st, stato, "torch")
    stato.inizia_passo()
    stato.salta_dd_superflui = dd
    g = torch.Generator().manual_seed(2910)
    B, T, V = 2, 12, 248320
    lotti = {nome: (torch.randint(0, V, (B, T), generator=g), None, torch.randn(B * T, 2048, generator=g))
             for nome in ("A", "M1")}

    def misura(nome):
        ids, iz, G = lotti[nome]
        with torch.no_grad():
            h = st.nascosto(ids, inizi=iz)
        return float((h.double() * G.double()).sum())

    def primitiva(k, n, seme):
        L, nome = k
        f = fams[k]
        gg = torch.Generator().manual_seed(seme)
        righe = np.zeros(n, dtype=PV.REG_DTYPE)
        usati, i = set(), 0
        while i < n:
            e = int(torch.randint(0, f.E, (1,), generator=gg))
            r = int(torch.randint(0, f.R, (1,), generator=gg))
            c = int(torch.randint(0, f.C, (1,), generator=gg))
            if (e, r, c // f.gs) in usati:                   # un salto per gruppo (come la selezione)
                continue
            t = int(S.trit_esperto(f, e)[r, c])
            tn = t - 1 if t == 1 else (t + 1 if t == -1 else (1 if int(torch.randint(0, 2, (1,), generator=gg)) else -1))
            usati.add((e, r, c // f.gs))
            righe[i] = (L, PV.FAMIGLIE_IDX[nome], e, r, c, t, tn)
            i += 1
        return PV.Registro(righe)

    def gradiente(nome, regs):
        ids, iz, G = lotti[nome]
        salti = {}
        for reg in regs:
            for (L, fi) in reg.chiavi():
                k = (L, PV.NOMI_IDX[fi])
                rr = reg.di(L, fi)
                gs = fams[k].gs
                c = torch.from_numpy(rr["c"].astype(np.int64))
                salti[k] = (torch.from_numpy(rr["e"].astype(np.int64)), torch.from_numpy(rr["r"].astype(np.int64)),
                            c // gs, (c % gs).to(torch.uint8))
        salvati = [p.grad for p in st.parametri()]
        for p in st.parametri():
            p.grad = None
        stato.prepara_ricentro(salti)
        try:
            s = st.avanti(ids, inizi=iz)
            xL = s["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                (st.esce(xL) * G).sum().backward()
            st.indietro(s, xL.grad)
        finally:
            stato.fine_raccolta()
            for p, gp in zip(st.parametri(), salvati):
                p.grad = gp
        return {f"{k[0]}.{k[1]}": stato.buffer(fams[k]).S[salti[k][0], salti[k][1], salti[k][2]].clone() for k in salti}

    P = [((3, "attn_q2"), 6), ((1, "ssm_out"), 6), ((2, "down"), 12), ((0, "attn_qkv2"), 6), ((3, "gate"), 12)]
    regs = [primitiva(k, n, 91 + i) for i, (k, n) in enumerate(P)]
    mis, grads = [], {}
    PV.CONTI_DD["saltati"] = 0
    with RP.RiusoProve(st, stato, max_lotti=2):
        with (_patch(PV, "dd_superfluo", mut) if mut is not None else _patch(PV, "CONTI_DD", PV.CONTI_DD)):
            mis.append(misura("A"))
            grads["g0_A"] = gradiente("A", regs)
            for i, reg in enumerate(regs):
                PV.rigioca(fams, [reg])
                grads[f"g1_A_{i}"] = gradiente("A", [reg])
                mis.append(misura("M1"))
                PV.annulla(fams, reg)
            tenuti = [regs[0], regs[2], regs[3]]
            for reg in tenuti:
                PV.rigioca(fams, [reg])
            grads["ricentro"] = gradiente("A", tenuti)
            mis.append(misura("A"))
            for reg in reversed(tenuti):
                PV.annulla(fams, reg)
    DP.imposta(modo="")
    return {"misure": mis, "grad": grads, "dd_saltati": PV.CONTI_DD["saltati"],
            "byte": {f"{k[0]}.{k[1]}": f.blocchi.clone() for k, f in fams.items()}}


def _uguali_r1(a, b) -> dict:
    return {"misure": a["misure"] == b["misure"],
            "grad": a["grad"].keys() == b["grad"].keys() and all(
                a["grad"][k].keys() == b["grad"][k].keys() and all(torch.equal(a["grad"][k][j], b["grad"][k][j])
                                                                   for j in a["grad"][k]) for k in a["grad"]),
            "grad_non_nulli": all(any(bool((v != 0).any()) for v in d.values()) for d in a["grad"].values()),
            "byte": all(torch.equal(a["byte"][k], b["byte"][k]) for k in a["byte"])}


def prova_f3() -> bool:
    import e2e_test_riuso_prove as TR
    t0 = time.time()
    gg = TR.giocattolo()
    out, ok = {}, True
    for ric, pieg in (("B3_ind32", ""), ("B3_ind2", ""), ("B3_ind2", "x2")):
        a = _sequenza_r1(gg, ric, pieg, dd=False)
        b = _sequenza_r1(gg, ric, pieg, dd=True)
        u = _uguali_r1(a, b)
        out[f"{ric}{'_piegati_' + pieg if pieg else ''}"] = {**u, "dd_saltati": (a["dd_saltati"], b["dd_saltati"])}
        ok = ok and all(u.values()) and b["dd_saltati"] > 0 and a["dd_saltati"] == 0
        print(f"  F3 {ric} {pieg or '-'}: {u} dd {a['dd_saltati']} -> {b['dd_saltati']} ({time.time() - t0:.0f} s)",
              flush=True)
    a = _sequenza_r1(gg, "B3_ind2", "", dd=False)
    m = _sequenza_r1(gg, "B3_ind2", "", dd=True, mut=_mut_dd_tutti)
    um = _uguali_r1(a, m)
    out["mutazione_dd_tutti"] = um
    ok = ok and not um["grad"]
    return registra("F3_strati_veri_R1_dd_solo_salti", ok, **out, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ F4 / F5: ciclo allena ═══════════════════════════════════
_KW_CICLO = dict(pv_sonda_dosi=True, pv_ricentra=True, pv_dosi_frazioni=(1 / 16, 1 / 4, 1.0), pv_punto_medio=True,
                 pv_pm_max=4, pv_pm_golosa=True, pv_riuso=True)
_CURE_ON = dict(pv_dd_solo_salti=True, pv_pm_ordine_strati=True, pv_golosa_pigra=True, pv_tempi_v=True)


def _ciclo(nome: str, **kw):
    import e2e_allena as A
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    d = DIR / f"ciclo_{nome}"
    TI._pulisci(d)
    st = TI._studente_int()
    z = A.allena(st, src, d, TI._ca_integrata(**_KW_CICLO, **kw), WS, WS, log=TI.muto)
    return st, z, A.carica_ripresa(d, "cpu"), d


def _confronta_cicli(x, y) -> dict:
    import e2e_test_integrata as TI
    import e2e_cancello_fase_v as CV
    (sa, za, ra, _), (sb, zb, rb, _) = x, y
    a_, b_ = TI._stato_confrontabile(za), TI._stato_confrontabile(zb)
    for s_ in (a_, b_):
        for c in ("pv_dd_solo_salti", "pv_pm_ordine_strati", "pv_golosa_pigra", "pv_tempi_v",
                  "scritture_sosta_dispositivo"):
            s_["config"].pop(c, None)
        for v in s_.get("storia", []):
            if v.get("pv"):
                for c in ("tempi_v", "riuso", "dd_saltati"):
                    v["pv"].pop(c, None)
    for va, vb in zip(a_.get("storia", []), b_.get("storia", [])):
        if va.get("pv") and vb.get("pv"):
            va["pv"] = CV.allinea_pigra(va["pv"], vb["pv"])
    return {"theta": TI._uguali_lista(sb.parametri(), sa.parametri()),
            "f32": TI._uguali_lista(sb.parametri_f32(), sa.parametri_f32()),
            "lora": TI._uguali_lora_ga(sb, sa), "byte": TI._uguali_lista(TI._byte(sb), TI._byte(sa)),
            "opt": TI._uguali_stato_opt(rb["opt"], ra["opt"]),
            "stato": _j(a_) == _j(b_), "registri_sha": zb["pv"]["sha"] == za["pv"]["sha"]}


def prova_f4() -> bool:
    t0 = time.time()
    a = _ciclo("spento")
    b = _ciclo("acceso", **_CURE_ON)
    u = _confronta_cicli(a, b)
    pv_b = [x["pv"] for x in b[1]["storia"] if x.get("pv")]
    lavoro = {"dd_saltati": [x.get("dd_saltati") for x in pv_b], "tempi_v": [bool(x.get("tempi_v")) for x in pv_b],
              "decisioni": [(x.get("salti"), x.get("tenuto"), x.get("salti_tenuti")) for x in pv_b]}
    ok = all(u.values()) and any((x or 0) > 0 for x in lavoro["dd_saltati"]) and all(lavoro["tempi_v"])
    return registra("F4_ciclo_acceso_uguale_spento", ok, uguali=u, lavoro=lavoro, secondi=round(time.time() - t0, 1))


def prova_f5() -> bool:
    import e2e_cancello_fase_v as CV
    t0 = time.time()
    a = _ciclo("spento_f5")
    out, ok = {}, True
    for nome, mut in (("verde", None), ("rosso_mutazione_dd", _mut_dd_tutti)):
        js = DIR / f"cancello_{nome}.json"
        js.unlink(missing_ok=True)
        os.environ["E2E_CANCELLO_V"] = str(js)
        os.environ["E2E_CANCELLO_V_MODO"] = "corsa"
        CV._FATTO.update(scattato=False, cure_spente=False)
        try:
            if mut is not None:
                with _patch(PV, "dd_superfluo", mut):
                    b = _ciclo(f"cancello_{nome}")
            else:
                b = _ciclo(f"cancello_{nome}")
        finally:
            os.environ.pop("E2E_CANCELLO_V", None)
            os.environ.pop("E2E_CANCELLO_V_MODO", None)
        e = json.loads(js.read_text()) if js.exists() else {}
        u = _confronta_cicli(a, b)
        atteso = (nome == "verde")
        out[nome] = {"uguali_alla_corsa_spenta": u, "cancello_ok": e.get("ok"), "uguali_cancello": e.get("uguali"),
                     "cure_spente_dopo": CV._FATTO["cure_spente"]}
        ok = ok and all(u.values()) and e.get("ok") is atteso and CV._FATTO["cure_spente"] is (not atteso)
    CV._FATTO.update(scattato=False, cure_spente=False)
    return registra("F5_cancello_corsa", ok, **out, secondi=round(time.time() - t0, 1))


_FIGLIO_F0 = r'''
import json, os, sys, torch
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
import e2e_allena as A, e2e_test_integrata as TI, e2e_test_lora as TL
kw = json.loads(sys.argv[3])
kw["pv_dosi_frazioni"] = tuple(kw["pv_dosi_frazioni"])
src, WS = TL.sorgente_lora(n_train=12, n_held=4)
src = TI.con_domini(src)
d = sys.argv[2] + "_dir"
TI._pulisci(d)
st = TI._studente_int()
z = A.allena(st, src, __import__("pathlib").Path(d), TI._ca_integrata(**kw), WS, WS, log=TI.muto)
r = A.carica_ripresa(__import__("pathlib").Path(d), "cpu")
torch.save({"theta": [p.detach().clone() for p in st.parametri()], "f32": [p.detach().clone() for p in st.parametri_f32()],
            "byte": TI._byte(st), "opt": r["opt"], "stato": json.dumps(TI._stato_confrontabile(z), sort_keys=True,
            default=repr), "sha": z["pv"]["sha"]}, sys.argv[2])
'''


def _senza_mtime(z: dict) -> dict:
    """Il file dello studente minuscolo si riscrive a ogni processo: la sua mtime non e' una differenza di calcolo."""
    if isinstance(z.get("studente"), dict):
        z["studente"].pop("mtime", None)
    return z


def prova_f0() -> bool:
    """F0 SPENTO == d2958b5 AL BIT (il codice della corsa viva, in un processo figlio, mai scritto: niente .pyc): il ciclo
    `allena` di F4 col ramo a cure spente e col codice congelato della corsa."""
    import subprocess
    import e2e_test_integrata as TI
    t0 = time.time()
    base = os.getenv("E2E_FASE_V_BASE", "@@ROOT:depot@@/e2e/codice/TT5-ottimizzato/thor_v2")
    f = DIR / "f0_figlio.py"
    f.write_text(_FIGLIO_F0)
    u = DIR / "f0_base.pt"
    subprocess.run([sys.executable, str(f), base, str(u), json.dumps(_KW_CICLO)], check=True, cwd=str(DIR),
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    a = torch.load(u, weights_only=False)
    sb, zb, rb, _ = _ciclo("f0_ramo")
    b_ = TI._stato_confrontabile(zb)
    for c in ("pv_dd_solo_salti", "pv_pm_ordine_strati", "pv_golosa_pigra", "pv_tempi_v", "scritture_sosta_dispositivo"):
        b_["config"].pop(c, None)
    u_ = {"theta": TI._uguali_lista(a["theta"], [p.detach() for p in sb.parametri()]),
          "f32": TI._uguali_lista(a["f32"], [p.detach() for p in sb.parametri_f32()]),
          "byte": TI._uguali_lista(a["byte"], TI._byte(sb)), "opt": TI._uguali_stato_opt(a["opt"], rb["opt"]),
          "stato": _senza_mtime(json.loads(a["stato"])) == _senza_mtime(json.loads(json.dumps(b_, sort_keys=True,
                                                                                               default=repr))), "registri_sha": a["sha"] == zb["pv"]["sha"]}
    return registra("F0_spento_uguale_d2958b5", all(u_.values()), base=base, uguali=u_, secondi=round(time.time() - t0, 1))


def _sha_file_ripresa(d: Path) -> dict:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.glob("*.pt"))
            if f.name.startswith(("ripresa_", "bop_pv_"))}


def prova_s1() -> bool:
    """S1 --scritture-sosta-dispositivo nel ciclo `allena` (V-Bop): asincrona di oggi contro sosta (forzata: sulla CPU la
    sosta e' un clone CPU, si prova tutta la strada fino al file) e contro sosta NON possibile (ripiego alla copia CPU):
    stessi file ripresa_*.pt e bop_pv_*.pt AL BIT. L'etichetta «cpu» delle storage del dispositivo la prova
    e2e_prova_sosta.py cuda (cancello GPU al confine)."""
    import e2e_allena as A
    t0 = time.time()
    a = _ciclo("s1_asincrona", scritture_asincrone=True)
    vp = A.sosta_possibile
    A.sosta_possibile = lambda *x: True
    try:
        b = _ciclo("s1_sosta", scritture_asincrone=True, scritture_sosta_dispositivo=True)
    finally:
        A.sosta_possibile = vp
    c = _ciclo("s1_ripiego", scritture_asincrone=True, scritture_sosta_dispositivo=True)
    fa, fb, fc = _sha_file_ripresa(a[3]), _sha_file_ripresa(b[3]), _sha_file_ripresa(c[3])
    bop = {k: v for k, v in fa.items() if k.startswith("bop_pv_")}
    # i file ripresa contengono la config (il campo stesso cambia fra le corse) e gli orari: si confrontano i TENSORI;
    # i file V-Bop (nessuna config) si confrontano byte per byte. Il file intero al bit: e2e_prova_sosta.py.
    tens = {}
    for k in fa:
        if k.startswith("ripresa_"):
            za, zb, zc = (torch.load(x[3] / k, map_location="cpu", weights_only=False) for x in (a, b, c))
            tens[k] = all(_tensori_uguali(za[c_], z[c_]) for z in (zb, zc) for c_ in za if c_ != "stato")
    ok = bool(bop) and all(fb.get(k) == v and fc.get(k) == v for k, v in bop.items()) and bool(tens) \
        and all(tens.values())
    return registra("S1_sosta_nel_ciclo", ok, bop_uguali=len(bop), riprese_tensori_uguali=tens,
                    secondi=round(time.time() - t0, 1))


def _tensori_uguali(x, y) -> bool:
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and \
            x.stride() == y.stride() and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(_tensori_uguali(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(_tensori_uguali(u, v) for u, v in zip(x, y))
    return x == y


PROVE = {"S1": prova_s1, "F0": prova_f0, "F1": prova_f1, "F2": prova_f2, "F3": prova_f3, "F4": prova_f4, "F5": prova_f5}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = sys.argv[1:] or list(PROVE)
    for n in scelte:
        try:
            PROVE[n]()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            registra(f"{n}_eccezione", False, errore=repr(e))
        p = DIR / "esito_test_fase_v.json"
        vecchio = json.loads(p.read_text()) if p.exists() else {"prove": {}}
        vecchio["prove"].update(ESITO["prove"])
        vecchio["data"] = ESITO["data"]
        p.write_text(json.dumps(vecchio, indent=1, default=str))
    return 0 if all(v["ok"] for v in ESITO["prove"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
