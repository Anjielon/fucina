#!/usr/bin/env python3
"""Prove del ramo `integrata` (27/9/2026): TUTTE le leve accese insieme, la ripresa da uno stato di tipo EB2 e
l'attribuzione per componente. SOLO CPU (la GPU e' di EB2), studente VERO minuscolo (GGUF sintetico con esperti
TQ1_0 e non-esperti k-quant, 2 strati: GDN + attenzione piena), passo V vero.

  I1  tutte insieme: V-Bop + rho adattivo + CAKLD (gamma misurato) + F32 (tutte le famiglie, anche ssm_a con la cache
      per versione) + LoRA-GA (gamma 64) + Sophia (gamma 2,4) + pesi per token kl + V senza BH (guadagno, t_g > 1) +
      EMA + held-out rapido/pieno + attribuzione: 3 passi; voci del registro presenti e finite; registro_componenti.jsonl
      (una riga per passo); L_tutto(t) == L0(t+1) AL BIT; ripresa 1 + 2 == 3 filati AL BIT (theta, F32, LoRA A/B/A0/B0,
      Sophia, AdamW di F32/LoRA, V-Bop, registri, EMA, stato); attribuzione accesa == spenta AL BIT (theta, F32, LoRA,
      byte dei codici, ottimizzatori, stato senza la voce "attribuzione").
  I2  ripresa da uno stato di tipo EB2 (LoRA a init zero GIA' allenata, V-Bop, rho tarato, AdamW, F32, CAKLD, EMA 0,9)
      verso la configurazione integrata con --lora-reinit-ga: accettata; al primo passo theta, F32, byte (registri
      rigiocati) == la ripresa; Sophia parte da m = exp_avg di Adam; AdamW di F32 ripreso al bit, LoRA con Adam vuoto;
      V-Bop ripreso dal file (sha); rho == tarato; LoRA scartata (norma registrata) e re-inizializzata GA (A = A0,
      B = B0); EMA del passo 3 == lerp(EMA della ripresa, theta_3) al bit; storia dei passi 1-2 intatta; una seconda
      ripresa con la stessa riga di lancio non re-inizializza (idempotente).
  I3  identita' esatte dell'attribuzione: sola theta -> senza_theta == L0 e solo == tutto; theta + F32 -> senza_theta ==
      solo_f32 e senza_f32 == solo_theta; theta + codici (salti tenuti per costruzione) -> senza_codici == solo_theta e
      senza_theta == solo_codici; allenamento con/ senza attribuzione identico al bit anche coi salti; un ripristino
      guastato apposta viene SCOPERTO (solleva).
  I5 istantanee a ogni passo · I6 cambio di ricetta alla ripresa (bf16x3) · I7 studente vero di EB3 (sottoprocesso) ·
  I8 --non-consumare · I9 righe mc + bf16x3 + attribuzione.
  I10 ripresa da un codice all'altro (modo b di riavvio_integrato.sh): stato scritto dal codice CONGELATO con cui EB3
      parte (E2E_CODICE_VECCHIO, default @@ROOT:depot@@/e2e/codice/EB3-ce17ad7/thor_v2; sottoprocesso) ripreso dal
      codice di oggi: stessa config -> passo 3 == quello del codice vecchio AL BIT; + righe mc / peso risposta / lettere
      -> decisione accettata, stato intatto, LoRA GA non re-inizializzata. Mutazione (theta ripreso +1 ulp): ROSSA.

Uso (4 fili, nice, niente GPU):
    systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=6G nice -n 19 env CUDA_VISIBLE_DEVICES= \\
        HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 E2E_ALLENA_PROVE=@@ROOT:lavoro@@/e2e/prove_integrata/x \\
        @@ROOT:venv@@/bin/python3 e2e_test_integrata.py [I1 I2 I3]
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("E2E_ALLENA_PROVE", "@@ROOT:lavoro@@/e2e/prove_allena")
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_attribuzione as AT  # noqa: E402
import e2e_lora as LR  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_studente as S  # noqa: E402
import e2e_test_allena as TA  # noqa: E402
import e2e_test_f32 as TF  # noqa: E402
import e2e_test_lora as TL  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "integrata"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
FAM = S.famiglie_f32(TF.TUTTE)                    # tutte le famiglie F32 (ssm_a: cache per versione del valore HF)


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:2500]}", flush=True)
    return bool(ok)


def _pulisci(*dd) -> None:
    for d in dd:
        shutil.rmtree(d, ignore_errors=True)


def _ca_integrata(**kw):
    """La configurazione di EB3 in piccolo (valori dei rami: Sophia 2,4, LoRA-GA gamma 64, pesi kl, t_g > 1)."""
    base = dict(lora_init="ga", lora_ga_gamma=64.0, allena_f32=FAM, lr_f32=3e-3, righe_per_passo=4, righe_per_micro=1,
                pv_codici=True, pv_modo="vbop", pv_rho=0.05, pv_rho_adattivo=True, pv_micro_curvatura=1,
                perdita="cakld", ottimizzatore_theta="sophia", sophia_gamma=2.4, peso_token="kl",
                pv_accetta="guadagno", pv_z_guadagno_min=1.0, ema_theta=0.5, heldout_ogni=1, heldout_righe_rapide=2,
                heldout_pieno_ogni=2, attribuzione=True, attribuzione_righe=2, tieni_riprese=10)
    base.update(kw)
    return TL._ca_lora(**base)


def con_domini(src, domini=("edu", "strumenti", "it_chat")):
    """La stessa sorgente in memoria con le righe di train assegnate a rotazione ai `domini` (domini diversi nel passo)."""
    r = {k: list(v) for k, v in src.r.items()}
    r["train"] = [(ids, h, domini[i % len(domini)], iz) for i, (ids, h, _, iz) in enumerate(r["train"])]
    return A.SorgenteMemoria(r, src.b, src.modo)


def _senza_ombra(sd):
    """Stato dell'ottimizzatore senza l'ombra di AdamW (exp_avg_sq/step_ombra di Sophia: solo registro)."""
    sd = copy.copy(sd)
    if isinstance(sd, dict) and sd.get("tipo") == "sophia":
        so = copy.copy(sd["sophia"])
        so["state"] = {i: {k: v for k, v in s.items() if k not in ("exp_avg_sq", "step_ombra")}
                       for i, s in so["state"].items()}
        sd["sophia"] = so
    return sd


def _studente_int():
    return TL.studente("A32", TL.cfg_lora(4), allena_f32=FAM)


def _byte(st) -> list:
    return [st.esperti(L).fam[n].blocchi.clone() for (L, n) in st.chiavi()]


def _uguali_lista(a, b) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _uguali_lora_ga(sa, sb) -> bool:
    return TL._uguali_lora(sa, sb) and all(
        (a.A0 is None) == (b.A0 is None) and (a.A0 is None or (torch.equal(a.A0, b.A0) and torch.equal(a.B0, b.B0)))
        for a, b in zip(sa.lora.values(), sb.lora.values()))


def _uguali_stato_opt(x, y) -> bool:
    """Stato di ottimizzatore (dict annidati di tensori e numeri) uguale al bit."""
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(_uguali_stato_opt(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(_uguali_stato_opt(a, b) for a, b in zip(x, y))
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.dtype == y.dtype and torch.equal(x, y)
    return x == y


def _stato_confrontabile(z: dict, senza_attribuzione: bool = False) -> dict:
    s = TA._senza_tempi(z)
    TA._togli_rumore(s)
    for x in s["storia"]:
        a = x.get("attribuzione")
        if a is not None:
            a.pop("secondi", None)
            if senza_attribuzione:
                x.pop("attribuzione")
        if senza_attribuzione:                         # solo registro con --attribuzione: min/mediana dei pesi, origine
            for k in ("min_w", "mediana_w"):
                (x.get("pesi") or {}).pop(k, None)
            x.pop("per_origine", None)
    if senza_attribuzione:
        s["config"].pop("attribuzione", None)
    return s


def _righe_jsonl(d: Path) -> list:
    p = d / "registro_componenti.jsonl"
    return [json.loads(r) for r in p.read_text().splitlines() if r.strip()] if p.exists() else []


def _finiti(a: dict) -> bool:
    val = [a["L0"]["perdita"], a["L_tutto"]["perdita"], *a["delta"].values(), *a["dsolo"].values()]
    return all(math.isfinite(v) for v in val)


# ─────────────────────────────────────────── I1 ───────────────────────────────────────────
def prova_i1() -> bool:
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = con_domini(src)
    ca = _ca_integrata()
    d_f, d_r, d_o = DIR / "i1_filata", DIR / "i1_ripresa", DIR / "i1_spenta"
    _pulisci(d_f, d_r, d_o)
    reg: list = []
    st_f = _studente_int()
    z_f = A.allena(st_f, src, d_f, ca, WS, WS, log=reg.append)
    storia = z_f["storia"]
    voci = {"passi": z_f["passo"] == 3,
            "attribuzione": all("attribuzione" in x and "theta" in x["attribuzione"]["componenti"] for x in storia),
            # min/mediana dei pesi: solo con --attribuzione, dal primo passo dopo G1 (il G1 ha i suoi pesi senza)
            "pesi_min_mediana": all("pesi" in x and "min_w" in x["pesi"] and "mediana_w" in x["pesi"] for x in storia[1:]),
            "sophia": all("sophia" in x for x in storia[1:]),
            "pv": all("pv" in x for x in storia[1:]),
            "lora_ga": z_f.get("lora_ga", {}).get("passo") == 0 and LR.ha_ga(st_f),
            "diagnostica": all({"theta", "f32", "lora"} <= set(x["attribuzione"]["diagnostica"]) for x in storia),
            "finiti": all(_finiti(x["attribuzione"]) for x in storia),
            "accetta_guadagno": any("SENZA i test BH" in m for m in reg),
            "log_attribuzione": sum("attribuzione:" in m for m in reg) == 3,
            "adam_controfattuale": all("theta_adam" in x["attribuzione"] for x in storia[1:])
            and all(math.isfinite(x["attribuzione"]["theta_adam"]["delta"]) for x in storia[1:]),
            "ombra_nello_stato": all("exp_avg_sq" in s for s in A.carica_ripresa(d_f, "cpu")["opt"]["sophia"]["state"]
                                     .values()) if (d_f / "ripresa_ultima.json").exists() else False}
    righe = _righe_jsonl(d_f)
    jl = (len(righe) == 3 and [r["passo"] for r in righe] == [1, 2, 3]
          and all({"attribuzione", "tempi", "kl_train"} <= set(r) for r in righe)
          and all({"avanti", "indietro", "gradiente", "passo_P", "passo_V", "attribuzione"} <= set(r["tempi"])
                  for r in righe[1:])
          and "heldout_rapido" in righe[0] and "heldout" in righe[1] and "heldout" in righe[2]
          and all("kl_per_dominio" in (r.get("heldout") or r.get("heldout_rapido")) for r in righe))
    catena = all(storia[i]["attribuzione"]["L_tutto"]["perdita"] == storia[i + 1]["attribuzione"]["L0"]["perdita"]
                 and storia[i]["attribuzione"]["L_tutto"]["kl"] == storia[i + 1]["attribuzione"]["L0"]["kl"]
                 for i in range(len(storia) - 1))
    # ripresa: 1 passo + ripresa + 2 == 3 filati
    A.allena(_studente_int(), src, d_r, ca, WS, WS, ferma_a=1, log=muto)
    st_r = _studente_int()
    z_r = A.allena(st_r, src, d_r, ca, WS, WS, riprendi=True, log=muto)
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    rip = {"theta": _uguali_lista(st_f.parametri(), st_r.parametri()),
           "f32": _uguali_lista(st_f.parametri_f32(), st_r.parametri_f32()),
           "lora": _uguali_lora_ga(st_f, st_r), "byte": _uguali_lista(_byte(st_f), _byte(st_r)),
           "opt": _uguali_stato_opt(rf["opt"], rr["opt"]),
           "ema": _uguali_stato_opt(rf.get("ema"), rr.get("ema")) and rf.get("ema") is not None,
           "stato": _stato_confrontabile(z_f) == _stato_confrontabile(z_r),
           "bop_sha": z_f["pv"].get("bop", {}).get("sha") == z_r["pv"].get("bop", {}).get("sha") is not None,
           "registri_sha": z_f["pv"]["sha"] == z_r["pv"]["sha"]}
    # attribuzione spenta: stesso allenamento al bit
    st_o = _studente_int()
    z_o = A.allena(st_o, src, d_o, _ca_integrata(attribuzione=False), WS, WS, log=muto)
    ro = A.carica_ripresa(d_o, "cpu")
    spenta = {"theta": _uguali_lista(st_f.parametri(), st_o.parametri()),
              "f32": _uguali_lista(st_f.parametri_f32(), st_o.parametri_f32()),
              "lora": _uguali_lora_ga(st_f, st_o), "byte": _uguali_lista(_byte(st_f), _byte(st_o)),
              "opt": _uguali_stato_opt(_senza_ombra(rf["opt"]), _senza_ombra(ro["opt"])),
              "stato": _stato_confrontabile(z_f, True) == _stato_confrontabile(z_o, True),
              "nessuna_riga": not (d_o / "registro_componenti.jsonl").exists()
              and all("attribuzione" not in x for x in z_o["storia"])}
    ok = all(voci.values()) and jl and catena and all(rip.values()) and all(spenta.values())
    return registra("I1_tutte_le_leve_insieme", ok, voci=voci, jsonl=jl, L_tutto_uguale_L0_dopo=catena, ripresa=rip,
                    attribuzione_spenta_al_bit=spenta,
                    attribuzione=[{k: x["attribuzione"][k] for k in ("componenti", "delta", "dsolo", "peggiore",
                                                                       "valutazioni")} for x in storia],
                    pv=[(x["pv"]["salti"], x["pv"]["tenuto"], x["pv"]["rho"]) for x in storia[1:]],
                    sophia=[x["sophia"]["quota_non_clippati"] for x in storia[1:]],
                    heldout=[(x["passo"], round(x["metrica"], 6)) for x in z_f["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I2 ───────────────────────────────────────────
def _ca_eb2(**kw):
    """Lo stato di EB2 in piccolo: LoRA a init zero, V-Bop + rho adattivo, AdamW, F32, CAKLD, EMA 0,9, rapido."""
    base = dict(allena_f32=FAM, lr_f32=3e-3, righe_per_passo=4, righe_per_micro=1, pv_codici=True, pv_modo="vbop",
                pv_rho=0.05, pv_rho_adattivo=True, pv_micro_curvatura=1, perdita="cakld", ema_theta=0.9,
                heldout_ogni=1, heldout_righe_rapide=2, heldout_pieno_ogni=16, tieni_riprese=1)
    base.update(kw)
    return TL._ca_lora(**base)


def prova_i2() -> bool:
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    d = DIR / "i2_eb2"
    _pulisci(d)
    st1 = _studente_int()
    ca1 = _ca_eb2()
    o_esito = PV.RegioneFiducia.esito                 # fase 1: salti TENUTI per costruzione (registri da rigiocare)

    def tiene_sempre(self, rho, prima, dopo, n, **kw):
        o_esito(self, rho, prima, dopo, n, **kw)
        return True
    PV.RegioneFiducia.esito = tiene_sempre
    try:
        z1 = A.allena(st1, src, d, ca1, WS, WS, ferma_a=2, log=muto)
    finally:
        PV.RegioneFiducia.esito = o_esito
    r1 = A.carica_ripresa(d, "cpu")
    lora_allenata = any(float(v["B"].abs().max()) > 0 for v in r1["lora"].values()) and \
        not any("A0" in v for v in r1["lora"].values())
    byte1, theta1, f321 = _byte(st1), [p.detach().clone() for p in st1.parametri()], \
        [p.detach().clone() for p in st1.parametri_f32()]
    n_th = len(st1.parametri())
    n_f = len(st1.parametri_f32())
    bop1 = dict(z1["pv"]["bop"])
    rho1 = z1["pv"]["rho"]
    # spie: stato alla prima sonda (prima di ogni aggiornamento), ottimizzatore caricato, V-Bop ripreso
    visto: dict = {}
    o_sonda, o_sophia, o_bop = AT.perdita_sonda, A.carica_ottimizzatore_sophia, PV.StatoV.ripristina_bop

    def spia_sonda(st, *a, **k):
        if "theta" not in visto:
            visto["theta"] = [p.detach().clone() for p in st.parametri()]
            visto["f32"] = [p.detach().clone() for p in st.parametri_f32()]
            visto["byte"] = _byte(st)
            visto["lora"] = {n: {k2: v.clone() for k2, v in ad.valori().items()} for n, ad in st.lora.items()}
        return o_sonda(st, *a, **k)

    def spia_sophia(opt, sd, st, nomi_prima, lora_prima, log):
        o_sophia(opt, sd, st, nomi_prima, lora_prima, log)
        visto["sophia"] = copy.deepcopy(opt.sophia.state_dict())
        visto["adam"] = copy.deepcopy(opt.adam.state_dict())

    def spia_bop(self, ist):
        o_bop(self, ist)
        visto["bop"] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ist.items()}
        visto["bop_dopo"] = self.istantanea_bop()
    AT.perdita_sonda, A.carica_ottimizzatore_sophia, PV.StatoV.ripristina_bop = spia_sonda, spia_sophia, spia_bop
    reg: list = []
    try:
        ca2 = _ca_integrata(ema_theta=0.9, heldout_pieno_ogni=16, tieni_riprese=1)
        st2 = _studente_int()
        z2 = A.allena(st2, src, d, ca2, WS, WS, riprendi=True, ferma_a=3, log=reg.append, reinit_lora_ga=True)
    finally:
        AT.perdita_sonda, A.carica_ottimizzatore_sophia, PV.StatoV.ripristina_bop = o_sonda, o_sophia, o_bop
    r2 = A.carica_ripresa(d, "cpu")
    so = visto.get("sophia", {}).get("state", {})
    ad = visto.get("adam", {}).get("state", {})
    s1 = r1["opt"]["state"]
    c = {"lora_allenata_a_init_zero_nella_ripresa": lora_allenata,
         "accettata": z2["passo"] == 3,
         "theta_alla_prima_sonda": _uguali_lista(visto.get("theta", []), theta1),
         "f32_alla_prima_sonda": _uguali_lista(visto.get("f32", []), f321),
         "byte_alla_prima_sonda": _uguali_lista(visto.get("byte", []), byte1),
         "lora_reinizializzata_ga": all("A0" in v and torch.equal(v["A"], v["A0"]) and torch.equal(v["B"], v["B0"])
                                        for v in visto.get("lora", {}).values()) and bool(visto.get("lora")),
         "lora_diversa_dalla_scartata": any(not torch.equal(visto["lora"][n]["B"], r1["lora"][n]["B"].to(
             visto["lora"][n]["B"].dtype)) for n in r1["lora"]) if visto.get("lora") else False,
         "sophia_m_da_adam": len(so) == n_th and all(torch.equal(so[i]["exp_avg"], s1[i]["exp_avg"])
                                                      and so[i]["n_hess"] == 0 for i in range(n_th)),
         "ombra_v_da_adam": len(so) == n_th and all(torch.equal(so[i]["exp_avg_sq"], s1[i]["exp_avg_sq"])
                                                     and float(so[i]["step_ombra"]) == float(s1[i]["step"])
                                                     for i in range(n_th)),
         "adamw_f32_ripreso": all(_uguali_stato_opt(ad[j], s1[n_th + j]) for j in range(n_f)),
         "adamw_lora_vuoto": all(j not in ad for j in range(n_f, n_f + len(st2.parametri_lora()))),
         "vbop_dal_file": "bop" in visto and bop1.get("sha") is not None
         and all(_uguali_stato_opt(visto["bop"][k], visto["bop_dopo"][k]) for k in visto["bop"]),
         "rho_tarato": z2["storia"][2]["pv"]["rho"] == rho1,
         "ema_passo3": all(torch.equal(e2, e1.clone().lerp_(t3, 1.0 - 0.9)) for e1, e2, t3 in
                           zip((r1["ema"][A._chiave(k)] for k in st2.chiavi()),
                               (r2["ema"][A._chiave(k)] for k in st2.chiavi()),
                               (r2["theta"][A._chiave(k)] for k in st2.chiavi()))),
         "storia_1_2_intatta": _stato_confrontabile(z2)["storia"][:2] == _stato_confrontabile(z1)["storia"][:2],
         "registro_reinit": z2.get("lora_reinit_ga", {}).get("passo") == 2
         and z2["lora_reinit_ga"]["norma_totale"] > 0 and z2.get("lora_ga", {}).get("passo") == 2,
         "registri_da_rigiocare": int(z1["pv"].get("registri") or 0) > 0,
         "log": {s: any(s in m for m in reg) for s in ("SCARTATA", "adamw -> sophia", "stato V-Bop ripreso",
                                                       "rigiocati", "attribuzione per componente accesa")},
         "campi_cambiati": any("ripresa con campi cambiati" in m and "ottimizzatore_theta" in m and "peso_token" in m
                               and "pv_accetta" in m and "lora_init" in m and "attribuzione" in m for m in reg)}
    # la stessa riga di lancio a una ripresa successiva NON re-inizializza (LoRA gia' GA)
    reg3: list = []
    st3 = _studente_int()
    A.allena(st3, src, d, ca2, WS, WS, riprendi=True, ferma_a=3, log=reg3.append, reinit_lora_ga=True)
    c["idempotente"] = any("senza effetto" in m for m in reg3) and _uguali_lora_ga(st2, st3) and \
        not any("SCARTATA" in m for m in reg3)
    ok = all(all(v.values()) if isinstance(v, dict) else v for v in c.values())
    return registra("I2_ripresa_tipo_EB2_verso_integrata", ok, **c,
                    scartata=z2.get("lora_reinit_ga"), secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I3 ───────────────────────────────────────────
def _attr(z) -> list:
    return [x["attribuzione"] for x in z["storia"]]


def prova_i3() -> bool:
    t0 = time.time()
    esiti = {}
    # (a) sola theta: niente F32, LoRA, PV
    src, WS = TF.sorgente_minuscola(n_train=8)
    d = DIR / "i3_theta"
    _pulisci(d)
    z = A.allena(TF.studente("A32"), src, d, TF._ca(attribuzione=True, heldout_ogni=1, heldout_righe_rapide=2), WS, WS,
                 log=muto)
    at = _attr(z)
    esiti["sola_theta"] = all(a["componenti"] == ["theta"] and a["senza"]["theta"]["perdita"] == a["L0"]["perdita"]
                              and a["solo"]["theta"]["perdita"] == a["L_tutto"]["perdita"] for a in at) and len(at) == 4
    esiti["catena_theta"] = all(at[i]["L_tutto"] == at[i + 1]["L0"] for i in range(len(at) - 1))
    # (b) theta + F32
    d = DIR / "i3_f32"
    _pulisci(d)
    fam = S.famiglie_f32("attn_norm,ffn_gate_inp,ssm_a,ssm_dt")
    z = A.allena(TF.studente("A32", fam), src, d, TF._ca(allena_f32=fam, attribuzione=True), WS, WS, log=muto)
    at = _attr(z)
    esiti["theta_f32"] = len(at) == 4 and all(
        a["componenti"] == ["theta", "f32"] and a["senza"]["theta"] == a["solo"]["f32"]
        and a["senza"]["f32"] == a["solo"]["theta"] for a in at)
    # (c) theta + codici con i salti TENUTI per costruzione (la sonda V non annulla): con e senza attribuzione
    o_esito = PV.RegioneFiducia.esito

    def tiene_sempre(self, rho, prima, dopo, n, **kw):
        o_esito(self, rho, prima, dopo, n, **kw)
        return True
    PV.RegioneFiducia.esito = tiene_sempre
    try:
        corse = {}
        for nome, att in (("con", True), ("senza", False)):
            d = DIR / f"i3_codici_{nome}"
            _pulisci(d)
            st = TF.studente("A32")
            z = A.allena(st, src, d, TF._ca(pv_codici=True, pv_rho=0.2, pv_micro_curvatura=1, attribuzione=att,
                                             righe_per_passo=4), WS, WS, log=muto)
            corse[nome] = (st, z, A.carica_ripresa(d, "cpu"))
    finally:
        PV.RegioneFiducia.esito = o_esito
    (sc, zc, rc), (ss, zs, rs) = corse["con"], corse["senza"]
    at = _attr(zc)
    con_codici = [a for a in at if "codici" in a["componenti"]]
    esiti["codici_presenti"] = len(con_codici) >= 1
    esiti["theta_codici"] = bool(con_codici) and all(
        a["senza"]["codici"] == a["solo"]["theta"] and a["senza"]["theta"] == a["solo"]["codici"] for a in con_codici)
    esiti["codici_con_senza_al_bit"] = (_uguali_lista(sc.parametri(), ss.parametri()) and _uguali_lista(_byte(sc), _byte(ss))
                                        and _uguali_stato_opt(rc["opt"], rs["opt"])
                                        and _stato_confrontabile(zc, True) == _stato_confrontabile(zs, True))
    # (d) un ripristino guastato apposta viene scoperto
    o_metti = AT.Commutatore.metti

    def metti_rotto(self, comp, quale):
        if comp == "f32" and quale == "nuovo":
            self.stato[comp] = quale                       # finge di aver rimesso i master nuovi
            return
        return o_metti(self, comp, quale)
    AT.Commutatore.metti = metti_rotto
    d = DIR / "i3_guasto"
    _pulisci(d)
    try:
        A.allena(TF.studente("A32", fam), src, d, TF._ca(allena_f32=fam, attribuzione=True), WS, WS, ferma_a=2,
                 log=muto)
        esiti["guasto_scoperto"] = False
    except RuntimeError as e:
        esiti["guasto_scoperto"] = "NON ripristinato" in str(e) and "f32" in str(e)
    finally:
        AT.Commutatore.metti = o_metti
    ok = all(esiti.values())
    return registra("I3_attribuzione_identita_esatte", ok, **esiti,
                    salti_tenuti=[x.get("pv", {}).get("salti") for x in zc["storia"]],
                    esempio=[{k: a[k] for k in ("componenti", "delta", "dsolo")} for a in con_codici[:2]],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I5 ───────────────────────────────────────────
def prova_i5() -> bool:
    """--istantanea-ogni-passo: passo_<N> (theta + ema/, registri dei codici in entrambe) a ogni passo in una cartella a
    parte, le ultime 2; formato del migliore (carica_istantanea == studente e EMA della ripresa); campo "istantanea" nel
    registro_componenti.jsonl; allenamento identico senza; spazio insufficiente o cartella impossibile = saltata, la
    corsa continua."""
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    d, d_i, d_o = DIR / "i5_corsa", DIR / "i5_istantanee", DIR / "i5_senza"
    _pulisci(d, d_i, d_o)
    ca = _ca_integrata(istantanea_ogni_passo=True, istantanee_dir=str(d_i), istantanee_tieni=2)
    st = _studente_int()
    reg: list = []
    z = A.allena(st, src, d, ca, WS, WS, log=reg.append)
    r = A.carica_ripresa(d, "cpu")
    cartelle = sorted(x.name for x in d_i.glob("passo_*"))
    c = {"ultime_due": cartelle == ["passo_000002", "passo_000003"]}
    ist = A.carica_istantanea(d_i / "passo_000003")
    ist_e = A.carica_istantanea(d_i / "passo_000003" / "ema")
    th = torch.load(d_i / "passo_000003" / "theta.pt", weights_only=True)
    th_e = torch.load(d_i / "passo_000003" / "ema" / "theta.pt", weights_only=True)
    c["theta_uguale"] = all(torch.equal(th[A._chiave(k)], p.detach()) for k, p in zip(st.chiavi(), st.parametri()))
    c["ema_uguale"] = all(torch.equal(th_e[A._chiave(k)], r["ema"][A._chiave(k)]) for k in st.chiavi())
    sc = st.scale_per_export()
    c["scale_uguali"] = set(sc) == set(ist["scale"]) and all(
        np_eq(ist["scale"][n], sc[n]) for n in sc)
    c["f32_lora_in_entrambe"] = all((d_i / "passo_000003" / s / f).exists() for s in ("", "ema")
                                    for f in ("f32.npz", "lora.npz", "registri_pv.npz", "meta.json"))
    c["lora_ga_meta"] = ist["lora_meta"].get("init") == "ga" and ist_e["lora_meta"].get("init") == "ga"
    righe = _righe_jsonl(d)
    c["campo_nel_registro"] = [rr.get("istantanea", {}).get("dir", "").endswith(f"passo_{rr['passo']:06d}")
                               for rr in righe] == [True, True, True]
    c["log"] = sum("istantanea del passo" in m for m in reg) == 3
    st_o = _studente_int()
    A.allena(st_o, src, d_o, _ca_integrata(), WS, WS, log=muto)
    c["allenamento_identico"] = _uguali_lista(st.parametri(), st_o.parametri()) and \
        _uguali_stato_opt(r["opt"], A.carica_ripresa(d_o, "cpu")["opt"])
    esiti = {}
    for nome, kw in (("spazio", dict(istantanee_dir=str(DIR / "i5_spazio"), istantanee_libero_min_gib=1e9)),
                     ("cartella_impossibile", dict(istantanee_dir="/proc/cartella_impossibile"))):
        dd = DIR / f"i5_{nome}"
        _pulisci(dd)
        rg: list = []
        zz = A.allena(_studente_int(), src, dd, _ca_integrata(istantanea_ogni_passo=True, **kw), WS, WS, log=rg.append)
        rr = _righe_jsonl(dd)
        esiti[nome] = zz["passo"] == 3 and all(("saltata" if nome == "spazio" else "errore") in x["istantanea"]
                                               for x in rr) and len(rr) == 3
    c["saltata_ma_la_corsa_continua"] = esiti
    ok = all(all(v.values()) if isinstance(v, dict) else v for v in c.values())
    return registra("I5_istantanea_a_ogni_passo", ok, **c, byte=righe[-1].get("istantanea", {}).get("byte"),
                    secondi=round(time.time() - t0, 1))


def np_eq(a, b) -> bool:
    import numpy as np
    return np.array_equal(np.asarray(a), np.asarray(b))


# ─────────────────────────────────────────── I6 ───────────────────────────────────────────
def prova_i6() -> bool:
    """bf16x3fix nella ripresa di EB3: da uno stato a ricetta senza bf16x3 (teste fp32) alla ripresa con la stessa ricetta
    + proiezioni non-esperte bf16x3 (come B_ind32 -> B3_ind32), testa KL bf16x3 e logit del maestro bf16, piu' Sophia,
    attribuzione, LoRA-GA con reinit: accettata (cambio di ricetta = decisione registrata), la LoRA avvolge i
    `LineareBF16x3`, passo finito; cambi di ricetta che toccano l'indietro degli esperti RIFIUTATI.
    Sul GGUF giocattolo la rotazione di B non si applica (dimensioni sotto il blocco da 1024): la coppia provata e' A32 ->
    A32x3 (A32 + non_esperti_x3, registrata qui), con i lineari non-esperti in bf16x3 (soglia 8 uscite invece di 256)."""
    import dataclasses as _dc
    import e2e_bf16x3 as X3
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    d = DIR / "i6_ricetta"
    _pulisci(d)
    S.RICETTE["A32x3"] = _dc.replace(S.RICETTE["A32"], nome="A32x3", non_esperti_x3=True)
    o_sost = X3.sostituisci_lineari
    # soglia 8 invece di 256 (giocattolo: famiglie LoRA 64-512 uscite): in bf16x3 le proiezioni delle famiglie LoRA; restano
    # nn.Linear shared_expert_gate (1) e in_proj_a/in_proj_b (4), che sono F32 allenati come nel Thor vero (li' 1 e 32 < 256)
    X3.sostituisci_lineari = lambda modulo, min_uscite=8, classe=None: o_sost(modulo, 8, classe)
    reg: list = []
    try:
        st1 = TL.studente("A32", TL.cfg_lora(4), allena_f32=FAM)
        A.allena(st1, src, d, _ca_eb2(), WS, WS, ferma_a=2, log=muto)
        st2 = TL.studente("A32x3", TL.cfg_lora(4), allena_f32=FAM)
        avvolti = [type(m.base).__name__ for m in st2.moduli[0].modules() if isinstance(m, LR.LinearLoRA)]
        WS3, WT3 = X3.TestaX3(WS), WS.to(torch.bfloat16)
        z2 = A.allena(st2, src, d, _ca_integrata(ema_theta=0.9, heldout_pieno_ogni=16, tieni_riprese=1, testa_kl="bf16x3",
                                                 logit_maestro="bf16"), WS3, WT3, riprendi=True, ferma_a=3,
                      log=reg.append, reinit_lora_ga=True)
        c = {"lora_su_bf16x3": bool(avvolti) and all(x == "LineareBF16x3" for x in avvolti),
             "accettata": z2["passo"] == 3 and z2["studente"]["ricetta"] == "A32x3",
             "registrata": z2.get("ricette_cambiate") == [{"passo": 2, "da": "A32", "a": "A32x3"}]
             and any("ricetta A32 -> A32x3" in m for m in reg),
             "teste_cambiate": any("testa_kl" in m and "logit_maestro" in m for m in reg if "campi cambiati" in m),
             "lora_ga_su_bf16x3": LR.ha_ga(st2) and z2.get("lora_ga", {}).get("passo") == 2,
             "passo_finito": all(math.isfinite(x) for x in (z2["storia"][-1]["kl_train"],
                                                            z2["storia"][-1]["attribuzione"]["L_tutto"]["perdita"]))}
        rifiuti = {}
        for nome, ric in (("verso_B3_indietro_fp16", "B3"), ("verso_B", "B"), ("verso_A", "A")):
            try:
                A.allena(TL.studente(ric, TL.cfg_lora(4), allena_f32=FAM), src, d, _ca_integrata(ema_theta=0.9,
                         heldout_pieno_ogni=16, tieni_riprese=1), WS, WS, riprendi=True, ferma_a=4, log=muto)
                rifiuti[nome] = False
            except RuntimeError as e:
                rifiuti[nome] = "altro studente" in str(e)
        c["rifiuti"] = rifiuti
    finally:
        X3.sostituisci_lineari = o_sost
        S.RICETTE.pop("A32x3", None)
    c["regola"] = (A.ricetta_cambiabile("B_ind32", S.RICETTE["B3_ind32"]) and A.ricetta_cambiabile("B", S.RICETTE["B3"])
                   and A.ricetta_cambiabile("B3_ind32", S.RICETTE["B_ind32"])
                   and not A.ricetta_cambiabile("B_ind32", S.RICETTE["B3"])
                   and not A.ricetta_cambiabile("B", S.RICETTE["B_ind32"]) and not A.ricetta_cambiabile(None, S.RICETTE["B"]))
    ok = all(all(v.values()) if isinstance(v, dict) else v for v in c.values())
    return registra("I6_ripresa_verso_ricetta_bf16x3_teste_bf16", ok, **c, lineari_avvolti=len(avvolti),
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I7 ───────────────────────────────────────────
_I7_FIGLIO = r"""
import json, sys, torch
sys.path.insert(0, %r)
torch.set_num_threads(4)
import e2e_studente as S, e2e_lora as LR
f32 = S.famiglie_f32("attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,"
                     "ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm")
cfg = LR.ConfigLoRA(rango=16, alpha=None, famiglie=LR.famiglie_lora("tutte"), seme=20260926, tipi=("Q8_0",))
kw = dict(strati=[0, 3], ricetta="B3_ind32", motore_nome="torch", device="cpu", log=lambda *a, **k: None)
st = S.Studente(allena_f32=f32, lora=cfg, **kw)
st0 = S.Studente(allena_f32=f32, **kw)
avvolti = {f"{L}.{n}": type(m.base).__name__ for L in st.strati for n, m in st.moduli[L].named_modules()
           if isinstance(m, LR.LinearLoRA)}
x3 = sum(len(getattr(st.moduli[L], "lineari_x3", [])) for L in st.strati)
g = torch.Generator().manual_seed(3)
ids = torch.randint(0, st.cfg.vocab_size, (1, 32), generator=g)
with torch.no_grad():
    y, y0 = st.nascosto(ids), st0.nascosto(ids)
print(json.dumps({"avvolti": avvolti, "x3": x3, "adattatori": len(st.chiavi_lora()), "f32": len(st.chiavi_f32()),
                  "al_bit": bool(torch.equal(y, y0)), "finita": bool(torch.isfinite(y).all()),
                  "lora_conteggi": st.lora_conteggi}))
""" % str(QUI)


def prova_i7() -> bool:
    """(Thor-T VERO, strati 0 = GDN e 3 = attenzione piena; lenta, in un processo a parte: le altre prove sostituiscono la
    configurazione dello studente con quella del giocattolo) lo studente di EB3 si costruisce: ricetta B3_ind32 + tutte le
    famiglie F32 di EB3 + LoRA rango 16 sui Q8_0 delle famiglie di default. I F32 restano nn.Linear/parametri (router,
    shared_expert_gate, in_proj_a/b NON diventano bf16x3), la LoRA avvolge i `LineareBF16x3`; con B = 0 l'andata (32
    token) e' AL BIT quella dello stesso studente B3_ind32 senza LoRA."""
    import subprocess
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", _I7_FIGLIO], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""})
    if r.returncode != 0:
        return registra("I7_studente_EB3_vero_B3_lora_f32", False, errore=r.stderr[-2500:])
    o = json.loads(r.stdout.strip().splitlines()[-1])
    c = {"lora_su_bf16x3": bool(o["avvolti"]) and all(v == "LineareBF16x3" for v in o["avvolti"].values()),
         "adattatori": o["adattatori"] > 0, "f32": o["f32"] > 0, "lineari_x3": o["x3"] > 0,
         "andata_al_bit": o["al_bit"], "finita": o["finita"]}
    return registra("I7_studente_EB3_vero_B3_lora_f32", all(c.values()), **c, avvolti=o["avvolti"],
                    lora_conteggi=o["lora_conteggi"], secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I8 ───────────────────────────────────────────
def prova_i8() -> bool:
    """--non-consumare: con --rotazione i blocchi del teacher usati NON si consumano (EB3 legge il teacher di EB2), di
    default si' (come prima); allenamento identico al bit nei due casi; campo di esecuzione; CLI."""
    t0 = time.time()
    src0, WS = TL.sorgente_lora(n_train=12, n_held=4)
    esiti = {}
    for nome, cons in (("consuma", True), ("non_consuma", False)):
        src = A.SorgenteMemoria(src0.r, {"train": [("train/b0", 0, 4), ("train/b1", 4, 8), ("train/b2", 8, 12)]})
        d = DIR / f"i8_{nome}"
        _pulisci(d)
        st = TL.studente("A32", TL.cfg_lora(4))
        reg: list = []
        z = A.allena(st, src, d, TL._ca_lora(righe_per_passo=4, rotazione=True, consuma_blocchi=cons), WS, WS,
                     log=reg.append)
        esiti[nome] = (st, z, list(src.consumati), reg)
    (sc, zc, cc, _), (sn, zn, cn, rn) = esiti["consuma"], esiti["non_consuma"]
    c = {"consuma_di_default": cc == ["train/b0", "train/b1", "train/b2"] and zc["consumati"] == cc,
         "non_consuma": cn == [] and zn["consumati"] == [] and any("consumo SPENTO" in m for m in rn),
         "stesso_allenamento": _uguali_lista(sc.parametri(), sn.parametri()) and TL._uguali_lora(sc, sn),
         "esecuzione": "consuma_blocchi" in A._CAMPI_DI_ESECUZIONE,
         "cli": A._config_da_arg(TF._arg(non_consumare=True)).consuma_blocchi is False
         and A._config_da_arg(TF._arg()).consuma_blocchi is True}
    return registra("I8_non_consumare", all(c.values()), **c, consumati=cc, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I9 ───────────────────────────────────────────
def prova_i9() -> bool:
    """trainer-mc nell'integrazione (studente finto e sorgente composta delle prove M): righe mc + --peso-token risposta +
    --gamma-lettere con la testa bf16x3 (le 4 righe delle lettere da hi + lo: `righe_testa` == W[id4] a ~1e-5) e con
    l'attribuzione: voce `per_origine` (perdita di allenamento delle righe base e mc, token contati), held-out mc nel
    registro_componenti.jsonl, allenamento con/senza attribuzione identico al bit."""
    import e2e_bf16x3 as X3
    import e2e_test_mc as TM
    t0 = time.time()
    fx = TM.Fixture(n_train=24, n_mc=5, n_mc_held=3)
    src = fx.sorgente(nome="i9_fixture")
    W3 = X3.TestaX3(fx.WS)
    idx = torch.tensor(TM.ID4, dtype=torch.long)
    err_righe = float((A.righe_testa(W3, idx) - fx.WS[idx]).norm() / fx.WS[idx].norm())
    corse = {}
    for nome, att in (("con", True), ("senza", False)):
        d = DIR / f"i9_{nome}"
        _pulisci(d)
        ca = TM._ca(righe_per_passo=4, righe_mc=1, peso_token="risposta", gamma_lettere=1.0, testa_kl="bf16x3",
                    heldout_ogni=1, heldout_righe_rapide=2, heldout_pieno_ogni=2, ripresa_ogni=1, attribuzione=att)
        st = TA.StudenteFinto()
        z = A.allena(st, src, d, ca, W3, fx.WS, log=muto)
        corse[nome] = (st, z, d)
    (sc, zc, dc), (ss, zs, ds) = corse["con"], corse["senza"]
    storia = zc["storia"][1:]                            # il passo 1 e' il G1
    righe = _righe_jsonl(dc)
    c = {"righe_testa_bf16x3": err_righe < 1e-4,
         "per_origine": bool(storia) and all(
             x.get("per_origine") and x["per_origine"]["mc_token"] > 0 and x["per_origine"]["base_token"] > 0
             and math.isfinite(x["per_origine"]["mc"]) and math.isfinite(x["per_origine"]["base"]) for x in storia),
         "lettere": all(x.get("pesi", {}).get("lettere_n", 0) > 0 for x in storia),
         "attribuzione": all("attribuzione" in x for x in zc["storia"]),
         "heldout_mc_nel_json": any("mc" in (r.get("heldout") or r.get("heldout_rapido") or {}) for r in righe),
         "senza_attribuzione_niente_per_origine": all("per_origine" not in x for x in zs["storia"]),
         "stesso_allenamento": _uguali_lista(sc.parametri(), ss.parametri())}
    return registra("I9_mc_con_bf16x3_e_attribuzione", all(c.values()), **c, errore_righe_testa=err_righe,
                    valori_per_origine=[x.get("per_origine") for x in storia], secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I10 ──────────────────────────────────────────
# Fase 1 nel CODICE VECCHIO (quello con cui EB3 parte: @@ROOT:depot@@/e2e/codice/EB3-<commit>, E2E_CODICE_VECCHIO), in un
# sottoprocesso con quel codice in testa a sys.path: corsa integrata filata a 3 passi e la stessa fermata al passo 2;
# righe e maestro (anche le righe mc, fatte dallo stesso maestro) salvati su disco. Fase 2 qui, col codice NUOVO.
_FASE1_I10 = r'''
import copy, json, os, sys
from pathlib import Path
vecchio, d_old, d_rip, uscita = sys.argv[1:5]
solo_dati = sys.argv[5:] == ["solo_dati"]   # I11: solo righe e maestro, col codice nuovo
sys.path.insert(0, vecchio)
os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np, torch
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
import e2e_allena as A, e2e_pv_codici as PV, e2e_test_integrata as TI, e2e_test_lora as TL
assert Path(A.__file__).resolve().parent == Path(vecchio).resolve(), A.__file__
assert solo_dati or not hasattr(A, "SorgenteConMC"), "il codice vecchio ha gia' trainer-mc"
src, WS = TL.sorgente_lora(n_train=12, n_held=4)
src = TI.con_domini(src)
m = TL.studente("A32", TL.cfg_lora(4))                     # il maestro di sorgente_lora, rifatto (stessi semi)
g = torch.Generator().manual_seed(5)
with torch.no_grad():
    for p in m.parametri():
        p.copy_(0.2 * torch.randn(p.shape, generator=g))
TL._b_casuale(m, 0.05, 6)
ID4, T = (357, 417, 351, 414), 16
gm = torch.Generator().manual_seed(77)
def righe_mc(n, dom):
    rr, pos, giu, fonte = [], [], [], []
    for i in range(n):
        ids = torch.randint(0, m.cfg.vocab_size, (T,), generator=gm)
        for t in (3, 9)[: 1 + i % 2]:
            k = int(torch.randint(0, 4, (1,), generator=gm))
            ids[t + 1] = ID4[k]; pos.append((i, t)); giu.append(k); fonte.append(f"{dom}{i}")
        with torch.no_grad():
            h = m.esce(m.avanti(ids[None])["xL"]).to(torch.bfloat16).clone()
        rr.append((ids.numpy().astype(np.int32), h, dom, None))
    return rr, {"pos": torch.tensor(pos), "giusta": torch.tensor(giu), "id_lettere": torch.tensor(ID4), "fonte": fonte}
mc, held = righe_mc(4, "mc"), righe_mc(2, "mch")
o_esito = PV.RegioneFiducia.esito                          # salti TENUTI: registri da rigiocare alla ripresa
def tiene_sempre(self, rho, prima, dopo, n, **kw):
    o_esito(self, rho, prima, dopo, n, **kw)
    return True
PV.RegioneFiducia.esito = tiene_sempre
ca = TI._ca_integrata(ema_theta=0.9)
if not solo_dati:
    A.allena(TI._studente_int(), src, Path(d_old), ca, WS, WS, log=lambda *a, **k: None)
    A.allena(TI._studente_int(), src, Path(d_rip), ca, WS, WS, ferma_a=2, log=lambda *a, **k: None)
torch.save({"r": src.r, "b": src.b, "modo": src.modo, "WS": WS, "mc": mc, "held": held,
            "config": json.loads(json.dumps(A.asdict(ca), default=str))}, uscita)
print("FASE1 OK", A.__file__)
'''


def _sorgente_mc_i10(dati: dict, d: Path):
    """La sorgente base (le righe della fase 1) + righe mc e held-out mc in memoria (come `e2e_test_mc.Fixture`)."""
    base = A.SorgenteMemoria(dati["r"], dati["b"], dati["modo"])
    d.mkdir(parents=True, exist_ok=True)
    firma = {"teacher_mc": "memoria", "train": None, "heldout": None}
    parti = []
    for chi, (righe, risp) in (("train", dati["mc"]), ("heldout", dati["held"])):
        s = A.SorgenteMemoria({"train": righe}, {})
        p = d / f"risposte_{chi}.pt"
        torch.save(risp, p)
        r = A.leggi_risposte(p, lambda i, s=s: s.ids("train", i), s.righe_previste("train"), 16)
        firma[chi] = {"sha256": s.firma()["sha256"], "risposte_sha256": r.sha256}
        parti.append((s, r))
    return base, A.SorgenteConMC(base, parti[0], (parti[1][0], "train", parti[1][1]), firma)


def _ripresa_confrontabile(r: dict) -> dict:
    """La ripresa senza lo stato (confrontato a parte) e senza i campi di solo registro del codice nuovo."""
    return {k: v for k, v in r.items() if k != "stato"}


def _stato_i10(z: dict) -> dict:
    s = _stato_confrontabile(z)
    s.pop("config", None)
    for x in s["storia"]:
        for k in ("min_w", "mediana_w"):                 # solo registro, nati dopo il codice vecchio
            (x.get("pesi") or {}).pop(k, None)
    return s


def prova_i10() -> bool:
    """Ripresa di EB3 da un codice all'altro (modo b dello script): lo stato scritto dal codice CONGELATO con cui EB3
    parte (fase 1, sottoprocesso) ripreso dal codice di oggi. (a) stessa configurazione: il passo 3 fatto dal codice nuovo
    == il passo 3 del codice vecchio AL BIT (ripresa intera: theta, F32, LoRA/GA, Sophia + ombra, AdamW, EMA, V-Bop,
    registri; stato e storia), nessun campo cambiato; (b) + righe mc (1 su 4), --peso-token risposta, --gamma-lettere,
    --lora-reinit-ga: accettata come decisione, stato alla prima sonda == la ripresa, LoRA GA NON re-inizializzata,
    segmento mc aperto al passo 2 dal cursore base, righe mc nei passi dopo, per_origine, registro_componenti continuo."""
    import subprocess
    vecchio = Path(os.environ.get("E2E_CODICE_VECCHIO", "@@ROOT:depot@@/e2e/codice/EB3-ce17ad7/thor_v2"))
    if not (vecchio / "e2e_allena.py").exists():
        return registra("I10_ripresa_da_codice_vecchio", False, errore=f"codice vecchio assente: {vecchio}")
    t0 = time.time()
    d_old, d_rip, d_new, d_mc = (DIR / f"i10_{n}" for n in ("vecchio", "ripresa", "nuovo", "mc"))
    _pulisci(d_old, d_rip, d_new, d_mc)
    uscita = DIR / "i10_fase1.pt"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    f1 = subprocess.run([sys.executable, "-c", _FASE1_I10, str(vecchio), str(d_old), str(d_rip), str(uscita)],
                        cwd=str(DIR), env=env, capture_output=True, text=True)
    if f1.returncode != 0 or "FASE1 OK" not in f1.stdout:
        return registra("I10_ripresa_da_codice_vecchio", False, fase1=f1.stdout[-1500:] + f1.stderr[-3000:])
    dati = torch.load(uscita, map_location="cpu", weights_only=False)
    TL._GGUF = TL.DIR / "sintetico_lora.gguf"          # il GGUF della fase 1 (rigenerarlo cambia l'mtime nella firma)
    WS = dati["WS"]
    base, src_mc = _sorgente_mc_i10(dati, DIR / "i10_sorgente_mc")
    o_esito = PV.RegioneFiducia.esito

    def tiene_sempre(self, rho, prima, dopo, n, **kw):
        o_esito(self, rho, prima, dopo, n, **kw)
        return True
    ca = _ca_integrata(ema_theta=0.9)
    cfg_vecchia = dati["config"]
    cfg_nuova = json.loads(json.dumps(A.asdict(ca), default=str))
    c: dict = {"fase1_codice_vecchio": str(vecchio) in f1.stdout,
               "config_uguale": all(cfg_nuova.get(k) == v for k, v in cfg_vecchia.items())
               and all(cfg_vecchia.get(k, A._CONFIG_DEFAULT.get(k)) == v for k, v in cfg_nuova.items())}
    # (a) stessa configurazione, codice nuovo
    shutil.copytree(d_rip, d_new)
    reg: list = []
    PV.RegioneFiducia.esito = tiene_sempre
    try:
        z_new = A.allena(_studente_int(), base, d_new, ca, WS, WS, riprendi=True, log=reg.append, reinit_lora_ga=True)
    finally:
        PV.RegioneFiducia.esito = o_esito
    r_old, r_new = A.carica_ripresa(d_old, "cpu"), A.carica_ripresa(d_new, "cpu")
    z_old = r_old["stato"]
    c["a_passo_3"] = z_new["passo"] == 3 == z_old["passo"]
    c["a_nessun_campo_cambiato"] = not any("ripresa con campi cambiati" in m for m in reg)
    c["a_ripresa_al_bit"] = _uguali_stato_opt(_ripresa_confrontabile(r_old), _ripresa_confrontabile(r_new))
    if not c["a_ripresa_al_bit"]:
        c["a_chiavi_diverse"] = sorted(k for k in set(r_old) | set(r_new) if k != "stato" and not _uguali_stato_opt(
            r_old.get(k), r_new.get(k)))
    zn = r_new["stato"]                                 # lo stato SCRITTO nelle due riprese del passo 3
    c["a_stato_al_bit"] = _stato_i10(z_old) == _stato_i10(zn)
    if not c["a_stato_al_bit"]:
        so, sn = _stato_i10(z_old), _stato_i10(zn)
        c["a_stato_chiavi_diverse"] = sorted(k for k in set(so) | set(sn) if so.get(k) != sn.get(k))
    c["a_registri_sha"] = z_old["pv"]["sha"] == zn["pv"]["sha"] and int(z_old["pv"].get("registri") or 0) > 0
    c["a_lora_ga_senza_effetto"] = any("senza effetto" in m for m in reg)
    jo, jn = _righe_jsonl(d_old), _righe_jsonl(d_new)
    c["a_jsonl_continuo"] = [r["passo"] for r in jn] == [1, 2, 3] == [r["passo"] for r in jo]
    # (b) + righe mc, codice nuovo, dalla stessa ripresa del passo 2
    shutil.copytree(d_rip, d_mc)
    r2 = A.carica_ripresa(d_mc, "cpu")
    visto: dict = {}
    o_sonda = AT.perdita_sonda

    def spia_sonda(st, *a, **k):
        if "theta" not in visto:
            visto["theta"] = [p.detach().clone() for p in st.parametri()]
            visto["f32"] = [p.detach().clone() for p in st.parametri_f32()]
            visto["lora"] = {n: {k2: v.clone() for k2, v in ad.valori().items()} for n, ad in st.lora.items()}
        return o_sonda(st, *a, **k)
    ca_mc = _ca_integrata(ema_theta=0.9, righe_mc=1, peso_token="risposta", beta_risposta=0.1, gamma_lettere=1.0)
    reg2: list = []
    AT.perdita_sonda = spia_sonda
    try:
        st_mc = _studente_int()
        z_mc = A.allena(st_mc, src_mc, d_mc, ca_mc, WS, WS, riprendi=True, log=reg2.append, reinit_lora_ga=True)
    finally:
        AT.perdita_sonda = o_sonda
    th2 = [r2["theta"][A._chiave(k)] for k in st_mc.chiavi()]
    seg = (z_mc.get("mc") or {}).get("segmenti") or []
    dopo = z_mc["storia"][2:]
    c["b_accettata_come_decisione"] = any("ripresa con campi cambiati" in m and "righe_mc" in m and "peso_token" in m
                                          and "gamma_lettere" in m for m in reg2)
    c["b_theta_alla_prima_sonda"] = _uguali_lista(visto.get("theta", []), th2)
    c["b_lora_uguale_alla_ripresa"] = bool(visto.get("lora")) and all(
        torch.equal(visto["lora"][n][k], r2["lora"][n][k].to(visto["lora"][n][k].dtype))
        for n in r2["lora"] for k in ("A", "B", "A0", "B0"))
    c["b_lora_ga_senza_effetto"] = any("senza effetto" in m for m in reg2) and not any("SCARTATA" in m for m in reg2)
    c["b_segmento_mc"] = bool(seg) and seg[0]["passo"] == 2 and seg[0]["righe_mc"] == 1 and seg[0]["cursore_mc"] == 0
    c["b_passi_dopo"] = z_mc["passo"] >= 3 and bool(dopo)
    c["b_per_origine"] = bool(dopo) and all((x.get("per_origine") or {}).get("mc_token", 0) > 0 for x in dopo)
    c["b_storia_1_2_intatta"] = _stato_i10(z_mc)["storia"][:2] == _stato_i10(z_old)["storia"][:2]
    jm = _righe_jsonl(d_mc)
    c["b_jsonl_continuo"] = [r["passo"] for r in jm] == list(range(1, z_mc["passo"] + 1))
    c["b_heldout_mc"] = any("mc" in (r.get("heldout") or r.get("heldout_rapido") or {}) for r in jm[2:])
    # (c) la riga del modo b SENZA mc: + --peso-dominio, --strati-allenati (qui 1: lo strato 0 fa la parte di 0-19),
    #     --pv-sonda-dosi con la griglia di EB3 e --pv-ricentra; (d) la riga CON mc: (c) + righe mc permutate, peso
    #     risposta, lettere
    varianti = {"c": dict(peso_dominio=_PD, strati_allenati=(1,), **_DOSI),
                "d": dict(peso_dominio=_PD, strati_allenati=(1,), righe_mc=1, righe_mc_permutate=True,
                          peso_token="risposta", beta_risposta=0.1, gamma_lettere=1.0, **_DOSI)}
    for v, extra in varianti.items():
        d_c = DIR / f"i10_modo_b_{v}"
        _pulisci(d_c)
        shutil.copytree(d_rip, d_c)
        src_c = _sorgente_mc_i10(dati, DIR / f"i10_sorgente_mc_{v}")[1] if "righe_mc" in extra else base
        reg3: list = []
        st_c = _studente_int()
        z_c = A.allena(st_c, src_c, d_c, _ca_integrata(ema_theta=0.9, **extra), WS, WS, riprendi=True,
                       log=reg3.append, reinit_lora_ga=True)
        zero = [k for k in st_c.chiavi() if k[0] == 0]
        cambiati = ("peso_dominio", "strati_allenati", "pv_sonda_dosi", "pv_ricentra", "pv_dosi_frazioni") + (
            ("righe_mc", "righe_mc_permutate") if "righe_mc" in extra else ())
        c[f"{v}_accettata"] = z_c["passo"] >= 3 and any("ripresa con campi cambiati" in m and all(x in m for x in cambiati)
                                                        for m in reg3)
        c[f"{v}_strato0_theta_al_bit"] = bool(zero) and _uguali_lista(
            [p for k, p in zip(st_c.chiavi(), st_c.parametri()) if k[0] == 0], [r2["theta"][A._chiave(k)] for k in zero])
        c[f"{v}_strato0_f32_al_bit"] = _uguali_lista(
            [st_c.f32[n].master for n in st_c.chiavi_f32() if st_c.f32[n].L == 0],
            [r2["f32"][n] for n in st_c.chiavi_f32() if st_c.f32[n].L == 0])
        c[f"{v}_strato0_lora_al_bit"] = all(torch.equal(x, r2["lora"][n][k].to(x.dtype)) for n, a in st_c.lora.items()
                                            if a.L == 0 for k, x in a.valori().items())
        c[f"{v}_strato1_in_moto"] = not _uguali_lista(
            [p for k, p in zip(st_c.chiavi(), st_c.parametri()) if k[0] == 1],
            [r2["theta"][A._chiave(k)] for k in st_c.chiavi() if k[0] == 1])
        c[f"{v}_peso_dominio_nel_registro"] = all("peso_dominio" in x for x in z_c["storia"][2:])
        c[f"{v}_dosi_nel_registro"] = all("dosi" in x["pv"] for x in z_c["storia"][2:] if x.get("pv"))
        c[f"{v}_lora_ga_senza_effetto"] = any("senza effetto" in m for m in reg3)
        if "righe_mc" in extra:
            c[f"{v}_mc_permutate"] = bool(((z_c.get("mc") or {}).get("segmenti") or [{}])[0].get("permutate")) and all(
                "mc" in (x.get("peso_dominio") or {}).get("per_dominio", {}) for x in z_c["storia"][2:])
    ok = all(v for k, v in c.items() if not k.endswith("_diverse"))
    return registra("I10_ripresa_da_codice_vecchio", ok, **c, codice_vecchio=str(vecchio),
                    passi_mc=z_mc["passo"], segmenti=seg, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── I11 ──────────────────────────────────────────
_PD = (("edu", 1.5), ("it_chat", 0.5), ("mc", 2.0))    # domini del giocattolo (`con_domini`) e delle righe mc


def _foto_strato(st, L: int) -> dict:
    return {"theta": [p.detach().clone() for (L2, _), p in zip(st.chiavi(), st.parametri()) if L2 == L],
            "f32": [st.f32[n].master.detach().clone() for n in st.chiavi_f32() if st.f32[n].L == L],
            "lora": [v.clone() for n, a in st.lora.items() if a.L == L for v in a.valori().values()],
            "byte": [st.esperti(L2).fam[n].blocchi.clone() for (L2, n) in st.chiavi() if L2 == L]}


def _uguali_foto(a: dict, b: dict) -> bool:
    return all(_uguali_lista(a[k], b[k]) for k in a)


def prova_i11(nome: str = "I11_dominio_strati_mc_tutte", **extra) -> bool:
    """dominio-strati-b + trainer-mc + tutte le leve di integrata, sul giocattolo (studente vero minuscolo, 2 strati):
    righe mc 1 su 4 + --peso-token risposta + --gamma-lettere + --peso-dominio (anche «mc») + --strati-allenati 1 + Sophia +
    V-Bop + LoRA-GA + F32 + attribuzione. (a) 3 passi: strato 0 AL BIT (theta, F32, LoRA, byte), voce peso_dominio con
    «mc», per_origine, pesi delle risposte; (b) ripresa 1 + 2 == 3 filati AL BIT (theta, F32, LoRA, ottimizzatori, EMA,
    stato); (c) --peso-dominio con un dominio ignoto rifiutato, LoTA con --strati-allenati rifiutata in ConfigAllena."""
    import subprocess
    t0 = time.time()
    uscita = DIR / "i11_sorgente.pt"                   # righe e maestro come in I10 (stesso script, codice di OGGI)
    f1 = subprocess.run([sys.executable, "-c", _FASE1_I10, str(QUI), "-", "-", str(uscita), "solo_dati"], cwd=str(DIR),
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    if f1.returncode != 0:
        return registra(nome, False, fase1=f1.stderr[-3000:])
    dati = torch.load(uscita, map_location="cpu", weights_only=False)
    TL._GGUF = TL.DIR / "sintetico_lora.gguf"
    WS = dati["WS"]
    d_f, d_r = DIR / f"{nome}_filata", DIR / f"{nome}_ripresa"
    _pulisci(d_f, d_r)
    ca = _ca_integrata(ema_theta=0.9, righe_mc=1, peso_token="risposta", beta_risposta=0.1, gamma_lettere=1.0,
                       peso_dominio=_PD, strati_allenati=(1,), **extra)
    o_esito = PV.RegioneFiducia.esito

    def tiene_sempre(self, rho, prima, dopo, n, **kw):
        o_esito(self, rho, prima, dopo, n, **kw)
        return True
    reg: list = []
    PV.RegioneFiducia.esito = tiene_sempre
    try:
        _, src = _sorgente_mc_i10(dati, DIR / "i11_mc_f")
        st_f = _studente_int()
        f0 = _foto_strato(st_f, 0)
        z_f = A.allena(st_f, src, d_f, ca, WS, WS, log=reg.append)
        f1_ = _foto_strato(st_f, 0)
        _, src_r = _sorgente_mc_i10(dati, DIR / "i11_mc_r")
        A.allena(_studente_int(), src_r, d_r, ca, WS, WS, ferma_a=1, log=muto)
        st_r = _studente_int()
        z_r = A.allena(st_r, src_r, d_r, ca, WS, WS, riprendi=True, log=muto)
    finally:
        PV.RegioneFiducia.esito = o_esito
    storia = z_f["storia"][1:]
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    c = {"passi": z_f["passo"] >= 3,
         "strato0_al_bit": _uguali_foto(f0, f1_),
         "strato1_in_moto": not _uguali_lista(_foto_strato(_studente_int(), 1)["theta"], _foto_strato(st_f, 1)["theta"]),
         "peso_dominio_con_mc": all("mc" in (x.get("peso_dominio") or {}).get("per_dominio", {}) for x in storia),
         "per_origine": all((x.get("per_origine") or {}).get("mc_token", 0) > 0 for x in storia),
         "lettere": all((x.get("pesi") or {}).get("lettere_n", 0) > 0 for x in storia),
         "attribuzione": all("attribuzione" in x for x in z_f["storia"]),
         "log_strati": any("strati allenati 1 " in m for m in reg),
         "ripresa_theta": _uguali_lista(st_f.parametri(), st_r.parametri()),
         "ripresa_f32": _uguali_lista(st_f.parametri_f32(), st_r.parametri_f32()),
         "ripresa_lora": _uguali_lora_ga(st_f, st_r), "ripresa_byte": _uguali_lista(_byte(st_f), _byte(st_r)),
         "ripresa_opt": _uguali_stato_opt(rf["opt"], rr["opt"]),
         "ripresa_ema": _uguali_stato_opt(rf.get("ema"), rr.get("ema")) and rf.get("ema") is not None,
         "ripresa_stato": _stato_confrontabile(z_f) == _stato_confrontabile(z_r)}
    try:
        A.allena(_studente_int(), src, DIR / "i11_ignoto", _ca_integrata(peso_dominio=(("nessuno", 2.0),)), WS, WS,
                 log=muto)
        c["dominio_ignoto_rifiutato"] = False
    except ValueError as e:
        c["dominio_ignoto_rifiutato"] = "assenti dal teacher" in str(e)
    try:
        A.ConfigAllena(lota_rango=4, strati_allenati=(1,))
        c["lota_con_strati_rifiutata"] = False
    except ValueError as e:
        c["lota_con_strati_rifiutata"] = "strati_allenati" in str(e)
    if extra.get("pv_sonda_dosi"):                   # I13: la curva delle dosi nei passi V (registro e storia)
        c["dosi_nel_registro"] = all("dosi" in (x.get("pv") or {}) for x in storia if x.get("pv"))
        c["log_dosi"] = any("dosi " in m for m in reg)
    if extra.get("righe_mc_permutate"):
        c["segmento_permutato"] = bool(((z_f.get("mc") or {}).get("segmenti") or [{}])[0].get("permutate"))
    return registra(nome, all(c.values()), **c,
                    peso_dominio=[x.get("peso_dominio") for x in storia],
                    valori_per_origine=[x.get("per_origine") for x in storia], secondi=round(time.time() - t0, 1))


_DOSI = dict(pv_sonda_dosi=True, pv_dosi_frazioni=(1 / 4096, 1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0),
             pv_ricentra=True, pv_ricentra_giri=3)


def prova_i13() -> bool:
    """I11 con in piu' le leve di v-interazione-b (sonda a dosi con la griglia di EB3 + ricentro, 3 giri) e le righe mc
    PERMUTATE: tutte insieme, ripresa 1 + 2 == 3 al bit, strato congelato al bit, curva delle dosi nel registro."""
    return prova_i11("I13_tutte_con_dosi_e_mc_permutate", righe_mc_permutate=True, **_DOSI)


# ─────────────────────────────────────────── I14 ──────────────────────────────────────────
def prova_i14() -> bool:
    """--righe-mc-permutate sul piano (`PianoRighe`, niente modello): corpus mc finto a QUARTI di rotazione come quello
    vero (768 righe, rotazione = indice // 192), 8 righe mc su 32 a passo. Nell'ordine del teacher i primi 24 passi
    vedono SOLO la rotazione 0 (il difetto); permutate: tutte e 4 le rotazioni gia' nei primi 2 passi, il primo giro
    e' una permutazione (96 passi senza ripetizioni), i giri dopo identici a prima; spento: il segmento non ha la chiave;
    alla ripresa cambiare l'ordine apre un segmento nuovo (decisione registrata)."""
    t0 = time.time()
    train = list(range(4096))
    mc = [A.OFFSET_MC + j for j in range(768)]
    rot = lambda r: (r - A.OFFSET_MC) // 192            # noqa: E731
    c = {}
    piani = {}
    for perm in (False, True):
        ca = A.ConfigAllena(righe_per_passo=32, righe_mc=8, righe_mc_permutate=perm)
        stato = {"passo": 0}
        seg = A._segmenti_mc(stato, ca, SimpleNamespaceMC(mc), train, muto)
        piani[perm] = (A.PianoRighe(train, ca, mc, seg), seg)
    (pt, st_), (pp, sp) = piani[False], piani[True]
    mc_di = lambda pi, t: [r for r in pi.righe(t)[0] if r >= A.OFFSET_MC]   # noqa: E731
    c["teacher_24_passi_una_rotazione"] = {rot(r) for t in range(24) for r in mc_di(pt, t)} == {0}
    c["permutate_4_rotazioni_in_2_passi"] = {rot(r) for t in range(2) for r in mc_di(pp, t)} == {0, 1, 2, 3}
    giro0 = [r for t in range(96) for r in mc_di(pp, t)]
    c["giro0_permutazione"] = sorted(giro0) == mc and giro0 != mc
    c["giri_dopo_uguali"] = all(mc_di(pp, t) == mc_di(pt, t) for t in range(96, 130))
    c["base_uguali"] = all([r for r in pp.righe(t)[0] if r < A.OFFSET_MC] == [r for r in pt.righe(t)[0] if r < A.OFFSET_MC]
                           for t in range(0, 120, 7))
    c["chiave_solo_se_accesa"] = "permutate" not in st_[0] and sp[0].get("permutate") is True
    # ripresa al passo 5 con l'ordine cambiato: segmento nuovo dai cursori di ora
    stato = {"passo": 5, "mc": {"firme": [{"teacher_mc": "finto"}], "segmenti": [dict(st_[0])]}}
    reg: list = []
    seg2 = A._segmenti_mc(stato, A.ConfigAllena(righe_per_passo=32, righe_mc=8, righe_mc_permutate=True),
                          SimpleNamespaceMC(mc), train, reg.append)
    c["ripresa_segmento_nuovo"] = len(seg2) == 2 and seg2[1]["passo"] == 5 and seg2[1]["cursore_mc"] == 40 \
        and seg2[1].get("permutate") is True and any("PERMUTATO" in m for m in reg)
    try:
        A.ConfigAllena(righe_per_passo=32, righe_mc_permutate=True)
        c["rifiutata_senza_righe_mc"] = False
    except ValueError:
        c["rifiutata_senza_righe_mc"] = True
    c["decisione"] = "righe_mc_permutate" in A._CAMPI_DI_DECISIONE
    return registra("I14_righe_mc_permutate", all(c.values()), **c, secondi=round(time.time() - t0, 1))


class SimpleNamespaceMC:
    """Sorgente mc minima per `_segmenti_mc` (firma e righe mc)."""

    def __init__(self, mc: list):
        self._mc = mc

    def firma_mc(self) -> dict:
        return {"teacher_mc": "finto"}

    def righe_mc(self) -> list:
        return list(self._mc)


# ─────────────────────────────────────────── I12 ──────────────────────────────────────────
_I12_FIGLIO = r"""
import json, sys, time, torch
sys.path.insert(0, %r)
torch.set_num_threads(4)
import e2e_studente as S, e2e_lora as LR
f32 = S.famiglie_f32("attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,"
                     "ssm_a,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt,output_norm")
cfg = LR.ConfigLoRA(rango=16, alpha=None, famiglie=LR.famiglie_lora("tutte"), seme=20260926, tipi=("Q8_0",))
st = S.Studente(allena_f32=f32, lora=cfg, strati=[0, 3], ricetta="B3_ind32", motore_nome="torch", device="cpu",
                log=lambda *a, **k: None)
g = torch.Generator().manual_seed(4)
with torch.no_grad():
    for p in st.parametri():
        p.copy_(0.02 * torch.randn(p.shape, generator=g))
    for a in st.lora.values():
        a.B.copy_(0.01 * torch.randn(a.B.shape, generator=g, dtype=torch.float64).to(a.B.dtype))
ids = torch.randint(0, st.cfg.vocab_size, (1, 32), generator=g)
R = torch.randn(32, st.cfg.hidden_size, generator=g)
tutti = lambda: st.parametri() + st.parametri_f32() + st.parametri_lora()
def grad():
    for p in tutti():
        p.grad = None
    t0 = time.time()
    stato = st.avanti(ids)
    xL = stato["xL"].detach().requires_grad_(True)
    with torch.enable_grad():
        hS = st.esce(xL)
        (hS.float().reshape(32, -1) * R).sum().backward()
    t1 = time.time()
    st.indietro(stato, xL.grad)
    t2 = time.time()
    return [None if p.grad is None else p.grad.clone() for p in tutti()], t2 - t1
strato = ([L for (L, _) in st.chiavi()] + [st.f32[n].L for n in st.chiavi_f32()]
          + [st.lora[n].L for n in st.chiavi_lora() for _ in (0, 1)])
assert len(strato) == len(tutti()), (len(strato), len(tutti()))
g_i, t_i = grad()
out = {"n": len(strato), "t_intero": round(t_i, 2)}
for allenato in (3, 0):
    st.congela_strati([allenato])
    g_c, t_c = grad()
    st.congela_strati(None)
    out[f"allenato_{allenato}"] = {
        "uguali_al_bit": all(torch.equal(a, b) for a, b, L in zip(g_i, g_c, strato) if L in (allenato, -1)),
        "congelati_none": all(b is None for b, L in zip(g_c, strato) if L not in (allenato, -1)),
        "non_vuoti": sum(b is not None and bool(b.abs().sum() > 0) for b, L in zip(g_c, strato) if L == allenato),
        "t_indietro": round(t_c, 2)}
out["finiti"] = all(torch.isfinite(x).all().item() for x in g_i if x is not None)
print(json.dumps(out))
""" % str(QUI)


def prova_i12() -> bool:
    """(Thor-T VERO, strati 0 = GDN e 3 = attenzione piena; lenta, processo a parte) --strati-allenati sullo studente di
    EB3 (B3_ind32 + 13 famiglie F32 + LoRA Q8_0 sui LineareBF16x3, theta e B casuali): con lo strato 3 allenato
    (l'indietro si ferma al 3, come 20-39 in EB3) e con lo 0 allenato (il 3 congelato propaga solo dL/dx attraverso i
    LineareBF16x3 e la LoRA) i gradienti degli strati allenati e di output_norm sono AL BIT quelli dell'indietro intero;
    quelli degli strati congelati None."""
    import subprocess
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", _I12_FIGLIO], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""})
    if r.returncode != 0:
        return registra("I12_strati_congelati_thor_vero_B3", False, errore=r.stderr[-2500:])
    o = json.loads(r.stdout.strip().splitlines()[-1])
    c = {f"{k}_{v}": o[k][v] > 0 if v == "non_vuoti" else o[k][v]
         for k in ("allenato_3", "allenato_0") for v in ("uguali_al_bit", "congelati_none", "non_vuoti")}
    c["finiti"] = o["finiti"]
    return registra("I12_strati_congelati_thor_vero_B3", all(c.values()), **c, misure=o,
                    secondi=round(time.time() - t0, 1))


PROVE = {"I1": prova_i1, "I2": prova_i2, "I3": prova_i3, "I5": prova_i5, "I6": prova_i6, "I7": prova_i7, "I8": prova_i8,
         "I9": prova_i9, "I10": prova_i10, "I11": prova_i11, "I12": prova_i12, "I13": prova_i13, "I14": prova_i14}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = [a.upper() for a in sys.argv[1:]] or list(PROVE)
    ok = True
    t0 = time.time()
    for k in scelte:
        t1 = time.time()
        try:
            ok &= PROVE[k]()
        except Exception as e:                       # una prova che solleva e' ROSSA, con la traccia
            ok = False
            registra(f"{k}_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-3000:])
        print(f"   ({k}: {time.time() - t1:.1f} s)", flush=True)
    ESITO["tutto_verde"] = ok
    ESITO["secondi"] = round(time.time() - t0, 1)
    suff = "_" + "_".join(scelte) if sys.argv[1:] else ""
    (DIR / f"esito_test_integrata{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
