#!/usr/bin/env python3
"""Prova su CPU di `strumenti/prova_strumenti_f32.py` (28/9/2026) sul modello MINUSCOLO delle prove integrate (2 strati:
GDN + attenzione piena, esperti TQ1_0, non-esperti k-quant, LoRA-GA, 17 F32, passo V vero). Niente GPU, niente /mnt.

  P1 una corsa minuscola con la configurazione di EB3 in piccolo (Sophia, PV V-Bop, LoRA-GA, tutte le famiglie F32,
     istantanea a ogni passo) scrive passo_000002 e passo_000003; la prova:
     - `config_corsa` + `carica_studente` ricostruiscono dallo SOLO istantanea (GGUF + F32 + LoRA + theta + registri PV)
       lo studente della corsa AL BIT (theta, master F32, A/B/A0/B0, byte dei codici) e la KL per riga == `A.valuta`
       sullo studente della corsa;
     - `esegui` sui 6 casi: ogni caso == uno studente costruito A PARTE nel modo indipendente (famiglie riportate =
       studente che NON le allena e le carica dal GGUF; lora_spenta = studente SENZA LoRA; theta_p12 = theta del passo 2),
       KL in avanti e inversa per riga uguali al bit; dopo i 6 casi lo stato e' quello dell'istantanea al bit;
     - `tabella`/`scrivi` producono JSON e tabella; `guardia_gpu` risponde (lista di motivi).
Uso:
    nice -n 15 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 E2E_ALLENA_PROVE=<cartella> \\
        @@ROOT:venv@@/bin/python3 e2e_test_prova_strumenti.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(QUI / "strumenti"))
import e2e_allena as A  # noqa: E402
import e2e_test_integrata as TI  # noqa: E402  (modello minuscolo, config di EB3 in piccolo)
import e2e_test_lora as TL  # noqa: E402
import prova_strumenti_f32 as P  # noqa: E402

DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "prova_strumenti"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:3000]}", flush=True)
    return bool(ok)


def _per_riga(st, car, righe, ca, WS) -> dict:
    return P.valuta_righe(st, car, righe, ca, WS, WS)


def _uguali_righe(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[i]["kl"] == b[i]["kl"] and a[i]["kl_inv"] == b[i]["kl_inv"] for i in a)


def prova_p1() -> bool:
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    d, d_i = DIR / "p1_corsa", DIR / "p1_istantanee"
    TI._pulisci(d, d_i)
    ca = TI._ca_integrata(istantanea_ogni_passo=True, istantanee_dir=str(d_i), istantanee_tieni=2)
    st = TI._studente_int()
    z = A.allena(st, src, d, ca, WS, WS, log=muto)
    ist3, ist2 = d_i / "passo_000003", d_i / "passo_000002"
    # EB3 ha il gamma DECISO nella config (0,552); la corsa minuscola lo misura al passo 0: lo si scrive come EB3
    cfg = json.loads(json.dumps(dataclasses.asdict(dataclasses.replace(ca, gamma=z["gamma"]["valore"])), default=str))
    (d / "config.json").write_text(json.dumps({"config": cfg, "gguf": str(TL.gguf_lora()), "ricetta": "A32",
                                               "teacher": "(memoria)"}))
    c: dict = {}
    ca_v, cj = P.config_corsa(d)
    c["config_corsa"] = (ca_v.lora_rango == ca.lora_rango and ca_v.lora_init == "ga" and ca_v.gamma == z["gamma"]["valore"]
                         and ca_v.perdita == ca.perdita and len(ca_v.allena_f32) == 13)
    sp, gg, regs, passo = P.carica_studente(cj, ca_v, ist3, False, "cpu", log=muto)
    c["passo_e_registri"] = passo == 3 and len(regs) >= 1
    c["theta_al_bit"] = TI._uguali_lista(sp.parametri(), st.parametri())
    c["f32_al_bit"] = sp.chiavi_f32() == st.chiavi_f32() and TI._uguali_lista(sp.parametri_f32(), st.parametri_f32())
    c["lora_al_bit"] = TI._uguali_lora_ga(sp, st)
    c["codici_al_bit"] = TI._uguali_lista(TI._byte(sp), TI._byte(st))
    car = A.Caricatore(src, torch.device("cpu"))
    righe = src.righe_previste("heldout")
    rif_tutto = _per_riga(st, car, righe, ca_v, WS)
    theta2 = torch.load(ist2 / "theta.pt", map_location="cpu")
    esito = {"casi": {}}
    P.esegui(sp, gg, car, righe, ca_v, WS, WS, list(P.CASI), theta2, esito, uscita=DIR / "p1_esito", log=muto)
    cs = esito["casi"]
    c["tutto_uguale_alla_corsa"] = _uguali_righe(cs["tutto"]["per_riga"], rif_tutto)

    def a_parte(togli=(), **kw):
        cav = dataclasses.replace(ca_v, allena_f32=tuple(f for f in ca_v.allena_f32 if f not in togli), **kw)
        s2, _, _, _ = P.carica_studente(cj, cav, ist3, False, "cpu", log=muto)
        return s2

    rif = {"f32_5_gguf": a_parte(P.F32_5), "router_gguf": a_parte(("ffn_gate_inp",)),
           "ssm_ab_gguf": a_parte(("ssm_alpha", "ssm_beta")),
           # lora_spenta a parte = studente SENZA LoRA (nessun adattatore): la forma GA con A = A0, B = B0 deve dare lo stesso
           "lora_spenta": a_parte(lora_rango=0, lora_famiglie=(), lora_tipi=(), lora_alpha=None, lora_init="zero")}
    s12 = a_parte()
    P.imposta_theta(s12, theta2)
    rif["theta_p12"] = s12
    for caso, s2 in rif.items():
        c[f"{caso}_uguale_a_parte"] = _uguali_righe(cs[caso]["per_riga"], _per_riga(s2, car, righe, ca_v, WS))
        c[f"{caso}_diverso_da_tutto"] = not _uguali_righe(cs[caso]["per_riga"], cs["tutto"]["per_riga"])
    c["stato_finale_al_bit"] = (TI._uguali_lista(sp.parametri(), st.parametri()) and
                                TI._uguali_lista(sp.parametri_f32(), st.parametri_f32()) and TI._uguali_lora_ga(sp, st))
    c["di_nuovo_tutto"] = _uguali_righe(_per_riga(sp, car, righe, ca_v, WS), rif_tutto)
    c["uscite"] = (DIR / "p1_esito.json").exists() and "| tutto |" in (DIR / "p1_esito.md").read_text()
    c["guardia_risponde"] = isinstance(P.guardia_gpu(), list)
    car.chiudi()
    ok = all(c.values())
    return registra("P1_prova_strumenti_f32", ok, **c,
                    kl=[(k, round(v["per_dominio"].get("a", {}).get("kl", float("nan")), 6)) for k, v in cs.items()],
                    secondi=round(time.time() - t0, 1))


def prova_p2() -> bool:
    """La riga di riavvio proposta per EB3 in piccolo: corsa con LoRA-GA + tutte le famiglie F32 + PV (AdamW), poi ripresa
    con --allena-f32 senza le 5 famiglie, --f32-ripristina di quelle 5 e --strati-allenati 1. Lo stato di Adam dei F32
    rimasti e della LoRA (che nell'ordine dei parametri viene DOPO i F32: lo spostamento degli indici) deve arrivare ai
    parametri giusti: step di Adam = quello della ripresa + 1 negli strati allenati, invariato in quelli congelati; i
    tensori tolti senza stato (l'avanti con le famiglie al GGUF e' gia' provato al bit in A28 di e2e_test_f32)."""
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    d = DIR / "p2_corsa"
    TI._pulisci(d)
    ca = TI._ca_integrata(ottimizzatore_theta="adamw")
    A.allena(TI._studente_int(), src, d, ca, WS, WS, ferma_a=2, log=muto)
    r2 = A.carica_ripresa(d, "cpu")
    resto = tuple(f for f in ca.allena_f32 if f not in P.F32_5)
    ca3 = TI._ca_integrata(ottimizzatore_theta="adamw", allena_f32=resto, f32_ripristina=P.F32_5, strati_allenati=(1,))
    reg: list = []
    st3 = TL.studente("A32", TL.cfg_lora(4), allena_f32=resto)
    z3 = A.allena(st3, src, d, ca3, WS, WS, riprendi=True, ferma_a=3, log=reg.append)
    r3 = A.carica_ripresa(d, "cpu")
    n_th = len(r2["theta"])

    def passi(r):
        st_ = r["opt"]["state"]
        f = {n: int(st_[n_th + j]["step"]) if (n_th + j) in st_ else 0 for j, n in enumerate(r["f32_nomi"])}
        o = n_th + len(r["f32_nomi"])
        lo = {f"{n}#{k}": (int(st_[o + 2 * j + i]["step"]) if (o + 2 * j + i) in st_ else 0,
                           tuple(st_[o + 2 * j + i]["exp_avg"].shape) if (o + 2 * j + i) in st_ else None)
              for j, n in enumerate(r["lora_nomi"]) for i, k in enumerate(("A", "B"))}
        return f, lo

    f2, l2 = passi(r2)
    f3, l3 = passi(r3)
    strato = lambda n: int(n.split(".")[1]) if n.startswith("blk.") else -1  # noqa: E731
    c: dict = {}
    c["ripresa_passa"] = z3["passo"] == 3 and any("TOLTI" in x for x in reg)
    c["tolti_senza_stato"] = sorted(r3["f32_nomi"]) == sorted(n for n in r2["f32_nomi"]
                                                               if S_fam(n) not in P.F32_5)
    c["f32_rimasti_step"] = all(f3[n] == f2[n] + (1 if strato(n) in (1, -1) else 0) for n in r3["f32_nomi"])
    c["lora_step_e_forme"] = (l3.keys() == l2.keys() and all(
        l3[k][1] == l2[k][1] and l3[k][0] == l2[k][0] + (1 if int(k.split(".")[1]) == 1 else 0) for k in l3))
    c["famiglie_fuori_dallo_studente"] = not any(t.famiglia in P.F32_5 for t in st3.f32.values())
    ok = all(c.values())
    return registra("P2_riavvio_EB3_in_piccolo", ok, **c, voce=z3.get("f32_ripristini"), secondi=round(time.time() - t0, 1))


def S_fam(n: str) -> str:
    import e2e_studente as S
    return S.famiglia_del_tensore_f32(n)


PROVE = {"P1": prova_p1, "P2": prova_p2}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = [a.upper() for a in sys.argv[1:]] or list(PROVE)
    ok, t0 = True, time.time()
    for k in scelte:
        try:
            ok &= PROVE[k]()
        except Exception as e:
            ok = False
            registra(f"{k}_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-3000:])
    ESITO["tutto_verde"] = ok
    ESITO["secondi"] = round(time.time() - t0, 1)
    (DIR / "esito_test_prova_strumenti.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
