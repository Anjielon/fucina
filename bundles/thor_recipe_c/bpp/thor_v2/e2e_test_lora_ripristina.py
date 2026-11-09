#!/usr/bin/env python3
"""Prove di `--lora-ripristina {zero,ferma}` di e2e_allena (28/9/2026). SOLO CPU, niente /mnt tranne la cartella delle
prove: il modello MINUSCOLO delle prove LoRA/integrate (2 strati: GDN + attenzione piena, esperti TQ1_0, non-esperti
k-quant veri, 13 adattatori). Il riferimento di ogni ripresa e' costruito A PARTE, con un altro meccanismo: la ripresa
«operata a mano» (A/B scritti nel file, stato di Adam della LoRA tolto per INDICE, oppure LoRA tolta del tutto e studente
senza LoRA), ripresa con la riga di sempre; la corsa con l'opzione deve darne lo stesso stato AL BIT.

  R1 config e riga di comando: default "" (campo di decisione, default nella config, gruppi dell'ottimizzatore di sempre),
     argparse vero (zero/ferma accettati, altro rifiutato), senza LoRA rifiutato, modo ignoto rifiutato; corsa NUOVA con
     zero == senza (al bit, e il registro lo dice).
  R2 zero, LoRA di sempre (init zero), AdamW + F32: alla ripresa Delta W = 0 esatto (B = 0, A tenuto) e avanti == studente
     SENZA LoRA; un passo == ripresa operata a mano (B = 0 nel file, Adam della LoRA tolto per indice) AL BIT (theta, F32,
     LoRA, tutto lo stato di Adam, stato della corsa); voce in stato["lora_ripristini"] con |Delta W|_F per adattatore ==
     calcolo indipendente in fp64; registro; step di Adam (LoRA 1, theta e F32 +1); rilancio con la stessa riga: niente
     secondo ripristino, == riferimento al passo 4; errori (zero + --lora-reinit-ga su LoRA a init zero).
  R3 la riga di EB3 in piccolo (analoga a P2 di e2e_test_prova_strumenti): LoRA-GA + tutte le famiglie F32 + PV V-Bop +
     CAKLD + attribuzione, AdamW, alla ripresa --lora-ripristina zero --lr-lora 1e-5 --strati-allenati 1
     --lora-reinit-ga: alla ripresa A = A0, B = B0 e avanti == studente SENZA LoRA; un passo == ripresa operata a mano
     (A = A0, B = B0 nel file) AL BIT, byte dei codici compresi; negli adattatori dello
     strato congelato A = A0, B = B0 e nessuno stato di Adam, in quelli allenati step 1 e forme giuste; F32 e theta con lo
     step giusto (indici della LoRA spostati dietro ai F32); rilancio idempotente; zero con --lora-init zero su una LoRA GA
     rifiutato. Variante Sophia (gruppo AdamW di F32/LoRA dentro Sophia) == operata a mano AL BIT.
  R4 ferma: (a) AdamW + --f32-ripristina (5 famiglie TOLTE: indici dei F32 spostati) + LoRA-GA ferma == corsa SENZA LoRA
     ripresa dal file operato a mano (LoRA e suo gruppo tolti) AL BIT (theta, F32, Adam, stato, byte dei codici); niente
     A/B nella ripresa e nell'istantanea; 2 gruppi nell'ottimizzatore; adattatori mai toccati; rilancio idempotente; export
     dell'istantanea == export della corsa senza LoRA (sha256); (b) Sophia senza F32 (l'AdamW di Sophia sparisce) == senza
     LoRA AL BIT; (c) togliere ferma alla ripresa = LoRA accesa ora (LoRA-GA, Adam nuovo).
  R5 voce del ripristino (forma zero / ga / mista, adattatori GA, norme == calcolo indipendente) ed export con Delta W = 0
     esatto (LoRA di sempre azzerata, LoRA-GA con A = A0, B = B0, e le due mescolate), modi griglia E ggml: GGUF ==
     quello senza LoRA (sha256); il modo ggml da capo avrebbe cambiato i byte (lo si misura: la prova ha un oggetto); con
     una LoRA azzerata solo in parte i tensori nulli tengono i byte del sorgente e gli altri == riquantizza(W + dW).
  R6 la riga di ripresa di EB3 dal passo 20 (argv della config di EB3 + --lora-ripristina zero --lr-lora 1e-5): il parser
     vero la accetta e la config differisce da quella di EB3 SOLO in lr_lora e lora_ripristina, campi di decisione (lettura
     di @@ROOT:depot@@/e2e/allena/EB3/config.json; saltata se assente).
Uso (CPU limitata, niente GPU):
    systemd-run --user --scope -p CPUQuota=200% nice -n 19 ionice -c3 env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \\
        OMP_NUM_THREADS=2 E2E_ALLENA_PROVE=<cartella> @@ROOT:venv@@/bin/python3 e2e_test_lora_ripristina.py [R1 ...]
    esito: $E2E_ALLENA_PROVE/lora_ripristina/esito_test_lora_ripristina.json
"""
from __future__ import annotations

import argparse
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
import e2e_allena as A  # noqa: E402
import e2e_lora as LR  # noqa: E402
import e2e_studente as S  # noqa: E402
import e2e_test_integrata as TI  # noqa: E402  (modello minuscolo, configurazione di EB3 in piccolo)
import e2e_test_f32 as TF  # noqa: E402
import e2e_test_lora as TL  # noqa: E402
from e2e_test_f32 import X, GGUFReader  # noqa: E402

DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "lora_ripristina"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
FAM_R2 = S.famiglie_f32("attn_norm,ffn_gate_inp,ssm_dt")
F32_5 = S.famiglie_f32("ffn_gate_inp,ffn_gate_inp_shexp,ssm_alpha,ssm_beta,ssm_conv1d")   # le 5 di prova_strumenti
_LORA_SPENTA = {k: A._CONFIG_DEFAULT[k] for k in ("lora_rango", "lora_alpha", "lora_famiglie", "lora_tipi", "lora_init",
                                                  "lora_ga_gamma", "lora_ga_righe")}
EB3_CONFIG = Path(os.environ.get("E2E_EB3_CONFIG", "@@ROOT:depot@@/e2e/allena/EB3/config.json"))


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:3000]}", flush=True)
    return bool(ok)


# ─────────────────────────────── aiuti ───────────────────────────────
def _pulita(nome: str) -> Path:
    d = DIR / nome
    shutil.rmtree(d, ignore_errors=True)
    return d


def _copia(d0: Path, nome: str) -> Path:
    d = _pulita(nome)
    shutil.copytree(d0, d)
    return d


def _riscrivi_ripresa(d: Path, r: dict) -> None:
    """La ripresa operata a mano torna nel file dell'ultima ripresa, col suo sha256 nel manifesto e nell'indice."""
    man = json.loads((d / "ripresa_ultima.json").read_text())
    sha = A._torch_save_atomico(r, d / man["file"])
    man["sha256"] = sha
    (d / "ripresa_ultima.json").write_text(json.dumps(man, indent=1))
    idx = json.loads((d / "riprese.json").read_text())
    for v in idx:
        if v["file"] == man["file"]:
            v["sha256"] = sha
    (d / "riprese.json").write_text(json.dumps(idx, indent=1))


def _chirurgia_zero(d: Path) -> None:
    """Riferimento di `zero`: nel FILE A/B azzerati a mano (B = 0; forma GA A = A0, B = B0) e lo stato di Adam dei
    parametri della LoRA tolto per INDICE (il gruppo resta): la ripresa di sempre li ricarica e Adam riparte da zero."""
    r = A.carica_ripresa(d, "cpu")
    for v in r["lora"].values():
        if "A0" in v:
            v["A"], v["B"] = v["A0"].clone(), v["B0"].clone()
        else:
            v["B"] = torch.zeros_like(v["B"])
    sophia = r["opt"].get("tipo") == "sophia"
    sd = r["opt"]["adam"] if sophia else r["opt"]
    off = (0 if sophia else len(r["theta"])) + len(r.get("f32_nomi", []))
    for j in range(2 * len(r["lora_nomi"])):
        sd["state"].pop(off + j, None)
    _riscrivi_ripresa(d, r)


def _chirurgia_senza_lora(d: Path) -> None:
    """Riferimento di `ferma`: la ripresa di una corsa SENZA LoRA (A/B e ordine tolti, gruppo della LoRA tolto dallo stato
    di Adam, config della LoRA ai default), da riprendere con uno studente senza adattatori."""
    r = A.carica_ripresa(d, "cpu")
    n = 2 * len(r.pop("lora_nomi"))
    r.pop("lora")
    sophia = r["opt"].get("tipo") == "sophia"
    sd = r["opt"]["adam"] if sophia else r["opt"]
    g = sd["param_groups"].pop()
    assert len(g["params"]) == n, (len(g["params"]), n)
    for i in g["params"]:
        sd["state"].pop(i, None)
    if sophia and not sd["param_groups"]:
        r["opt"]["adam"] = None
    r["stato"]["config"].update(_LORA_SPENTA)
    _riscrivi_ripresa(d, r)


def _senza_lora(ca):
    return dataclasses.replace(ca, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in _LORA_SPENTA.items()},
                               lora_ripristina="")


def _uguali_lora_dati(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[n].keys() == b[n].keys() and all(torch.equal(a[n][k], b[n][k]) for k in a[n])
                                        for n in a)


def _riprese_uguali(ra: dict, rb: dict) -> dict:
    """Confronto AL BIT di due riprese: theta, F32 (valori e ordine), LoRA (A, B, A0, B0), EMA, stato dell'ottimizzatore."""
    c = {"theta": ra["theta"].keys() == rb["theta"].keys() and all(torch.equal(ra["theta"][k], rb["theta"][k])
                                                                    for k in ra["theta"]),
         "f32": ra.get("f32_nomi") == rb.get("f32_nomi") and all(torch.equal(torch.as_tensor(ra["f32"][n]),
                                                                            torch.as_tensor(rb["f32"][n]))
                                                                for n in ra.get("f32_nomi", [])),
         "lora": ra.get("lora_nomi") == rb.get("lora_nomi") and _uguali_lora_dati(ra.get("lora", {}), rb.get("lora", {})),
         "ema": (ra.get("ema") is None) == (rb.get("ema") is None) and (
             ra.get("ema") is None or all(torch.equal(ra["ema"][k], rb["ema"][k]) for k in ra["ema"])),
         "adam": TI._uguali_stato_opt(ra["opt"], rb["opt"])}
    return c


def _stato_uguale(za: dict, zb: dict, via=("config", "lora_ripristini")) -> bool:
    a, b = TI._stato_confrontabile(za), TI._stato_confrontabile(zb)
    for k in via:
        a.pop(k, None)
        b.pop(k, None)
    return a == b


def _uscita(st):
    ids, x, _ = TF._ingresso(st)
    with torch.no_grad():
        return st.avanti(ids, x0=x)["xL"]


def _senza_lora_come(st, fam):
    """Studente SENZA LoRA con theta e F32 di `st`."""
    sn = TL.studente("A32", None, allena_f32=fam)
    with torch.no_grad():
        for p, q in zip(sn.parametri(), st.parametri()):
            p.copy_(q)
    if fam:
        sn.imposta_f32(st.valori_f32())
    return sn


def _norme_indipendenti(lora: dict, st) -> dict:
    """|Delta W|_F per adattatore formando Delta W in fp64 (s B A, o s (B A - B0 A0)): il calcolo che la voce evita."""
    out = {}
    for n, v in lora.items():
        s = st.lora[n].scala
        A_, B_ = v["A"].double(), v["B"].double()
        D = B_ @ A_ if "A0" not in v else B_ @ A_ - v["B0"].double() @ v["A0"].double()
        out[n] = float(torch.linalg.norm(D * s))
    return out


def _voce_ok(voce: dict, passo: int, modo: str, forma: str, lora: dict, st) -> dict:
    ind = _norme_indipendenti(lora, st)
    per = voce.get("per_adattatore", {})
    tot = math.sqrt(sum(x * x for x in ind.values()))
    return {"passo_modo_forma": (voce.get("passo"), voce.get("modo"), voce.get("forma")) == (passo, modo, forma),
            "adattatori": voce.get("adattatori") == len(lora) and per.keys() == ind.keys(),
            "norme_per_adattatore": all(ind[n] > 0 and abs(per[n] - ind[n]) <= 1e-9 * ind[n] for n in ind),
            "norma_totale": abs(voce.get("norma_totale", -1) - tot) <= 1e-9 * tot}


def _passi(r: dict) -> tuple:
    """(theta, F32, LoRA): {nome: step di Adam} dalla ripresa AdamW (0 = senza stato); per la LoRA anche la forma."""
    st_ = r["opt"]["state"]
    n_th = len(r["theta"])
    th = {k: int(st_[i]["step"]) if i in st_ else 0 for i, k in enumerate(r["theta"])}
    f = {n: int(st_[n_th + j]["step"]) if (n_th + j) in st_ else 0 for j, n in enumerate(r.get("f32_nomi", []))}
    o = n_th + len(r.get("f32_nomi", []))
    lo = {f"{n}#{k}": (int(st_[o + 2 * j + i]["step"]) if (o + 2 * j + i) in st_ else 0,
                       tuple(st_[o + 2 * j + i]["exp_avg"].shape) if (o + 2 * j + i) in st_ else None)
          for j, n in enumerate(r.get("lora_nomi", [])) for i, k in enumerate(("A", "B"))}
    return th, f, lo


def _strato(nome: str) -> int:
    return int(nome.split(".")[1]) if nome.startswith("blk.") else -1


def _parse_cli(argv: list):
    """Il Namespace del parser VERO di `e2e_allena.main` (fermato subito dopo parse_args: nessun effetto)."""
    class Preso(Exception):
        pass
    orig = argparse.ArgumentParser.parse_args

    def spia(self, args=None, namespace=None):
        raise Preso(orig(self, args, namespace))
    argparse.ArgumentParser.parse_args = spia
    try:
        A.main(argv)
    except Preso as e:
        return e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError("parse_args non raggiunto")


# ─────────────────────────────── R1 ───────────────────────────────
def prova_r1() -> bool:
    t0 = time.time()
    c: dict = {}
    c["default"] = (A.ConfigAllena().lora_ripristina == "" and A._CONFIG_DEFAULT.get("lora_ripristina") == ""
                    and "lora_ripristina" in A._CAMPI_DI_DECISIONE
                    and A.nomi_gruppi(TL._ca_lora()) == ["theta", "lora"]
                    and A.nomi_gruppi(TL._ca_lora(lora_ripristina="zero")) == ["theta", "lora"]
                    and A.nomi_gruppi(TL._ca_lora(lora_ripristina="ferma", allena_f32=FAM_R2)) == ["theta", "f32"])
    base = ["allena", "--corsa", "x", "--teacher", "y", "--lora-nonesperti", "4"]
    a_z, a_f, a_0 = _parse_cli(base + ["--lora-ripristina", "zero"]), _parse_cli(base + ["--lora-ripristina", "ferma"]), \
        _parse_cli(base)
    c["argparse"] = (A._config_da_arg(a_z).lora_ripristina == "zero" and A._config_da_arg(a_f).lora_ripristina == "ferma"
                     and A._config_da_arg(a_0).lora_ripristina == "")
    try:
        _parse_cli(base + ["--lora-ripristina", "tutto"])
        c["modo_ignoto_rifiutato"] = False
    except SystemExit:
        c["modo_ignoto_rifiutato"] = True
    rif = {}
    for nome, kw in (("senza_lora", dict(lora_ripristina="zero")), ("modo", dict(lora_rango=4, lora_famiglie=("attn_q",),
                                                                                lora_ripristina="zero,ferma"))):
        try:
            A.ConfigAllena(**kw)
            rif[nome] = False
        except ValueError:
            rif[nome] = True
    try:
        A._config_da_arg(_parse_cli(["allena", "--corsa", "x", "--teacher", "y", "--lora-ripristina", "zero"]))
        rif["riga_senza_lora"] = False
    except ValueError:
        rif["riga_senza_lora"] = True
    c["rifiuti"] = all(rif.values())
    # corsa NUOVA con zero == senza (al bit), e il registro dice che non ha effetto
    src, WS = TL.sorgente_lora(n_train=8)
    d_a, d_b = _pulita("r1_senza"), _pulita("r1_zero")
    za = A.allena(TL.studente("A32", TL.cfg_lora(4)), src, d_a, TL._ca_lora(), WS, WS, ferma_a=2, log=muto)
    reg: list = []
    zb = A.allena(TL.studente("A32", TL.cfg_lora(4)), src, d_b, TL._ca_lora(lora_ripristina="zero"), WS, WS, ferma_a=2,
                  log=reg.append)
    ug = _riprese_uguali(A.carica_ripresa(d_a, "cpu"), A.carica_ripresa(d_b, "cpu"))
    c["corsa_nuova_zero_uguale"] = all(ug.values()) and _stato_uguale(za, zb) and "lora_ripristini" not in zb
    c["corsa_nuova_registro"] = any("--lora-ripristina zero senza effetto" in x for x in reg)
    ok = all(c.values())
    return registra("R1_config_e_riga", ok, **c, rifiuti_dettaglio=rif, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── R2 ───────────────────────────────
def prova_r2() -> bool:
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12)                       # 6 passi da 2 righe
    ca = TL._ca_lora(allena_f32=FAM_R2)
    nuovo = lambda: TL.studente("A32", TL.cfg_lora(4), allena_f32=FAM_R2)  # noqa: E731
    d0 = _pulita("r2_base")
    A.allena(nuovo(), src, d0, ca, WS, WS, ferma_a=2, log=muto)
    r2 = A.carica_ripresa(d0, "cpu")
    caz = dataclasses.replace(ca, lora_ripristina="zero", lr_lora=1e-3)
    car = dataclasses.replace(ca, lr_lora=1e-3)                  # riferimento: la riga di sempre (stesso lr_lora)
    c: dict = {}
    c["lora_allenata_nella_base"] = all(float(v["B"].abs().max()) > 0 for v in r2["lora"].values())
    # (a) alla ripresa, prima di ogni passo (ferma_a = passo della ripresa: nessun passo, nulla salvato)
    d_a = _copia(d0, "r2_a")
    st_a = nuovo()
    A.allena(st_a, src, d_a, caz, WS, WS, riprendi=True, ferma_a=2, log=muto)
    c["delta_zero_esatto"] = all(not a.delta_file().any() for a in st_a.lora.values())
    c["B_zero_A_tenuto"] = all(float(a.B.abs().max()) == 0 and torch.equal(a.A, r2["lora"][n]["A"])
                               for n, a in st_a.lora.items())
    c["avanti_come_senza_lora"] = torch.equal(_uscita(st_a), _uscita(_senza_lora_come(st_a, FAM_R2)))
    # (b) un passo == la ripresa operata a mano, al bit
    d_z, d_r = _copia(d0, "r2_zero"), _copia(d0, "r2_rif")
    _chirurgia_zero(d_r)
    reg: list = []
    st_z = nuovo()
    z3 = A.allena(st_z, src, d_z, caz, WS, WS, riprendi=True, ferma_a=3, log=reg.append)
    f3 = A.allena(nuovo(), src, d_r, car, WS, WS, riprendi=True, ferma_a=3, log=muto)
    rz, rr = A.carica_ripresa(d_z, "cpu"), A.carica_ripresa(d_r, "cpu")
    ug = _riprese_uguali(rz, rr)
    c["uguale_alla_ripresa_operata"] = all(ug.values()) and _stato_uguale(z3, f3)
    voci = z3.get("lora_ripristini", [])
    vo = _voce_ok(voci[0], 2, "zero", "zero", r2["lora"], st_z) if len(voci) == 1 else {"una_voce": False}
    c["voce"] = all(vo.values())
    c["registro"] = (any("--lora-ripristina zero al passo 2" in x and "B = 0, A tenuto" in x for x in reg)
                     and sum("|Delta W| scartato strato" in x for x in reg) == 2
                     and any("lora_ripristina  -> zero" in x and "lr_lora 0.003 -> 0.001" in x for x in reg))
    th2, f2, _ = _passi(r2)
    th3, f3p, lo3 = _passi(rz)
    c["adam_step"] = (all(th3[k] == th2[k] + 1 for k in th3) and all(f3p[n] == f2[n] + 1 for n in f3p)
                      and all(v[0] == 1 for v in lo3.values()) and rz["opt"]["param_groups"][2]["lr"] == 1e-3
                      and len(rz["opt"]["param_groups"]) == 3)
    # (c) rilancio con la stessa riga: niente secondo ripristino (== riferimento al passo 4)
    reg4: list = []
    z4 = A.allena(nuovo(), src, d_z, caz, WS, WS, riprendi=True, ferma_a=4, log=reg4.append)
    f4 = A.allena(nuovo(), src, d_r, car, WS, WS, riprendi=True, ferma_a=4, log=muto)
    ug4 = _riprese_uguali(A.carica_ripresa(d_z, "cpu"), A.carica_ripresa(d_r, "cpu"))
    c["idempotente"] = (all(ug4.values()) and _stato_uguale(z4, f4) and len(z4.get("lora_ripristini", [])) == 1
                        and not any("--lora-ripristina zero al passo" in x for x in reg4)
                        and z4["config"]["lora_ripristina"] == "zero")
    # (d) zero + --lora-reinit-ga su una LoRA a init zero: entrambi scartano, si sceglie (errore chiaro)
    d_e = _copia(d0, "r2_errore")
    try:
        A.allena(nuovo(), src, d_e, dataclasses.replace(caz, lora_init="ga"), WS, WS, riprendi=True, ferma_a=3,
                 log=muto, reinit_lora_ga=True)
        c["zero_con_reinit_rifiutato"] = False
    except RuntimeError as e:
        c["zero_con_reinit_rifiutato"] = "--lora-reinit-ga insieme" in str(e)
    ok = all(c.values())
    return registra("R2_zero_lora_di_sempre", ok, **c, confronto=ug, voce_dettaglio=vo, confronto_4=ug4,
                    norma_scartata=voci[0].get("norma_totale") if voci else None, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── R3 ───────────────────────────────
def prova_r3() -> bool:
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=24, n_held=4)            # 4 righe a passo: 6 passi
    ca = TI._ca_integrata(ottimizzatore_theta="adamw")
    d0 = _pulita("r3_base")
    A.allena(TI._studente_int(), src, d0, ca, WS, WS, ferma_a=2, log=muto)
    r2 = A.carica_ripresa(d0, "cpu")
    ca3 = TI._ca_integrata(ottimizzatore_theta="adamw", lora_ripristina="zero", lr_lora=1e-5, strati_allenati=(1,))
    ca3r = dataclasses.replace(ca3, lora_ripristina="")
    c: dict = {}
    c["base_ga_allenata"] = all("A0" in v and not torch.equal(v["B"], v["B0"]) for v in r2["lora"].values())
    # alla ripresa, prima di ogni passo: forma GA con A = A0, B = B0 (A0/B0 della ripresa), avanti == SENZA LoRA
    st_a = TI._studente_int()
    A.allena(st_a, src, _copia(d0, "r3_a"), ca3, WS, WS, riprendi=True, ferma_a=2, log=muto, reinit_lora_ga=True)
    c["ga_A0_B0_e_avanti_senza_lora"] = (
        all(torch.equal(a.A, r2["lora"][n]["A0"]) and torch.equal(a.B, r2["lora"][n]["B0"]) and a.A0 is not None
            and not a.delta_file().any() for n, a in st_a.lora.items())
        and torch.equal(_uscita(st_a), _uscita(_senza_lora_come(st_a, TI.FAM))))
    d_z, d_r = _copia(d0, "r3_zero"), _copia(d0, "r3_rif")
    _chirurgia_zero(d_r)
    reg: list = []
    st_z, st_r = TI._studente_int(), TI._studente_int()
    z3 = A.allena(st_z, src, d_z, ca3, WS, WS, riprendi=True, ferma_a=3, log=reg.append, reinit_lora_ga=True)
    f3 = A.allena(st_r, src, d_r, ca3r, WS, WS, riprendi=True, ferma_a=3, log=muto, reinit_lora_ga=True)
    rz, rr = A.carica_ripresa(d_z, "cpu"), A.carica_ripresa(d_r, "cpu")
    ug = _riprese_uguali(rz, rr)
    c["uguale_alla_ripresa_operata"] = (all(ug.values()) and _stato_uguale(z3, f3)
                                        and TI._uguali_lista(TI._byte(st_z), TI._byte(st_r)))
    voci = z3.get("lora_ripristini", [])
    vo = _voce_ok(voci[0], 2, "zero", "ga", r2["lora"], st_z) if len(voci) == 1 else {"una_voce": False}
    c["voce"] = all(vo.values())
    c["registro"] = (any("--lora-ripristina zero al passo 2" in x and "A = A0, B = B0" in x for x in reg)
                     and any("--lora-reinit-ga senza effetto" in x for x in reg))
    lz = rz["lora"]
    c["strato_congelato_a_zero"] = all(torch.equal(v["A"], v["A0"]) and torch.equal(v["B"], v["B0"])
                                       for n, v in lz.items() if _strato(n) == 0)
    c["strato_allenato_mosso"] = any(not torch.equal(v["B"], v["B0"]) for n, v in lz.items() if _strato(n) == 1)
    th2, f2, l2 = _passi(r2)
    th3, f3p, l3 = _passi(rz)
    c["lora_step_e_forme"] = (l3.keys() == l2.keys() and all(
        (l3[k][0], l3[k][1]) == ((1, l2[k][1]) if _strato(k) == 1 else (0, None)) for k in l3))
    c["f32_theta_step"] = (all(f3p[n] == f2[n] + (1 if _strato(n) in (1, -1) else 0) for n in f3p)
                           and all(th3[k] == th2[k] + (1 if k.startswith("1.") else 0) for k in th3)
                           and rz["opt"]["param_groups"][2]["lr"] == 1e-5)
    # rilancio con la stessa riga (idempotente) == riferimento al passo 4
    reg4: list = []
    z4 = A.allena(TI._studente_int(), src, d_z, ca3, WS, WS, riprendi=True, ferma_a=4, log=reg4.append, reinit_lora_ga=True)
    f4 = A.allena(TI._studente_int(), src, d_r, ca3r, WS, WS, riprendi=True, ferma_a=4, log=muto, reinit_lora_ga=True)
    ug4 = _riprese_uguali(A.carica_ripresa(d_z, "cpu"), A.carica_ripresa(d_r, "cpu"))
    c["idempotente"] = (all(ug4.values()) and _stato_uguale(z4, f4) and len(z4.get("lora_ripristini", [])) == 1
                        and not any("--lora-ripristina zero al passo" in x for x in reg4))
    # zero con --lora-init zero su una LoRA GA: la forma resterebbe GA, rifiutato
    try:
        A.allena(TI._studente_int(), src, _copia(d0, "r3_errore"), dataclasses.replace(
            ca3, lora_init="zero", lora_ga_gamma=16.0), WS, WS, riprendi=True, ferma_a=3, log=muto)
        c["init_diverso_rifiutato"] = False
    except RuntimeError as e:
        c["init_diverso_rifiutato"] = "--lora-ripristina zero la LoRA resta nella sua forma" in str(e)
    # variante Sophia: il gruppo AdamW di F32/LoRA dentro l'ottimizzatore di Sophia
    cs = TI._ca_integrata()
    ds0 = _pulita("r3s_base")
    A.allena(TI._studente_int(), src, ds0, cs, WS, WS, ferma_a=2, log=muto)
    ds_z, ds_r = _copia(ds0, "r3s_zero"), _copia(ds0, "r3s_rif")
    _chirurgia_zero(ds_r)
    zs = A.allena(TI._studente_int(), src, ds_z, dataclasses.replace(cs, lora_ripristina="zero", lr_lora=1e-5), WS, WS,
                  riprendi=True, ferma_a=3, log=muto)
    fs = A.allena(TI._studente_int(), src, ds_r, dataclasses.replace(cs, lr_lora=1e-5), WS, WS, riprendi=True, ferma_a=3,
                  log=muto)
    rs = A.carica_ripresa(ds_z, "cpu")
    ugs = _riprese_uguali(rs, A.carica_ripresa(ds_r, "cpu"))
    c["sophia_uguale_alla_ripresa_operata"] = (all(ugs.values()) and _stato_uguale(zs, fs) and rs["opt"]["tipo"] == "sophia"
                                               and len(zs.get("lora_ripristini", [])) == 1)
    ok = all(c.values())
    return registra("R3_riga_EB3_in_piccolo_zero", ok, **c, confronto=ug, confronto_4=ug4, confronto_sophia=ugs,
                    voce_dettaglio=vo, norma_scartata=voci[0].get("norma_totale") if voci else None,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────── R4 ───────────────────────────────
def _esporta_sha(d_ist: Path, nome: str) -> str:
    ist = A.carica_istantanea(d_ist)
    dst = DIR / f"{nome}.gguf"
    for p in (dst, Path(str(dst) + ".provenienza.json"), dst.with_name(dst.name + ".parziale")):
        p.unlink(missing_ok=True)
    X.esporta(TL.gguf_lora(), dst, ist["scale"], ist["meta"], controlla_disco=False, log=muto, f32=ist.get("f32"),
              lora=ist.get("lora"), lora_meta=ist.get("lora_meta"), lora_modo="ggml" if TL.LIB_OK else "griglia")
    sha = X.sha256_file(dst)
    dst.unlink()
    return sha


def prova_r4() -> bool:
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=24, n_held=4)
    resto = tuple(f for f in TI.FAM if f not in F32_5)
    ca = TI._ca_integrata(ottimizzatore_theta="adamw")
    d0 = _pulita("r4_base")
    A.allena(TI._studente_int(), src, d0, ca, WS, WS, ferma_a=2, log=muto)
    r2 = A.carica_ripresa(d0, "cpu")
    c: dict = {}
    # (a) ferma + 5 famiglie F32 tolte (indici spostati) == corsa SENZA LoRA dalla ripresa operata a mano, al bit
    d_f, d_r = _copia(d0, "r4_ferma"), _copia(d0, "r4_rif")
    _chirurgia_senza_lora(d_r)
    kw = dict(ottimizzatore_theta="adamw", allena_f32=resto, f32_ripristina=F32_5, istantanea_ogni_passo=True,
              istantanee_tieni=2)
    caf = TI._ca_integrata(lora_ripristina="ferma", istantanee_dir=str(d_f / "ist"), **kw)
    car = _senza_lora(TI._ca_integrata(istantanee_dir=str(d_r / "ist"), **kw))
    st_f = TL.studente("A32", TL.cfg_lora(4), allena_f32=resto)
    st_r = TL.studente("A32", None, allena_f32=resto)
    costruzione = {n: a.valori() for n, a in TL.studente("A32", TL.cfg_lora(4), allena_f32=resto).lora.items()}
    reg: list = []
    z3 = A.allena(st_f, src, d_f, caf, WS, WS, riprendi=True, ferma_a=3, log=reg.append, reinit_lora_ga=True)
    f3 = A.allena(st_r, src, d_r, car, WS, WS, riprendi=True, ferma_a=3, log=muto)
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    ug = _riprese_uguali(rf, rr)
    c["uguale_alla_corsa_senza_lora"] = (all(ug.values()) and _stato_uguale(z3, f3)
                                         and TI._uguali_lista(TI._byte(st_f), TI._byte(st_r)))
    voci = z3.get("lora_ripristini", [])
    vo = _voce_ok(voci[0], 2, "ferma", "ga", r2["lora"], st_f) if len(voci) == 1 else {"una_voce": False}
    c["voce"] = all(vo.values()) and len(z3.get("f32_ripristini", [])) == 1
    c["niente_lora_nella_ripresa"] = "lora" not in rf and "lora_nomi" not in rf and len(rf["opt"]["param_groups"]) == 2
    th2, f2, _ = _passi(r2)
    _, f3p, _ = _passi(rf)
    c["f32_rimasti_step"] = (sorted(rf["f32_nomi"]) == sorted(n for n in r2["f32_nomi"]
                                                             if S.famiglia_del_tensore_f32(n) not in F32_5)
                             and all(f3p[n] == f2[n] + 1 for n in rf["f32_nomi"]))
    c["adattatori_mai_toccati"] = (all(_uguali_lora_dati({n: a.valori()}, {n: costruzione[n]}) for n, a in st_f.lora.items())
                                   and all(p.requires_grad for p in st_f.parametri_lora())
                                   and not getattr(st_f, "lora_ferma", False))
    c["registro"] = (any("LoRA FERMA" in x for x in reg) and any("--lora-ripristina ferma al passo 2" in x for x in reg))
    # rilancio con la stessa riga (idempotente) == riferimento al passo 4; istantanea senza A/B; export == senza LoRA
    reg4: list = []
    z4 = A.allena(TL.studente("A32", TL.cfg_lora(4), allena_f32=resto), src, d_f, caf, WS, WS, riprendi=True, ferma_a=4,
                  log=reg4.append, reinit_lora_ga=True)
    f4 = A.allena(TL.studente("A32", None, allena_f32=resto), src, d_r, car, WS, WS, riprendi=True, ferma_a=4, log=muto)
    ug4 = _riprese_uguali(A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu"))
    c["idempotente"] = (all(ug4.values()) and _stato_uguale(z4, f4) and len(z4.get("lora_ripristini", [])) == 1
                        and not any("--lora-ripristina ferma al passo" in x for x in reg4))
    i_f, i_r = d_f / "ist" / "passo_000004", d_r / "ist" / "passo_000004"
    meta_f = json.loads((i_f / "meta.json").read_text())
    c["istantanea_senza_lora"] = not (i_f / "lora.npz").exists() and "lora" not in meta_f
    c["export_uguale_senza_lora"] = _esporta_sha(i_f, "r4_ferma") == _esporta_sha(i_r, "r4_rif")
    # (b) Sophia senza F32: l'AdamW dentro Sophia aveva solo la LoRA e sparisce
    cs = TI._ca_integrata(allena_f32=())
    ds0 = _pulita("r4s_base")
    A.allena(TL.studente("A32", TL.cfg_lora(4)), src, ds0, cs, WS, WS, ferma_a=2, log=muto)
    ds_f, ds_r = _copia(ds0, "r4s_ferma"), _copia(ds0, "r4s_rif")
    _chirurgia_senza_lora(ds_r)
    zs = A.allena(TL.studente("A32", TL.cfg_lora(4)), src, ds_f, dataclasses.replace(cs, lora_ripristina="ferma"), WS, WS,
                  riprendi=True, ferma_a=3, log=muto)
    fs = A.allena(TL.studente("A32", None), src, ds_r, _senza_lora(cs), WS, WS, riprendi=True, ferma_a=3, log=muto)
    rs = A.carica_ripresa(ds_f, "cpu")
    ugs = _riprese_uguali(rs, A.carica_ripresa(ds_r, "cpu"))
    c["sophia_senza_adamw"] = (all(ugs.values()) and _stato_uguale(zs, fs) and rs["opt"]["tipo"] == "sophia"
                               and rs["opt"]["adam"] is None)
    # (c) togliere ferma alla ripresa = LoRA accesa ora (LoRA-GA prima del passo, Adam nuovo)
    reg5: list = []
    kw_c = {k: v for k, v in kw.items() if k != "istantanea_ogni_passo"}
    z5 = A.allena(TL.studente("A32", TL.cfg_lora(4), allena_f32=resto), src, d_f,
                  TI._ca_integrata(istantanee_dir=str(d_f / "ist"), **kw_c), WS, WS, riprendi=True, ferma_a=5,
                  log=reg5.append)
    r5 = A.carica_ripresa(d_f, "cpu")
    _, _, l5 = _passi(r5)
    c["tolta_ferma_accesa_ora"] = (any("LoRA accesa ora" in x for x in reg5) and (z5.get("lora_ga") or {}).get("passo") == 4
                                   and r5.get("lora_nomi") == st_f.chiavi_lora() and l5 and all(v[0] == 1 for v in l5.values()))
    ok = all(c.values())
    return registra("R4_ferma", ok, **c, confronto=ug, confronto_4=ug4, confronto_sophia=ugs, voce_dettaglio=vo,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────── R5 ───────────────────────────────
def _esporta(ist: dict, nome: str, lora: bool, modo: str) -> tuple:
    dst = DIR / f"{nome}.gguf"
    for p in (dst, Path(str(dst) + ".provenienza.json"), dst.with_name(dst.name + ".parziale")):
        p.unlink(missing_ok=True)
    prov = X.esporta(TL.gguf_lora(), dst, ist["scale"], ist["meta"], controlla_disco=False, log=muto,
                     lora=ist.get("lora") if lora else None, lora_meta=ist.get("lora_meta") if lora else None,
                     lora_modo=modo)
    return dst, prov


def prova_r5() -> bool:
    t0 = time.time()
    modi = ("griglia", "ggml") if TL.LIB_OK else ("griglia",)
    src = TL.gguf_lora()
    c: dict = {}
    dett: dict = {}
    for forma in ("zero", "ga", "mista"):
        st = TL.studente("A32", TL.cfg_lora(4))
        TF._theta_casuale(st, seme=4, s=0.1)                     # scale diverse dal sorgente: l'export non e' una copia
        g = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for i, a in enumerate(st.lora.values()):
                if forma == "ga" or (forma == "mista" and i % 2 == 0):
                    a.imposta_ga(torch.randn(a.A.shape, generator=g, dtype=torch.float64) * 0.05,
                                 torch.randn(a.B.shape, generator=g, dtype=torch.float64) * 0.05)
                    a.A.add_(0.3 * torch.randn(a.A.shape, generator=g, dtype=torch.float64).to(a.A.dtype))
                a.B.add_(0.5 * torch.randn(a.B.shape, generator=g, dtype=torch.float64).to(a.B.dtype))
        mosse = sum(1 for a in st.lora.values() if a.delta_file().any())
        reg: list = []                                           # la voce del ripristino (forma, norme) su questa LoRA
        voce = A.ripristino_lora(st, st.valori_lora(), "zero", 7, TL._ca_lora(), reg.append)
        vo = _voce_ok(voce, 7, "zero", forma, st.valori_lora(), st)
        n_ga = sum(1 for a in st.lora.values() if a.A0 is not None)
        c[f"{forma}_voce"] = (all(vo.values()) and voce["adattatori_ga"] == n_ga
                              and (forma != "mista" or (0 < n_ga < len(st.lora)
                                                        and any(f"sui {n_ga} in forma GA" in x for x in reg))))
        for a in st.lora.values():
            a.azzera_delta()
        d0 = _pulita(f"r5_{forma}_ist")
        d0.mkdir(parents=True)
        ist = A.carica_istantanea(d0 / A.salva_istantanea(d0, st, 1, {"metrica": 0.0}))
        dst0, _ = _esporta(ist, f"r5_{forma}_senza", False, "griglia")
        sha0 = X.sha256_file(dst0)
        per = {"adattatori_mossi_prima": mosse, "init_ga_nel_file": ist["lora_meta"].get("init") == (
            None if forma == "zero" else "ga")}
        for modo in modi:
            dst, prov = _esporta(ist, f"r5_{forma}_{modo}", True, modo)
            sp = prov["lora"]["sopravvivenza_per_tensore"]
            per[modo] = (X.sha256_file(dst) == sha0 and len(sp) == len(st.lora)
                         and all(v.get("delta_nullo") for v in sp.values()))
            dst.unlink()
        dst0.unlink()
        c[f"{forma}_uguale_senza_lora"] = mosse == len(st.lora) and all(v for v in per.values() if isinstance(v, bool))
        dett[forma] = per
    # il modo ggml da capo avrebbe spostato i byte: la prova ha un oggetto (misura, non si assume)
    if TL.LIB_OK:
        ts = {t.name: t for t in GGUFReader(str(src)).tensors}
        spostati = 0
        for n in TL.studente("A32", TL.cfg_lora(4)).lora:
            raw = np.ascontiguousarray(np.asarray(ts[n].data))
            tipo = ts[n].tensor_type.name
            spostati += int(not np.array_equal(LR.riquantizza(raw, LR.dequantizza(raw, tipo), tipo, "ggml"), raw))
        c["ggml_da_capo_sposterebbe"] = spostati > 0
        dett["ggml_tensori_che_cambierebbero"] = spostati
    # LoRA azzerata solo in parte (modo ggml, o griglia senza la libreria): nulli = byte del sorgente, altri = riquantizza
    modo = modi[-1]
    st = TL.studente("A32", TL.cfg_lora(4))
    TL._b_casuale(st, 0.5, seme=31)
    zero = set(list(st.lora)[::2])
    for n in zero:
        st.lora[n].azzera_delta()
    d1 = _pulita("r5_parte_ist")
    d1.mkdir(parents=True)
    ist = A.carica_istantanea(d1 / A.salva_istantanea(d1, st, 1, {"metrica": 0.0}))
    dst, prov = _esporta(ist, "r5_parte", True, modo)
    ta, tb = ({t.name: t for t in GGUFReader(str(p)).tensors} for p in (src, dst))
    parte = True
    for n, a in st.lora.items():
        va, vb = np.asarray(ta[n].data), np.asarray(tb[n].data)
        tipo = ta[n].tensor_type.name
        atteso = va if n in zero else LR.riquantizza(va, LR.dequantizza(va, tipo) + a.delta_file(), tipo, modo)
        parte &= np.array_equal(vb, atteso)
        parte &= bool(prov["lora"]["sopravvivenza_per_tensore"][n].get("delta_nullo")) == (n in zero)
    del ta, tb
    dst.unlink()
    c["in_parte"] = parte and 0 < len(zero) < len(st.lora)
    ok = all(c.values())
    return registra("R5_export_delta_zero", ok, **c, dettaglio=dett, modi=modi, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── R6 ───────────────────────────────
def prova_r6() -> bool:
    t0 = time.time()
    if not EB3_CONFIG.exists():
        return registra("R6_riga_EB3", True, saltata=f"{EB3_CONFIG} assente")
    cj = json.loads(EB3_CONFIG.read_text())
    argv = list(cj["argv"][1:]) + ["--lora-ripristina", "zero", "--lr-lora", "1e-5"]
    ca = A._config_da_arg(_parse_cli(argv))
    ora = json.loads(json.dumps(dataclasses.asdict(ca), default=str))
    prima = cj["config"]
    diversi = {k for k in set(prima) | set(ora) if prima.get(k, A._CONFIG_DEFAULT.get(k)) != ora.get(k)}
    ammessi = A._CAMPI_DI_ESECUZIONE | A._CAMPI_DI_DECISIONE | A._CAMPI_SOLO_G1
    c = {"riga_accettata": ca.lora_ripristina == "zero" and ca.lr_lora == 1e-5,
         "diversi_solo_lora": diversi == {"lr_lora", "lora_ripristina"} and not (diversi - ammessi),
         "lora_di_EB3": (ca.lora_init, ca.lora_rango, ca.lora_tipi, A.nomi_gruppi(ca)) == (
             "ga", 16, ("Q8_0",), ["theta", "f32", "lora"])}
    ok = all(c.values())
    return registra("R6_riga_EB3", ok, **c, diversi=sorted(diversi), argv_in_piu=argv[-4:],
                    secondi=round(time.time() - t0, 1))


PROVE = {"R1": prova_r1, "R2": prova_r2, "R3": prova_r3, "R4": prova_r4, "R5": prova_r5, "R6": prova_r6}


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
    (DIR / f"esito_test_lora_ripristina{'_' + '_'.join(scelte) if sys.argv[1:] else ''}.json").write_text(
        json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
