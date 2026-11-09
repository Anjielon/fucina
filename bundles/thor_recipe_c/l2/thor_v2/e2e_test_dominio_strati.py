#!/usr/bin/env python3
"""Prove del ramo `dominio-strati-b` (27/9/2026): --peso-dominio e --strati-allenati. SOLO CPU (la GPU e' di EB2), studente
VERO minuscolo (GGUF sintetico di e2e_test_lora: esperti TQ1_0, non-esperti k-quant, 2 strati GDN + attenzione piena) e,
per il tempo, un giocattolo a 8 strati (2 x [GDN, GDN, GDN, attenzione piena], come Thor-T in piccolo).

  D1  default AL BIT: (a) contro il commit di partenza 39d514a (ramo integrata, git archive, UN ALTRO processo): la
      configurazione integrata (Sophia, V-Bop, rho adattivo, CAKLD, F32 tutte, LoRA-GA, pesi kl, V senza BH, EMA,
      attribuzione) per 3 passi -> theta, F32, LoRA (A, B, A0, B0), byte dei codici, stato dell'ottimizzatore, EMA, stato
      del ciclo IDENTICI; (b) pesi per dominio tutti uguali (normalizzati = 1) == spento al bit; (c) strati allenati =
      tutti == spento al bit.
  D2  pesi per dominio: (a) testo -> forma canonica e rifiuti (peso <= 0, NaN, doppi, sintassi, strati); (b)
      normalizzazione: media 1 (1e-12), rapporti chiesti, righe mc -> dominio `mc`, non elencati 1; (c) gradiente del
      passo con i pesi == (1/n) sum_r w_r g_r (gradienti delle righe una per una) a 1e-5 relativo, anche per il gradiente
      di F32 e LoRA; (d) moltiplicativo coi pesi per token kl (stessa identita'); (e) curvatura del passo V: dL/dxL del
      vettore campionato == sqrt(w) x quello senza pesi; (f) ciclo: voce "peso_dominio" (c, quote a somma 1), allenamento
      diverso da spento, dominio ignoto rifiutato, ripresa da una corsa senza pesi accettata (decisione registrata).
  D3  strati congelati sullo studente minuscolo (strato 0 congelato): (a) gradienti dello strato 1 (theta, F32, LoRA) e
      di output_norm IDENTICI al bit all'indietro intero, strato 0 senza gradiente; (b) ciclo integrato 3 passi: tutto lo
      stato dello strato 0 (theta, F32, LoRA, byte, stato di Sophia/AdamW assente, V-Bop, EMA) identico all'inizio al bit,
      salti solo nello strato 1, strato 1 mosso, studente restituito tutto allenabile; ripresa 1 + 2 == 3 filati al bit;
      (c) AdamW: 2 passi tutto, 1 congelato (stato di Adam dello strato 0 al bit), 1 scongelato (passi di Adam diversi:
      nessun errore, rumore precondizionato finito, lo strato 0 si muove).
  D4  ripresa di tipo EB2 (tutto allenato: LoRA a init zero allenata, AdamW, V-Bop, F32, CAKLD, EMA 0,9) verso EB3 con
      --strati-allenati 1 e --lora-reinit-ga: accettata; dopo 1 passo lo strato 0 ha theta, F32, LoRA (valori e Adam),
      byte, V-Bop ed EMA della ripresa AL BIT e m di Sophia = exp_avg di Adam; strato 1: LoRA re-inizializzata GA (A = A0,
      B = B0 al primo passo), Adam della LoRA da zero, theta mosso.
  D5  giocattolo a 8 strati, allenati 4-7: gradienti degli allenati identici al bit all'indietro intero; TEMPO
      dell'indietro e del passo intero (mediane di 5, alternati) e rapporto.

Uso (4 fili, nice, niente GPU):
    systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=6G nice -n 19 env CUDA_VISIBLE_DEVICES= \\
        HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 E2E_ALLENA_PROVE=@@ROOT:depot@@/prove_dominio_strati_b/x \\
        @@ROOT:venv@@/bin/python3 e2e_test_dominio_strati.py [D1 D2 D3 D4 D5]
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("E2E_ALLENA_PROVE", "@@ROOT:depot@@/prove_dominio_strati_b/prove")
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_lora as LR  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_studente as S  # noqa: E402
import e2e_test_allena as TA  # noqa: E402
import e2e_test_f32 as TF  # noqa: E402
import e2e_test_integrata as TI  # noqa: E402
import e2e_test_lora as TL  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "dominio_strati"
BASE = "39d514a"                       # ramo integrata al momento della nascita di dominio-strati-b
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
FAM = TI.FAM


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:3000]}", flush=True)
    return bool(ok)


def _pulisci(*dd) -> None:
    for d in dd:
        shutil.rmtree(d, ignore_errors=True)


def _uguali(a, b) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _foto(st) -> dict:
    """Tutti i valori allenabili e i byte, per strato (copie)."""
    return {"theta": {k: p.detach().clone() for k, p in zip(st.chiavi(), st.parametri())},
            "f32": {n: t.master.detach().clone() for n, t in st.f32.items()},
            "lora": {n: {k: v.clone() for k, v in a.valori().items()} for n, a in st.lora.items()},
            "byte": {k: st.esperti(k[0]).fam[k[1]].blocchi.clone() for k in st.chiavi()}}


def _strato_di_nome(n: str) -> int:
    return int(n.split(".")[1]) if n.startswith("blk.") else -1


def _uguali_strato(f1: dict, f2: dict, L: int) -> dict:
    """Confronto al bit dei valori dello strato L fra due foto."""
    return {"theta": all(torch.equal(f1["theta"][k], f2["theta"][k]) for k in f1["theta"] if k[0] == L),
            "f32": all(torch.equal(f1["f32"][n], f2["f32"][n]) for n in f1["f32"] if _strato_di_nome(n) == L),
            "lora": all(set(f1["lora"][n]) == set(f2["lora"][n]) and all(torch.equal(f1["lora"][n][k],
                                                                                    f2["lora"][n][k].to(
                                                                                        f1["lora"][n][k].dtype))
                                                                       for k in f1["lora"][n])
                        for n in f1["lora"] if _strato_di_nome(n) == L),
            "byte": all(torch.equal(f1["byte"][k], f2["byte"][k]) for k in f1["byte"] if k[0] == L)}


def _mosso_strato(f1: dict, f2: dict, L: int) -> dict:
    return {"theta": any(not torch.equal(f1["theta"][k], f2["theta"][k]) for k in f1["theta"] if k[0] == L),
            "f32": any(not torch.equal(f1["f32"][n], f2["f32"][n]) for n in f1["f32"] if _strato_di_nome(n) == L)}


def _indici(st, L: int) -> dict:
    """Indici nello stato dell'ottimizzatore dei parametri dello strato L: theta (gruppo 0), F32 e LoRA (AdamW, ordine
    di `nuovo_ottimizzatore`; con Sophia gli indici dell'AdamW ripartono da 0)."""
    th = [i for i, (Lk, _) in enumerate(st.chiavi()) if Lk == L]
    f = [j for j, n in enumerate(st.chiavi_f32()) if _strato_di_nome(n) == L]
    lo = [2 * j + k for j, n in enumerate(st.chiavi_lora()) if _strato_di_nome(n) == L for k in (0, 1)]
    return {"theta": th, "f32": f, "lora": lo}


# ─────────────────────────────────────────── D1 ───────────────────────────────────────────
_ESEGUI = r'''
import json, os, sys, torch
from pathlib import Path
alb, uscita, corsa, campi = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
sys.path.insert(0, alb)
os.chdir(alb)
import e2e_allena as A, e2e_test_integrata as TI, e2e_test_lora as TL
src, WS = TL.sorgente_lora(n_train=12, n_held=4)
src = TI.con_domini(src)
ca = TI._ca_integrata(**{k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
                         for k, v in campi.items()})
st = TI._studente_int()
z = A.allena(st, src, Path(corsa), ca, WS, WS, log=lambda *a, **k: None)
r = A.carica_ripresa(Path(corsa), "cpu")
s = TI._stato_confrontabile(z)
for k in ("studente",):
    s.pop(k, None)
for k in ("peso_dominio", "strati_allenati"):
    s["config"].pop(k, None)
voci_pd = [x.pop("peso_dominio", None) for x in s["storia"]]      # solo registro (con --peso-dominio): a parte
torch.save({"voci_peso_dominio": voci_pd,"theta": [p.detach().clone() for p in st.parametri()], "f32": [p.detach().clone() for p in st.parametri_f32()],
            "lora": st.valori_lora(), "byte": [st.esperti(L).fam[n].blocchi.clone() for (L, n) in st.chiavi()],
            "opt": r["opt"], "ema": r.get("ema"), "stato": json.dumps(s, sort_keys=True, default=str)}, uscita)
'''


def _corsa_in_processo(albero: Path, nome: str, campi: dict) -> dict:
    """La corsa integrata di 3 passi in un processo a parte con i moduli dell'`albero` (thor_v2)."""
    d = DIR / f"d1_{nome}"
    _pulisci(d)
    d.mkdir(parents=True)
    script = d / "esegui.py"
    script.write_text(_ESEGUI)
    env = dict(os.environ, E2E_ALLENA_PROVE=str(d / "prove"), TRITON_INTERPRET="1")
    r = subprocess.run([sys.executable, str(script), str(albero), str(d / "esito.pt"), str(d / "corsa"),
                        json.dumps(campi)], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"corsa {nome} fallita: {r.stderr[-3000:]}")
    return torch.load(d / "esito.pt", weights_only=False)


def _uguali_esiti(a: dict, b: dict) -> dict:
    return {"theta": _uguali(a["theta"], b["theta"]), "f32": _uguali(a["f32"], b["f32"]),
            "lora": a["lora"].keys() == b["lora"].keys() and all(
                a["lora"][n].keys() == b["lora"][n].keys() and all(torch.equal(a["lora"][n][k], b["lora"][n][k])
                                                                   for k in a["lora"][n]) for n in a["lora"]),
            "byte": _uguali(a["byte"], b["byte"]), "opt": TI._uguali_stato_opt(a["opt"], b["opt"]),
            "ema": TI._uguali_stato_opt(a["ema"], b["ema"]) and a["ema"] is not None,
            "stato": a["stato"] == b["stato"]}


def _stato_uguale_a_meno_dei_nuovi(js_prima: str, js_ora: str) -> tuple:
    """(uguali, campi nuovi della config): stato di 39d514a contro quello di oggi, senza i campi della config che 39d514a
    non conosceva (devono valere il loro default: allora sono spenti) e senza min_w/mediana_w (solo registro; 39d514a li
    scriveva sempre, anche nei pesi del G1, oggi solo con --attribuzione e non nel G1: 8af000d)."""
    p, o = json.loads(js_prima), json.loads(js_ora)
    nuovi = sorted(set(o["config"]) - set(p["config"]))
    ai_default = all(o["config"][k] == A._CONFIG_DEFAULT.get(k) for k in nuovi) and not set(p["config"]) - set(o["config"])
    for z in (p, o):
        for k in nuovi:
            z["config"].pop(k, None)
        for pz in [x.get("pesi") for x in z["storia"]] + [(z.get("g1") or {}).get("pesi_passo0")]:
            for k in ("min_w", "mediana_w"):
                (pz or {}).pop(k, None)
    return ai_default and p == o, nuovi


def prova_d1() -> bool:
    t0 = time.time()
    base = DIR / "d1_albero_base"
    _pulisci(base)
    base.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(QUI.parent), "archive", BASE, "thor_v2"], check=True,
                         capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(base)], input=arc, check=True)
    npz = QUI / "mappa_completa_gguf_hf_perm.npz"
    if npz.exists():
        shutil.copy2(npz, base / "thor_v2" / npz.name)
    prima = _corsa_in_processo(base / "thor_v2", "prima", {})
    ora = _corsa_in_processo(QUI, "ora", {})
    a = _uguali_esiti(prima, ora)
    # integrata (merge dopo trainer-mc, lota-qaf e min/mediana dei pesi): lo stato si confronta tolti i campi nati dopo
    # 39d514a che sono SOLO registro (min_w/mediana_w dei pesi con --attribuzione) o config ai loro default
    a["stato"], a["stato_campi_nuovi"] = _stato_uguale_a_meno_dei_nuovi(prima["stato"], ora["stato"])
    # (b) pesi tutti uguali (i tre domini di con_domini a 2) e (c) strati allenati = tutti, nello stesso albero
    uguali = _corsa_in_processo(QUI, "pesi_uguali", {"peso_dominio": [["edu", 2.0], ["it_chat", 2.0],
                                                                        ["strumenti", 2.0]]})
    tutti = _corsa_in_processo(QUI, "strati_tutti", {"strati_allenati": [0, 1]})
    b = _uguali_esiti(ora, uguali)
    # la voce "peso_dominio" del registro c'e' solo con l'opzione accesa: pesi effettivi tutti 1
    b["voce_pesi_1"] = (all(v is None for v in ora["voci_peso_dominio"])
                        and all(v is not None and all(x["peso"] == 1.0 for x in v["per_dominio"].values())
                                for v in uguali["voci_peso_dominio"]))
    c = _uguali_esiti(ora, tutti)
    ok = all(v for k, v in a.items() if k != "stato_campi_nuovi") and all(b.values()) and all(c.values())
    return registra("D1_default_al_bit", ok, base=BASE, a_contro_39d514a=a, b_pesi_uguali=b, c_strati_tutti=c,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── D2 ───────────────────────────────────────────
class _SorgenteDomini:
    """Solo `dominio` (normalizzazione)."""

    def __init__(self, domini: dict):
        self.d = domini

    def dominio(self, lato, i):
        return self.d[i]


def _sorgente_d2(domini=("a", "b", "a", "c")):
    """sorgente_lora con le righe di train assegnate a rotazione ai domini dati."""
    src, WS = TL.sorgente_lora(n_train=8, n_held=4)
    return TI.con_domini(src, domini), WS


def _grad(st, src, WS, righe: list, ca, K: float) -> dict:
    """Gradienti (theta, F32, LoRA) di `gradiente_passo` su `righe`."""
    A._azzera(st)
    car = A.Caricatore(src, torch.device("cpu"))
    try:
        A.gradiente_passo(st, car, "train", righe, ca, WS, WS, K)
    finally:
        car.chiudi()
    return {"theta": [None if p.grad is None else p.grad.clone() for p in st.parametri()],
            "f32": [None if p.grad is None else p.grad.clone() for p in st.parametri_f32()],
            "lora": [None if p.grad is None else p.grad.clone() for p in st.parametri_lora()]}


def _err_combinazione(g: dict, singoli: list, pesi: list, n: int) -> float:
    """max su tutti i tensori di |g - (1/n) sum_r w_r g_r| / max |riferimento|."""
    peggiore = 0.0
    for gruppo in ("theta", "f32", "lora"):
        for j, x in enumerate(g[gruppo]):
            rif = sum(w * s[gruppo][j].double() for w, s in zip(pesi, singoli)) / n
            m = float(rif.abs().max())
            if m == 0.0:
                continue
            peggiore = max(peggiore, float((x.double() - rif).abs().max()) / m)
    return peggiore


def _studente_grad():
    st = TI._studente_int()
    TF._theta_casuale(st, seme=4, s=0.05)
    TL._b_casuale(st, 0.05, seme=9)
    return st


def prova_d2() -> bool:
    t0 = time.time()
    es: dict = {}
    # (a) testo e rifiuti
    rif = (("edu", 1.5), ("it_chat", 0.5), ("strumenti", 0.5))
    es["testo"] = (A.peso_dominio_da_testo(" strumenti=0.5, it_chat=0.5,edu=1.5 ") == rif
                   and A.peso_dominio_da_testo("") == () and A.strati_da_testo("20-39") == tuple(range(20, 40))
                   and A.strati_da_testo("0-1, 30-31,5") == (0, 1, 5, 30, 31) and A.strati_da_testo("") == ()
                   and A._testo_strati(tuple(range(20, 40))) == "20-39" and A._testo_strati((0, 1, 5)) == "0-1,5")
    rifiuti = {}
    for nome, f in (("peso_zero", lambda: A.ConfigAllena(peso_dominio=(("a", 0.0),))),
                    ("peso_negativo", lambda: A.ConfigAllena(peso_dominio=A.peso_dominio_da_testo("a=-1"))),
                    ("peso_nan", lambda: A.ConfigAllena(peso_dominio=A.peso_dominio_da_testo("a=nan"))),
                    ("peso_inf", lambda: A.ConfigAllena(peso_dominio=A.peso_dominio_da_testo("a=inf"))),
                    ("doppio", lambda: A.peso_dominio_da_testo("a=1,a=2")),
                    ("sintassi", lambda: A.peso_dominio_da_testo("a")),
                    ("senza_nome", lambda: A.peso_dominio_da_testo("=1")),
                    ("disordinato", lambda: A.ConfigAllena(peso_dominio=(("b", 1.0), ("a", 1.0)))),
                    ("strati_vuoto", lambda: A.strati_da_testo("5-3")),
                    ("strati_sintassi", lambda: A.strati_da_testo("x")),
                    ("strati_disordinati", lambda: A.ConfigAllena(strati_allenati=(3, 1)))):
        try:
            f()
            rifiuti[nome] = False
        except ValueError:
            rifiuti[nome] = True
    es["rifiuti"] = all(rifiuti.values())
    # (b) normalizzazione
    doms = {0: "it_chat", 1: "edu", 2: "strumenti", 3: "it_chat", 4: "cot", 5: "altro", A.OFFSET_MC_PESO + 7: "x"}
    sd = _SorgenteDomini(doms)
    ca_n = A.ConfigAllena(peso_dominio=(("cot", 1.25), ("edu", 1.5), ("it_chat", 0.5), ("mc", 3.0), ("strumenti", 0.75)))
    righe = list(doms)
    w = A.pesi_dominio_passo(sd, "train", righe, ca_n)
    grezzi = {"it_chat": 0.5, "edu": 1.5, "strumenti": 0.75, "cot": 1.25, "altro": 1.0, "mc": 3.0}
    dom_p = [A.dominio_per_peso(sd, "train", i) for i in righe]
    es["normalizzazione"] = (abs(math.fsum(w.values()) / len(w) - 1.0) < 1e-12
                             and dom_p[-1] == "mc" and dom_p[:6] == [doms[i] for i in righe[:6]]
                             and all(abs(w[i] / w[righe[0]] - grezzi[d] / grezzi[dom_p[0]]) < 1e-12
                                     for i, d in zip(righe, dom_p)))
    es["uguali_none"] = (A.pesi_dominio_passo(sd, "train", [1, 1], ca_n) is None
                         and A.pesi_dominio_passo(sd, "train", righe, A.ConfigAllena()) is None)
    rias = A.riassunto_peso_dominio(sd, "train", righe, ca_n)
    es["riassunto"] = (abs(sum(rias["quota_perdita"].values()) - 1.0) < 1e-12
                       and rias["per_dominio"]["it_chat"]["righe"] == 2 and abs(rias["c"] * math.fsum(
                           grezzi[d] for d in dom_p) - len(righe)) < 1e-12)
    # (c) gradiente del passo pesato == combinazione dei gradienti delle righe (theta, F32, LoRA)
    src, WS = _sorgente_d2()
    K = 2.0 ** 12
    pd = (("a", 0.5), ("b", 2.0), ("c", 1.25))
    grezzi_c = [dict(pd)[src.dominio("train", i)] for i in range(4)]
    wc = [x * 4 / sum(grezzi_c) for x in grezzi_c]
    st = _studente_grad()
    base = dict(allena_f32=FAM, righe_per_micro=2)
    g_w = _grad(st, src, WS, [0, 1, 2, 3], TL._ca_lora(peso_dominio=pd, **base), K)
    singoli = [_grad(st, src, WS, [r], TL._ca_lora(**base), K) for r in range(4)]
    err_c = _err_combinazione(g_w, singoli, wc, 4)
    g_u = _grad(st, src, WS, [0, 1, 2, 3], TL._ca_lora(**base), K)
    err_u = _err_combinazione(g_u, singoli, [1.0] * 4, 4)
    es["gradiente_pesato"] = err_c < 1e-5 and err_u < 1e-5
    # (d) moltiplicativo coi pesi per token kl (una riga per micro-lotto: i pesi per token si normalizzano nella riga)
    base_d = dict(allena_f32=FAM, righe_per_micro=1, peso_token="kl")
    g_wd = _grad(st, src, WS, [0, 1, 2, 3], TL._ca_lora(peso_dominio=pd, **base_d), K)
    singoli_d = [_grad(st, src, WS, [r], TL._ca_lora(**base_d), K) for r in range(4)]
    err_d = _err_combinazione(g_wd, singoli_d, wc, 4)
    es["moltiplicativo_peso_token"] = err_d < 1e-5
    # (e) curvatura: dL/dxL del vettore campionato, riga pesata == sqrt(w) x senza pesi (stesso seme)
    visto = []

    class _PVFinto:
        def curvatura(self):
            return 3

    o_ind = st.indietro
    st.indietro = lambda stato, g: visto.append(g.clone())
    try:
        car = A.Caricatore(src, torch.device("cpu"))
        ca_c = TL._ca_lora(allena_f32=FAM, righe_per_micro=1)
        A._curvatura_passo(st, car, "train", [1], ca_c, WS, 1.0, _PVFinto(), 77)
        A._curvatura_passo(st, car, "train", [1], ca_c, WS, 1.0, _PVFinto(), 77, pesi_d={1: 2.25})
        car.chiudi()
    finally:
        del st.indietro
    es["curvatura_sqrt_w"] = (len(visto) == 2 and float((visto[1].double() - 1.5 * visto[0].double()).abs().max())
                              <= 1e-6 * float(visto[0].abs().max()))
    es["studente_torna_quello_di_prima"] = st.indietro.__func__ is o_ind.__func__
    # (f) ciclo: voce nel registro, allenamento diverso, dominio ignoto, ripresa da spento
    src3, WS3 = TL.sorgente_lora(n_train=12, n_held=4)
    src3 = TI.con_domini(src3)
    pdz = (("edu", 1.5), ("it_chat", 0.5), ("strumenti", 0.75))
    d_on, d_off, d_ign, d_rip = (DIR / x for x in ("d2_on", "d2_off", "d2_ignoto", "d2_ripresa"))
    _pulisci(d_on, d_off, d_ign, d_rip)
    reg: list = []
    st_on = TI._studente_int()
    z_on = A.allena(st_on, src3, d_on, TI._ca_integrata(peso_dominio=pdz), WS3, WS3, ferma_a=2, log=reg.append)
    st_off = TI._studente_int()
    A.allena(st_off, src3, d_off, TI._ca_integrata(), WS3, WS3, ferma_a=2, log=muto)
    voci = [x.get("peso_dominio") for x in z_on["storia"]]
    es["voce_registro"] = all(v is not None and abs(sum(v["quota_perdita"].values()) - 1.0) < 1e-12
                              and set(v["per_dominio"]) <= {"edu", "it_chat", "strumenti"} for v in voci) and len(voci) == 2
    es["log"] = any("pesi per dominio (--peso-dominio)" in m for m in reg)
    es["allenamento_diverso"] = not _uguali(st_on.parametri(), st_off.parametri())
    try:
        A.allena(TI._studente_int(), src3, d_ign, TI._ca_integrata(peso_dominio=(("strumento", 0.5),)), WS3, WS3,
                 ferma_a=1, log=muto)
        es["dominio_ignoto_rifiutato"] = False
    except ValueError as e:
        es["dominio_ignoto_rifiutato"] = "strumento" in str(e)
    A.allena(TI._studente_int(), src3, d_rip, TI._ca_integrata(), WS3, WS3, ferma_a=1, log=muto)
    reg_r: list = []
    z_r = A.allena(TI._studente_int(), src3, d_rip, TI._ca_integrata(peso_dominio=pdz), WS3, WS3, riprendi=True,
                   ferma_a=2, log=reg_r.append)
    es["ripresa_da_spento"] = (z_r["passo"] == 2 and "peso_dominio" not in z_r["storia"][0]
                               and "peso_dominio" in z_r["storia"][1]
                               and any("ripresa con campi cambiati" in m and "peso_dominio" in m for m in reg_r))
    ok = all(es.values())
    return registra("D2_pesi_per_dominio", ok, **es, rifiuti_dettaglio=rifiuti, err_combinazione=err_c,
                    err_uniforme=err_u, err_moltiplicativo=err_d, pesi_normalizzati=wc, voce=voci[-1],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── D3 ───────────────────────────────────────────
def prova_d3() -> bool:
    t0 = time.time()
    es: dict = {}
    # (a) gradienti dello strato 1 == indietro intero, al bit
    src, WS = _sorgente_d2()
    K = 2.0 ** 12
    ca = TL._ca_lora(allena_f32=FAM, righe_per_micro=2)
    st_i, st_c = _studente_grad(), _studente_grad()
    g_i = _grad(st_i, src, WS, [0, 1, 2, 3], ca, K)
    st_c.congela_strati([1])
    g_c = _grad(st_c, src, WS, [0, 1, 2, 3], ca, K)
    L_th = [L for (L, _) in st_c.chiavi()]
    L_f = [st_c.f32[n].L for n in st_c.chiavi_f32()]
    L_l = [st_c.lora[n].L for n in st_c.chiavi_lora() for _ in (0, 1)]
    ug = {"theta_1": all(torch.equal(a, b) for a, b, L in zip(g_i["theta"], g_c["theta"], L_th) if L == 1),
          "theta_0_none": all(b is None for b, L in zip(g_c["theta"], L_th) if L == 0),
          "f32_1_e_output_norm": all(torch.equal(a, b) for a, b, L in zip(g_i["f32"], g_c["f32"], L_f) if L != 0)
          and any(L == -1 for L in L_f),
          "f32_0_none": all(b is None for b, L in zip(g_c["f32"], L_f) if L == 0),
          "lora_1": all(torch.equal(a, b) for a, b, L in zip(g_i["lora"], g_c["lora"], L_l) if L == 1),
          "lora_0_none": all(b is None for b, L in zip(g_c["lora"], L_l) if L == 0),
          "requires_grad": all(not p.requires_grad for (L, _), p in zip(st_c.chiavi(), st_c.parametri()) if L == 0)}
    st_c.congela_strati(None)
    ug["scongelato_tutto"] = all(p.requires_grad for p in st_c.parametri() + st_c.parametri_f32() + st_c.parametri_lora())
    es["a_gradienti"] = all(ug.values())
    # (b) ciclo integrato con lo strato 0 congelato
    src3, WS3 = TL.sorgente_lora(n_train=12, n_held=4)
    src3 = TI.con_domini(src3)
    ca_b = TI._ca_integrata(strati_allenati=(1,), tieni_riprese=10)
    d_f, d_r = DIR / "d3_filata", DIR / "d3_ripresa"
    _pulisci(d_f, d_r)
    st_f = TI._studente_int()
    f0 = _foto(st_f)
    reg: list = []
    o_esito = PV.RegioneFiducia.esito                 # salti TENUTI per costruzione: si vede DOVE atterrano

    def tiene_sempre(self, rho, prima, dopo, n, **kw):
        o_esito(self, rho, prima, dopo, n, **kw)
        return True
    PV.RegioneFiducia.esito = tiene_sempre
    try:
        z_f = A.allena(st_f, src3, d_f, ca_b, WS3, WS3, log=reg.append)
        A.allena(TI._studente_int(), src3, d_r, ca_b, WS3, WS3, ferma_a=1, log=muto)
        st_r = TI._studente_int()
        z_r = A.allena(st_r, src3, d_r, ca_b, WS3, WS3, riprendi=True, log=muto)
    finally:
        PV.RegioneFiducia.esito = o_esito
    f1 = _foto(st_f)
    r_f = A.carica_ripresa(d_f, "cpu")
    idx0 = _indici(st_f, 0)
    so = r_f["opt"]["sophia"]["state"]
    ad = r_f["opt"]["adam"]["state"]
    bop = torch.load(d_f / z_f["pv"]["bop"]["file"], weights_only=True)
    registri = PV.Registro.carica_lista(d_f / z_f["pv"]["file"], z_f["pv"]["sha"])
    strati_salti = sorted({int(L) for rg in registri for L in np.asarray(rg.righe["L"]).tolist()})
    freschi = {n: PV.StatoBop.crea(st_f.esperti(0).fam[n]) for n in S.FAMIGLIE}
    n_f_tot = len(st_f.parametri_f32())
    b = {"passi": z_f["passo"] == 3, "strato0_al_bit": _uguali_strato(f0, f1, 0),
         "strato1_mosso": _mosso_strato(f0, f1, 1),
         "sophia_strato0_assente": all(i not in so for i in idx0["theta"]) and all(
             i in so for i, (L, _) in enumerate(st_f.chiavi()) if L == 1),
         "adamw_f32_lora_strato0_assente": all(j not in ad for j in idx0["f32"]) and all(
             n_f_tot + j not in ad for j in idx0["lora"]) and len(ad) > 0,
         "vbop_strato0_iniziale": all(torch.equal(bop["stato"][f"0.{n}"][c], getattr(freschi[n], c).cpu())
                                      for n in S.FAMIGLIE for c in PV._CAMPI_BOP)
         and any(int((bop["stato"][f"1.{n}"]["eta"] != 0).sum()) > 0 for n in S.FAMIGLIE),
         "ema_strato0": all(torch.equal(r_f["ema"][A._chiave(k)], f0["theta"][k]) for k in st_f.chiavi() if k[0] == 0),
         "salti_solo_strato1": strati_salti == [1],
         "lora_ga_solo_strato1": all((a.A0 is not None) == (a.L == 1) for a in st_f.lora.values()),
         "attribuzione": all("attribuzione" in x for x in z_f["storia"]),
         "statistiche_theta_solo_allenati": all(
             x["attribuzione"]["diagnostica"]["theta"]["vivi"] == sum(
                 int((st_f.esperti(1).fam[n].d0 != 0).sum()) for n in S.FAMIGLIE) for x in z_f["storia"]),
         "log": any("strati allenati 1" in m and "indietro fermo allo strato 1" in m for m in reg)
         and any("famiglie degli strati allenati" in m for m in reg),
         "studente_restituito": all(p.requires_grad for p in st_f.parametri() + st_f.parametri_f32()
                                    + st_f.parametri_lora())}
    b["strato0_al_bit"] = all(b["strato0_al_bit"].values())
    b["strato1_mosso"] = all(b["strato1_mosso"].values())
    # ripresa 1 + 2 == 3 filati, al bit (corse fatte sopra, coi salti tenuti)
    r_r = A.carica_ripresa(d_r, "cpu")
    b["ripresa_al_bit"] = (_uguali(st_f.parametri(), st_r.parametri()) and _uguali(st_f.parametri_f32(),
                                                                                    st_r.parametri_f32())
                           and TI._uguali_lora_ga(st_f, st_r) and _uguali(TI._byte(st_f), TI._byte(st_r))
                           and TI._uguali_stato_opt(r_f["opt"], r_r["opt"])
                           and TI._uguali_stato_opt(r_f["ema"], r_r["ema"])
                           and TI._stato_confrontabile(z_f) == TI._stato_confrontabile(z_r)
                           and z_f["pv"].get("bop", {}).get("sha") == z_r["pv"].get("bop", {}).get("sha"))
    es["b_ciclo"] = all(b.values())
    # (c) AdamW: 2 passi tutto, 1 congelato, 1 scongelato (24 righe: 6 passi da 4, 2 micro-lotti a passo)
    d_c = DIR / "d3_scongela"
    _pulisci(d_c)
    src4, WS4 = TL.sorgente_lora(n_train=24, n_held=4)
    ca_c = TL._ca_lora(allena_f32=FAM, righe_per_passo=4, righe_per_micro=2, tieni_riprese=10, heldout_ogni=1)
    A.allena(TI._studente_int(), src4, d_c, ca_c, WS4, WS4, ferma_a=2, log=muto)
    r2 = A.carica_ripresa(d_c, "cpu")
    reg_c: list = []
    st3 = TI._studente_int()
    A.allena(st3, src4, d_c, TL._ca_lora(allena_f32=FAM, righe_per_passo=4, righe_per_micro=2, tieni_riprese=10,
                                         heldout_ogni=1, strati_allenati=(1,)), WS4, WS4, riprendi=True, ferma_a=3,
             log=reg_c.append)
    r3 = A.carica_ripresa(d_c, "cpu")
    i0 = _indici(st3, 0)
    n_th, n_f = len(st3.parametri()), len(st3.parametri_f32())
    tutti0 = i0["theta"] + [n_th + j for j in i0["f32"]] + [n_th + n_f + j for j in i0["lora"]]
    c = {"adam_strato0_al_bit": all(TI._uguali_stato_opt(r2["opt"]["state"][i], r3["opt"]["state"][i]) for i in tutti0),
         "theta0_al_bit": all(torch.equal(r2["theta"][A._chiave(k)], r3["theta"][A._chiave(k)])
                              for k in st3.chiavi() if k[0] == 0),
         "campi_cambiati": any("ripresa con campi cambiati" in m and "strati_allenati" in m for m in reg_c)}
    reg_s: list = []
    st4 = TI._studente_int()
    z4 = A.allena(st4, src4, d_c, ca_c, WS4, WS4, riprendi=True, ferma_a=4, log=reg_s.append)
    r4 = A.carica_ripresa(d_c, "cpu")
    pre = (z4["storia"][-1].get("rumore") or {}).get("precondizionata") or {}
    c.update({"scongelato": z4["passo"] == 4,
              "theta0_mosso": any(not torch.equal(r3["theta"][A._chiave(k)], r4["theta"][A._chiave(k)])
                                  for k in st4.chiavi() if k[0] == 0),
              "passi_adam_diversi": {float(r4["opt"]["state"][i]["step"]) for i in range(n_th)} == {3.0, 4.0},
              "rumore_precondizionato_finito": pre.get("stima") is not None
              and all(math.isfinite(v) for v in pre["g2_micro"])})
    es["c_scongela"] = all(c.values())
    ok = all(es.values())
    return registra("D3_strati_congelati", ok, a=ug, b=b, c=c, salti_per_strato=strati_salti,
                    heldout=[(x["passo"], round(x["metrica"], 6)) for x in z_f["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── D4 ───────────────────────────────────────────
def prova_d4() -> bool:
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    d = DIR / "d4_eb2"
    _pulisci(d)
    st1 = TI._studente_int()
    o_esito = PV.RegioneFiducia.esito                 # EB2: salti TENUTI per costruzione (registri da rigiocare)

    def tiene_sempre(self, rho, prima, dopo, n, **kw):
        o_esito(self, rho, prima, dopo, n, **kw)
        return True
    PV.RegioneFiducia.esito = tiene_sempre
    try:
        z1 = A.allena(st1, src, d, TI._ca_eb2(), WS, WS, ferma_a=2, log=muto)
    finally:
        PV.RegioneFiducia.esito = o_esito
    r1 = A.carica_ripresa(d, "cpu")
    f_rip = _foto(st1)
    bop1 = torch.load(d / z1["pv"]["bop"]["file"], weights_only=True)
    visto: dict = {}
    o_ga = LR.inizializza_ga

    def spia_ga(bersaglio, *a, **k):                   # stato appena prima di LoRA-GA
        visto["lora_prima_ga"] = {n: {kk: v.clone() for kk, v in ad.valori().items()} for n, ad in st2.lora.items()}
        visto["ga_su"] = sorted(bersaglio.lora)
        return o_ga(bersaglio, *a, **k)
    LR.inizializza_ga = spia_ga
    reg: list = []
    try:
        ca2 = TI._ca_integrata(ema_theta=0.9, heldout_pieno_ogni=16, tieni_riprese=1, strati_allenati=(1,))
        st2 = TI._studente_int()
        z2 = A.allena(st2, src, d, ca2, WS, WS, riprendi=True, ferma_a=3, log=reg.append, reinit_lora_ga=True)
    finally:
        LR.inizializza_ga = o_ga
    r2 = A.carica_ripresa(d, "cpu")
    f2 = _foto(st2)
    n_th, n_f = len(st2.parametri()), len(st2.parametri_f32())
    i0, i1 = _indici(st2, 0), _indici(st2, 1)
    s1 = r1["opt"]["state"]
    so, ad = r2["opt"]["sophia"]["state"], r2["opt"]["adam"]["state"]
    bop2 = torch.load(d / z2["pv"]["bop"]["file"], weights_only=True)
    lora0 = [n for n in st2.chiavi_lora() if _strato_di_nome(n) == 0]
    lora1 = [n for n in st2.chiavi_lora() if _strato_di_nome(n) == 1]
    c = {"lora_eb2_allenata": any(float(v["B"].abs().max()) > 0 for v in r1["lora"].values())
         and not any("A0" in v for v in r1["lora"].values()),
         "accettata": z2["passo"] == 3,
         "strato0_al_bit": _uguali_strato(f_rip, f2, 0),
         "theta1_mosso": _mosso_strato(f_rip, f2, 1)["theta"],
         "sophia_m0_da_adam": all(torch.equal(so[i]["exp_avg"], s1[i]["exp_avg"]) and so[i]["n_hess"] == 0
                                  and float(so[i]["step"]) == float(s1[i]["step"]) for i in i0["theta"]),
         "sophia_strato1_avanzato": all(float(so[i]["step"]) == float(s1[i]["step"]) + 1 for i in i1["theta"]),
         "adamw_f32_0_al_bit": all(TI._uguali_stato_opt(ad[j], s1[n_th + j]) for j in i0["f32"]),
         "adamw_lora_0_al_bit": all(TI._uguali_stato_opt(ad[n_f + j], s1[n_th + n_f + j]) for j in i0["lora"]),
         "adamw_lora_1_da_zero": all(float(ad[n_f + j]["step"]) == 1.0 for j in i1["lora"]),
         "vbop_strato0_al_bit": all(torch.equal(bop1["stato"][f"0.{n}"][k], bop2["stato"][f"0.{n}"][k])
                                    for n in S.FAMIGLIE for k in PV._CAMPI_BOP),
         "ema_strato0_al_bit": all(torch.equal(r1["ema"][A._chiave(k)], r2["ema"][A._chiave(k)])
                                   for k in st2.chiavi() if k[0] == 0),
         "ema_strato1_lerp": all(torch.equal(r2["ema"][A._chiave(k)], r1["ema"][A._chiave(k)].clone().lerp_(
             r2["theta"][A._chiave(k)], 1.0 - 0.9)) for k in st2.chiavi() if k[0] == 1),
         "ga_solo_strato1": visto.get("ga_su") == sorted(lora1),
         "lora0_ripresa_prima_di_ga": all(torch.equal(visto["lora_prima_ga"][n]["B"], r1["lora"][n]["B"].to(
             visto["lora_prima_ga"][n]["B"].dtype)) for n in lora0) if visto.get("lora_prima_ga") else False,
         "lora1_scartata_prima_di_ga": all(float(visto["lora_prima_ga"][n]["B"].abs().max()) == 0.0 for n in lora1)
         if visto.get("lora_prima_ga") else False,
         "lora1_ga": all("A0" in f2["lora"][n] for n in lora1) and all("A0" not in f2["lora"][n] for n in lora0),
         "registro_reinit": z2.get("lora_reinit_ga", {}).get("conservati_strati_congelati") == len(lora0)
         and z2["lora_reinit_ga"]["adattatori"] == len(lora1),
         "log": any("CONSERVATA al bit" in m for m in reg)}
    c["strato0_al_bit"] = all(c["strato0_al_bit"].values())
    ok = all(c.values())
    return registra("D4_ripresa_tipo_EB2_strati_congelati", ok, **c, reinit=z2.get("lora_reinit_ga"),
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── D5 ───────────────────────────────────────────
TIPI_8 = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2


def config_otto():
    cfg = TF.config_minuscola()
    cfg.num_hidden_layers = 8
    cfg.layer_types = list(TIPI_8)
    return cfg


def crea_gguf_otto(p: Path, seme: int = 11) -> Path:
    """Come `e2e_test_f32.crea_gguf` (F32 ovunque tranne gli esperti TQ1_0), 8 strati."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    CSG, GQ, GT = TF.CSG, TF.GQ, TF.GT
    cfg = config_otto()
    g = np.random.default_rng(seme)
    w = TF.GGUFWriter(str(p), "qwen35moe")
    H = cfg.hidden_size
    w.add_tensor("token_embd.weight", (0.5 * g.standard_normal((cfg.vocab_size, H))).astype(np.float32))
    for L in range(8):
        tipo = "full" if cfg.layer_types[L] == "full_attention" else "linear"
        forme = {k: tuple(v.shape) for k, v in Qwen3_5MoeDecoderLayer(cfg, layer_idx=L).state_dict().items()}
        for fam, voce in CSG.MAPPA[tipo].items():
            if voce.get("per_esperto"):
                continue
            forma = (cfg.num_experts, H) if voce["hf"] == "mlp.gate.weight" else forme[voce["hf"]]
            n = int(np.prod(forma[1:])) if len(forma) > 1 else forma[0]
            if voce["hf"].endswith("A_log"):
                v = g.uniform(-1.0, 0.5, forma)
            elif voce.get("f") == "piu_uno" or voce["hf"].endswith("norm.weight") and "layernorm" in voce["hf"]:
                v = 0.1 * g.standard_normal(forma)
            elif voce["hf"].endswith("linear_attn.norm.weight"):
                v = 1.0 + 0.1 * g.standard_normal(forma)
            elif voce["hf"].endswith("dt_bias"):
                v = g.uniform(-0.5, 0.5, forma)
            else:
                v = g.standard_normal(forma) / np.sqrt(max(n, 1))
            if voce["hf"] == "mlp.gate.weight":
                v = v * 3.0
            w.add_tensor(f"blk.{L}.{fam}", np.ascontiguousarray(TF._hf_in_gguf(voce, v), dtype=np.float32))
        I = cfg.moe_intermediate_size
        for nome, (R, C) in (("ffn_gate_exps", (I, H)), ("ffn_up_exps", (I, H)), ("ffn_down_exps", (H, I))):
            x = (g.standard_normal((cfg.num_experts * R, C)) / np.sqrt(C)).astype(np.float32)
            q = GQ.quantize(x, GT.TQ1_0).reshape(cfg.num_experts, R, -1)
            w.add_tensor(f"blk.{L}.{nome}.weight", np.ascontiguousarray(q), raw_dtype=GT.TQ1_0)
    w.add_tensor("output_norm.weight", (1.0 + 0.1 * g.standard_normal(H)).astype(np.float32))
    w.add_tensor("output.weight", (g.standard_normal((cfg.vocab_size, H)) / 16).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return p


def studente_otto(gguf: Path):
    vecchio = S.config_testo
    S.config_testo = config_otto
    try:
        return S.Studente(gguf=gguf, strati=range(8), ricetta="A32", motore_nome="torch", device="cpu", log=muto,
                          allena_f32=FAM)
    finally:
        S.config_testo = vecchio


def _sync_tempo() -> float:
    return time.perf_counter()


def prova_d5() -> bool:
    t0 = time.time()
    DIR.mkdir(parents=True, exist_ok=True)
    gguf = crea_gguf_otto(DIR / "otto_strati.gguf")
    st = studente_otto(gguf)
    TF._theta_casuale(st, seme=5, s=0.05)
    B, T = 2, int(os.getenv("E2E_D5_TOKEN", "128"))
    g = torch.Generator().manual_seed(8)
    ids = torch.randint(0, st.cfg.vocab_size, (B, T), generator=g)
    R = torch.randn(B * T, st.cfg.hidden_size, generator=g, dtype=torch.float64)
    allenati = [4, 5, 6, 7]

    def passo(misura_indietro: bool):
        A._azzera(st)
        a = _sync_tempo()
        stato = st.avanti(ids)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            (st.esce(xL).to(R.dtype) * R).sum().backward()
        b = _sync_tempo()
        st.indietro(stato, xL.grad)
        c = _sync_tempo()
        del stato, xL
        return c - b, c - a

    # gradienti al bit
    passo(False)
    g_int = [None if p.grad is None else p.grad.clone() for p in st.parametri()] + \
        [None if p.grad is None else p.grad.clone() for p in st.parametri_f32()]
    st.congela_strati(allenati)
    passo(False)
    g_con = [None if p.grad is None else p.grad.clone() for p in st.parametri()] + \
        [None if p.grad is None else p.grad.clone() for p in st.parametri_f32()]
    L_tutti = [L for (L, _) in st.chiavi()] + [st.f32[n].L for n in st.chiavi_f32()]
    al_bit = all(((a is not None and b is not None and torch.equal(a, b)) if (L in allenati or L == -1) else b is None)
                 for a, b, L in zip(g_int, g_con, L_tutti))
    st.congela_strati(None)
    # tempi: alternati, mediane di 5 (dopo un giro di riscaldamento per parte)
    tempi = {"intero": [], "congelato": []}
    passo(True)
    st.congela_strati(allenati)
    passo(True)
    st.congela_strati(None)
    n_rip = int(os.getenv("E2E_D5_RIPETIZIONI", "5"))
    for _ in range(n_rip):
        tempi["intero"].append(passo(True))
        st.congela_strati(allenati)
        tempi["congelato"].append(passo(True))
        st.congela_strati(None)
    med = {k: {"indietro_s": statistics.median(x[0] for x in v), "passo_s": statistics.median(x[1] for x in v)}
           for k, v in tempi.items()}
    rap = {"indietro": med["congelato"]["indietro_s"] / med["intero"]["indietro_s"],
           "passo": med["congelato"]["passo_s"] / med["intero"]["passo_s"]}
    ok = al_bit and rap["indietro"] < 0.75
    return registra("D5_tempo_indietro_otto_strati", ok, gradienti_allenati_al_bit=al_bit, allenati=allenati,
                    token=B * T, mediane=med, rapporto_congelato_su_intero=rap,
                    guadagno=({k: 1.0 / v for k, v in rap.items()}),
                    tempi_grezzi={k: [[round(a, 4), round(b, 4)] for a, b in v] for k, v in tempi.items()},
                    secondi=round(time.time() - t0, 1))


PROVE = {"D1": prova_d1, "D2": prova_d2, "D3": prova_d3, "D4": prova_d4, "D5": prova_d5}


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
    (DIR / f"esito_test_dominio_strati{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
