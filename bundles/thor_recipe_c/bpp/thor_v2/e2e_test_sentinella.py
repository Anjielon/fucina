#!/usr/bin/env python3
"""Prove della sentinella di progresso (e2e_sentinella.py, 28/9/2026, blocco di EB3). SOLO CPU: ogni scenario gira in un
PROCESSO FIGLIO (la sentinella chiude il processo), il padre guarda codice d'uscita, tempo e registro.

  S0 (la diagnosi, ripetibile) la curvatura del passo V ripreso con la riga di EB3 del 28/9 (--lora-ripristina zero
     --lr-lora 1e-5) e con quella di a014d42 che il 27/9 e' passata (niente ripristino, 4e-5), stessa ripresa, modello
     minuscolo: sotto una TorchDispatchMode le op aten sono le STESSE, nello stesso ordine e con le stesse forme e dtype;
     nessun tensore vuoto in piu' (i due di `unique` dell'embedding su CPU ci sono in entrambe). L'unica differenza sono i valori (A - A0 e B - B0 = 0: prodotti
     con un operando tutto zero), che non cambiano i kernel lanciati: la causa del blocco NON e' nel diff a014d42..4ed4a95.
  S1 blocco che RILASCIA il GIL (come `rocr::BusyWaitSignal::WaitRelaxed` sotto `.item()`/`.all()`): uscita 75 entro la
     soglia + un giro di controllo; nel registro la riga ⛔ SENTINELLA (con chi altro tiene la GPU, dal controllo) e le pile
     di tutti i fili, compresa la funzione ferma.
  S2 nessun falso allarme: segni ogni 0,2 s per 4 soglie -> uscita 0, nessuna riga; dopo il blocco la rete in C e' spenta
     (si resta fermi oltre soglia + margine senza essere uccisi).
  S3 blocco col GIL TENUTO (sleep di libc via ctypes.PyDLL: il filo Python non puo' girare): scatta la rete in C
     (faulthandler) a soglia + margine, pile nel registro, uscita 1.
  S4 E2E_SENTINELLA_S=0: spenta (nessun filo, nessuna rete): un vuoto lungo non chiude nulla; lettura dell'ambiente.
  S5 il percorso VERO, la riga di EB3 in piccolo (LoRA-GA + F32 + PV V-Bop con curvatura su 2 micro-lotti + dosi +
     ricentro + pesi per dominio, AdamW, ripresa con --lora-ripristina zero --lr-lora 1e-5, strato 1 allenato):
     (a) un passo sano con la sentinella accesa (soglia 5 s) finisce senza allarmi (i segni di prendi / sonda / ricentro /
     registro bastano); (b) la curvatura del passo V bloccata per sempre (`_curvatura_passo` che aspetta un evento mai
     segnalato, come EB3 alle 15:4x): uscita 75, registro con ⛔ SENTINELLA e le pile passo_v -> curvatura_passo;
     (c) CONTROLLO NEGATIVO: lo stesso blocco con la sentinella SPENTA e' ancora vivo dopo 40 s (la prova sa fallire:
     senza la cura il figlio non esce, e (b) scadrebbe).
  S6 cablaggio: `main` accende la sentinella attorno ad `allena` (solo su cuda); segno() in Caricatore.prendi,
     perdita_sonda_lotti, gradiente_ricentro e nel registro; i due lanciatori (riavvio_integrato.sh, lancia_eb4.sh)
     ripartono SOLO sul codice 75 (RestartForceExitStatus=75) e al piu' 3 volte.
Uso (CPU limitata, niente GPU):
    systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=12G nice -n 19 env HIP_VISIBLE_DEVICES= \\
        CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 E2E_ALLENA_PROVE=<cartella> @@ROOT:venv@@/bin/python3 \\
        e2e_test_sentinella.py [S1 ...]
    esito: $E2E_ALLENA_PROVE/sentinella/esito_test_sentinella.json
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_sentinella as SE  # noqa: E402

DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "sentinella"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
PY = sys.executable


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:3000]}", flush=True)
    return bool(ok)


def _log_su(reg: Path):
    def log(msg: str) -> None:
        SE.segno()
        with open(reg, "a") as f:
            f.write(msg + "\n")
    return log


# ─────────────────────────────── figli (girano in un processo a parte) ───────────────────────────────
def attesa_segnale_gpu_mai_arrivato() -> None:
    """Come EB3: un'attesa senza fine che RILASCIA il GIL (qui un evento mai segnalato)."""
    threading.Event().wait()


def attesa_col_gil() -> None:
    import ctypes
    ctypes.PyDLL(None).sleep(60)                  # PyDLL: il GIL resta tenuto per tutta la chiamata


def figlio(nome: str, d: Path) -> None:
    reg = d / "registro.log"
    log = _log_su(reg)
    gpu = lambda: "finto-pid 4242 (Vulkan, 9 GiB)"  # noqa: E731
    if nome == "s1":
        with SE.sentinella(reg, 1.0, log, controllo_gpu=gpu, margine=60.0):
            attesa_segnale_gpu_mai_arrivato()
    elif nome == "s2":
        with SE.sentinella(reg, 1.0, log, controllo_gpu=gpu, margine=1.0):
            for _ in range(20):
                time.sleep(0.2)
                SE.segno()
        time.sleep(3.0)                           # oltre soglia + margine: la rete in C deve essere spenta
    elif nome == "s3":
        with SE.sentinella(reg, 1.0, log, controllo_gpu=gpu, margine=1.0):
            attesa_col_gil()
    elif nome == "s4":
        s, m = SE.secondi_da_env()
        with SE.sentinella(reg, s, log, controllo_gpu=gpu, margine=m) as sn:
            log(f"spenta={sn is None} secondi={s}")
            time.sleep(2.5)
    elif nome in ("s5a", "s5b", "s5c"):
        figlio_s5(nome, d, reg, log)
    else:
        raise SystemExit(f"figlio ignoto {nome}")


def riga_eb3(**kw):
    """La riga di EB3 in piccolo: AdamW, LoRA-GA gamma 64, tutte le F32 (lr 5e-5), PV V-Bop con curvatura su DUE
    micro-lotti (--pv-micro-curvatura 2 con 2 righe a micro), sonda a dosi + ricentro 3 giri, pesi per dominio, CAKLD,
    attribuzione; lr_lora 4e-5 come la riga del 27/9 (a014d42)."""
    import e2e_test_integrata as TI
    base = dict(ottimizzatore_theta="adamw", lr_f32=5e-5, righe_per_passo=8, righe_per_micro=2,
                pv_micro_curvatura=2, pv_sonda_dosi=True,
                pv_dosi_frazioni=(1 / 4096, 1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0), pv_ricentra=True,
                pv_ricentra_giri=3, peso_dominio=(("a", 1.0), ("b", 1.5)), peso_token="uniforme", lr_lora=4e-5)
    base.update(kw)
    return TI._ca_integrata(**base)


def figlio_s5(nome: str, d: Path, reg: Path, log) -> None:
    import e2e_allena as A
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    import torch
    torch.manual_seed(0)
    riga = riga_eb3
    src, WS = TL.sorgente_lora(n_train=48, n_held=4)
    dc = d / "corsa"
    A.allena(TI._studente_int(), src, dc, riga(), WS, WS, ferma_a=2, log=lambda *a, **k: None)   # base, senza sentinella
    if nome in ("s5b", "s5c"):
        def _curvatura_ferma(*a, **k):
            attesa_segnale_gpu_mai_arrivato()
        A._curvatura_passo = _curvatura_ferma
    secondi = 0.0 if nome == "s5c" else 5.0
    with SE.sentinella(reg, secondi, log, controllo_gpu=lambda: "finto", margine=60.0):
        A.allena(TI._studente_int(), src, dc, riga(strati_allenati=(1,), lora_ripristina="zero", lr_lora=1e-5), WS, WS,
                 riprendi=True, ferma_a=3, log=log, reinit_lora_ga=True)
    log("FINE PASSO")


def lancia(nome: str, scadenza: float, env_extra=None) -> dict:
    d = DIR / nome
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    env = dict(os.environ, **(env_extra or {}))
    t0 = time.time()
    p = subprocess.Popen([PY, "-W", "ignore", str(Path(__file__).resolve()), "--figlio", nome, str(d)], env=env,
                         stdout=open(d / "stdout.txt", "w"), stderr=subprocess.STDOUT)
    try:
        rc = p.wait(timeout=scadenza)
        vivo = False
    except subprocess.TimeoutExpired:
        vivo = True
        p.kill()
        rc = p.wait()
    reg = d / "registro.log"
    testo = reg.read_text() if reg.exists() else ""
    return {"rc": None if vivo else rc, "vivo_alla_scadenza": vivo, "secondi": round(time.time() - t0, 2),
            "registro": testo, "stdout": (d / "stdout.txt").read_text()[-2000:]}


# ─────────────────────────────── prove ───────────────────────────────
def _traccia_curvatura(A, fn):
    """Esegue fn() con `_curvatura_passo` sotto una TorchDispatchMode: per ogni curvatura la lista delle op aten
    (nome, forme/dtype di ingressi e uscite), i tensori a 0 elementi, le uscite non finite, i prodotti con un operando
    tutto zero."""
    import torch
    from torch.utils import _pytree as pytree
    from torch.utils._python_dispatch import TorchDispatchMode
    prodotti = {"aten.mm.default", "aten.addmm.default", "aten.bmm.default"}

    class Spia(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops, self.vuoti, self.non_finiti, self.zero = [], 0, 0, 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            ins = [t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
            outs = [t for t in pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
            self.ops.append((str(func), tuple((tuple(t.shape), str(t.dtype)) for t in ins),
                             tuple((tuple(t.shape), str(t.dtype)) for t in outs)))
            self.vuoti += any(t.numel() == 0 for t in ins + outs)
            self.non_finiti += any(t.is_floating_point() and t.numel() and not bool(torch.isfinite(t).all())
                                   for t in outs)
            self.zero += str(func) in prodotti and any(t.is_floating_point() and t.numel() and not bool(t.any())
                                                        for t in ins)
            return out
    spie, orig = [], A._curvatura_passo

    def spiato(*a, **k):
        s = Spia()
        with s:
            r = orig(*a, **k)
        spie.append(s)
        return r
    A._curvatura_passo = spiato
    try:
        fn()
    finally:
        A._curvatura_passo = orig
    return spie


def prova_s0() -> bool:
    import e2e_allena as A
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    import torch
    t0 = time.time()
    torch.manual_seed(0)
    src, WS = TL.sorgente_lora(n_train=48, n_held=4)
    d0 = DIR / "s0_base"
    shutil.rmtree(d0, ignore_errors=True)
    A.allena(TI._studente_int(), src, d0, riga_eb3(), WS, WS, ferma_a=2, log=lambda *a, **k: None)
    tracce = {}
    for nome, ca in (("zero_28_9", riga_eb3(strati_allenati=(1,), lora_ripristina="zero", lr_lora=1e-5)),
                     ("a014d42_27_9", riga_eb3(strati_allenati=(1,)))):
        d = DIR / f"s0_{nome}"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(d0, d)
        tracce[nome] = _traccia_curvatura(A, lambda: A.allena(TI._studente_int(), src, d, ca, WS, WS, riprendi=True,
                                                              ferma_a=3, log=lambda *a, **k: None,
                                                              reinit_lora_ga=True))
    z, n = tracce["zero_28_9"], tracce["a014d42_27_9"]
    c = {"una_curvatura_ciascuna": len(z) == len(n) == 1,
         "stesse_op_stesse_forme": [s.ops for s in z] == [s.ops for s in n],
         "vuoti_uguali": [s.vuoti for s in z] == [s.vuoti for s in n],
         "solo_valori_diversi_operandi_zero": all(a.zero > b.zero for a, b in zip(z, n))}
    return registra("S0_curvatura_identica_con_e_senza_ripristino", all(c.values()), **c,
                    op=[len(s.ops) for s in z], operandi_zero={"zero_28_9": [s.zero for s in z],
                                                               "a014d42_27_9": [s.zero for s in n]},
                    tensori_vuoti=[s.vuoti for s in z], secondi=round(time.time() - t0, 1),
                    uscite_non_finite={"zero_28_9": [s.non_finiti for s in z], "a014d42_27_9": [s.non_finiti for s in n],
                                       "nota": "buffer torch.empty non ancora scritti e loro viste: memoria non inizializzata"})


def prova_s1() -> bool:
    r = lancia("s1", 30)
    t = r["registro"]
    c = {"uscita_75": r["rc"] == SE.CODICE_USCITA, "in_tempo": r["secondi"] < 15,
         "riga_sentinella": "⛔ SENTINELLA: nessun progresso" in t and "finto-pid 4242" in t,
         "pile_con_la_funzione_ferma": "attesa_segnale_gpu_mai_arrivato" in t and "Thread 0x" in t}
    return registra("S1_blocco_senza_gil", all(c.values()), **c, rc=r["rc"], secondi=r["secondi"])


def prova_s2() -> bool:
    r = lancia("s2", 30)
    c = {"uscita_0": r["rc"] == 0, "nessun_allarme": "SENTINELLA:" not in r["registro"] and "Timeout" not in r["registro"],
         "accesa": "sentinella di progresso accesa" in r["registro"], "durata": 6.0 <= r["secondi"] < 20}
    return registra("S2_nessun_falso_allarme_e_rete_spenta", all(c.values()), **c, rc=r["rc"], secondi=r["secondi"])


def prova_s3() -> bool:
    r = lancia("s3", 30)
    t = r["registro"]
    c = {"uscita_1": r["rc"] == 1, "in_tempo": r["secondi"] < 15, "rete_in_c": "Timeout (0:00:02)!" in t,
         "pile_con_la_funzione_ferma": "attesa_col_gil" in t, "filo_python_non_girato": "SENTINELLA:" not in t}
    return registra("S3_blocco_col_gil_rete_in_c", all(c.values()), **c, rc=r["rc"], secondi=r["secondi"])


def prova_s4() -> bool:
    r = lancia("s4", 30, {"E2E_SENTINELLA_S": "0"})
    c = {"uscita_0": r["rc"] == 0, "spenta": "spenta=True secondi=0.0" in r["registro"],
         "nessun_allarme": "SENTINELLA:" not in r["registro"],
         "default_1800": SE.secondi_da_env({}) == (1800.0, 600.0),
         "env_letto": SE.secondi_da_env({"E2E_SENTINELLA_S": "90", "E2E_SENTINELLA_MARGINE_S": "7"}) == (90.0, 7.0)}
    try:
        SE.Sentinella(DIR / "x.log", 5.0, print, margine=0.0)
        c["margine_zero_rifiutato"] = False
    except ValueError:
        c["margine_zero_rifiutato"] = True
    return registra("S4_spenta_e_ambiente", all(c.values()), **c, rc=r["rc"])


def prova_s5() -> bool:
    a = lancia("s5a", 600)
    b = lancia("s5b", 600)
    n = lancia("s5c", 40 + b["secondi"])          # la base + il passo fino alla curvatura, poi 40 s di blocco
    tb = b["registro"]
    c = {"a_passo_sano_finito": a["rc"] == 0 and "FINE PASSO" in a["registro"],
         "a_nessun_allarme": "SENTINELLA:" not in a["registro"],
         "a_ripristino_fatto": "--lora-ripristina zero al passo 2" in a["registro"],
         "b_uscita_75": b["rc"] == SE.CODICE_USCITA,
         "b_riga_sentinella": "⛔ SENTINELLA: nessun progresso" in tb,
         "b_pile_passo_v_curvatura": all(x in tb for x in ("attesa_segnale_gpu_mai_arrivato", "curvatura_passo",
                                                            "passo_v", "in allena")),
         "c_senza_sentinella_ancora_vivo": n["vivo_alla_scadenza"] and "FINE PASSO" not in n["registro"]}
    return registra("S5_riga_EB3_in_piccolo_curvatura_bloccata", all(c.values()), **c,
                    rc=[a["rc"], b["rc"], n["rc"]], secondi=[a["secondi"], b["secondi"], n["secondi"]],
                    stdout_b=b["stdout"][-600:] if not c["b_uscita_75"] else "")


def prova_s6() -> bool:
    import e2e_allena as A
    m = inspect.getsource(A.main)
    i_s, i_a = m.find("SE.sentinella("), m.find("stato = allena(st, src, dir_c")
    c = {"main_accende_attorno_ad_allena": 0 <= i_s < i_a and "dev.type == \"cuda\"" in m[i_s:i_a],
         "segno_prendi": "SE.segno()" in inspect.getsource(A.Caricatore.prendi),
         "segno_sonda": "SE.segno()" in inspect.getsource(A.perdita_sonda_lotti),
         "segno_ricentro": "SE.segno()" in inspect.getsource(A.gradiente_ricentro),
         "segno_registro": "SE.segno()" in inspect.getsource(A._registro)}
    for f in ("riavvio_integrato.sh", "lancia_eb4.sh"):
        t = (QUI / "strumenti" / f).read_text()
        riga = next((x for x in t.splitlines() if x.startswith("LANCIO=(systemd-run")), "")
        c[f"lanciatore_{f}"] = ("-p RestartForceExitStatus=75" in riga and "-p StartLimitBurst=3" in riga
                                and "Restart=on-failure" not in riga and "Restart=always" not in riga)
    return registra("S6_cablaggio", all(c.values()), **c)


PROVE = {"S0": prova_s0, "S1": prova_s1, "S2": prova_s2, "S3": prova_s3, "S4": prova_s4, "S5": prova_s5, "S6": prova_s6}


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--figlio":
        figlio(sys.argv[2], Path(sys.argv[3]))
        return 0
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = sys.argv[1:] or list(PROVE)
    ok = True
    for k in scelte:
        try:
            ok &= PROVE[k]()
        except Exception as e:  # noqa: BLE001 - una prova rotta e' un rosso, le altre girano
            import traceback
            ok &= registra(k, False, eccezione=f"{type(e).__name__}: {e}", traccia=traceback.format_exc()[-1500:])
    ESITO["tutte_ok"] = bool(ok)
    (DIR / "esito_test_sentinella.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print("TUTTE OK" if ok else "QUALCUNA ROSSA", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
