#!/usr/bin/env python3
"""Collaudo delle patch di VELOCITA' della forgia scorrevole (25/9 notte) — CPU, scala giocattolo.

Non tocca NESSUN file di thor_v2: copia i sorgenti in una cartella di lavoro, ci applica le
patch (una alla volta, tutte insieme, tutte in ordine inverso) e confronta le corse con la copia
NON patchata. Ogni corsa gira in un processo a parte, solo CPU:

    cd @@ROOT:forge@@/thor_v2 && env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \\
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 strumenti/collaudo_patch_tempo.py --lavoro DIR [--suite]

La cartella di lavoro deve avere accanto `kernel-rotazione` (gptq_1024_lib lo cerca in ../):
il collaudo crea il collegamento se manca.

Scenari (strati Qwen3_5MoeDecoderLayer VERI a dimensioni giocattolo, attenzione LINEARE e
PIENA mescolate come in Thor: piena agli strati 3, 7, ...):
  prod8   ricetta di produzione (solo fase dura, rotazione, nucleo bf16, routing aggiornato,
          diagnostica spenta), strati bf16, 8 strati, W=2, 5 passi, scrittura asincrona,
          interfaccia vera (raccordo), verifica del seme dalla finestra 3, cache d'attenzione
          verificata alla prima visita di OGNI batch.
  base6   ricetta di base (fase morbida + dura: beta che cambia, diagnostica ACCESA), fp32, 6 strati.
  ripresa6 produzione, 6 strati: corsa intera; corsa interrotta nel thread principale
          (prima_transizione@2) e ripresa; corsa interrotta nel thread di scrittura
          (dopo_ripresa@3) e ripresa.
  gara6   oggetti portati allenati IN PLACE nella finestra dopo (SostitutoA del collaudo della
          forgia) e scrittura del punto di ripresa rallentata di 1 s nel thread: il punto di
          ripresa deve restare quello della fine della finestra. Nelle copie con la patch
          prepara_asincrona gira anche gara6_senza_copia (fotografia SENZA copia): DEVE differire
          (prova che il collaudo vede la gara).
Per ogni copia e ogni scenario: codici su disco (tutti i .npy) IDENTICI AL BIT alla copia non
patchata, sequenza delle loss dei passi identica (float.hex), punti di ripresa (stato portato e
semi) identici al bit; contatori che provano che ogni leva ha lavorato davvero.
Con --suite: test_forgia_scorrevole.py e test_raccordo_scorrevole.py (base e produzione) sulla
copia con TUTTE le patch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

QUI = Path(__file__).resolve()
THOR_V2 = QUI.parent.parent
PATCH = ["prepara_asincrona", "riuso_teacher", "prefetch_strato", "cache_attenzione", "beta_costante",
         "congelo_senza_moe", "attenzione_confine"]
# attenzione_confine RICHIEDE cache_attenzione (aggancio scritto da quella): nell'ordine inverso resta ultima
VARIANTI = {
    "base": [],
    "C_prepara": ["prepara_asincrona"],
    "B_teacher": ["riuso_teacher"],
    "A_prefetch": ["prefetch_strato"],
    "L1_attenzione": ["cache_attenzione"],
    "L1b_confine": ["cache_attenzione", "attenzione_confine"],
    "L4_beta": ["beta_costante"],
    "L6_congelo": ["congelo_senza_moe"],
    "tutte": PATCH,
    "tutte_inverso": list(reversed(PATCH[:-1])) + ["attenzione_confine"],
}
# stesse variabili della forgia veloce/scorrevole (test_raccordo_scorrevole.RICETTE["produzione"])
RICETTA_PRODUZIONE = {
    "THOR_VALIDA_FRAZIONE_DURA": "1.0", "THOR_ROTAZIONE_INGRESSO": "1", "THOR_NUCLEO_BF16": "1",
    "THOR_ROUTING_AGGIORNATO": "1", "THOR_ROUTING_VIVO": "0", "THOR_DIAGNOSTICA_PASSO": "0",
    "THOR_HELDOUT_OGNI": "0", "THOR_RICONGELA_OGNI": "0", "THOR_FORGIA_DOWN_TERNARIO": "1",
    "THOR_LR_Z": "0.01", "THOR_LR_LOGS": "0.01", "THOR_LR_DELTA": "0.01", "THOR_VALIDA_LR": "0.01",
    "THOR_ESPERTI_THOR": "1", "THOR_SALTA_NON_FINITI": "1", "THOR_COMMIT_DA_CODICI": "1",
    "THOR_COMMIT_SCALA_FP16": "1",
}
SCENARI = {
    "prod8": {"ricetta": RICETTA_PRODUZIONE, "extra": {"THOR_CACHE_ATTENZIONE_VERIFICA": "ogni_batch"}},
    "base6": {"ricetta": {"THOR_FORGIA_DOWN_TERNARIO": "1"}, "extra": {"THOR_CACHE_ATTENZIONE_VERIFICA": "ogni_batch"}},
    "ripresa6": {"ricetta": RICETTA_PRODUZIONE, "extra": {}},
    "gara6": {"ricetta": {"THOR_FORGIA_DOWN_TERNARIO": "1"}, "extra": {}},
    "gara6_senza_copia": {"ricetta": {"THOR_FORGIA_DOWN_TERNARIO": "1"}, "extra": {}},
    # modo Adam "conserva" (25/9, altro lavoro sul raccordo): la finestra dopo ADOTTA i momenti di Adam dello
    # strato portato e li aggiorna sul posto; scrittura del punto di ripresa rallentata di 1 s nel thread
    "adam6_gara": {"ricetta": {**RICETTA_PRODUZIONE, "THOR_SCORREVOLE_ADAM": "conserva"}, "extra": {}},
    "adam6_gara_copia_parziale": {"ricetta": {**RICETTA_PRODUZIONE, "THOR_SCORREVOLE_ADAM": "conserva"}, "extra": {}},
}
SCENARI_CONTROPROVA = {"gara6_senza_copia": "gara6", "adam6_gara_copia_parziale": "adam6_gara"}
ENV_CPU = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONDONTWRITEBYTECODE": "1"}
ESITI: list = []


def ok(nome: str, cond, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(cond)))
    print(f"    {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(cond)


# ═══════════════════════════ processo FIGLIO: una corsa giocattolo ═══════════════════════════
def figlio(cartella: Path, scenario: str, uscita: Path) -> int:
    cartella, uscita = Path(cartella).resolve(), Path(uscita).resolve()
    os.chdir(cartella)
    sys.path.insert(0, str(cartella))
    import numpy as np  # noqa: F401
    import torch
    assert not torch.cuda.is_available(), "solo CPU"
    import valida_finestra_8_12 as V
    import forgia_scorrevole as FS
    import test_forgia_scorrevole as TFS
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

    uscita.mkdir(parents=True, exist_ok=True)
    reg = {"perdite": [], "teacher_chiamate": {}, "attenzione_classe": {"lineare": 0, "piena": 0}, "moe_blocchi": 0,
           "carica": [], "prepara_thread": [], "snap": []}
    blocco = threading.Lock()

    # -- loss di ogni passo, al bit
    orig_passo = V.passo_ottimizzazione

    def passo_spia(*a, **k):
        r = orig_passo(*a, **k)
        reg["perdite"].append([float(r["loss"].detach().float().item()).hex(),
                               float(r["loss_somma"].detach().float().item()).hex(), r.get("saltato")])
        return r
    V.passo_ottimizzazione = passo_spia

    # -- attenzione davvero calcolata (la cache la scavalca a livello d'istanza)
    orig_moe = M.Qwen3_5MoeSparseMoeBlock.forward

    def moe_contato(self, *a, **k):
        with blocco:
            reg["moe_blocchi"] += 1
        return orig_moe(self, *a, **k)
    M.Qwen3_5MoeSparseMoeBlock.forward = moe_contato
    for classe, nome in ((M.Qwen3_5MoeGatedDeltaNet, "lineare"), (M.Qwen3_5MoeAttention, "piena")):
        orig = classe.forward

        def contata(self, *a, _orig=orig, _nome=nome, **k):
            with blocco:
                reg["attenzione_classe"][_nome] += 1
            return _orig(self, *a, **k)
        classe.forward = contata

    # -- punti di ripresa: copia di ogni punto completo prima che venga cancellato
    orig_completa = FS.Ripresa.completa

    def completa_spia(self, K, tmp, seme_h, seme_e, portati):
        orig_completa(self, K, tmp, seme_h, seme_e, portati)
        dest = uscita / "ripresa_snap" / f"{len(reg['snap']):02d}_finestra_{K:02d}"
        shutil.copytree(self.cartella(K), dest)
        reg["snap"].append(dest.name)
    FS.Ripresa.completa = completa_spia
    for nome_m in ("prepara", "prepara_da_foto"):
        if hasattr(FS.Ripresa, nome_m):
            orig_m = getattr(FS.Ripresa, nome_m)

            def spia_m(self, *a, _orig=orig_m, _n=nome_m, **k):
                reg["prepara_thread"].append([_n, threading.current_thread().name])
                return _orig(self, *a, **k)
            setattr(FS.Ripresa, nome_m, spia_m)

    # -- strati giocattolo con attenzione lineare e piena (piena a 3, 7, ... come Thor)
    class GiocattoloMisto(TFS.Giocattolo):
        def __init__(self, n, dtype):
            super().__init__(n, dtype)
            c = self.cfg
            c.layer_types = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(n)]
            c.linear_conv_kernel_dim, c.linear_key_head_dim, c.linear_value_head_dim = 2, 4, 4
            c.linear_num_key_heads, c.linear_num_value_heads = 2, 2

        def carica(self, layers):
            reg["carica"].append([list(layers), threading.current_thread().name])
            cfg, emb, pul, qua = super().carica(layers)
            for L, s in pul.items():
                def conta(_m, _a, _o, _L=L):
                    with blocco:
                        reg["teacher_chiamate"][str(_L)] = reg["teacher_chiamate"].get(str(_L), 0) + 1
                s.register_forward_hook(conta)
            return cfg, emb, pul, qua

    def esegui(dirc: Path, gioc, n_passi=5, interrompi=None, ottimizza_fn=None, verifica_seme_a=None,
               asincrona=True, corsa_id="collaudo"):
        V.OUT_DIR = dirc / "out"
        V.ID_CORSA = corsa_id
        FS.esegui(gioc.n, 2, n_passi, "teacher", "", "", salta_export=True, carica_strati=gioc.carica,
                  carica_batch=gioc.batch, seme_iniziale=gioc.seme0, ottimizza_fn=ottimizza_fn,
                  ripresa_dir=dirc / "ripresa", scrittura_asincrona=asincrona,
                  verifica_seme_a=verifica_seme_a or [], interrompi=interrompi)

    def interruzione(punto_nome, k_int):
        def f(p, k):
            if p == punto_nome and k == k_int:
                raise FS.InterruzioneSimulata(f"{p}@{k}")
        return f

    t0 = time.time()
    if scenario == "prod8":
        esegui(uscita / "corsa", GiocattoloMisto(8, torch.bfloat16), verifica_seme_a=[3])
    elif scenario == "base6":
        esegui(uscita / "corsa", GiocattoloMisto(6, torch.float32), n_passi=6)
    elif scenario == "ripresa6":
        esegui(uscita / "intera", GiocattoloMisto(6, torch.bfloat16))
        for etichetta, (pn, kk) in (("int_principale", ("prima_transizione", 2)), ("int_scrittura", ("dopo_ripresa", 3))):
            d = uscita / etichetta
            try:
                esegui(d, GiocattoloMisto(6, torch.bfloat16), interrompi=interruzione(pn, kk))
                reg.setdefault("interruzioni_mancate", []).append(etichetta)
            except FS.InterruzioneSimulata:
                pass
            reg.setdefault("commessi_dopo_interruzione", {})[etichetta] = FS.strati_commessi(d / "out", 6, "collaudo", True)
            esegui(d, GiocattoloMisto(6, torch.bfloat16))
    elif scenario in ("gara6", "gara6_senza_copia", "adam6_gara", "adam6_gara_copia_parziale"):
        if scenario in SCENARI_CONTROPROVA:
            if not hasattr(FS, "_stato_ternario_copia"):
                (uscita / "risultato.json").write_text(json.dumps({"non_applicabile": True}))
                return 0
            if scenario == "gara6_senza_copia":
                FS._stato_ternario_copia = FS._stato_ternario        # fotografia SENZA copia: la gara deve vedersi
            else:
                # copia del SOLO stato latente (la prima stesura della patch): i momenti di Adam che la finestra
                # dopo aggiorna sul posto finirebbero su disco a meta' aggiornamento
                def _parziale(obj):
                    s = FS._stato_ternario(obj)
                    s["stato"] = FS._copia_profonda(s["stato"])
                    return s
                FS._stato_ternario_copia = _parziale
        orig_save = torch.save

        def save_lento(obj, f, *a, **k):
            if threading.current_thread() is not threading.main_thread() and Path(str(f)).name.startswith("portato_"):
                time.sleep(1.0)
            return orig_save(obj, f, *a, **k)
        torch.save = save_lento
        if scenario.startswith("adam6"):
            esegui(uscita / "corsa", GiocattoloMisto(6, torch.bfloat16))           # raccordo: interfaccia vera
        else:
            esegui(uscita / "corsa", GiocattoloMisto(6, torch.float32), ottimizza_fn=TFS.SostitutoA().ottimizza)
        torch.save = orig_save
    else:
        raise SystemExit(f"scenario sconosciuto {scenario}")
    reg["secondi"] = round(time.time() - t0, 2)
    quant = sys.modules.get("quant_ternaria_globale_batch_v2")
    reg["beta_tensori_in_cache"] = len(getattr(quant, "_BETA_TENSORI", {})) if quant else None
    (uscita / "risultato.json").write_text(json.dumps(reg, indent=1))
    return 0


# ═══════════════════════════ processo PADRE: copie, corse, confronti ═══════════════════════════
def _sorgenti(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "strumenti").mkdir(parents=True)
    for pat in ("*.py", "*.json", "*.npz"):
        for f in THOR_V2.glob(pat):
            shutil.copy2(f, dest / f.name)
    for f in (THOR_V2 / "strumenti").glob("*.py"):
        shutil.copy2(f, dest / "strumenti" / f.name)


def prepara_copie(lavoro: Path, py: str) -> dict:
    print("\n══ copie e applicazione delle patch ══")
    lavoro.mkdir(parents=True, exist_ok=True)
    kr = lavoro / "kernel-rotazione"
    if not kr.exists():
        kr.symlink_to(THOR_V2.parent / "kernel-rotazione")
    firme_vere = {f.name: f.read_bytes() for f in THOR_V2.glob("*.py")}
    cartelle = {}
    for nome, patch in VARIANTI.items():
        d = lavoro / nome
        _sorgenti(d)
        for p in patch:
            r = subprocess.run([py, str(d / "strumenti" / f"patch_tempo_{p}.py"), "--cartella", str(d)],
                               capture_output=True, text=True, env={**os.environ, **ENV_CPU})
            if not ok(f"{nome}: patch {p} applicata", r.returncode == 0, (r.stdout + r.stderr).strip()[-160:]):
                raise SystemExit(1)
        cartelle[nome] = d
    # rifiuti: patch gia' applicata, aggancio mancante (nessuna scrittura parziale)
    d = cartelle["tutte"]
    prima = {f.name: f.read_bytes() for f in d.glob("*.py")}
    for p in PATCH:
        r = subprocess.run([py, str(d / "strumenti" / f"patch_tempo_{p}.py"), "--cartella", str(d)],
                           capture_output=True, text=True, env={**os.environ, **ENV_CPU})
        ok(f"patch {p} applicata due volte: RIFIUTATA", r.returncode != 0 and "gia' applicata" in (r.stdout + r.stderr))
    rotta = lavoro / "_rotta"
    _sorgenti(rotta)
    fs = rotta / "forgia_scorrevole.py"
    fs.write_text(fs.read_text().replace("    scrittore = ThreadPoolExecutor(max_workers=1)\n", "    scrittore = ThreadPoolExecutor(1)\n"))
    valida_prima = (rotta / "valida_finestra_8_12.py").read_bytes()
    r = subprocess.run([py, str(rotta / "strumenti" / "patch_tempo_prefetch_strato.py"), "--cartella", str(rotta)],
                       capture_output=True, text=True, env={**os.environ, **ENV_CPU})
    ok("aggancio mancante in UN file: patch RIFIUTATA e NESSUN file scritto (nemmeno l'altro)",
       r.returncode != 0 and "aggancio trovato 0 volte" in (r.stdout + r.stderr)
       and (rotta / "valida_finestra_8_12.py").read_bytes() == valida_prima,
       (r.stdout + r.stderr).strip()[-120:])
    shutil.rmtree(rotta)
    sola = lavoro / "_sola_confine"
    _sorgenti(sola)
    prima_sola = {f.name: f.read_bytes() for f in sola.glob("*.py")}
    r = subprocess.run([py, str(sola / "strumenti" / "patch_tempo_attenzione_confine.py"), "--cartella", str(sola)],
                       capture_output=True, text=True, env={**os.environ, **ENV_CPU})
    ok("attenzione_confine SENZA cache_attenzione: RIFIUTATA, nessun file scritto",
       r.returncode != 0 and {f.name: f.read_bytes() for f in sola.glob("*.py")} == prima_sola,
       (r.stdout + r.stderr).strip()[-110:])
    shutil.rmtree(sola)
    ok("i file di thor_v2 NON sono stati toccati", {f.name: f.read_bytes() for f in THOR_V2.glob("*.py")} == firme_vere)
    ok("i file della copia 'tutte' non cambiano dopo i rifiuti", {f.name: f.read_bytes() for f in d.glob("*.py")} == prima)
    return cartelle


def corri(py: str, cartella: Path, scenario: str, uscita: Path) -> dict:
    if uscita.exists():
        shutil.rmtree(uscita)
    uscita.mkdir(parents=True)
    env = {**os.environ, **ENV_CPU, **SCENARI[scenario]["ricetta"], **SCENARI[scenario]["extra"]}
    t0 = time.time()
    with open(uscita / "log.txt", "w") as flog:
        r = subprocess.run([py, str(QUI), "--figlio", "--cartella", str(cartella), "--scenario", scenario,
                            "--uscita", str(uscita)], stdout=flog, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        coda = (uscita / "log.txt").read_text()[-1500:]
        print(coda)
        if cartella.name == "base" and scenario.startswith("adam6"):
            return None      # il modo conserva del sorgente di oggi non gira: scenario non eseguibile (non conta)
        raise SystemExit(f"corsa {cartella.name}/{scenario} fallita (rc {r.returncode}), log in {uscita / 'log.txt'}")
    ris = json.loads((uscita / "risultato.json").read_text())
    ris["_secondi_processo"] = round(time.time() - t0, 1)
    ris["_log"] = (uscita / "log.txt").read_text()
    return ris


def _codici(d: Path) -> dict:
    import numpy as np
    return {f.relative_to(d).as_posix(): np.load(f) for f in sorted(d.rglob("*.npy"))}


def _uguali_codici(a: Path, b: Path) -> "tuple[bool, str]":
    import numpy as np
    ca, cb = _codici(a), _codici(b)
    if sorted(ca) != sorted(cb):
        return False, f"file diversi: {sorted(set(ca) ^ set(cb))[:6]}"
    if not ca:
        return False, "nessun codice"
    diversi = [k for k in ca if ca[k].dtype != cb[k].dtype or ca[k].shape != cb[k].shape or not np.array_equal(ca[k], cb[k])]
    return not diversi, f"{len(ca)} file .npy" + (f", diversi: {diversi[:6]}" if diversi else ", tutti identici al bit")


def _struttura_uguale(x, y) -> bool:
    import torch
    if isinstance(x, dict):
        return isinstance(y, dict) and list(x) == list(y) and all(_struttura_uguale(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(_struttura_uguale(a, b) for a, b in zip(x, y))
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and bool(torch.equal(x, y))
    return x == y


def _uguali_snap(a: Path, b: Path) -> "tuple[bool, str]":
    import torch
    fa = sorted(p.relative_to(a).as_posix() for p in a.rglob("*.pt")) if a.exists() else []
    fb = sorted(p.relative_to(b).as_posix() for p in b.rglob("*.pt")) if b.exists() else []
    if fa != fb:
        return False, f"insiemi di file diversi ({len(fa)} contro {len(fb)})"
    if not fa:
        return True, "nessun punto di ripresa"
    diversi = [f for f in fa if not _struttura_uguale(torch.load(a / f, weights_only=True),
                                                        torch.load(b / f, weights_only=True))]
    return not diversi, f"{len(fa)} file .pt" + (f", diversi: {diversi[:4]}" if diversi else ", tutti identici al bit")


def confronta(nome: str, scenario: str, rif: dict, var: dict, d_rif: Path, d_var: Path) -> None:
    sotto_rif = sorted({p.parent.parent.relative_to(d_rif).as_posix() for p in d_rif.rglob("*.npy")
                        if p.parent.name == "out"})
    ok(f"{nome}/{scenario}: stesse corse della copia NON patchata ({sotto_rif})", bool(sotto_rif) and sotto_rif == sorted(
        {p.parent.parent.relative_to(d_var).as_posix() for p in d_var.rglob("*.npy") if p.parent.name == "out"}))
    for sotto in sotto_rif:
        uguali, det = _uguali_codici(d_rif / sotto / "out", d_var / sotto / "out")
        ok(f"{nome}/{scenario}/{sotto}: codici su disco identici alla copia NON patchata", uguali, det)
    ok(f"{nome}/{scenario}: loss di ogni passo identiche al bit ({len(rif['perdite'])} passi)",
       rif["perdite"] == var["perdite"] and len(rif["perdite"]) > 0,
       "" if rif["perdite"] == var["perdite"] else f"{len(var['perdite'])} passi, primo diverso "
       f"{next((i for i, (a, b) in enumerate(zip(rif['perdite'], var['perdite'])) if a != b), '-')}")
    uguali, det = _uguali_snap(d_rif / "ripresa_snap", d_var / "ripresa_snap")
    ok(f"{nome}/{scenario}: punti di ripresa (stato portato + semi) identici al bit", uguali, det)


def attivita(nome: str, scenario: str, rif: dict, var: dict, patch: list) -> None:
    """Le leve hanno lavorato davvero (non solo 'numeri uguali perche' spente')."""
    tr = sum(rif["teacher_chiamate"].values())
    tv = sum(var["teacher_chiamate"].values())
    if "riuso_teacher" in patch:
        ok(f"{nome}/{scenario}: teacher calcolato MENO volte (riuso fra finestre)", tv < tr, f"{tr} -> {tv} passaggi di strato")
        ok(f"{nome}/{scenario}: riuso annunciato e controllato al bit nel log",
           "RIUSATO dalla finestra precedente" in var["_log"])
    ar, av = rif["attenzione_classe"], var["attenzione_classe"]
    if "cache_attenzione" in patch:
        ok(f"{nome}/{scenario}: attenzione calcolata MENO volte (cache del primo strato)",
           av["lineare"] + av["piena"] < ar["lineare"] + ar["piena"], f"{ar} -> {av}")
        ok(f"{nome}/{scenario}: controllo al bit della cache superato e annunciato",
           "IDENTICA AL BIT a quella del congelamento" in var["_log"])
    if "prefetch_strato" in patch:
        dal_thread = [c for c in var["carica"] if c[1].startswith("prefetch_strato")]
        ok(f"{nome}/{scenario}: strati entranti caricati dal thread di prefetch", len(dal_thread) > 0,
           f"{len(dal_thread)} caricamenti nel thread: {[c[0] for c in dal_thread]}")
    if "prepara_asincrona" in patch and scenario != "gara6_senza_copia":
        foto = [p for p in var["prepara_thread"] if p[0] == "prepara_da_foto"]
        ok(f"{nome}/{scenario}: stato portato scritto nel thread di scrittura",
           len(foto) > 0 and all(p[1] != "MainThread" for p in foto)
           and not any(p[0] == "prepara" for p in var["prepara_thread"]),
           f"{len(foto)} scritture: thread {sorted({p[1] for p in foto})}")
    if "congelo_senza_moe" in patch:
        ok(f"{nome}/{scenario}: blocchi MoE calcolati MENO volte (congelamento fermo al router dell'ultimo strato)",
           var["moe_blocchi"] < rif["moe_blocchi"], f"{rif['moe_blocchi']} -> {var['moe_blocchi']}")
    if "attenzione_confine" in patch:
        ok(f"{nome}/{scenario}: confine con l'attenzione dalla cache, batch 0 verificato al bit (log)",
           "dalla cache della finestra (batch 0 ricalcolato: identico al bit)" in var["_log"])
    if "beta_costante" in patch:
        ok(f"{nome}/{scenario}: beta dal tensore in cache", (var.get("beta_tensori_in_cache") or 0) > 0,
           f"{var.get('beta_tensori_in_cache')} tensori beta distinti")


def suite(py: str, cartella: Path, lavoro: Path) -> None:
    print("\n══ suite esistenti sulla copia con TUTTE le patch ══")
    env = {**os.environ, **ENV_CPU}
    comandi = [("test_forgia_scorrevole", [py, "test_forgia_scorrevole.py", "--cartella", str(lavoro / "suite_forgia")]),
               ("test_raccordo_scorrevole base", [py, "test_raccordo_scorrevole.py", "--ricetta", "base",
                                                  "--cartella", str(lavoro / "suite_racc_base")]),
               ("test_raccordo_scorrevole produzione", [py, "test_raccordo_scorrevole.py", "--ricetta", "produzione",
                                                        "--cartella", str(lavoro / "suite_racc_prod")])]
    for nome, cmd in comandi:
        t0 = time.time()
        r = subprocess.run(cmd, cwd=cartella, env=env, capture_output=True, text=True)
        (lavoro / f"{nome.replace(' ', '_')}.log").write_text(r.stdout + r.stderr)
        riga = [x for x in r.stdout.splitlines() if "controlli" in x and ("TUTTI" in x or "FALLITI" in x)]
        ok(f"suite {nome}", r.returncode == 0, (riga[-1] if riga else r.stderr[-200:]).strip() + f" [{time.time()-t0:.0f}s]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lavoro", type=Path, help="cartella di lavoro (copie e corse)")
    ap.add_argument("--suite", action="store_true", help="anche le suite esistenti sulla copia con tutte le patch")
    ap.add_argument("--scenari", default="prod8,base6,ripresa6,gara6,gara6_senza_copia,adam6_gara,"
                                         "adam6_gara_copia_parziale")
    ap.add_argument("--figlio", action="store_true")
    ap.add_argument("--cartella", type=Path)
    ap.add_argument("--scenario")
    ap.add_argument("--uscita", type=Path)
    args = ap.parse_args()
    if args.figlio:
        return figlio(args.cartella, args.scenario, args.uscita)
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "" or os.environ.get("HIP_VISIBLE_DEVICES", None) != "":
        raise SystemExit("lanciare con CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= (la GPU e' della forgia)")
    py = sys.executable
    t0 = time.time()
    cartelle = prepara_copie(args.lavoro, py)
    scenari = [s for s in args.scenari.split(",") if s]
    risultati: dict = {}
    for scenario in scenari:
        print(f"\n══ scenario {scenario} ══")
        rif_scen = SCENARI_CONTROPROVA.get(scenario, scenario)
        if ("base", rif_scen) not in risultati:
            r0 = corri(py, cartelle["base"], rif_scen, args.lavoro / "corse" / "base" / rif_scen)
            if r0 is None:
                print(f"   ⚠️ scenario {rif_scen}: la copia NON patchata non lo esegue (sorgente di oggi): NON eseguito, "
                      "nessun controllo contato")
                continue
            risultati[("base", rif_scen)] = r0
        for nome, d in cartelle.items():
            if nome == "base" and scenario == rif_scen:
                continue
            if scenario in SCENARI_CONTROPROVA and ("prepara_asincrona" not in VARIANTI[nome] or nome == "base"):
                continue
            risultati[(nome, scenario)] = corri(py, d, scenario, args.lavoro / "corse" / nome / scenario)
            print(f"   {nome:15s} {scenario:26s} {risultati[(nome, scenario)]['_secondi_processo']:6.1f} s", flush=True)
        rif = risultati[("base", rif_scen)]
        d_rif = args.lavoro / "corse" / "base" / rif_scen
        for nome in cartelle:
            if nome == "base" or (nome, scenario) not in risultati:
                continue
            var, d_var = risultati[(nome, scenario)], args.lavoro / "corse" / nome / scenario
            if scenario in SCENARI_CONTROPROVA:
                uguali, det = _uguali_snap(d_rif / "ripresa_snap", d_var / "ripresa_snap")
                ok(f"{nome}/{scenario}: controprova — il punto di ripresa scritto nel thread DIFFERISCE da quello "
                   "della copia non patchata (il collaudo vede la gara)", not uguali, det)
                continue
            confronta(nome, scenario, rif, var, d_rif, d_var)
            attivita(nome, scenario, rif, var, VARIANTI[nome])
        if scenario == "ripresa6":
            for nome in cartelle:
                d_var = args.lavoro / "corse" / nome / scenario
                for etichetta in ("int_principale", "int_scrittura"):
                    uguali, det = _uguali_codici(d_rif / "intera" / "out", d_var / etichetta / "out")
                    comm = risultati[(nome, scenario)].get("commessi_dopo_interruzione", {}).get(etichetta)
                    ok(f"{nome}/ripresa6/{etichetta}: interrotta ({comm} strati commessi) e ripresa = corsa intera "
                       "della copia NON patchata", uguali and not risultati[(nome, scenario)].get("interruzioni_mancate"),
                       det)
    if args.suite:
        suite(py, cartelle["tutte"], args.lavoro)
    print("\n══ tempi delle corse giocattolo (CPU, informativi: NON sono tempi della forgia) ══")
    for (nome, scenario), r in sorted(risultati.items()):
        print(f"   {nome:15s} {scenario:26s} corsa {r.get('secondi', '-')} s · attenzione {r.get('attenzione_classe')} "
              f"· teacher {sum(r.get('teacher_chiamate', {}).values()) if 'teacher_chiamate' in r else '-'} "
              f"· blocchi MoE {r.get('moe_blocchi', '-')}")
    falliti = [n for n, o in ESITI if not o]
    print(f"\n{'✅ TUTTI' if not falliti else '⛔ FALLITI ' + str(len(falliti)) + ' su'} {len(ESITI)} controlli in "
          f"{time.time() - t0:.0f}s" + ("" if not falliti else ": " + " | ".join(falliti)))
    return 0 if not falliti else 1


if __name__ == "__main__":
    raise SystemExit(main())
