#!/usr/bin/env python3
"""Radiografia PER BLOCCO del passo VERO della forgia (ricetta S4) con torch.profiler — 25/9 mattina.

Diverso da `strumenti/radiografia_blocchi.py` (00:40, misure dei blocchi ISOLATI con eventi GPU,
fuori dal passo): qui gira la forgia vera — `forgia_globale_40_strati.esegui` sui primi 4 strati,
cioe' le due prime finestre della ricetta S4 separata, [0,1] (lineare+lineare) e [2,3] (con
attenzione piena allo strato 3) — e il tempo GPU di ogni passo si attribuisce a blocchi:

  attenzione lineare (fla) · attenzione piena · norme dello strato · esperto condiviso · MoE (somma)
  · router / instradamento congelato · esperti: grouped mm · esperti: attivazione SwiGLU
  · esperti: smistamento e ricombinazione · rotazione (FWHT) · nucleo ternario · perdita (Lemma 1,
  resto) · backward (con la ripartizione per blocco d'origine, via sequence_nr) · ottimizzatore
  AdamW · ottimizzatore zero_grad · norma gradienti e controlli · sincronizzazioni (.item/bool)
  · fuori dai blocchi.

Come: ogni kernel GPU e' attribuito all'operazione CPU che l'ha lanciato (self_device_time_total),
e quella al blocco dell'etichetta (record_function) piu' vicina fra i suoi antenati. Le etichette
le mettono ganci installati in QUESTO processo (nessun file di thor_v2 toccato): ganci globali dei
moduli (sottomoduli dello strato, riconosciuti al primo passaggio), involucri di funzioni cercate
per nome a ogni chiamata (`_grouped_linear`, `_ruota_se_possibile`, `forward_duro`,
`errore_lemma1_finestra`, `passo_ottimizzazione`, `Tensor.backward/item/__bool__`); l'ottimizzatore
e' gia' etichettato da torch. Gli involucri lavorano SOLO mentre il profiler raccoglie: i passi
cronometrati senza profiler non pagano niente.

Per finestra (default --passi 48): passi 0-4 riscaldamento; 5-14 cronometrati SENZA profiler
(il passo vero, per la colonna «% del passo»); 15-16 riscaldamento del profiler; 17-46 (30 passi)
profilati. I codici NON vengono scritti (scrivi_codici_su_disco sostituita, nel solo processo),
nessun export. Uscita: tabella ms/passo e % per blocco (stampata e in --tabella .md) + --json.

Uso VERO: SOLO col lanciatore, a GPU libera — strumenti/radiografia_blocchi_passo.sh (guardia GPU, ambiente
della ricetta, PYTHONPATH di fla). Lanciato col solo python il passo NON sarebbe quello della ricetta (senza
einops fla non si carica e transformers ripiega su torch; senza le THOR_* rotazione d'ingresso, nucleo bf16 ed
esperti_thor sono spenti): dal 25/9 il python lo verifica e rifiuta (rc 4). Controllo senza GPU:
    <ambiente del lanciatore> python3 strumenti/radiografia_blocchi_passo.py --solo-verifica-ricetta
Collaudo su CPU (strati Qwen3_5MoeDecoderLayer veri a scala giocattolo, lineare+piena):
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
      @@ROOT:venv@@/bin/python3 strumenti/radiografia_blocchi_passo.py --giocattolo --json J --tabella T
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
THOR_V2 = QUI.parent
sys.path.insert(0, str(THOR_V2))

B_LINEARE = "attenzione lineare (fla)"
B_PIENA = "attenzione piena"
B_NORME = "norme dello strato"
B_CONDIVISO = "esperto condiviso"
B_MOE = "MoE: somma condiviso + instradati"
B_STRATO = "strato: resto (connessioni residue)"
B_ROUTER = "router / instradamento congelato"
B_GMM = "esperti: grouped mm (gate_up, down)"
B_ATT = "esperti: attivazione SwiGLU"
B_SMISTA = "esperti: smistamento e ricombinazione"
B_ROT = "rotazione (FWHT)"
B_NUCLEO = "nucleo ternario (pesi da z/s/delta)"
B_PERDITA = "perdita (Lemma 1, resto)"
B_BACKWARD = "backward"
B_ADAM = "ottimizzatore AdamW"
B_ZERO = "ottimizzatore zero_grad"
B_NORMA = "norma gradienti e controlli (resto del passo)"
B_SINCRO = "sincronizzazioni (.item / bool)"
B_FUORI = "fuori dai blocchi (ciclo del passo)"
ORDINE = [B_LINEARE, B_PIENA, B_NORME, B_CONDIVISO, B_MOE, B_STRATO, B_ROUTER, B_GMM, B_ATT, B_SMISTA, B_ROT, B_NUCLEO,
          B_PERDITA, B_BACKWARD, B_ADAM, B_ZERO, B_NORMA, B_SINCRO, B_FUORI]
PREFISSO = "blocco:"


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [radiografia]", *a, flush=True)


# ═══════════════════════════ etichette ═══════════════════════════
class Etichette:
    """Etichette record_function, attive solo quando `self.attive` (il profiler raccoglie)."""

    def __init__(self):
        self.attive = False
        self._pila = threading.local()
        self._maniglie = []
        self.M = None

    # -- involucro di funzione o metodo
    def avvolgi(self, fn, blocco: str):
        from torch.autograd.profiler import record_function
        etichetta = PREFISSO + blocco

        @functools.wraps(fn)
        def involucro(*a, **k):
            if not self.attive:
                return fn(*a, **k)
            with record_function(etichetta):
                return fn(*a, **k)
        involucro._radiografia_originale = fn
        return involucro

    def sostituisci(self, oggetto, nome: str, blocco: str) -> None:
        setattr(oggetto, nome, self.avvolgi(getattr(oggetto, nome), blocco))

    # -- ganci globali dei moduli: etichetta ai sottomoduli dello strato, riconosciuti al primo passaggio
    def _etichetta_strato(self, strato) -> None:
        coppie = [("input_layernorm", B_NORME), ("post_attention_layernorm", B_NORME), ("linear_attn", B_LINEARE),
                  ("self_attn", B_PIENA), ("mlp", B_MOE), ("mlp.shared_expert", B_CONDIVISO),
                  ("mlp.shared_expert_gate", B_CONDIVISO), ("mlp.gate", B_ROUTER), ("mlp.experts", B_SMISTA)]
        moduli = dict(strato.named_modules())
        strato._radiografia_blocco = B_STRATO
        for nome, blocco in coppie:
            if nome in moduli:
                moduli[nome]._radiografia_blocco = blocco
        strato._radiografia_etichettato = True

    def _pre(self, modulo, _ingressi):
        if not self.attive:
            return
        if isinstance(modulo, self.M.Qwen3_5MoeDecoderLayer) and not getattr(modulo, "_radiografia_etichettato", False):
            self._etichetta_strato(modulo)
        blocco = getattr(modulo, "_radiografia_blocco", None)
        if blocco is None:
            return
        from torch.autograd.profiler import record_function
        rf = record_function(PREFISSO + blocco)
        rf.__enter__()
        pila = getattr(self._pila, "v", None)
        if pila is None:
            pila = self._pila.v = []
        pila.append((id(modulo), rf))

    def _post(self, modulo, _ingressi, _uscita):
        pila = getattr(self._pila, "v", None)
        if pila and pila[-1][0] == id(modulo):
            pila.pop()[1].__exit__(None, None, None)

    def accendi(self) -> None:
        import torch.nn.modules.module as mm
        self._maniglie = [mm.register_module_forward_pre_hook(self._pre), mm.register_module_forward_hook(self._post)]
        self.attive = True

    def spegni(self) -> None:
        self.attive = False
        for h in self._maniglie:
            h.remove()
        self._maniglie = []
        pila = getattr(self._pila, "v", None)
        while pila:
            pila.pop()[1].__exit__(None, None, None)

    def installa(self, V, F, FG, Q, M, moe) -> None:
        import torch
        self.M = M
        self.sostituisci(moe, "_grouped_linear", B_GMM)                # esperti_thor lo cerca a ogni chiamata
        self.sostituisci(M.Qwen3_5MoeExperts, "_apply_gate", B_ATT)
        self.sostituisci(V, "_ruota_se_possibile", B_ROT)              # dal gancio d'ingresso ruotato
        self.sostituisci(V, "_gate_up_per_forward", B_NUCLEO)
        self.sostituisci(Q.TernarioDiffBatchFuso, "forward_duro", B_NUCLEO)
        self.sostituisci(Q.TernarioDiffBatchFuso, "forward", B_NUCLEO)
        self.sostituisci(FG, "errore_lemma1_finestra", B_PERDITA)      # passo_ottimizzazione lo cerca per nome
        self.sostituisci(torch.Tensor, "backward", B_BACKWARD)
        self.sostituisci(torch.Tensor, "item", B_SINCRO)
        self.sostituisci(torch.Tensor, "__bool__", B_SINCRO)


# ═══════════════════════════ regia del profiler, per finestra ═══════════════════════════
class Regista:
    def __init__(self, etichette: Etichette, n_passi: int, riscaldo: int, cronometrati: int, prof_riscaldo: int,
                 profilati: int, su_gpu: bool):
        assert n_passi >= riscaldo + cronometrati + prof_riscaldo + profilati + 1, "passi insufficienti per lo schema"
        self.et = etichette
        self.n_passi, self.riscaldo, self.cronometrati = n_passi, riscaldo, cronometrati
        self.prof_riscaldo, self.profilati, self.su_gpu = prof_riscaldo, profilati, su_gpu
        self.finestre: list = []
        self._corrente = None

    def avvolgi_finestra(self, fn):
        @functools.wraps(fn)
        def involucro(strati_correnti, *a, **k):
            self._corrente = {"strati": sorted(strati_correnti), "ingressi": [], "prof": None, "eventi": None}
            try:
                return fn(strati_correnti, *a, **k)
            finally:
                self._chiudi()
        return involucro

    def avvolgi_passo(self, fn):
        from torch.autograd.profiler import record_function

        @functools.wraps(fn)
        def involucro(*a, **k):
            c = self._corrente
            if c is None:
                return fn(*a, **k)
            i = len(c["ingressi"])
            c["ingressi"].append(time.perf_counter())
            inizio = self.riscaldo + self.cronometrati
            if i == inizio:
                self._accendi(c)
            elif c["prof"] is not None and i > inizio:
                c["prof"].step()
                if c["eventi"] is not None and self.et.attive:
                    self.et.spegni()            # finiti i passi attivi: niente etichette dopo
            if not self.et.attive:
                return fn(*a, **k)
            with record_function(PREFISSO + B_NORMA):
                return fn(*a, **k)
        return involucro

    def _accendi(self, c) -> None:
        import torch
        from torch.profiler import ProfilerActivity, profile, schedule
        att = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.su_gpu else [])

        def pronto(p):
            c["eventi"] = p.events()
        c["prof"] = profile(activities=att, schedule=schedule(wait=0, warmup=self.prof_riscaldo,
                                                              active=self.profilati, repeat=1),
                            on_trace_ready=pronto, record_shapes=False, with_stack=False, acc_events=False)
        if self.su_gpu:
            torch.cuda.synchronize()
        self.et.accendi()
        c["prof"].start()
        log(f"finestra {c['strati']}: profiler acceso al passo {len(c['ingressi']) - 1} "
            f"({self.prof_riscaldo} di riscaldamento + {self.profilati} profilati)")

    def _chiudi(self) -> None:
        c, self._corrente = self._corrente, None
        if c is None:
            return
        if self.et.attive:
            self.et.spegni()
        if c["prof"] is not None:
            try:
                c["prof"].stop()
            except Exception:  # noqa: BLE001 — gia' fermato dal ciclo
                pass
        ing = c["ingressi"]
        a, b = self.riscaldo, self.riscaldo + self.cronometrati
        intervalli = [ing[j + 1] - ing[j] for j in range(a, min(b, len(ing) - 1))]
        ris = {"strati": c["strati"], "passi_eseguiti": len(ing), "passi_cronometrati": len(intervalli),
               "passo_senza_profiler_s": {"mediana": statistics.median(intervalli) if intervalli else None,
                                          "media": statistics.fmean(intervalli) if intervalli else None,
                                          "min": min(intervalli) if intervalli else None,
                                          "max": max(intervalli) if intervalli else None}}
        if c["eventi"] is None:
            ris["errore"] = "il profiler non ha consegnato eventi (passi insufficienti?)"
        else:
            ris.update(analizza(c["eventi"], self.profilati, self.su_gpu))
        self.finestre.append(ris)
        log(f"finestra {c['strati']}: analisi pronta")


# ═══════════════════════════ analisi degli eventi ═══════════════════════════
def _blocco_di(e):
    """(blocco, evento evaluate_function del backward o None) dall'antenato piu' vicino."""
    p = e
    while p is not None:
        n = p.name
        if n.startswith("autograd::engine::evaluate_function"):
            return B_BACKWARD, p
        if n.startswith(PREFISSO):
            return n[len(PREFISSO):], None
        if n.startswith("Optimizer.step"):
            return B_ADAM, None
        if n.startswith("Optimizer.zero_grad"):
            return B_ZERO, None
        p = p.cpu_parent
    return B_FUORI, None


def _unione_us(intervalli: list) -> float:
    tot, fine = 0.0, None
    inizio_c = None
    for a, b in sorted(intervalli):
        if fine is None or a > fine:
            if fine is not None:
                tot += fine - inizio_c
            inizio_c, fine = a, b
        else:
            fine = max(fine, b)
    if fine is not None:
        tot += fine - inizio_c
    return tot


def analizza(eventi, n_passi: int, su_gpu: bool) -> dict:
    from torch.autograd import DeviceType
    per_blocco: dict = collections.defaultdict(float)
    bw_per_origine: dict = collections.defaultdict(float)
    kernel: dict = collections.defaultdict(collections.Counter)
    origine_seq: dict = {}
    cpu_ev = [e for e in eventi if e.device_type == DeviceType.CPU]
    info = [(e, *_blocco_di(e)) for e in cpu_ev]
    for e, blocco, _ in info:                                      # blocco d'origine dei nodi autograd
        if blocco != B_BACKWARD and e.sequence_nr is not None and e.sequence_nr >= 0:
            origine_seq.setdefault((e.thread, e.sequence_nr), blocco)
    n_bw_senza = 0
    for e, blocco, valuta in info:
        t = e.self_device_time_total if su_gpu else e.self_cpu_time_total
        if t <= 0:
            continue
        per_blocco[blocco] += t
        if blocco == B_BACKWARD:
            orig = origine_seq.get((valuta.fwd_thread, valuta.sequence_nr)) if valuta is not None else None
            if orig is None:
                n_bw_senza += 1
            bw_per_origine[orig or "non attribuito"] += t
        if su_gpu:
            for kk in e.kernels:
                kernel[blocco][kk.name[:90]] += kk.duration
        else:
            kernel[blocco][e.name[:90]] += t
    totale = sum(per_blocco.values())
    passi_prof = [e.cpu_time_total for e in cpu_ev if e.name.startswith("ProfilerStep")]
    ris = {"unita": "tempo GPU dei kernel" if su_gpu else "tempo CPU (self) delle operazioni",
           "passi_profilati": n_passi,
           "blocchi_ms_per_passo": {b: per_blocco[b] / n_passi / 1e3 for b in ORDINE if b in per_blocco},
           "altri_blocchi_ms_per_passo": {b: v / n_passi / 1e3 for b, v in per_blocco.items() if b not in ORDINE},
           "backward_per_origine_ms_per_passo": {b: v / n_passi / 1e3 for b, v in
                                                 sorted(bw_per_origine.items(), key=lambda kv: -kv[1])},
           "backward_operazioni_senza_origine": n_bw_senza,
           "totale_attribuito_ms_per_passo": totale / n_passi / 1e3,
           "passo_sotto_profiler_ms": (statistics.fmean(passi_prof) / 1e3) if passi_prof else None,
           "kernel_principali": {b: [[n, v / n_passi / 1e3] for n, v in c.most_common(3)] for b, c in kernel.items()}}
    if su_gpu:
        dev = [(e.time_range.start, e.time_range.end) for e in eventi
               if e.device_type != DeviceType.CPU and not getattr(e, "is_user_annotation", False)]
        ris["gpu_occupata_unione_ms_per_passo"] = _unione_us(dev) / n_passi / 1e3
        # copertura: quanto del tempo dei kernel e' stato ricondotto a un'operazione CPU (e quindi a un blocco)
        tot_dev = sum(b - a for a, b in dev)
        ris["kernel_totale_ms_per_passo"] = tot_dev / n_passi / 1e3
        ris["copertura_attribuzione"] = (totale / tot_dev) if tot_dev else None
    return ris


def controlli_giocattolo(finestre: list) -> bool:
    """Il collaudo su CPU: due finestre, ogni blocco richiesto presente, la piena SOLO in [2,3]."""
    esiti = []

    def ok(nome, cond):
        esiti.append(bool(cond))
        print(f"  {'✅' if cond else '⛔'} {nome}")
    ok("due finestre radiografate, nessun errore", len(finestre) == 2 and all("errore" not in f for f in finestre))
    if len(finestre) != 2 or any("errore" in f for f in finestre):
        return False
    richiesti = [B_LINEARE, B_ROUTER, B_GMM, B_ATT, B_SMISTA, B_ROT, B_NUCLEO, B_PERDITA, B_BACKWARD, B_ADAM,
                 B_ZERO, B_NORMA, B_SINCRO, B_NORME, B_CONDIVISO]
    for f in finestre:
        presenti = f["blocchi_ms_per_passo"]
        mancano = [b for b in richiesti if b not in presenti]
        ok(f"finestra {f['strati']}: tutti i {len(richiesti)} blocchi richiesti presenti" + (f" (mancano {mancano})" if mancano else ""),
           not mancano)
        ok(f"finestra {f['strati']}: nessun blocco fuori dall'elenco ({list(f['altri_blocchi_ms_per_passo'])})",
           not f["altri_blocchi_ms_per_passo"])
        ok(f"finestra {f['strati']}: backward ripartito per origine (almeno 5 blocchi d'origine)",
           len([b for b in f["backward_per_origine_ms_per_passo"] if b != "non attribuito"]) >= 5)
        ok(f"finestra {f['strati']}: passo senza profiler cronometrato ({f['passi_cronometrati']} intervalli)",
           f["passi_cronometrati"] >= 2 and f["passo_senza_profiler_s"]["mediana"] > 0)
    ok("attenzione piena SOLO nella finestra [2,3]", B_PIENA not in finestre[0]["blocchi_ms_per_passo"]
       and B_PIENA in finestre[1]["blocchi_ms_per_passo"])
    return all(esiti)


def tabella(finestre: list, su_gpu: bool) -> str:
    righe = []
    for f in finestre:
        tipo = "con attenzione piena" if any(L % 4 == 3 for L in f["strati"]) else "lineare+lineare"
        passo = f["passo_senza_profiler_s"]["mediana"]
        righe.append(f"\n### finestra {f['strati']} ({tipo})\n")
        if "errore" in f:
            righe.append(f"⛔ {f['errore']}\n")
            continue
        righe.append(f"passo senza profiler (mediana di {f['passi_cronometrati']}): **{passo * 1e3:.1f} ms** · "
                     f"sotto profiler {f['passo_sotto_profiler_ms'] or float('nan'):.1f} ms · unita': {f['unita']}"
                     f" · {f['passi_profilati']} passi profilati\n")
        tot = f["totale_attribuito_ms_per_passo"]
        righe.append("| blocco | ms/passo | % del totale attribuito | % del passo (orologio) | kernel principali |")
        righe.append("|---|---:|---:|---:|---|")
        voci = list(f["blocchi_ms_per_passo"].items()) + list(f["altri_blocchi_ms_per_passo"].items())
        for b, v in sorted(voci, key=lambda kv: -kv[1]):
            kp = "; ".join(f"{n} {ms:.1f}" for n, ms in f["kernel_principali"].get(b, []))
            righe.append(f"| {b} | {v:.2f} | {100 * v / tot:.1f}% | {100 * v / (passo * 1e3):.1f}% | {kp[:160]} |")
        righe.append(f"| **totale attribuito** | **{tot:.2f}** | 100% | {100 * tot / (passo * 1e3):.1f}% | |")
        if su_gpu:
            righe.append(f"| kernel totali (somma delle durate) | {f['kernel_totale_ms_per_passo']:.2f} | "
                         f"copertura {100 * (f['copertura_attribuzione'] or 0):.1f}% | | |")
            occ = f["gpu_occupata_unione_ms_per_passo"]
            righe.append(f"| GPU occupata (unione degli intervalli) | {occ:.2f} | | {100 * occ / (passo * 1e3):.1f}% | |")
            righe.append(f"| GPU inattiva (passo − unione): attese, lanci, sincronizzazioni | "
                         f"{passo * 1e3 - occ:.2f} | | {100 * (passo * 1e3 - occ) / (passo * 1e3):.1f}% | |")
        righe.append("\nbackward per blocco d'origine (sequence_nr del nodo autograd):\n")
        righe.append("| origine | ms/passo |")
        righe.append("|---|---:|")
        for b, v in f["backward_per_origine_ms_per_passo"].items():
            righe.append(f"| {b} | {v:.2f} |")
        righe.append(f"\n(operazioni del backward senza origine ritrovata: {f['backward_operazioni_senza_origine']})")
    return "\n".join(righe)


# ═══════════════════════════ la ricetta: si pretende, non si presume ═══════════════════════════
# ⛔ 25/9 (verifica avversaria): lanciato col solo python (come diceva «Uso VERO» qui sopra) il passo
# misurato NON era quello della ricetta: senza PYTHONPATH=strumenti/pylib_fla (einops) l'import di fla
# fallisce in silenzio e transformers ripiega sull'attenzione lineare in torch; senza le THOR_* valida
# parte con rotazione d'ingresso, nucleo bf16 ed esperti_thor SPENTI. Ora, a GPU, prima di misurare:
# le THOR_* della ricetta (lette dal suo comune(), non copiate qui), PYTHONPATH, HSA_OVERRIDE assente,
# cartella di lavoro, e che l'attenzione lineare sia DAVVERO fla. Una differenza = nessuna misura.
RICETTA = THOR_V2 / "ricetta_base_nuova_6h.sh"
PER_CORSA = {"THOR_FORGIA_N_STRATI", "THOR_VALIDA_N_PASSI", "THOR_CRONO_OGNI", "THOR_ID_CORSA", "THOR_VALIDA_OUT",
             "THOR_VALIDA_GGUF_OUT", "THOR_CALIB_CORPUS", "THOR_CALIB_MISCELA"}   # non cambiano il costo del passo


def variabili_ricetta(percorso: Path) -> "tuple[dict, str | None]":
    """({THOR_X: valore} del comune() della ricetta, con $VAR espanse dalle assegnazioni dello script;
    PYTHONPATH esportato dallo script)."""
    import re
    testo = Path(percorso).read_text()
    var = {k: v.strip('"') for k, v in re.findall(r"^([A-Za-z_]\w*)=(\S+)", testo, re.M)}

    def espandi(v: str) -> str:
        return re.sub(r"\$\{?(\w+)\}?", lambda m: var.get(m.group(1), m.group(0)), v)
    i = testo.index("comune() {")
    blocco = testo[i:testo.index("\n}", i)]
    thor = {k: espandi(a if a else b) for k, a, b in re.findall(r"\b(THOR_[A-Z0-9_]+)=(?:\"([^\"]*)\"|(\S*))", blocco)}
    pp = re.search(r"^export PYTHONPATH=(\S+)", testo, re.M)
    return thor, (espandi(pp.group(1)) if pp else None)


def differenze_dalla_ricetta(ricetta: Path = RICETTA) -> "list[str]":
    attese, pp = variabili_ricetta(ricetta)
    diff = [f"{k}: ricetta {v!r}, qui {os.environ.get(k)!r}" for k, v in sorted(attese.items())
            if k not in PER_CORSA and os.environ.get(k) != v]
    if pp is not None and os.environ.get("PYTHONPATH") != pp:
        diff.append(f"PYTHONPATH: ricetta {pp!r}, qui {os.environ.get('PYTHONPATH')!r}")
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        diff.append("HSA_OVERRIDE_GFX_VERSION presente (la ricetta la toglie)")
    if Path.cwd().resolve() != THOR_V2.resolve():
        diff.append(f"cartella di lavoro {Path.cwd()} (la ricetta lavora in {THOR_V2})")
    return diff


def attenzione_lineare_fla(M) -> "str | None":
    """Il modulo dell'implementazione che transformers ha scelto per chunk_gated_delta_rule (None = ignota)."""
    for c in (M.torch_chunk_gated_delta_rule.__closure__ or ()):
        f = c.cell_contents
        if callable(f) and getattr(f, "__name__", "") == "chunk_gated_delta_rule":
            return getattr(f, "__module__", None)
    return None


# ═══════════════════════════ i due modi ═══════════════════════════
def _moduli():
    import finestra_globale as FG
    import forgia_globale_40_strati as F
    import quant_ternaria_globale_batch_v2 as Q
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    import valida_finestra_8_12 as V
    from transformers.integrations import moe
    return V, F, FG, Q, M, moe


def prepara(regista: Regista, etichette: Etichette):
    V, F, FG, Q, M, moe = _moduli()
    etichette.installa(V, F, FG, Q, M, moe)
    F.ottimizza_finestra_JOINT = regista.avvolgi_finestra(F.ottimizza_finestra_JOINT)   # forgia lo cerca per nome
    V.passo_ottimizzazione = regista.avvolgi_passo(V.passo_ottimizzazione)            # ottimizza lo cerca per nome
    F.scrivi_codici_su_disco = lambda risultato: log(f"codici degli strati {sorted(risultato)} NON scritti (radiografia)")
    return V, F


def giocattolo(F, V) -> None:
    """Strati veri a scala giocattolo, lineari e pieni come Thor (piena a 3), esperti Thor se chiesti."""
    import torch
    sys.path.insert(0, str(THOR_V2))
    import test_forgia_scorrevole as TFS

    class GiocattoloMisto(TFS.Giocattolo):
        def __init__(self, n, dtype):
            super().__init__(n, dtype)
            c = self.cfg
            c.layer_types = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(n)]
            c.linear_conv_kernel_dim, c.linear_key_head_dim, c.linear_value_head_dim = 2, 4, 4
            c.linear_num_key_heads, c.linear_num_value_heads = 2, 2
            # torch._grouped_mm su CPU vuole passi multipli di 16 byte: intermedio 16 (bf16: 32 byte)
            c.moe_intermediate_size = c.shared_expert_intermediate_size = 16
            c._experts_implementation = "grouped_mm"
            if os.getenv("THOR_ESPERTI_THOR", "0") == "1":
                import esperti_thor
                esperti_thor.registra()
                c._experts_implementation = esperti_thor.NOME

    g = GiocattoloMisto(4, torch.bfloat16)
    F.carica_finestra_corrente = lambda layers: g.carica(layers)
    F.carica_batch_calibrazione = g.batch
    F.seme_embedding_quantizzato = g.seme0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--giocattolo", action="store_true", help="collaudo su CPU, strati giocattolo")
    ap.add_argument("--passi", type=int, default=48)
    ap.add_argument("--riscaldo", type=int, default=5)
    ap.add_argument("--cronometrati", type=int, default=10)
    ap.add_argument("--prof-riscaldo", type=int, default=2)
    ap.add_argument("--profilati", type=int, default=30)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--tabella", type=Path)
    ap.add_argument("--ricetta", type=Path, default=RICETTA, help="lo script da cui leggere le THOR_* della ricetta")
    ap.add_argument("--solo-verifica-ricetta", action="store_true",
                    help="controlla ambiente e fla contro la ricetta e termina (niente GPU): rc 0 = misurerebbe la ricetta")
    args = ap.parse_args()
    if not args.solo_verifica_ricetta and (args.json is None or args.tabella is None):
        ap.error("servono --json e --tabella")
    if not args.giocattolo:
        diff = differenze_dalla_ricetta(args.ricetta)          # puro: nessun import di torch
        if diff:
            print(f"⛔ ambiente diverso dalla ricetta {args.ricetta.name}: il passo misurato non sarebbe il suo "
                  "(lancia strumenti/radiografia_blocchi_passo.sh):\n  " + "\n  ".join(diff))
            return 4
        if args.solo_verifica_ricetta:                        # nessuna misura: la GPU non si tocca nemmeno per sbaglio
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"] = ""
        else:                                                 # la guardia GPU PRIMA di importare torch/transformers
            r = subprocess.run([str(QUI / "gpu_occupata.sh")], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"⛔ GPU occupata, non misuro (una corsia pesante per volta):\n{r.stdout}")
                return 3
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _M
        impl = attenzione_lineare_fla(_M)
        if not (impl or "").startswith("fla."):
            print(f"⛔ l'attenzione lineare NON e' fla ({impl or 'ripiego torch'}): PYTHONPATH senza einops? "
                  "Misurerei un altro passo.")
            return 4
        print(f"✅ ambiente = ricetta {args.ricetta.name} (tranne {sorted(PER_CORSA)}), attenzione lineare {impl}")
        if args.solo_verifica_ricetta:
            return 0
    elif args.solo_verifica_ricetta:
        ap.error("--solo-verifica-ricetta non ha senso con --giocattolo")
    import torch
    su_gpu = not args.giocattolo
    if su_gpu and not torch.cuda.is_available():
        raise SystemExit("serve la GPU (o --giocattolo)")
    if args.giocattolo and torch.cuda.is_available():
        raise SystemExit("--giocattolo va lanciato con CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=")
    et = Etichette()
    reg = Regista(et, args.passi, args.riscaldo, args.cronometrati, args.prof_riscaldo, args.profilati, su_gpu)
    V, F = prepara(reg, et)
    if args.giocattolo:
        giocattolo(F, V)
    t0 = time.time()
    log(f"forgia_globale.esegui su 4 strati (finestre [0,1] e [2,3]), {args.passi} passi per finestra")
    F.esegui(4, 2, False, args.passi, True)
    ris = {"quando": time.strftime("%F %T"), "giocattolo": args.giocattolo, "secondi": round(time.time() - t0, 1),
           "schema_passi": {"riscaldo": args.riscaldo, "cronometrati": args.cronometrati,
                            "prof_riscaldo": args.prof_riscaldo, "profilati": args.profilati, "totale": args.passi},
           "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("THOR_")}, "finestre": reg.finestre}
    args.json.write_text(json.dumps(ris, indent=1, ensure_ascii=False))
    testo = (f"# Radiografia per blocco del passo — {'GIOCATTOLO (CPU)' if args.giocattolo else 'ricetta S4 (GPU)'}"
             f" — {ris['quando']}\n" + tabella(reg.finestre, su_gpu))
    args.tabella.write_text(testo + "\n")
    print(testo)
    ok = len(reg.finestre) == 2 and all("errore" not in f for f in reg.finestre)
    if args.giocattolo:
        print("\ncontrolli del collaudo su CPU:")
        ok = controlli_giocattolo(reg.finestre) and ok
    print(f"\n{'✅' if ok else '⛔'} radiografia {'completa' if ok else 'INCOMPLETA'}: {args.json} · {args.tabella}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
