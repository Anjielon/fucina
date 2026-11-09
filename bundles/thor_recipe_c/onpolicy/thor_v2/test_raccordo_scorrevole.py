#!/usr/bin/env python3
"""Collaudo del RACCORDO A<->B della forgia scorrevole (verificatore, 24/9 notte).

CPU, scala giocattolo, su una COPIA della cartella con ENTRAMBE le patch applicate
(la catena di stanotte importa gli originali: non si toccano):

    cp thor_v2/*.py thor_v2/*.json thor_v2/*.npz thor_v2/*.sh COPIA/ ; cp thor_v2/strumenti/*.py COPIA/strumenti/
    python3 COPIA/strumenti/patch_ripartenza.py
    python3 COPIA/strumenti/patch_raccordo_scorrevole.py
    cd COPIA && env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 test_raccordo_scorrevole.py [--ricetta base|produzione] [--senza-patch-A DIR]

Riusa l'infrastruttura di test_forgia_scorrevole.py (Giocattolo, corsa, controlli_catena,
t2/t4/t6/t7), ma con l'interfaccia VERA: ottimizza_finestra_JOINT patchata dal
compito A, attraverso raccordo_scorrevole. Prove in piu', indipendenti:
  R1 interfaccia: patch A e raccordo visti da forgia_scorrevole (anche --verifica-interfaccia
     in un processo a parte); senza patch A la forgia RIFIUTA.
  R2 ripartenza verificata DA FUORI (spia su applica_ripartenza_): i codici con cui lo strato
     entra nella finestra = quelli con cui e' uscito dalla precedente (via parametri) o quelli
     della corsa sorgente (via codici), letti con _estrai_famiglia, al bit.
  R3 memoria: all'ingresso della finestra nessun oggetto ternario pieno oltre agli stati da
     applicare (uno per strato portato); dopo ogni commit solo gli oggetti portati.
  R4 guasti: finestra fallita PRIMA della copia -> oggetto portato ripristinato identico;
     DOPO la copia -> oggetto svuotato e marcato, il riuso solleva.
  --ricetta produzione: variabili della forgia veloce (solo fase dura, rotazione
     sull'ingresso, nucleo bf16, routing aggiornato, diagnostica spenta, lr 0,01) e strati
     bf16: catena completa + modalita' (b).
  R8 (25/9) THOR_SCORREVOLE_ADAM=conserva — `--adam` per girare solo questi (+R1):
     R8a momenti/step del portato al passo 0 della finestra k+1 = fine della finestra k, al
         bit; entrante con Adam nuovo; misura del primo passo (scossa) nei due modi.
     R8b equivalenza: 2 chiamate da N passi con Adam conservato = 1 da 2N (loss, parametri,
         momenti, codici al bit) dove l'obiettivo e' lo stesso; riparti diverge.
     R8c ripresa dopo interruzione (5 punti) con Adam conservato: momenti ricevuti e codici
         identici alla corsa intera; punto senza momenti / forme sbagliate / modo cambiato RIFIUTATI.
     R8d errori rumorosi sulla chiamata diretta, oggetti intatti; guasti prima/dopo la copia.
     R8e catena completa con Adam conservato (stessi controlli di R2/R3 + memoria dei momenti).
     R8f peso su disco del punto di ripresa (misurato, scalato a Thor).
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RICETTE = {
    "base": {},
    "produzione": {
        "THOR_VALIDA_FRAZIONE_DURA": "1.0", "THOR_ROTAZIONE_INGRESSO": "1", "THOR_NUCLEO_BF16": "1",
        "THOR_ROUTING_AGGIORNATO": "1", "THOR_ROUTING_VIVO": "0", "THOR_DIAGNOSTICA_PASSO": "0",
        "THOR_HELDOUT_OGNI": "0", "THOR_RICONGELA_OGNI": "0", "THOR_FORGIA_DOWN_TERNARIO": "1",
        "THOR_LR_Z": "0.01", "THOR_LR_LOGS": "0.01", "THOR_LR_DELTA": "0.01", "THOR_VALIDA_LR": "0.01",
        "THOR_ESPERTI_THOR": "1",
    },
}
ESITI: list = []


def ok(nome: str, cond, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(cond)))
    print(f"    {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(cond)


class _Tee(io.TextIOBase):
    def __init__(self, *f):
        self.f = f

    def write(self, s):
        for x in self.f:
            x.write(s)
        return len(s)

    def flush(self):
        for x in self.f:
            x.flush()


def _conta_ternari_pieni(TDBF) -> int:
    gc.collect()
    return sum(1 for o in gc.get_objects() if isinstance(o, TDBF) and o.z.numel() > 0)


def _conta_stati_vivi(rc) -> int:
    gc.collect()
    return sum(1 for o in gc.get_objects() if isinstance(o, rc.StatoStrato) and o.parametri is not None)


class Spie:
    """R2 + R3: spia su applica_ripartenza_ (codici all'ingresso, letti da fuori) e su
    V.ottimizza_finestra_JOINT (oggetti vivi all'ingresso della finestra)."""

    def __init__(self, V, rc, R, TDBF):
        self.V, self.rc, self.R, self.TDBF = V, rc, R, TDBF
        self.ingressi: list = []          # (L, via, identici?)
        self.memoria: list = []           # (pieni, stati, attesi)
        self._orig_app = rc.applica_ripartenza_
        self._orig_ott = V.ottimizza_finestra_JOINT

    def __enter__(self):
        V, rc, R = self.V, self.rc, self.R
        orig_app, orig_ott = self._orig_app, self._orig_ott

        def applica(L, voce, t_gu, t_dw, *a, **k):
            if isinstance(voce, rc.StatoStrato):
                tmp_gu = R._nuovo_da_stato(voce.parametri["gu"], voce.meta["gu"])
                atteso = {"gu": V._estrai_famiglia(tmp_gu)}
                if voce.parametri["dw"] is not None:
                    atteso["dw"] = V._estrai_famiglia(R._nuovo_da_stato(voce.parametri["dw"], voce.meta["dw"]))
                del tmp_gu
                via = "parametri"
            else:
                D, Q = rc.codici_gate_up(voce)
                atteso = {"gu": (D.numpy(), Q.numpy())}
                if rc.FAMIGLIA_DW in voce:
                    atteso["dw"] = tuple(voce[rc.FAMIGLIA_DW])
                via = "codici"
            esito = orig_app(L, voce, t_gu, t_dw, *a, **k)
            import numpy as np
            letti = {"gu": V._estrai_famiglia(t_gu)}
            if t_dw is not None:
                letti["dw"] = V._estrai_famiglia(t_dw)
            uguali = sorted(letti) == sorted(atteso) and all(
                np.array_equal(letti[f][0], atteso[f][0]) and np.array_equal(letti[f][1], atteso[f][1]) for f in letti)
            self.ingressi.append((L, via, uguali))
            return esito

        @functools.wraps(orig_ott)
        def ottimizza(strati_correnti, *a, **k):
            rip = k.get("ripartenza_per_strato") or {}
            n_stati = sum(1 for v in rip.values() if isinstance(v, rc.StatoStrato))
            self.memoria.append((_conta_ternari_pieni(self.TDBF), _conta_stati_vivi(rc), n_stati))
            return orig_ott(strati_correnti, *a, **k)

        rc.applica_ripartenza_ = applica
        V.ottimizza_finestra_JOINT = ottimizza
        return self

    def __exit__(self, *_):
        self.rc.applica_ripartenza_ = self._orig_app
        self.V.ottimizza_finestra_JOINT = self._orig_ott
        return False


def r1_interfaccia(FS, R, V, senza_patch_A: "str | None") -> None:
    print("\n── R1 interfaccia: patch A + raccordo visti da forgia_scorrevole ──")
    R.verifica_patch_A(V)
    fn = FS._funzione_ottimizza()
    ok("forgia_scorrevole prende A1 dal raccordo", fn is R.ottimizza_finestra_JOINT_B)
    try:
        FS.verifica_interfaccia_A(fn, serve_da_codici=True)
        ok("verifica_interfaccia_A (A1 + A2) passa", FS._funzione_da_codici() is R.ternari_da_codici)
    except RuntimeError as e:
        ok("verifica_interfaccia_A (A1 + A2) passa", False, str(e)[:120])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "THOR_SCORREVOLE_INIT": "teacher"}
    p = subprocess.run([sys.executable, "forgia_scorrevole.py", "--verifica-interfaccia"], env=env,
                       capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(FS.__file__)))
    ok("forgia_scorrevole.py --verifica-interfaccia (processo a parte): rc 0", p.returncode == 0,
       (p.stdout.strip().splitlines() or [""])[0][:100] if p.returncode == 0 else p.stderr[-160:])
    if senza_patch_A:
        p = subprocess.run([sys.executable, "forgia_scorrevole.py", "--verifica-interfaccia"], env=env,
                           capture_output=True, text=True, cwd=senza_patch_A)
        ok("SENZA patch A: --verifica-interfaccia RIFIUTA", p.returncode != 0 and "PATCH A ASSENTE" in p.stderr,
           p.stderr.strip().splitlines()[-1][:120] if p.stderr.strip() else "")


def _righe_ripartenza(testo: str) -> list:
    return [r for r in testo.splitlines() if re.search(r"ripartenza strato \d+ da (PARAMETRI|CODICI)", r)]


def _tutte_esatte(righe: list) -> bool:
    perc = [float(x) for r in righe for x in re.findall(r"codici identici ([\d.]+)%", r)]
    sc = [float(x) for r in righe for x in re.findall(r"scarto massimo pesi ([\d.e+-]+)", r)]
    return bool(perc) and all(p == 100.0 for p in perc) and bool(sc) and all(s == 0.0 for s in sc)


def r2_catena(TB, FS, R, V, rc, TDBF, base: Path, dtype, toll_q: float, etichetta: str):
    print(f"\n── R2/R3 catena scorrevole con l'interfaccia VERA (raccordo), 10 strati, {etichetta} ──")
    n = 10
    gioc = TB.Giocattolo(n, dtype)
    buf = io.StringIO()
    oss_mem: list = []
    cartella = base / f"r2_{etichetta}"

    class Oss(TB.Osservatore):
        def __call__(self, evento, **d):
            super().__call__(evento, **d)
            if evento == "commit" and d.get("strato") == d.get("finestra"):
                oss_mem.append((d["finestra"], _conta_ternari_pieni(TDBF), _conta_stati_vivi(rc)))

    oss = Oss()
    V.OUT_DIR, V.ID_CORSA = cartella / "out", "prova"
    with Spie(V, rc, R, TDBF) as spie, contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        FS.esegui(n, 2, 3, "teacher", "", "", salta_export=True, carica_strati=gioc.carica,
                  carica_batch=gioc.batch, seme_iniziale=gioc.seme0, ottimizza_fn=R.ottimizza_finestra_JOINT_B,
                  da_codici_fn=R.ternari_da_codici, ripresa_dir=cartella / "ripresa", osservatore=oss)
    TB.controlli_catena(f"R2-{etichetta}", cartella, gioc, oss, 2, "prova", toll_q=toll_q)
    righe = _righe_ripartenza(buf.getvalue())
    ok(f"R2-{etichetta}: una ripartenza per PARAMETRI per ogni finestra dopo la prima (8), tutte esatte",
       len(righe) == n - 2 and all("da PARAMETRI" in r for r in righe) and _tutte_esatte(righe), f"{len(righe)} righe")
    ok(f"R2-{etichetta}: spia — codici all'ingresso = codici d'uscita della finestra precedente, al bit",
       len(spie.ingressi) == n - 2 and all(u and via == "parametri" for _, via, u in spie.ingressi),
       f"{[(L, via, u) for L, via, u in spie.ingressi][:3]}...")
    ok(f"R3-{etichetta}: all'ingresso di ogni finestra zero oggetti ternari pieni e uno stato per strato portato",
       all(p == 0 and s == atteso for p, s, atteso in spie.memoria), f"(pieni, stati, attesi) {spie.memoria[:4]}...")
    per_strato = 2 if V.DOWN_TERNARIO else 1           # oggetti ternari per strato: gate_up (+ down)
    ok(f"R3-{etichetta}: dopo ogni commit restano solo gli oggetti dello strato portato "
       f"({per_strato}; 0 all'ultima finestra), zero stati",
       len(oss_mem) == n - 1 and all(p == (0 if k == n - 2 else per_strato) and s == 0 for k, p, s in oss_mem),
       f"(finestra, pieni, stati) {oss_mem[:2]} ... {oss_mem[-2:]}")
    return cartella, gioc


def r5_codici(TB, FS, R, V, rc, TDBF, base: Path, sorgente: Path, dtype, toll_q: float, etichetta: str):
    print(f"\n── R5 modalita' (b) con l'interfaccia VERA: entranti dai codici di una corsa precedente ({etichetta}) ──")
    import numpy as np
    n = 10
    gioc = TB.Giocattolo(n, dtype)
    buf = io.StringIO()
    oss = TB.Osservatore()
    cartella = base / f"r5_{etichetta}"
    V.OUT_DIR, V.ID_CORSA = cartella / "out", "prova_b"
    with Spie(V, rc, R, TDBF) as spie, contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        FS.esegui(n, 2, 3, "codici", str(sorgente / "out"), "prova", salta_export=True, carica_strati=gioc.carica,
                  carica_batch=gioc.batch, seme_iniziale=gioc.seme0, ottimizza_fn=R.ottimizza_finestra_JOINT_B,
                  da_codici_fn=R.ternari_da_codici, ripresa_dir=cartella / "ripresa", osservatore=oss)
    TB.controlli_catena(f"R5-{etichetta}", cartella, gioc, oss, 2, "prova_b", toll_q=toll_q)
    righe = _righe_ripartenza(buf.getvalue())
    cod = [r for r in righe if "da CODICI" in r]
    par = [r for r in righe if "da PARAMETRI" in r]
    ok(f"R5-{etichetta}: {n} ingressi dai CODICI (2 alla finestra 0, poi 1 per finestra) e {n-2} dai PARAMETRI, tutti esatti",
       len(cod) == n and len(par) == n - 2 and _tutte_esatte(righe), f"codici {len(cod)} · parametri {len(par)}")
    prec = {L: FS.leggi_codici_strato(sorgente / "out", L, "prova", True) for L in range(n)}
    ingressi_cod = [(L, u) for L, via, u in spie.ingressi if via == "codici"]
    ok(f"R5-{etichetta}: spia — ogni entrante parte ESATTAMENTE dai codici della corsa sorgente (letti nel modulo)",
       sorted(L for L, _ in ingressi_cod) == list(range(n)) and all(u for _, u in ingressi_cod))
    ok(f"R5-{etichetta}: gli oggetti del solo controllo (z saturo) sono liberati prima della finestra",
       all(p == 0 and s == atteso for p, s, atteso in spie.memoria), f"(pieni, stati, attesi) {spie.memoria[:3]}...")
    # i codici della corsa sorgente sono proprio quelli che la spia ha visto nel modulo: confronto diretto
    ok(f"R5-{etichetta}: i codici sorgente usati sono quelli su disco (Q identici)",
       all(np.array_equal(FS._codici_gate_up_fusi(prec[L])[1], oss.init_portati[k][L]["gu"][1])
           for k in oss.entranti for L in oss.entranti[k]))
    return cartella


def r4_guasti(TB, R, V, rc, TDBF) -> None:
    print("\n── R4 guasti dentro la finestra: prima e dopo la copia dello stato ──")
    import torch
    gioc = TB.Giocattolo(3)
    cfg, emb, pul, qua = gioc.carica([0, 1, 2])
    ids, n_tr = gioc.batch()
    usc: dict = {}
    R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (0, 1)}, ids, cfg, 2, strati_teacher={L: pul[L] for L in (0, 1)},
                                 emb=emb, ternari_iniziali=None, ternari_uscita=usc, n_batch_addestramento=n_tr)
    portato = usc[1]
    prima = {f: {n: getattr(o, n).detach().clone() for n in rc.NOMI_PARAMETRI} for f, o in portato.items()}
    orig = V.ottimizza_finestra_JOINT

    @functools.wraps(orig)
    def rotta_prima(*a, **k):
        raise KeyboardInterrupt("guasto simulato prima della copia")
    V.ottimizza_finestra_JOINT = rotta_prima
    try:
        R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (1, 2)}, ids, cfg, 2, strati_teacher={L: pul[L] for L in (1, 2)},
                                     emb=emb, ternari_iniziali={1: portato}, ternari_uscita={}, n_batch_addestramento=n_tr)
        ok("guasto prima della copia: propagato", False)
    except KeyboardInterrupt:
        ok("guasto prima della copia: propagato", True)
    finally:
        V.ottimizza_finestra_JOINT = orig
    ok("guasto prima della copia: oggetto portato RIPRISTINATO identico al bit",
       all(torch.equal(getattr(o, n), prima[f][n]) for f, o in portato.items() for n in rc.NOMI_PARAMETRI)
       and not portato["gu"]._svuotato_dal_raccordo)
    orig_passo = V.passo_ottimizzazione

    def passo_rotto(*a, **k):
        raise RuntimeError("guasto simulato durante l'addestramento")
    V.passo_ottimizzazione = passo_rotto
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (1, 2)}, ids, cfg, 2,
                                         strati_teacher={L: pul[L] for L in (1, 2)}, emb=emb,
                                         ternari_iniziali={1: portato}, ternari_uscita={}, n_batch_addestramento=n_tr)
        ok("guasto dopo la copia: propagato", False)
    except RuntimeError as e:
        ok("guasto dopo la copia: propagato", "guasto simulato" in str(e))
    finally:
        V.passo_ottimizzazione = orig_passo
    ok("guasto dopo la copia: oggetto svuotato e marcato (lo stato vero e' nel punto di ripresa)",
       portato["gu"].z.numel() == 0 and portato["gu"]._svuotato_dal_raccordo)
    try:
        R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (1, 2)}, ids, cfg, 2, strati_teacher={L: pul[L] for L in (1, 2)},
                                     emb=emb, ternari_iniziali={1: portato}, ternari_uscita={}, n_batch_addestramento=n_tr)
        ok("riuso di un oggetto svuotato: RIFIUTATO", False)
    except RuntimeError as e:
        ok("riuso di un oggetto svuotato: RIFIUTATO", "SVUOTATO" in str(e), str(e)[:90])
    for nome, kw in (("strato fuori finestra", {"ternari_iniziali": {7: portato}}),
                     ("argomento di A passato a mano", {"adam_ripartenza": "conserva"})):
        try:
            R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (1, 2)}, ids, cfg, 2,
                                         strati_teacher={L: pul[L] for L in (1, 2)}, emb=emb,
                                         n_batch_addestramento=n_tr, **kw)
            ok(f"{nome}: RIFIUTATO", False)
        except (ValueError, TypeError) as e:
            ok(f"{nome}: RIFIUTATO", True, str(e)[:80])


def giocattolo_largo(TB, V, n: int, dtype):
    """Strati VERI alla larghezza di Thor dove conta per il formato: hidden 2048 (rotazione
    a blocchi di 1024 ATTIVA, gruppo gate/up 128), moe_intermediate 512 (gruppo down 256);
    pochi esperti, poche teste, 6 token. A scala giocattolo n_in=8 la rotazione e' l'identita'
    e i gruppi sono l'intera riga: qui passano rotazione dei pesi, rotazione sull'ingresso,
    de-rotazione del commit e dell'export, esattamente come su Thor."""
    import torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    base_cls = getattr(TB, "_GiocattoloOriginale", TB.Giocattolo)

    class Largo(base_cls):
        def __init__(self, *_a, **_k):
            super().__init__(n, dtype)
            self.cfg = Qwen3_5MoeTextConfig(
                vocab_size=32, hidden_size=2048, num_hidden_layers=n, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, moe_intermediate_size=512,
                shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
                layer_types=["full_attention"] * n, partial_rotary_factor=1.0)
            self.cfg._attn_implementation = "sdpa"
            if os.getenv("THOR_ESPERTI_THOR", "0") == "1":
                import esperti_thor
                esperti_thor.registra()
                self.cfg._experts_implementation = esperti_thor.NOME
            else:
                self.cfg._experts_implementation = "grouped_mm"
            g = torch.Generator().manual_seed(7)
            self.emb = (torch.randn(32, 2048, generator=g) * 0.5).to(dtype)
    gioc = Largo()
    assert V._group_giocattolo(gioc.cfg.hidden_size, V.GRUPPO_GATEUP_REALE) == 128
    assert V._group_giocattolo(gioc.cfg.moe_intermediate_size, V.GRUPPO_DOWN_REALE) == 256
    return gioc


def r7_largo(TB, FS, R, V, rc, TDBF, base: Path, dtype, toll_q: float) -> None:
    print("\n── R7 LARGHEZZA VERA (hidden 2048, rotazione attiva, gruppi 128/256): catena + modalita' (b) ──")
    n = 4
    for modo, init_dir, corsa_id in (("teacher", "", "largo"), ("codici", str(base / "r7_teacher" / "out"), "largo_b")):
        gioc = giocattolo_largo(TB, V, n, dtype)
        buf = io.StringIO()
        oss = TB.Osservatore()
        cartella = base / f"r7_{modo}"
        V.OUT_DIR, V.ID_CORSA = cartella / "out", corsa_id
        with Spie(V, rc, R, TDBF) as spie, contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            FS.esegui(n, 2, 3, modo, init_dir, "largo" if init_dir else "", salta_export=True,
                      carica_strati=gioc.carica, carica_batch=gioc.batch, seme_iniziale=gioc.seme0,
                      ottimizza_fn=R.ottimizza_finestra_JOINT_B, da_codici_fn=R.ternari_da_codici,
                      ripresa_dir=cartella / "ripresa", osservatore=oss)
        # controlli_catena ricostruisce il seme con un Giocattolo NUOVO: deve essere largo anche lui
        TB._GiocattoloOriginale = TB.Giocattolo
        TB.Giocattolo = type(gioc)
        try:
            TB.controlli_catena(f"R7-{modo}", cartella, gioc, oss, 2, corsa_id, toll_q=toll_q)
        finally:
            TB.Giocattolo = TB._GiocattoloOriginale
            del TB._GiocattoloOriginale
        righe = _righe_ripartenza(buf.getvalue())
        attese = (n - 2) + (n if modo == "codici" else 0)
        ok(f"R7-{modo}: {attese} ripartenze, tutte con codici identici 100% e scarto pesi 0",
           len(righe) == attese and _tutte_esatte(righe), f"{len(righe)} righe")
        ok(f"R7-{modo}: spia — codici letti nel modulo all'ingresso = attesi, al bit",
           len(spie.ingressi) == attese and all(u for _, _, u in spie.ingressi))
        if modo == "codici":
            print("    " + next((r.strip() for r in righe if "da CODICI" in r), "")[:250])


# ═══════════════════════ R8 — Adam conservato (THOR_SCORREVOLE_ADAM=conserva, 25/9) ═══════════════════════
@contextlib.contextmanager
def _env(**kv):
    prima = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in prima.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _patch(modulo, **attr):
    prima = {k: getattr(modulo, k) for k in attr}
    for k, v in attr.items():
        setattr(modulo, k, v)
    try:
        yield
    finally:
        for k, v in prima.items():
            setattr(modulo, k, v)


def _clona(x):
    import torch
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _clona(v) for k, v in x.items()}
    return x


def _uguali_al_bit(a, b) -> bool:
    import torch
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape \
            and torch.equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and sorted(a) == sorted(b) \
            and all(_uguali_al_bit(a[k], b[k]) for k in a)
    return a == b


def _parametri_di(voce: dict, rc) -> dict:
    return {f: {n: getattr(o, n).detach().clone() for n in rc.NOMI_PARAMETRI} for f, o in voce.items() if o is not None}


class SpiaPassi:
    """Spia sull'ottimizzatore DENTRO la finestra: moduli degli strati ripartiti (applica_ripartenza_),
    stato di Adam e parametri all'ingresso del passo 0 (prima di qualunque aggiornamento) e parametri
    all'ingresso del passo 1 (= dopo il primo aggiornamento)."""

    def __init__(self, V, rc):
        self.V, self.rc = V, rc
        self.moduli: dict = {}
        self.n = 0
        self.stato0 = self.par0 = self.par1 = None
        self.altri_vuoti0 = None

    def __enter__(self):
        V, rc = self.V, self.rc
        orig_passo, orig_app = V.passo_ottimizzazione, rc.applica_ripartenza_
        self._orig = (orig_passo, orig_app)

        def app(L, voce, t_gu, t_dw, *a, **k):
            r = orig_app(L, voce, t_gu, t_dw, *a, **k)
            self.moduli[L] = {"gu": t_gu, "dw": t_dw}
            return r

        def passo(*a, **k):
            opt = a[3]
            if self.n == 0:
                self.stato0 = {L: {f: {n: _clona(opt.state.get(getattr(o, n))) for n in rc.NOMI_PARAMETRI}
                                   for f, o in m.items() if o is not None} for L, m in self.moduli.items()}
                self.par0 = {L: _parametri_di(m, rc) for L, m in self.moduli.items()}
                ids = {id(getattr(o, n)) for m in self.moduli.values() for o in m.values() if o is not None
                       for n in rc.NOMI_PARAMETRI}
                self.altri_vuoti0 = all(not opt.state.get(p) for g in opt.param_groups for p in g["params"]
                                        if id(p) not in ids)
            elif self.n == 1:
                self.par1 = {L: _parametri_di(m, rc) for L, m in self.moduli.items()}
            self.n += 1
            return orig_passo(*a, **k)

        V.passo_ottimizzazione, rc.applica_ripartenza_ = passo, app
        return self

    def __exit__(self, *_):
        self.V.passo_ottimizzazione, self.rc.applica_ripartenza_ = self._orig
        return False


def r8a_passaggio(TB, R, V, rc, dtype, etichetta: str, segno_atteso: bool) -> None:
    """Collaudo (2): stato di Adam dello strato portato alla fine della finestra k = stato
    nell'ottimizzatore della finestra k+1 al passo 0, al bit. Piu' la misura della «scossa»:
    con Adam nuovo il primo passo vale lr*g/(|g|+eps), cioe' ~lr su ogni peso con |g| >> eps
    (controllato solo con `segno_atteso`: in fase morbida i gradienti sono ~1e-5 >> eps 1e-8; in
    sola fase dura il surrogato con beta 16 da' gradienti anche sotto eps). Con Adam conservato
    il passo e' normalizzato dalla storia: NON e' garantito piu' piccolo (gradiente costante ->
    stesso passo di segno), quindi il confronto fra i due resta una misura, non un controllo."""
    print(f"\n── R8a passaggio di Adam fra finestre ({etichetta}) ──")
    import torch
    n_passi = 4
    scossa = {}
    for modo in ("conserva", "riparti"):
        gioc = TB.Giocattolo(3, dtype)
        cfg, emb, pul, qua = gioc.carica([0, 1, 2])
        ids, n_tr = gioc.batch()
        usc: dict = {}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (0, 1)}, ids, cfg, n_passi,
                                         strati_teacher={L: pul[L] for L in (0, 1)}, emb=emb, ternari_iniziali=None,
                                         ternari_uscita=usc, n_batch_addestramento=n_tr, modo_adam_esplicito=modo)
        con_adam = [(L, f) for L in (0, 1) for f, o in usc[L].items()
                    if o is not None and getattr(o, R.CHIAVE_ADAM, None) is not None]
        portato = usc[1]
        if modo == "conserva":
            ok(f"R8a-{etichetta}: a fine finestra ogni oggetto che esce porta il suo stato di Adam (gu e dw, 2 strati)",
               len(con_adam) == 4, f"{con_adam}")
            passi_z = {f: float(getattr(o, R.CHIAVE_ADAM)["z"]["step"]) for f, o in portato.items()}
            ok(f"R8a-{etichetta}: step di z = {n_passi} (passi della finestra)",
               all(v == n_passi for v in passi_z.values()), f"{passi_z}")
            fine = {f: _clona(getattr(o, R.CHIAVE_ADAM)) for f, o in portato.items()}
        else:
            ok(f"R8a-{etichetta}: riparti (default): nessun oggetto porta stato di Adam", not con_adam, f"{con_adam}")
        par_fine = _parametri_di(portato, rc)
        usc2: dict = {}
        buf = io.StringIO()
        with SpiaPassi(V, rc) as spia, contextlib.redirect_stdout(buf):
            R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (1, 2)}, ids, cfg, n_passi,
                                         strati_teacher={L: pul[L] for L in (1, 2)}, emb=emb,
                                         ternari_iniziali={1: portato}, ternari_uscita=usc2,
                                         n_batch_addestramento=n_tr, modo_adam_esplicito=modo)
        ok(f"R8a-{etichetta}-{modo}: parametri dello strato portato al passo 0 = fine della finestra precedente, al bit",
           _uguali_al_bit(spia.par0[1], par_fine))
        ok(f"R8a-{etichetta}-{modo}: strato entrante con Adam NUOVO (nessuno stato al passo 0)", spia.altri_vuoti0)
        if modo == "conserva":
            uguali = _uguali_al_bit(spia.stato0[1], fine)
            ok(f"R8a-{etichetta}: exp_avg / exp_avg_sq / step dello strato portato al passo 0 della finestra k+1 = "
               f"fine della finestra k, AL BIT", uguali,
               f"step z al passo 0: {float(spia.stato0[1]['gu']['z']['step'])}")
            righe = [r for r in buf.getvalue().splitlines() if "stato di Adam CONSERVATO" in r]
            ok(f"R8a-{etichetta}: A dichiara lo stato conservato per il solo strato portato", len(righe) == 1
               and "strato 1" in righe[0], righe[0].strip()[:110] if righe else "nessuna riga")
            passi_z2 = {L: float(getattr(usc2[L]["gu"], R.CHIAVE_ADAM)["z"]["step"]) for L in (1, 2)}
            ok(f"R8a-{etichetta}: dopo la finestra k+1 il contatore continua (portato {2 * n_passi}, entrante {n_passi})",
               passi_z2 == {1: 2 * n_passi, 2: n_passi}, f"{passi_z2}")
        else:
            ok(f"R8a-{etichetta}: riparti: stato del portato VUOTO al passo 0 (AdamW nuovo)",
               all(v is None for fam in spia.stato0[1].values() for v in fam.values()))
        lr = float(os.getenv("THOR_LR_Z", str(V.NUCLEO_LR)))
        dz = (spia.par1[1]["gu"]["z"].float() - spia.par0[1]["gu"]["z"].float()).abs().flatten() / lr
        q = torch.quantile(dz, torch.tensor([0.1, 0.5, 0.9]))
        scossa[modo] = (q.tolist(), float((dz > 0.5).float().mean()))
        print(f"    (misura) {modo}: primo passo sullo strato portato, |Δz|/lr quantili 10/50/90% "
              f"{[round(x, 3) for x in q.tolist()]} · frazione con |Δz| > lr/2: {scossa[modo][1]*100:.1f}%")
    if segno_atteso:
        ok(f"R8a-{etichetta}: riparti = passo di SEGNO sullo strato gia' ottimizzato (mediana |Δz|/lr in [0,95; 1])",
           0.95 <= scossa["riparti"][0][1] <= 1.0,
           f"mediane riparti {scossa['riparti'][0][1]:.4f} · conserva {scossa['conserva'][0][1]:.4f}")


def r8b_equivalenza(TB, R, V, rc) -> None:
    """Collaudo (3): con l'obiettivo IDENTICO fra le due chiamate (stessa finestra, stessi dati),
    2 chiamate da N passi con Adam conservato = 1 chiamata da 2N passi, al bit. Condizioni per
    cui l'obiettivo e' davvero lo stesso: solo fase dura (beta costante = BETA_MAX: con la fase
    morbida la schedula di beta riparte a ogni chiamata), routing vivo (il routing congelato si
    ricalcola all'ingresso di ogni chiamata), N multiplo delle batch di addestramento (il ciclo
    b = t mod n_tr riparte da 0), lr costante. Nella scorrevole vera l'obiettivo cambia (lo strato
    k e' commesso duro, entra k+2): li' e' confrontabile solo lo STATO (R8a), non la traiettoria."""
    print("\n── R8b equivalenza: 2 finestre da N passi con Adam conservato = 1 finestra da 2N passi ──")
    N = 4

    class Prova:
        """Strati giocattolo NUOVI (stessi numeri: fabbrica deterministica) per ogni prova."""

        def __init__(self):
            gioc = TB.Giocattolo(2)
            self.cfg, self.emb, self.pul, self.qua = gioc.carica([0, 1])
            self.ids, self.n_tr = gioc.batch()
            assert N % self.n_tr == 0

        def finestra(self, n_passi, iniziali, modo):
            usc, diag = {}, []
            with contextlib.redirect_stdout(io.StringIO()):
                ris = R.ottimizza_finestra_JOINT_B({L: self.qua[L] for L in (0, 1)}, self.ids, self.cfg, n_passi,
                                                   strati_teacher={L: self.pul[L] for L in (0, 1)}, emb=self.emb,
                                                   ternari_iniziali=iniziali, ternari_uscita=usc,
                                                   n_batch_addestramento=self.n_tr, raccogli_diagnostica=diag,
                                                   modo_adam_esplicito=modo)
            return usc, ris, diag[0]["perdite"]

    with _patch(V, NUCLEO_FRAZIONE_DURA=1.0, ROUTING_VIVO=True):
        assert V.LR_SCHEDULA == "costante"
        usc_A, ris_A, perd_A = Prova().finestra(2 * N, None, "conserva")
        esiti = {}
        for modo in ("conserva", "riparti"):
            p = Prova()
            usc_1, _, perd_1 = p.finestra(N, None, modo)
            usc_2, ris_2, perd_2 = p.finestra(N, {0: usc_1[0], 1: usc_1[1]}, modo)
            esiti[modo] = {
                "perdite": perd_1 + perd_2 == perd_A,
                "parametri": all(_uguali_al_bit(_parametri_di(usc_2[L], rc), _parametri_di(usc_A[L], rc)) for L in (0, 1)),
                "codici": TB.uguali(ris_2[0], ris_A[0]) and TB.uguali(ris_2[1], ris_A[1]),
                "adam": modo == "conserva" and all(
                    _uguali_al_bit(getattr(usc_2[L][f], R.CHIAVE_ADAM), getattr(usc_A[L][f], R.CHIAVE_ADAM))
                    for L in (0, 1) for f in ("gu", "dw")),
                "prima_diversa": next((i for i, (a, b) in enumerate(zip(perd_1 + perd_2, perd_A)) if a != b), None),
            }
    e = esiti["conserva"]
    ok(f"R8b: conserva — loss passo per passo ({2 * N}) identiche alla finestra unica", e["perdite"],
       f"prima differenza al passo {e['prima_diversa']}")
    ok("R8b: conserva — parametri finali (z, log_s, delta_raw; gu e dw; 2 strati) identici AL BIT", e["parametri"])
    ok("R8b: conserva — stato di Adam finale identico AL BIT", e["adam"])
    ok("R8b: conserva — codici esportati identici", e["codici"])
    r = esiti["riparti"]
    ok("R8b: controllo di sensibilita' — riparti DIVERGE dalla finestra unica gia' al primo passo dopo la ripartenza",
       not r["perdite"] and not r["parametri"] and r["prima_diversa"] == N + 1,
       f"prima loss diversa al passo {r['prima_diversa']} (il passo {N} usa ancora i parametri uguali)")


def _riscrivi_portato(cartella_punto: Path, L: int, modifica) -> None:
    """Riscrive portato_LL.pt con `modifica(dict)` e aggiorna sha256 e byte nel manifest, come se
    il file l'avesse scritto cosi' un altro codice (il controllo d'integrita' non deve intervenire)."""
    import hashlib
    import json
    import torch
    p = cartella_punto / f"portato_{L:02d}.pt"
    d = torch.load(p, weights_only=True)
    modifica(d)
    torch.save(d, p)
    man_p = cartella_punto / "manifest.json"
    man = json.loads(man_p.read_text())
    man["file"][p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "byte": p.stat().st_size}
    man_p.write_text(json.dumps(man, indent=1, ensure_ascii=False))


class SpiaAdamIniettato:
    """Stato di Adam che ogni finestra riceve per i suoi strati portati (clonato all'ingresso di A)."""

    def __init__(self, V, rc):
        self.V, self.rc, self.visti = V, rc, {}

    def __enter__(self):
        orig = self._orig = self.V.ottimizza_finestra_JOINT

        @functools.wraps(orig)
        def ottimizza(*a, **k):
            for L, st in (k.get("ripartenza_per_strato") or {}).items():
                if isinstance(st, self.rc.StatoStrato):
                    self.visti[L] = _clona(st.adam)
            return orig(*a, **k)
        self.V.ottimizza_finestra_JOINT = ottimizza
        return self

    def __exit__(self, *_):
        self.V.ottimizza_finestra_JOINT = self._orig
        return False


def r8c_ripresa(TB, FS, R, V, rc, base: Path) -> None:
    """Collaudo (4) + (5, lato ripresa): catena con Adam conservato interrotta e ripresa: stato di
    Adam ricevuto da ogni finestra e codici IDENTICI alla corsa senza interruzioni; punto di
    ripresa senza momenti o con forme sbagliate RIFIUTATO senza toccare nulla."""
    import torch
    print("\n── R8c ripresa con Adam conservato (6 strati, scrittura sincrona) ──")
    n = 6
    corri = functools.partial(TB.corsa, asincrona=False, ottimizza_fn=R.ottimizza_finestra_JOINT_B)
    with _env(THOR_SCORREVOLE_ADAM="conserva"):
        with SpiaAdamIniettato(V, rc) as spia_rif, contextlib.redirect_stdout(io.StringIO()):
            corri(base / "r8c_rif", TB.Giocattolo(n))
        rif = TB.codici_su_disco(base / "r8c_rif", n, "prova")
        ok("R8c: ricetta della corsa con THOR_SCORREVOLE_ADAM=conserva (le THOR_* contano tutte)",
           FS.Ripresa(base / "r8c_ricetta", "x", n, 2).ricetta.get("THOR_SCORREVOLE_ADAM") == "conserva"
           and "THOR_SCORREVOLE_ADAM" not in FS._NON_RICETTA)
        ok("R8c: corsa di riferimento: ogni finestra dopo la prima riceve lo stato di Adam del portato",
           sorted(spia_rif.visti) == list(range(1, n - 1)) and all(v is not None for v in spia_rif.visti.values()))
    with contextlib.redirect_stdout(io.StringIO()):
        corri(base / "r8c_rif_riparti", TB.Giocattolo(n))
    rif_rip = TB.codici_su_disco(base / "r8c_rif_riparti", n, "prova")
    diversi = [L for L in range(n) if not TB.uguali(rif[L], rif_rip[L])]
    ok("R8c: sensibilita' — con riparti i codici cambiano (il confronto sotto puo' fallire)", bool(diversi),
       f"strati con codici diversi: {diversi}")

    casi = [("prima_transizione", 2), ("dopo_ripresa", 2), ("dopo_commit", 2), ("dopo_commit_strato", 4),
            ("prima_transizione", 3)]
    pesi = {}
    for punto_nome, k_int in casi:
        d = base / f"r8c_{punto_nome}_{k_int}"

        def interrompi(p, k, _pn=punto_nome, _k=k_int):
            if p == _pn and k == _k:
                raise FS.InterruzioneSimulata(f"{p}@{k}")
        with _env(THOR_SCORREVOLE_ADAM="conserva"):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    corri(d, TB.Giocattolo(n), interrompi=interrompi)
                ok(f"R8c: interruzione {punto_nome}@{k_int} avvenuta", False)
                continue
            except FS.InterruzioneSimulata:
                pass
            punti = sorted((d / "ripresa").glob("finestra_*/portato_*.pt"))
            if punti:
                s = torch.load(punti[0], weights_only=True)
                pesi[punti[0].name] = (punti[0].stat().st_size, "adam" in s["gu"] and "adam" in s["dw"])
            with SpiaAdamIniettato(V, rc) as spia, contextlib.redirect_stdout(io.StringIO()):
                oss = corri(d, TB.Giocattolo(n))
        fatte = [k for (e, k, _) in oss.eventi if e == "finestra_inizio"]
        uguali_cod = all(TB.uguali(rif[L], v) for L, v in TB.codici_su_disco(d, n, "prova").items())
        uguali_adam = sorted(spia.visti) == [L for L in range(1, n - 1) if L >= fatte[0]] and all(
            _uguali_al_bit(spia.visti[L], spia_rif.visti[L]) for L in spia.visti)
        ok(f"R8c: interruzione {punto_nome}@{k_int} -> ripresa dalla finestra {fatte[0]}: stato di Adam ricevuto "
           f"(finestre {sorted(spia.visti)}) e codici IDENTICI alla corsa senza interruzioni", uguali_cod and uguali_adam,
           f"codici {uguali_cod} · adam {uguali_adam}")
    ok("R8c: i punti di ripresa con conserva contengono i momenti (gu e dw)", pesi and all(a for _, a in pesi.values()),
       f"{pesi}")

    # punto di ripresa senza momenti / con forme sbagliate / ricetta con modo diverso -> rifiuti, niente toccato
    d = base / "r8c_rifiuti"

    def stop(p, k):
        if p == "prima_transizione" and k == 3:
            raise FS.InterruzioneSimulata("stop")
    with _env(THOR_SCORREVOLE_ADAM="conserva"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                corri(d, TB.Giocattolo(n), interrompi=stop)
        except FS.InterruzioneSimulata:
            pass
    punto = d / "ripresa" / "finestra_03"
    originali = {p.name: p.read_bytes() for p in punto.iterdir()}
    c0 = FS.strati_commessi(d / "out", n, "prova", True)
    with _env(THOR_SCORREVOLE_ADAM="riparti"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                corri(d, TB.Giocattolo(n))
            ok("R8c: ripresa con THOR_SCORREVOLE_ADAM cambiato (conserva -> riparti) RIFIUTATA", False)
        except RuntimeError as e:
            ok("R8c: ripresa con THOR_SCORREVOLE_ADAM cambiato (conserva -> riparti) RIFIUTATA",
               "ricetta diversa" in str(e) and "THOR_SCORREVOLE_ADAM" in str(e), str(e)[:120])

    def senza_adam(s):
        del s["gu"]["adam"], s["dw"]["adam"]

    def forma_sbagliata(s):
        s["gu"]["adam"]["z"]["exp_avg"] = s["gu"]["adam"]["z"]["exp_avg"][..., :1].clone()

    def senza_step(s):
        del s["dw"]["adam"]["z"]["step"]

    for nome, modifica, attesa in (("senza momenti (codice che non li salvava)", senza_adam, "MANCANTE"),
                                   ("momento di z con forma sbagliata", forma_sbagliata, "forma"),
                                   ("stato senza 'step'", senza_step, "chiavi")):
        for p in punto.iterdir():
            p.write_bytes(originali[p.name])
        _riscrivi_portato(punto, 3, modifica)
        chiamate = []
        with _env(THOR_SCORREVOLE_ADAM="conserva"), \
                _patch(V, ottimizza_finestra_JOINT=functools.wraps(V.ottimizza_finestra_JOINT)(
                    lambda *a, _o=V.ottimizza_finestra_JOINT, **k: (chiamate.append(1), _o(*a, **k))[1])):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    corri(d, TB.Giocattolo(n))
                ok(f"R8c: punto di ripresa {nome}: RIFIUTATO", False)
            except RuntimeError as e:
                ok(f"R8c: punto di ripresa {nome}: RIFIUTATO prima di ottimizzare, nessuna marca nuova",
                   attesa in str(e) and not chiamate and FS.strati_commessi(d / "out", n, "prova", True) == c0,
                   str(e)[:110])
    for p in punto.iterdir():
        p.write_bytes(originali[p.name])
    with _env(THOR_SCORREVOLE_ADAM="conserva"), contextlib.redirect_stdout(io.StringIO()):
        corri(d, TB.Giocattolo(n))
    ok("R8c: rimesso il punto originale, la ripresa completa e combacia con la corsa senza interruzioni",
       all(TB.uguali(rif[L], v) for L, v in TB.codici_su_disco(d, n, "prova").items()))
    # corsa di DEFAULT (variabile assente) interrotta: riprenderla con THOR_SCORREVOLE_ADAM=riparti scritta a
    # mano e' un'altra ricetta (assente != "riparti"): rifiutata; senza variabile riprende
    d = base / "r8c_default_esplicito"
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            corri(d, TB.Giocattolo(n), interrompi=stop)
        except FS.InterruzioneSimulata:
            pass
    with _env(THOR_SCORREVOLE_ADAM="riparti"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                corri(d, TB.Giocattolo(n))
            ok("R8c: corsa di default ripresa con THOR_SCORREVOLE_ADAM=riparti esplicita: RIFIUTATA", False)
        except RuntimeError as e:
            ok("R8c: corsa di default ripresa con THOR_SCORREVOLE_ADAM=riparti esplicita: RIFIUTATA (assente != 'riparti')",
               "ricetta diversa" in str(e) and "THOR_SCORREVOLE_ADAM: None -> 'riparti'" in str(e), str(e)[:120])
    with contextlib.redirect_stdout(io.StringIO()):
        corri(d, TB.Giocattolo(n))
    ok("R8c: ... e senza la variabile la stessa ripresa completa e combacia con la corsa di default intera",
       all(TB.uguali(rif_rip[L], v) for L, v in TB.codici_su_disco(d, n, "prova").items()))


def r8d_errori(TB, R, V, rc) -> None:
    """Collaudo (5): errori rumorosi sulla chiamata diretta, SEMPRE prima di toccare gli oggetti."""
    print("\n── R8d errori rumorosi (chiamata diretta) ──")
    import torch
    gioc = TB.Giocattolo(3)
    cfg, emb, pul, qua = gioc.carica([0, 1, 2])
    ids, n_tr = gioc.batch()

    def finestra(iniziali, modo, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (1, 2)}, ids, cfg, 2,
                                                strati_teacher={L: pul[L] for L in (1, 2)}, emb=emb,
                                                ternari_iniziali=iniziali, ternari_uscita={}, n_batch_addestramento=n_tr,
                                                modo_adam_esplicito=modo, **kw)

    def portato_da(modo):
        usc = {}
        with contextlib.redirect_stdout(io.StringIO()):
            R.ottimizza_finestra_JOINT_B({L: qua[L] for L in (0, 1)}, ids, cfg, 2, strati_teacher={L: pul[L] for L in (0, 1)},
                                         emb=emb, ternari_iniziali=None, ternari_uscita=usc, n_batch_addestramento=n_tr,
                                         modo_adam_esplicito=modo)
        return usc[1]

    def intatto(voce, par, adam) -> bool:
        return (not voce["gu"]._svuotato_dal_raccordo and _uguali_al_bit(_parametri_di(voce, rc), par)
                and all(_uguali_al_bit(getattr(voce[f], R.CHIAVE_ADAM, None), adam.get(f)) for f in ("gu", "dw")))

    casi = []
    v = portato_da("riparti")
    casi.append(("conserva con oggetto SENZA stato di Adam", v, "conserva", "MANCANTE"))
    v = portato_da("conserva")
    casi.append(("riparti con oggetto CON stato di Adam", v, "riparti", "incoerente"))
    for nome, alterazione, attesa in (
            ("dw senza stato (gu con)", lambda v: delattr(v["dw"], R.CHIAVE_ADAM), "MANCANTE"),
            ("exp_avg_sq di log_s con forma sbagliata",
             lambda v: getattr(v["gu"], R.CHIAVE_ADAM)["log_s"].__setitem__(
                 "exp_avg_sq", torch.zeros(3)), "forma"),
            ("exp_avg di z in float64", lambda v: getattr(v["dw"], R.CHIAVE_ADAM)["z"].__setitem__(
                "exp_avg", getattr(v["dw"], R.CHIAVE_ADAM)["z"]["exp_avg"].double()), "forma"),
            ("chiave in piu' (max_exp_avg_sq)", lambda v: getattr(v["gu"], R.CHIAVE_ADAM)["z"].__setitem__(
                "max_exp_avg_sq", torch.zeros(1)), "chiavi"),
            ("step = 0", lambda v: getattr(v["gu"], R.CHIAVE_ADAM)["z"].__setitem__("step", torch.tensor(0.0)), "step"),
            ("parametro mancante nello stato", lambda v: getattr(v["gu"], R.CHIAVE_ADAM).pop("delta_raw"), "chiavi")):
        v = portato_da("conserva")
        alterazione(v)
        casi.append((nome, v, "conserva", attesa))
    for nome, v, modo, attesa in casi:
        par = _parametri_di(v, rc)
        adam = {f: getattr(v[f], R.CHIAVE_ADAM, None) for f in ("gu", "dw")}
        try:
            finestra({1: v}, modo)
            ok(f"R8d: {nome}: RIFIUTATO", False)
        except RuntimeError as e:
            ok(f"R8d: {nome}: RIFIUTATO, oggetto e stato intatti", attesa in str(e) and intatto(v, par, adam),
               str(e)[:100])
    try:
        finestra(None, "boh")
        ok("R8d: modo sconosciuto RIFIUTATO", False)
    except ValueError as e:
        ok("R8d: modo sconosciuto RIFIUTATO", "THOR_SCORREVOLE_ADAM" in str(e), str(e)[:80])
    with _env(THOR_SCORREVOLE_ADAM="boh"):
        try:
            finestra(None, None)
            ok("R8d: THOR_SCORREVOLE_ADAM=boh RIFIUTATO", False)
        except ValueError as e:
            ok("R8d: THOR_SCORREVOLE_ADAM=boh RIFIUTATO", "boh" in str(e), str(e)[:80])
    # guasti con Adam conservato: prima della copia -> oggetto E momenti ripristinati; dopo -> svuotato
    v = portato_da("conserva")
    par = _parametri_di(v, rc)
    adam = {f: getattr(v[f], R.CHIAVE_ADAM) for f in ("gu", "dw")}
    adam_val = _clona(adam)

    def rotta(*a, **k):
        raise KeyboardInterrupt("guasto simulato prima della copia")
    with _patch(V, ottimizza_finestra_JOINT=functools.wraps(V.ottimizza_finestra_JOINT)(rotta)):
        try:
            finestra({1: v}, "conserva")
        except KeyboardInterrupt:
            pass
    ok("R8d: guasto prima della copia: parametri E stato di Adam ripristinati (stessi tensori, stessi valori)",
       intatto(v, par, adam) and _uguali_al_bit({f: getattr(v[f], R.CHIAVE_ADAM) for f in ("gu", "dw")}, adam_val))

    def passo_rotto(*a, **k):
        raise RuntimeError("guasto simulato durante l'addestramento")
    with _patch(V, passo_ottimizzazione=passo_rotto):
        try:
            finestra({1: v}, "conserva")
        except RuntimeError:
            pass
    ok("R8d: guasto dopo la copia: oggetto svuotato, marcato e SENZA stato di Adam (vero solo su disco)",
       v["gu"].z.numel() == 0 and v["gu"]._svuotato_dal_raccordo and getattr(v["gu"], R.CHIAVE_ADAM, None) is None)
    try:
        finestra({1: v}, "conserva")
        ok("R8d: riuso dell'oggetto svuotato RIFIUTATO", False)
    except RuntimeError as e:
        ok("R8d: riuso dell'oggetto svuotato RIFIUTATO", "SVUOTATO" in str(e), str(e)[:80])


def r8e_catena(TB, FS, R, V, rc, TDBF, base: Path, dtype, toll_q: float, etichetta: str) -> None:
    """Catena scorrevole completa con Adam conservato: stessi controlli di R2/R3 (codici, semi,
    memoria) piu': ogni finestra dopo la prima riprende i momenti del portato; dopo ogni commit lo
    stato di Adam vive SOLO sugli oggetti portati (quello dello strato commesso e' liberato)."""
    print(f"\n── R8e catena scorrevole con Adam conservato, 10 strati, {etichetta} ──")
    n = 10
    gioc = TB.Giocattolo(n, dtype)
    buf = io.StringIO()
    con_adam: list = []
    cartella = base / f"r8e_{etichetta}"

    def _oggetti_con_adam() -> int:
        gc.collect()
        return sum(1 for o in gc.get_objects()
                   if isinstance(o, TDBF) and o.z.numel() > 0 and getattr(o, R.CHIAVE_ADAM, None) is not None)

    class Oss(TB.Osservatore):
        def __call__(self, evento, **d):
            super().__call__(evento, **d)
            if evento == "commit" and d.get("strato") == d.get("finestra"):
                con_adam.append((d["finestra"], _oggetti_con_adam()))
    oss = Oss()
    V.OUT_DIR, V.ID_CORSA = cartella / "out", "prova"
    with _env(THOR_SCORREVOLE_ADAM="conserva"), SpiaAdamIniettato(V, rc) as spia, \
            contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        FS.esegui(n, 2, 3, "teacher", "", "", salta_export=True, carica_strati=gioc.carica,
                  carica_batch=gioc.batch, seme_iniziale=gioc.seme0, ottimizza_fn=R.ottimizza_finestra_JOINT_B,
                  da_codici_fn=R.ternari_da_codici, ripresa_dir=cartella / "ripresa", osservatore=oss)
    TB.controlli_catena(f"R8e-{etichetta}", cartella, gioc, oss, 2, "prova", toll_q=toll_q)
    righe = [r for r in buf.getvalue().splitlines() if "stato di Adam CONSERVATO" in r]
    ok(f"R8e-{etichetta}: una ripresa di Adam per finestra dopo la prima ({n - 2}), sullo strato portato",
       len(righe) == n - 2 and all(f"strato {k}:" in r for k, r in zip(range(1, n - 1), righe)),
       righe[0].strip()[:100] if righe else "nessuna")
    passi = [float(spia.visti[L]["gu"]["z"]["step"]) for L in sorted(spia.visti)]
    ok(f"R8e-{etichetta}: ogni portato arriva con step = passi della finestra precedente (3)",
       sorted(spia.visti) == list(range(1, n - 1)) and all(p == 3 for p in passi), f"{passi}")
    per_strato = 2 if V.DOWN_TERNARIO else 1
    ok(f"R8e-{etichetta}: dopo ogni commit lo stato di Adam vive solo sugli oggetti portati ({per_strato}; 0 alla fine)",
       len(con_adam) == n - 1 and all(c == (0 if k == n - 2 else per_strato) for k, c in con_adam), f"{con_adam}")


def r8f_peso(FS, R, rc, base: Path) -> None:
    """Peso su disco: file portato_LL.pt con e senza momenti a scala giocattolo (misurato) e a scala
    Thor (parametri contati su oggetti costruiti sul device 'meta': forme vere, zero memoria)."""
    import torch
    from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso as TDBF
    print("\n── R8f peso del punto di ripresa: momenti di Adam ──")
    ok("R8f: stesso nome dell'attributo in forgia (Ripresa) e raccordo", FS.ATTR_ADAM == R.CHIAVE_ADAM)
    # misura: UN esperto alla larghezza vera di Thor (gate_up 1024x2048 gruppo 128, down 2048x512 gruppo 256),
    # salvato con _stato_ternario come fa Ripresa.prepara, senza e con i momenti (~36 MiB in tutto)
    g = torch.Generator().manual_seed(11)
    gu = TDBF(torch.randn(1, 1024, 2048, generator=g) * 0.02, group=128)
    dw = TDBF(torch.randn(1, 2048, 512, generator=g) * 0.02, group=256)

    def stato_finto(o):
        return {n: {"step": torch.tensor(75.0), "exp_avg": torch.randn(getattr(o, n).shape, generator=g),
                    "exp_avg_sq": torch.rand(getattr(o, n).shape, generator=g)} for n in rc.NOMI_PARAMETRI}
    d = base / "r8f"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"gu": FS._stato_ternario(gu), "dw": FS._stato_ternario(dw)}, d / "senza.pt")
    adam = {"gu": stato_finto(gu), "dw": stato_finto(dw)}
    setattr(gu, R.CHIAVE_ADAM, adam["gu"])
    setattr(dw, R.CHIAVE_ADAM, adam["dw"])
    torch.save({"gu": FS._stato_ternario(gu), "dw": FS._stato_ternario(dw)}, d / "con.pt")
    b_senza, b_con = (d / "senza.pt").stat().st_size, (d / "con.pt").stat().st_size
    n_par = sum(getattr(o, n).numel() for o in (gu, dw) for n in rc.NOMI_PARAMETRI)
    ok(f"R8f: con i momenti il file dello strato portato pesa 3x (misurato: {b_senza / 2**20:.2f} -> "
       f"{b_con / 2**20:.2f} MiB; aggiunta = {(b_con - b_senza) / (8 * n_par):.5f} x 8 byte per parametro)",
       abs(b_con / b_senza - 3.0) < 0.01)
    rilett = torch.load(d / "con.pt", weights_only=True)
    obj = FS._ricrea_ternario(rilett["gu"], "cpu")
    try:
        R._controlla_adam(obj, getattr(obj, R.CHIAVE_ADAM), "riletto")
        accettato = True
    except RuntimeError:
        accettato = False
    ok("R8f: _ricrea_ternario rimette lo stato sull'oggetto (valori identici al bit) e il raccordo lo accetta",
       _uguali_al_bit(getattr(obj, R.CHIAVE_ADAM), adam["gu"]) and accettato)
    senza = torch.load(d / "senza.pt", weights_only=True)
    obj2 = FS._ricrea_ternario(senza["gu"], "cpu")
    ok("R8f: senza momenti: nel file nessuna chiave 'adam' e sull'oggetto nessun attributo (default come prima)",
       getattr(obj2, R.CHIAVE_ADAM, None) is None and "adam" not in senza["gu"] and "adam" not in senza["dw"])
    shutil.rmtree(d)
    # scala Thor: E=256 — parametri contati su oggetti 'meta' (forme vere, zero memoria)
    thor = {"gu": TDBF(torch.empty(256, 1024, 2048, device="meta"), group=128),
            "dw": TDBF(torch.empty(256, 2048, 512, device="meta"), group=256)}
    n_thor = {f: sum(getattr(o, n).numel() for n in rc.NOMI_PARAMETRI) for f, o in thor.items()}
    tot = sum(n_thor.values())
    agg = (b_con - b_senza) / n_par * tot           # byte misurati per parametro, scalati a Thor
    print(f"    (misura) Thor, uno strato portato: {tot:,} parametri fp32 (gu {n_thor['gu']:,} · dw {n_thor['dw']:,}) "
          f"= {4 * tot / 2**30:.3f} GiB; momenti exp_avg+exp_avg_sq +{agg / 2**30:.3f} GiB "
          f"({(b_con - b_senza) / n_par:.4f} byte/parametro misurati)")
    print(f"    (misura) punto di ripresa W=2 (1 strato portato): 5,10 GiB misurati (conf6b_S1.log) -> "
          f"{5.10 + agg / 2**30:.2f} GiB con conserva; alla transizione coesistono i punti K e K+1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ricetta", choices=sorted(RICETTE), default="base")
    ap.add_argument("--largo", action="store_true", help="solo R7: strati alla larghezza vera (2048/512)")
    ap.add_argument("--senza-patch-A", default=None, help="copia con il SOLO raccordo (per il rifiuto)")
    ap.add_argument("--adam", action="store_true", help="solo R1 + R8 (THOR_SCORREVOLE_ADAM=conserva)")
    ap.add_argument("--cartella", default=None)
    args = ap.parse_args()
    if os.environ.get("THOR_SCORREVOLE_ADAM") is not None:
        raise SystemExit("THOR_SCORREVOLE_ADAM impostata nell'ambiente: il collaudo la gestisce da se' (default = riparti)")
    os.environ.update(RICETTE[args.ricetta])        # PRIMA di importare valida (costanti di modulo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    assert not torch.cuda.is_available(), "solo CPU"
    import valida_finestra_8_12 as V
    import forgia_scorrevole as FS
    import raccordo_scorrevole as R
    import ripartenza_codici as rc
    import test_forgia_scorrevole as TB
    from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso as TDBF
    base = Path(args.cartella) if args.cartella else Path(tempfile.mkdtemp(prefix="test_raccordo_"))
    if base.exists() and args.cartella:
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(0)
    print(f"ricetta {args.ricetta}: " + (" ".join(f"{k}={v}" for k, v in RICETTE[args.ricetta].items()) or "default"))
    r1_interfaccia(FS, R, V, args.senza_patch_A)
    dt_largo = torch.bfloat16 if args.ricetta == "produzione" else torch.float32

    def r8_tutti():
        if args.ricetta == "base":
            r8a_passaggio(TB, R, V, rc, torch.float32, "fp32", segno_atteso=True)
            r8b_equivalenza(TB, R, V, rc)
            r8c_ripresa(TB, FS, R, V, rc, base)
            r8d_errori(TB, R, V, rc)
            r8e_catena(TB, FS, R, V, rc, TDBF, base, torch.float32, 1e-5, "fp32")
            r8f_peso(FS, R, rc, base)
        else:
            r8a_passaggio(TB, R, V, rc, torch.bfloat16, "bf16-produzione", segno_atteso=False)
            r8e_catena(TB, FS, R, V, rc, TDBF, base, torch.bfloat16, 5e-2, "bf16-produzione")
    if args.adam:
        r8_tutti()
    elif args.largo:
        r7_largo(TB, FS, R, V, rc, TDBF, base, dt_largo, 5e-2 if dt_largo == torch.bfloat16 else 1e-5)
    elif args.ricetta == "base":
        TB.t7_contratto_A(R.ottimizza_finestra_JOINT_B, "RACCORDO + patch A")
        sorgente, _ = r2_catena(TB, FS, R, V, rc, TDBF, base, torch.float32, 1e-5, "fp32")
        r5_codici(TB, FS, R, V, rc, TDBF, base, sorgente, torch.float32, 1e-5, "fp32")
        TB.t4_ripresa(base, ottimizza_fn=R.ottimizza_finestra_JOINT_B, etichetta="raccordo")
        # teacher catturato vs ricalcolato, con l'interfaccia vera
        gioc = TB.Giocattolo(10)
        TB.corsa(base / "r6", gioc, riusa_teacher=False, ottimizza_fn=R.ottimizza_finestra_JOINT_B)
        ok("R6: teacher ricalcolato = teacher catturato (codici identici), interfaccia vera",
           all(TB.uguali(a, b) for a, b in zip(TB.codici_su_disco(sorgente, 10, "prova").values(),
                                                 TB.codici_su_disco(base / "r6", 10, "prova").values())))
        r4_guasti(TB, R, V, rc, TDBF)
        r8_tutti()
    else:
        sorgente, _ = r2_catena(TB, FS, R, V, rc, TDBF, base, torch.bfloat16, 5e-2, "bf16-produzione")
        r5_codici(TB, FS, R, V, rc, TDBF, base, sorgente, torch.bfloat16, 5e-2, "bf16-produzione")
        r8_tutti()
    tutti = ESITI + TB.ESITI
    falliti = [n for n, o in tutti if not o]
    print(f"\n{'✅ TUTTI' if not falliti else '⛔ FALLITI ' + str(len(falliti)) + ' su'} {len(tutti)} controlli "
          f"(ricetta {args.ricetta}) in {time.time()-t0:.0f}s" + ("" if not falliti else ": " + "; ".join(falliti)))
    if not args.cartella:
        shutil.rmtree(base, ignore_errors=True)
    return 0 if not falliti else 1


if __name__ == "__main__":
    raise SystemExit(main())
