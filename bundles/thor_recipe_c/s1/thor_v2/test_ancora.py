#!/usr/bin/env python3
"""Collaudo CPU dell'ANCORA (THOR_ANCORA_RAPPORTO / THOR_ANCORA_A) e del held-out di CONOSCENZA
(THOR_HELDOUT_CONOSCENZA) della forgia scorrevole — 26/9/2026.

Uso (regola di sessione: la GPU e' della forgia, qui solo CPU):
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 test_ancora.py [--verboso] [--tieni]

Cartella di lavoro: @@ROOT:models@@/gguf/tony-tern/test_ancora/<data>-<uuid>/ (MAI /tmp: e' RAM, e la cache
per lotto la rifiuta), rimossa a fine collaudo (--tieni la lascia). File giocattolo: pochi KiB.

Ogni corsa gira in un PROCESSO FIGLIO con la ricetta nell'ambiente PRIMA di importare valida (costanti
di modulo), su due ricette: "base" (fase morbida + dura, fp32) e "produzione" (solo fase dura STE,
rotazione sull'ingresso, nucleo bf16, routing aggiornato, bf16: quella di lancia_forgia_scorrevole.sh).
Forgia giocattolo: 5 strati Qwen3.5-MoE VERI a dimensioni giocattolo, W=2 (4 finestre), 6 lotti (5 di
addestramento + 1 held-out), 9 passi, interfaccia VERA (raccordo_scorrevole + ottimizza_finestra_JOINT).

(a) SPENTO = c316fe3, AL BIT. Il figlio "vecchio" importa i moduli da un'estrazione di c316fe3
    (`git archive`, cartella sua): e' il codice di prima, non una sua imitazione. Codici su disco,
    perdite di ogni passo, held-out, semi di ogni finestra, sha1 del giornale e il LOG (senza orari e
    tempi) identici fra vecchio, nuovo senza variabili e nuovo con THOR_ANCORA_RAPPORTO=0 (con
    THOR_ANCORA_A=init, che a r=0 non deve contare); nessun passo riceve il gancio dopo_backward.
(b) ANCORA r > 0 (teacher): per ogni finestra λ = r·L0/A0 al passo 0 (passo_lambda 0, L0 = perdita
    dati derivata del passo 0, A0 = ancora al passo 0), termine d'ancora del passo 0 = r·L0; A0 della
    finestra 0 = ricalcolo indipendente (teacher + init LS rifatti qui, ancora a tensore intero);
    ancora FINALE = ricalcolo dai codici su disco; codici diversi da r = 0. Ancora finale PIU' BASSA
    che con r = 0 in ogni finestra: PRETESA nella ricetta base (fase morbida: 0,21-0,58x misurato);
    nella ricetta di produzione (solo fase dura STE + AdamW) MISURATA e stampata, non pretesa: il
    26/9 e' risultata falsa alla finestra 0 del giocattolo (0,663 contro 0,603, uguale per r da 0,1 a
    1000) e la causa e' isolata — dall'init LS la discesa della SOLA ancora col gradiente STE fa
    SALIRE l'ancora dura (AdamW lr 0,01 a larghezza vera: 0,1874 -> 0,1974 in 9 passi). E' un
    risultato sul METODO (L2 nella perdita + Adam + STE, cfr. Loshchilov-Hutter), non un difetto
    del codice: il gradiente del gancio e' verificato esatto qui sotto. Gradiente: il gancio somma nei .grad ESATTAMENTE il gradiente
    di (dati + λ·ancora) calcolato con autograd sul tensore intero (blocchi da 1 e da 32 esperti, fase
    dura e morbida, larghezza vera con rotazione attiva). init: con passi morbidi funziona (A0 > 0),
    con la fase dura dal passo 0 RIFIUTATA prima di caricare uno strato. Variabili non valide rifiutate.
    Ricetta: THOR_ANCORA_* entrano nella ricetta registrata (ripresa con r diverso RIFIUTATA).
(c) HELD-OUT DI CONOSCENZA: con 2 lotti di conoscenza in coda i codici, le perdite di ogni passo,
    il held-out di sempre e i semi dei lotti comuni sono IDENTICI alla corsa senza (anche con l'ancora
    accesa: stesso λ, stessi codici; anche a cache per lotto accesa, budget 0 = tutto dal deposito);
    nessun passo usa un lotto di conoscenza; una riga «held-out CONOSCENZA» a init e a fine per
    finestra; con 1 lotto uguale al lotto held-out le misure CONOSCENZA = held-out, al bit. Fuori
    ricetta. Ripresa con i lotti di conoscenza spenti su un punto scritto accesi RIFIUTATA; ripresa
    accesa = corsa intera. Caricatore: manifesto con sha diverso, L_SEQ diversa, troppi lotti, file
    assente RIFIUTATI.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

RADICE_COLLAUDO = Path("@@ROOT:models@@/gguf/tony-tern/test_ancora")
QUI = Path(__file__).resolve().parent
REPO = QUI.parent
COMMIT_RIF = "c316fe3"
PY = sys.executable
N_STRATI, W, N_LOTTI, N_PASSI = 5, 2, 6, 9
R_PROVA = 1.0
ESITI: list = []
MISURE: list = []         # misure da leggere (non requisiti): stampate anche in fondo


def ok(nome: str, cond, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(cond)))
    print(f"    {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(cond)


@contextlib.contextmanager
def env(**kv):
    prima = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in prima.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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


# ═══════════════════════════════════ FIGLIO: una ricetta, un codice ═══════════════════════════════════
class Ctx:
    """I moduli importati dal figlio (dalla radice scelta: estrazione di c316fe3 o albero di lavoro)."""

    def __init__(self, spec: dict):
        import torch
        import valida_finestra_8_12 as V
        import forgia_scorrevole as FS
        import raccordo_scorrevole as R
        import test_forgia_scorrevole as TB
        self.torch, self.V, self.FS, self.R, self.TB = torch, V, FS, R, TB
        self.spec = spec
        self.base = Path(spec["cartella"])
        self.dtype = torch.bfloat16 if spec["ricetta"] == "produzione" else torch.float32
        self.verboso = spec.get("verboso", False)
        self.S = torch.from_numpy(V._S1024_NP)

    def giocattolo(self, seme_contenuto: bool = False):
        """`seme_contenuto`: il rumore del seme della finestra 0 dipende dai TOKEN del lotto e non dal suo
        indice (come in produzione, dove il seme e' l'embedding quantizzato dei token): due lotti con gli
        stessi token hanno lo stesso seme — serve alla prova di uguaglianza esatta CONOSCENZA = held-out."""
        torch, TB = self.torch, self.TB

        class GiocattoloLotti(TB.Giocattolo):
            """Il Giocattolo di test_forgia_scorrevole con 6 lotti (5 + 1 held-out), come test_cache_lotti."""

            def __init__(self, n, dtype):
                super().__init__(n, dtype)
                g = torch.Generator().manual_seed(11)
                self.ids = torch.randint(0, self.cfg.vocab_size, (N_LOTTI, 2, 6), generator=g)

            def batch(self):
                return self.ids.clone(), N_LOTTI - 1

            if seme_contenuto:
                def seme0(self, ids_batch, emb):
                    h, e = {}, {}
                    for b in range(ids_batch.shape[0]):
                        h[b] = torch.nn.functional.embedding(ids_batch[b], emb).to(emb.dtype)
                        g = torch.Generator().manual_seed(500 + int(ids_batch[b].sum()))
                        e[b] = (torch.randn(h[b].shape, generator=g) * 0.05).float()
                    return h, e
        return GiocattoloLotti(N_STRATI, self.dtype)


def corsa(ctx: Ctx, nome: str, variabili: "dict | None" = None, asincrona: bool = True, interrompi=None,
          cartella: "Path | None" = None, pulisci: bool = True, corsa_id: str = "tan", seme_contenuto: bool = False,
          **kw) -> dict:
    """Una forgia giocattolo; ritorna tutto cio' che si confronta. `variabili`: ambiente in piu' (None =
    togli). `kw` vanno a FS.esegui (solo il codice nuovo li conosce: heldout_conoscenza, ...)."""
    torch, V, FS, R, TB = ctx.torch, ctx.V, ctx.FS, ctx.R, ctx.TB
    cartella = cartella or (ctx.base / f"corsa_{nome}")
    gioc = ctx.giocattolo(seme_contenuto)
    diag: list = []
    passi: list = []

    @functools.wraps(R.ottimizza_finestra_JOINT_B)      # la firma (interfaccia A) resta visibile a verifica_interfaccia_A
    def ottimizza(*a, **k):
        d: list = []
        ris = R.ottimizza_finestra_JOINT_B(*a, raccogli_diagnostica=d, **k)
        diag.extend(d)
        return ris
    vero_passo = V.passo_ottimizzazione

    def spia_passo(tf, qf, h0, opt, e0=None, **k):
        passi.append({"h0": h0.detach().clone(), "gancio": "dopo_backward" in k})
        return vero_passo(tf, qf, h0, opt, e0, **k)
    buf = io.StringIO()
    V.passo_ottimizzazione = spia_passo
    errore = None
    try:
        with env(**(variabili or {})), contextlib.redirect_stdout(_Tee(sys.stdout, buf) if ctx.verboso else buf):
            oss = TB.corsa(cartella, gioc, n_passi=N_PASSI, w=W, corsa_id=corsa_id, asincrona=asincrona,
                           interrompi=interrompi, ottimizza_fn=ottimizza, **kw)
    except BaseException as e:   # noqa: BLE001 — il chiamante decide se l'errore era atteso
        errore = e
        oss = None
    finally:
        V.passo_ottimizzazione = vero_passo
    # il percorso della corsa cambia fra figlio vecchio e nuovo (cartelle diverse): nel log diventa <CORSA>
    reg = {"log": buf.getvalue().replace(str(cartella), "<CORSA>"), "diag": diag, "passi": passi, "caricati": [list(c) for c in gioc.caricati],
           "errore": None if errore is None else (type(errore).__name__, str(errore)), "gioc": gioc,
           "cartella": cartella}
    if errore is None:
        reg["codici"] = TB.codici_su_disco(cartella, N_STRATI, corsa_id)
        reg["risultati"] = oss.risultati
        reg["semi"] = oss.semi
        reg["giornale"] = [(g["strato"], g["sha1_Q"]) for g in
                           (json.loads(r) for r in (cartella / "out" / "giornale_scorrevole.jsonl").read_text()
                            .splitlines())]
    if pulisci and errore is None:
        shutil.rmtree(cartella, ignore_errors=True)
    return reg


def serializzabile(reg: dict) -> dict:
    return {k: v for k, v in reg.items() if k not in ("gioc", "cartella")}


# ── ancora ricalcolata a parte (tensore intero, nessuna riga di ancora_forgia) ──
def _A(deq, W) -> float:
    d = (deq.double() - W.double())
    return float((d * d).sum() / (W.double() * W.double()).sum())


def ancora_da_codici(ctx: Ctx, gioc, codici: dict) -> dict:
    """A per strato e famiglia dai CODICI (D x Q, scale fp32), contro il teacher del giocattolo nello spazio
    della forgia (gate/up ruotato con la funzione della forgia, down diretto)."""
    V, FS = ctx.V, ctx.FS
    out = {}
    for L, cod in codici.items():
        t = gioc.strato(L, False).mlp.experts
        Wgu = V._ruota_se_possibile(t.gate_up_proj.detach().float(), ctx.S)
        D, Q = FS._codici_gate_up_fusi(cod)
        out[f"L{L}.gu"] = _A(V.pesi_da_DQ(D, Q, "cpu", scala_fp16=False), Wgu)
        if "ffn_down_exps" in cod:
            out[f"L{L}.dw"] = _A(V.pesi_da_DQ(*cod["ffn_down_exps"], "cpu", scala_fp16=False),
                                 t.down_proj.detach().float())
    return out


def ancora_init_finestra0(ctx: Ctx, gioc) -> dict:
    """A0 della finestra 0 rifatto qui: strati 0 e 1 entrano dal teacher + init LS (come controlli_catena di
    test_forgia_scorrevole), ricostruzione del passo 0 (morbida a beta_0 se la ricetta ha passi morbidi,
    dura dai codici altrimenti) contro il teacher, tensore intero."""
    torch, V = ctx.torch, ctx.V
    from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso
    n_morbida = N_PASSI - V.passi_duri(N_PASSI)
    out = {}
    for L in (0, 1):
        t = gioc.strato(L, False).mlp.experts
        for fam, Wt, g in (("gu", V._ruota_se_possibile(t.gate_up_proj.detach().float(), ctx.S),
                            V._group_giocattolo(gioc.cfg.hidden_size, V.GRUPPO_GATEUP_REALE)),
                           ("dw", t.down_proj.detach().float(),
                            V._group_giocattolo(gioc.cfg.moe_intermediate_size, V.GRUPPO_DOWN_REALE))):
            obj = TernarioDiffBatchFuso(Wt, group=g)
            V.reinit_ls_(obj, Wt)
            with torch.no_grad():
                if n_morbida > 0:
                    beta0 = V.schedula_beta(1.0, V.NUCLEO_BETA_MAX, max(1, n_morbida))[0]
                    deq = obj.forward(beta0)
                else:
                    deq = V.pesi_da_DQ(*V._estrai_famiglia(obj), "cpu", scala_fp16=False)
            out[f"L{L}.{fam}"] = _A(deq, Wt)
    return out


def scenario_gradiente(ctx: Ctx) -> dict:
    """Il gancio somma nei .grad il gradiente di λ·A: confronto con autograd su (dati + λ·A) a tensore
    intero. Scala giocattolo e larghezza vera (hidden 2048: rotazione attiva, gruppi 128/256)."""
    import ancora_forgia as AF
    torch, V = ctx.torch, ctx.V
    from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso
    esiti = []
    casi = (("giocattolo", 4, 6, 16, 8, 16, 8, 8),      # nome, E, righe gu, n_in gu, gruppo gu, righe dw, n_in dw, gruppo dw
            ("larghezza vera", 2, 4, 2048, 128, 8, 512, 256))
    for nome, E, r_gu, n_gu, g_gu, r_dw, n_dw, g_dw in casi:
        g = torch.Generator().manual_seed(21)
        W_gu = (torch.randn(E, r_gu, n_gu, generator=g) * 0.02).to(torch.bfloat16)   # teacher bf16 come Thor
        W_dw = (torch.randn(E, r_dw, n_dw, generator=g) * 0.02).to(torch.bfloat16)
        ruota = (lambda X: V._ruota_se_possibile(X, ctx.S))
        Wr_gu = ruota(W_gu.float())
        x_gu = torch.randn(n_gu, 3, generator=g)
        x_dw = torch.randn(n_dw, 3, generator=g)
        for duro in (True, False):
            for blocco in (1, AF.BLOCCO_ESPERTI):
                o_gu = TernarioDiffBatchFuso(Wr_gu, group=g_gu)
                V.reinit_ls_(o_gu, Wr_gu)
                o_dw = TernarioDiffBatchFuso(W_dw.float(), group=g_dw)
                V.reinit_ls_(o_dw, W_dw.float())
                with torch.no_grad():
                    for o in (o_gu, o_dw):
                        o.z.add_(torch.randn(o.z.shape, generator=g) * 0.3)
                        o.log_s.add_(torch.randn(o.log_s.shape, generator=g) * 0.1)
                beta = 16.0 if duro else 3.0

                def dati():
                    f = (lambda o: o.forward_duro(beta)) if duro else (lambda o: o.forward(beta))
                    return (f(o_gu) @ x_gu).pow(2).sum() + (f(o_dw) @ x_dw).pow(2).sum()
                parametri = [p for o in (o_gu, o_dw) for p in o.parameters()]
                # 1) il gancio, come in passo_ottimizzazione: backward dei dati, poi dopo_backward
                vecchio = AF.BLOCCO_ESPERTI
                AF.BLOCCO_ESPERTI = blocco
                try:
                    for p in parametri:
                        p.grad = None
                    Ld = dati()
                    Ld.backward()
                    anc = AF.Ancora(0.7, "teacher", [0], {0: o_gu}, {0: o_dw}, {0: (W_gu, W_dw)}, ruota,
                                    log=lambda *a: None)
                    info = anc.gancio(0, beta, duro)(Ld.detach())
                finally:
                    AF.BLOCCO_ESPERTI = vecchio
                g_gancio = [p.grad.clone() for p in parametri]
                # 2) riferimento: autograd sulla perdita TOTALE a tensore intero, stesso λ
                for p in parametri:
                    p.grad = None
                f = (lambda o: o.forward_duro(beta)) if duro else (lambda o: o.forward(beta))
                A_rif = (((f(o_gu).double() - Wr_gu.double()) ** 2).sum() / (Wr_gu.double() ** 2).sum()
                         + ((f(o_dw).double() - W_dw.double()) ** 2).sum() / (W_dw.double() ** 2).sum())
                (dati() + anc.lam * A_rif).backward()
                g_rif = [p.grad.clone() for p in parametri]
                scarto = max(float(((a - b).abs().max() / b.abs().max().clamp_min(1e-30))) for a, b in zip(g_gancio, g_rif))
                esiti.append({"caso": nome, "duro": duro, "blocco": blocco, "scarto_grad": scarto,
                              "A0": anc.A0, "A_rif": float(A_rif), "L0": anc.L0, "lam": anc.lam, "r": anc.r,
                              "termine": info["termine"], "ancora": info["ancora"]})
    return {"gradiente": esiti}


def scenario_rifiuti_caricatore(ctx: Ctx, file_ok: Path) -> dict:
    torch, FS = ctx.torch, ctx.FS
    d = ctx.base / "caricatore"
    d.mkdir(parents=True, exist_ok=True)
    esiti = {}

    def prova(nome, fn):
        try:
            fn()
            esiti[nome] = ("nessuna eccezione", "")
        except Exception as e:  # noqa: BLE001
            esiti[nome] = (type(e).__name__, str(e))
    ids = torch.load(file_ok, weights_only=True)
    sha_sbagliato = d / "sha_sbagliato.pt"
    torch.save(ids, sha_sbagliato)
    m = json.loads(file_ok.with_suffix(".json").read_text())
    m["sha256_token"] = "0" * 64
    sha_sbagliato.with_suffix(".json").write_text(json.dumps(m))
    prova("sha", lambda: FS.carica_heldout_conoscenza(str(sha_sbagliato), 1, 2, 6))
    lungo = d / "lseq7.pt"
    torch.save(torch.zeros(6, 7, dtype=torch.int64), lungo)
    prova("lseq", lambda: FS.carica_heldout_conoscenza(str(lungo), 1, 2, 6))
    prova("troppi", lambda: FS.carica_heldout_conoscenza(str(file_ok), 4, 2, 6))
    prova("assente", lambda: FS.carica_heldout_conoscenza(str(d / "non_esiste.pt"), 1, 2, 6))
    buono = FS.carica_heldout_conoscenza(str(file_ok), 3, 2, 6)
    esiti["buono_forma"] = tuple(buono.shape)
    esiti["buono_uguale"] = bool(torch.equal(buono.reshape(6, 6), ids))
    return esiti


def prepara_conoscenza_giocattolo(ctx: Ctx) -> "tuple[Path, Path]":
    """File di conoscenza giocattolo (n_seq 6 x L_SEQ 6): righe 0-1 = il lotto held-out della corsa (lotto 5),
    righe 2-5 casuali. Due file: col manifesto (sha256 dei token come prepara_conoscenza.py) e senza."""
    import hashlib
    torch = ctx.torch
    gioc = ctx.giocattolo()
    g = torch.Generator().manual_seed(13)
    ids = torch.cat([gioc.ids[N_LOTTI - 1].clone(), torch.randint(0, gioc.cfg.vocab_size, (4, 6), generator=g)])
    d = ctx.base / "conoscenza"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "heldout_conoscenza.pt"
    torch.save(ids, p)
    sha = hashlib.sha256(ids.contiguous().numpy().astype("<i8").tobytes()).hexdigest()
    p.with_suffix(".json").write_text(json.dumps({"sha256_token": sha, "L_SEQ": 6}))
    return p, ids


def figlio(spec: dict) -> None:
    radice = Path(spec["radice"]).resolve()
    # i moduli di thor_v2 vengono SOLO dalla radice scelta (l'albero di lavoro NON deve comparire per il vecchio)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != QUI]
    sys.path.insert(0, str(radice))
    os.environ.update(spec["env"])                 # PRIMA di importare valida (costanti di modulo)
    import torch
    assert not torch.cuda.is_available(), "solo CPU (CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=)"
    torch.manual_seed(0)
    ctx = Ctx(spec)
    assert Path(ctx.V.__file__).resolve().parent == radice and Path(ctx.FS.__file__).resolve().parent == radice, \
        f"moduli importati da {ctx.V.__file__}, attesi da {radice}"
    ctx.base.mkdir(parents=True, exist_ok=True)
    uscita: dict = {"radice": str(radice), "ha_passi_duri": hasattr(ctx.V, "passi_duri")}
    for s in spec["scenari"]:
        t0 = time.time()
        print(f"  [figlio {spec['nome']}] scenario {s} ...", flush=True)
        if s == "spenta":
            reg = corsa(ctx, s, {"THOR_ANCORA_RAPPORTO": None, "THOR_ANCORA_A": None})
            if spec.get("ancora_indipendente") and reg["errore"] is None:
                # ancora ricalcolata dai codici della corsa SENZA ancora (il riferimento r = 0 della prova (b))
                reg["A_indip"] = {k: ancora_da_codici(ctx, reg["gioc"], dict(reg["risultati"][k]))
                                  for k in reg["risultati"]}
            uscita[s] = serializzabile(reg)
        elif s == "zero":
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_ANCORA_RAPPORTO": "0", "THOR_ANCORA_A": "init"}))
        elif s == "ancora":
            reg = corsa(ctx, s, {"THOR_ANCORA_RAPPORTO": str(R_PROVA), "THOR_ANCORA_A": "teacher"}, pulisci=False)
            if reg["errore"] is None:
                reg["A_indip"] = {k: ancora_da_codici(ctx, reg["gioc"], {L: reg["risultati"][k][L] for L in
                                                                         reg["risultati"][k]})
                                  for k in reg["risultati"]}
                reg["A0_indip_f0"] = ancora_init_finestra0(ctx, reg["gioc"])
                reg["ricetta_json"] = json.loads((reg["cartella"] / "out" / "ricetta_scorrevole.json").read_text())
                shutil.rmtree(reg["cartella"], ignore_errors=True)
            uscita[s] = serializzabile(reg)
        elif s == "init":
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_ANCORA_RAPPORTO": str(R_PROVA), "THOR_ANCORA_A": "init"}))
        elif s == "non_validi":
            uscita[s] = {n: serializzabile(corsa(ctx, f"nv_{n}", v)) for n, v in (
                ("r_negativo", {"THOR_ANCORA_RAPPORTO": "-1"}), ("r_testo", {"THOR_ANCORA_RAPPORTO": "abc"}),
                ("r_nan", {"THOR_ANCORA_RAPPORTO": "nan"}),
                ("rif", {"THOR_ANCORA_RAPPORTO": "1", "THOR_ANCORA_A": "pesi"}))}
        elif s == "ricetta":
            with env(THOR_ANCORA_RAPPORTO="1", THOR_ANCORA_A="teacher", THOR_HELDOUT_CONOSCENZA="x.pt",
                     THOR_HELDOUT_CONOSCENZA_LOTTI="2"):
                ric = ctx.FS._ricetta()
            cart = ctx.base / "ripresa_ricetta"

            def ferma(p, k):
                if p == "prima_transizione" and k == 1:
                    raise ctx.FS.InterruzioneSimulata("stop")
            prima = corsa(ctx, "ric1", {"THOR_ANCORA_RAPPORTO": "1"}, asincrona=False, interrompi=ferma,
                          cartella=cart, pulisci=False)
            dopo = corsa(ctx, "ric2", {"THOR_ANCORA_RAPPORTO": "2"}, asincrona=False, cartella=cart, pulisci=False)
            shutil.rmtree(cart, ignore_errors=True)
            uscita[s] = {"ricetta": ric, "interrotta": prima["errore"], "ripresa_r2": dopo["errore"]}
        elif s.startswith("conoscenza"):
            p, ids_con = prepara_conoscenza_giocattolo(ctx)
            uscita["ids_conoscenza"] = ids_con
            if s == "conoscenza":
                uscita[s] = serializzabile(corsa(ctx, s, {"THOR_ANCORA_RAPPORTO": None},
                                                 heldout_conoscenza=str(p), lotti_conoscenza=2))
            elif s == "conoscenza_1":
                uscita[s] = serializzabile(corsa(ctx, s, {"THOR_ANCORA_RAPPORTO": None}, seme_contenuto=True,
                                                 heldout_conoscenza=str(p), lotti_conoscenza=1))
            elif s == "conoscenza_ancora":
                uscita[s] = serializzabile(corsa(ctx, s, {"THOR_ANCORA_RAPPORTO": str(R_PROVA), "THOR_ANCORA_A": "teacher"},
                                                 heldout_conoscenza=str(p), lotti_conoscenza=2))
            elif s == "conoscenza_cache":
                uscita[s] = serializzabile(corsa(
                    ctx, s, {"THOR_ANCORA_RAPPORTO": None, "THOR_CACHE_LOTTI": "1", "THOR_CACHE_LOTTI_GPU_GIB": "0",
                             "THOR_CACHE_LOTTI_DIR": str(ctx.base / "cache_lotti")},
                    heldout_conoscenza=str(p), lotti_conoscenza=2))
                uscita[s]["cache_vuota"] = not any((ctx.base / "cache_lotti").iterdir()) \
                    if (ctx.base / "cache_lotti").exists() else True
            elif s == "conoscenza_ripresa":
                cart = ctx.base / "ripresa_conoscenza"

                def ferma(pp, k):
                    if pp == "prima_transizione" and k == 1:
                        raise ctx.FS.InterruzioneSimulata("stop")
                prima = corsa(ctx, "cr1", {}, asincrona=False, interrompi=ferma, cartella=cart, pulisci=False,
                              heldout_conoscenza=str(p), lotti_conoscenza=2)
                spenta = corsa(ctx, "cr2", {}, asincrona=False, cartella=cart, pulisci=False)
                accesa = corsa(ctx, "cr3", {}, asincrona=False, cartella=cart, pulisci=False,
                               heldout_conoscenza=str(p), lotti_conoscenza=2)
                shutil.rmtree(cart, ignore_errors=True)
                uscita[s] = {"interrotta": prima["errore"], "ripresa_spenta": spenta["errore"],
                             "ripresa_accesa": serializzabile(accesa)}
            elif s == "conoscenza_caricatore":
                uscita[s] = scenario_rifiuti_caricatore(ctx, p)
        elif s == "gradiente":
            uscita[s] = scenario_gradiente(ctx)
        else:
            raise ValueError(f"scenario {s!r} sconosciuto")
        print(f"  [figlio {spec['nome']}] scenario {s} in {time.time() - t0:.0f}s", flush=True)
    torch.save(uscita, spec["uscita"])


# ═══════════════════════════════════ GENITORE: lancia e confronta ═══════════════════════════════════
def lancia_figlio(nome: str, radice: Path, ricetta: str, ricetta_env: dict, scenari: list, base: Path,
                  verboso: bool, ancora_indipendente: bool = False) -> dict:
    import torch
    uscita = base / f"esito_{nome}.pt"
    spec = {"nome": nome, "radice": str(radice), "ricetta": ricetta, "env": ricetta_env, "scenari": scenari,
            "cartella": str(base / nome), "uscita": str(uscita), "verboso": verboso,
            "ancora_indipendente": ancora_indipendente}
    ambiente = {k: v for k, v in os.environ.items() if not k.startswith("THOR_")}
    ambiente.update({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "THOR_CRONO_OGNI": "3"})
    t0 = time.time()
    p = subprocess.run([PY, str(QUI / "test_ancora.py"), "--figlio", json.dumps(spec)], env=ambiente,
                       cwd=str(base), capture_output=not verboso, text=True)
    if p.returncode != 0:
        print((p.stdout or "")[-4000:], (p.stderr or "")[-4000:])
        ok(f"figlio {nome}: terminato senza errori", False, f"rc {p.returncode}")
        return {}
    print(f"  figlio {nome} ({ricetta}, {len(scenari)} scenari) in {time.time() - t0:.0f}s", flush=True)
    return torch.load(uscita, weights_only=False)


_ORARIO = re.compile(r"^\[\d\d:\d\d:\d\d\] ", re.M)
_TEMPI = [(re.compile(r"cronometro [0-9.]+ s/passo"), "cronometro X s/passo"),
          (re.compile(r"FATTA in [0-9.]+s"), "FATTA in Xs"),
          (re.compile(r"finestre in [0-9.]+s"), "finestre in Xs"),
          (re.compile(r"\(batch 0, [0-9]+s\)"), "(batch 0, Xs)")]


# righe scritte dal THREAD di scrittura della forgia (scrittura asincrona): il loro posto fra le righe del thread
# principale dipende dai tempi, non dal codice (la prima esecuzione del collaudo e' passata per caso)
_ASINCRONE = re.compile(r"^(codici scritti in |  punto di ripresa finestra \d+ scritto |  impronta del punto di ripresa )")


def log_normalizzato(testo: str) -> str:
    t = _ORARIO.sub("", testo)
    for rx, sost in _TEMPI:
        t = rx.sub(sost, t)
    return t


def log_confrontabile(testo: str) -> "tuple[list, list]":
    """(righe del thread principale IN ORDINE, righe del thread di scrittura ORDINATE: stesso multinsieme)."""
    righe = log_normalizzato(testo).splitlines()
    return [r for r in righe if not _ASINCRONE.match(r)], sorted(r for r in righe if _ASINCRONE.match(r))


def _uguali_codici(a: dict, b: dict) -> bool:
    import numpy as np
    return sorted(a) == sorted(b) and all(
        sorted(a[L]) == sorted(b[L]) and all(np.array_equal(a[L][f][0], b[L][f][0]) and np.array_equal(a[L][f][1], b[L][f][1])
                                             for f in a[L]) for L in a)


def _perdite(reg: dict) -> list:
    return [(d["perdite"], d["heldout"]) for d in reg["diag"]]


def _semi_uguali(a: dict, b: dict, lotti: "list | None" = None) -> bool:
    import torch
    if sorted(a) != sorted(b):
        return False
    for k in a:
        for x, y in zip(a[k], b[k]):
            if (x is None) != (y is None):
                return False
            if x is None:
                continue
            chiavi = sorted(x) if lotti is None else lotti
            if lotti is None and sorted(x) != sorted(y):
                return False
            if not all(x[i].dtype == y[i].dtype and torch.equal(x[i], y[i]) for i in chiavi):
                return False
    return True


def confronta_identiche(etichetta: str, rif: dict, reg: dict, con_log: bool = True) -> None:
    ok(f"{etichetta}: nessun errore", rif.get("errore") is None and reg.get("errore") is None,
       f"{rif.get('errore')} / {reg.get('errore')}")
    if rif.get("errore") is not None or reg.get("errore") is not None:
        return
    ok(f"{etichetta}: codici su disco IDENTICI (tutti gli strati, D e Q)", _uguali_codici(rif["codici"], reg["codici"]))
    ok(f"{etichetta}: perdite di OGNI passo e held-out identiche (==)", _perdite(rif) == _perdite(reg),
       f"{len(reg['diag'])} finestre")
    ok(f"{etichetta}: semi di ogni finestra identici (torch.equal)", _semi_uguali(rif["semi"], reg["semi"]))
    ok(f"{etichetta}: sha1 dei codici nel giornale identici", rif["giornale"] == reg["giornale"])
    if con_log:
        (pa, sa), (pb, sb) = log_confrontabile(rif["log"]), log_confrontabile(reg["log"])
        diverse = [(x, y) for x, y in zip(pa, pb) if x != y]
        ok(f"{etichetta}: log identico (orari e tempi esclusi; thread principale riga per riga, righe del thread "
           f"di scrittura come multinsieme)", pa == pb and sa == sb and len(sa) > 0,
           f"{len(pa)} + {len(sa)} righe" if pa == pb and sa == sb else
           f"prima differenza: {diverse[:1]} · righe {len(pa)}+{len(sa)} vs {len(pb)}+{len(sb)}")


def parte_a(ricetta: str, vecchio: dict, nuovo: dict) -> None:
    print(f"\n── (a) [{ricetta}] spento = c316fe3 al bit ──")
    ok(f"(a) [{ricetta}] il figlio vecchio usa davvero i moduli di {COMMIT_RIF} (nessun passi_duri)",
       vecchio.get("ha_passi_duri") is False and nuovo.get("ha_passi_duri") is True, vecchio.get("radice", ""))
    confronta_identiche(f"(a) [{ricetta}] c316fe3 vs nuovo senza variabili", vecchio["spenta"], nuovo["spenta"])
    confronta_identiche(f"(a) [{ricetta}] c316fe3 vs nuovo con THOR_ANCORA_RAPPORTO=0 (A=init)", vecchio["spenta"],
                        nuovo["zero"])
    for nome in ("spenta", "zero"):
        reg = nuovo[nome]
        ok(f"(a) [{ricetta}] {nome}: nessun passo riceve il gancio dopo_backward",
           reg["passi"] and not any(p["gancio"] for p in reg["passi"]), f"{len(reg['passi'])} passi")
        segni = ("ANCORA accesa", "λ = r·L0/A0", "ancora FINALE", "ancora λ·A", "ancora (", "CONOSCENZA")
        trovate = [r for r in reg["log"].splitlines() if any(s in r for s in segni)]
        ok(f"(a) [{ricetta}] {nome}: nessuna riga di ancora o conoscenza nel log, nessuna chiave nuova nella diagnostica",
           not trovate and all(sorted(d) == ["chunk", "codici", "heldout", "perdite"] for d in reg["diag"]),
           "; ".join(trovate[:2]) or f"chiavi {sorted(reg['diag'][0]) if reg['diag'] else '-'}")


def parte_b(ricetta: str, nuovo: dict) -> None:
    print(f"\n── (b) [{ricetta}] ancora r = {R_PROVA} (teacher) ──")
    reg, zero = nuovo["ancora"], nuovo["spenta"]
    ok(f"(b) [{ricetta}] corsa con l'ancora senza errori", reg.get("errore") is None, str(reg.get("errore")))
    if reg.get("errore") is not None:
        return
    fin = len(reg["diag"])
    ok(f"(b) [{ricetta}] diagnostica dell'ancora in ognuna delle {fin} finestre",
       fin == N_STRATI - W + 1 and all("ancora" in d for d in reg["diag"]))
    for i, d in enumerate(reg["diag"]):
        a = d["ancora"]
        st = a["storia"]
        ok(f"(b) [{ricetta}] finestra {i}: λ = r·L0/A0 fissato al passo 0 (λ {a['lambda']:.4e}, L0 {a['L0']:.4e}, "
           f"A0 {a['A0']:.4e})",
           a["passo_lambda"] == 0 and a["r"] == R_PROVA and a["lambda"] == R_PROVA * a["L0"] / a["A0"]
           and st[0][0] == 0 and st[0][1] == a["L0"] and a["A0"] > 0)
        rel = abs(a["lambda"] * st[0][2] - R_PROVA * a["L0"]) / (R_PROVA * a["L0"])
        ok(f"(b) [{ricetta}] finestra {i}: al passo 0 il termine d'ancora λ·A della perdita = r·L0 (scarto {rel:.1e})",
           rel < 1e-12 and len(st) == N_PASSI)
        A_ind = sum(reg["A_indip"][i].values())
        rel_f = abs(a["A_fine"] - A_ind) / A_ind
        ok(f"(b) [{ricetta}] finestra {i}: ancora FINALE {a['A_fine']:.5e} = ricalcolo dai codici {A_ind:.5e} "
           f"(scarto {rel_f:.1e})", rel_f < 1e-5)
        A_zero = sum(zero["A_indip"][i].values())
        testo = (f"(b) [{ricetta}] finestra {i}: ancora finale con r={R_PROVA} {A_ind:.5e} vs r=0 {A_zero:.5e} "
                 f"({A_ind / A_zero:.4f}x; A0 {a['A0']:.5e})")
        if ricetta == "base":
            ok(testo.replace(" vs r=0", " < con r=0"), A_ind < A_zero)
        else:
            # ⚠️ 26/9: con la sola fase dura (STE) e AdamW NON e' una proprieta' del codice ma della dinamica, e sul
            # giocattolo e' risultata FALSA alla finestra 0 (0,663 contro 0,603; stessa cosa per r da 0,1 a 1000).
            # Causa isolata: dall'init LS la discesa della SOLA ancora col gradiente STE la fa SALIRE (AdamW lr 0,01,
            # larghezza vera: 0,1874 -> 0,1974 in 9 passi). Si misura e si stampa, non si pretende (docstring (b)).
            riga = f"📏 MISURA {testo} · {'piu\' bassa' if A_ind < A_zero else 'PIU\' ALTA'} che senza ancora"
            MISURE.append(riga)
            print("    " + riga, flush=True)
    a0 = reg["diag"][0]["ancora"]
    A0_ind = sum(reg["A0_indip_f0"].values())
    rel0 = abs(a0["A0"] - A0_ind) / A0_ind
    ok(f"(b) [{ricetta}] finestra 0: A0 {a0['A0']:.6e} = ricalcolo indipendente (teacher + init LS, tensore intero) "
       f"{A0_ind:.6e} (scarto {rel0:.1e})", rel0 < 1e-5)
    ok(f"(b) [{ricetta}] i codici con l'ancora sono DIVERSI da quelli senza", not _uguali_codici(reg["codici"], zero["codici"]))
    ok(f"(b) [{ricetta}] ogni passo riceve il gancio dopo_backward", all(p["gancio"] for p in reg["passi"])
       and len(reg["passi"]) == fin * N_PASSI)
    log = reg["log"]
    ok(f"(b) [{ricetta}] log: riga di accensione, λ per finestra, crono coi due termini, ancora FINALE per finestra",
       "ANCORA accesa: r = 1" in log and log.count("λ = r·L0/A0 =") == fin
       and re.search(r"cronometro .*\(loss [0-9.e+-]+\) · dati [0-9.e+-]+ · ancora λ·A [0-9.e+-]+", log) is not None
       and log.count("ancora FINALE") == fin)
    ric = reg["ricetta_json"]["ricetta"]
    ok(f"(b) [{ricetta}] THOR_ANCORA_* nella ricetta registrata (ricetta_scorrevole.json)",
       ric.get("THOR_ANCORA_RAPPORTO") == str(R_PROVA) and ric.get("THOR_ANCORA_A") == "teacher")


def parte_b_init(ricetta: str, nuovo: dict) -> None:
    reg = nuovo["init"]
    if ricetta == "produzione":
        e = reg.get("errore")
        ok(f"(b) [{ricetta}] THOR_ANCORA_A=init con la fase dura dal passo 0: RIFIUTATA prima di caricare uno strato",
           e is not None and e[0] == "ValueError" and "per costruzione" in e[1] and not reg["caricati"],
           str(e)[:160] if e else "nessun errore")
    else:
        ok(f"(b) [{ricetta}] THOR_ANCORA_A=init con passi morbidi: corsa completa, λ finito e A0 > 0 in ogni finestra",
           reg.get("errore") is None and all(math.isfinite(d["ancora"]["lambda"]) and d["ancora"]["A0"] > 0
                                             and d["ancora"]["riferimento"] == "init" for d in reg["diag"]),
           str(reg.get("errore"))[:160])


def parte_b_varie(ricetta: str, nuovo: dict) -> None:
    for nome, reg in nuovo["non_validi"].items():
        e = reg.get("errore")
        ok(f"(b) [{ricetta}] variabile non valida ({nome}) RIFIUTATA prima di caricare uno strato",
           e is not None and e[0] == "ValueError" and not reg["caricati"], str(e)[:120] if e else "nessun errore")
    r = nuovo["ricetta"]
    ok(f"(b) [{ricetta}] _ricetta(): THOR_ANCORA_* dentro, THOR_HELDOUT_CONOSCENZA* fuori",
       "THOR_ANCORA_RAPPORTO" in r["ricetta"] and "THOR_ANCORA_A" in r["ricetta"]
       and "THOR_HELDOUT_CONOSCENZA" not in r["ricetta"] and "THOR_HELDOUT_CONOSCENZA_LOTTI" not in r["ricetta"])
    ok(f"(b) [{ricetta}] ripresa con THOR_ANCORA_RAPPORTO cambiato (1 -> 2) RIFIUTATA",
       r["interrotta"] is not None and r["interrotta"][0] == "InterruzioneSimulata" and r["ripresa_r2"] is not None
       and "ricetta diversa" in r["ripresa_r2"][1] and "THOR_ANCORA_RAPPORTO" in r["ripresa_r2"][1],
       str(r["ripresa_r2"])[:160])
    for e in nuovo["gradiente"]["gradiente"]:
        ok(f"(b) [{ricetta}] gradiente del gancio = autograd di (dati + λ·A) a tensore intero · {e['caso']}, "
           f"{'dura' if e['duro'] else 'morbida'}, blocco {e['blocco']} (scarto relativo {e['scarto_grad']:.1e})",
           e["scarto_grad"] < 1e-5 and abs(e["A0"] - e["A_rif"]) / e["A_rif"] < 1e-5
           and abs(e["lam"] * e["ancora"] - e["r"] * e["L0"]) / (e["r"] * e["L0"]) < 1e-12)


def parte_c(ricetta: str, nuovo: dict) -> None:
    import torch
    print(f"\n── (c) [{ricetta}] held-out di CONOSCENZA ──")
    base, con = nuovo["spenta"], nuovo["conoscenza"]
    fin = N_STRATI - W + 1
    ok(f"(c) [{ricetta}] corse con la conoscenza senza errori",
       all(nuovo[s].get("errore") is None for s in ("conoscenza", "conoscenza_1", "conoscenza_ancora", "conoscenza_cache")),
       "; ".join(f"{s}: {nuovo[s].get('errore')}" for s in ("conoscenza", "conoscenza_1", "conoscenza_ancora",
                                                             "conoscenza_cache") if nuovo[s].get("errore")))
    if con.get("errore") is not None:
        return
    lotti_comuni = list(range(N_LOTTI))
    for etich, reg in (("2 lotti", con), ("2 lotti a cache accesa (budget 0)", nuovo["conoscenza_cache"])):
        ok(f"(c) [{ricetta}] {etich}: codici su disco IDENTICI alla corsa senza conoscenza",
           _uguali_codici(base["codici"], reg["codici"]))
        ok(f"(c) [{ricetta}] {etich}: perdite di ogni passo e held-out di sempre identici (==)", _perdite(base) == _perdite(reg))
        ok(f"(c) [{ricetta}] {etich}: semi dei lotti comuni identici, piu' {N_LOTTI + 2 - N_LOTTI} lotti in coda",
           _semi_uguali(base["semi"], reg["semi"], lotti_comuni)
           and all(sorted(reg["semi"][k][0]) == list(range(N_LOTTI + 2)) for k in reg["semi"]))
        ok(f"(c) [{ricetta}] {etich}: sha1 del giornale identici", base["giornale"] == reg["giornale"])
        ok(f"(c) [{ricetta}] {etich}: ogni passo t di ogni finestra usa il lotto t % {N_LOTTI - 1} (seme identico al "
           f"bit), mai un lotto di conoscenza", len(reg["passi"]) == fin * N_PASSI and all(
               torch.equal(reg["passi"][i * N_PASSI + t]["h0"], reg["semi"][i][0][t % (N_LOTTI - 1)])
               and not any(torch.equal(reg["passi"][i * N_PASSI + t]["h0"], reg["semi"][i][0][b])
                           for b in range(N_LOTTI, N_LOTTI + 2))
               for i in range(fin) for t in range(N_PASSI)))
        hc = [d.get("heldout_conoscenza") for d in reg["diag"]]
        ok(f"(c) [{ricetta}] {etich}: held-out CONOSCENZA misurato a init e a fine in ogni finestra (v20, init, fine finiti)",
           all(h is not None and all(math.isfinite(h[x]) for x in ("v20", "init", "fine")) for h in hc)
           and reg["log"].count("held-out CONOSCENZA (2 batch) loss DURA: v20") == fin
           and reg["log"].count("held-out CONOSCENZA loss DURA finale") == fin
           and re.search(r"held-out CONOSCENZA loss DURA finale [0-9.e+-]+ · rispetto a init [0-9.]+x · rispetto a v20 "
                         r"[0-9.]+x", reg["log"]) is not None)
    ok(f"(c) [{ricetta}] a cache accesa: misure CONOSCENZA identiche alla cache spenta, nessuna cartella di cache rimasta",
       [d["heldout_conoscenza"] for d in con["diag"]] == [d["heldout_conoscenza"] for d in nuovo["conoscenza_cache"]["diag"]]
       and nuovo["conoscenza_cache"].get("cache_vuota") is True)
    uno = nuovo["conoscenza_1"]
    ok(f"(c) [{ricetta}] 1 lotto di conoscenza = il lotto held-out: misure CONOSCENZA = held-out di sempre, al bit",
       all(d["heldout_conoscenza"] == d["heldout"] for d in uno["diag"]) and len(uno["diag"]) == fin,
       str(uno["diag"][0]["heldout_conoscenza"]) + " vs " + str(uno["diag"][0]["heldout"]))
    anc, anc_con = nuovo["ancora"], nuovo["conoscenza_ancora"]
    if anc.get("errore") is None and anc_con.get("errore") is None:
        ok(f"(c) [{ricetta}] con l'ancora accesa: codici, perdite e λ identici con e senza conoscenza (λ non la legge)",
           _uguali_codici(anc["codici"], anc_con["codici"]) and _perdite(anc) == _perdite(anc_con)
           and [d["ancora"]["lambda"] for d in anc["diag"]] == [d["ancora"]["lambda"] for d in anc_con["diag"]]
           and [d["ancora"]["storia"] for d in anc["diag"]] == [d["ancora"]["storia"] for d in anc_con["diag"]])
    rip = nuovo["conoscenza_ripresa"]
    ok(f"(c) [{ricetta}] ripresa con la conoscenza SPENTA su un punto scritto ACCESA: RIFIUTATA",
       rip["interrotta"] is not None and rip["interrotta"][0] == "InterruzioneSimulata"
       and rip["ripresa_spenta"] is not None and "lotti" in rip["ripresa_spenta"][1], str(rip["ripresa_spenta"])[:160])
    acc = rip["ripresa_accesa"]
    ok(f"(c) [{ricetta}] ripresa con la conoscenza accesa: codici IDENTICI alla corsa intera",
       acc.get("errore") is None and _uguali_codici(con["codici"], acc["codici"]), str(acc.get("errore"))[:120])
    car = nuovo["conoscenza_caricatore"]
    for nome, testo in (("sha", "sha256"), ("lseq", "L_SEQ"), ("troppi", "tetto"), ("assente", "assente")):
        ok(f"(c) [{ricetta}] caricatore: {nome} RIFIUTATO", car[nome][0] != "nessuna eccezione" and testo in car[nome][1],
           f"{car[nome][0]}: {car[nome][1][:100]}")
    ok(f"(c) [{ricetta}] caricatore: 3 lotti = le prime 6 sequenze del file, forma (3, 2, 6)",
       car["buono_forma"] == (3, 2, 6) and car["buono_uguale"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figlio", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--verboso", action="store_true", help="mostra il log delle corse giocattolo")
    ap.add_argument("--tieni", action="store_true", help="non cancellare la cartella di collaudo")
    ap.add_argument("--ricette", default="base,produzione")
    args = ap.parse_args()
    if args.figlio:
        figlio(json.loads(args.figlio))
        return 0
    for v in [k for k in os.environ if k.startswith(("THOR_ANCORA", "THOR_HELDOUT_CONOSCENZA", "THOR_CACHE_LOTTI"))]:
        raise SystemExit(f"{v} impostata nell'ambiente: il collaudo le gestisce da se'")
    sys.path.insert(0, str(QUI))
    import test_raccordo_scorrevole as TR
    base = RADICE_COLLAUDO / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True)
    t0 = time.time()
    print(f"cartella di collaudo {base}")
    try:
        # c316fe3 estratto da git: il codice di PRIMA, intero (tutti i moduli della cartella thor_v2)
        vecchio_dir = base / f"moduli_{COMMIT_RIF}"
        vecchio_dir.mkdir()
        # thor_v2 e kernel-rotazione (gptq_1024_lib la importa per percorso relativo, ../kernel-rotazione)
        arch = subprocess.run(["git", "-C", str(REPO), "archive", COMMIT_RIF, "thor_v2", "kernel-rotazione"],
                              capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(vecchio_dir)], input=arch.stdout, check=True)
        radice_vecchia = vecchio_dir / "thor_v2"
        ok(f"estrazione di {COMMIT_RIF}: valida_finestra_8_12.py presente e senza ancora_forgia",
           (radice_vecchia / "valida_finestra_8_12.py").exists()
           and "ancora_forgia" not in (radice_vecchia / "valida_finestra_8_12.py").read_text())
        scenari_nuovo = ["spenta", "zero", "ancora", "init", "non_validi", "ricetta", "conoscenza",
                         "conoscenza_1", "conoscenza_ancora", "conoscenza_cache", "conoscenza_ripresa",
                         "conoscenza_caricatore", "gradiente"]
        for ricetta in args.ricette.split(","):
            renv = dict(TR.RICETTE[ricetta])
            print(f"\n══ ricetta {ricetta}: " + (" ".join(f"{k}={v}" for k, v in renv.items()) or "default") + " ══")
            vecchio = lancia_figlio(f"vecchio_{ricetta}", radice_vecchia, ricetta, renv, ["spenta"], base, args.verboso)
            nuovo = lancia_figlio(f"nuovo_{ricetta}", QUI, ricetta, renv, scenari_nuovo, base, args.verboso,
                                  ancora_indipendente=True)
            if not vecchio or not nuovo:
                continue
            parte_a(ricetta, vecchio, nuovo)
            parte_b(ricetta, nuovo)
            parte_b_init(ricetta, nuovo)
            parte_b_varie(ricetta, nuovo)
            parte_c(ricetta, nuovo)
    finally:
        if not args.tieni:
            shutil.rmtree(base, ignore_errors=True)
    if MISURE:
        print("\nMISURE da leggere (non requisiti):\n" + "\n".join("  " + m for m in MISURE))
    falliti = [n for n, o in ESITI if not o]
    print(f"\n{'✅ TUTTI' if not falliti else '⛔ FALLITI ' + str(len(falliti)) + ' su'} {len(ESITI)} controlli "
          f"in {time.time() - t0:.0f}s" + ("" if not falliti else ": " + "; ".join(falliti)))
    return 0 if not falliti else 1


if __name__ == "__main__":
    raise SystemExit(main())
