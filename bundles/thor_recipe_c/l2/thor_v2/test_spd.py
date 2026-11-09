#!/usr/bin/env python3
"""Collaudo CPU della SELECTIVE PROJECTION DECAY (THOR_SPD_LAMBDA / THOR_SPD_SEMPRE / THOR_SPD_GRANULARITA) della
forgia scorrevole — 26/9/2026 sera. Perche' e come: docstring di spd_forgia.py.

Uso (la GPU e' della forgia, qui solo CPU; con una prova in corsa sulla GPU, confinato a tre livelli):
    systemd-run --user --scope -p CPUQuota=400% nice -n 19 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
        OPENBLAS_NUM_THREADS=4 THOR_TORCH_THREADS=4 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \\
        @@ROOT:venv@@/bin/python3 test_spd.py [--verboso] [--tieni] [--ricette base,produzione]

Cartella di lavoro: @@ROOT:models@@/gguf/tony-tern/test_spd/<data>-<uuid>/ (MAI /tmp: e' RAM), rimossa a fine collaudo
(--tieni la lascia). Stessa struttura di test_ancora.py: ogni corsa gira in un PROCESSO FIGLIO con la ricetta
nell'ambiente prima di importare valida; forgia giocattolo VERA (5 strati Qwen3.5-MoE a dimensioni giocattolo, W=2,
4 finestre, 6 lotti di cui 1 held-out, 9 passi, raccordo_scorrevole + ottimizza_finestra_JOINT), ricette "base"
(fase morbida + dura, fp32) e "produzione" (solo fase dura STE, rotazione sull'ingresso, nucleo bf16, routing
aggiornato: quella di lancia_forgia_scorrevole.sh). Una SPIA su passo_ottimizzazione fotografa per conto suo i
parametri dell'ottimizzatore al primo passo di ogni finestra e misura, dopo l'ultimo, distanza e codici duri
cambiati con formule sue (nessuna riga di spd_forgia): vale anche per la corsa senza SPD, che non fotografa nulla.

(a) SPENTA = 4b4553f AL BIT, ricetta base e produzione. Il figlio "vecchio" importa i moduli da un'estrazione di
    4b4553f (`git archive`): codici su disco, perdite di ogni passo, held-out, semi di ogni finestra, sha1 del
    giornale e LOG (senza orari e tempi) identici fra vecchio, nuovo senza variabili e nuovo con THOR_SPD_LAMBDA=0;
    nessun passo riceve `proiezione`; nessuna riga SPD nel log; diagnostica con le sole chiavi di prima.
(b) λ = 1, RICETTA DI PRODUZIONE, entrambe le granularita': in OGNI finestra la distanza finale dai latenti di
    partenza e' MINORE che con λ = 0 e i codici duri cambiati sono MENO (verifica: se e' falsa il test fallisce).
    Nella ricetta base la stessa cosa si misura e si stampa. Coerenza: distanza e codici della riga FINALE di SPD
    = misura indipendente della spia (scarto relativo < 1e-6, codici al conteggio esatto). Ogni passo riceve la
    proiezione; righe di log (accensione, per finestra, cronometro, FINALE); THOR_SPD_* nella ricetta registrata;
    la granularita' di default scritta esplicitamente da' codici IDENTICI a quella implicita.
    ⚠️ ESITO del 26/9 sera (prima esecuzione, lasciato com'e': e' una verifica, non una misura): la distanza scende in
    OGNI finestra (per esperto 0,85-0,92x, per tensore 0,92-0,97x), ma i codici cambiati NO alla finestra 1 — 29 contro
    29 per esperto, 31 contro 29 per tensore (finestre 0, 2, 3: meno). Quindi il test FALLISCE con 2 controlli. Con
    finestre da 40 passi (MISURA «lunghe») per esperto i codici scendono in ogni finestra (446 contro 480 in totale),
    per tensore no (finestra 0: 152 contro 149). Nella ricetta base, per esperto, meno in ogni finestra anche a 9 passi.
(c) FORMULA su tensori sintetici (ternari veri, un chunk, ottimizzatore finto: p~ imposto a mano): c, γ_prev, γ, r,
    k e i parametri proiettati coincidono con un calcolo indipendente in float64 (per esperto e per tensore, λ 0,5 /
    1 / 2, casi: c<0 con crescita, c<0 senza crescita -> r = 0, c>0, d = 0 -> c = 0, γ = 0 -> r = 0 senza NaN); dove
    k = 0 il parametro resta IDENTICO al bit; con λ = 1 la deviazione proiettata torna esattamente a γ_prev.
(d) THOR_SPD_SEMPRE: sui sintetici proietta anche dove c >= 0 (r dell'articolo); nella forgia giocattolo al primo
    passo di ogni finestra (c = 0) proietta con SEMPRE e mai senza; codici diversi da λ = 1 senza SEMPRE.
(e) AGGANCIO: con passo_ottimizzazione vero (finestra giocattolo di finestra_globale) `prima_del_passo` vede i
    parametri PRIMA del passo e i gradienti del backward, `dopo_il_passo` i parametri DOPO AdamW, al bit rispetto a
    una copia senza proiezione; micro-passo di accumulo e passo saltato (non finito) non la chiamano.
(f) RIPRESA con SPD accesa (revisione avversaria, 26/9 notte): θ0 NON si salva nel punto di ripresa perche' la
    ripresa e' solo a confine di finestra (Ripresa.punto_di_ripresa: si rifa' la finestra K intera, mai da meta'
    chunk) e θ0 e' l'init della finestra = strato portato ricaricato al bit (_applica_parametri_ verifica
    torch.equal) + strato entrante dal teacher (deterministico). Qui lo si PROVA invece di dirlo: λ = 1, corsa
    interrotta in due punti che prendono le due strade della ripresa (prima_transizione@1 -> si rifa' la finestra 1
    dal punto finestra_01; dopo_commit_strato@3 -> ultima finestra interrotta a meta' delle marche, si rifa' intera),
    poi ripresa con la STESSA ricetta: codici su disco IDENTICI alla corsa λ = 1 senza interruzioni e, per ogni
    finestra rifatta, distanza FINALE da θ0, codici cambiati rispetto a θ0, statistiche della condizione c<0 e perdite
    di ogni passo IDENTICHE (==): tutte dipendono da θ0, quindi θ0 ritrovata al bit.
Varie: valori non validi (λ negativo, NaN, testo; SEMPRE=2; granularita' sconosciuta; SEMPRE o GRANULARITA con
λ = 0) RIFIUTATI prima di caricare uno strato; THOR_SPD_* nella ricetta (_ricetta); ripresa con λ cambiato
RIFIUTATA.
MISURE (stampate, non pretese): granularita' esperto contro tensore (esperti in disaccordo col segno del loro
tensore, unita' proiettate, r medio, distanza e codici); θ0 tenuta in bf16 invece che fp32 (codici e distanza).
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
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

import torch

torch.set_num_threads(int(os.environ.get('THOR_TORCH_THREADS', '0')) or int(os.environ.get('SPD_TEST_THREADS', '0'))
                      or torch.get_num_threads())

RADICE_COLLAUDO = Path("@@ROOT:models@@/gguf/tony-tern/test_spd")
QUI = Path(__file__).resolve().parent
REPO = QUI.parent
COMMIT_RIF = "4b4553f"
PY = sys.executable
N_STRATI, W, N_LOTTI, N_PASSI = 5, 2, 6, 9
N_PASSI_LUNGHE = 40          # solo per le MISURE a finestre lunghe (la verifica (b) resta a N_PASSI, come test_ancora)
LAM_PROVA = 1.0
ESITI: list = []
MISURE: list = []


def ok(nome: str, cond, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(cond)))
    print(f"    {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(cond)


def misura(riga: str) -> None:
    MISURE.append(riga)
    print("    📏 MISURA " + riga, flush=True)


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
    def __init__(self, spec: dict):
        import valida_finestra_8_12 as V
        import forgia_scorrevole as FS
        import raccordo_scorrevole as R
        import test_forgia_scorrevole as TB
        from quant_ternaria_globale_batch import TernarioDiffBatch
        self.V, self.FS, self.R, self.TB = V, FS, R, TB
        self.spec = spec
        self.base = Path(spec["cartella"])
        self.dtype = torch.bfloat16 if spec["ricetta"] == "produzione" else torch.float32
        self.verboso = spec.get("verboso", False)
        # delta_max dei ternari della forgia: il default del costruttore (_costruisci_ternari_layer non lo passa)
        self.delta_max = inspect.signature(TernarioDiffBatch.__init__).parameters["delta_max"].default

    def giocattolo(self):
        TB = self.TB

        class GiocattoloLotti(TB.Giocattolo):
            """Il Giocattolo di test_forgia_scorrevole con 6 lotti (5 + 1 held-out), come test_ancora."""

            def __init__(self, n, dtype):
                super().__init__(n, dtype)
                g = torch.Generator().manual_seed(11)
                self.ids = torch.randint(0, self.cfg.vocab_size, (N_LOTTI, 2, 6), generator=g)

            def batch(self):
                return self.ids.clone(), N_LOTTI - 1
        return GiocattoloLotti(N_STRATI, self.dtype)


def misura_spia(ctx: Ctx, opt, foto: list) -> dict:
    """Distanza e codici duri cambiati dei parametri dell'ottimizzatore rispetto alla fotografia della spia, con
    formule PROPRIE (float64; codice = sign(z)·(|z| > δ), equivalente a z > δ / z < -δ). Gruppi dell'AdamW della
    forgia nell'ordine di lr_per_tipo: z, log_s, delta_raw."""
    gruppi = [list(g["params"]) for g in opt.param_groups]
    assert len(gruppi) == 3, f"attesi 3 gruppi (z, log_s, delta_raw), trovati {len(gruppi)}"
    par = [p for g in gruppi for p in g]
    num = sum(float(((p.detach().double() - f.double()) ** 2).sum()) for p, f in zip(par, foto))
    den = sum(float((f.double() ** 2).sum()) for f in foto)
    nz, nl = len(gruppi[0]), len(gruppi[1])
    cambiati = totale = 0

    def q(z, dr):
        return torch.sign(z) * (z.abs() > torch.sigmoid(dr) * ctx.delta_max)
    for z, z0, dr, dr0 in zip(gruppi[0], foto[:nz], gruppi[2], foto[nz + nl:]):
        assert tuple(dr.shape) == tuple(z.shape[:-1]) + (1,), (tuple(z.shape), tuple(dr.shape))
        cambiati += int((q(z.detach(), dr.detach()) != q(z0, dr0)).sum())
        totale += z.numel()
    return {"dist": math.sqrt(num / den), "cambiati": cambiati, "totale": totale}


def corsa(ctx: Ctx, nome: str, variabili: "dict | None" = None, asincrona: bool = True, interrompi=None,
          cartella: "Path | None" = None, pulisci: bool = True, corsa_id: str = "tspd", n_passi: int = N_PASSI,
          **kw) -> dict:
    """Una forgia giocattolo; ritorna tutto cio' che si confronta, piu' la misura della spia per finestra."""
    V, FS, R, TB = ctx.V, ctx.FS, ctx.R, ctx.TB
    cartella = cartella or (ctx.base / f"corsa_{nome}")
    gioc = ctx.giocattolo()
    diag: list = []
    passi: list = []
    spia: list = []

    @functools.wraps(R.ottimizza_finestra_JOINT_B)      # la firma resta visibile a verifica_interfaccia_A
    def ottimizza(*a, **k):
        d: list = []
        ris = R.ottimizza_finestra_JOINT_B(*a, raccogli_diagnostica=d, **k)
        diag.extend(d)
        return ris
    vero_passo = V.passo_ottimizzazione
    stato = {"n": 0, "foto": None}

    def spia_passo(tf, qf, h0, opt, e0=None, **k):
        t = stato["n"] % n_passi
        par = [p for g in opt.param_groups for p in g["params"]]
        if t == 0:
            stato["foto"] = [p.detach().clone() for p in par]
        passi.append({"proiezione": "proiezione" in k, "gancio": "dopo_backward" in k})
        ris = vero_passo(tf, qf, h0, opt, e0, **k)
        stato["n"] += 1
        if t == n_passi - 1:
            spia.append(misura_spia(ctx, opt, stato["foto"]))
            stato["foto"] = None
        return ris
    buf = io.StringIO()
    V.passo_ottimizzazione = spia_passo
    errore = None
    try:
        with env(**(variabili or {})), contextlib.redirect_stdout(_Tee(sys.stdout, buf) if ctx.verboso else buf):
            oss = TB.corsa(cartella, gioc, n_passi=n_passi, w=W, corsa_id=corsa_id, asincrona=asincrona,
                           interrompi=interrompi, ottimizza_fn=ottimizza, **kw)
    except BaseException as e:   # noqa: BLE001 — il chiamante decide se l'errore era atteso
        errore = e
        oss = None
    finally:
        V.passo_ottimizzazione = vero_passo
    reg = {"log": buf.getvalue().replace(str(cartella), "<CORSA>"), "diag": diag, "passi": passi, "spia": spia,
           "caricati": [list(c) for c in gioc.caricati],
           "errore": None if errore is None else (type(errore).__name__, str(errore)), "cartella": cartella}
    if errore is None:
        reg["codici"] = TB.codici_su_disco(cartella, N_STRATI, corsa_id)
        reg["semi"] = oss.semi
        reg["giornale"] = [(g["strato"], g["sha1_Q"]) for g in
                           (json.loads(r) for r in (cartella / "out" / "giornale_scorrevole.jsonl").read_text()
                            .splitlines())]
        reg["ricetta_json"] = json.loads((cartella / "out" / "ricetta_scorrevole.json").read_text())
    if pulisci and errore is None:
        shutil.rmtree(cartella, ignore_errors=True)
    return reg


def serializzabile(reg: dict) -> dict:
    return {k: v for k, v in reg.items() if k != "cartella"}


# ── (c) la formula su tensori sintetici ──
def _ternari_sintetici(E: int, seme: int):
    from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso
    g = torch.Generator().manual_seed(seme)
    gu = TernarioDiffBatchFuso(torch.randn(E, 6, 16, generator=g) * 0.02, group=8)
    dw = TernarioDiffBatchFuso(torch.randn(E, 8, 6, generator=g) * 0.02, group=6)
    return gu, dw


def _casi_per_esperto(p0: torch.Tensor, g: torch.Generator):
    """(p, grad, p~) con un caso per esperto (E = 5): 0 c<0 e crescita · 1 c<0 senza crescita · 2 c>0 e crescita ·
    3 d = 0 (c = 0) e crescita · 4 p~ = θ0 (γ = 0)."""
    d = torch.randn(p0.shape, generator=g) * 0.1
    p = p0.clone()
    gr = torch.randn(p0.shape, generator=g)
    pt = p0.clone()
    for e in range(p0.shape[0]):
        if e == 0:
            p[e] = p0[e] + d[e]; gr[e] = 0.7 * d[e]; pt[e] = p0[e] + 1.5 * d[e]
        elif e == 1:
            p[e] = p0[e] + d[e]; gr[e] = 0.7 * d[e]; pt[e] = p0[e] + 0.5 * d[e]
        elif e == 2:
            p[e] = p0[e] + d[e]; gr[e] = -0.7 * d[e]; pt[e] = p0[e] + 1.3 * d[e]
        elif e == 3:
            gr[e] = torch.randn(p0[e].shape, generator=g); pt[e] = p0[e] + d[e]
        else:
            p[e] = p0[e] + d[e]; gr[e] = 0.7 * d[e]; pt[e] = p0[e]
    return p, gr, pt


def _riferimento(p0, p, gr, pt, lam, sempre, gran):
    """Calcolo INDIPENDENTE in float64 (per esperto o per tensore): c, γ_prev, γ, r, k e il p proiettato."""
    P0, P, G, PT = (x.double() for x in (p0, p, gr, pt))
    E = P.shape[0]
    c = -(G * (P - P0)).reshape(E, -1).sum(1)
    gp = (P - P0).reshape(E, -1).norm(dim=1)
    ga = (PT - P0).reshape(E, -1).norm(dim=1)
    if gran == "tensore":
        c, gp, ga = c.sum().reshape(1), gp.square().sum().sqrt().reshape(1), ga.square().sum().sqrt().reshape(1)
    r = torch.tensor([max(0.0, float(a - b)) / float(a) if float(a) > 0 else 0.0 for a, b in zip(ga, gp)],
                     dtype=torch.float64)
    cond = torch.ones_like(c, dtype=torch.bool) if sempre else (c < 0)
    k = lam * r * cond.double()
    kk = k.expand(E) if gran == "tensore" else k
    nuovo = PT - kk.view(-1, *([1] * (PT.dim() - 1))) * (PT - P0)
    return {"c": c, "gp": gp, "ga": ga, "r": r, "cond": cond, "k": k, "p": nuovo}


def scenario_formula(ctx: Ctx) -> dict:
    import spd_forgia as S
    esiti = []
    for gran in S.GRANULARITA:
        for sempre in (False, True):
            for lam in (0.5, 1.0, 2.0):
                gu, dw = _ternari_sintetici(5, 31)
                opt = torch.optim.AdamW([p for o in (gu, dw) for p in o.parameters()], lr=0.01)
                g = torch.Generator().manual_seed(77)
                casi = {}
                with torch.no_grad():
                    for o in (gu, dw):
                        for n in S.NOMI_PARAMETRI:
                            prm = getattr(o, n)
                            p0 = prm.detach().clone()
                            p, gr, pt = _casi_per_esperto(p0, g)
                            casi[id(prm)] = (p0, p, gr, pt)
                spd = S.ProiezioneSPD(lam, sempre, gran, [0], {0: gu}, {0: dw}, opt, log=lambda *a: None)
                # la fotografia e' stata presa ORA (p = p0); ora si porta p a θ_{t-1} e si mette il gradiente
                with torch.no_grad():
                    for o in (gu, dw):
                        for n in S.NOMI_PARAMETRI:
                            prm = getattr(o, n)
                            p0, p, gr, pt = casi[id(prm)]
                            prm.copy_(p)
                            prm.grad = gr.clone()
                spd.prima_del_passo()
                visti = {v.etichetta: (v.c_e.clone(), v.gam_prev_e.clone()) for v in spd.voci}
                with torch.no_grad():                    # «opt.step()»: p~ imposto a mano
                    for o in (gu, dw):
                        for n in S.NOMI_PARAMETRI:
                            getattr(o, n).copy_(casi[id(getattr(o, n))][3])
                spd.dopo_il_passo()
                storia = spd._storia[-1][1].tolist()
                for fam, o in (("gu", gu), ("dw", dw)):
                    for n in S.NOMI_PARAMETRI:
                        prm = getattr(o, n)
                        p0, p, gr, pt = casi[id(prm)]
                        rif = _riferimento(p0, p, gr, pt, lam, sempre, gran)
                        c_e, gp_e = visti[f"L0.{fam}.{n}"]
                        c_mod = c_e.sum().reshape(1) if gran == "tensore" else c_e
                        gp_mod = gp_e.square().sum().sqrt().reshape(1) if gran == "tensore" else gp_e
                        nuovo = prm.detach()
                        k_e = rif["k"].expand(p0.shape[0]) if gran == "tensore" else rif["k"]
                        fermi = [e for e in range(p0.shape[0]) if float(k_e[e]) == 0.0]
                        scala = float((pt.double() - p0.double()).abs().max()) or 1.0
                        err_p = float((nuovo.double() - rif["p"]).abs().max()) / scala
                        # con λ = 1 la deviazione di un'unita' proiettata torna a γ_prev
                        dev_nuova = (nuovo.double() - p0.double()).reshape(p0.shape[0], -1).norm(dim=1)
                        if gran == "tensore":
                            dev_nuova = dev_nuova.square().sum().sqrt().reshape(1)
                        proiettate = [j for j in range(len(rif["k"])) if float(rif["k"][j]) > 0]
                        err_lam1 = (max((abs(float(dev_nuova[j] - rif["gp"][j])) / float(rif["gp"][j])
                                         for j in proiettate if float(rif["gp"][j]) > 0), default=0.0)
                                    if lam == 1.0 else 0.0)
                        esiti.append({
                            "gran": gran, "sempre": sempre, "lam": lam, "voce": f"{fam}.{n}",
                            "err_c": float((c_mod - rif["c"]).abs().max() / rif["c"].abs().max().clamp_min(1e-30)),
                            "err_gp": float((gp_mod - rif["gp"]).abs().max() / rif["gp"].abs().max().clamp_min(1e-30)),
                            "err_p": err_p, "fermi_identici": all(torch.equal(nuovo[e], pt[e]) for e in fermi),
                            "n_fermi": len(fermi), "err_lam1": err_lam1, "nan": bool(torch.isnan(nuovo).any()),
                            "cond": rif["cond"].tolist(), "k": rif["k"].tolist(), "storia": storia})
                spd.libera()
    return {"formula": esiti}


# ── (e) l'aggancio in passo_ottimizzazione ──
def scenario_aggancio(ctx: Ctx) -> dict:
    import copy
    import finestra_globale as FG

    class Registratore:
        def __init__(self, params):
            self.params, self.eventi = params, []

        def prima_del_passo(self):
            self.eventi.append(("prima", [p.detach().clone() for p in self.params],
                                [None if p.grad is None else p.grad.clone() for p in self.params]))

        def dopo_il_passo(self):
            self.eventi.append(("dopo", [p.detach().clone() for p in self.params], None))
    st_t, st_q, pesi, h0, _ = FG.costruisci_finestra_giocattolo(n_strati=1, seed=5)
    FG._perturba_z(pesi, scala=0.6, seed=10)
    params = FG.parametri_ternari(pesi[0])
    gemelli = copy.deepcopy(pesi)                       # stessa finestra, senza proiezione
    strato2 = FG.StratoMoEGiocattolo(gemelli[0], st_q[0].keywords["strato"].routing_fisso, ternario=True)
    st_q2 = [functools.partial(FG._chiama_quant, strato=strato2, beta=16.0)]
    params2 = FG.parametri_ternari(gemelli[0])
    opt = torch.optim.AdamW(params, lr=0.05)
    opt2 = torch.optim.AdamW(params2, lr=0.05)
    reg = Registratore(params)
    prima = [p.detach().clone() for p in params]
    FG.passo_ottimizzazione(st_t, st_q, h0, opt, proiezione=reg)
    FG.passo_ottimizzazione(st_t, st_q2, h0, opt2)
    uguali_prima = len(reg.eventi) == 2 and reg.eventi[0][0] == "prima" and all(
        torch.equal(a, b) for a, b in zip(reg.eventi[0][1], prima))
    grad_uguali = all(g is not None and torch.equal(g, p2.grad) for g, p2 in zip(reg.eventi[0][2], params2))
    dopo_uguali = len(reg.eventi) == 2 and reg.eventi[1][0] == "dopo" and all(
        torch.equal(a, p2.detach()) for a, p2 in zip(reg.eventi[1][1], params2))
    # micro-passo di accumulo (indice 0 di 2): nessun passo, nessuna chiamata
    reg2 = Registratore(params)
    FG.passo_ottimizzazione(st_t, st_q, h0, opt, indice_accumulo=0, passi_accumulo=2, proiezione=reg2)
    accumulo_muto = reg2.eventi == []
    FG.passo_ottimizzazione(st_t, st_q, h0, opt, indice_accumulo=1, passi_accumulo=2, proiezione=reg2)
    accumulo_ultimo = [e[0] for e in reg2.eventi] == ["prima", "dopo"]
    # passo saltato: un gradiente non finito (salta_non_finiti) -> nessuna chiamata
    reg3 = Registratore(params)

    class Avvelena:
        def __call__(self, dati):
            params[0].grad.view(-1)[0] = float("nan")
    FG.passo_ottimizzazione(st_t, st_q, h0, opt, salta_non_finiti=True, dopo_backward=Avvelena(), proiezione=reg3)
    saltato_muto = reg3.eventi == []
    return {"aggancio": {"prima_vede_p_prima": uguali_prima, "prima_vede_grad": grad_uguali,
                         "dopo_vede_p_adamw": dopo_uguali, "accumulo_muto": accumulo_muto,
                         "accumulo_ultimo": accumulo_ultimo, "saltato_muto": saltato_muto}}


def figlio(spec: dict) -> None:
    radice = Path(spec["radice"]).resolve()
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != QUI]
    sys.path.insert(0, str(radice))
    os.environ.update(spec["env"])                 # PRIMA di importare valida (costanti di modulo)
    assert not torch.cuda.is_available(), "solo CPU (CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=)"
    torch.manual_seed(0)
    ctx = Ctx(spec)
    assert Path(ctx.V.__file__).resolve().parent == radice and Path(ctx.FS.__file__).resolve().parent == radice, \
        f"moduli importati da {ctx.V.__file__}, attesi da {radice}"
    ctx.base.mkdir(parents=True, exist_ok=True)
    uscita: dict = {"radice": str(radice), "ha_spd": (radice / "spd_forgia.py").exists()
                    and hasattr(ctx.V, "spd_forgia")}
    lam = str(LAM_PROVA)
    for s in spec["scenari"]:
        t0 = time.time()
        print(f"  [figlio {spec['nome']}] scenario {s} ...", flush=True)
        if s == "spenta":
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_SPD_LAMBDA": None}))
        elif s == "zero":
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_SPD_LAMBDA": "0"}))
        elif s == "spd":
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_SPD_LAMBDA": lam}))
        elif s == "spd_esplicita":
            import spd_forgia as S
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_SPD_LAMBDA": lam,
                                                      "THOR_SPD_GRANULARITA": S.GRANULARITA_DEFAULT}))
        elif s == "spd_altra":
            import spd_forgia as S
            altra = [g for g in S.GRANULARITA if g != S.GRANULARITA_DEFAULT][0]
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_SPD_LAMBDA": lam, "THOR_SPD_GRANULARITA": altra}))
        elif s == "sempre":
            uscita[s] = serializzabile(corsa(ctx, s, {"THOR_SPD_LAMBDA": lam, "THOR_SPD_SEMPRE": "1"}))
        elif s == "bf16":
            # MISURA: θ0 tenuta in bf16 (arrotondata alla fotografia) invece che in fp32
            import spd_forgia as S
            vero_init = S._Voce.__init__

            def init_bf16(self, *a, **k):
                vero_init(self, *a, **k)
                self.p0 = self.p0.to(torch.bfloat16).to(self.p0.dtype)
                self.norma02_e = S._norme_righe(self.p0).square()
            S._Voce.__init__ = init_bf16
            try:
                uscita[s] = serializzabile(corsa(ctx, s, {"THOR_SPD_LAMBDA": lam}))
            finally:
                S._Voce.__init__ = vero_init
        elif s == "dose":
            # MISURA: λ 0,5 e 2 (granularita' di default), 9 passi; e SEMPRE a λ 0,5 (a λ 1 SEMPRE annulla ogni passo)
            uscita[s] = {x: serializzabile(corsa(ctx, f"dose_{x}", {"THOR_SPD_LAMBDA": x})) for x in ("0.5", "2")}
            uscita[s]["0.5 SEMPRE"] = serializzabile(corsa(ctx, "dose_sempre", {"THOR_SPD_LAMBDA": "0.5",
                                                                                "THOR_SPD_SEMPRE": "1"}))
        elif s == "lunghe":
            # MISURA: finestre da N_PASSI_LUNGHE passi (piu' vicine ai 150 della forgia vera): λ 0, λ 1 per esperto
            # e per tensore
            import spd_forgia as S
            altra = [g for g in S.GRANULARITA if g != S.GRANULARITA_DEFAULT][0]
            uscita[s] = {n: serializzabile(corsa(ctx, f"lunga_{n}", v, n_passi=N_PASSI_LUNGHE)) for n, v in (
                ("zero", {"THOR_SPD_LAMBDA": None}), (S.GRANULARITA_DEFAULT, {"THOR_SPD_LAMBDA": lam}),
                (altra, {"THOR_SPD_LAMBDA": lam, "THOR_SPD_GRANULARITA": altra}))}
        elif s == "non_validi":
            uscita[s] = {n: serializzabile(corsa(ctx, f"nv_{n}", v)) for n, v in (
                ("lam_negativo", {"THOR_SPD_LAMBDA": "-1"}), ("lam_testo", {"THOR_SPD_LAMBDA": "abc"}),
                ("lam_nan", {"THOR_SPD_LAMBDA": "nan"}), ("sempre_2", {"THOR_SPD_LAMBDA": "1", "THOR_SPD_SEMPRE": "2"}),
                ("gran", {"THOR_SPD_LAMBDA": "1", "THOR_SPD_GRANULARITA": "strato"}),
                ("sempre_senza_lam", {"THOR_SPD_SEMPRE": "1"}),
                ("gran_senza_lam", {"THOR_SPD_LAMBDA": "0", "THOR_SPD_GRANULARITA": "tensore"}))}
        elif s == "ricetta":
            with env(THOR_SPD_LAMBDA="1", THOR_SPD_SEMPRE="1", THOR_SPD_GRANULARITA="tensore"):
                ric = ctx.FS._ricetta()
            cart = ctx.base / "ripresa_ricetta"

            def ferma(p, k):
                if p == "prima_transizione" and k == 1:
                    raise ctx.FS.InterruzioneSimulata("stop")
            prima = corsa(ctx, "ric1", {"THOR_SPD_LAMBDA": "1"}, asincrona=False, interrompi=ferma, cartella=cart,
                          pulisci=False)
            dopo = corsa(ctx, "ric2", {"THOR_SPD_LAMBDA": "0.5"}, asincrona=False, cartella=cart, pulisci=False)
            shutil.rmtree(cart, ignore_errors=True)
            uscita[s] = {"ricetta": ric, "interrotta": prima["errore"], "ripresa_lam": dopo["errore"]}
        elif s == "ripresa":
            # (f) la ripresa con la STESSA ricetta (λ = 1) ritrova θ0: due punti d'interruzione, due strade della ripresa
            casi = {}
            for punto_nome, k_int in (("prima_transizione", 1), ("dopo_commit_strato", N_STRATI - W)):
                cart = ctx.base / f"ripresa_{punto_nome}_{k_int}"

                def ferma(p, k, _pn=punto_nome, _k=k_int):
                    if p == _pn and k == _k:
                        raise ctx.FS.InterruzioneSimulata(f"{p}@{k}")
                prima = corsa(ctx, f"rip1_{punto_nome}", {"THOR_SPD_LAMBDA": lam}, asincrona=False, interrompi=ferma,
                              cartella=cart, pulisci=False)
                dopo = corsa(ctx, f"rip2_{punto_nome}", {"THOR_SPD_LAMBDA": lam}, asincrona=False, cartella=cart,
                             pulisci=False)
                shutil.rmtree(cart, ignore_errors=True)
                casi[f"{punto_nome}@{k_int}"] = {"interrotta": prima["errore"], "ripresa": serializzabile(dopo)}
            uscita[s] = casi
        elif s == "formula":
            uscita[s] = scenario_formula(ctx)
        elif s == "aggancio":
            uscita[s] = scenario_aggancio(ctx)
        else:
            raise ValueError(f"scenario {s!r} sconosciuto")
        print(f"  [figlio {spec['nome']}] scenario {s} in {time.time() - t0:.0f}s", flush=True)
    torch.save(uscita, spec["uscita"])


# ═══════════════════════════════════ GENITORE: lancia e confronta ═══════════════════════════════════
def lancia_figlio(nome: str, radice: Path, ricetta: str, ricetta_env: dict, scenari: list, base: Path,
                  verboso: bool) -> dict:
    uscita = base / f"esito_{nome}.pt"
    spec = {"nome": nome, "radice": str(radice), "ricetta": ricetta, "env": ricetta_env, "scenari": scenari,
            "cartella": str(base / nome), "uscita": str(uscita), "verboso": verboso}
    ambiente = {k: v for k, v in os.environ.items() if not k.startswith("THOR_")}
    ambiente.update({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "THOR_CRONO_OGNI": "3",
                     # i thread passano al figlio fuori dal prefisso THOR_ (che entrerebbe nella ricetta)
                     "SPD_TEST_THREADS": os.environ.get("THOR_TORCH_THREADS", "0")})
    t0 = time.time()
    p = subprocess.run([PY, str(QUI / "test_spd.py"), "--figlio", json.dumps(spec)], env=ambiente,
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
_ASINCRONE = re.compile(r"^(codici scritti in |  punto di ripresa finestra \d+ scritto |  impronta del punto di ripresa )")


def log_confrontabile(testo: str) -> "tuple[list, list]":
    t = _ORARIO.sub("", testo)
    for rx, sost in _TEMPI:
        t = rx.sub(sost, t)
    righe = t.splitlines()
    return [r for r in righe if not _ASINCRONE.match(r)], sorted(r for r in righe if _ASINCRONE.match(r))


def _uguali_codici(a: dict, b: dict) -> bool:
    import numpy as np
    return sorted(a) == sorted(b) and all(
        sorted(a[L]) == sorted(b[L]) and all(np.array_equal(a[L][f][0], b[L][f][0]) and np.array_equal(a[L][f][1], b[L][f][1])
                                             for f in a[L]) for L in a)


def _codici_diversi(a: dict, b: dict) -> int:
    import numpy as np
    return sum(int(np.sum(a[L][f][1] != b[L][f][1])) for L in a for f in a[L])


def _perdite(reg: dict) -> list:
    return [(d["perdite"], d["heldout"]) for d in reg["diag"]]


def _semi_uguali(a: dict, b: dict) -> bool:
    if sorted(a) != sorted(b):
        return False
    for k in a:
        for x, y in zip(a[k], b[k]):
            if (x is None) != (y is None):
                return False
            if x is None:
                continue
            if sorted(x) != sorted(y) or not all(x[i].dtype == y[i].dtype and torch.equal(x[i], y[i]) for i in x):
                return False
    return True


def confronta_identiche(etichetta: str, rif: dict, reg: dict) -> None:
    ok(f"{etichetta}: nessun errore", rif.get("errore") is None and reg.get("errore") is None,
       f"{rif.get('errore')} / {reg.get('errore')}")
    if rif.get("errore") is not None or reg.get("errore") is not None:
        return
    ok(f"{etichetta}: codici su disco IDENTICI (tutti gli strati, D e Q)", _uguali_codici(rif["codici"], reg["codici"]))
    ok(f"{etichetta}: perdite di OGNI passo e held-out identiche (==)", _perdite(rif) == _perdite(reg),
       f"{len(reg['diag'])} finestre")
    ok(f"{etichetta}: semi di ogni finestra identici (torch.equal)", _semi_uguali(rif["semi"], reg["semi"]))
    ok(f"{etichetta}: sha1 dei codici nel giornale identici", rif["giornale"] == reg["giornale"])
    (pa, sa), (pb, sb) = log_confrontabile(rif["log"]), log_confrontabile(reg["log"])
    diverse = [(x, y) for x, y in zip(pa, pb) if x != y]
    ok(f"{etichetta}: log identico (orari e tempi esclusi; thread principale riga per riga, thread di scrittura "
       "come multinsieme)", pa == pb and sa == sb and len(sa) > 0,
       f"{len(pa)} + {len(sa)} righe" if pa == pb and sa == sb else
       f"prima differenza: {diverse[:1]} · righe {len(pa)}+{len(sa)} vs {len(pb)}+{len(sb)}")


def parte_a(ricetta: str, vecchio: dict, nuovo: dict) -> None:
    print(f"\n── (a) [{ricetta}] spenta = {COMMIT_RIF} al bit ──")
    ok(f"(a) [{ricetta}] il figlio vecchio usa davvero i moduli di {COMMIT_RIF} (niente spd_forgia), il nuovo si'",
       vecchio.get("ha_spd") is False and nuovo.get("ha_spd") is True, vecchio.get("radice", ""))
    confronta_identiche(f"(a) [{ricetta}] {COMMIT_RIF} vs nuovo senza variabili", vecchio["spenta"], nuovo["spenta"])
    confronta_identiche(f"(a) [{ricetta}] {COMMIT_RIF} vs nuovo con THOR_SPD_LAMBDA=0", vecchio["spenta"], nuovo["zero"])
    for nome in ("spenta", "zero"):
        reg = nuovo[nome]
        ok(f"(a) [{ricetta}] {nome}: nessun passo riceve `proiezione`",
           reg["passi"] and not any(p["proiezione"] for p in reg["passi"]), f"{len(reg['passi'])} passi")
        trovate = [r for r in reg["log"].splitlines() if "SPD" in r or "θ0" in r]
        ok(f"(a) [{ricetta}] {nome}: nessuna riga SPD nel log, nessuna chiave nuova nella diagnostica",
           not trovate and all(sorted(d) == ["chunk", "codici", "heldout", "perdite"] for d in reg["diag"]),
           "; ".join(trovate[:2]) or f"chiavi {sorted(reg['diag'][0]) if reg['diag'] else '-'}")


def parte_b(ricetta: str, nuovo: dict) -> None:
    import spd_forgia as S
    print(f"\n── (b) [{ricetta}] λ = {LAM_PROVA} ──")
    zero = nuovo["spenta"]
    fin = N_STRATI - W + 1
    altra = [g for g in S.GRANULARITA if g != S.GRANULARITA_DEFAULT][0]
    for scen, gran in (("spd", S.GRANULARITA_DEFAULT), ("spd_altra", altra)):
        reg = nuovo[scen]
        et = f"(b) [{ricetta}] {gran}"
        ok(f"{et}: corsa senza errori", reg.get("errore") is None, str(reg.get("errore")))
        if reg.get("errore") is not None:
            continue
        ok(f"{et}: diagnostica SPD in ognuna delle {fin} finestre, con la granularita' giusta",
           len(reg["diag"]) == fin and all(d.get("spd", {}).get("granularita") == gran for d in reg["diag"]))
        ok(f"{et}: ogni passo riceve `proiezione` e fa la sua storia ({fin} x {N_PASSI})",
           len(reg["passi"]) == fin * N_PASSI and all(p["proiezione"] for p in reg["passi"])
           and all(d["spd"]["passi"] == N_PASSI for d in reg["diag"]))
        log = reg["log"]
        ok(f"{et}: log: accensione, fotografia e FINALE per finestra, cronometro con la coda SPD",
           "SPD accesa: λ 1" in log and log.count("fotografia dei latenti di partenza θ0") == fin
           and log.count("distanza FINALE dai latenti di partenza") == fin
           and re.search(r"cronometro .*\(loss [0-9.e+-]+\) · SPD \(3 passi\): c<0 tensori", log) is not None)
        ric = reg["ricetta_json"]["ricetta"]
        ok(f"{et}: THOR_SPD_* nella ricetta registrata (ricetta_scorrevole.json)",
           ric.get("THOR_SPD_LAMBDA") == str(LAM_PROVA)
           and (ric.get("THOR_SPD_GRANULARITA") == (None if scen == "spd" else gran)))
        ok(f"{et}: codici DIVERSI da quelli senza SPD", not _uguali_codici(reg["codici"], zero["codici"]),
           f"{_codici_diversi(reg['codici'], zero['codici'])} codici diversi")
        for i, d in enumerate(reg["diag"]):
            sp, s0, m = reg["spia"][i], zero["spia"][i], d["spd"]
            rel = abs(m["distanza_fine"] - sp["dist"]) / sp["dist"]
            ok(f"{et} finestra {i}: riga FINALE = misura indipendente della spia: distanza {m['distanza_fine']:.6e} vs "
               f"{sp['dist']:.6e} (scarto {rel:.1e}) · codici cambiati {m['codici_cambiati']:.6f} vs "
               f"{sp['cambiati']}/{sp['totale']}", rel < 1e-6 and m["codici_cambiati"] == sp["cambiati"] / sp["totale"])
            testo = (f"{et} finestra {i}: distanza dai latenti di partenza {sp['dist']:.4e} vs λ=0 {s0['dist']:.4e} "
                     f"({sp['dist'] / s0['dist']:.3f}x) · codici duri cambiati {sp['cambiati']} vs λ=0 "
                     f"{s0['cambiati']} (su {sp['totale']})")
            verifica = sp["dist"] < s0["dist"] and sp["cambiati"] < s0["cambiati"]
            if ricetta == "produzione":
                ok(testo.replace(" vs λ=0", " < λ=0") + " — MINORI entrambe", verifica)
            else:
                misura(testo + f" · {'minori entrambe' if verifica else 'NON minori entrambe'}")
            tot = m["totali"]
            misura(f"{et} finestra {i}: esperti in disaccordo col segno del loro tensore "
                   f"{100 * tot['disaccordo'] / tot['esperti']:.1f}% · c<0 tensori {100 * tot['tensori_neg'] / tot['tensori']:.1f}%"
                   f" · esperti {100 * tot['esperti_neg'] / tot['esperti']:.1f}% · unita' proiettate "
                   f"{100 * tot['unita_proiettate'] / tot['unita']:.1f}% · r medio "
                   f"{(tot['somma_r_cond'] / tot['unita_cond']) if tot['unita_cond'] else float('nan'):.3e}")
    esp = nuovo["spd_esplicita"]
    ok(f"(b) [{ricetta}] granularita' di default ({S.GRANULARITA_DEFAULT}) scritta esplicitamente: codici IDENTICI "
       "alla variabile assente", esp.get("errore") is None and _uguali_codici(esp["codici"], nuovo["spd"]["codici"])
       and _perdite(esp) == _perdite(nuovo["spd"]))
    b16 = nuovo["bf16"]
    if b16.get("errore") is None:
        for i, (a, b) in enumerate(zip(b16["spia"], nuovo["spd"]["spia"])):
            misura(f"(bf16) [{ricetta}] finestra {i}: θ0 in bf16 invece che fp32 -> codici cambiati {a['cambiati']} vs "
                   f"{b['cambiati']} (su {a['totale']}) · distanza {a['dist']:.4e} vs {b['dist']:.4e}")
        misura(f"(bf16) [{ricetta}] codici finali diversi da θ0 fp32: {_codici_diversi(b16['codici'], nuovo['spd']['codici'])}"
               f" su {sum(int(v[1].size) for L in b16['codici'] for v in b16['codici'][L].values())}")
    else:
        ok(f"(bf16) [{ricetta}] corsa senza errori", False, str(b16.get("errore")))
    zero_spia = nuovo["spenta"]["spia"]
    for x, reg in nuovo["dose"].items():
        if reg.get("errore") is not None:
            ok(f"(dose) [{ricetta}] λ {x}: corsa senza errori", False, str(reg.get("errore")))
            continue
        misura(f"(dose) [{ricetta}] λ {x} ({S.GRANULARITA_DEFAULT}), {N_PASSI} passi: per finestra distanza/λ=0 "
               + " · ".join(f"{a['dist'] / b['dist']:.3f}x" for a, b in zip(reg["spia"], zero_spia))
               + " · codici cambiati " + " · ".join(f"{a['cambiati']} (λ=0 {b['cambiati']})"
                                                    for a, b in zip(reg["spia"], zero_spia)))
    lu = nuovo["lunghe"]
    if all(r.get("errore") is None for r in lu.values()):
        z = lu["zero"]["spia"]
        for n in [k for k in lu if k != "zero"]:
            misura(f"(lunghe) [{ricetta}] λ 1 {n}, {N_PASSI_LUNGHE} passi: per finestra distanza/λ=0 "
                   + " · ".join(f"{a['dist'] / b['dist']:.3f}x" for a, b in zip(lu[n]["spia"], z))
                   + " · codici cambiati " + " · ".join(f"{a['cambiati']} (λ=0 {b['cambiati']})"
                                                        for a, b in zip(lu[n]["spia"], z))
                   + f" · totale {sum(a['cambiati'] for a in lu[n]['spia'])} contro {sum(b['cambiati'] for b in z)}")
    else:
        ok(f"(lunghe) [{ricetta}] corse senza errori", False, "; ".join(str(r.get("errore")) for r in lu.values()))


def parte_c(ricetta: str, nuovo: dict) -> None:
    print(f"\n── (c) [{ricetta}] formula su tensori sintetici ──")
    es = nuovo["formula"]["formula"]
    for gran in ("esperto", "tensore"):
        for sempre in (False, True):
            sel = [e for e in es if e["gran"] == gran and e["sempre"] == sempre]
            ok(f"(c) [{ricetta}] {gran}{' SEMPRE' if sempre else ''}: c, γ_prev = calcolo indipendente float64 "
               f"(scarto max {max(max(e['err_c'], e['err_gp']) for e in sel):.1e}), p proiettato "
               f"(scarto max {max(e['err_p'] for e in sel):.1e} della deviazione), nessun NaN",
               sel and all(e["err_c"] < 1e-6 and e["err_gp"] < 1e-6 and e["err_p"] < 1e-6 and not e["nan"] for e in sel))
            ok(f"(c) [{ricetta}] {gran}{' SEMPRE' if sempre else ''}: dove k = 0 il parametro resta IDENTICO al bit "
               f"({sum(e['n_fermi'] for e in sel)} unita' ferme)", all(e["fermi_identici"] for e in sel))
            ok(f"(c) [{ricetta}] {gran}{' SEMPRE' if sempre else ''}: con λ = 1 la deviazione proiettata torna a "
               f"γ_prev (scarto max {max(e['err_lam1'] for e in sel):.1e})", all(e["err_lam1"] < 1e-6 for e in sel))
    per_esp = [e for e in es if e["gran"] == "esperto" and not e["sempre"]]
    # casi per esperto: 0 c<0 e crescita -> proiettato; 1 c<0 senza crescita, 2 c>0, 3 d=0, 4 γ=0 -> fermi
    ok(f"(c) [{ricetta}] per esperto, senza SEMPRE: proietta SOLO l'esperto con c<0 e deviazione in crescita",
       all(e["k"][0] > 0 and e["k"][1:] == [0.0, 0.0, 0.0, 0.0] and e["cond"][:3] == [True, True, False]
           and e["cond"][3] is False for e in per_esp), str(per_esp[0]["k"]))
    per_esp_s = [e for e in es if e["gran"] == "esperto" and e["sempre"]]
    print(f"\n── (d) [{ricetta}] THOR_SPD_SEMPRE ──")
    ok(f"(d) [{ricetta}] per esperto, SEMPRE: proietta anche c>0 (esperto 2) e d=0 (esperto 3, r = 1), non chi non "
       "cresce (1) ne' γ=0 (4)",
       all(e["k"][0] > 0 and e["k"][2] > 0 and e["k"][3] > 0 and e["k"][1] == 0.0 and e["k"][4] == 0.0
           for e in per_esp_s), str(per_esp_s[0]["k"]))


def parte_d(ricetta: str, nuovo: dict) -> None:
    sem, spd = nuovo["sempre"], nuovo["spd"]
    ok(f"(d) [{ricetta}] forgia con SEMPRE senza errori", sem.get("errore") is None, str(sem.get("errore")))
    if sem.get("errore") is not None:
        return
    primi = [(d["spd"]["storia"][0][0], d["spd"]["storia"][0][1]) for d in sem["diag"]]
    primi_c = [(d["spd"]["storia"][0][0], d["spd"]["storia"][0][1]) for d in spd["diag"]]
    ok(f"(d) [{ricetta}] al primo passo di ogni finestra (c = 0): SEMPRE proietta ({[int(s['unita_proiettate']) for _, s in primi]} "
       f"unita'), senza SEMPRE nessuna ({[int(s['unita_proiettate']) for _, s in primi_c]})",
       all(t == 0 and s["unita_proiettate"] > 0 for t, s in primi)
       and all(t == 0 and s["unita_proiettate"] == 0 and s["unita_cond"] == 0 for t, s in primi_c))
    ok(f"(d) [{ricetta}] codici con SEMPRE diversi da λ = 1 con la condizione",
       not _uguali_codici(sem["codici"], spd["codici"]), f"{_codici_diversi(sem['codici'], spd['codici'])} diversi")
    for i, (a, b) in enumerate(zip(sem["spia"], nuovo["spenta"]["spia"])):
        misura(f"(d) [{ricetta}] SEMPRE finestra {i}: distanza {a['dist']:.4e} (λ=0 {b['dist']:.4e}) · codici "
               f"cambiati {a['cambiati']} (λ=0 {b['cambiati']})")


def parte_e(ricetta: str, nuovo: dict) -> None:
    print(f"\n── (e) [{ricetta}] aggancio in passo_ottimizzazione ──")
    a = nuovo["aggancio"]["aggancio"]
    ok(f"(e) [{ricetta}] prima_del_passo vede i parametri PRIMA del passo e i gradienti del backward (al bit)",
       a["prima_vede_p_prima"] and a["prima_vede_grad"])
    ok(f"(e) [{ricetta}] dopo_il_passo vede i parametri DOPO AdamW (al bit contro la copia senza proiezione)",
       a["dopo_vede_p_adamw"])
    ok(f"(e) [{ricetta}] micro-passo di accumulo: nessuna chiamata; ultimo micro-passo: prima e dopo",
       a["accumulo_muto"] and a["accumulo_ultimo"])
    ok(f"(e) [{ricetta}] passo saltato per gradienti non finiti: nessuna chiamata", a["saltato_muto"])


def parte_f(ricetta: str, nuovo: dict) -> None:
    print(f"\n── (f) [{ricetta}] ripresa con SPD accesa (λ = {LAM_PROVA}): θ0 ritrovata al bit ──")
    rif = nuovo["spd"]
    if rif.get("errore") is not None:
        ok(f"(f) [{ricetta}] corsa di riferimento λ = {LAM_PROVA} senza errori", False, str(rif.get("errore")))
        return
    for caso, x in nuovo["ripresa"].items():
        rip = x["ripresa"]
        et = f"(f) [{ricetta}] {caso}"
        ok(f"{et}: interruzione avvenuta, ripresa senza errori",
           x["interrotta"] is not None and x["interrotta"][0] == "InterruzioneSimulata" and rip.get("errore") is None,
           f"{x['interrotta']} / {rip.get('errore')}")
        if rip.get("errore") is not None:
            continue
        ok(f"{et}: codici su disco IDENTICI alla corsa λ = {LAM_PROVA} senza interruzioni (tutti gli strati, D e Q)",
           _uguali_codici(rip["codici"], rif["codici"]),
           f"{_codici_diversi(rip['codici'], rif['codici'])} codici diversi")
        n = len(rip["diag"])
        coppie = list(zip(rip["diag"], rif["diag"][len(rif["diag"]) - n:])) if n else []
        uguali = n > 0 and all(
            a["chunk"] == b["chunk"] and a["perdite"] == b["perdite"]
            and a["spd"]["distanza_fine"] == b["spd"]["distanza_fine"]
            and a["spd"]["codici_cambiati"] == b["spd"]["codici_cambiati"]
            and a["spd"]["totali"] == b["spd"]["totali"] for a, b in coppie)
        ok(f"{et}: finestre rifatte {[a['chunk'] for a, _ in coppie]}: distanza FINALE da θ0, codici cambiati rispetto "
           "a θ0, statistiche della condizione e perdite di ogni passo IDENTICHE (==) alla corsa senza interruzioni",
           uguali, "; ".join(f"{a['chunk']}: dist {a['spd']['distanza_fine']:.9e} vs {b['spd']['distanza_fine']:.9e}"
                             for a, b in coppie))


def parte_varie(ricetta: str, nuovo: dict) -> None:
    print(f"\n── varie [{ricetta}] ──")
    for nome, reg in nuovo["non_validi"].items():
        e = reg.get("errore")
        ok(f"[{ricetta}] configurazione non valida ({nome}) RIFIUTATA prima di caricare uno strato",
           e is not None and e[0] == "ValueError" and not reg["caricati"], str(e)[:120] if e else "nessun errore")
    r = nuovo["ricetta"]
    ok(f"[{ricetta}] _ricetta(): THOR_SPD_LAMBDA / SEMPRE / GRANULARITA dentro",
       all(k in r["ricetta"] for k in ("THOR_SPD_LAMBDA", "THOR_SPD_SEMPRE", "THOR_SPD_GRANULARITA")))
    ok(f"[{ricetta}] ripresa con THOR_SPD_LAMBDA cambiato (1 -> 0.5) RIFIUTATA",
       r["interrotta"] is not None and r["interrotta"][0] == "InterruzioneSimulata" and r["ripresa_lam"] is not None
       and "ricetta diversa" in r["ripresa_lam"][1] and "THOR_SPD_LAMBDA" in r["ripresa_lam"][1],
       str(r["ripresa_lam"])[:160])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figlio", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--verboso", action="store_true")
    ap.add_argument("--tieni", action="store_true")
    ap.add_argument("--ricette", default="base,produzione")
    args = ap.parse_args()
    if args.figlio:
        figlio(json.loads(args.figlio))
        return 0
    for v in [k for k in os.environ if k.startswith(("THOR_SPD", "THOR_ANCORA", "THOR_HELDOUT_CONOSCENZA",
                                                     "THOR_CACHE_LOTTI"))]:
        raise SystemExit(f"{v} impostata nell'ambiente: il collaudo le gestisce da se'")
    sys.path.insert(0, str(QUI))
    import test_raccordo_scorrevole as TR
    base = RADICE_COLLAUDO / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True)
    t0 = time.time()
    print(f"cartella di collaudo {base}")
    try:
        vecchio_dir = base / f"moduli_{COMMIT_RIF}"
        vecchio_dir.mkdir()
        arch = subprocess.run(["git", "-C", str(REPO), "archive", COMMIT_RIF, "thor_v2", "kernel-rotazione"],
                              capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(vecchio_dir)], input=arch.stdout, check=True)
        radice_vecchia = vecchio_dir / "thor_v2"
        ok(f"estrazione di {COMMIT_RIF}: valida_finestra_8_12.py presente e senza spd_forgia",
           (radice_vecchia / "valida_finestra_8_12.py").exists() and not (radice_vecchia / "spd_forgia.py").exists()
           and "spd_forgia" not in (radice_vecchia / "valida_finestra_8_12.py").read_text())
        scenari_nuovo = ["spenta", "zero", "spd", "spd_esplicita", "spd_altra", "sempre", "bf16", "dose", "lunghe",
                         "non_validi", "ricetta", "ripresa", "formula", "aggancio"]
        for ricetta in args.ricette.split(","):
            renv = dict(TR.RICETTE[ricetta])
            print(f"\n══ ricetta {ricetta}: " + (" ".join(f"{k}={v}" for k, v in renv.items()) or "default") + " ══")
            vecchio = lancia_figlio(f"vecchio_{ricetta}", radice_vecchia, ricetta, renv, ["spenta"], base, args.verboso)
            nuovo = lancia_figlio(f"nuovo_{ricetta}", QUI, ricetta, renv, scenari_nuovo, base, args.verboso)
            if not vecchio or not nuovo:
                continue
            parte_a(ricetta, vecchio, nuovo)
            parte_b(ricetta, nuovo)
            parte_c(ricetta, nuovo)
            parte_d(ricetta, nuovo)
            parte_e(ricetta, nuovo)
            parte_f(ricetta, nuovo)
            parte_varie(ricetta, nuovo)
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
