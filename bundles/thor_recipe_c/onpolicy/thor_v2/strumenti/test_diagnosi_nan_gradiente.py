#!/usr/bin/env python3
"""Collaudo a scala GIOCATTOLO di strumenti/diagnosi_nan_gradiente.py (CPU, zero Thor, zero GPU).

Uso (la GPU e' della forgia):
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 strumenti/test_diagnosi_nan_gradiente.py [--cartella DIR]

Finestra giocattolo di 3 strati (W=3; la forgia usa W=2, ma il primo strato di una finestra
ha l'ingresso dal SEME, senza gradiente: un NaN iniettato li' non arriva a nessun parametro,
come nella forgia vera). Strati 0 e 2 ad attenzione LINEARE (gated delta rule, qui il ripiego
PyTorch: fla vuole la GPU), strato 1 ad attenzione PIENA; esperti con grouped_mm_thor
(esperti_thor.py: _RaccogliRighe/_PermutaRighe), bf16, rotazione sull'ingresso, nucleo
ternario bf16 compilato, routing aggiornato, solo fase dura, salto dei passi non finiti: gli
stessi interruttori di lancia_forgia_scorrevole.sh.

  T1  senza iniezione: nessuna segnalazione, sentinella DAVVERO attiva (nodi e regioni
      contati, Function personalizzate e grafi compilati agganciati), e gradienti di OGNI
      passo, perdite e codici finali IDENTICI AL BIT con e senza ganci (piu' una seconda
      corsa senza ganci: il confronto ha senso solo se l'ambiente e' deterministico).
  P1  NaN iniettato nel backward di L1.self_attn (attenzione piena) al passo 2: lo strumento
      indica L1.self_attn, passo 2, batch 0; grad USCITA finito, INGRESSO non finito; replay
      (veleno una tantum, come il NaN vero) NON riprodotto ma loss del replay identica al bit.
  P2  come P1 su L2.linear_attn (attenzione lineare), veleno persistente: replay RIPRODOTTO
      nello stesso punto con conteggi NaN esatti; `riproduci` sull'ingresso salvato: uscita
      identica al bit, gradiente finito senza veleno, NaN in 3/3 col veleno rimesso.
  P3  NaN nato DENTRO una autograd.Function personalizzata (backward di
      esperti_thor._RaccogliRighe, strato 1): indicato il nodo _RaccogliRigheBackward in
      L1.mlp.experts; `riproduci` degli esperti isolati: uscita al bit, NaN solo col veleno.
  P4  NaN nel nucleo ternario compilato (gate/up dello strato 1): indicato
      L1.pesi_ternari › gu.forward_duro; `riproduci` del metodo: uscita al bit.
  P5  passo NON saltato (salta_non_finiti spento): dump scritto PRIMA di opt.step (parametri
      intatti: loss del replay identica al bit) e .grad parcheggiati restituiti IDENTICI al
      bit (NaN compresi) contro la stessa corsa senza diagnosi.
  P6  loss non finita (NaN nel forward): dump con il primo e_l non finito e, nel replay, la
      prima regione con uscita non finita.
  A1  origine di tipo ACCUMULO: due contributi finiti la cui SOMMA trabocca a Inf.
  L1  lanciatore `finestra` (k=0) col suo ciclo vero e caricatori giocattolo: ripete la finestra
      finche' il NaN scatta; senza NaN lo ferma il tetto di tempo; ogni giro riparte identico.
  L2  lanciatore `finestra` k=1 da un punto di ripresa della forgia (seme e strato portato);
      k>0 senza ripresa rifiutato.
  N1  (verifica avversaria 25/9) gradienti TUTTI FINITI x1e30: la norma L2 di passo_ottimizzazione
      trabocca e la forgia salta per 'gradienti'; il rapporto dice 'certo', max|g|, il nodo che
      amplifica (AmplificaBackward in L1.self_attn) e il replay misura le norme come la forgia.
  G1-G4  (verifica avversaria 25/9) guardie GPU: VRAM DRM per processo (Vulkan, invisibile a
      gpu_occupata.sh) su /proc finto; script della coda vivi (processi veri); `forgia` e
      `riproduci --device cuda` rifiutano a GPU occupata; un lavoro comparso ferma la finestra (rc 4).
  M1-M3  il collaudo sa FALLIRE: un gancio che altera i gradienti, un'attribuzione sbagliata e
      la scelta di un nodo che propaga invece dell'origine vengono scoperti.
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RICETTA = {
    "THOR_ROTAZIONE_INGRESSO": "1", "THOR_NUCLEO_BF16": "1", "THOR_ROUTING_VIVO": "0", "THOR_ROUTING_AGGIORNATO": "1",
    "THOR_ESPERTI_THOR": "1", "THOR_DIAGNOSTICA_PASSO": "0", "THOR_FORGIA_DOWN_TERNARIO": "1",
    "THOR_VALIDA_FRAZIONE_DURA": "1.0", "THOR_SALTA_NON_FINITI": "1", "THOR_LR_Z": "0.01", "THOR_LR_LOGS": "0.01",
    "THOR_LR_DELTA": "0.01", "THOR_VALIDA_LR": "0.01", "THOR_GROUPED_BMM": "0", "THOR_HELDOUT_OGNI": "0",
    "THOR_RICONGELA_OGNI": "0", "THOR_CRONO_OGNI": "0", "THOR_COMMIT_DA_CODICI": "1", "THOR_COMMIT_SCALA_FP16": "1",
    "THOR_DIAGNOSI_NAN": "1", "THOR_ID_CORSA": "collaudo-diagnosi-nan",
}
os.environ.update(RICETTA)

import torch  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent))
sys.path.insert(0, str(QUI))

import valida_finestra_8_12 as V  # noqa: E402
import finestra_globale as FG  # noqa: E402
import esperti_thor  # noqa: E402
import quant_ternaria_globale_batch_v2 as QB  # noqa: E402
import diagnosi_nan_gradiente as D  # noqa: E402

ESITI: list = []
N_PASSI = 5
PASSO_BERSAGLIO = 2          # batch = 2 % 2 = 0 (2 batch di addestramento)
STRATI = (0, 1, 2)


def verifica(nome: str, ok: bool, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(ok)))
    print(f"    {'✅' if ok else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(ok)


# ═══════════════════════ finestra giocattolo ═══════════════════════
class Giocattolo:
    """Strati Qwen3_5MoeDecoderLayer VERI a dimensioni giocattolo (multipli di 8: grouped_mm
    su CPU vuole passi di 16 byte), deterministici: teacher seme 100+L, corrente = teacher +
    rumore 900+L (come Giocattolo di test_forgia_scorrevole.py)."""

    def __init__(self):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        esperti_thor.registra()
        self.cfg = Qwen3_5MoeTextConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, moe_intermediate_size=8, shared_expert_intermediate_size=8, num_experts=4,
            num_experts_per_tok=2, layer_types=["linear_attention", "full_attention", "linear_attention"],
            partial_rotary_factor=1.0,
            linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
            linear_num_value_heads=2)
        self.cfg._attn_implementation = "sdpa"
        self.cfg._experts_implementation = esperti_thor.NOME
        g = torch.Generator().manual_seed(7)
        self.emb = torch.randn(self.cfg.vocab_size, self.cfg.hidden_size, generator=g).to(torch.bfloat16)
        self.ids = torch.randint(0, self.cfg.vocab_size, (3, 2, 6), generator=g)   # 2 addestramento + 1 held-out

    def strato(self, L: int, quant: bool):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
        p = Qwen3_5MoeDecoderLayer(self.cfg, layer_idx=L)
        V._init_pesi_giocattolo(p, seed=100 + L)
        if quant:
            q = Qwen3_5MoeDecoderLayer(self.cfg, layer_idx=L)
            with torch.no_grad():
                g = torch.Generator().manual_seed(900 + L)
                for (_, a), (_, b) in zip(p.named_parameters(), q.named_parameters()):
                    b.copy_(a + torch.randn(a.shape, generator=g) * 0.15)
            p = q
        p = p.to(torch.bfloat16).eval()
        for par in p.parameters():
            par.requires_grad_(False)
        return p

    def finestra(self):
        pul = {L: self.strato(L, False) for L in STRATI}
        qua = {L: self.strato(L, True) for L in STRATI}
        h0 = {b: torch.nn.functional.embedding(self.ids[b], self.emb).to(torch.bfloat16) for b in range(3)}
        e0 = {b: (torch.randn(h0[b].shape, generator=torch.Generator().manual_seed(500 + b)) * 0.05).float()
              for b in range(3)}
        return pul, qua, h0, e0


class Osservatore:
    """Involucro PIU' ESTERNO di V.passo_ottimizzazione (installato dopo la diagnosi): conta i
    passi, fotografa i .grad e l'esito di ogni passo, spegne i veleni dopo il passo bersaglio."""

    def __init__(self, veleni=()):
        self.passo = -1
        self.grad: list = []
        self.saltati: list = []
        self.veleni = list(veleni)

    def __enter__(self):
        self._prima = V.passo_ottimizzazione

        def avvolto(*a, **k):
            self.passo += 1
            for v in self.veleni:
                v.passo = self.passo
            try:
                r = self._prima(*a, **k)
            finally:
                for v in self.veleni:
                    v.dopo_passo(self.passo)
            opt = a[3] if len(a) > 3 else k["opt"]
            self.grad.append([None if p.grad is None else p.grad.detach().clone()
                              for g in opt.param_groups for p in g["params"]])
            self.saltati.append(r.get("saltato"))
            return r
        V.passo_ottimizzazione = avvolto
        return self

    def __exit__(self, *_):
        V.passo_ottimizzazione = self._prima
        return False


def corsa(gioc: Giocattolo, n_passi: int = N_PASSI, preparazione=None, veleni=(), tollera_fpe: bool = False):
    """Una finestra [0,2] completa con ottimizza_finestra_JOINT VERA sugli strati giocattolo."""
    pul, qua, h0, e0 = gioc.finestra()
    if preparazione is not None:
        veleni = list(veleni) + list(preparazione(pul, qua))
    diag: list = []
    errore = ris = None
    try:
        with Osservatore(veleni) as oss:
            try:
                ris = V.ottimizza_finestra_JOINT(qua, gioc.ids, gioc.cfg, n_passi, strati_teacher=pul, emb=gioc.emb,
                                                 h0_esterno_per_b=h0, e0_esterno_per_b=e0, n_batch_addestramento=2,
                                                 raccogli_diagnostica=diag)
            except FloatingPointError as e:
                if not tollera_fpe:
                    raise
                errore = str(e)
    finally:
        for v in reversed(veleni):
            v.rimuovi()
    return {"risultato": ris, "perdite": diag[0]["perdite"] if diag else [], "grad": oss.grad,
            "saltati": oss.saltati, "qua": qua, "errore": errore}


# ═══════════════════════ veleni: un NaN nel backward in un punto NOTO ═══════════════════════
class Avvelena(torch.autograd.Function):
    """Identita' nel forward; nel backward mette NaN nel PRIMO elemento del gradiente."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        g.view(-1)[0] = float("nan")
        return g


class VelenoIngressoModulo:
    """Pre-gancio sul modulo: al passo bersaglio l'ingresso principale passa per Avvelena, cosi'
    il backward DI QUEL MODULO produce un gradiente d'ingresso non finito (uscita finita).
    persistente=True: resta attivo per tutto il passo bersaglio (anche nei replay)."""

    def __init__(self, modulo, passo_bersaglio: int, persistente: bool = False):
        self.passo, self.bersaglio, self.persistente = -1, passo_bersaglio, persistente
        self.colpi = 0
        self.spento = False
        self.maniglia = modulo.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _m, args, kwargs):
        if not torch.is_grad_enabled() or self.spento or self.passo != self.bersaglio:
            return None
        if self.colpi and not self.persistente:
            return None
        self.colpi += 1
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": Avvelena.apply(kwargs["hidden_states"])}
        return (Avvelena.apply(args[0]), *args[1:]), kwargs

    def dopo_passo(self, passo):
        if passo == self.bersaglio:
            self.spento = True

    def rimuovi(self):
        self.maniglia.remove()


class VelenoRaccogliRighe:
    """Il NaN nasce DENTRO la Function personalizzata esperti_thor._RaccogliRighe: il suo backward
    restituisce un gradiente con un NaN (il gradiente che riceve e' finito). Si sostituisce il
    nome globale `_RaccogliRighe` (letto da grouped_mm_experts_forward_thor a ogni chiamata) con
    una sottoclasse con lo STESSO nome (il nodo resta «_RaccogliRigheBackward») e un apply che
    sceglie lo strato: nel passo bersaglio le chiamate con gradiente sono L0, L1, L2 per ogni
    forward (sentinella, poi replay) -> colpisce quelle con indice % n_strati == strato."""

    def __init__(self, passo_bersaglio: int, strato: int = 1, n_strati: int = 3, persistente: bool = True):
        self.passo, self.bersaglio, self.persistente = -1, passo_bersaglio, persistente
        self.strato, self.n_strati = strato, n_strati
        self.spento = False
        self.k = 0
        self.colpi = 0
        self._prossimo = False
        self._orig = esperti_thor._RaccogliRighe
        orig, veleno = self._orig, self

        class _RaccogliRighe(orig):
            @staticmethod
            def forward(ctx, h, perm, k):
                ctx.avvelena = veleno._prossimo
                veleno._prossimo = False
                return orig.forward(ctx, h, perm, k)

            @staticmethod
            def backward(ctx, g):
                dh, a, b = orig.backward(ctx, g)
                if getattr(ctx, "avvelena", False):
                    veleno.colpi += 1
                    dh = dh.clone()
                    dh.view(-1)[0] = float("nan")
                return dh, a, b

        class _Scelta:
            @staticmethod
            def apply(h, perm, k):
                if torch.is_grad_enabled() and veleno.passo == veleno.bersaglio and not veleno.spento:
                    i = veleno.k
                    veleno.k += 1
                    veleno._prossimo = (i % veleno.n_strati == veleno.strato
                                        and (veleno.persistente or i < veleno.n_strati))
                return _RaccogliRighe.apply(h, perm, k)
        esperti_thor._RaccogliRighe = _Scelta

    def dopo_passo(self, passo):
        if passo == self.bersaglio:
            self.spento = True

    def rimuovi(self):
        esperti_thor._RaccogliRighe = self._orig


class VelenoNucleo:
    """Il NaN nasce nel ramo del nucleo ternario COMPILATO (bf16) di gate/up dello strato 1: z passa
    per Avvelena prima del kernel compilato. Chiamate con gradiente per passo (costruzione dei
    pesi, PRIMA del passo): L0.gu, L0.dw, L1.gu, L1.dw, L2.gu, L2.dw -> bersaglio = 6*passo + 2
    (e ogni 6 dopo, se persistente: i replay ricostruiscono i pesi nello stesso ordine)."""

    def __init__(self, passo_bersaglio: int, persistente: bool = False):
        self.passo, self.bersaglio, self.persistente = -1, passo_bersaglio, persistente
        self.spento = False
        self.n = 0
        self.colpi = 0
        self._orig = QB._nucleo_duro_compilato_bf16
        veleno = self

        def nucleo(z, log_s, delta_raw, delta_max, beta_t):
            if torch.is_grad_enabled():
                k = veleno.n
                veleno.n += 1
                primo = 6 * veleno.bersaglio + 2
                colpisce = (k == primo) or (veleno.persistente and not veleno.spento and k > primo
                                            and (k - primo) % 6 == 0)
                if colpisce:
                    veleno.colpi += 1
                    z = Avvelena.apply(z)
            return veleno._orig(z, log_s, delta_raw, delta_max, beta_t)
        QB._nucleo_duro_compilato_bf16 = nucleo

    def dopo_passo(self, passo):
        if passo == self.bersaglio:
            self.spento = True

    def rimuovi(self):
        QB._nucleo_duro_compilato_bf16 = self._orig


class VelenoAvanti:
    """NaN nel FORWARD (ingresso di L1.self_attn) al passo bersaglio: loss non finita."""

    def __init__(self, modulo, passo_bersaglio: int):
        self.passo, self.bersaglio, self.spento = -1, passo_bersaglio, False
        self.maniglia = modulo.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _m, args, kwargs):
        if not torch.is_grad_enabled() or self.spento or self.passo != self.bersaglio:
            return None
        x = kwargs["hidden_states"].clone()
        x.view(-1)[3] = float("nan")
        return args, {**kwargs, "hidden_states": x}

    def dopo_passo(self, passo):
        if passo == self.bersaglio:
            self.spento = True

    def rimuovi(self):
        self.maniglia.remove()


# ═══════════════════════ utilita' dei confronti ═══════════════════════
def bit(t: torch.Tensor) -> torch.Tensor:
    """Vista intera dei bit (NaN confrontabili: torch.equal su NaN darebbe sempre falso)."""
    tipi = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16,
            torch.float64: torch.int64}
    return t.contiguous().view(tipi[t.dtype]) if t.dtype in tipi else t


def stessi_bit(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bit(a), bit(b))


def grad_identici(g1: list, g2: list) -> "tuple[bool, str]":
    if len(g1) != len(g2):
        return False, f"passi {len(g1)} vs {len(g2)}"
    for t, (a, b) in enumerate(zip(g1, g2)):
        if len(a) != len(b):
            return False, f"passo {t}: {len(a)} vs {len(b)} parametri"
        for j, (x, y) in enumerate(zip(a, b)):
            if not stessi_bit(x, y):
                return False, f"passo {t} parametro {j} diverso"
    return True, f"{len(g1)} passi x {len(g1[0]) if g1 else 0} parametri"


def codici_identici(r1: dict, r2: dict) -> bool:
    import numpy as np
    return sorted(r1) == sorted(r2) and all(
        np.array_equal(r1[L][f][0], r2[L][f][0]) and np.array_equal(r1[L][f][1], r2[L][f][1])
        for L in r1 for f in r1[L])


def nuova_diagnosi(cartella: Path, **kw) -> D.Diagnosi:
    if cartella.exists():
        shutil.rmtree(cartella)
    return D.Diagnosi(V, FG, cartella=cartella, severo=True, **kw).installa()


def controlla_origine(nome: str, diag: D.Diagnosi, regione: str, nodo: str) -> dict:
    verifica(f"{nome}: UN solo dump scritto", len(diag.dump_scritti) == 1, f"{diag.dump_scritti}")
    if not diag.eventi:
        return {}
    ev = diag.eventi[0]
    o = ev.get("origine") or {}
    verifica(f"{nome}: passo e batch del dump = quelli dell'iniezione",
             ev.get("t") == PASSO_BERSAGLIO and ev.get("b") == PASSO_BERSAGLIO % 2, f"t={ev.get('t')} b={ev.get('b')}")
    verifica(f"{nome}: regione indicata = {regione!r}", o.get("regione") == regione, f"indicata {o.get('regione')!r}")
    verifica(f"{nome}: nodo indicato = {nodo!r}", o.get("nodo") == nodo, f"indicato {o.get('nodo')!r}")
    gu = [s for s in o.get("grad_uscita", []) if s]
    gi = [s for s in o.get("grad_ingresso", []) if s]
    verifica(f"{nome}: grad USCITA finito, grad INGRESSO non finito (tipo 'nodo')",
             o.get("tipo") == "nodo" and gu and all(s["finito"] for s in gu) and any(not s["finito"] for s in gi),
             f"uscita {[s['max_abs'] for s in gu]} · ingresso {[s['max_abs'] for s in gi]}")
    bordo = o.get("bordo_regione") or {}
    verifica(f"{nome}: al bordo della regione grad uscite finite e grad ingressi NON finiti",
             bordo.get("grad_uscite_finite") is True and bordo.get("grad_ingressi_finiti") is False,
             f"uscite {bordo.get('grad_uscite_finite')} ingressi {bordo.get('grad_ingressi_finiti')}")
    return ev


# ═══════════════════════ i collaudi ═══════════════════════
def t1_senza_iniezione(gioc: Giocattolo, base: Path) -> None:
    print("\n── T1 senza iniezione: nessuna segnalazione, sentinella attiva, numeri identici al bit ──", flush=True)
    corsa(gioc, 2)                                    # riscaldamento: compilazioni del nucleo stabilizzate
    a = corsa(gioc)
    diag = nuova_diagnosi(base / "t1")
    try:
        b = corsa(gioc)
        nodi, regioni = diag.nodi_agganciati_ultimo_passo, diag.regioni_ultimo_passo
        nomi, percorsi = diag.nomi_nodi_ultimo_passo, set(diag.percorsi_ultimo_passo)
        passi, dump = diag.passi_osservati, list(diag.dump_scritti)
    finally:
        diag.disinstalla()
    c = corsa(gioc)
    ok_ac, det_ac = grad_identici(a["grad"], c["grad"])
    verifica("T1: ambiente deterministico (due corse SENZA ganci identiche al bit)", ok_ac, det_ac)
    ok_ab, det_ab = grad_identici(a["grad"], b["grad"])
    verifica("T1: gradienti di OGNI passo identici al bit con e senza ganci", ok_ab, det_ab)
    verifica("T1: perdite identiche al bit con e senza ganci", a["perdite"] == b["perdite"],
             f"{a['perdite']} vs {b['perdite']}")
    verifica("T1: codici finali identici con e senza ganci", codici_identici(a["risultato"], b["risultato"]))
    verifica("T1: nessuna segnalazione senza iniezione", dump == [] and not any(b["saltati"]),
             f"dump {dump} · saltati {b['saltati']}")
    verifica(f"T1: sentinella attiva su tutti i {N_PASSI} passi", passi == N_PASSI, f"{passi} passi osservati")
    attesi_nodi = ("CompiledFunctionBackward", "_RaccogliRigheBackward", "_PermutaRigheBackward")
    verifica("T1: agganciati anche nodi di Function personalizzate e grafi compilati",
             nodi > 50 and all(nomi.get(n, 0) > 0 for n in attesi_nodi),
             f"{nodi} nodi · " + ", ".join(f"{n} x{nomi.get(n, 0)}" for n in attesi_nodi))
    attese = {"L0", "L1", "L2", "L0.linear_attn", "L0.linear_attn › gated_delta_rule", "L2.linear_attn › causal_conv1d",
              "L1.self_attn", "L1.self_attn.q_proj", "L0.mlp.experts", "L1.mlp.experts › rotazione_ingresso",
              "L1.mlp.experts › grouped_linear#1", "L1.mlp.experts › grouped_linear#2", "L2.mlp.shared_expert",
              "L1.pesi_ternari", "L1.pesi_ternari › gu.forward_duro", "L0.pesi_ternari › dw.forward_duro"}
    verifica("T1: regioni riconosciute (moduli, funzioni, metodi del nucleo)", attese <= percorsi,
             f"{regioni} regioni · mancano {sorted(attese - percorsi)}")


def p1_attenzione_piena(gioc: Giocattolo, base: Path) -> None:
    print("\n── P1 NaN nel backward di L1.self_attn (una tantum, come il NaN vero) ──", flush=True)
    diag = nuova_diagnosi(base / "p1")
    try:
        r = corsa(gioc, preparazione=lambda pul, qua: [
            VelenoIngressoModulo(qua[1].self_attn, PASSO_BERSAGLIO, persistente=False)])
    finally:
        diag.disinstalla()
    verifica("P1: il passo bersaglio e' stato saltato (gradienti), gli altri no",
             r["saltati"] == [None, None, "gradienti", None, None], f"{r['saltati']}")
    ev = controlla_origine("P1", diag, "L1.self_attn", "AvvelenaBackward")
    rep = ev.get("replay") or []
    verifica("P1: replay eseguito: loss identica al bit, NaN non riprodotto (veleno una tantum)",
             len(rep) == 3 and all(e.get("loss_identica_al_bit") and e.get("riprodotto") is False for e in rep),
             f"{[(e.get('loss_identica_al_bit'), e.get('riprodotto'), e.get('errore')) for e in rep]}")
    verifica("P1: ingresso della regione catturato anche senza riproduzione",
             (ev.get("cattura") or {}).get("percorso") == "L1.self_attn")


def p2_attenzione_lineare(gioc: Giocattolo, base: Path) -> None:
    print("\n── P2 NaN nel backward di L2.linear_attn (persistente: replay riprodotto) ──", flush=True)
    diag = nuova_diagnosi(base / "p2")
    try:
        corsa(gioc, preparazione=lambda pul, qua: [
            VelenoIngressoModulo(qua[2].linear_attn, PASSO_BERSAGLIO, persistente=True)])
    finally:
        diag.disinstalla()
    ev = controlla_origine("P2", diag, "L2.linear_attn", "AvvelenaBackward")
    rep = ev.get("replay") or []
    ok = len(rep) == 3 and all(e.get("riprodotto") and (e.get("origine") or {}).get("stesso_punto_della_sentinella")
                               for e in rep)
    verifica("P2: replay riprodotto 3/3 nello stesso punto della sentinella", ok,
             f"{[(e.get('riprodotto'), (e.get('origine') or {}).get('regione'), e.get('errore')) for e in rep]}")
    gi = [s for s in ((rep[0].get("origine") or {}).get("grad_ingresso") or []) if s] if rep else []
    verifica("P2: conteggi del replay esatti (1 NaN, 0 Inf nel gradiente d'ingresso)",
             sum(s["n_nan"] for s in gi) == 1 and sum(s["n_inf"] for s in gi) == 0,
             f"{[(s['n_nan'], s['n_inf'], s['forma']) for s in gi]}")
    salvati = (rep[0].get("salvati_forward_origine") if rep else None) or {}
    verifica("P2: pre-gancio sul nodo d'origine eseguito nel replay (Avvelena non salva tensori)",
             salvati.get("nodo") == "AvvelenaBackward" and not salvati.get("saved_tensors"), f"{salvati}")
    pt = ev.get("file_pt")
    es = D.riproduci(pt, ripetizioni=3, device="cpu")
    verifica("P2: riproduzione isolata di L2.linear_attn: uscita identica al bit, gradiente finito senza veleno",
             es["uscita_identica_al_bit"] is True and es["non_finite"] == 0,
             f"uscita al bit {es['uscita_identica_al_bit']} · non finite {es['non_finite']}/3")

    def rimetti_veleno(fn, args, kwargs, oggetto):
        return fn, args, {**kwargs, "hidden_states": Avvelena.apply(kwargs["hidden_states"])}
    es2 = D.riproduci(pt, ripetizioni=3, device="cpu", prepara=rimetti_veleno)
    verifica("P2: col veleno rimesso la riproduzione isolata conta NaN in 3/3", es2["non_finite"] == 3,
             f"{es2['non_finite']}/3")


def p3_function_esperti(gioc: Giocattolo, base: Path) -> None:
    print("\n── P3 NaN nato DENTRO esperti_thor._RaccogliRighe.backward (strato 1) ──", flush=True)
    diag = nuova_diagnosi(base / "p3")
    try:
        corsa(gioc, veleni=[VelenoRaccogliRighe(PASSO_BERSAGLIO, strato=1, persistente=True)])
    finally:
        diag.disinstalla()
    ev = controlla_origine("P3", diag, "L1.mlp.experts", "_RaccogliRigheBackward")
    rep = ev.get("replay") or []
    verifica("P3: replay riprodotto nello stesso nodo", bool(rep) and all(
        (e.get("origine") or {}).get("stesso_punto_della_sentinella") for e in rep),
        f"{[((e.get('origine') or {}).get('nodo'), e.get('errore')) for e in rep]}")
    salvati = (rep[0].get("salvati_forward_origine") if rep else None) or {}
    st = salvati.get("saved_tensors") or []
    verifica("P3: tensori SALVATI dal forward di _RaccogliRighe letti prima del backward (perm, int64, T*k)",
             len(st) == 1 and isinstance(st[0], dict) and st[0].get("dtype") == "int64"
             and st[0].get("forma") == [2 * 6 * 2], f"{st}")
    pt = ev.get("file_pt")
    es = D.riproduci(pt, ripetizioni=2, device="cpu")
    verifica("P3: esperti isolati: uscita identica al bit, gradiente finito senza veleno",
             es["uscita_identica_al_bit"] is True and es["non_finite"] == 0,
             f"uscita al bit {es['uscita_identica_al_bit']} · non finite {es['non_finite']}/2")
    veleno = VelenoRaccogliRighe(0, strato=0, n_strati=1, persistente=True)   # ogni chiamata (una per ripetizione)
    veleno.passo = 0
    try:
        es2 = D.riproduci(pt, ripetizioni=2, device="cpu")
    finally:
        veleno.rimuovi()
    verifica("P3: col veleno rimesso nella Function gli esperti isolati danno NaN 2/2", es2["non_finite"] == 2,
             f"{es2['non_finite']}/2")


def p4_nucleo(gioc: Giocattolo, base: Path) -> None:
    print("\n── P4 NaN nel nucleo ternario compilato (gate/up dello strato 1) ──", flush=True)
    diag = nuova_diagnosi(base / "p4")
    try:
        corsa(gioc, veleni=[VelenoNucleo(PASSO_BERSAGLIO, persistente=False)])
    finally:
        diag.disinstalla()
    ev = controlla_origine("P4", diag, "L1.pesi_ternari › gu.forward_duro", "AvvelenaBackward")
    par = ev.get("parametri_con_gradiente_non_finito") or []
    verifica("P4: parametro con gradiente non finito = solo z di L1.gu", par == ["L1.gu.z"], f"{par}")
    es = D.riproduci(ev.get("file_pt"), ripetizioni=2, device="cpu")
    verifica("P4: forward_duro isolato: uscita identica al bit, gradiente finito",
             es["uscita_identica_al_bit"] is True and es["non_finite"] == 0,
             f"uscita al bit {es['uscita_identica_al_bit']} · non finite {es['non_finite']}/2")


def p5_passo_non_saltato(gioc: Giocattolo, base: Path) -> None:
    print("\n── P5 passo NON saltato: analisi prima di opt.step, .grad restituiti al bit ──", flush=True)

    def prep(pul, qua):
        return [VelenoIngressoModulo(qua[1].self_attn, PASSO_BERSAGLIO, persistente=False)]
    V.SALTA_NON_FINITI = False
    try:
        senza = corsa(gioc, preparazione=prep, tollera_fpe=True)
        diag = nuova_diagnosi(base / "p5")
        try:
            con = corsa(gioc, preparazione=prep, tollera_fpe=True)
        finally:
            diag.disinstalla()
    finally:
        V.SALTA_NON_FINITI = True
    ev = diag.eventi[0] if diag.eventi else {}
    verifica("P5: senza salto la forgia si ferma al passo dopo (loss NaN), con e senza diagnosi",
             all(e and "loss non finita al passo 3" in e for e in (senza["errore"], con["errore"])),
             f"{senza['errore']} | {con['errore']}")
    verifica("P5: dump scritto PRIMA di opt.step", "prima di opt.step" in (ev.get("fase") or ""), f"{ev.get('fase')}")
    o = ev.get("origine") or {}
    verifica("P5: regione e nodo indicati", o.get("regione") == "L1.self_attn" and o.get("nodo") == "AvvelenaBackward",
             f"{o.get('regione')} / {o.get('nodo')}")
    rep = ev.get("replay") or []
    verifica("P5: replay a parametri intatti (loss identica al bit)",
             bool(rep) and all(e.get("loss_identica_al_bit") for e in rep),
             f"{[(e.get('loss_identica_al_bit'), e.get('errore')) for e in rep]}")
    ok, det = grad_identici(senza["grad"], con["grad"])
    verifica("P5: .grad di ogni passo (NaN compresi) identici al bit con e senza diagnosi", ok, det)


def p6_loss(gioc: Giocattolo, base: Path) -> None:
    print("\n── P6 loss non finita (NaN nel forward di L1.self_attn) ──", flush=True)
    diag = nuova_diagnosi(base / "p6")
    try:
        r = corsa(gioc, preparazione=lambda pul, qua: [VelenoAvanti(qua[1].self_attn, PASSO_BERSAGLIO)])
    finally:
        diag.disinstalla()
    ev = diag.eventi[0] if diag.eventi else {}
    verifica("P6: passo saltato per loss, dump scritto", r["saltati"][PASSO_BERSAGLIO] == "loss" and len(diag.dump_scritti) == 1,
             f"{r['saltati']} · {diag.dump_scritti}")
    lnf = ev.get("loss_non_finita") or {}
    verifica("P6: primo e_l non finito = uscita dello strato 1 (e_l[2])", lnf.get("primo_e_l_non_finito") == 2,
             f"{lnf.get('primo_e_l_non_finito')}")
    rep = ev.get("replay") or []
    pf = (rep[0].get("primo_forward_non_finito") or {}) if rep else {}
    verifica("P6: nel replay la prima regione con uscita non finita sta in L1.self_attn",
             str(pf.get("regione", "")).startswith("L1.self_attn"), f"{pf.get('regione')}")


def a1_accumulo(gioc: Giocattolo, base: Path) -> None:
    print("\n── A1 origine di tipo ACCUMULO: due contributi finiti la cui somma trabocca ──", flush=True)

    class Grande(torch.autograd.Function):
        """Gradiente d'ingresso FINITO ma enorme (2,5e38): due copie sommate fanno Inf in fp32."""

        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, g):
            return torch.full_like(g, 2.5e38)
    diag = D.Diagnosi(V, FG, cartella=base / "a1", severo=True)       # non installata: si prova il nucleo
    x = torch.randn(8, 4, requires_grad=True)
    y = x.clone()                                                      # CloneBackward0 riceve i due contributi
    loss = (Grande.apply(y) + Grande.apply(y)).sum()
    tr = D._Traccia("sentinella")
    diag._aggancia_grafo(tr, [loss.grad_fn])
    loss.backward()
    origine = diag._esito_sentinella(tr)
    rapp = diag._rapporto(tr, origine, "collaudo", None) if origine else {}
    o = rapp.get("origine") or {}
    contrib = o.get("contributi_in_arrivo") or []
    verifica("A1: origine = accumulo nel buffer d'ingresso di CloneBackward0",
             o.get("tipo") == "accumulo" and o.get("nodo") == "CloneBackward0", f"{o.get('tipo')} {o.get('nodo')}")
    verifica("A1: i due contributi in arrivo sono FINITI (2,5e38) e vengono da GrandeBackward",
             len(contrib) == 2 and all(c["da_nodo"] == "GrandeBackward" and c["finito"]
                                       and c["max_abs"] == float(torch.tensor(2.5e38, dtype=torch.float32))
                                       for c in contrib), f"{[(c['da_nodo'], c['max_abs']) for c in contrib]}")
    verifica("A1: il gradiente arrivato su x e' davvero Inf", bool(torch.isinf(x.grad).all()))
    tr.svuota()


# ═══════════════════════ lanciatore `finestra` (percorso vero, caricatori giocattolo) ═══════════════════════
class VelenoAllaChiamata:
    """Avvelena l'ingresso del modulo alla sua n-esima chiamata con gradiente (dall'installazione)."""

    def __init__(self, modulo, n: int):
        self.n, self.k = n, 0
        self.maniglia = modulo.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _m, args, kwargs):
        if not torch.is_grad_enabled():
            return None
        k = self.k
        self.k += 1
        if k != self.n:
            return None
        return args, {**kwargs, "hidden_states": Avvelena.apply(kwargs["hidden_states"])}

    def rimuovi(self):
        self.maniglia.remove()


class Lanciatore:
    """Sostituisce SOLO i caricatori (dati e pesi veri -> giocattolo) e la guardia GPU; il ciclo del
    lanciatore, la ricetta, il raccordo e ottimizza_finestra_JOINT sono quelli veri."""

    def __init__(self, gioc: Giocattolo, veleno=None):
        import forgia_scorrevole as FS
        self.FS, self.gioc, self.veleno = FS, gioc, veleno
        self.righe: list = []
        self.perdite_per_giro: list = []
        self.veleni: list = []

    def __enter__(self):
        FS, gioc, V_ = self.FS, self.gioc, V
        self._orig = (V_.carica_batch_calibrazione, V_.carica_finestra_corrente, FS._seme_iniziale_produzione,
                      D._gpu_occupata, V_.log, FS.log, V_.ottimizza_finestra_JOINT, D._gpu_visibile,
                      D._altri_sulla_gpu)

        def carica_finestra(layers):
            pul, qua, _h, _e = gioc.finestra()
            if self.veleno is not None:
                self.veleni.append(self.veleno(qua))
            return gioc.cfg, gioc.emb.clone(), {L: pul[L] for L in layers}, {L: qua[L] for L in layers}

        def seme(ids, emb):
            _p, _q, h, e = gioc.finestra()
            return h, e

        def registra(*a):
            self.righe.append(" ".join(str(x) for x in a))
            self._orig[4](*a)
        ottimizza = self._orig[6]

        @functools.wraps(ottimizza)            # la firma resta quella vera: il raccordo la controlla
        def ottimizza_con_perdite(*a, **k):
            diagnostica: list = []
            try:
                return ottimizza(*a, **{**k, "raccogli_diagnostica": diagnostica})
            finally:
                self.perdite_per_giro.append(diagnostica[0]["perdite"] if diagnostica else None)
        V_.carica_batch_calibrazione = lambda: (gioc.ids.clone(), 2)
        V_.carica_finestra_corrente = carica_finestra
        FS._seme_iniziale_produzione = seme
        D._gpu_occupata = lambda: None
        D._gpu_visibile = lambda: True         # i caricatori giocattolo stanno su CPU
        D._altri_sulla_gpu = self.altri          # guardia a ogni giro: comandata dal collaudo (G4)
        V_.log = FS.log = registra
        V_.ottimizza_finestra_JOINT = ottimizza_con_perdite
        return self

    altri_dal_giro = None                      # G4: da quel giro in poi "qualcuno e' comparso sulla GPU"

    def altri(self):
        giri = sum(1 for r in self.righe if "diagnosi NaN: giro" in r)
        return ("1234 llama-perplexit 68074 MiB VRAM (finto)"
                if self.altri_dal_giro is not None and giri + 1 >= self.altri_dal_giro else None)

    def __exit__(self, *_):
        (V.carica_batch_calibrazione, V.carica_finestra_corrente, self.FS._seme_iniziale_produzione,
         D._gpu_occupata, V.log, self.FS.log, V.ottimizza_finestra_JOINT, D._gpu_visibile,
         D._altri_sulla_gpu) = self._orig
        for v in self.veleni:
            v.rimuovi()
        return False


def argomenti(**kw):
    a = argparse.Namespace(k=0, passi=4, tetto_minuti=5.0, ripresa="", corsa_ripresa="", n_strati=3)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def l1_lanciatore(gioc: Giocattolo, base: Path) -> None:
    print("\n── L1 lanciatore `finestra` k=0: giri ripetuti fino al NaN; tetto di tempo ──", flush=True)
    os.environ["THOR_DIAGNOSI_NAN_DIR"] = str(base / "l1")
    try:
        # veleno alla 6a chiamata con gradiente di L1.self_attn = giro 2, passo 1 (4 passi per giro)
        with Lanciatore(gioc, veleno=lambda qua: VelenoAllaChiamata(qua[1].self_attn, 5)) as lan:
            rc = D.comando_finestra(argomenti())
        giri = [r for r in lan.righe if "diagnosi NaN: giro" in r]
        verifica("L1: il lanciatore ripete la finestra e si ferma al giro del NaN (rc 0, 2 giri)",
                 rc == 0 and len(giri) == 2, f"rc {rc} · {len(giri)} giri")
        dump = sorted((base / "l1").glob("*.json"))
        import json
        ev = json.loads(dump[0].read_text()) if dump else {}
        o = ev.get("origine") or {}
        verifica("L1: dump del lanciatore: passo 1, batch 1, L1.self_attn",
                 len(dump) == 1 and ev.get("t") == 1 and ev.get("b") == 1 and o.get("regione") == "L1.self_attn",
                 f"{len(dump)} dump · t={ev.get('t')} b={ev.get('b')} {o.get('regione')}")
        with Lanciatore(gioc) as lan2:
            rc2 = D.comando_finestra(argomenti(tetto_minuti=0.25))
        pg = [x for x in lan2.perdite_per_giro if x]
        verifica("L1: senza NaN il tetto di tempo ferma il lanciatore (rc 1, nessun dump)",
                 rc2 == 1 and any("tetto di tempo" in r for r in lan2.righe)
                 and not list((base / "l1").glob("*.json"))[1:], f"rc {rc2} · {len(lan2.perdite_per_giro)} giri")
        verifica("L1: ogni giro riparte dalle STESSE condizioni (perdite dei giri completi identiche al bit)",
                 len(pg) >= 2 and all(x == pg[0] for x in pg), f"{len(pg)} giri completi")
    finally:
        os.environ.pop("THOR_DIAGNOSI_NAN_DIR", None)


def l2_lanciatore_ripresa(gioc: Giocattolo, base: Path) -> None:
    print("\n── L2 lanciatore `finestra` k=1 dal punto di ripresa della forgia (seme e strato portato) ──", flush=True)
    import forgia_scorrevole as FS
    FS._NON_RICETTA.update(D.VAR_DIAGNOSI)
    cartella = base / "l2_ripresa"
    if cartella.exists():
        shutil.rmtree(cartella)
    pul, _qua, h, e = gioc.finestra()
    S = torch.from_numpy(V._S1024_NP)
    ggu = V._group_giocattolo(gioc.cfg.hidden_size, V.GRUPPO_GATEUP_REALE)
    gdw = V._group_giocattolo(gioc.cfg.moe_intermediate_size, V.GRUPPO_DOWN_REALE)
    portati = {1: {"gu": V._costruisci_ternari_layer(pul[1].mlp.experts.gate_up_proj, S, ggu, ruota=True),
                   "dw": V._costruisci_ternari_layer(pul[1].mlp.experts.down_proj, S, gdw, ruota=False)}}
    with torch.no_grad():
        portati[1]["gu"].z.add_(0.05)          # uno stato che il teacher non darebbe: deve arrivare dalla ripresa
    scrittore = FS.Ripresa(cartella, "collaudo-ripresa", 3, 2)
    tmp = scrittore.prepara(1, portati)
    scrittore.completa(1, tmp, h, e, [1])
    os.environ["THOR_DIAGNOSI_NAN_DIR"] = str(base / "l2")
    try:
        with Lanciatore(gioc, veleno=lambda qua: VelenoAllaChiamata(qua[2].linear_attn, 1)) as lan:
            rc = D.comando_finestra(argomenti(k=1, passi=3, ripresa=str(cartella), corsa_ripresa="collaudo-ripresa"))
        dump = sorted((base / "l2").glob("*.json"))
        import json
        ev = json.loads(dump[0].read_text()) if dump else {}
        o = ev.get("origine") or {}
        verifica("L2: seme e strato portato caricati dal punto di ripresa (sha256, ricetta) e usati dalla finestra",
                 any("ripresa dalla finestra 1" in r for r in lan.righe)
                 and any("ripartenza strato 1 da" in r for r in lan.righe),
                 " | ".join(r for r in lan.righe if "ripres" in r or "ripartenza" in r)[:300])
        verifica("L2: NaN diagnosticato nella finestra [1,2]: passo 1, L2.linear_attn",
                 rc == 0 and len(dump) == 1 and ev.get("t") == 1 and o.get("regione") == "L2.linear_attn"
                 and ev.get("layers") == [1, 2], f"rc {rc} · {len(dump)} dump · t={ev.get('t')} {o.get('regione')}")
        rc3 = D.comando_finestra(argomenti(k=1))
        verifica("L2: k>0 SENZA punto di ripresa rifiutato (niente condizioni inventate)", rc3 == 2, f"rc {rc3}")
    finally:
        os.environ.pop("THOR_DIAGNOSI_NAN_DIR", None)


# ═══════════════════════ norma L2 traboccata con gradienti FINITI (verifica avversaria 25/9) ═══════════════════════
class Amplifica(torch.autograd.Function):
    """Identita' nel forward; nel backward il gradiente x fattore (FINITO, niente NaN/Inf)."""

    @staticmethod
    def forward(ctx, x, fattore):
        ctx.fattore = fattore
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return g * ctx.fattore, None


class VelenoAmplifica(VelenoIngressoModulo):
    """Al passo bersaglio il gradiente d'ingresso del modulo x1e30: tutti i gradienti restano FINITI,
    ma vector_norm L2 (quella di passo_ottimizzazione) trabocca a Inf e la forgia salta il passo."""

    def _pre(self, _m, args, kwargs):
        if not torch.is_grad_enabled() or self.spento or self.passo != self.bersaglio:
            return None
        return args, {**kwargs, "hidden_states": Amplifica.apply(kwargs["hidden_states"], 1e30)}


def n1_norma_traboccata(gioc: Giocattolo, base: Path) -> None:
    print("\n── N1 gradienti FINITI x1e30: la forgia salta per 'gradienti' (L2 traboccata), il rapporto lo dice ──",
          flush=True)
    diag = nuova_diagnosi(base / "n1")
    try:
        r = corsa(gioc, preparazione=lambda pul, qua: [
            VelenoAmplifica(qua[1].self_attn, PASSO_BERSAGLIO, persistente=True)])
    finally:
        diag.disinstalla()
    ev = diag.eventi[0] if diag.eventi else {}
    gr = ev.get("grandezze") or {}
    verifica("N1: passo saltato per 'gradienti', UN dump, nessun nodo non finito (origine vuota)",
             r["saltati"] == [None, None, "gradienti", None, None] and len(diag.dump_scritti) == 1
             and not ev.get("origine"), f"{r['saltati']} · {len(diag.dump_scritti)} dump · origine {ev.get('origine')}")
    mx = gr.get("max_abs_globale") or 0.0
    verifica("N1: rapporto: 'gradienti_finiti_ma_L2_traboccata' = certo, max|g| finito e oltre la soglia L2 1.84e19",
             ev.get("gradienti_finiti_ma_L2_traboccata") == "certo" and D._SOGLIA_L2 < mx < float("inf"),
             f"{ev.get('gradienti_finiti_ma_L2_traboccata')} · max|g| {mx:.3e}")
    amp = (gr.get("amplificazioni") or [{}])[0]
    verifica("N1: il nodo che amplifica di piu' = AmplificaBackward in L1.self_attn (fattore ~1e30)",
             amp.get("nodo") == "AmplificaBackward" and amp.get("regione") == "L1.self_attn"
             and amp.get("fattore", 0) > 1e29, f"{amp.get('nodo')} in {amp.get('regione')} x{amp.get('fattore')}")
    par = gr.get("parametri") or []
    verifica("N1: per parametro max|g| finito e giudizio L2 'certo' su almeno uno",
             bool(par) and all(p["finito"] for p in par) and any(p["l2_trabocca"] == "certo" for p in par),
             f"{[(p['parametro'], p['l2_trabocca']) for p in par[:4]]}")
    rep = ev.get("replay") or []
    verifica("N1: replay completo 3/3: gradienti finiti, norma COME LA FORGIA non finita (salto riprodotto), L2 fp64 finita",
             len(rep) == 3 and all(e.get("gradienti_finiti") is True and e.get("salto_come_la_forgia") is True
                                   and any(n["l2_come_la_forgia_finita"] is False and math.isfinite(n["l2_fp64"])
                                           for n in e.get("norme_parametri") or []) for e in rep),
             f"{[(e.get('gradienti_finiti'), e.get('salto_come_la_forgia'), e.get('errore')) for e in rep]}")


# ═══════════════════════ guardie GPU del lanciatore (verifica avversaria 25/9) ═══════════════════════
def g1_vram_drm(base: Path) -> None:
    print("\n── G1 guardia DRM (Vulkan): /proc finto con fdinfo ──", flush=True)
    radice = base / "proc_finto"
    if radice.exists():
        shutil.rmtree(radice)

    def processo(pid, comm, fdinfo: dict, cmd="prog arg"):
        d = radice / str(pid)
        (d / "fdinfo").mkdir(parents=True)
        (d / "comm").write_text(comm + "\n")
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
        for fd, testo in fdinfo.items():
            (d / "fdinfo" / fd).write_text(testo)

    def drm(cid, vram, driver="amdgpu"):
        return f"pos:\t0\nflags:\t02100002\ndrm-driver:\t{driver}\ndrm-client-id:\t{cid}\ndrm-memory-vram:\t{vram}\n"
    processo(101, "llama-perplexit", {"3": drm(7, "69708020 KiB"), "4": drm(7, "69708020 KiB")})  # 2 fd, 1 client
    processo(102, "llama-server", {"3": drm(8, "30000000 KiB")})        # la casa: esente
    processo(103, "gnome-shell", {"3": drm(9, "102400 KiB")})           # 100 MiB: sotto soglia
    processo(104, "python3", {"3": "pos:\t0\nflags:\t02\n"})              # fd non DRM
    processo(105, "altro", {"5": drm(10, "300 MiB")})                   # unita' MiB
    processo(106, "estraneo", {"3": drm(11, "9000000 KiB", driver="i915")})  # altro driver
    out = D._vram_drm(proc=radice, soglia_mib=256)
    pids = sorted(int(x.split()[0]) for x in out)
    verifica("G1: visti il lavoro Vulkan (68074 MiB, contato UNA volta su due fd) e quello da 300 MiB; esclusi "
             "casa, sotto soglia, fd non DRM, altro driver", pids == [101, 105]
             and any(x.startswith("101 ") and " 68074 MiB" in x for x in out), f"{out}")
    verifica("G1: `escludi` (se stessi e gli antenati) funziona",
             [int(x.split()[0]) for x in D._vram_drm(proc=radice, soglia_mib=256, escludi={101})] == [105])


def g2_coda_viva(base: Path) -> None:
    print("\n── G2 guardia coda: script .sh vivi della radice (anche in attesa) ──", flush=True)
    cart = base / "coda_finta"
    if cart.exists():
        shutil.rmtree(cart)
    (cart / "strumenti").mkdir(parents=True)
    for nome in ("anello_coda.sh", "strumenti/gpu_occupata.sh"):
        (cart / nome).write_text("#!/bin/bash\nsleep 120\n")
    (base / "fuori_radice.sh").write_text("#!/bin/bash\nsleep 120\n")
    figli = [subprocess.Popen(["bash", "anello_coda.sh", "4182106"], cwd=cart),          # relativo + cwd
             subprocess.Popen(["bash", "-c", f"sleep 120; : {cart / 'anello_coda.sh'}"]),   # comando, non script
                                                                                            # (2 comandi: bash resta)
             subprocess.Popen(["bash", str(cart / "strumenti/gpu_occupata.sh")]),         # la guardia stessa
             subprocess.Popen(["bash", str(base / "fuori_radice.sh")])]                   # fuori dalla radice
    try:
        argv_c = (Path(f"/proc/{figli[1].pid}/cmdline").read_bytes().split(b"\0"))
        verifica("G2: il `bash -c` e' davvero un bash vivo con -c (il controllo non e' vuoto)",
                 argv_c[0].endswith(b"bash") and argv_c[1] == b"-c", f"{argv_c[:2]}")
        out = D._coda_viva(radice=cart)
        pids = sorted(int(x.split()[0]) for x in out)
        verifica("G2: visto SOLO l'anello della coda (non `bash -c`, non gpu_occupata.sh, non script fuori radice)",
                 pids == [figli[0].pid], f"{out}")
        verifica("G2: `escludi` funziona", D._coda_viva(radice=cart, escludi={figli[0].pid}) == [])
    finally:
        for p in figli:
            p.kill()
            p.wait()
    verifica("G2: finita la coda, nessuno script visto", D._coda_viva(radice=cart) == [])


def g3_guardie_sottocomandi(base: Path) -> None:
    print("\n── G3 `forgia` e `riproduci --device cuda` passano dalla guardia ──", flush=True)
    import forgia_scorrevole as FS
    orig = (D._gpu_occupata, D._gpu_visibile, FS.main, sys.argv)
    avviata: list = []
    FS.main = lambda: avviata.append(1) or 0
    try:
        D._gpu_occupata = lambda: "1234 llama-perplexit 68074 MiB VRAM (finto)"
        D._gpu_visibile = lambda: True
        rc = D.comando_forgia(["--salta-export"])
        verifica("G3: `forgia` con GPU occupata: rc 3, forgia NON avviata", rc == 3 and not avviata, f"rc {rc}")
        sys.argv = ["diagnosi_nan_gradiente.py", "riproduci", str(base / "non_esiste.pt"), "--device", "cuda"]
        rc2 = D.main()
        verifica("G3: `riproduci --device cuda` con GPU occupata: rc 3 prima di caricare il dump", rc2 == 3, f"rc {rc2}")
        D._gpu_occupata = lambda: None
        D._gpu_visibile = lambda: False
        rc3 = D.comando_forgia(["--salta-export"])
        verifica("G3: `forgia` senza GPU visibile: rc 3, forgia NON avviata", rc3 == 3 and not avviata, f"rc {rc3}")
    finally:
        D._gpu_occupata, D._gpu_visibile, FS.main, sys.argv = orig


def g4_ferma_per_altri(gioc: Giocattolo, base: Path) -> None:
    print("\n── G4 lanciatore `finestra`: un lavoro/coda comparso durante la diagnosi la ferma al giro dopo ──", flush=True)
    os.environ["THOR_DIAGNOSI_NAN_DIR"] = str(base / "g4")
    try:
        with Lanciatore(gioc) as lan:
            lan.altri_dal_giro = 2
            rc = D.comando_finestra(argomenti(tetto_minuti=5.0))
        giri = [r for r in lan.righe if "diagnosi NaN: giro" in r]
        verifica("G4: rc 4 dopo UN giro completo, riga «FERMATA» nel log, nessun dump",
                 rc == 4 and len(giri) == 1 and any("FERMATA" in r for r in lan.righe)
                 and not list((base / "g4").glob("*.json")), f"rc {rc} · {len(giri)} giri")
    finally:
        os.environ.pop("THOR_DIAGNOSI_NAN_DIR", None)


# ═══════════════════════ il collaudo sa FALLIRE ═══════════════════════
def m_mutazioni(gioc: Giocattolo, base: Path) -> None:
    print("\n── M il collaudo sa fallire: tre strumenti ROTTI apposta devono essere scoperti ──", flush=True)
    riferimento = corsa(gioc)
    orig_misura = D._misura_rapida

    def misura_che_altera(t):             # un gancio che tocca i gradienti (0,4% in piu')
        if t is not None and t.is_floating_point() and t.numel():
            try:
                t.mul_(1.004)
            except RuntimeError:          # gradiente espanso (una sola cella per molti elementi)
                pass
        return orig_misura(t)
    D._misura_rapida = misura_che_altera
    diag = nuova_diagnosi(base / "m1")
    try:
        alterata = corsa(gioc)
    finally:
        diag.disinstalla()
        D._misura_rapida = orig_misura
    ok, _ = grad_identici(riferimento["grad"], alterata["grad"])
    verifica("M1: il confronto al bit SCOPRE un gancio che altera i gradienti", not ok)

    orig_attr = D.Diagnosi._attribuisci

    def attribuisci_senza_annidate(self, tr):   # vecchia versione: si ferma ai nodi gia' attribuiti
        attr, ordine = {}, {}
        for r, fr in enumerate(tr.chiamate):
            pila = [n for n, _ in fr.get("uscite", [])]
            while pila:
                n = pila.pop()
                if n is None or n in fr["stop"] or n in attr or n.name() == D._ACCUMULA:
                    continue
                attr[n] = r
                ordine.setdefault(r, []).append(n)
                pila += [m for m, _ in n.next_functions]
        tr.ordine_regione = ordine
        return attr
    D.Diagnosi._attribuisci = attribuisci_senza_annidate
    diag = nuova_diagnosi(base / "m2", tentativi_replay=0)
    try:
        corsa(gioc, preparazione=lambda pul, qua: [VelenoIngressoModulo(qua[1].self_attn, PASSO_BERSAGLIO)])
    finally:
        diag.disinstalla()
        D.Diagnosi._attribuisci = orig_attr
    reg = ((diag.eventi[0] if diag.eventi else {}).get("origine") or {}).get("regione")
    verifica("M2: un'attribuzione sbagliata viene SCOPERTA (regione diversa da L1.self_attn)",
             reg != "L1.self_attn", f"indicata {reg!r}")

    orig_esito = D.Diagnosi._esito_sentinella

    def esito_ultimo(self, tr):                 # sceglie l'ULTIMO nodo non finito invece del primo
        o = orig_esito(self, tr)
        if o is None:
            return None
        for pos in range(len(tr.finitezza) - 1, -1, -1):
            i, fu, fi = tr.finitezza[pos]
            if not (fu and fi):
                return {"posizione": pos, "indice": i, "tipo": "nodo" if fu else "accumulo"}
        return o
    D.Diagnosi._esito_sentinella = esito_ultimo
    diag = nuova_diagnosi(base / "m3", tentativi_replay=0)
    try:
        corsa(gioc, preparazione=lambda pul, qua: [VelenoIngressoModulo(qua[1].self_attn, PASSO_BERSAGLIO)])
    finally:
        diag.disinstalla()
        D.Diagnosi._esito_sentinella = orig_esito
    o = (diag.eventi[0] if diag.eventi else {}).get("origine") or {}
    verifica("M3: scegliere un nodo che PROPAGA invece dell'origine viene SCOPERTO",
             not (o.get("regione") == "L1.self_attn" and o.get("nodo") == "AvvelenaBackward"
                  and o.get("tipo") == "nodo"), f"indicato {o.get('nodo')} in {o.get('regione')} ({o.get('tipo')})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartella", default="")
    a = ap.parse_args()
    base = Path(a.cartella) if a.cartella else Path(tempfile.mkdtemp(prefix="collaudo_diagnosi_nan_"))
    base.mkdir(parents=True, exist_ok=True)
    print(f"cartella dei dump di collaudo: {base}")
    t0 = time.time()
    gioc = Giocattolo()
    for prova in (t1_senza_iniezione, p1_attenzione_piena, p2_attenzione_lineare, p3_function_esperti, p4_nucleo,
                  p5_passo_non_saltato, p6_loss, a1_accumulo, n1_norma_traboccata, l1_lanciatore, l2_lanciatore_ripresa,
                  g1_vram_drm, g2_coda_viva, g3_guardie_sottocomandi, g4_ferma_per_altri, m_mutazioni):
        try:
            if prova in (g1_vram_drm, g2_coda_viva, g3_guardie_sottocomandi):
                prova(base)
                continue
            prova(gioc, base)
        except Exception as e:  # noqa: BLE001 — un collaudo che esplode e' un collaudo fallito, gli altri girano
            import traceback
            traceback.print_exc()
            verifica(f"{prova.__name__}: eseguito senza eccezioni", False, repr(e))
    falliti = [n for n, ok in ESITI if not ok]
    print(f"\n{'✅ TUTTI I COLLAUDI PASSANO' if not falliti else '⛔ COLLAUDI FALLITI: ' + str(falliti)} "
          f"({len(ESITI) - len(falliti)}/{len(ESITI)}) in {time.time() - t0:.0f} s")
    return 0 if not falliti else 1


if __name__ == "__main__":
    sys.exit(main())
