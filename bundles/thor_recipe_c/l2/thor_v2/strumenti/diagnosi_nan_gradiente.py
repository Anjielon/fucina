#!/usr/bin/env python3
"""DIAGNOSI del NaN SPORADICO nei GRADIENTI della forgia Thor (25/9/2026).

## Il sintomo
Nella forgia scorrevole (valida_finestra_8_12.ottimizza_finestra_JOINT ->
finestra_globale.passo_ottimizzazione con salta_non_finiti) circa 2 passi su
~1000 hanno GRADIENTI non finiti con LOSS finita (log conf6e_S2: finestra [0,1]
t=50 batch 18 e finestra [3,4] t=66 batch 2; conf6b_S1: finestra [4,5] t=130).
Il passo viene saltato e la forgia prosegue: il sintomo e' contenuto, la causa
no. Non deterministico: la stessa ricetta rilanciata non lo riproduce nello
stesso passo.

## Cosa fa questo strumento
Al PRIMO passo con gradiente non finito salva (JSON leggibile + .pt):
  - passo t, batch b, strati della finestra, loss, esito del salto;
  - il PRIMO punto del backward in cui il gradiente diventa non finito: il nodo
    autograd con gradiente rispetto alle USCITE del suo forward FINITO e
    gradiente rispetto ai suoi INGRESSI NON finito (o, se tutti i contributi
    in arrivo erano finiti e la loro SOMMA non lo e', l'accumulo nel buffer
    d'ingresso di quel nodo), con nome, posizione nell'ordine di esecuzione,
    forma/dtype/max|g| di ogni gradiente coinvolto, e la REGIONE che lo ha
    creato (sotto-modulo, funzione o metodo: "L0.linear_attn › gated_delta_rule",
    "L1.mlp.experts", "L1.pesi_ternari › gu.forward_duro", ...);
  - un REPLAY dello stesso passo (stessi parametri, stessa batch, stesso
    routing) con controllo completo nodo per nodo: conteggi NaN/Inf, tensori
    del punto d'origine, statistiche dei tensori salvati dal forward di quel
    nodo, e l'INGRESSO DEL FORWARD della regione d'origine (argomenti, stato
    del modulo, uscite, gradiente arrivato sulle uscite) per riprodurla ISOLATA
    con `riproduci()` / sottocomando `riproduci`.

## Come (nessun file esistente toccato: solo monkeypatch dall'esterno)
- `V.passo_ottimizzazione` (nome globale letto da ottimizza_finestra_JOINT a
  ogni chiamata) -> involucro che conosce t/b (dal frame del chiamante), il
  contesto del passo, e fa l'analisi DOPO il backward: nel gancio pre-step
  dell'ottimizzatore (passo non saltato: prima che AdamW corrompa i parametri)
  oppure al ritorno (passo saltato: parametri intatti).
- `FG.errore_lemma1_finestra` (nome globale letto da passo_ottimizzazione) ->
  dopo il forward si percorre il grafo dalla loss e si aggancia un gancio
  (`Node.register_hook`) a OGNI nodo autograd: funzioni native, autograd.Function
  personalizzate (esperti_thor._RaccogliRighe/_PermutaRighe), grafi compilati
  (CompiledFunctionBackward del nucleo ternario e della FWHT), operatori custom
  (grouped_mm di ripiego), kernel fla. register_full_backward_hook non basta:
  vede solo i confini dei moduli, non le Function dentro. I nodi AccumulateGrad
  NON si agganciano: sono pozzi, e il gradiente che ricevono e' gia' misurato
  come uscita del nodo che lo produce (una seconda lettura dei 3 GiB di z/log_s
  per passo non aggiungerebbe nulla). Misurato (torch 2.11, 25/9): con i ganci
  il .grad resta ADOTTATO, nessuna copia in piu'.
- Sentinella a costo basso: per ogni gradiente un solo `vector_norm(ord=inf)`
  (max|g|, propaga NaN e Inf; verificato all'installazione sul device), nessuna
  sincronizzazione nel backward; UNA sincronizzazione a fine passo.
- Regioni: ganci forward (prepend, with_kwargs, always_call) su ogni sotto-modulo
  degli strati quantizzati; involucri su `modeling.torch_chunk_gated_delta_rule`
  (fla o ripiego), `modeling.causal_conv1d_fn`, `moe._grouped_linear` (#1 gate_up,
  #2 down), `V._ruota_se_possibile` (rotazione sull'ingresso),
  `TernarioDiffBatchFuso.forward/forward_duro` (nucleo ternario compilato),
  `V._stratofn_quant_reale` (costruzione dei pesi dei ternari). L'attribuzione
  nodo->regione si calcola SOLO quando serve (dopo un NaN), percorrendo il grafo
  dalle uscite di ogni regione fino ai suoi ingressi.
- Nessun numero cambia: i ganci osservano e basta (collaudo: gradienti identici
  al bit con e senza, test_diagnosi_nan_gradiente.py).

## Uso
Si attiva SOLO con THOR_DIAGNOSI_NAN=1 (altrimenti il lanciatore rifiuta).
    diagnosi_nan_gradiente.py finestra --k 0 --tetto-minuti 360
        ripete la finestra [k, k+W-1] con la ricetta dell'ambiente finche' il NaN
        scatta o finisce il tempo (k=0: condizioni ESATTE della forgia; k>0 solo
        con --ripresa DIR --corsa-ripresa ID: seme e strato portato dal punto di
        ripresa della forgia, niente approssimazioni silenziose).
    diagnosi_nan_gradiente.py forgia -- [argomenti di forgia_scorrevole.py]
        la forgia scorrevole VERA, strumentata (non si ferma al NaN salvo
        THOR_DIAGNOSI_NAN_FERMA=1).
    diagnosi_nan_gradiente.py riproduci DUMP.pt [--ripetizioni N] [--device cuda]
        riesegue la regione d'origine isolata dall'ingresso salvato.
Variabili: THOR_DIAGNOSI_NAN_DIR (cartella, default @@ROOT:lavoro@@/thor-diagnosi-nan/
<THOR_ID_CORSA>; mai /tmp per i tensori grandi: e' RAM), THOR_DIAGNOSI_NAN_REPLAY
(tentativi di replay, default 3), THOR_DIAGNOSI_NAN_MAX_DUMP (default 1),
THOR_DIAGNOSI_NAN_FERMA, THOR_DIAGNOSI_NAN_MAX_MB (tetto per tensore salvato,
default 4096). Collaudo: test_diagnosi_nan_gradiente.py (CPU, giocattolo).

## ⚠️ «gradienti non finiti» nel log della forgia NON prova un NaN (verifica 25/9)
passo_ottimizzazione salta se vector_norm L2 dei gradienti non e' finita: quella norma (fp32, senza
riscalamento) trabocca a Inf con gradienti TUTTI FINITI oltre |g| = 1.84e19. Se la sentinella non vede
nessun nodo non finito, il rapporto dice le grandezze (max|g| per parametro, nodi piu' grandi, nodi che
AMPLIFICANO) e il replay misura le norme per parametro (esatta fp64 e come la forgia).

## Comando GPU (solo a GPU libera E coda ferma; ricetta = conf6e_S2 / lancia_forgia_scorrevole.sh, 75 passi)
Finestra [0,1] ripetuta finche' il NaN scatta (il NaN di S2 e' caduto proprio li', t=50), tetto 6 h.
Il lanciatore rifiuta da se' (rc 3) se: strumenti/gpu_occupata.sh vede un lavoro ROCm; un processo non
della casa ha VRAM nel DRM fdinfo (Vulkan: llama-perplexity/llama-cli della coda, che gpu_occupata.sh
NON vede); uno script .sh di thor_v2 e' vivo (la coda, anche in attesa del PID precedente: al suo turno
troverebbe noi ed uscirebbe con 3). Gli ultimi due si ricontrollano a ogni giro: se compaiono, rc 4.
    cd @@ROOT:forge@@/thor_v2 && strumenti/gpu_occupata.sh && \\
    env -u HSA_OVERRIDE_GFX_VERSION PYTHONPATH=@@ROOT:forge@@/thor_v2/strumenti/pylib_fla \\
      THOR_DIAGNOSI_NAN=1 THOR_DIAGNOSI_NAN_DIR=@@ROOT:lavoro@@/thor-diagnosi-nan/finestra0 \\
      THOR_ID_CORSA=diagnosi-nan-f0 THOR_CALIB_MISCELA=cot:2,edu:3,wiki:3 THOR_SALTA_NON_FINITI=1 \\
      THOR_COMMIT_DA_CODICI=1 THOR_COMMIT_SCALA_FP16=1 THOR_CACHE_STRATI=@@ROOT:models@@/gguf/tony-tern/cache_strati \\
      THOR_GROUPED_BMM=0 THOR_ROTAZIONE_INGRESSO=1 THOR_NUCLEO_BF16=1 THOR_ROUTING_VIVO=0 \\
      THOR_ROUTING_AGGIORNATO=1 THOR_ESPERTI_THOR=1 THOR_DIAGNOSTICA_PASSO=0 THOR_FORGIA_DOWN_TERNARIO=1 \\
      THOR_FORGIA_N_STRATI=40 THOR_FORGIA_W=2 THOR_FORGIA_ALPHA=1.0 THOR_VALIDA_BATCH_SEQ=8 \\
      THOR_VALIDA_N_BATCH=32 THOR_VALIDA_N_HELDOUT=1 THOR_VALIDA_N_PASSI=75 THOR_VALIDA_LR=0.01 \\
      THOR_VALIDA_FRAZIONE_DURA=1.0 THOR_LR_Z=0.01 THOR_LR_LOGS=0.01 THOR_LR_DELTA=0.01 \\
      THOR_HELDOUT_OGNI=0 THOR_RICONGELA_OGNI=0 THOR_CRONO_OGNI=25 \\
      nohup @@ROOT:venv@@/bin/python3 strumenti/diagnosi_nan_gradiente.py finestra --k 0 \\
        --tetto-minuti 360 > @@ROOT:forge@@/stato/velocita/diagnosi_nan_f0.log 2>&1 &
Esito: rc 0 = dump scritto (riga «⛔ DIAGNOSI NaN» nel log), rc 1 = tetto senza NaN (NON prova che il
NaN sia sparito: i ganci aggiungono lanci di kernel fra un nodo e l'altro del backward, e un NaN da
corsa critica puo' cambiare frequenza), rc 3 = GPU/coda occupata alla partenza, rc 4 = fermata per
lasciare il posto a un altro lavoro/coda comparso. Poi, isolata (la guardia vale anche qui su cuda):
    env -u HSA_OVERRIDE_GFX_VERSION PYTHONPATH=@@ROOT:forge@@/thor_v2/strumenti/pylib_fla \\
      @@ROOT:venv@@/bin/python3 strumenti/diagnosi_nan_gradiente.py riproduci \\
      @@ROOT:lavoro@@/thor-diagnosi-nan/finestra0/nan_*.pt --ripetizioni 200 --device cuda
Nel log la riga «diagnosi NaN INSTALLATA ... gated delta rule da [...]» deve nominare fla
(senza PYTHONPATH=pylib_fla transformers usa in silenzio il ripiego PyTorch: ricetta diversa).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import functools
import importlib
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
import weakref
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
RADICE = QUI.parent                     # thor_v2
if str(RADICE) not in sys.path:
    sys.path.insert(0, str(RADICE))

VAR_ATTIVA = "THOR_DIAGNOSI_NAN"
VAR_DIAGNOSI = ("THOR_DIAGNOSI_NAN", "THOR_DIAGNOSI_NAN_DIR", "THOR_DIAGNOSI_NAN_REPLAY", "THOR_DIAGNOSI_NAN_MAX_DUMP",
                "THOR_DIAGNOSI_NAN_FERMA", "THOR_DIAGNOSI_NAN_MAX_MB", "THOR_DIAGNOSI_NAN_TETTO_S")
VERSIONE = 2
_ACCUMULA = "torch::autograd::AccumulateGrad"
_INF = float("inf")
# la forgia decide il salto con vector_norm L2 (accumulo fp32 anche per bf16, NESSUN riscalamento):
# trabocca a Inf con gradienti TUTTI FINITI appena un |g| > sqrt(FLT_MAX) = 1.84e19, o la somma dei
# quadrati > 3.4e38 (misurato 25/9: un solo elemento a 2e19 -> L2 = inf). «gradienti non finiti» nel
# log della forgia NON prova un NaN/Inf: puo' essere un gradiente finito enorme.
_FLT_MAX = float(torch.finfo(torch.float32).max)
_SOGLIA_L2 = math.sqrt(_FLT_MAX)
_CASA = ("llama-server", "whisper-server")
_DIZ_GANCI = tuple(k for k, v in torch.nn.Module().__dict__.items() if isinstance(v, dict) and "hook" in k)


class NanTrovato(RuntimeError):
    """Sollevata (solo con ferma_al_primo) al ritorno del passo in cui e' stato scritto un dump."""


class TettoDiTempo(RuntimeError):
    """Sollevata all'inizio di un passo quando il tetto di tempo e' superato."""


# ═══════════════════════════ utilita' sui tensori ═══════════════════════════
def _appiattisci(x) -> list:
    """Tensori di una struttura annidata (tuple/list/dict), in ordine deterministico."""
    out: list = []

    def vai(o):
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            for e in o:
                vai(e)
        elif isinstance(o, dict):
            for k in o:
                vai(o[k])
    vai(x)
    return out


def _mappa(x, fn):
    """Stessa struttura, tensori trasformati da fn (namedtuple comprese)."""
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*[_mappa(e, fn) for e in x])
    if isinstance(x, tuple):
        return tuple(_mappa(e, fn) for e in x)
    if isinstance(x, list):
        return [_mappa(e, fn) for e in x]
    if isinstance(x, dict):
        return {k: _mappa(v, fn) for k, v in x.items()}
    return x


def _nbyte(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _clona_cpu(t: torch.Tensor, tetto_byte: int):
    """Copia staccata su CPU (foglia, requires_grad conservato) oppure, oltre il tetto,
    un segnaposto con le sole statistiche (mai un file grande dove non ci sta)."""
    if _nbyte(t) > tetto_byte:
        return {"__omesso__": True, "motivo": f"{_nbyte(t)} byte > tetto {tetto_byte}", **statistiche(t)}
    c = t.detach().to("cpu", copy=True)
    if t.requires_grad and (c.is_floating_point() or c.is_complex()):
        c.requires_grad_(True)
    return c


def _picklabile(o) -> bool:
    try:
        pickle.dumps(o)
        return True
    except Exception:  # noqa: BLE001 — qualunque rifiuto di pickle = non salvabile, lo si dice nel dump
        return False


def _clona_struttura(x, tetto_byte: int):
    def voce(o):
        if isinstance(o, torch.Tensor):
            return _clona_cpu(o, tetto_byte)
        return o
    y = _mappa(x, voce)
    return _mappa_nonpicklabili(y)


def _mappa_nonpicklabili(x):
    if isinstance(x, torch.Tensor) or x is None or isinstance(x, (int, float, bool, str)):
        return x
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*[_mappa_nonpicklabili(e) for e in x])
    if isinstance(x, (list, tuple)):
        return type(x)(_mappa_nonpicklabili(e) for e in x)
    if isinstance(x, dict):
        return {k: _mappa_nonpicklabili(v) for k, v in x.items()}
    return x if _picklabile(x) else {"__non_salvabile__": repr(x)[:200]}


def statistiche(t: "torch.Tensor | None") -> "dict | None":
    """Statistiche COMPLETE (sincronizza): forma, dtype, NaN, Inf, max|x| (tutti e soli i finiti)."""
    if t is None:
        return None
    with torch.no_grad():
        d = {"forma": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "device": str(t.device),
             "numel": int(t.numel())}
        if t.numel() == 0 or not (t.is_floating_point() or t.is_complex()):
            d.update(finito=True, n_nan=0, n_inf=0, max_abs=0.0 if t.numel() == 0 else float(t.abs().max()),
                     max_abs_finiti=None)
            return d
        fin = torch.isfinite(t)
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        a = t.detach().abs()
        m_fin = float(a.masked_fill(~fin, 0).max())
        d.update(finito=(n_nan + n_inf == 0), n_nan=n_nan, n_inf=n_inf,
                 max_abs=(m_fin if n_nan + n_inf == 0 else (math.nan if n_nan else _INF)),
                 max_abs_finiti=m_fin,
                 prime_posizioni_non_finite=[int(i) for i in torch.nonzero(~fin.reshape(-1))[:5].flatten()])
    return d


def _misura_rapida(t):
    """Sentinella: (forma, dtype, device, max|g| come tensore 0-d SENZA sincronizzare)."""
    if t is None:
        return None
    if t.numel() == 0 or not (t.is_floating_point() or t.is_complex()):
        return (tuple(t.shape), str(t.dtype).replace("torch.", ""), str(t.device), None)
    return (tuple(t.shape), str(t.dtype).replace("torch.", ""), str(t.device),
            torch.linalg.vector_norm(t.detach(), ord=_INF))


def verifica_sentinella(device) -> None:
    """La sentinella si fida di vector_norm(ord=inf) che propaga NaN e Inf: lo si PROVA sul
    device (un controllo che non puo' fallire non e' un controllo)."""
    for dt in (torch.float32, torch.bfloat16):
        for val, atteso in ((math.nan, "nan"), (_INF, "inf"), (-_INF, "inf")):
            x = torch.zeros(100_003, dtype=dt, device=device)
            x[77_777] = val
            m = float(torch.linalg.vector_norm(x, ord=_INF))
            if not ((atteso == "nan" and math.isnan(m)) or (atteso == "inf" and m == _INF)):
                raise RuntimeError(f"sentinella NON affidabile su {device}/{dt}: vector_norm(inf) di un tensore con "
                                   f"{val} da' {m} (atteso {atteso}) — lo strumento non si installa")
        x = torch.full((4097,), 3.0, dtype=dt, device=device)
        if float(torch.linalg.vector_norm(x, ord=_INF)) != 3.0:
            raise RuntimeError(f"sentinella NON affidabile su {device}/{dt}: max|x| sbagliato")


def _json(o):
    """Oggetto JSON-abile (float non finiti come stringhe: JSON stretto)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else ("nan" if math.isnan(o) else ("inf" if o > 0 else "-inf"))
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, torch.Tensor):
        return statistiche(o)
    return repr(o)[:300]


# ═══════════════════════════ clonazione di un modulo ═══════════════════════════
def clona_modulo(m: torch.nn.Module, tetto_byte: int) -> "tuple[torch.nn.Module, list]":
    """Copia su CPU del modulo COM'E' nel momento del forward: parametri, buffer, tensori
    SOSTITUITI in __dict__ (es. gate_up_proj ricostruito dai ternari: diventa una foglia
    che richiede gradiente, come il ramo del grafo che lo alimentava), sotto-moduli. Niente
    ganci, niente forward sostituiti (elencati nel secondo valore). Si salva con torch.save."""
    cls = type(m)
    nuovo = cls.__new__(cls)
    sostituiti: list = []
    d = {}
    for k, v in m.__dict__.items():
        if k in ("_parameters", "_buffers", "_modules"):
            continue
        if k in _DIZ_GANCI:
            d[k] = collections.OrderedDict()
        elif isinstance(v, torch.Tensor):
            d[k] = _clona_cpu(v, tetto_byte)
        elif k == "forward" and getattr(v, "_diagnosi_cattura", False):
            continue                                        # l'involucro di cattura di questo strumento
        elif k == "forward" and callable(v):
            difetti = [x for x in (getattr(v, "__defaults__", None) or ()) if isinstance(x, torch.Tensor)]
            sostituiti.append({"attributo": "forward", "funzione": getattr(v, "__qualname__", repr(v)),
                               "tensori_default": [_clona_cpu(x, tetto_byte) for x in difetti]})
        elif _picklabile(v):
            d[k] = v
        else:
            sostituiti.append({"attributo": k, "non_salvabile": repr(v)[:200]})
    nuovo.__dict__.update(d)
    nuovo.__dict__["_parameters"] = {k: (None if p is None else torch.nn.Parameter(
        p.detach().to("cpu", copy=True), requires_grad=p.requires_grad)) for k, p in m._parameters.items()}
    nuovo.__dict__["_buffers"] = {k: (None if b is None else b.detach().to("cpu", copy=True))
                                  for k, b in m._buffers.items()}
    figli = {}
    for k, s in m._modules.items():
        if s is None:
            figli[k] = None
            continue
        figli[k], sost_figlio = clona_modulo(s, tetto_byte)
        sostituiti += [{**x, "sottomodulo": k + ("." + x["sottomodulo"] if "sottomodulo" in x else "")}
                       for x in sost_figlio]
    nuovo.__dict__["_modules"] = figli
    return nuovo, sostituiti


def _modulo_su_device(m: torch.nn.Module, device) -> torch.nn.Module:
    """.to() sposta parametri e buffer; i tensori SOSTITUITI in __dict__ si spostano a mano
    (restano foglie, con il loro requires_grad)."""
    m = m.to(device)
    for sub in m.modules():
        for k, v in list(sub.__dict__.items()):
            if isinstance(v, torch.Tensor) and k not in ("_parameters", "_buffers"):
                nv = v.detach().to(device)
                if v.requires_grad:
                    nv.requires_grad_(True)
                object.__setattr__(sub, k, nv)
    return m


# ═══════════════════════════ stato di un passo ═══════════════════════════
class _Traccia:
    """Tutto cio' che riguarda UN passo (o un replay): regioni, nodi, misure."""

    def __init__(self, modo: str, bersaglio=None):
        self.modo = modo                    # "sentinella" | "replay"
        self.bersaglio = bersaglio          # (percorso, occorrenza) della regione da catturare (replay)
        self.pila: list = []                # regioni aperte
        self.chiamate: list = []            # regioni chiuse, in ordine di chiusura (piu' interne prima)
        self.occorrenze: collections.Counter = collections.Counter()
        self.nodi: list = []                # indice -> nodo autograd agganciato
        self.indice: dict = {}              # nodo -> indice
        self.registro: list = []            # (indice, misure_uscita, misure_ingresso) in ordine di ESECUZIONE
        self.radici: list = []
        self.argomenti_stratofn: dict = {}  # id(closure) -> argomenti di _stratofn_quant_reale
        self.ctx = None                     # argomenti del passo
        self.contesto: dict = {}            # t, b, chunk, layers
        self.ris = None
        self.loss_orig = None
        self.analizzato = False
        self.lock = threading.Lock()
        # solo replay
        self.visto_non_finito = False
        self.origine = None
        self.cattura = None
        self.primo_forward_non_finito = None
        self.salvati_origine = None

    def svuota(self) -> None:
        """I ganci nei nodi tengono un riferimento a questa traccia e la traccia ai nodi: si rompe
        il ciclo a fine passo, cosi' grafo e tensori salvati muoiono quando devono."""
        self.pila.clear()
        self.chiamate.clear()
        self.nodi.clear()
        self.indice.clear()
        self.registro.clear()
        self.radici.clear()
        self.argomenti_stratofn.clear()
        self.ctx = self.ris = self.loss_orig = self.cattura = self.origine = None
        for attr in ("valori", "finitezza", "ordine_regione"):
            if hasattr(self, attr):
                delattr(self, attr)


class _Cattura:
    """Ingresso del forward della regione d'origine (replay), per riprodurla isolata."""

    def __init__(self, diag: "Diagnosi", frame: dict, args, kwargs, oggetto):
        self.info = {"percorso": frame["percorso"], "occorrenza": frame["occorrenza"], "tipo": frame["tipo"],
                     "funzione": frame.get("funzione"), "metodo": frame.get("metodo")}
        self.ingressi = {"args": _clona_struttura(args, diag.tetto_byte),
                         "kwargs": _clona_struttura(kwargs, diag.tetto_byte)}
        self.oggetto, self.sostituiti = (None, [])
        if oggetto is not None:
            self.oggetto, self.sostituiti = clona_modulo(oggetto, diag.tetto_byte)
        self.uscite = None
        self.grad_uscite: list = []
        self._tetto = diag.tetto_byte

    def fine(self, out) -> None:
        self.uscite = _clona_struttura(out, self._tetto)
        tensori = _appiattisci(out)
        self.grad_uscite = [None] * len(tensori)
        for i, t in enumerate(tensori):
            if t.requires_grad:
                t.register_hook(functools.partial(self._gancio, i))

    def _gancio(self, i, g):
        self.grad_uscite[i] = _clona_cpu(g, self._tetto)

    def per_salvataggio(self) -> dict:
        return {**self.info, "ingressi": self.ingressi, "oggetto": self.oggetto, "sostituiti": self.sostituiti,
                "uscite": self.uscite, "grad_uscite": self.grad_uscite}


# ═══════════════════════════ la diagnosi ═══════════════════════════
class Diagnosi:
    def __init__(self, V, FG, cartella: "str | Path | None" = None, massimo_dump: "int | None" = None,
                 tentativi_replay: "int | None" = None, ferma_al_primo: "bool | None" = None,
                 tetto_s: "float | None" = None, severo: bool = False, log=None):
        self.V, self.FG = V, FG
        corsa = getattr(V, "ID_CORSA", time.strftime("%Y%m%d-%H%M%S"))
        self.cartella = Path(cartella or os.getenv("THOR_DIAGNOSI_NAN_DIR", "")
                             or f"@@ROOT:lavoro@@/thor-diagnosi-nan/{corsa}")
        self.massimo_dump = int(os.getenv("THOR_DIAGNOSI_NAN_MAX_DUMP", "1") if massimo_dump is None else massimo_dump)
        self.tentativi_replay = int(os.getenv("THOR_DIAGNOSI_NAN_REPLAY", "3")
                                    if tentativi_replay is None else tentativi_replay)
        self.ferma_al_primo = (os.getenv("THOR_DIAGNOSI_NAN_FERMA", "0") == "1"
                               if ferma_al_primo is None else ferma_al_primo)
        tetto = float(os.getenv("THOR_DIAGNOSI_NAN_TETTO_S", "0") if tetto_s is None else tetto_s)
        self.scadenza = (time.time() + tetto) if tetto > 0 else None
        mb = int(os.getenv("THOR_DIAGNOSI_NAN_MAX_MB", "4096"))
        # /tmp e /dev/shm sono RAM: li' niente tensori grandi, qualunque cosa dica l'ambiente
        in_ram = str(self.cartella.resolve()).startswith(("/tmp", "/dev/shm"))
        self.tetto_byte = min(mb, 64) * 2**20 if in_ram else mb * 2**20
        self.severo = severo
        self._log = log or getattr(V, "log", None) or (lambda *a: print(*a, flush=True))
        self.modo = "spento"
        self.traccia = _Traccia("sentinella")
        self._originali: list = []
        self._moduli: dict = {}
        self._ottimizzatori: dict = {}
        self._etichette_ternari: dict = {}
        self._nomi_parametri: dict = {}
        self.eventi: list = []
        self.dump_scritti: list = []
        self.passi_osservati = 0
        self.passi_saltati = 0
        self.nodi_agganciati_ultimo_passo = 0
        self.regioni_ultimo_passo = 0
        self.nomi_nodi_ultimo_passo: collections.Counter = collections.Counter()
        self.percorsi_ultimo_passo: list = []
        self._da_fermare = False

    # ── installazione ──────────────────────────────────────────────────────
    def installa(self, forza: bool = False) -> "Diagnosi":
        if not forza and os.getenv(VAR_ATTIVA) != "1":
            raise RuntimeError(f"{VAR_ATTIVA} non e' 1: diagnosi NON installata (si attiva solo esplicitamente)")
        if self._originali:
            raise RuntimeError("diagnosi gia' installata")
        verifica_sentinella("cpu")
        if torch.cuda.is_available():
            verifica_sentinella("cuda")
        V, FG = self.V, self.FG
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
        from transformers.integrations import moe as MOE
        from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso

        self._passo_orig = V.passo_ottimizzazione
        self._lemma1_orig = FG.errore_lemma1_finestra
        self._stratofn_orig = V._stratofn_quant_reale
        self._firma_passo = inspect.signature(self._passo_orig)
        self._firma_stratofn = inspect.signature(self._stratofn_orig)
        self._sostituisci(V, "passo_ottimizzazione", self._passo)
        self._sostituisci(FG, "errore_lemma1_finestra", self._lemma1)
        self._sostituisci(V, "_stratofn_quant_reale", self._stratofn)
        for modulo, attr, etichetta, contatore in (
                (V, "_ruota_se_possibile", "rotazione_ingresso", False),
                (M, "torch_chunk_gated_delta_rule", "gated_delta_rule", False),
                (M, "causal_conv1d_fn", "causal_conv1d", False),
                (MOE, "_grouped_linear", "grouped_linear", True)):
            orig = getattr(modulo, attr)
            self._sostituisci(modulo, attr, self._avvolgi_funzione(orig, etichetta, modulo.__name__, attr, contatore))
        for metodo in ("forward", "forward_duro"):
            orig = TernarioDiffBatchFuso.__dict__[metodo]
            self._sostituisci(TernarioDiffBatchFuso, metodo, self._avvolgi_metodo(orig, metodo))
        self.impl_gated_delta = [getattr(c.cell_contents, "__module__", "?") for c in
                                 (getattr(self._originale(M.torch_chunk_gated_delta_rule), "__closure__", None) or ())
                                 if callable(c.cell_contents)]
        self.modo = "sentinella"
        self._log(f"🔬 diagnosi NaN INSTALLATA · cartella {self.cartella} · replay {self.tentativi_replay} · "
                  f"massimo dump {self.massimo_dump} · ferma al primo {self.ferma_al_primo} · "
                  f"gated delta rule da {self.impl_gated_delta}")
        return self

    def _sostituisci(self, oggetto, attr, nuovo) -> None:
        self._originali.append((oggetto, attr, oggetto.__dict__[attr] if isinstance(oggetto, type)
                                else getattr(oggetto, attr)))
        setattr(oggetto, attr, nuovo)

    @staticmethod
    def _originale(f):
        return getattr(f, "_diagnosi_originale", f)

    def disinstalla(self) -> None:
        for oggetto, attr, orig in reversed(self._originali):
            setattr(oggetto, attr, orig)
        self._originali.clear()
        for _rif, pre, post, _p in self._moduli.values():
            pre.remove()
            post.remove()
        self._moduli.clear()
        for maniglia in self._ottimizzatori.values():
            maniglia[1].remove()
        self._ottimizzatori.clear()
        self.modo = "spento"
        self.traccia = _Traccia("sentinella")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disinstalla()
        return False

    # ── regioni ─────────────────────────────────────────────────────────────
    def _attivo(self) -> bool:
        return self.modo != "spento" and torch.is_grad_enabled()

    def _apri(self, nome: str, tipo: str, ingressi: list, oggetto=None, assoluto: bool = False,
              contatore: bool = False, **extra) -> dict:
        tr = self.traccia
        if assoluto or not tr.pila:
            percorso = nome
        else:
            genitore = tr.pila[-1]
            if contatore:
                genitore["contatori"][nome] += 1
                nome = f"{nome}#{genitore['contatori'][nome]}"
            percorso = f"{genitore['percorso']} › {nome}"
        occ = tr.occorrenze[percorso]
        tr.occorrenze[percorso] += 1
        stop = set()
        for t in ingressi:
            if isinstance(t, torch.Tensor) and t.grad_fn is not None:
                stop.add(t.grad_fn)
        frame = {"percorso": percorso, "occorrenza": occ, "tipo": tipo, "stop": stop, "oggetto": oggetto,
                 "catena": [f["percorso"] for f in tr.pila] + [percorso],
                 "antenati": [id(f) for f in tr.pila],
                 "oggetto_id": id(oggetto) if oggetto is not None else None,
                 "contatori": collections.Counter(), **extra}
        tr.pila.append(frame)
        return frame

    def _chiudi(self, frame: dict, uscite) -> None:
        tr = self.traccia
        if not tr.pila or tr.pila[-1] is not frame:
            self._log(f"⚠️ diagnosi: pila delle regioni incoerente alla chiusura di {frame['percorso']}: azzero")
            tr.pila.clear()
            return
        tr.pila.pop()
        tensori = _appiattisci(uscite)
        frame["uscite"] = [(t.grad_fn, t.output_nr) for t in tensori if t.grad_fn is not None]
        tr.chiamate.append(frame)
        if tr.modo == "replay" and tr.primo_forward_non_finito is None:
            for t in tensori:
                if (t.is_floating_point() or t.is_complex()) and t.numel() and not bool(torch.isfinite(t).all()):
                    tr.primo_forward_non_finito = {"regione": frame["percorso"], "uscita": statistiche(t)}
                    break

    def _tensori_modulo(self, m: torch.nn.Module) -> list:
        out = [p for p in m._parameters.values() if p is not None]
        out += [b for b in m._buffers.values() if b is not None]
        out += [v for k, v in m.__dict__.items() if isinstance(v, torch.Tensor)]
        return out

    def _registra_moduli(self, strato: torch.nn.Module, L) -> None:
        for nome, m in strato.named_modules():
            voce = self._moduli.get(id(m))
            if voce is not None and voce[0]() is m:
                continue
            percorso = f"L{L}" + (f".{nome}" if nome else "")
            pre = m.register_forward_pre_hook(functools.partial(self._pre_modulo, percorso),
                                              prepend=True, with_kwargs=True)
            post = m.register_forward_hook(functools.partial(self._post_modulo, percorso),
                                           with_kwargs=True, always_call=True)
            self._moduli[id(m)] = (weakref.ref(m), pre, post, percorso)

    def _pre_modulo(self, percorso, modulo, args, kwargs):
        if not self._attivo():
            return None
        self._apri(percorso, "modulo", _appiattisci((args, kwargs)) + self._tensori_modulo(modulo),
                   oggetto=modulo, assoluto=True)
        return None

    def _post_modulo(self, percorso, modulo, args, kwargs, output):
        if not self._attivo():
            return None
        tr = self.traccia
        if tr.pila and tr.pila[-1]["oggetto_id"] == id(modulo) and tr.pila[-1]["percorso"] == percorso:
            self._chiudi(tr.pila[-1], output)
        return None

    def _avvolgi_funzione(self, fn, etichetta: str, nome_modulo: str, attr: str, contatore: bool):
        diag = self

        @functools.wraps(fn)
        def involucro(*args, **kwargs):
            if not diag._attivo():
                return fn(*args, **kwargs)
            frame = diag._apri(etichetta, "funzione", _appiattisci((args, kwargs)), contatore=contatore,
                               funzione={"modulo": nome_modulo, "attributo": attr})
            cattura = diag._forse_cattura(frame, args, kwargs, None)
            out = None
            try:
                out = fn(*args, **kwargs)
            finally:
                diag._chiudi(frame, out)
            if cattura is not None:
                cattura.fine(out)
            return out
        involucro._diagnosi_originale = fn
        return involucro

    def _avvolgi_metodo(self, fn, nome_metodo: str):
        diag = self

        @functools.wraps(fn)
        def involucro(obj, *args, **kwargs):
            if not diag._attivo():
                return fn(obj, *args, **kwargs)
            etichetta = diag._etichette_ternari.get(id(obj), type(obj).__name__)
            frame = diag._apri(f"{etichetta}.{nome_metodo}", "metodo",
                               _appiattisci((args, kwargs)) + diag._tensori_modulo(obj), oggetto=obj,
                               metodo=nome_metodo)
            cattura = diag._forse_cattura(frame, args, kwargs, obj)
            out = None
            try:
                out = fn(obj, *args, **kwargs)
            finally:
                diag._chiudi(frame, out)
            if cattura is not None:
                cattura.fine(out)
            return out
        involucro._diagnosi_originale = fn
        return involucro

    def _forse_cattura(self, frame: dict, args, kwargs, oggetto):
        tr = self.traccia
        if tr.modo != "replay" or tr.cattura is not None or tr.bersaglio is None:
            return None
        if (frame["percorso"], frame["occorrenza"]) != tr.bersaglio:
            return None
        tr.cattura = _Cattura(self, frame, args, kwargs, oggetto)
        return tr.cattura

    @contextlib.contextmanager
    def _cattura_forward_modulo(self, modulo: torch.nn.Module):
        """Per le regioni-modulo l'ingresso si prende all'ENTRATA di forward(), dopo i ganci
        (es. la rotazione dell'ingresso degli esperti): e' cio' che forward() vede davvero, e
        `riproduci` chiama forward() direttamente."""
        diag = self
        aveva = "forward" in modulo.__dict__
        prima = modulo.__dict__.get("forward")
        originale = modulo.forward

        def forward_con_cattura(*args, **kwargs):
            tr = diag.traccia
            cima = tr.pila[-1] if tr.pila else None
            if cima is not None and cima["oggetto_id"] == id(modulo) and torch.is_grad_enabled():
                cattura = diag._forse_cattura(cima, args, kwargs, modulo)
                if cattura is not None:
                    out = originale(*args, **kwargs)
                    cattura.fine(out)
                    return out
            return originale(*args, **kwargs)
        forward_con_cattura._diagnosi_cattura = True
        modulo.__dict__["forward"] = forward_con_cattura
        try:
            yield
        finally:
            if aveva:
                modulo.__dict__["forward"] = prima
            else:
                modulo.__dict__.pop("forward", None)

    # ── i punti d'aggancio nella forgia ──────────────────────────────────────
    def _stratofn(self, *args, **kwargs):
        a = self._firma_stratofn.bind(*args, **kwargs)
        a.apply_defaults()
        p = dict(a.arguments)
        strato = p["strato"]
        # indice VERO dello strato: layer_idx dei sotto-moduli di attenzione (il DecoderLayer non
        # lo porta), poi `chiave` (None nelle valutazioni held-out, che arrivano per prime)
        L = next((m.layer_idx for m in strato.modules() if isinstance(getattr(m, "layer_idx", None), int)),
                 p["chiave"] if p["chiave"] is not None else "?")
        self._registra_moduli(strato, L)
        for fam, obj in (("gu", p["ternari_gu"]), ("dw", p["ternari_dw"])):
            if obj is not None:
                self._etichette_ternari[id(obj)] = fam
                for nome, par in obj.named_parameters():
                    self._nomi_parametri[id(par)] = f"L{L}.{fam}.{nome}"
        if not (self._attivo() and p["pesi_pronti"] is None):
            return self._stratofn_orig(**p)
        tensori_ingresso = [t for o in (p["ternari_gu"], p["ternari_dw"]) if o is not None for t in o.parameters()]
        frame = self._apri(f"L{L}.pesi_ternari", "costruzione_pesi", tensori_ingresso, assoluto=True)
        f = None
        try:
            f = self._stratofn_orig(**p)
        finally:
            celle = [c.cell_contents for c in (getattr(f, "__closure__", None) or ())
                     if isinstance(getattr(c, "cell_contents", None), torch.Tensor)]
            self._chiudi(frame, celle)
        self.traccia.argomenti_stratofn[id(f)] = p
        return f

    def _lemma1(self, *args, **kwargs):
        ris = self._lemma1_orig(*args, **kwargs)
        tr = self.traccia
        if tr.modo == "sentinella" and self._attivo() and ris["loss_somma"].requires_grad:
            tr.radici = [ris["loss_somma"].grad_fn, ris["loss"].grad_fn]
            tr.loss_orig = (ris["loss"].detach().clone(), ris["loss_somma"].detach().clone())
            self._aggancia_grafo(tr, tr.radici)
        return ris

    def _aggancia_grafo(self, tr: _Traccia, radici: list) -> None:
        pila = [r for r in radici if r is not None]
        while pila:
            n = pila.pop()
            if n is None or n in tr.indice:
                continue
            i = len(tr.nodi)
            tr.nodi.append(n)
            tr.indice[n] = i
            if n.name() != _ACCUMULA:
                n.register_hook(functools.partial(self._gancio_nodo, tr, i))
            for m, _ in n.next_functions:
                if m is not None and m not in tr.indice:
                    pila.append(m)

    def _gancio_nodo(self, tr: _Traccia, i: int, grad_ingresso, grad_uscita):
        """grad_uscita: gradienti rispetto alle USCITE del forward del nodo (arrivano dal basso);
        grad_ingresso: rispetto ai suoi INGRESSI (li produce il nodo). Nessuna modifica."""
        with torch.no_grad():
            if tr.modo == "sentinella":
                rec = (i, [_misura_rapida(t) for t in grad_uscita], [_misura_rapida(t) for t in grad_ingresso])
                with tr.lock:
                    tr.registro.append(rec)
                return None
            # replay: controllo completo, con sincronizzazione (un passo solo)
            fin_u = all(t is None or t.numel() == 0 or not t.is_floating_point() or bool(torch.isfinite(t).all())
                        for t in grad_uscita)
            fin_i = all(t is None or t.numel() == 0 or not t.is_floating_point() or bool(torch.isfinite(t).all())
                        for t in grad_ingresso)
            with tr.lock:
                tr.registro.append((i, fin_u, fin_i))
                if tr.origine is None and not (fin_u and fin_i):
                    tr.origine = {"indice": i, "posizione": len(tr.registro) - 1,
                                  "tipo": "nodo" if fin_u else "accumulo",
                                  "grad_uscita": [statistiche(t) for t in grad_uscita],
                                  "grad_ingresso": [statistiche(t) for t in grad_ingresso],
                                  "tensori": {"grad_uscita": [None if t is None else _clona_cpu(t, self.tetto_byte)
                                                              for t in grad_uscita],
                                              "grad_ingresso": [None if t is None else _clona_cpu(t, self.tetto_byte)
                                                                for t in grad_ingresso]}}
        return None

    def _aggancia_ottimizzatore(self, opt) -> None:
        voce = self._ottimizzatori.get(id(opt))
        if voce is not None and voce[0]() is opt:
            return
        self._ottimizzatori[id(opt)] = (weakref.ref(opt), opt.register_step_pre_hook(self._pre_step))

    def _pre_step(self, opt, _args, _kwargs):
        tr = self.traccia
        if self.modo != "spento" and tr.ctx is not None and tr.ctx["opt"] is opt and not tr.analizzato:
            self._analizza("prima di opt.step (passo NON saltato)", parcheggia_grad=True)
        return None

    def _contesto_chiamante(self) -> dict:
        f = sys._getframe(2)
        for _ in range(10):
            if f is None:
                break
            if f.f_code.co_name == "ottimizza_finestra_JOINT":
                loc = f.f_locals
                return {k: _json(loc.get(k)) for k in ("t", "b", "chunk", "layers", "n_passi", "n_tr")}
            f = f.f_back
        return {}

    def _passo(self, *args, **kwargs):
        if self.modo == "spento":
            return self._passo_orig(*args, **kwargs)
        if self.scadenza is not None and time.time() > self.scadenza:
            raise TettoDiTempo(f"tetto di tempo raggiunto dopo {self.passi_osservati} passi osservati")
        a = self._firma_passo.bind(*args, **kwargs)
        a.apply_defaults()
        tr = self.traccia
        tr.ctx = dict(a.arguments)
        tr.contesto = self._contesto_chiamante()
        self._aggancia_ottimizzatore(tr.ctx["opt"])
        try:
            ris = self._passo_orig(*args, **kwargs)
            tr.ris = ris
            if not tr.analizzato:
                self._analizza("dopo il passo" + (f" (saltato: {ris['saltato']})" if ris.get("saltato") else ""))
        finally:
            self.nodi_agganciati_ultimo_passo = len(tr.nodi)
            self.regioni_ultimo_passo = len(tr.chiamate)
            self.nomi_nodi_ultimo_passo = collections.Counter(n.name() for n in tr.nodi)
            self.percorsi_ultimo_passo = [fr["percorso"] for fr in tr.chiamate]
            tr.svuota()
            self.traccia = _Traccia("sentinella")
            if len(self.dump_scritti) >= self.massimo_dump and self.modo != "spento":
                self.modo = "spento"
                self._log(f"🔬 diagnosi NaN: {len(self.dump_scritti)} dump scritti (massimo {self.massimo_dump}): "
                          "sentinella spenta, la forgia prosegue senza ganci")
        self.passi_osservati += 1
        if ris.get("saltato"):
            self.passi_saltati += 1
        if self._da_fermare:
            self._da_fermare = False
            raise NanTrovato(f"NaN diagnosticato: {self.dump_scritti[-1]}")
        return ris

    # ── analisi ─────────────────────────────────────────────────────────────
    def _esito_sentinella(self, tr: _Traccia) -> "dict | None":
        """UNA sincronizzazione: max|g| di tutti i gradienti visti -> primo punto non finito."""
        if not tr.registro:
            return None
        per_device: dict = collections.defaultdict(list)
        for pos, (_i, u, g) in enumerate(tr.registro):
            for lato, misure in (("u", u), ("i", g)):
                for s, m in enumerate(misure):
                    if m is not None and m[3] is not None:
                        per_device[m[2]].append(((pos, lato, s), m[3]))
        valori: dict = {}
        for _dev, voci in per_device.items():
            vv = torch.stack([v.to(torch.float32).reshape(()) for _k, v in voci]).cpu().tolist()
            for (k, _v), x in zip(voci, vv):
                valori[k] = x
        tr.valori = valori

        def fin(pos, lato, n):
            return all(math.isfinite(valori[(pos, lato, s)]) for s in range(n) if (pos, lato, s) in valori)
        tr.finitezza = []
        origine = None
        for pos, (i, u, g) in enumerate(tr.registro):
            fu, fi = fin(pos, "u", len(u)), fin(pos, "i", len(g))
            tr.finitezza.append((i, fu, fi))
            if origine is None and not (fu and fi):
                origine = {"posizione": pos, "indice": i, "tipo": "nodo" if fu else "accumulo"}
        return origine

    def _descrivi_slot(self, tr: _Traccia, pos: int, lato: str) -> list:
        _i, u, g = tr.registro[pos]
        out = []
        for s, m in enumerate(u if lato == "u" else g):
            if m is None:
                out.append(None)
                continue
            v = tr.valori.get((pos, lato, s))
            out.append({"forma": list(m[0]), "dtype": m[1], "device": m[2],
                        "max_abs": v if v is not None else 0.0,
                        "finito": True if v is None else math.isfinite(v)})
        return out

    def _attribuisci(self, tr: _Traccia) -> dict:
        """nodo -> indice della regione piu' interna che lo ha creato (percorrendo il grafo dalle
        uscite di ogni regione fino ai suoi ingressi; regioni in ordine di chiusura)."""
        attr: dict = {}
        ordine: dict = collections.defaultdict(list)
        for r, fr in enumerate(tr.chiamate):
            pila = [n for n, _nr in fr.get("uscite", [])]
            visti = set()
            while pila:
                n = pila.pop()
                if n is None or n in visti or n in fr["stop"]:
                    continue
                visti.add(n)
                if n.name() == _ACCUMULA:
                    continue
                r_prec = attr.get(n)
                if r_prec is None:
                    attr[n] = r
                    ordine[r].append(n)
                elif id(fr) not in tr.chiamate[r_prec]["antenati"]:
                    continue        # nodo di una regione ESTERNA (creato prima di questa): confine
                # nodo nuovo, o di una regione ANNIDATA (gia' attribuita): si prosegue verso gli ingressi
                for m, _ in n.next_functions:
                    pila.append(m)
        tr.ordine_regione = ordine
        return attr

    def _percorso_nodo(self, tr, attr, n) -> str:
        if n is None:
            return "?"
        if n.name() == _ACCUMULA:
            return "parametro " + self._nomi_parametri.get(id(getattr(n, "variable", None)), "?")
        r = attr.get(n)
        return tr.chiamate[r]["percorso"] if r is not None else "lemma1 (errore_lemma1_finestra: h+e, cast, e_next, loss)"

    def _bordo_regione(self, tr, attr, r: int) -> dict:
        """Gradiente sulle USCITE della regione (arriva) e sui suoi INGRESSI (prodotto dalla regione)."""
        fr = tr.chiamate[r]
        uscite, ingressi = [], []
        pos_di = {i: pos for pos, (i, _u, _g) in enumerate(tr.registro)}
        for n, nr in fr.get("uscite", []):
            i = tr.indice.get(n)
            if i is None or i not in pos_di:
                continue
            d = self._descrivi_slot(tr, pos_di[i], "u")
            uscite.append({"nodo": n.name(), "slot": nr, **(d[nr] or {})} if nr < len(d) else {"nodo": n.name()})
        # nodi della regione E delle regioni annidate (una funzione dentro un modulo): il bordo
        # e' fatto dagli archi che escono da questo insieme verso gli ingressi della regione
        interni = [n for r2, fr2 in enumerate(tr.chiamate) if r2 == r or id(fr) in fr2.get("antenati", ())
                   for n in tr.ordine_regione.get(r2, [])]
        dentro = set(interni)
        for n in interni:
            i = tr.indice.get(n)
            if i is None or i not in pos_di:
                continue
            d = self._descrivi_slot(tr, pos_di[i], "i")
            for j, (m, _nr) in enumerate(n.next_functions):
                if m is None or j >= len(d) or d[j] is None or m in dentro:
                    continue
                ingressi.append({"da_nodo": n.name(), "verso": self._percorso_nodo(tr, attr, m) + " :: " + m.name(),
                                 **d[j]})
        return {"regione": fr["percorso"], "occorrenza": fr["occorrenza"], "tipo": fr["tipo"],
                "grad_uscite_finite": all(x.get("finito", True) for x in uscite),
                "grad_ingressi_finiti": all(x.get("finito", True) for x in ingressi),
                "grad_uscite": uscite, "grad_ingressi": ingressi}

    def _catena_regioni(self, tr, r) -> list:
        return list(tr.chiamate[r]["catena"])

    def _grandezze(self, tr: _Traccia, attr: dict, quanti: int = 10) -> dict:
        """Dai max|g| gia' misurati dalla sentinella (nessuna sincronizzazione in piu'): i nodi coi
        gradienti piu' GRANDI, quelli che AMPLIFICANO di piu' (max|g ingresso| / max|g uscita|: un
        gradiente finito che esplode nasce li'), e per ogni parametro il max|g| arrivato al suo
        AccumulateGrad con il giudizio sulla norma L2 della forgia."""
        righe, per_param = [], {}
        for pos, (i, u, g) in enumerate(tr.registro):
            n = tr.nodi[i]
            vu = [tr.valori.get((pos, "u", s)) for s in range(len(u))]
            vi = [tr.valori.get((pos, "i", s)) for s in range(len(g))]
            mu = max((x for x in vu if x is not None and math.isfinite(x)), default=0.0)
            mi = max((x for x in vi if x is not None and math.isfinite(x)), default=0.0)
            righe.append((pos, n, mu, mi))
            for j, (m, _nr) in enumerate(n.next_functions):
                if m is None or m.name() != _ACCUMULA or j >= len(vi) or vi[j] is None or g[j] is None:
                    continue
                nome = self._nomi_parametri.get(id(getattr(m, "variable", None)), "?")
                d = per_param.setdefault(nome, {"parametro": nome, "forma": list(g[j][0]), "dtype": g[j][1],
                                                "numel": math.prod(g[j][0]), "max_abs": 0.0, "finito": True,
                                                "contributi": 0})
                d["contributi"] += 1
                if math.isfinite(vi[j]):
                    d["max_abs"] = max(d["max_abs"], vi[j])
                else:
                    d["finito"] = False
        esiti = []
        for d in per_param.values():
            d["limite_somma_quadrati"] = d["numel"] * d["max_abs"] ** 2       # float Python: fino a 1e308
            d["l2_trabocca"] = ("certo" if d["max_abs"] > _SOGLIA_L2 else
                                "possibile" if d["limite_somma_quadrati"] > _FLT_MAX else "no")
            if d["contributi"] > 1:
                d["nota"] = "piu' contributi sommati nel .grad: il max|g| vero puo' arrivare alla loro somma"
            esiti.append(d["l2_trabocca"])
        piu_grandi = sorted(righe, key=lambda r: -r[3])[:quanti]
        ampl = sorted(((r[3] / r[2], r) for r in righe if r[2] > 0 and r[3] > 0), key=lambda x: -x[0])[:quanti]

        def voce(r, **extra):
            pos, n, mu, mi = r
            return {"posizione": pos, "nodo": n.name(), "regione": self._percorso_nodo(tr, attr, n),
                    "max_abs_uscita": mu, "max_abs_ingresso": mi, **extra}
        return {"max_abs_globale": max((v for v in tr.valori.values() if math.isfinite(v)), default=0.0),
                "soglia_L2_fp32": _SOGLIA_L2,
                "l2_traboccata_con_gradienti_finiti": ("certo" if "certo" in esiti else
                                                       "possibile (somma dei quadrati: decide il replay)"
                                                       if "possibile" in esiti else "no"),
                "parametri": sorted(per_param.values(), key=lambda d: -d["max_abs"]),
                "piu_grandi": [voce(r) for r in piu_grandi],
                "amplificazioni": [voce(r, fattore=f) for f, r in ampl]}

    def _analizza(self, fase: str, parcheggia_grad: bool = False) -> None:
        tr = self.traccia
        tr.analizzato = True
        try:
            origine = self._esito_sentinella(tr)
            saltato = tr.ris.get("saltato") if tr.ris is not None else None
            if origine is None and saltato is None:
                return
            if len(self.dump_scritti) >= self.massimo_dump:
                self._log(f"🔬 diagnosi NaN: altro evento al passo {tr.contesto} ({saltato or 'gradienti'}), "
                          "massimo dump raggiunto")
                return
            rapporto = self._rapporto(tr, origine, fase, saltato)
            bersaglio = None
            if origine is not None and rapporto["origine"].get("regione_indice") is not None:
                fr = tr.chiamate[rapporto["origine"]["regione_indice"]]
                bersaglio = fr
            catture = None
            if self.tentativi_replay > 0 and (origine is not None or saltato in ("loss", "gradienti")):
                with (self._grad_parcheggiati(tr.ctx["opt"]) if parcheggia_grad else contextlib.nullcontext()):
                    # salto per 'gradienti' senza nodo non finito: replay COMPLETO (norme per parametro)
                    rapporto["replay"], catture = self._replay(tr, bersaglio, rapporto,
                                                               solo_forward=(origine is None and saltato == "loss"))
            self._scrivi(rapporto, catture)
        except Exception as e:  # noqa: BLE001 — la diagnosi non deve mai uccidere la forgia (salvo severo)
            self._log(f"⛔ diagnosi NaN: errore interno durante l'analisi: {e!r}\n{traceback.format_exc()}")
            if self.severo:
                raise

    def _rapporto(self, tr: _Traccia, origine, fase: str, saltato) -> dict:
        V = self.V
        loss = None if tr.loss_orig is None else float(tr.loss_orig[0])
        loss_somma = None if tr.loss_orig is None else float(tr.loss_orig[1])
        rapporto = {"versione": VERSIONE, "quando": time.strftime("%F %T"), "corsa": getattr(V, "ID_CORSA", None),
                    "fase": fase, "saltato": saltato, **tr.contesto, "loss": loss, "loss_somma": loss_somma,
                    "nodi_agganciati": len(tr.nodi), "nodi_eseguiti": len(tr.registro),
                    "regioni": len(tr.chiamate), "gated_delta_rule_da": getattr(self, "impl_gated_delta", None),
                    "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "ricetta": {k: v for k, v in sorted(os.environ.items()) if k.startswith("THOR_")},
                    "origine": {}}
        if saltato == "loss" and tr.ris is not None:
            e_l = tr.ris.get("e_l") or []
            primo = next((l for l, e in enumerate(e_l) if not bool(torch.isfinite(e).all())), None)
            rapporto["loss_non_finita"] = {"primo_e_l_non_finito": primo,
                                           "e_l": [statistiche(e) for e in e_l]}
        attr = self._attribuisci(tr) if tr.registro else {}
        if tr.registro:
            rapporto["grandezze"] = self._grandezze(tr, attr)
        if origine is None:
            if saltato == "gradienti":
                gr = rapporto.get("grandezze") or {}
                l2 = gr.get("l2_traboccata_con_gradienti_finiti", "sconosciuto")
                rapporto["gradienti_finiti_ma_L2_traboccata"] = l2
                rapporto["anomalia"] = (
                    "passo saltato per 'gradienti non finiti' ma NESSUN nodo agganciato ha visto un gradiente non "
                    f"finito (max|g| visto {gr.get('max_abs_globale')}). La forgia decide con la norma L2 "
                    f"(vector_norm, fp32, senza riscalamento), che trabocca a Inf con gradienti FINITI oltre "
                    f"{_SOGLIA_L2:.3e} (o somma dei quadrati > {_FLT_MAX:.3e}): esito '{l2}'. Vedi "
                    "grandezze.parametri / piu_grandi / amplificazioni e replay[].norme_parametri. Se il replay "
                    "dice L2 finita, resta l'accumulo nei .grad (nodi AccumulateGrad non agganciati).")
            return rapporto
        n = tr.nodi[origine["indice"]]
        r = attr.get(n)
        pos = origine["posizione"]
        o = {"tipo": origine["tipo"], "nodo": n.name(), "posizione_nel_backward": pos,
             "su_nodi_eseguiti": len(tr.registro), "regione": self._percorso_nodo(tr, attr, n),
             "regione_indice": r, "grad_uscita": self._descrivi_slot(tr, pos, "u"),
             "grad_ingresso": self._descrivi_slot(tr, pos, "i"),
             "ingressi_verso": [None if m is None else self._percorso_nodo(tr, attr, m) + " :: " + m.name()
                                for m, _ in n.next_functions]}
        if r is not None:
            o["occorrenza_regione"] = tr.chiamate[r]["occorrenza"]
            o["indice_nella_regione"] = tr.ordine_regione[r].index(n)
            o["catena_regioni"] = self._catena_regioni(tr, r)
            o["bordo_regione"] = self._bordo_regione(tr, attr, r)
        if origine["tipo"] == "accumulo":
            contrib = []
            for pos2, (i2, _u, _g) in enumerate(tr.registro[:pos]):
                n2 = tr.nodi[i2]
                d = self._descrivi_slot(tr, pos2, "i")
                for j, (m, _nr) in enumerate(n2.next_functions):
                    if m is n and j < len(d):
                        contrib.append({"da_nodo": n2.name(), "regione": self._percorso_nodo(tr, attr, n2), **(d[j] or {})})
            o["contributi_in_arrivo"] = contrib
        rapporto["origine"] = o
        primi = []
        for pos2, (i2, fu, fi) in enumerate(tr.finitezza):
            if not (fu and fi):
                n2 = tr.nodi[i2]
                primi.append({"posizione": pos2, "nodo": n2.name(), "regione": self._percorso_nodo(tr, attr, n2),
                              "grad_uscita_finito": fu, "grad_ingresso_finito": fi})
                if len(primi) >= 12:
                    break
        rapporto["primi_non_finiti"] = primi
        par = set()
        for pos2, (i2, _fu, fi) in enumerate(tr.finitezza):
            if fi:
                continue
            n2 = tr.nodi[i2]
            d = self._descrivi_slot(tr, pos2, "i")
            for j, (m, _nr) in enumerate(n2.next_functions):
                if m is not None and m.name() == _ACCUMULA and j < len(d) and d[j] and not d[j]["finito"]:
                    par.add(self._nomi_parametri.get(id(getattr(m, "variable", None)), "?"))
        rapporto["parametri_con_gradiente_non_finito"] = sorted(par)
        return rapporto

    @contextlib.contextmanager
    def _grad_parcheggiati(self, opt):
        """Passo NON saltato: i .grad (3+ GiB su Thor) vanno su CPU durante il replay e tornano
        IDENTICI (copia esatta) prima di opt.step."""
        salvati = []
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    salvati.append((p, p.grad.device, p.grad.to("cpu", copy=True)))
                    p.grad = None
        try:
            yield
        finally:
            for p, dev, g in salvati:
                p.grad = g.to(dev)

    def _replay(self, tr: _Traccia, bersaglio, rapporto, solo_forward: bool):
        """Lo STESSO passo rieseguito (parametri intatti: prima di opt.step o dopo un salto),
        controllato nodo per nodo; cattura dell'ingresso della regione d'origine."""
        esiti, cattura = [], None
        ctx = tr.ctx
        params = [p for g in ctx["opt"].param_groups for p in g["params"]]
        chiave = None if bersaglio is None else (bersaglio["percorso"], bersaglio["occorrenza"])
        for k in range(self.tentativi_replay):
            tr2 = _Traccia("replay", bersaglio=chiave if cattura is None else None)
            principale = self.traccia
            self.traccia = tr2
            fwd = (self._cattura_forward_modulo(bersaglio["oggetto"])
                   if bersaglio is not None and bersaglio["tipo"] == "modulo" and cattura is None
                   else contextlib.nullcontext())
            esito = {"tentativo": k + 1}
            try:
                with fwd, torch.enable_grad():
                    fns = []
                    for f in ctx["strati_quant"]:
                        p = principale.argomenti_stratofn.get(id(f))
                        if p is None:
                            raise RuntimeError("closure del ramo quantizzato non registrata: replay impossibile")
                        fns.append(self._stratofn(**{**p, "raccoglitore": None}))
                    ris2 = self._lemma1_orig(ctx["strati_teacher"], fns, ctx["h0"], ctx["e0"],
                                             tol_algebrica=ctx["tol_algebrica"],
                                             h_l_teacher_cache=ctx["h_l_teacher_cache"],
                                             diagnostica=ctx["diagnostica"])
                    esito["loss_identica_al_bit"] = (tr.loss_orig is not None
                                                     and torch.equal(ris2["loss"].detach(), tr.loss_orig[0])
                                                     and torch.equal(ris2["loss_somma"].detach(), tr.loss_orig[1]))
                    esito["loss"] = float(ris2["loss"])
                    esito["primo_forward_non_finito"] = tr2.primo_forward_non_finito
                    da = ris2["loss_somma"] if ctx["riduzione"] == "somma" else ris2["loss"]
                    if not solo_forward and bool(torch.isfinite(da)):
                        self._aggancia_grafo(tr2, [da.grad_fn])
                        attr2 = self._attribuisci(tr2)
                        self._aggancia_salvati(tr2, attr2, rapporto)
                        g = torch.autograd.grad(da, params, allow_unused=True)
                        esito["gradienti_finiti"] = all(x is None or bool(torch.isfinite(x).all()) for x in g)
                        esito["norme_parametri"] = [{"parametro": self._nomi_parametri.get(id(p), "?"), **_norme(x)}
                                                    for p, x in zip(params, g) if x is not None]
                        presenti = [x for x in g if x is not None]
                        if presenti:           # lo STESSO conto di passo_ottimizzazione (salta_non_finiti)
                            nf = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in presenti]))
                            esito["norma_come_la_forgia"] = float(nf)
                            esito["salto_come_la_forgia"] = not math.isfinite(float(nf))
                        del g, presenti
                        esito["riprodotto"] = tr2.origine is not None
                        if tr2.origine is not None:
                            n = tr2.nodi[tr2.origine["indice"]]
                            o = {kk: v for kk, v in tr2.origine.items() if kk != "tensori"}
                            o.update(nodo=n.name(), regione=self._percorso_nodo(tr2, attr2, n))
                            o["stesso_punto_della_sentinella"] = (
                                o["regione"] == rapporto["origine"].get("regione")
                                and o["nodo"] == rapporto["origine"].get("nodo"))
                            esito["origine"] = o
                            esito["_tensori_origine"] = tr2.origine["tensori"]
                        esito["salvati_forward_origine"] = tr2.salvati_origine
                    elif not solo_forward:
                        esito["riprodotto"] = False
                        esito["nota"] = "loss non finita nel replay"
                    del ris2, da, fns
            except Exception as e:  # noqa: BLE001 — un replay fallito si scrive nel dump, non ferma la forgia
                esito["errore"] = f"{e!r}"
                esito["traccia"] = traceback.format_exc()[-2000:]
            finally:
                self.traccia = principale
                if cattura is None and tr2.cattura is not None:
                    cattura = tr2.cattura
                tr2.svuota()
            esiti.append(esito)
            if "errore" in esito:
                break
        return esiti, cattura

    def _aggancia_salvati(self, tr2: _Traccia, attr2: dict, rapporto: dict) -> None:
        """Pre-gancio sul nodo corrispondente all'origine della sentinella (stessa regione e
        occorrenza, stesso posto nell'ordine della regione, stesso nome): statistiche dei tensori
        SALVATI dal suo forward, prima che il backward li liberi."""
        o = rapporto.get("origine") or {}
        if o.get("regione_indice") is None:
            return
        chiave = (o["regione"], o.get("occorrenza_regione"))
        for r, fr in enumerate(tr2.chiamate):
            if (fr["percorso"], fr["occorrenza"]) != chiave:
                continue
            nodi = tr2.ordine_regione.get(r, [])
            k = o.get("indice_nella_regione")
            if k is None or k >= len(nodi) or nodi[k].name() != o["nodo"]:
                tr2.salvati_origine = {"nota": "nodo corrispondente non trovato nel replay (grafo diverso?)"}
                return
            n = nodi[k]

            def pre(grad_uscita, _n=n):
                tr2.salvati_origine = _tensori_salvati(_n)
                return None
            n.register_prehook(pre)
            return

    def _scrivi(self, rapporto: dict, cattura) -> None:
        self.cartella.mkdir(parents=True, exist_ok=True)
        t = rapporto.get("t")
        b = rapporto.get("b")
        base = self.cartella / (f"nan_{rapporto.get('corsa')}_t{t if t is not None else 'x'}"
                                f"_b{b if b is not None else 'x'}_{time.strftime('%Y%m%d-%H%M%S')}")
        tensori_origine = []
        for e in rapporto.get("replay") or []:
            tensori_origine.append(e.pop("_tensori_origine", None))
        blocco = {"rapporto": rapporto, "cattura": None if cattura is None else cattura.per_salvataggio(),
                  "tensori_origine_replay": tensori_origine}
        try:
            torch.save(blocco, str(base) + ".pt")
        except Exception as e:  # noqa: BLE001 — salvataggio del modulo non riuscito: si riprova senza
            rapporto["errore_salvataggio"] = f"{e!r}"
            if blocco["cattura"] is not None:
                blocco["cattura"]["oggetto"] = None
            torch.save(blocco, str(base) + ".pt")
        rapporto["file_pt"] = str(base) + ".pt"
        rapporto["cattura"] = (None if cattura is None else
                               {**cattura.info, "sostituiti": _json(cattura.sostituiti),
                                "grad_uscite_arrivati": [g is not None for g in cattura.grad_uscite]})
        Path(str(base) + ".json").write_text(json.dumps(_json(rapporto), indent=1, ensure_ascii=False))
        self.dump_scritti.append(str(base) + ".json")
        self.eventi.append(rapporto)
        if self.ferma_al_primo:
            self._da_fermare = True
        o = rapporto.get("origine") or {}
        gi = [s for s in (o.get("grad_ingresso") or []) if s]
        gu = [s for s in (o.get("grad_uscita") or []) if s]
        rep = rapporto.get("replay") or []
        self._log(
            f"⛔ DIAGNOSI NaN · passo t={t} batch {b} · strati {rapporto.get('layers')} · saltato {rapporto.get('saltato')}"
            + (f" · PRIMO nodo non finito nel backward: {o.get('nodo')} ({o.get('tipo')}, "
               f"{o.get('posizione_nel_backward')}/{o.get('su_nodi_eseguiti')}) in «{o.get('regione')}» · grad USCITA "
               f"{'finito' if all(s['finito'] for s in gu) else 'NON finito'} (max|g| "
               f"{max((s['max_abs'] for s in gu if s['finito']), default=0):.3e}) · grad INGRESSO "
               + ", ".join(f"{tuple(s['forma'])} {s['dtype']} max|g| {s['max_abs']}" for s in gi if not s["finito"])
               if o else "")
            + (f" · NESSUN nodo non finito: max|g| {(rapporto.get('grandezze') or {}).get('max_abs_globale')} · L2 "
               f"traboccata con gradienti finiti: {rapporto.get('gradienti_finiti_ma_L2_traboccata')} · amplifica di piu': "
               + ", ".join(f"{a['nodo']} in «{a['regione']}» x{a['fattore']:.2e}"
                           for a in ((rapporto.get('grandezze') or {}).get('amplificazioni') or [])[:2])
               if not o and rapporto.get("saltato") == "gradienti" else "")
            + f" · replay: {sum(1 for e in rep if e.get('riprodotto'))}/{len(rep)} riprodotti"
            + f" · dump {base}.json/.pt")


def _norme(x: torch.Tensor) -> dict:
    """max|x|, NaN/Inf, L2 ESATTA (fp64, a pezzi da 16M elementi: mai una copia fp64 intera in VRAM)
    e L2 COME LA CALCOLA LA FORGIA (vector_norm di default: trabocca oltre 1.84e19)."""
    piatto = x.detach().reshape(-1)
    somma, n_nan, n_inf = 0.0, 0, 0
    for pezzo in piatto.split(1 << 24):
        somma += float(torch.linalg.vector_norm(pezzo, dtype=torch.float64)) ** 2
        n_nan += int(torch.isnan(pezzo).sum())
        n_inf += int(torch.isinf(pezzo).sum())
    mx = float(torch.linalg.vector_norm(piatto, ord=_INF))
    l2f = float(torch.linalg.vector_norm(x.detach()))
    return {"forma": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""), "max_abs": mx,
            "max_abs_finito": math.isfinite(mx), "n_nan": n_nan, "n_inf": n_inf, "l2_fp64": math.sqrt(somma),
            "l2_come_la_forgia": l2f, "l2_come_la_forgia_finita": math.isfinite(l2f)}


def _tensori_salvati(n) -> dict:
    """Tensori che il forward del nodo ha salvato per il backward (nativi: _saved_*;
    Function Python/compilate: saved_tensors)."""
    out = {"nodo": n.name()}
    for nome in dir(n):
        if not nome.startswith("_saved_"):
            continue
        try:
            v = getattr(n, nome)
        except Exception as e:  # noqa: BLE001 — gia' liberato o non leggibile: si dice
            out[nome] = f"illeggibile: {e!r}"[:200]
            continue
        if isinstance(v, torch.Tensor):
            out[nome] = statistiche(v)
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, torch.Tensor) for x in v):
            out[nome] = [statistiche(x) for x in v]
        else:
            out[nome] = _json(v)
    try:
        st = getattr(n, "saved_tensors", None)
        if st:
            out["saved_tensors"] = [statistiche(x) if isinstance(x, torch.Tensor) else _json(x) for x in st]
    except Exception as e:  # noqa: BLE001
        out["saved_tensors"] = f"illeggibile: {e!r}"[:200]
    return out




# ═══════════════════════════ riproduzione isolata ═══════════════════════════
def _risolvi_funzione(info: dict):
    f = getattr(importlib.import_module(info["modulo"]), info["attributo"])
    return getattr(f, "_diagnosi_originale", f)


def riproduci(percorso_dump: "str | Path", ripetizioni: int = 10, device=None, prepara=None) -> dict:
    """Riesegue ISOLATA la regione d'origine dall'ingresso salvato nel replay: forward (uscite
    confrontate al bit con quelle salvate) e backward col gradiente arrivato sulle uscite; conta
    le ripetizioni con gradienti d'ingresso non finiti. `prepara(fn, args, kwargs, oggetto)` ->
    (fn, args, kwargs): gancio per rimettere uno strumento (o un veleno, nei collaudi)."""
    blocco = torch.load(str(percorso_dump), map_location="cpu", weights_only=False)
    c = blocco.get("cattura")
    if c is None:
        raise RuntimeError(f"{percorso_dump}: nessun ingresso catturato (replay non eseguito o origine fuori regione)")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if c.get("oggetto") is not None:
        impl = {getattr(getattr(m, "config", None), "_experts_implementation", None) for m in c["oggetto"].modules()}
        import esperti_thor
        if esperti_thor.NOME in impl:
            esperti_thor.registra()
    esito = {"regione": c["percorso"], "tipo": c["tipo"], "ripetizioni": ripetizioni, "non_finite": 0,
             "uscita_identica_al_bit": None, "dettaglio": []}
    for k in range(ripetizioni):
        def su_dev(t):
            if isinstance(t, dict) and t.get("__omesso__"):
                raise RuntimeError(f"tensore d'ingresso non salvato ({t['motivo']}): riproduzione impossibile")
            n = t.detach().to(dev)
            if t.requires_grad:
                n.requires_grad_(True)
            return n
        args = _mappa(c["ingressi"]["args"], su_dev)
        kwargs = _mappa(c["ingressi"]["kwargs"], su_dev)
        oggetto = None
        if c["tipo"] in ("modulo", "metodo"):
            if c["oggetto"] is None:
                raise RuntimeError("stato del modulo non salvato: riproduzione impossibile")
            oggetto = _modulo_su_device(copy.deepcopy(c["oggetto"]), dev)
        if c["tipo"] == "modulo":
            fn = oggetto.forward
        elif c["tipo"] == "metodo":
            fn = getattr(type(oggetto), c["metodo"])
            fn = functools.partial(getattr(fn, "_diagnosi_originale", fn), oggetto)
        elif c["tipo"] == "funzione":
            fn = _risolvi_funzione(c["funzione"])
        else:
            raise RuntimeError(f"regione di tipo {c['tipo']}: non riproducibile isolata")
        # le foglie si prendono PRIMA di `prepara`: un involucro sugli ingressi non deve nasconderle
        foglie = [(f"ingresso[{i}]", t) for i, t in enumerate(_appiattisci((args, kwargs))) if t.requires_grad]
        if prepara is not None:
            fn, args, kwargs = prepara(fn, args, kwargs, oggetto)
        out = fn(*args, **kwargs)
        uscite = _appiattisci(out)
        if k == 0:
            salvate = _appiattisci(c["uscite"])
            esito["uscita_identica_al_bit"] = (len(salvate) == len(uscite) and all(
                isinstance(a, torch.Tensor) and torch.equal(a.detach().cpu(), b.detach()) for a, b in zip(uscite, salvate)))
        coppie = [(t, g) for t, g in zip(uscite, c["grad_uscite"]) if g is not None and t.requires_grad]
        if not coppie:
            raise RuntimeError("nessun gradiente d'uscita salvato: riproduzione del backward impossibile")
        torch.autograd.backward([t for t, _ in coppie], [g.detach().to(dev) for _, g in coppie])
        if oggetto is not None:
            for nome, p in oggetto.named_parameters():
                if p.requires_grad:
                    foglie.append((nome, p))
            for nome_m, sub in oggetto.named_modules():
                for kk, v in sub.__dict__.items():
                    if isinstance(v, torch.Tensor) and v.requires_grad and v.is_leaf:
                        foglie.append(((nome_m + "." if nome_m else "") + kk, v))
        stat = {nome: statistiche(t.grad) for nome, t in foglie}
        non_finito = any(s is not None and not s["finito"] for s in stat.values())
        esito["non_finite"] += int(non_finito)
        esito["dettaglio"].append({"ripetizione": k + 1, "non_finito": non_finito, "grad": stat})
    return esito


# ═══════════════════════════ lanciatore ═══════════════════════════
def _antenati() -> set:
    """Questo processo e i suoi antenati (la shell che l'ha lanciato non e' «la coda»)."""
    pid, visti = os.getpid(), set()
    while pid > 1 and pid not in visti:
        visti.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return visti


_UNITA_KIB = {"": 1 / 1024, "KiB": 1, "MiB": 1024, "GiB": 1024 ** 2}


def _vram_drm(proc: Path = Path("/proc"), soglia_mib: "int | None" = None, escludi=()) -> list:
    """Processi con VRAM amdgpu attribuita dal kernel nel DRM fdinfo — Vulkan COMPRESO. La guardia
    gpu_occupata.sh legge /sys/class/kfd, che vede solo ROCm: misurato 25/9 02:25, llama-perplexity
    (Vulkan) della coda con 68 GiB in VRAM e gpu_occupata.sh rispondeva «libera» (rc 0). Esenti la
    casa e `escludi`; un client DRM condiviso da piu' fd si conta una volta."""
    soglia = int(os.getenv("GPU_SOGLIA_MIB", "256")) if soglia_mib is None else soglia_mib
    out = []
    for d in proc.iterdir():
        if not d.name.isdigit() or int(d.name) in escludi:
            continue
        try:
            comm = (d / "comm").read_text().strip()
            fds = list((d / "fdinfo").iterdir())
        except OSError:
            continue                                    # processo sparito o di un altro utente
        if comm in _CASA:
            continue
        clienti: dict = {}
        for fi in fds:
            try:
                testo = fi.read_text()
            except OSError:
                continue
            if "drm-driver" not in testo:
                continue
            campi = {k.strip(): v.strip() for k, _, v in (r.partition(":") for r in testo.splitlines())}
            v = campi.get("drm-memory-vram") or campi.get("drm-total-vram")
            if campi.get("drm-driver") != "amdgpu" or not v:
                continue
            num, _, unita = v.partition(" ")
            try:
                kib = int(num) * _UNITA_KIB.get(unita.strip(), 1)
            except ValueError:
                continue
            cid = campi.get("drm-client-id", fi.name)
            clienti[cid] = max(clienti.get(cid, 0), kib)
        mib = int(sum(clienti.values()) // 1024)
        if mib > soglia:
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:120]
            except OSError:
                cmd = "?"
            out.append(f"{d.name} {comm} {mib} MiB VRAM (DRM: Vulkan o altro, invisibile a gpu_occupata.sh) {cmd}")
    return out


def _coda_viva(proc: Path = Path("/proc"), radice: Path = RADICE, escludi=()) -> list:
    """Script della coda (bash <radice>/**.sh) vivi, ANCHE IN ATTESA: l'anello che aspetta un PID poi
    interroga gpu_occupata.sh UNA volta ed esce con 3 («NON partita») se trova la GPU presa — una
    diagnosi partita nella pausa fra due anelli fa abortire l'anello dopo (confronto_scorrevole_6f.sh,
    prova_c_torch.sh, 25/9). `bash -c "..."` non conta (e' un comando, non uno script della coda)."""
    radice = radice.resolve()
    out = []
    for d in proc.iterdir():
        if not d.name.isdigit() or int(d.name) in escludi:
            continue
        try:
            argv = [a.decode(errors="replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if len(argv) < 2 or Path(argv[0]).name not in ("bash", "sh", "dash", "zsh") or "-c" in argv[1:]:
            continue
        script = next((a for a in argv[1:] if not a.startswith("-")), "")
        if not script.endswith(".sh"):
            continue
        p = Path(script)
        try:
            if not p.is_absolute():
                p = Path(os.readlink(d / "cwd")) / p
            p = p.resolve()
        except OSError:
            continue
        if p.name != "gpu_occupata.sh" and radice in p.parents:
            out.append(f"{d.name} coda in corso: {p} {' '.join(argv[2:])[:60]}")
    return out


def _gpu_occupata() -> "str | None":
    """Tre guardie, e tutte devono dire libera: (1) strumenti/gpu_occupata.sh (VRAM ROCm/KFD, la
    guardia che usa la coda), (2) VRAM DRM per processo (i lavori Vulkan della coda: la (1) non li
    vede), (3) nessuno script della coda di thor_v2 vivo, nemmeno in attesa."""
    motivi = []
    r = subprocess.run([str(QUI / "gpu_occupata.sh")], capture_output=True, text=True)
    if r.returncode != 0:
        motivi.append("gpu_occupata.sh: " + ((r.stdout or "").strip() or f"rc {r.returncode}"))
    mio = _antenati()
    motivi += _vram_drm(escludi=mio)
    motivi += _coda_viva(escludi=mio)
    return "\n".join(motivi) if motivi else None


def _altri_sulla_gpu() -> "str | None":
    """Durante la diagnosi (a ogni giro): qualcun altro sulla GPU (DRM) o una coda comparsa? La (1)
    di _gpu_occupata qui non serve: vedrebbe noi stessi."""
    mio = _antenati()
    motivi = _vram_drm(escludi=mio) + _coda_viva(escludi=mio)
    return "\n".join(motivi) if motivi else None


def _gpu_visibile() -> bool:
    return torch.cuda.is_available()


def comando_finestra(a) -> int:
    if os.getenv(VAR_ATTIVA) != "1":
        print(f"⛔ {VAR_ATTIVA} non e' 1: lo strumento si attiva solo esplicitamente")
        return 2
    if a.k > 0 and not (a.ripresa and a.corsa_ripresa):
        print("⛔ finestra k>0: servono --ripresa e --corsa-ripresa (seme e strato portato veri della forgia); "
              "senza, la finestra partirebbe da condizioni che la forgia non ha mai visto")
        return 2
    occ = _gpu_occupata()
    if occ:
        print(f"⛔ GPU occupata da processi che non sono la casa, o coda di thor_v2 in corso:\n{occ}")
        return 3
    if not _gpu_visibile():
        print("⛔ nessuna GPU visibile a torch (CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES?): la finestra vera "
              "carica gli strati su GPU — niente caricamento a vuoto")
        return 3
    import valida_finestra_8_12 as V
    import finestra_globale as FG
    import forgia_scorrevole as FS
    import raccordo_scorrevole as RS
    FS._NON_RICETTA.update(VAR_DIAGNOSI)      # strumentazione, non ricetta (i numeri non cambiano)
    W = int(os.getenv("THOR_FORGIA_W", "2"))
    layers = list(range(a.k, a.k + W))
    n_passi = a.passi if a.passi else V.N_PASSI
    diag = Diagnosi(V, FG, ferma_al_primo=True, tetto_s=a.tetto_minuti * 60, massimo_dump=1).installa()
    try:
        return _ripeti_finestra(a, V, FS, RS, diag, layers, n_passi, W)
    finally:
        diag.disinstalla()


def _ripeti_finestra(a, V, FS, RS, diag: Diagnosi, layers: list, n_passi: int, W: int) -> int:
    ids, n_tr = V.carica_batch_calibrazione()
    cfg, emb, puliti, quant = V.carica_finestra_corrente(layers)
    istantanea = {L: (quant[L].mlp.experts.gate_up_proj.detach().to("cpu", copy=True),
                      quant[L].mlp.experts.down_proj.detach().to("cpu", copy=True)) for L in layers}
    rip = None
    if a.ripresa:
        rip = FS.Ripresa(Path(a.ripresa), a.corsa_ripresa, a.n_strati, W)
    else:
        h_seme, e_seme = FS._seme_iniziale_produzione(ids, emb)
    t0 = time.time()
    giro = 0
    while True:
        giro += 1
        altri = _altri_sulla_gpu()
        if altri:
            V.log(f"⏸ diagnosi NaN FERMATA prima del giro {giro} ({diag.passi_osservati} passi osservati): la GPU "
                  f"serve ad altri o una coda e' comparsa — si lascia il posto:\n{altri}")
            return 4
        with torch.no_grad():
            for L in layers:
                quant[L].mlp.experts.gate_up_proj.data.copy_(istantanea[L][0])
                quant[L].mlp.experts.down_proj.data.copy_(istantanea[L][1])
        iniziali = None
        if rip is not None:
            h_seme, e_seme, portati = rip.carica(a.k, emb.device)
            iniziali = portati or None
        V.log(f"🔁 diagnosi NaN: giro {giro} della finestra {layers} · {n_passi} passi · "
              f"{diag.passi_osservati} passi osservati finora · {(time.time()-t0)/60:.1f} min")
        try:
            RS.ottimizza_finestra_JOINT_B({L: quant[L] for L in layers}, ids, cfg, n_passi,
                                          strati_teacher={L: puliti[L] for L in layers}, emb=emb, lr=V.NUCLEO_LR,
                                          ternari_iniziali=iniziali, ternari_uscita={},
                                          h0_esterno_per_b=h_seme, e0_esterno_per_b=e_seme, n_batch_addestramento=n_tr)
        except NanTrovato as e:
            V.log(f"✅ NaN trovato e diagnosticato al giro {giro} dopo {diag.passi_osservati} passi: {e}")
            return 0
        except TettoDiTempo as e:
            V.log(f"⏱ {e}: {giro} giri, {diag.passi_osservati} passi, {diag.passi_saltati} saltati, nessun dump")
            return 1
        except FloatingPointError as e:
            if diag.dump_scritti:
                V.log(f"✅ la finestra si e' fermata ({e}) DOPO il dump: {diag.dump_scritti[-1]}")
                return 0
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if diag.dump_scritti:
            return 0


def comando_forgia(argv: list) -> int:
    if os.getenv(VAR_ATTIVA) != "1":
        print(f"⛔ {VAR_ATTIVA} non e' 1: lo strumento si attiva solo esplicitamente")
        return 2
    occ = _gpu_occupata()
    if occ:
        print(f"⛔ GPU occupata o coda in corso:\n{occ}")
        return 3
    if not _gpu_visibile():
        print("⛔ nessuna GPU visibile a torch: la forgia vera carica gli strati su GPU")
        return 3
    import valida_finestra_8_12 as V
    import finestra_globale as FG
    import forgia_scorrevole as FS
    FS._NON_RICETTA.update(VAR_DIAGNOSI)
    Diagnosi(V, FG).installa()
    sys.argv = ["forgia_scorrevole.py", *argv]
    return FS.main()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finestra", help="ripete una finestra finche' il NaN scatta (tetto di tempo)")
    f.add_argument("--k", type=int, default=0, help="primo strato della finestra (default 0)")
    f.add_argument("--passi", type=int, default=0, help="passi per giro (default THOR_VALIDA_N_PASSI)")
    f.add_argument("--tetto-minuti", type=float, default=360.0)
    f.add_argument("--ripresa", default="", help="cartella dei punti di ripresa della forgia (k>0)")
    f.add_argument("--corsa-ripresa", default="", help="THOR_ID_CORSA della forgia che ha scritto la ripresa")
    f.add_argument("--n-strati", type=int, default=int(os.getenv("THOR_FORGIA_N_STRATI", "40")))
    g = sub.add_parser("forgia", help="forgia_scorrevole.py strumentata (argomenti dopo --)")
    g.add_argument("resto", nargs=argparse.REMAINDER)
    r = sub.add_parser("riproduci", help="riesegue isolata la regione d'origine di un dump .pt")
    r.add_argument("dump")
    r.add_argument("--ripetizioni", type=int, default=20)
    r.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.cmd == "finestra":
        return comando_finestra(a)
    if a.cmd == "forgia":
        return comando_forgia([x for x in a.resto if x != "--"])
    if str(a.device or ("cuda" if _gpu_visibile() else "cpu")).startswith("cuda"):
        occ = _gpu_occupata()
        if occ:
            print(f"⛔ riproduci su GPU: GPU occupata o coda in corso (usare --device cpu, o aspettare):\n{occ}")
            return 3
    esito = riproduci(a.dump, a.ripetizioni, a.device)
    print(json.dumps(_json({k: v for k, v in esito.items() if k != "dettaglio"}), indent=1, ensure_ascii=False))
    print(json.dumps(_json(esito["dettaglio"][:3]), indent=1, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
