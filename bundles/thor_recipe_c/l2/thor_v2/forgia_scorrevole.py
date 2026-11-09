#!/usr/bin/env python3
"""FORGIA SCORREVOLE a PASSO 1 — Thor ternario, finestre [0,1],[1,2],...,[38,39].

24/9/2026, decisione di Angelo: la ricetta giusta non e' la partizione a
finestre separate ([0,1],[2,3],... di `forgia_globale_40_strati.py`) ma la
finestra SCORREVOLE a passo 1, in UNA sola catena sequenziale, come
SliderQuant (arXiv:2603.25284) e CBQ (arXiv:2312.07950).

## Perche' (la proprieta' che la partizione non ha)

Con la partizione lo strato 2j+1 (ultimo della sua finestra) riceve la sua
ottimizzazione finale SENZA avere in vista lo strato che gli viene dopo: il suo
errore viene giudicato solo sulla sua uscita, non su come lo strato 2j+2 lo
propaga. Con il passo 1 OGNI strato k (tranne il 39, l'ultimo della rete)
riceve l'ultima ottimizzazione dentro la finestra [k,k+1], cioe' con lo strato
successivo in vista, partendo dall'uscita VERA di tutti gli strati 0..k-1 gia'
commessi. Ogni strato viene inoltre ottimizzato DUE volte (entra come secondo
in [k-1,k], chiude come primo in [k,k+1]), e la seconda volta parte dai codici
appena ottimizzati, non dal teacher.

## Il ciclo (W generico, oggi W=2)

    finestra k = strati [k, k+W)              k = 0 .. n-W   (n-W+1 finestre: 39 per 40 strati)
    seme(k)    = (h_k, e_k): uscita vera (teacher) e errore del flusso commesso
                 dopo gli strati 0..k-1
    1. ottimizza la finestra (ottimizza_finestra_JOINT, ricetta INVARIATA) con:
         strati k..k+W-2  INIZIALIZZATI dai loro codici della finestra k-1
                          (oggetti TernarioDiffBatchFuso portati tali e quali:
                          z, log_s, delta_raw — lo stato latente completo)
         strato k+W-1     NUOVO: (a) teacher + LS come oggi, oppure
                          (b) codici di una corsa precedente (.npy + marca .ok)
    2. COMMIT dello strato k, DEFINITIVO: i pesi duri de-ruotati sono gia' nel
       modulo (commit di ottimizza_finestra_JOINT); i codici vanno su disco
       (scrivi_codici_su_disco, stesso formato e stessa marca .ok dell'export).
    3. seme(k+1) = seme(k) fatto passare per il SOLO strato k commesso:
       confine a META' finestra. Non serve una formula nuova:
       `calcola_confine_finestra` attraversa TUTTI gli strati del dizionario che
       riceve, quindi chiamata sul sotto-dizionario {k} e' esattamente il confine
       con offset interno 1 — stessa maschera, stesso posizionale, stesso codice
       (`calcola_confine_a_meta`, sotto). L'uscita del teacher dello strato k
       non si ricalcola: si cattura con un gancio sul modulo teacher mentre
       `precalcola_teacher` la calcola DENTRO l'ottimizzazione
       (`_CatturaUscitaTeacher`: conteggio esatto delle chiamate verificato,
       batch 0 ricalcolato e confrontato a OGNI finestra).
    4. l'ultima finestra [n-W, n) commette TUTTI i suoi strati.

## ⛔ INTERFACCIA ATTESA DAL COMPITO A (un altro agente la sta scrivendo)

Il passaggio dello strato k+1 alla finestra successiva richiede due cose che
`valida_finestra_8_12.py` oggi NON ha. Questo file le chiama SOLO attraverso i
nomi nel blocco `INTERFACCIA_A` qui sotto (se il compito A usa nomi diversi, si
cambia quel blocco e nient'altro), e rifiuta di partire se mancano
(`verifica_interfaccia_A`, anche da riga di comando: `--verifica-interfaccia`).

A1. `ottimizza_finestra_JOINT(..., ternari_iniziali=None, ternari_uscita=None)`
    - `ternari_iniziali`: `{L: {"gu": TernarioDiffBatchFuso, "dw": TernarioDiffBatchFuso | None}}`
      Per ogni strato L presente, la finestra usa QUEGLI oggetti TALI E QUALI
      (stesso oggetto Python, gia' sul device della finestra, parametri
      z/log_s/delta_raw NON toccati: niente reinit_ls_, niente ricostruzione dal
      teacher). "gu" e' nel dominio RUOTATO (come oggi), gruppo 128; "dw" senza
      rotazione, gruppo 256; "dw" None solo con THOR_FORGIA_DOWN_TERNARIO=0.
      Gli strati assenti dal dizionario si inizializzano come oggi
      (THOR_INIT_DA / THOR_INIT_LS).
    - `ternari_uscita`: dict vuoto passato dal chiamante; a fine chiamata contiene
      `{L: {"gu": ..., "dw": ...}}` per OGNI strato della finestra: gli oggetti
      ottimizzati (dopo l'ultimo passo). Per gli strati portati devono essere
      GLI STESSI oggetti ricevuti (qui si verifica con `is`).
    - Con entrambi None: comportamento identico a oggi (default invariati).
    - Lo stato di Adam NON si porta (l'ottimizzatore e' per finestra): i momenti
      dello strato portato ripartono da zero. Scelta del compito A, dichiarata.
      25/9: con THOR_SCORREVOLE_ADAM=conserva il raccordo lo porta come attributo
      `ATTR_ADAM` dell'oggetto (vedi raccordo_scorrevole.py); qui lo si salva e
      ricarica soltanto nel punto di ripresa (§Ripresa), senza interpretarlo.
A2. `ternari_da_codici(codici, cfg, device) -> {"gu": ..., "dw": ... | None}`
    (solo per la modalita' (b)). `codici` ha lo STESSO contratto di
    `risultato[L]` di ottimizza_finestra_JOINT e dei .npy su disco:
    `{"ffn_gate_exps": (D, Q), "ffn_up_exps": (D, Q)[, "ffn_down_exps": (D, Q)]}`.
    Garanzia richiesta (verificata qui a ogni uso, `_da_codici_verificato`):
    `_estrai_famiglia` dell'oggetto costruito ridà Q IDENTICO al bit e D entro
    1e-6 relativo (exp(log D) puo' differire di un ULP da D).

## Ripresa da interruzione (protocollo, nessuno stato implicito)

- Strato commesso = marca `fatto_forgia_LL.ok` in THOR_VALIDA_OUT con
  `corsa=<THOR_ID_CORSA> down_ternario=<0|1>` (la stessa che l'export esige).
- Punto di ripresa della finestra K = cartella `RIPRESA/finestra_KK/`:
  seme(K) per ogni batch (esatto al bit) + stato latente degli strati portati
  K..K+W-2 + manifest (sha256 di ogni file, ricetta = tutte le variabili THOR_*
  della corsa). Scritta in `.tmp_finestra_KK/` e rinominata (atomica).
  Se l'oggetto portato ha lo stato di Adam (`ATTR_ADAM`, THOR_SCORREVOLE_ADAM=
  conserva) va nello stesso `portato_LL.pt` (chiave "adam": +2x i parametri) e
  torna sull'oggetto alla ripresa; senza, il file e' quello di prima al byte. Il
  controllo (stato mancante, forme) e' del raccordo, alla finestra che lo usa.
- Ordine a fine finestra K: (1) punto di ripresa K+1 completo e rinominato,
  (2) codici + marca dello strato K, (3) cancellazione del punto K. Qualunque
  interruzione lascia o il punto K (si rifa' la finestra K) o il punto K+1 con
  la marca K (si riparte da K+1). Nell'ultima finestra una marca parziale fa
  rifare l'ultima finestra intera (marche dei suoi strati cancellate prima).
- La catena dei semi e' anche RICOSTRUIBILE dai soli codici su disco
  (`ricostruisci_seme`: strato per strato, stessa funzione di confine): serve
  a verificare un punto di ripresa (THOR_SCORREVOLE_VERIFICA_SEME_A="k1,k2")
  e, nel collaudo, a dimostrare che il seme della finestra k coincide con il
  forward diretto degli strati 0..k-1 commessi.
- Ripresa con ricetta diversa, marche di un'altra corsa, marche non contigue o
  punto di ripresa mancante: RIFIUTO rumoroso, mai un ripiego.

## Memoria (una finestra alla volta)

Restano caricati SOLO gli strati della finestra corrente: lo strato uscente k
(teacher + corrente) si libera dopo il confine, lo strato entrante k+W si carica
(`carica_finestra_corrente(layers=[k+W])`, l'embedding che ricarica si butta
subito), gli strati che restano NON si ricaricano. Degli oggetti ternari restano
solo quelli portati (gradienti azzerati: il gradiente di z da solo sono 3 GiB).

## Tempi (stima dalle misure del 24/9, `stato/v2_ThorT-veloce/forgia.log`)

Finestra [16,17] della forgia a finestre separate: 393,5 s = carico 2 strati
12 s + init LS 6 s + teacher/routing/held-out iniziale 38 s + 150 passi a
1,90 s + estrazione e scrittura 15 s + confine ~35 s. Passo 1,24 s con uno
strato ad attenzione piena, 1,89-1,90 s con due lineari. A passo 1: 19 finestre
lineare+piena e 20 lineare+lineare, passo medio 1,57 s. Sovraccarico stimato
per finestra ~75 s (1 strato caricato 6 s, embedding ~1 s, init 3 s,
teacher/routing 38 s, estrazione ~5 s, confine su UNO strato ~17 s, controllo
teacher 1 s, verifica commit ~2 s, stato portato ~3 s; le scritture in thread).
    T(N passi) ~ 39 x (75 + 1,57 N) s:  N=75 -> 2,1 h · N=100 -> 2,5 h · N=150 -> 3,4 h
Da MISURARE alla prima corsa (THOR_CRONO_OGNI e i tempi per finestra nel log).

## Ancora e held-out di conoscenza (26/9, entrambi SPENTI di default)

La forgia B perde 3,6 punti di MMLU sulla base mentre la PPL migliora: la perdita vede solo le
direzioni che la calibrazione attiva (ancora_forgia.py, §Perche').
- THOR_ANCORA_RAPPORTO=r (THOR_ANCORA_A=teacher|init): perdita = dati + λ·ancora dentro ogni
  finestra (valida_finestra_8_12.ottimizza_finestra_JOINT). Ricetta: entra nel confronto della ripresa.
- THOR_HELDOUT_CONOSCENZA=<.pt> (+ THOR_HELDOUT_CONOSCENZA_LOTTI, default 1): lotti di domande MMLU
  di validation in coda ai lotti della corsa, misurati come il held-out di sempre, mai addestrati.
  Fuori ricetta; il punto di ripresa rifiuta un numero di lotti diverso.
Collaudo CPU: test_ancora.py. Prova GPU sulle prime 8 finestre: prova_ancora.sh.

## Selective Projection Decay (26/9 sera, SPENTA di default)

L'ancora nella perdita non riavvicina i pesi sotto AdamW + STE (risultati/verifica_ancora.md). SPD (Tian et
al., NeurIPS 2024, arXiv 2411.01713, Alg. 2) agisce DOPO il passo sui parametri: dove c = -g·(p - θ0) < 0
riporta la deviazione da θ0 (i latenti all'inizio della finestra) al piu' dov'era, p <- p~ - λ·r·(p~ - θ0).
- THOR_SPD_LAMBDA=λ (0 = spenta: nessuna fotografia, identica al bit), THOR_SPD_SEMPRE=1 (controllo senza la
  condizione), THOR_SPD_GRANULARITA=esperto|tensore. Tutte e tre sono RICETTA (cambiano i pesi): entrano nel
  confronto della ripresa, una ripresa con λ diverso e' rifiutata. Dettaglio: spd_forgia.py. Collaudo CPU:
  test_spd.py. Prova GPU: prova_ancora.sh, campo spd=λ.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import valida_finestra_8_12 as V  # noqa: E402 — SOLA LETTURA: nessuna riga toccata
# 25/9 (PROGETTO_CACHE_LOTTI.md): stato per lotto oltre il budget GPU su disco, THOR_CACHE_LOTTI spento
# di default (ogni aggancio qui sotto e' allora identita' o no-op). Alias DL: D e' un nome locale (scale).
import deposito_lotti as DL  # noqa: E402
import ancora_forgia  # noqa: E402 — 26/9: solo i controlli della configurazione prima di partire
import spd_forgia  # noqa: E402 — 26/9 sera: idem per THOR_SPD_* (la proiezione vive in valida)
from ottimizzazione_globale_loop import pianifica_finestre  # noqa: E402 — pura, riusata per il confronto
from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso  # noqa: E402

log = V.log

# ═══════════════════════ INTERFACCIA_A (unico punto da adattare) ═══════════════════════
PARAM_INIZIALI = "ternari_iniziali"      # A1: parametro di ottimizza_finestra_JOINT
PARAM_USCITA = "ternari_uscita"          # A1: parametro di ottimizza_finestra_JOINT
MODULO_DA_CODICI = "raccordo_scorrevole"    # A2: raccordo A<->B (24/9 notte), ternari_da_codici
MODULO_OTTIMIZZA = "raccordo_scorrevole"    # A1: raccordo A<->B, sopra la patch A di valida_finestra_8_12
FUNZ_OTTIMIZZA = "ottimizza_finestra_JOINT_B"
FUNZ_DA_CODICI = "ternari_da_codici"        # A2: (codici, cfg, device) -> {"gu", "dw"}
TOLLERANZA_D_DA_CODICI = 1e-6               # A2: scarto relativo massimo su D (Q: identico)
ATTR_ADAM = "_adam_raccordo"                # A1 (25/9): stato di Adam dell'oggetto portato (= raccordo.CHIAVE_ADAM)

# ═══════════════════════ configurazione (solo da ambiente) ═══════════════════════
N_STRATI = int(os.getenv("THOR_FORGIA_N_STRATI", "40"))
W_FINESTRA = int(os.getenv("THOR_FORGIA_W", "2"))
N_PASSI = int(os.getenv("THOR_VALIDA_N_PASSI", "200"))      # STESSA variabile che l'export scrive nei metadati
MODO_INIT = os.getenv("THOR_SCORREVOLE_INIT", "teacher")    # teacher | codici
INIT_DIR = os.getenv("THOR_SCORREVOLE_INIT_DIR", "")
INIT_CORSA = os.getenv("THOR_SCORREVOLE_INIT_CORSA", "")
RIUSA_TEACHER = os.getenv("THOR_SCORREVOLE_RIUSA_TEACHER", "1") == "1"
SCRITTURA_ASINCRONA = os.getenv("THOR_SCORREVOLE_SCRITTURA_ASINCRONA", "1") == "1"
VERIFICA_COMMIT = os.getenv("THOR_SCORREVOLE_VERIFICA_COMMIT", "1") == "1"
VERIFICA_SEME_A = [int(x) for x in os.getenv("THOR_SCORREVOLE_VERIFICA_SEME_A", "").split(",") if x.strip()]
# tolleranza del confronto seme della catena vs seme ricostruito da disco (stesso algoritmo:
# differenze solo se i pesi ricostruiti dai codici differiscono dal commit in memoria)
TOLL_SEME_RICOSTRUITO = float(os.getenv("THOR_SCORREVOLE_TOLL_SEME", "1e-3"))
# 26/9: held-out di CONOSCENZA (domande MMLU di VALIDATION con la risposta, prepara_conoscenza.py): i primi
# THOR_HELDOUT_CONOSCENZA_LOTTI lotti del file si accodano ai lotti della corsa e viaggiano nella catena come
# il held-out di sempre (seme, confine, punto di ripresa, cache per lotto); valutati a init e a fine chunk,
# MAI un passo. Vuoto (default) = spento. Non e' ricetta: non cambia un peso (test_ancora.py, prova (c)).
HELDOUT_CONOSCENZA = os.getenv("THOR_HELDOUT_CONOSCENZA", "")
LOTTI_CONOSCENZA = int(os.getenv("THOR_HELDOUT_CONOSCENZA_LOTTI", "1"))

# variabili THOR_* che NON fanno parte della ricetta (strumentazione): escluse dal
# confronto alla ripresa. Tutte le altre, anche quelle nate dopo questo file, contano.
_NON_RICETTA = {
    "THOR_CRONO_OGNI", "THOR_HELDOUT_OGNI", "THOR_PROFILO", "THOR_DIAGNOSTICA_PASSO",
    "THOR_SCORREVOLE_SCRITTURA_ASINCRONA", "THOR_SCORREVOLE_VERIFICA_COMMIT",
    "THOR_SCORREVOLE_VERIFICA_SEME_A", "THOR_SCORREVOLE_TOLL_SEME", "THOR_SCORREVOLE_RIUSA_TEACHER",
    "THOR_EXPORT_SOVRASCRIVI", "THOR_VALIDA_PORTA_SONDA", "THOR_VALIDA_UNITA_SONDA",
    "THOR_LLAMA_BIN_K1024", "THOR_B9_DIR",
    # 25/9: la cache per lotto e' esatta al bit (PROGETTO_CACHE_LOTTI.md §6), quindi FUORI ricetta: una
    # corsa interrotta a cache accesa riprende spenta e viceversa. L'impronta del punto e' strumentazione.
    "THOR_CACHE_LOTTI", "THOR_CACHE_LOTTI_GPU_GIB", "THOR_CACHE_LOTTI_DIR", "THOR_CACHE_LOTTI_RISERVA_GIB",
    "THOR_CACHE_LOTTI_VERIFICA", "THOR_SCORREVOLE_IMPRONTA_RIPRESA",
    # 26/9: il held-out di conoscenza si misura e basta (codici identici con e senza, test_ancora.py (c)).
    # Una ripresa con un numero di lotti diverso la rifiuta il controllo dei lotti del punto di ripresa.
    # ⚠️ THOR_ANCORA_RAPPORTO / THOR_ANCORA_A cambiano i pesi: sono ricetta, qui NON vanno.
    # ⚠️ 26/9 sera: nemmeno THOR_SPD_LAMBDA / THOR_SPD_SEMPRE / THOR_SPD_GRANULARITA (spd_forgia.py).
    "THOR_HELDOUT_CONOSCENZA", "THOR_HELDOUT_CONOSCENZA_LOTTI",
}


class InterruzioneSimulata(RuntimeError):
    """Solo per il collaudo della ripresa: sollevata da `interrompi(punto, k)`."""


def _libera() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════ 1. pianificazione ═══════════════════════
def pianifica_scorrevole(n_strati: int, w: int) -> "list[tuple[int, int]]":
    """[(k, k+w)] per k = 0..n-w: finestre semiaperte a passo 1. Per w=2 coincide
    con `pianifica_finestre(n, 2, overlap=True)` (stride ceil(2/2)=1): verificato
    qui a ogni chiamata, non assunto. Ogni strato k < n-w+1 e' il PRIMO di
    esattamente una finestra (quella in cui si commette); gli ultimi w-1 strati
    si commettono con l'ultima finestra."""
    if w < 2:
        raise ValueError(f"W={w}: la finestra scorrevole serve almeno 2 strati (uno commesso, uno in vista)")
    if n_strati < w:
        raise ValueError(f"W={w} piu' grande della rete ({n_strati} strati)")
    finestre = [(k, k + w) for k in range(n_strati - w + 1)]
    assert finestre[0][0] == 0 and finestre[-1][1] == n_strati
    assert all(b[0] == a[0] + 1 for a, b in zip(finestre, finestre[1:])), "passo diverso da 1"
    if w == 2:
        rif = pianifica_finestre(n_strati, 2, True)
        if rif != finestre:
            raise RuntimeError(f"pianificazione divergente da pianifica_finestre(n,2,overlap): {rif} != {finestre}")
    return finestre


def strati_commessi_da(finestra: "tuple[int, int]", n_strati: int) -> "list[int]":
    """Il primo strato della finestra; nell'ultima finestra tutti i suoi strati."""
    k0, k1 = finestra
    return list(range(k0, k1)) if k1 == n_strati else [k0]


# ═══════════════════════ 2. interfaccia A (verifica e uso) ═══════════════════════
def verifica_interfaccia_A(ottimizza_fn, serve_da_codici: bool) -> None:
    """Rifiuta di partire se l'interfaccia del compito A non c'e' (mai un ripiego
    silenzioso sul teacher: sarebbe un'altra ricetta con lo stesso nome)."""
    parametri = inspect.signature(ottimizza_fn).parameters
    mancano = [p for p in (PARAM_INIZIALI, PARAM_USCITA) if p not in parametri]
    if mancano:
        raise RuntimeError(
            f"INTERFACCIA A ASSENTE: {getattr(ottimizza_fn, '__qualname__', ottimizza_fn)} non ha i parametri "
            f"{mancano}. Serve la patch del compito A su valida_finestra_8_12.py (vedi docstring di "
            "forgia_scorrevole.py, §INTERFACCIA ATTESA DAL COMPITO A), oppure aggiornare il blocco INTERFACCIA_A.")
    if serve_da_codici:
        _funzione_da_codici()


def _funzione_da_codici():
    import importlib
    modulo = importlib.import_module(MODULO_DA_CODICI)
    fn = getattr(modulo, FUNZ_DA_CODICI, None)
    if fn is None:
        raise RuntimeError(f"INTERFACCIA A ASSENTE: {MODULO_DA_CODICI}.{FUNZ_DA_CODICI} non esiste "
                           "(serve alla modalita' THOR_SCORREVOLE_INIT=codici)")
    return fn


def _funzione_ottimizza():
    """A1 attraverso il raccordo; rifiuta se la patch del compito A non c'e'."""
    import importlib
    modulo = importlib.import_module(MODULO_OTTIMIZZA)
    modulo.verifica_patch_A()
    return getattr(modulo, FUNZ_OTTIMIZZA)


def _codici_gate_up_fusi(codici: dict) -> "tuple[np.ndarray, np.ndarray]":
    """gate e up (file separati su disco) riuniti nella famiglia FUSA gate_up come la
    vede il modulo: righe 0:meta = gate, meta: = up (stessa convenzione di _esporta_gate_up)."""
    Dg, Qg = codici["ffn_gate_exps"]
    Du, Qu = codici["ffn_up_exps"]
    return np.concatenate([Dg, Du], axis=1), np.concatenate([Qg, Qu], axis=1)


def _confronta_famiglia(nome: str, D: np.ndarray, Q: np.ndarray, D_att: np.ndarray, Q_att: np.ndarray,
                        toll_d: float) -> float:
    if Q.shape != Q_att.shape or D.shape != D_att.shape:
        raise RuntimeError(f"{nome}: forme {D.shape}/{Q.shape} != attese {D_att.shape}/{Q_att.shape}")
    n_q = int((Q != Q_att).sum())
    if n_q:
        raise RuntimeError(f"{nome}: {n_q} codici diversi da quelli attesi (devono essere IDENTICI)")
    rel = float(np.max(np.abs(D.astype(np.float64) - D_att) / np.maximum(np.abs(D_att), 1e-30)))
    if rel > toll_d:
        raise RuntimeError(f"{nome}: scale diverse, scarto relativo massimo {rel:.3e} > {toll_d:.0e}")
    return rel


def _da_codici_verificato(codici: dict, cfg, device, fn, etichetta: str) -> dict:
    """A2 + controllo: i codici caricati devono ridare ESATTAMENTE i pesi duri salvati
    (Q identico, D entro TOLLERANZA_D_DA_CODICI)."""
    obj = fn(codici, cfg=cfg, device=device)
    if not isinstance(obj, dict) or "gu" not in obj or "dw" not in obj:
        raise RuntimeError(f"{FUNZ_DA_CODICI} deve ritornare {{'gu': ..., 'dw': ...}}, ricevuto {type(obj)}")
    D_att, Q_att = _codici_gate_up_fusi(codici)
    D, Q = V._estrai_famiglia(obj["gu"])
    r_gu = _confronta_famiglia(f"{etichetta} gate_up", D, Q, D_att, Q_att, TOLLERANZA_D_DA_CODICI)
    r_dw = float("nan")
    if V.DOWN_TERNARIO:
        if obj["dw"] is None or "ffn_down_exps" not in codici:
            raise RuntimeError(f"{etichetta}: down ternario richiesto ma assente (oggetto o codici)")
        Dd, Qd = V._estrai_famiglia(obj["dw"])
        r_dw = _confronta_famiglia(f"{etichetta} down", Dd, Qd, *codici["ffn_down_exps"], TOLLERANZA_D_DA_CODICI)
    elif obj["dw"] is not None:
        raise RuntimeError(f"{etichetta}: THOR_FORGIA_DOWN_TERNARIO=0 ma {FUNZ_DA_CODICI} ha costruito un down")
    log(f"  {etichetta}: init dai codici su disco · Q identico · scarto D gu {r_gu:.1e} dw {r_dw:.1e}")
    return obj


# ═══════════════════════ 3. codici su disco ═══════════════════════
def leggi_marca(cartella: Path, L: int) -> "dict | None":
    p = Path(cartella) / f"fatto_forgia_{L:02d}.ok"
    if not p.exists():
        return None
    t = p.read_text()
    corsa = re.search(r"corsa=(\S+)", t)
    down = re.search(r"down_ternario=(\d)", t)
    return {"corsa": corsa[1] if corsa else None, "down": down[1] if down else None, "testo": t}


def strati_commessi(cartella: Path, n_strati: int, corsa: str, down_atteso: bool) -> int:
    """Numero c di strati commessi 0..c-1 (prefisso contiguo). Rifiuta marche di
    un'altra corsa, marche illeggibili (scrittura interrotta: trattata come assente
    SOLO se e' l'ultima del prefisso) e marche oltre un buco."""
    c = 0
    while c < n_strati:
        m = leggi_marca(cartella, c)
        if m is None:
            break
        if m["corsa"] is None or m["down"] is None:
            log(f"⚠️ marca dello strato {c} illeggibile ({m['testo']!r}): scrittura interrotta, strato NON commesso")
            break
        if m["corsa"] != corsa:
            raise RuntimeError(f"strato {c}: marca della corsa {m['corsa']!r}, questa corsa e' {corsa!r} "
                               f"(cartella {cartella} mista)")
        if m["down"] != str(int(down_atteso)):
            raise RuntimeError(f"strato {c}: marca down_ternario={m['down']}, atteso {int(down_atteso)}")
        c += 1
    oltre = [L for L in range(c + 1, n_strati) if leggi_marca(cartella, L) is not None]
    if oltre:
        raise RuntimeError(f"marche NON contigue in {cartella}: commessi 0..{c-1}, ma esistono marche per {oltre}")
    return c


def verifica_cartella_init(cartella: Path, n_strati: int, corsa_attesa: str, down: bool) -> None:
    """Modalita' (b): TUTTI gli strati della cartella sorgente devono esserci (marca
    della corsa dichiarata, down coerente, i sei .npy) PRIMA di partire — non alla
    finestra 25 dopo ore di lavoro."""
    cartella = Path(cartella)
    guasti = []
    for L in range(n_strati):
        m = leggi_marca(cartella, L)
        if m is None:
            guasti.append(f"{L}: nessuna marca")
            continue
        if m["corsa"] != corsa_attesa:
            guasti.append(f"{L}: corsa {m['corsa']!r}")
        if down and m["down"] != "1":
            guasti.append(f"{L}: down_ternario={m['down']}")
        for pref in ("Dg", "Qg", "Du", "Qu") + (("Dd", "Qd") if down else ()):
            if not (cartella / f"{pref}_{L:02d}.npy").exists():
                guasti.append(f"{L}: manca {pref}_{L:02d}.npy")
    if guasti:
        raise RuntimeError(f"cartella dei codici iniziali {cartella} (corsa attesa {corsa_attesa!r}) incompleta: "
                           + "; ".join(guasti[:12]) + (f" ... (+{len(guasti)-12})" if len(guasti) > 12 else ""))


def leggi_codici_strato(cartella: Path, L: int, corsa_attesa: "str | None", down: bool) -> dict:
    """Codici di uno strato nel contratto di risultato[L], con la marca verificata."""
    cartella = Path(cartella)
    m = leggi_marca(cartella, L)
    if m is None:
        raise RuntimeError(f"strato {L}: nessuna marca fatto_forgia_{L:02d}.ok in {cartella}")
    if corsa_attesa and m["corsa"] != corsa_attesa:
        raise RuntimeError(f"strato {L} in {cartella}: corsa {m['corsa']!r}, attesa {corsa_attesa!r}")
    if down and m["down"] != "1":
        raise RuntimeError(f"strato {L} in {cartella}: down_ternario={m['down']}, serve 1")
    t_marca = (cartella / f"fatto_forgia_{L:02d}.ok").stat().st_mtime
    for pref in ("Dg", "Qg", "Du", "Qu") + (("Dd", "Qd") if down else ()):
        f = cartella / f"{pref}_{L:02d}.npy"
        if f.exists() and f.stat().st_mtime > t_marca + 1.0:
            raise RuntimeError(f"PROVENIENZA: {f} e' piu' recente della marca fatto_forgia_{L:02d}.ok: "
                               "riscritto da un'altra corsa")
    cod = {"ffn_gate_exps": (np.load(cartella / f"Dg_{L:02d}.npy"), np.load(cartella / f"Qg_{L:02d}.npy")),
           "ffn_up_exps": (np.load(cartella / f"Du_{L:02d}.npy"), np.load(cartella / f"Qu_{L:02d}.npy"))}
    if down:
        cod["ffn_down_exps"] = (np.load(cartella / f"Dd_{L:02d}.npy"), np.load(cartella / f"Qd_{L:02d}.npy"))
    for fam, (D, Q) in cod.items():
        if Q.dtype != np.int8 or D.dtype != np.float32 or D.shape[:2] != Q.shape[:2]:
            raise RuntimeError(f"strato {L}/{fam}: dtype/forme inattese D={D.dtype}{D.shape} Q={Q.dtype}{Q.shape}")
        if Q.shape[-1] % D.shape[-1]:
            raise RuntimeError(f"strato {L}/{fam}: n_in={Q.shape[-1]} non multiplo di ng={D.shape[-1]}")
        if not set(np.unique(Q).tolist()) <= {-1, 0, 1}:
            raise RuntimeError(f"strato {L}/{fam}: codici fuori da {{-1,0,1}}")
    return cod


# 24/9 23:5x: codici -> pesi ha UNA sola formula, quella del commit (V.pesi_duri_da_codici,
# scale fp16 come nel GGUF). Qui si ri-esportano i nomi di prima: nessuna seconda copia.
_pesi_da_DQ = V.pesi_da_DQ
pesi_duri_da_codici = V.pesi_duri_da_codici


def inietta_codici_nel_modulo(strato_quant: torch.nn.Module, codici: dict, S: torch.Tensor) -> None:
    """Scrive nel modulo (ramo corrente, pesi GGUF di base) i pesi duri di codici su
    disco: stesso stato del modulo dopo il commit in memoria."""
    esp = strato_quant.mlp.experts
    gu, dw = pesi_duri_da_codici(codici, S, esp.gate_up_proj.device, esp.gate_up_proj.dtype)
    with torch.no_grad():
        esp.gate_up_proj.data.copy_(gu)
        if dw is not None:
            esp.down_proj.data.copy_(dw)


def verifica_commit(strato_quant: torch.nn.Module, codici: dict, S: torch.Tensor, L: int) -> dict:
    """I codici che vanno su disco devono ridare i pesi duri commessi nel modulo.
    Con V.COMMIT_DA_CODICI (default dal 24/9) il commit E' la ricostruzione dai codici:
    si pretende l'uguaglianza al bit, qualunque differenza e' un difetto. Senza (corse
    vecchie, commit da forward_duro compilato) resta la tolleranza di 1 ULP bf16."""
    esp = strato_quant.mlp.experts
    # de-rotazione sul tensore INTERO (stessa forma del commit: nessuna forma nuova per la
    # FWHT compilata della forgia); confronto a blocchi di esperti (niente copie fp32 da 2 GiB)
    gu, dw = pesi_duri_da_codici(codici, S, esp.gate_up_proj.device, esp.gate_up_proj.dtype)
    esito = {}
    for nome, atteso, reale in (("gate_up", gu, esp.gate_up_proj.data),
                                ("down", dw, esp.down_proj.data if dw is not None else None)):
        if atteso is None:
            continue
        if atteso.shape != reale.shape:
            raise RuntimeError(f"strato {L} {nome}: forma dai codici {tuple(atteso.shape)} != modulo {tuple(reale.shape)}")
        diversi = oltre = 0
        massimo = 0.0
        for e0 in range(0, reale.shape[0], 32):
            a, r = atteso[e0:e0 + 32], reale[e0:e0 + 32]
            diversi += int((a != r).sum().item())
            scarto = (a.float() - r.float()).abs()
            oltre += int((scarto > r.float().abs() * (2.0 ** -7) + 1e-30).sum().item())
            massimo = max(massimo, float(scarto.max().item()))
            del a, r, scarto
        esito[nome] = {"diversi": diversi, "totale": reale.numel(), "oltre_1ulp": oltre, "max": massimo}
        if V.COMMIT_DA_CODICI and diversi:
            raise RuntimeError(f"strato {L} {nome}: {diversi} pesi commessi diversi dai codici (max {massimo:.3e}) "
                               f"con commit DAI codici — i codici su disco NON ridanno il modulo")
        if oltre:
            raise RuntimeError(f"strato {L} {nome}: {oltre} pesi commessi differiscono dai codici oltre 1 ULP "
                               f"(max {esito[nome]['max']:.3e}) — i codici su disco NON ridanno il modulo")
    del gu, dw
    log(f"  strato {L}: commit verificato contro i codici · "
        + " · ".join(f"{k} {v['diversi']}/{v['totale']} elementi non identici (max {v['max']:.1e})"
                     for k, v in esito.items()))
    return esito


# ═══════════════════════ 4. confine a meta' finestra e cattura del teacher ═══════════════════════
def calcola_confine_a_meta(puliti: dict, quantizzati: dict, strati: "list[int]", ids_batch: torch.Tensor, cfg,
                           emb: torch.Tensor, h0_per_b, e0_per_b, h_teacher_uscita_per_b=None,
                           contenitori: "tuple | None" = None) -> tuple:
    """Confine con OFFSET INTERNO: seme della finestra fatto passare per i soli `strati`
    (un prefisso della finestra). E' `calcola_confine_finestra` sul sotto-dizionario:
    quella funzione attraversa tutti gli strati che riceve, con la SUA maschera e il
    SUO posizionale — nessuna seconda formula per lo stesso calcolo.
    `contenitori` (25/9, cache per lotto): depositi (h, e) del seme nuovo, inoltrati."""
    finestra = sorted(puliti)
    if strati != finestra[:len(strati)] or not strati:
        raise ValueError(f"strati {strati} non sono un prefisso della finestra {finestra}")
    return V.calcola_confine_finestra({L: puliti[L] for L in strati}, {L: quantizzati[L] for L in strati},
                                      ids_batch, cfg, emb, h0_ingresso_per_b=h0_per_b,
                                      e0_ingresso_per_b=e0_per_b, h_fine_teacher_per_b=h_teacher_uscita_per_b,
                                      contenitori=contenitori)


class _CatturaUscitaTeacher:
    """Gancio sul modulo teacher del PRIMO strato della finestra: `precalcola_teacher`
    (dentro ottimizza_finestra_JOINT) calcola F_k(h_k) una volta per batch, in ordine
    b = 0..n-1; qui si tiene il riferimento a quel tensore (nessuna copia) per il
    confine a meta' finestra. Il conteggio delle chiamate deve essere ESATTAMENTE il
    numero di batch: se l'ottimizzazione chiamasse il teacher altrove (altra patch,
    sotto-finestre) il riuso sarebbe ambiguo e si rifiuta.

    25/9 (cache per lotto): con `contenitore` (un deposito, THOR_CACHE_LOTTI=1) il gancio
    scrive l'uscita al lotto `len(depositati)` (stesso ordine b = 0..n-1): cosi' non si
    tengono sulla GPU le uscite di tutti i lotti fino al confine. E' lo stesso tensore di
    `teacher[b][1]` (provenienza: un file solo). `per_batch` restituisce il deposito, che
    la forgia chiude DOPO il confine (una volta sola: il seme della finestra dopo e' un
    deposito NUOVO che nomina gli stessi file, critica 1). None = la lista di prima."""

    def __init__(self, strato: torch.nn.Module, contenitore=None):
        self.strato = strato
        self.uscite = [] if contenitore is None else contenitore
        self._maniglia = None

    def __enter__(self):
        def _gancio(_m, _a, o):
            if isinstance(self.uscite, list):
                self.uscite.append(o[0] if isinstance(o, tuple) else o)
            else:
                self.uscite[len(self.uscite)] = o[0] if isinstance(o, tuple) else o
        self._maniglia = self.strato.register_forward_hook(_gancio)
        return self

    def __exit__(self, *_):
        self._maniglia.remove()
        return False

    def per_batch(self, n_batch: int) -> dict:
        if len(self.uscite) != n_batch:
            raise RuntimeError(f"cattura teacher: {len(self.uscite)} chiamate al teacher del primo strato, "
                               f"attese {n_batch} (una per batch in precalcola_teacher) — riuso rifiutato")
        if not isinstance(self.uscite, list):
            return self.uscite
        return {b: self.uscite[b] for b in range(n_batch)}


def _controlla_teacher_catturato(pulito, quant, ids, cfg, emb, h_seme, catturato: dict, k: int) -> float:
    """Batch 0 ricalcolato da zero e confrontato con l'uscita catturata (a ogni finestra)."""
    h_rif, _ = V.calcola_confine_finestra({k: pulito}, {k: quant}, ids[:1], cfg, emb,
                                          h0_ingresso_per_b=None if h_seme is None else {0: h_seme[0]},
                                          e0_ingresso_per_b=None)
    d = h_rif[0].float() - catturato[0].float()
    rel = (d.norm() / h_rif[0].float().norm().clamp_min(1e-30)).item()
    if rel > 1e-2:
        raise RuntimeError(f"finestra {k}: teacher catturato diverso dal ricalcolato (rel {rel:.3e})")
    return rel


# ═══════════════════════ 5. punto di ripresa ═══════════════════════
def _ricetta() -> dict:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("THOR_") and k not in _NON_RICETTA}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blocco in iter(lambda: f.read(1 << 24), b""):
            h.update(blocco)
    return h.hexdigest()


def _sha256_e_libera(p: Path) -> str:
    """25/9 (cache per lotto, P4): sha256 e poi il file fuori dalla page cache (sull'UMA e' RAM
    della casa: 12,5 GiB a finestra a 151 lotti). Stesso sha: si legge lo stesso file."""
    h = _sha256(p)
    DL.libera_pagine(p)
    return h


def _stato_ternario(obj) -> dict:
    salvato = {"meta": {"classe": type(obj).__name__, "shape": list(obj.shape), "group": obj.group,
                        "beta_0": obj.beta_0, "clip": obj.clip, "delta_max": obj.delta_max},
               "stato": obj.state_dict()}
    adam = getattr(obj, ATTR_ADAM, None)
    if adam is not None:
        # 25/9 (THOR_SCORREVOLE_ADAM=conserva): momenti e step dello strato portato. prepara() e'
        # SINCRONA: la finestra dopo li adotta e li aggiorna sul posto, quindi si salvano prima.
        salvato["adam"] = adam
    return salvato


def _ricrea_ternario(salvato: dict, device) -> TernarioDiffBatchFuso:
    m = salvato["meta"]
    if m["classe"] != TernarioDiffBatchFuso.__name__:
        raise RuntimeError(f"stato portato di classe {m['classe']}, attesa {TernarioDiffBatchFuso.__name__}")
    obj = TernarioDiffBatchFuso(torch.zeros(m["shape"], device=device), group=m["group"])
    for a in ("beta_0", "clip", "delta_max"):
        if getattr(obj, a) != m[a]:
            raise RuntimeError(f"stato portato: {a}={m[a]} ma il costruttore di oggi da' {getattr(obj, a)}")
    obj.load_state_dict(salvato["stato"], strict=True)
    if "adam" in salvato:
        setattr(obj, ATTR_ADAM, salvato["adam"])     # controllato dal raccordo alla finestra che lo usa
    return obj


class Ripresa:
    """Punti di ripresa `finestra_KK/` (vedi docstring del modulo, §Ripresa)."""

    def __init__(self, radice: Path, corsa: str, n_strati: int, w: int):
        self.radice = Path(radice)
        self.corsa, self.n, self.w = corsa, n_strati, w
        self.ricetta = _ricetta()
        self.radice.mkdir(parents=True, exist_ok=True)

    def cartella(self, K: int) -> Path:
        return self.radice / f"finestra_{K:02d}"

    def _tmp(self, K: int) -> Path:
        return self.radice / f".tmp_finestra_{K:02d}"

    def presenti(self) -> "list[int]":
        return sorted(int(p.name.split("_")[1]) for p in self.radice.glob("finestra_[0-9]*")
                      if (p / "manifest.json").exists())

    def prepara(self, K: int, portati: dict) -> Path:
        """SINCRONO (prima che la finestra successiva tocchi gli oggetti portati):
        stato latente degli strati portati in .tmp_finestra_KK/."""
        tmp = self._tmp(K)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for L, fam in portati.items():
            torch.save({"gu": _stato_ternario(fam["gu"]),
                        "dw": None if fam["dw"] is None else _stato_ternario(fam["dw"])},
                       tmp / f"portato_{L:02d}.pt")
        return tmp

    def completa(self, K: int, tmp: Path, seme_h: dict, seme_e: dict, portati: "list[int]") -> None:
        """(nel thread di scrittura) semi + manifest con sha256, poi rename atomico.

        25/9 (cache per lotto): il seme puo' essere un deposito; `DL.leggi_per_scrittura` lo
        rilegge sul device ORIGINALE (stesso torch.save, stessi byte, stessi sha). A cache
        accesa (P4, critica 15) ogni file si libera dalla page cache DOPO il suo sha256
        (calcolato subito, dalla page cache: niente rilettura dal disco) e con
        THOR_SCORREVOLE_IMPRONTA_RIPRESA=1 si scrive nel log l'impronta del punto (G1)."""
        n_batch = len(seme_h)
        cache = DL.attiva()
        sha_pronti: dict = {}
        for b in range(n_batch):
            p = tmp / f"seme_b{b:03d}.pt"
            torch.save({"h": DL.leggi_per_scrittura(seme_h, b), "e": DL.leggi_per_scrittura(seme_e, b)}, p)
            if cache:
                sha_pronti[p.name] = _sha256_e_libera(p)
        file = {p.name: {"sha256": sha_pronti[p.name] if p.name in sha_pronti
                         else (_sha256_e_libera(p) if cache else _sha256(p)), "byte": p.stat().st_size}
                for p in sorted(tmp.iterdir())}
        if os.getenv("THOR_SCORREVOLE_IMPRONTA_RIPRESA", "0") == "1":
            elenco = json.dumps({n: v["sha256"] for n, v in file.items()}, sort_keys=True)
            log(f"  impronta del punto di ripresa finestra {K}: {hashlib.sha256(elenco.encode()).hexdigest()}")
        manifest = {"finestra": K, "corsa": self.corsa, "n_strati": self.n, "W": self.w, "n_batch": n_batch,
                    "portati": portati, "ricetta": self.ricetta, "file": file, "quando": time.strftime("%F %T")}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        finale = self.cartella(K)
        if finale.exists():
            shutil.rmtree(finale)
        os.replace(tmp, finale)
        log(f"  punto di ripresa finestra {K} scritto ({sum(v['byte'] for v in file.values())/2**30:.2f} GiB)")

    def elimina(self, K: int) -> None:
        p = self.cartella(K)
        if p.exists():
            shutil.rmtree(p)

    def punto_di_ripresa(self, c: int, cartella_out: Path) -> "int | None":
        """Dato c = strati commessi, la finestra da cui ripartire (None = da capo).
        Pulisce .tmp_* e punti superati; rifiuta gli stati incoerenti."""
        for t in self.radice.glob(".tmp_finestra_*"):
            log(f"  ripresa: tolgo il punto incompleto {t.name}")
            shutil.rmtree(t)
        presenti = self.presenti()
        ultima = self.n - self.w
        if c == 0 and not presenti:
            return None
        if c == self.n:
            for K in presenti:
                self.elimina(K)
            return self.n                       # tutto commesso
        if c in presenti and c <= ultima:
            K = c
        elif ultima in presenti and ultima < c < self.n:
            K = ultima                           # ultima finestra interrotta a meta' delle marche
            log(f"⚠️ ripresa: ultima finestra interrotta con {c - ultima} marche su {self.w}: "
                f"cancello le marche {list(range(ultima, c))} e rifaccio la finestra {ultima} intera")
            for L in range(ultima, self.n):
                p = Path(cartella_out) / f"fatto_forgia_{L:02d}.ok"
                if p.exists():
                    p.unlink()
        elif c == 0:
            log(f"  ripresa: nessuno strato commesso, punti {presenti} superati: riparto da capo")
            for K in presenti:
                self.elimina(K)
            return None
        else:
            raise RuntimeError(f"ripresa impossibile: strati commessi 0..{c-1} ma nessun punto di ripresa "
                               f"finestra_{c:02d} in {self.radice} (presenti: {presenti}). Il seme e' "
                               "ricostruibile (ricostruisci_seme) ma lo stato latente degli strati portati no: "
                               "ripartire da capo in una cartella nuova.")
        for K2 in presenti:
            if K2 != K:
                log(f"  ripresa: tolgo il punto superato finestra_{K2:02d}")
                self.elimina(K2)
        return K

    def carica(self, K: int, device, contenitori: "tuple | None" = None,
               n_batch_atteso: "int | None" = None) -> "tuple[dict, dict, dict]":
        """`contenitori` (25/9, cache per lotto): depositi (h, e) riempiti lotto per lotto (sulla
        GPU un lotto alla volta, non 9,4 GiB) e file letti liberati dalla page cache. None = dict.
        `n_batch_atteso` (26/9): i lotti della corsa che riprende; il punto deve averne altrettanti.
        THOR_HELDOUT_CONOSCENZA e' fuori ricetta ma aggiunge lotti: una ripresa con i lotti di
        conoscenza accesi su un punto scritto senza (o viceversa) si RIFIUTA qui, invece di
        mescolare semi di lotti diversi."""
        d = self.cartella(K)
        cache = contenitori is not None
        man = json.loads((d / "manifest.json").read_text())
        if man["corsa"] != self.corsa or man["n_strati"] != self.n or man["W"] != self.w or man["finestra"] != K:
            raise RuntimeError(f"punto di ripresa {d}: corsa/n/W/finestra {man['corsa']}/{man['n_strati']}/"
                               f"{man['W']}/{man['finestra']} != {self.corsa}/{self.n}/{self.w}/{K}")
        if n_batch_atteso is not None and man["n_batch"] != n_batch_atteso:
            raise RuntimeError(f"ripresa RIFIUTATA: il punto di ripresa {d} ha {man['n_batch']} lotti, questa corsa "
                               f"{n_batch_atteso} (THOR_HELDOUT_CONOSCENZA/_LOTTI diversi dalla corsa interrotta?)")
        diverse = sorted(k for k in set(man["ricetta"]) | set(self.ricetta)
                         if man["ricetta"].get(k) != self.ricetta.get(k))
        if diverse:
            raise RuntimeError("ripresa RIFIUTATA: ricetta diversa da quella della corsa interrotta: "
                               + "; ".join(f"{k}: {man['ricetta'].get(k)!r} -> {self.ricetta.get(k)!r}"
                                           for k in diverse))
        for nome, info in man["file"].items():
            if (_sha256_e_libera(d / nome) if cache else _sha256(d / nome)) != info["sha256"]:
                raise RuntimeError(f"punto di ripresa {d}: {nome} corrotto (sha256 diverso)")
        h, e = contenitori if contenitori is not None else ({}, {})
        for b in range(man["n_batch"]):
            s = torch.load(d / f"seme_b{b:03d}.pt", map_location=device, weights_only=True)
            h[b], e[b] = s["h"], s["e"]
            if cache:
                del s
                DL.libera_pagine(d / f"seme_b{b:03d}.pt")
        portati = {}
        for L in man["portati"]:
            s = torch.load(d / f"portato_{L:02d}.pt", map_location=device, weights_only=True)
            if cache:
                DL.libera_pagine(d / f"portato_{L:02d}.pt")
            portati[L] = {"gu": _ricrea_ternario(s["gu"], device),
                          "dw": None if s["dw"] is None else _ricrea_ternario(s["dw"], device)}
        log(f"  ripresa dalla finestra {K}: seme di {man['n_batch']} batch e strati portati {man['portati']} "
            f"(sha256 verificati, ricetta identica)")
        return h, e, portati


# ═══════════════════════ 6. ricostruzione della catena dei semi ═══════════════════════
def ricostruisci_seme(k: int, ids_batch: torch.Tensor, cfg, emb: torch.Tensor, seme0: tuple, carica_strati,
                      cartella: Path, corsa: str, batch: "list[int] | None" = None) -> tuple:
    """seme(k) dai SOLI codici su disco: strato per strato 0..k-1 (teacher bf16 + ramo
    corrente con i codici iniettati), stessa funzione di confine della catena. Uno
    strato alla volta in memoria. `batch`: sottoinsieme di batch (per il controllo)."""
    sel = list(range(ids_batch.shape[0])) if batch is None else list(batch)
    ids = ids_batch[sel]
    h0, e0 = seme0
    h = None if h0 is None else {i: h0[b] for i, b in enumerate(sel)}
    e = None if e0 is None else {i: e0[b] for i, b in enumerate(sel)}
    S = torch.from_numpy(V._S1024_NP).to(emb.device)
    for L in range(k):
        _cfg, _emb, pul, qua = carica_strati([L])
        del _cfg, _emb
        inietta_codici_nel_modulo(qua[L], leggi_codici_strato(cartella, L, corsa, V.DOWN_TERNARIO), S)
        h, e = V.calcola_confine_finestra(pul, qua, ids, cfg, emb, h0_ingresso_per_b=h, e0_ingresso_per_b=e)
        del pul, qua
        _libera()
    return h, e


def _confronta_semi(h_a: dict, e_a: dict, h_b: dict, e_b: dict, mappa: dict) -> "tuple[float, float]":
    """max su batch di (rel teacher, rel quantizzato h+e). mappa: indice_b -> indice_a."""
    rt = rq = 0.0
    for ib, ia in mappa.items():
        ha, hb = h_a[ia].float(), h_b[ib].float()
        qa, qb = ha + e_a[ia].float(), hb + e_b[ib].float()
        rt = max(rt, ((ha - hb).norm() / ha.norm().clamp_min(1e-30)).item())
        rq = max(rq, ((qa - qb).norm() / qa.norm().clamp_min(1e-30)).item())
    return rt, rq


# ═══════════════════════ 7. il ciclo ═══════════════════════
def carica_heldout_conoscenza(percorso: str, n_lotti: int, batch_seq: int, l_seq: int) -> torch.Tensor:
    """26/9: i primi `n_lotti` lotti (n_lotti x batch_seq sequenze, nell'ordine del file) del held-out
    di CONOSCENZA, forma (n_lotti, batch_seq, l_seq) come i lotti di calibrazione. Il file e' quello di
    prepara_conoscenza.py (tensore int64 (n_seq, L_SEQ), domande MMLU di validation gia' mescolate col
    loro seme). Se accanto c'e' il manifesto (stesso nome, .json) si pretendono lo sha256 dei token
    dichiarato, L_SEQ e il tokenizzatore della forgia (THOR_HF_DIR). Rifiuto rumoroso su ogni scarto,
    mai un lotto corto o ripetuto."""
    p = Path(percorso)
    if not p.is_file():
        raise FileNotFoundError(f"THOR_HELDOUT_CONOSCENZA={p}: file assente")
    if n_lotti < 1:
        raise ValueError(f"THOR_HELDOUT_CONOSCENZA_LOTTI={n_lotti}: serve almeno 1 lotto")
    ids = torch.load(p, weights_only=True)
    if not (isinstance(ids, torch.Tensor) and ids.dtype == torch.int64 and ids.dim() == 2):
        raise ValueError(f"THOR_HELDOUT_CONOSCENZA={p}: atteso un tensore int64 (n_seq, L_SEQ), ricevuto "
                         f"{type(ids).__name__} {getattr(ids, 'dtype', '')} {tuple(getattr(ids, 'shape', ()))}")
    if ids.shape[1] != l_seq:
        raise ValueError(f"THOR_HELDOUT_CONOSCENZA={p}: sequenze da {ids.shape[1]} token, la forgia usa L_SEQ={l_seq}")
    if int(ids.min()) < 0:
        raise ValueError(f"THOR_HELDOUT_CONOSCENZA={p}: token negativi")
    serve = n_lotti * batch_seq
    if serve > ids.shape[0]:
        raise ValueError(f"THOR_HELDOUT_CONOSCENZA={p}: {n_lotti} lotti x {batch_seq} sequenze = {serve}, il file ne "
                         f"ha {ids.shape[0]} (tetto {ids.shape[0] // batch_seq} lotti)")
    man = p.with_suffix(".json")
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        sha = hashlib.sha256(np.ascontiguousarray(ids.numpy()).astype("<i8", copy=False).tobytes()).hexdigest()
        if m.get("sha256_token") != sha:
            raise RuntimeError(f"THOR_HELDOUT_CONOSCENZA={p}: sha256 dei token {sha} diverso dal manifesto "
                               f"{man.name} ({m.get('sha256_token')})")
        if "L_SEQ" in m and int(m["L_SEQ"]) != l_seq:
            raise RuntimeError(f"THOR_HELDOUT_CONOSCENZA={p}: manifesto con L_SEQ={m['L_SEQ']}, la forgia usa {l_seq}")
        tok = m.get("tokenizzatore") or {}
        if tok.get("cartella") and str(tok["cartella"]) != str(V.SRC):
            raise RuntimeError(f"THOR_HELDOUT_CONOSCENZA={p}: tokenizzato con {tok['cartella']}, la forgia usa "
                               f"THOR_HF_DIR={V.SRC}")
        if tok.get("sha256_tokenizer_json") and (Path(V.SRC) / "tokenizer.json").exists():
            if _sha256(Path(V.SRC) / "tokenizer.json") != tok["sha256_tokenizer_json"]:
                raise RuntimeError(f"THOR_HELDOUT_CONOSCENZA={p}: tokenizer.json di {V.SRC} diverso da quello del "
                                   "manifesto")
        log(f"held-out CONOSCENZA: {p} · manifesto verificato (sha256 token {sha[:16]}...)")
    else:
        log(f"⚠️ held-out CONOSCENZA: manifesto {man.name} assente, impronta dei token non verificata")
    return ids[:serve].reshape(n_lotti, batch_seq, l_seq).clone()


def _seme_iniziale_produzione(ids_batch: torch.Tensor, emb: torch.Tensor, contenitori: "tuple | None" = None) -> tuple:
    """Stessa scelta di forgia_globale_40_strati: embedding Q5_K della base (audit B12).
    `contenitori` (25/9, cache per lotto): depositi riempiti lotto per lotto (critica 2)."""
    if os.getenv("THOR_E0_EMBEDDING_QUANT", "1") == "1":
        return V.seme_embedding_quantizzato(ids_batch, emb, contenitori=contenitori)
    return None, None


def _arma_sigterm():
    """25/9 (cache per lotto, revisione risorse, difetto 2): con l'azione di default un SIGTERM
    (kill PID, systemctl stop) uccide il processo SENZA il finally di `esegui`, e la cartella
    della cache (fino a ~28 GiB) resta su @@ROOT:models@@, invisibile fino alla corsa dopo. Qui il
    primo SIGTERM diventa SystemExit(143) (lo stesso codice d'uscita del segnale) e risale al
    finally: scrittore svuotato (il punto di ripresa in scrittura si completa), cartella tolta.
    Il gestore rimette subito l'azione di default: un secondo SIGTERM uccide all'istante (la
    cartella allora la toglie la corsa dopo, o il lanciatore). Si arma solo nel thread
    principale e solo se nessuno ha gia' scelto un'azione per SIGTERM. Ritorna il gestore da
    rimettere, o None se non armato. La chiama `esegui` SOLO a cache accesa."""
    if threading.current_thread() is not threading.main_thread():
        return None
    prima = signal.getsignal(signal.SIGTERM)
    if prima != signal.SIG_DFL:
        return None

    def _termina(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        # os.write e non log(): un print dentro un gestore di segnale puo' rientrare nel buffer di stdout
        os.write(2, b"\n[SIGTERM] forgia fermata: chiudo la corsa della cache (un secondo SIGTERM uccide subito)\n")
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _termina)
    return prima


def esegui(n_strati: int = N_STRATI, w: int = W_FINESTRA, n_passi: int = N_PASSI, modo_init: str = MODO_INIT,
           init_dir: str = INIT_DIR, init_corsa: str = INIT_CORSA, salta_export: bool = False, *,
           carica_strati=None, carica_batch=None, seme_iniziale=None, ottimizza_fn=None, da_codici_fn=None,
           ripresa_dir: "Path | None" = None, scrittura_asincrona: bool = SCRITTURA_ASINCRONA,
           riusa_teacher: bool = RIUSA_TEACHER, verifica_commit_attiva: bool = VERIFICA_COMMIT,
           verifica_seme_a: "list[int] | None" = None, osservatore=None, interrompi=None,
           heldout_conoscenza: str = HELDOUT_CONOSCENZA, lotti_conoscenza: int = LOTTI_CONOSCENZA) -> dict:
    """La catena scorrevole completa. Gli argomenti con * sono punti di iniezione per il
    collaudo a scala giocattolo (default = produzione: valida_finestra_8_12).
    `heldout_conoscenza`/`lotti_conoscenza` (26/9): vedi HELDOUT_CONOSCENZA ("" = spento)."""
    carica_strati = carica_strati or (lambda layers: V.carica_finestra_corrente(layers=layers))
    carica_batch = carica_batch or V.carica_batch_calibrazione
    seme_iniziale = seme_iniziale or _seme_iniziale_produzione
    ottimizza_fn = ottimizza_fn or _funzione_ottimizza()
    verifica_seme_a = VERIFICA_SEME_A if verifica_seme_a is None else verifica_seme_a
    osserva = osservatore or (lambda evento, **dati: None)
    punto = interrompi or (lambda nome, k: None)

    if float(os.getenv("THOR_FORGIA_ALPHA", "1.0")) != 1.0:
        raise NotImplementedError("THOR_FORGIA_ALPHA != 1 non previsto dalla ricetta scorrevole")
    if modo_init not in ("teacher", "codici"):
        raise ValueError(f"THOR_SCORREVOLE_INIT={modo_init!r}: attesi 'teacher' o 'codici'")
    cartella_out = Path(V.OUT_DIR)
    corsa = V.ID_CORSA
    if modo_init == "codici":
        if not init_dir or not init_corsa:
            raise ValueError("modalita' codici: servono THOR_SCORREVOLE_INIT_DIR e THOR_SCORREVOLE_INIT_CORSA")
        if Path(init_dir).resolve() == cartella_out.resolve():
            raise ValueError(f"cartella dei codici iniziali = cartella d'uscita ({cartella_out}): rifiuto")
        verifica_cartella_init(Path(init_dir), n_strati, init_corsa, V.DOWN_TERNARIO)
        da_codici_fn = da_codici_fn or _funzione_da_codici()
    verifica_interfaccia_A(ottimizza_fn, serve_da_codici=False)
    # 26/9 (ancora_forgia.py): variabili non valide o THOR_ANCORA_A=init senza passi morbidi si rifiutano QUI,
    # prima di caricare uno strato (spenta: nessun effetto, nessuna riga di log in piu')
    _r_ancora, _rif_ancora = ancora_forgia.configurazione()
    ancora_forgia.verifica_ricetta(n_passi - V.passi_duri(n_passi), _r_ancora, _rif_ancora)
    if _r_ancora > 0:
        log(f"ANCORA accesa: r = {_r_ancora:g} (λ = r·L0/A0 al primo passo di ogni finestra), riferimento "
            f"{_rif_ancora}")
    # 26/9 sera (spd_forgia.py): stessa regola per THOR_SPD_* — valori non validi rifiutati prima di caricare uno
    # strato (spenta: nessun effetto, nessuna riga di log in piu')
    _spd = spd_forgia.configurazione()
    if _spd[0] > 0:
        log(f"SPD accesa: {spd_forgia.descrizione(*_spd)} — Selective Projection Decay (Tian et al. 2024, Alg. 2) "
            "dopo ogni passo di AdamW, θ0 = i latenti all'inizio di ogni finestra")

    finestre = pianifica_scorrevole(n_strati, w)
    log(f"═══ FORGIA SCORREVOLE {n_strati} strati, W={w}, passo 1 → {len(finestre)} finestre, "
        f"{n_passi} passi/finestra, init strato entrante: {modo_init}"
        + (f" ({init_dir}, corsa {init_corsa})" if modo_init == "codici" else "") + f" · corsa {corsa} ═══")

    cartella_out.mkdir(parents=True, exist_ok=True)
    ripresa = Ripresa(ripresa_dir or (cartella_out / "ripresa"), corsa, n_strati, w)
    (cartella_out / "ricetta_scorrevole.json").write_text(json.dumps(
        {"n_strati": n_strati, "W": w, "passo": 1, "n_passi": n_passi, "modo_init": modo_init,
         "init_dir": init_dir, "init_corsa": init_corsa, "ricetta": ripresa.ricetta}, indent=1))
    giornale = cartella_out / "giornale_scorrevole.jsonl"

    c = strati_commessi(cartella_out, n_strati, corsa, V.DOWN_TERNARIO)
    punto_ripresa = ripresa.punto_di_ripresa(c, cartella_out)
    ids_batch, n_tr = carica_batch()
    n_con = 0
    if heldout_conoscenza:
        # 26/9: i lotti di CONOSCENZA in CODA (dopo addestramento e held-out di sempre): da qui in poi sono lotti
        # come gli altri per seme, confine, punto di ripresa e cache; ottimizza_finestra_JOINT li riconosce da
        # n_batch_conoscenza e non li usa mai per un passo
        ids_con = carica_heldout_conoscenza(heldout_conoscenza, lotti_conoscenza, ids_batch.shape[1],
                                            ids_batch.shape[2])
        n_con = ids_con.shape[0]
        ids_batch = torch.cat([ids_batch, ids_con], dim=0)
        del ids_con
        log(f"held-out CONOSCENZA: {n_con} lotti x {ids_batch.shape[1]}x{ids_batch.shape[2]} token in coda "
            f"(lotti {ids_batch.shape[0] - n_con}..{ids_batch.shape[0] - 1}), mai addestrati")
    n_batch = ids_batch.shape[0]
    if punto_ripresa == n_strati:
        log(f"tutti i {n_strati} strati gia' commessi dalla corsa {corsa}: vado all'export")
        K0, finestre_da_fare = n_strati, []
    else:
        K0 = 0 if punto_ripresa is None else punto_ripresa
        finestre_da_fare = finestre[K0:]
        if K0:
            log(f"RIPRESA: strati 0..{K0-1} commessi, riparto dalla finestra [{K0},{K0+w-1}]")

    scrittore = ThreadPoolExecutor(max_workers=1)
    lavoro_prec = None

    def _consegna(fn, *a):
        """Una transizione alla volta, in ordine; un errore di scrittura emerge alla
        transizione successiva (o subito, in modalita' sincrona)."""
        nonlocal lavoro_prec
        if lavoro_prec is not None:
            lavoro_prec.result()
        if scrittura_asincrona:
            lavoro_prec = scrittore.submit(fn, *a)
        else:
            fn(*a)
            lavoro_prec = None

    def _transizione(k0, tmp, seme_h, seme_e, portati_ids, commit):
        if tmp is not None:
            ripresa.completa(k0 + 1, tmp, seme_h, seme_e, portati_ids)
        punto("dopo_ripresa", k0)
        for L, cod in commit:
            V.scrivi_codici_su_disco({L: cod})
            with open(giornale, "a") as f:
                f.write(json.dumps({"quando": time.strftime("%F %T"), "evento": "commit", "strato": L,
                                    "finestra": [k0, k0 + w - 1], "corsa": corsa,
                                    "sha1_Q": {fam: hashlib.sha1(np.ascontiguousarray(cod[fam][1])).hexdigest()
                                               for fam in sorted(cod)}}) + "\n")
            punto("dopo_commit_strato", L)
        punto("dopo_commit", k0)
        ripresa.elimina(k0)

    puliti: dict = {}
    quant: dict = {}
    portati: dict = {}
    cfg = emb = None
    h_seme = e_seme = None
    seme_vecchio = None
    t_tot = time.time()
    # 25/9 (cache per lotto): la corsa della cache si apre QUI, subito prima del try il cui finally la
    # chiude (anche dopo un'eccezione). Spenta: no-op. Protette: uscita e ripresa (la pulizia degli
    # orfani non deve mai poterle toccare).
    DL.apri_corsa(corsa, n_lotti=n_batch, protette=[cartella_out, ripresa.radice])
    sigterm_prima = _arma_sigterm() if DL.attiva() else None     # spenta: nessun gestore (difetto 2)
    try:
        for k0, k1 in finestre_da_fare:
            t0 = time.time()
            layers = list(range(k0, k1))
            ultima = k1 == n_strati
            log(f"── finestra [{k0},{k1-1}] ({k0+1}/{len(finestre)}) ──")
            # memoria: fuori gli strati non piu' in finestra, dentro solo i mancanti
            for L in [L for L in puliti if L not in layers]:
                del puliti[L], quant[L]
            _libera()
            mancanti = [L for L in layers if L not in puliti]
            for L in mancanti:
                cfg_n, emb_n, p_n, q_n = carica_strati([L])
                if cfg is None:
                    cfg, emb = cfg_n, emb_n
                else:
                    del emb_n                    # l'embedding c'e' gia': la copia ricaricata si butta
                puliti.update(p_n)
                quant.update(q_n)
            osserva("carica", finestra=k0, caricati=mancanti, attivi=sorted(puliti))
            S = torch.from_numpy(V._S1024_NP).to(emb.device)

            if k0 == K0:
                # 25/9 (cache per lotto): budget -> lotti residenti; seme nei depositi lotto per lotto
                # (spenta: fissa_residenti no-op, semina = seme_iniziale(ids_batch, emb), contenitori None)
                DL.fissa_residenti(ids_batch, cfg.hidden_size, emb.element_size(), w)
                if K0 == 0:
                    h_seme, e_seme = DL.semina(seme_iniziale, ids_batch, emb, finestra=0)
                else:
                    h_seme, e_seme, portati = ripresa.carica(K0, emb.device, contenitori=DL.contenitori_seme(K0),
                                                             n_batch_atteso=n_batch)
                if DL.attiva():
                    _libera()          # critica 2: i segmenti per lotto riversati tornano al driver
            if k0 > 0 and k0 in verifica_seme_a:
                # 25/9 00:58: la verifica LEGGE da disco i codici appena commessi, che la scrittura
                # asincrona puo' non aver ancora finito (S2 si e' fermata: marca dello strato 1 scritta
                # 5 s dopo la lettura; in S1b era andata bene per caso di tempi). Si aspetta la scrittura.
                if lavoro_prec is not None:
                    lavoro_prec.result()
                _verifica_seme_da_disco(k0, ids_batch, cfg, emb, seme_iniziale, carica_strati,
                                        cartella_out, corsa, h_seme, e_seme)

            # init: portati dalla finestra precedente; entranti dal teacher (default) o dai codici
            attesi_portati = layers[:-1] if k0 > 0 else []
            if sorted(portati) != attesi_portati:
                raise RuntimeError(f"finestra {k0}: strati portati {sorted(portati)} != attesi {attesi_portati}")
            iniziali = dict(portati)
            entranti = [L for L in layers if L not in portati]
            if modo_init == "codici":
                for L in entranti:
                    cod = leggi_codici_strato(Path(init_dir), L, init_corsa, V.DOWN_TERNARIO)
                    iniziali[L] = _da_codici_verificato(cod, cfg, emb.device, da_codici_fn, f"strato {L}")
            osserva("finestra_inizio", finestra=k0, strati=layers, h_seme=h_seme, e_seme=e_seme,
                    iniziali=iniziali, entranti=entranti)

            uscita: dict = {}
            kw = {PARAM_INIZIALI: iniziali or None, PARAM_USCITA: uscita}
            if n_con:
                kw["n_batch_conoscenza"] = n_con       # 26/9: solo se acceso (spento: la chiamata di prima)
            del iniziali
            usa_cattura = riusa_teacher and not ultima
            # 25/9 (cache per lotto): a cache accesa la cattura scrive in un deposito (contenitore None = lista)
            cattura = (_CatturaUscitaTeacher(puliti[k0], contenitore=DL.contenitore_cattura(k0 + 1))
                       if usa_cattura else None)
            with (cattura if usa_cattura else contextlib.nullcontext()):
                risultato = ottimizza_fn({L: quant[L] for L in layers}, ids_batch, cfg, n_passi,
                                         strati_teacher={L: puliti[L] for L in layers}, emb=emb, lr=V.NUCLEO_LR,
                                         h0_esterno_per_b=h_seme, e0_esterno_per_b=e_seme,
                                         n_batch_addestramento=n_tr, **kw)
            del kw
            if sorted(risultato) != layers or sorted(uscita) != layers:
                raise RuntimeError(f"finestra {k0}: risultato {sorted(risultato)} / uscita {sorted(uscita)} "
                                   f"!= strati {layers}")
            for L, fam in portati.items():
                if uscita[L]["gu"] is not fam["gu"] or uscita[L]["dw"] is not fam["dw"]:
                    raise RuntimeError(f"finestra {k0}: lo strato portato {L} NON e' stato ottimizzato "
                                       "dall'oggetto ricevuto (interfaccia A violata)")
            osserva("finestra_fine", finestra=k0, risultato=risultato, uscita=uscita)
            punto("prima_transizione", k0)

            commessi = strati_commessi_da((k0, k1), n_strati)
            # gli oggetti ternari degli strati che non si portano oltre si liberano QUI
            # (prima della verifica del commit, che alloca ~5 GiB transitori)
            portati = {} if ultima else {L: uscita[L] for L in layers[1:]}
            del uscita
            _libera()
            if verifica_commit_attiva:
                for L in commessi:
                    verifica_commit(quant[L], risultato[L], S, L)

            # 25/9 (cache per lotto, critica 8): il seme di QUESTA finestra si chiude dopo la consegna qui
            # sotto (che aspetta la transizione k-1, l'ultima a leggerlo), in ENTRAMBI i rami (anche
            # l'ultima finestra). Solo a cache accesa: spenta non si tiene nessun riferimento in piu'.
            seme_vecchio = (h_seme, e_seme) if DL.attiva() else None
            if not ultima:
                teacher_k = cattura.per_batch(n_batch) if usa_cattura else None
                if teacher_k is not None:
                    rel = _controlla_teacher_catturato(puliti[k0], quant[k0], ids_batch, cfg, emb,
                                                       h_seme, teacher_k, k0)
                    log(f"  teacher dello strato {k0} riusato (batch 0 ricalcolato: scarto relativo {rel:.1e})")
                h_seme, e_seme = calcola_confine_a_meta(puliti, quant, [k0], ids_batch, cfg, emb,
                                                        h_seme, e_seme, h_teacher_uscita_per_b=teacher_k,
                                                        contenitori=DL.contenitori_seme(k0 + 1))
                DL.chiudi(teacher_k)       # la cattura, una volta sola: il seme nuovo nomina gli stessi file
                del teacher_k
                for fam in portati.values():
                    for obj in (fam["gu"], fam["dw"]):
                        if obj is not None:
                            for p in obj.parameters():
                                p.grad = None
                tmp = ripresa.prepara(k0 + 1, portati)
                _consegna(_transizione, k0, tmp, h_seme, e_seme, sorted(portati),
                          [(L, risultato[L]) for L in commessi])
            else:
                _consegna(_transizione, k0, None, None, None, [], [(L, risultato[L]) for L in commessi])
            if seme_vecchio is not None:
                DL.chiudi(*seme_vecchio)
                seme_vecchio = None
            fam = obj = p = None     # (raccordo 24/9) i nomi di ciclo non tengono vivi gli oggetti portati
            for L in commessi:
                osserva("commit", finestra=k0, strato=L, codici=risultato[L])
            del risultato, cattura
            _libera()
            log(f"── finestra [{k0},{k1-1}] FATTA in {time.time()-t0:.1f}s · commessi {commessi} ──")
        if lavoro_prec is not None:
            lavoro_prec.result()
    finally:
        # (difetto 2) il gestore torna quello di prima IN TESTA: rimesso anche se la pulizia qui sotto
        # solleva; un SIGTERM durante la pulizia agisce come oggi (la cartella la toglie il lanciatore)
        if sigterm_prima is not None:
            signal.signal(signal.SIGTERM, sigterm_prima)
        scrittore.shutdown(wait=True)
        puliti.clear()
        quant.clear()
        portati = {}
        h_seme = e_seme = emb = None
        seme_vecchio = None
        # 25/9 (cache per lotto): DOPO lo shutdown dello scrittore (l'ultima transizione ha letto il
        # seme); rimuove la cartella anche dopo un'eccezione, e non maschera l'eccezione in volo.
        DL.chiudi_corsa()
        _libera()

    log(f"═══ {len(finestre_da_fare)} finestre in {time.time()-t_tot:.1f}s ═══")
    c = strati_commessi(cartella_out, n_strati, corsa, V.DOWN_TERNARIO)
    if c != n_strati:
        raise RuntimeError(f"a fine catena commessi {c} strati su {n_strati}")
    if salta_export:
        log("--salta-export: fermo prima dell'esportazione GGUF")
        return {"finestre": finestre, "cartella": str(cartella_out)}
    path = V.esporta_gguf_finestra(layers=list(range(n_strati)))
    log(f"GGUF finale: {path}")
    return {"finestre": finestre, "cartella": str(cartella_out), "gguf": str(path)}


def _verifica_seme_da_disco(k, ids_batch, cfg, emb, seme_iniziale, carica_strati, cartella, corsa, h_seme, e_seme):
    """Seme della catena (in memoria o dal punto di ripresa) contro il seme ricostruito
    dai codici su disco, batch 0. Stesso algoritmo: deve coincidere salvo gli ULP
    exp compilato/eager (vedi verifica_commit)."""
    t0 = time.time()
    seme0 = seme_iniziale(ids_batch[:1], emb)
    h_r, e_r = ricostruisci_seme(k, ids_batch[:1], cfg, emb, seme0, carica_strati, cartella, corsa)
    rt, rq = _confronta_semi(h_seme, e_seme, h_r, e_r, {0: 0})
    log(f"  seme della finestra {k} vs ricostruito dai codici su disco (batch 0, {time.time()-t0:.0f}s): "
        f"teacher {rt:.2e} · quantizzato {rq:.2e}")
    if rt > TOLL_SEME_RICOSTRUITO or rq > TOLL_SEME_RICOSTRUITO:
        raise RuntimeError(f"seme della finestra {k} diverso dalla ricostruzione da disco "
                           f"(teacher {rt:.2e}, quantizzato {rq:.2e} > {TOLL_SEME_RICOSTRUITO:.0e})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salta-export", action="store_true")
    ap.add_argument("--verifica-interfaccia", action="store_true",
                    help="controlla solo che l'interfaccia del compito A ci sia (niente GPU, niente dati)")
    args = ap.parse_args()
    if args.verifica_interfaccia:
        verifica_interfaccia_A(_funzione_ottimizza(), serve_da_codici=MODO_INIT == "codici")
        print(f"interfaccia A presente ({PARAM_INIZIALI}, {PARAM_USCITA}"
              + (f", {MODULO_DA_CODICI}.{FUNZ_DA_CODICI}" if MODO_INIT == "codici" else "") + ")")
        if MODO_INIT == "codici":
            if not INIT_DIR or not INIT_CORSA:
                raise SystemExit("modalita' codici: servono THOR_SCORREVOLE_INIT_DIR e THOR_SCORREVOLE_INIT_CORSA")
            verifica_cartella_init(Path(INIT_DIR), N_STRATI, INIT_CORSA, V.DOWN_TERNARIO)
            print(f"cartella dei codici iniziali completa: {INIT_DIR} ({N_STRATI} strati, corsa {INIT_CORSA})")
        print(f"pianificazione: {len(pianifica_scorrevole(N_STRATI, W_FINESTRA))} finestre")
        return 0
    esegui(salta_export=args.salta_export,
           ripresa_dir=Path(os.environ["THOR_SCORREVOLE_RIPRESA_DIR"])
           if os.getenv("THOR_SCORREVOLE_RIPRESA_DIR") else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
