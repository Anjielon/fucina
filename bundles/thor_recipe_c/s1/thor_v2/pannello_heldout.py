#!/usr/bin/env python3
"""PANNELLO HELD-OUT ALLARGATO per la finestra [0,1] di Thor (leva #1, prova A1 di
`thor_v2/PIANO_LEVE_GAP_Q2K.md`). 24/9/2026, sera.

## Perche' esiste

Oggi il cancello di finestra ha UN held-out: 1 batch da 9 esempi (6 math, 3 code),
8x1024 token, solo in formato chat. Le ricette Q1/Q2/Q4 (0,509 / 0,502 / 0,499x
v20) dentro quel rumore non si distinguono, e nessuno sa se il guadagno misurato
su math/code in formato chat vale anche su TESTO GREZZO, cioe' sulla metrica del
mandato (PPL wikitext a c=4096, che conta solo le posizioni 2048-4095).
Questo script costruisce un pannello piu' largo e rivaluta, SOLO IN AVANTI, i
codici gia' salvati su disco delle corse K2, Q1, Q2, Q3, Q4 (piu' esperti v20 e
init LS), cosi' le regole di decisione di A1 (scritte prima di guardare i
numeri, PIANO sez. 2) si possono applicare.

## Il pannello (costruito sempre allo stesso modo, deterministico)

| dominio | cosa | forma |
|---|---|---|
| `storico` | l'held-out di sempre (esempi 0-8 del mescolamento, seme 1234) | 1 x 8 x 1024 |
| `math_code` | esempi math/code in formato chat, DOPO l'ultimo esempio usato da qualunque corsa (vedi sotto) | 16 x 8 x 1024 |
| `wiki` | wikitext-2-raw-v1 VALIDATION, testo grezzo, token 0 .. 131071 | 16 x 8 x 1024 |
| `edu` | FineWeb-Edu sample-10BT, documenti dall'inizio uniti con `\\n\\n`, testo grezzo | 16 x 8 x 1024 |
| `lunghe` | wikitext VALIDATION, token 131072 .. 163839, sequenze da 4096 | 4 x 2 x 4096 |

Testo grezzo = SENZA template chat e senza token speciali (tokenizzatore di
`@@ROOT:models@@/gguf/thor-src-hf`). Fasce di posizione sulle `lunghe`: [0,1024),
[1024,2048), [2048,4096) — l'ultima e' quella che la PPL a c=4096 conta.

**Disgiunzione math/code.** Le corse impacchettano gli esempi mescolati (seme
1234) cosi' (`carica_batch_calibrazione`): held-out = i primi esempi finche'
riempiono N_HELDOUT x BATCH_SEQ x 1024 token, addestramento = gli esempi
successivi finche' riempiono N_BATCH x BATCH_SEQ x 1024. L'ultimo esempio preso
e' troncato ma conta come USATO. Il pannello parte dal primo esempio mai
toccato dalla configurazione piu' vorace fra quelle che hanno prodotto codici
(`CONFIG_ADDESTRAMENTO`: Q1-Q4 e la forgia ThorT-veloce 32x8, 1 held-out ->
esempi 0-278; K2 32x2, 4 held-out -> esempi 0-82). La replica dell'impacchettamento
e' verificata CONTRO la funzione vera (`verifica_replica`): stessi identici token.
⚠️ Vale per il seme 1234: una corsa con `THOR_CALIB_SEED` diverso (A4 lo prevede,
4321) mescola in un altro ordine e PUO' addestrarsi su esempi del pannello.

## Valutazione (finestra [0,1], sola andata, niente gradiente)

Stesso metro dell'held-out di `ottimizza_finestra_JOINT` (lo stesso numero, se
la batch e' la stessa): seme della finestra 0 = embedding Q5_K della base (B12),
teacher bf16 precalcolato, loss DURA = somma sui due strati di mean(e_{l+1}^2)
(`errore_lemma1_finestra`, diagnostica spenta: stessa loss al bit). Routing come
nella valutazione FINALE di JOINT: il router VERO sceglie gli esperti sul flusso
dello studente (`_congela_routing_chunk` coi codici della ricetta), poi la loss
si calcola con quella scelta. `--routing vivo` lascia decidere il router dentro
il forward della loss (differisce solo per l'arrotondamento bf16 dell'ingresso).
- `v20`: esperti della base di produzione (`_stratofn_reale`, router vero).
- `init_LS`: codici dell'init LS dai pesi originali, calcolati come fa JOINT.
- ricette: codici letti da disco (`Dg/Qg`, `Du/Qu`, `Dd/Qd` .npy, marca
  `fatto_forgia_LL.ok` con `down_ternario=1`), ricostruiti come W = D (x) Q per
  gruppo; gate e up si concatenano (righe 0:512 gate, 512:1024 up) e si
  DE-RUOTANO (R1024) come nel commit di JOINT; il down no. Scale in fp32 come
  in JOINT (`--scale-fp16` le arrotonda come le vede llama.cpp).

Uscite per ricetta e dominio: loss media; rapporto APPAIATO per batch contro
v20 (media +- errore standard); rapporto delle medie; rapporto appaiato contro
K2 (K2_pan); per fascia di posizione il rapporto appaiato per sequenza e la
pendenza R[ultima]/R[prima]; confronto dello `storico` coi numeri dei log;
segnali delle regole di A1 applicate meccanicamente (decide il coordinatore).
Tabella markdown + json in `@@ROOT:forge@@/stato/pannello/`.

## Uso

    # 1) dati pubblici (una volta): scarica_dati_pannello.py
    # 2) pannello su CPU (tokenizzazione + verifiche), salvato in @@ROOT:models@@/gguf/tony-tern/pannello/
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 pannello_heldout.py --solo-pannello
    # 3) prova della logica a scala minuscola, SOLO CPU (strati giocattolo + codici veri a campione)
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 pannello_heldout.py --prova-cpu
    # 4) esecuzione VERA (GPU, dopo la forgia; una corsia pesante per volta)
    unset HSA_OVERRIDE_GFX_VERSION
    @@ROOT:venv@@/bin/python3 pannello_heldout.py [--routing congelato|vivo] \\
        [--extra ThorT-veloce=@@ROOT:models@@/gguf/tony-tern/v2-ThorT-veloce] [--vram-max-gib 10]

La guardia GPU e' quella di `valida_finestra_8_12.verifica_gpu_libera` (VRAM
usata <= `--vram-max-gib`, GPU <= `--gpu-pct-max`, nessuna unita' thor-*/adhoc-*
attiva). Con la casa accesa serve alzare `--vram-max-gib` e avere ~30 GiB liberi.

## Stima per l'esecuzione VERA (dichiarata come stima, non misura)

Basi misurate (log `stato/velocita/Q4_aggiornato_lr0.01_150.log`): i due strati
pulito+corrente con la cache strati si caricano in ~8 s e occupano 7,23 GiB;
init LS dei due strati ~9 s; un held-out di 1 batch 8x1024 (ricostruzione dei
pesi + routing + loss) ~2 s.
- **VRAM, picco stimato 25-30 GiB**: strati + embedding 7,2; semi (h0 bf16 + e0
  fp32) e teacher precalcolati per 53 batch da 8192 token ~8,3; pesi di UNA
  ricetta alla volta 3,0; transitorio di ricostruzione (gate_up fp32 2 GiB +
  temporanei della FWHT) o dell'init LS ~6-8; attivazioni < 2. Il valore vero
  si legge nel json (`torch.cuda.max_memory_allocated`).
- **Tempo, stima ~10 min (forchetta 8-15)**: caricamento + pannello ~1 min;
  7 valutazioni (v20, init, 5 ricette) x 53 batch x ~1 s = ~6 min; letture dei
  codici (1,6 GiB per ricetta) e init LS ~1-2 min. Il json riporta i tempi veri.

## File letti in SOLA LETTURA

`valida_finestra_8_12.py` (importato: carica_finestra_corrente,
seme_embedding_quantizzato, carica_batch_calibrazione, _congela_routing_chunk,
_stratofn_reale, _stratofn_quant_reale, _gate_up_per_forward, _costruisci_
ternari_layer, _esporta_gate_up, _estrai_famiglia, scrivi_codici_su_disco,
verifica_gpu_libera ...) · `finestra_globale.py` · `init_ls.py` · i dati di
`@@ROOT:models@@/gguf/tony-tern/pannello/` (vedi `scarica_dati_pannello.py`).
Nessuno di questi file viene modificato: dove serve un parametro diverso (es.
N_BATCH per replicare una configurazione) si cambia l'attributo del modulo in
memoria e lo si ripristina subito dopo.
"""
from __future__ import annotations

import os
import sys

PROVA_CPU = "--prova-cpu" in sys.argv
SOLO_PANNELLO = "--solo-pannello" in sys.argv
if PROVA_CPU or SOLO_PANNELLO:
    # niente GPU per costruzione, niente compilazione (Inductor su CPU accende un
    # compilatore per core: vietato con la forgia accesa). Va fatto PRIMA di torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
for _k, _v in (("TOKENIZERS_PARALLELISM", "false"), ("RAYON_NUM_THREADS", "2"), ("OMP_NUM_THREADS", "2"),
               ("TORCHINDUCTOR_COMPILE_THREADS", "1")):
    os.environ.setdefault(_k, _v)
if not (PROVA_CPU or SOLO_PANNELLO):
    os.environ.setdefault("THOR_CACHE_STRATI", "@@ROOT:models@@/gguf/tony-tern/cache_strati")   # identica al bit

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import valida_finestra_8_12 as V  # noqa: E402  (SOLA LETTURA)
from finestra_globale import (  # noqa: E402  (SOLA LETTURA)
    errore_lemma1_finestra, precalcola_teacher, GRUPPO_GATEUP_REALE, GRUPPO_DOWN_REALE, _group_giocattolo,
)
from init_ls import reinit_ls_  # noqa: E402  (SOLA LETTURA)

log = V.log

# ─────────────────────────────── configurazione ───────────────────────────────
DIR_DATI = Path(os.getenv("THOR_PANNELLO_DATI", "@@ROOT:models@@/gguf/tony-tern/pannello"))
DIR_USCITA = Path(os.getenv("THOR_PANNELLO_USCITA", "@@ROOT:forge@@/stato/pannello"))
FILE_WIKI = DIR_DATI / "wikitext-2-raw-v1.validation.raw"
FILE_EDU = DIR_DATI / "fineweb-edu.sample-10BT.000_00000.rg0-1.jsonl"
FILE_PANNELLO = DIR_DATI / "pannello_ids.pt"
SEPARATORE_EDU = "\n\n"
SEME_CALIB = 1234
FINESTRA = (0, 1)
L_SEQ_ADDESTRAMENTO = 1024

_TT = Path("@@ROOT:models@@/gguf/tony-tern")
RICETTE = {   # nome -> cartella dei codici salvati da scrivi_codici_su_disco (finestra [0,1])
    "K2": _TT / "v2-sweep-K2_lr005_400_ric50",
    "Q1": _TT / "v2-veloce-Q1_bs8_150_lr005",
    "Q2": _TT / "v2-veloce-Q2_bs8_150_lr01",
    "Q3": _TT / "v2-veloce-Q3_completa_lr0.01_150",
    "Q4": _TT / "v2-veloce-Q4_aggiornato_lr0.01_150",
}
# Configurazioni che hanno visto dati di calibrazione (nome, N_BATCH, BATCH_SEQ, N_HELDOUT, seme).
CONFIG_ADDESTRAMENTO = (
    ("Q1-Q4 e forgia ThorT-veloce (8 seq x 32 batch, 1 held-out)", 32, 8, 1, SEME_CALIB),
    ("K2 (2 seq x 32 batch, 4 held-out)", 32, 2, 4, SEME_CALIB),
)
# held-out storico: valori finali dai log (stessi token dello `storico`). Percorsi numerici
# dichiarati: Q1-Q4 misurati con rotazione sull'ingresso (THOR_ROTAZIONE_INGRESSO=1),
# K2 con 4 batch da 2 sequenze: scarti di qualche 1e-4 relativo sono attesi.
RIFERIMENTI_STORICO = {
    "v20": (1.5576e-05, "stato/velocita/Q1..Q4 (1 batch 8x1024); K2 dava 1.5573e-05 su 4x2"),
    "init_LS": (3.3340e-05, "stato/velocita/Q1, Q2, Q4"),
    "K2": (8.8666e-06, "stato/sweep_v2/K2_lr005_400_ric50.log (4 batch x 2 seq, rotazione sui pesi)"),
    "Q1": (7.9320e-06, "stato/velocita/Q1_bs8_150_lr005.log (rot=1)"),
    "Q2": (7.8123e-06, "stato/velocita/Q2_bs8_150_lr01.log (rot=1)"),
    "Q3": (8.3273e-06, "stato/velocita/Q3_completa_lr0.01_150.log (rot=1, routing VIVO)"),
    "Q4": (7.7669e-06, "stato/velocita/Q4_aggiornato_lr0.01_150.log (rot=1)"),
}
TOLLERANZA_STORICO = 0.01   # 1% relativo: sopra, la ricostruzione o il protocollo differiscono davvero


@dataclass(frozen=True)
class Scala:
    n_batch: int            # batch per dominio (math_code, wiki, edu)
    seq_per_batch: int
    l_seq: int
    n_lunghe: int           # sequenze lunghe per le fasce di posizione
    l_lunga: int
    lunghe_per_batch: int
    fasce: tuple


SCALA_VERA = Scala(16, 8, 1024, 8, 4096, 2, ((0, 1024), (1024, 2048), (2048, 4096)))
SCALA_PROVA = Scala(1, 2, 32, 2, 64, 2, ((0, 16), (16, 32), (32, 64)))
DOMINI = ("storico", "math_code", "wiki", "edu", "lunghe")
# 25/9: domini NUOVI (italiano, chat, ragionamento, strumenti) da un file A PARTE costruito da
# costruisci_corpus_calibrazione.py, disgiunto dalla calibrazione. pannello_ids.pt non cambia.
DOMINI_STORICI = DOMINI
FILE_NUOVI = Path(os.getenv("THOR_PANNELLO_NUOVI", str(DIR_DATI / "pannello_nuovi_domini.pt")))
_SCELTI_NUOVI = os.getenv("THOR_PANNELLO_NUOVI_DOMINI", "")        # vuoto = tutti · "0" = nessuno
if _SCELTI_NUOVI == "0" or not FILE_NUOVI.with_suffix(".json").exists():
    DOMINI_NUOVI: tuple = ()
else:
    _FORME_NUOVI = json.loads(FILE_NUOVI.with_suffix(".json").read_text(encoding="utf-8"))["forme"]
    DOMINI_NUOVI = tuple(d for d in _FORME_NUOVI if not _SCELTI_NUOVI or d in _SCELTI_NUOVI.split(","))
    _ignoti = set(_SCELTI_NUOVI.split(",")) - set(_FORME_NUOVI) if _SCELTI_NUOVI else set()
    if _ignoti:
        raise ValueError(f"THOR_PANNELLO_NUOVI_DOMINI: domini {sorted(_ignoti)} assenti da {FILE_NUOVI}")
DOMINI = DOMINI_STORICI + DOMINI_NUOVI


# ═══════════════════════════ 1. DATI E PANNELLO (CPU) ═══════════════════════════
def tokenizzatore():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(V.SRC))


def esempi_calibrazione(tok, seme: int = SEME_CALIB) -> list:
    """Replica ESATTA della prima meta' di `carica_batch_calibrazione` (stessi file,
    stessi filtri, stesso template, stesso mescolamento). Ogni voce: (tipo, testo, q).
    `random.shuffle` permuta per indice: il terzo campo non cambia l'ordine."""
    cartella = V.CORPUS.parent
    esempi = []
    for tipo in ("math", "code"):
        for riga in (cartella / f"cot_{tipo}.jsonl").open(encoding="utf-8"):
            try:
                d = json.loads(riga)
            except json.JSONDecodeError:
                continue
            if d.get("errore") or not d.get("risposta"):
                continue
            testo = tok.apply_chat_template(
                [{"role": "user", "content": d["q"]},
                 {"role": "assistant", "content": d["risposta"], "reasoning_content": d.get("ragionamento", "")}],
                tokenize=False)
            esempi.append((tipo, testo, d["q"]))
    random.Random(seme).shuffle(esempi)
    return esempi


def impacca(tok, esempi: list, inizio: int, n_batch: int, batch_seq: int, l_seq: int, cache: dict):
    """Replica di `_impacca` (dentro carica_batch_calibrazione) a partire da `inizio`.
    Ritorna (ids (n_batch, batch_seq, l_seq) int64, esempi usati, conta per tipo)."""
    per_batch = batch_seq * l_seq
    flusso, usati, conta = [], 0, {"math": 0, "code": 0}
    while len(flusso) < n_batch * per_batch:
        i = inizio + usati
        if i >= len(esempi):
            raise RuntimeError(f"esempi finiti: servono {n_batch} batch da {per_batch} token da {inizio}")
        if i not in cache:
            cache[i] = tok(esempi[i][1], add_special_tokens=False).input_ids
        flusso += cache[i]
        conta[esempi[i][0]] += 1
        usati += 1
    ids = torch.tensor(flusso[:n_batch * per_batch], dtype=torch.int64)
    return ids.reshape(n_batch, batch_seq, l_seq), usati, conta


def primo_esempio_libero(tok, esempi: list, cache: dict) -> "tuple[int, list]":
    """Indice del primo esempio mai toccato da nessuna configurazione di CONFIG_ADDESTRAMENTO."""
    dettagli = []
    for nome, nb, bs, nho, seme in CONFIG_ADDESTRAMENTO:
        if seme != SEME_CALIB:
            raise ValueError(f"{nome}: seme {seme} diverso da {SEME_CALIB}: l'ordine degli esempi cambia")
        _, u_ho, c_ho = impacca(tok, esempi, 0, nho, bs, L_SEQ_ADDESTRAMENTO, cache)
        _, u_tr, c_tr = impacca(tok, esempi, u_ho, nb, bs, L_SEQ_ADDESTRAMENTO, cache)
        dettagli.append({"config": nome, "heldout_esempi": [0, u_ho], "addestramento_esempi": [u_ho, u_ho + u_tr],
                         "conta_heldout": c_ho, "conta_addestramento": c_tr})
    return max(d["addestramento_esempi"][1] for d in dettagli), dettagli


def verifica_replica(tok, esempi: list, cache: dict) -> dict:
    """La replica dell'impacchettamento deve dare GLI STESSI token di
    `carica_batch_calibrazione` per ogni configurazione (attributi del modulo e
    variabili d'ambiente cambiati in memoria e ripristinati subito)."""
    esito = {}
    for nome, nb, bs, nho, seme in CONFIG_ADDESTRAMENTO:
        salva_attr = (V.N_BATCH, V.BATCH_SEQ, V.L_SEQ)
        salva_env = {k: os.environ.get(k) for k in ("THOR_VALIDA_N_HELDOUT", "THOR_CALIB_SEED", "THOR_CALIB_MISCELA")}
        salva_corpus = getattr(V, "CALIB_CORPUS", None)
        try:
            V.N_BATCH, V.BATCH_SEQ, V.L_SEQ = nb, bs, L_SEQ_ADDESTRAMENTO
            os.environ.pop("THOR_CALIB_MISCELA", None)
            if salva_corpus is not None:
                V.CALIB_CORPUS = ""
            os.environ["THOR_VALIDA_N_HELDOUT"] = str(nho)
            os.environ["THOR_CALIB_SEED"] = str(seme)
            ids_rif, n_tr = V.carica_batch_calibrazione()
        finally:
            V.N_BATCH, V.BATCH_SEQ, V.L_SEQ = salva_attr
            if salva_corpus is not None:
                V.CALIB_CORPUS = salva_corpus
            for k, v in salva_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        ho, u_ho, _ = impacca(tok, esempi, 0, nho, bs, L_SEQ_ADDESTRAMENTO, cache)
        tr, _, _ = impacca(tok, esempi, u_ho, nb, bs, L_SEQ_ADDESTRAMENTO, cache)
        uguali = bool(n_tr == nb and torch.equal(torch.cat([tr, ho]), ids_rif))
        esito[nome] = uguali
        log(f"  replica di carica_batch_calibrazione [{nome}]: {'✅ token identici' if uguali else '⛔ DIVERSA'}")
        if not uguali:
            raise RuntimeError(f"replica dell'impacchettamento diversa dalla funzione vera per {nome}")
    return esito


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def costruisci_pannello(tok, sc: Scala, verifica: bool = True) -> dict:
    """Tutti i domini, deterministici. Ritorna {"domini": {nome: ids 3D}, "fasce": ..., "meta": ...}."""
    t0 = time.time()
    esempi = esempi_calibrazione(tok)
    cache: dict = {}
    replica = verifica_replica(tok, esempi, cache) if verifica else None
    inizio, dettagli = primo_esempio_libero(tok, esempi, cache)

    # storico: l'held-out delle corse Q (1 x 8 x 1024); a scala di prova, la sua testa
    storico, u_st, c_st = impacca(tok, esempi, 0, 1, 8, L_SEQ_ADDESTRAMENTO, cache)
    if sc != SCALA_VERA:
        storico = storico.reshape(-1)[:sc.seq_per_batch * sc.l_seq].reshape(1, sc.seq_per_batch, sc.l_seq)

    mc, u_mc, c_mc = impacca(tok, esempi, inizio, sc.n_batch, sc.seq_per_batch, sc.l_seq, cache)
    q_addestr = {esempi[i][2].strip() for i in range(inizio)}
    t_addestr = {esempi[i][1] for i in range(inizio)}
    q_dup = sum(1 for i in range(inizio, inizio + u_mc) if esempi[i][2].strip() in q_addestr)
    t_dup = sum(1 for i in range(inizio, inizio + u_mc) if esempi[i][1] in t_addestr)

    testo_wiki = FILE_WIKI.read_text(encoding="utf-8")
    ids_wiki = torch.tensor(tok(testo_wiki, add_special_tokens=False).input_ids, dtype=torch.int64)
    n_w = sc.n_batch * sc.seq_per_batch * sc.l_seq
    n_l = sc.n_lunghe * sc.l_lunga
    if ids_wiki.numel() < n_w + n_l:
        raise RuntimeError(f"wikitext validation: {ids_wiki.numel()} token, ne servono {n_w + n_l}")
    wiki = ids_wiki[:n_w].reshape(sc.n_batch, sc.seq_per_batch, sc.l_seq)
    lunghe = ids_wiki[n_w:n_w + n_l].reshape(sc.n_lunghe // sc.lunghe_per_batch, sc.lunghe_per_batch, sc.l_lunga)

    flusso, n_doc = [], 0
    with FILE_EDU.open(encoding="utf-8") as f:
        for riga in f:
            if len(flusso) >= n_w:
                break
            flusso += tok(json.loads(riga)["text"] + SEPARATORE_EDU, add_special_tokens=False).input_ids
            n_doc += 1
    if len(flusso) < n_w:
        raise RuntimeError(f"FineWeb-Edu: {len(flusso)} token, ne servono {n_w}")
    edu = torch.tensor(flusso[:n_w], dtype=torch.int64).reshape(sc.n_batch, sc.seq_per_batch, sc.l_seq)

    domini = {"storico": storico, "math_code": mc, "wiki": wiki, "edu": edu, "lunghe": lunghe}
    man = json.loads((DIR_DATI / "manifesto.json").read_text()) if (DIR_DATI / "manifesto.json").exists() else {}
    meta = {
        "scala": sc.__dict__ if hasattr(sc, "__dict__") else str(sc),
        "forme": {k: list(v.shape) for k, v in domini.items()},
        "sha256_token": {k: _sha(v) for k, v in domini.items()},
        "fasce": [list(f) for f in sc.fasce],
        "math_code": {"seme": SEME_CALIB, "esempi_disponibili": len(esempi), "primo_esempio_del_pannello": inizio,
                      "esempi_usati": [inizio, inizio + u_mc], "conta": c_mc,
                      "configurazioni_di_addestramento": dettagli,
                      "replica_verificata_contro_carica_batch_calibrazione": replica,
                      "domande_del_pannello_gia_viste_in_esempi_precedenti": q_dup,
                      "testi_identici_a_esempi_precedenti": t_dup},
        "storico": {"esempi": [0, u_st], "conta": c_st},
        "wiki": {"file": FILE_WIKI.name, "token_totali": int(ids_wiki.numel()), "token_batch": [0, n_w],
                 "token_lunghe": [n_w, n_w + n_l], "revisione_hf": man.get("wikitext", {}).get("revisione"),
                 "sha256_file": man.get("wikitext", {}).get("sha256_scritto")},
        "edu": {"file": FILE_EDU.name, "documenti_usati": [0, n_doc], "separatore": SEPARATORE_EDU,
                "nota": "chi usa FineWeb-Edu per ADDESTRARE parte dal documento " + str(n_doc),
                "revisione_hf": man.get("fineweb_edu", {}).get("revisione"),
                "sha256_file": man.get("fineweb_edu", {}).get("sha256_scritto")},
        "tokenizzatore": str(V.SRC), "secondi": round(time.time() - t0, 1),
    }
    log(f"pannello costruito in {meta['secondi']} s · math_code dagli esempi {inizio}..{inizio + u_mc - 1} "
        f"({c_mc['math']} math, {c_mc['code']} code; domande gia' viste prima: {q_dup}, testi identici: {t_dup}) · "
        f"wiki {ids_wiki.numel()} token · edu {n_doc} documenti · forme {meta['forme']}")
    return {"domini": domini, "fasce": sc.fasce, "meta": meta}


def aggiungi_nuovi_domini(p: dict, sc: Scala) -> dict:
    """Aggiunge a `p` i domini nuovi (file a parte, sha256 di ogni dominio verificato col suo
    manifesto). A scala di prova ne prende la testa, come `costruisci_pannello` con lo storico."""
    if not DOMINI_NUOVI:
        log(f"domini nuovi del pannello: nessuno ({FILE_NUOVI} assente o THOR_PANNELLO_NUOVI_DOMINI=0)")
        return p
    nuovi = torch.load(FILE_NUOVI, weights_only=False)
    meta = nuovi["meta"]
    for d in DOMINI_NUOVI:
        t = nuovi["domini"][d]
        if _sha(t) != meta["sha256_token"][d]:
            raise RuntimeError(f"{FILE_NUOVI}: dominio {d} non combacia con lo sha256 del suo manifesto")
        if sc != SCALA_VERA:
            t = t.reshape(-1)[:sc.seq_per_batch * sc.l_seq].reshape(1, sc.seq_per_batch, sc.l_seq)
        p["domini"][d] = t
        p["meta"]["forme"][d] = list(t.shape)
        p["meta"]["sha256_token"][d] = _sha(t)
    p["meta"]["nuovi_domini"] = {
        "file": str(FILE_NUOVI), "domini": list(DOMINI_NUOVI),
        "manifesto": {k: meta[k] for k in ("quando", "forme", "sha256_token", "seme", "soglia_pannello_hash",
                                           "k_massive", "costruttore") if k in meta},
        "disgiunzione_esito": (meta.get("disgiunzione") or {}).get("esito")}
    log(f"domini nuovi del pannello da {FILE_NUOVI.name}: " + ", ".join(
        f"{d} {tuple(p['domini'][d].shape)}" for d in DOMINI_NUOVI)
        + f" · disgiunzione {p['meta']['nuovi_domini']['disgiunzione_esito']}")
    return p


def pannello_vero(tok, ricostruisci: bool = False) -> dict:
    """Costruisce il pannello vero e lo confronta con quello salvato (determinismo):
    se differisce si ferma, salvo `--ricostruisci-pannello`."""
    p = costruisci_pannello(tok, SCALA_VERA)
    if FILE_PANNELLO.exists() and not ricostruisci:
        salvato = torch.load(FILE_PANNELLO, weights_only=False)
        diversi = [k for k in DOMINI_STORICI if not torch.equal(salvato["domini"][k], p["domini"][k])]
        if diversi:
            raise RuntimeError(f"pannello ricostruito DIVERSO da {FILE_PANNELLO} nei domini {diversi}: "
                               "tokenizzatore o dati cambiati. --ricostruisci-pannello per sovrascrivere")
        log(f"pannello identico a quello salvato in {FILE_PANNELLO} ✅")
    else:
        torch.save(p, FILE_PANNELLO)
        (FILE_PANNELLO.with_suffix(".json")).write_text(json.dumps(p["meta"], ensure_ascii=False, indent=2))
        log(f"pannello salvato: {FILE_PANNELLO} (+ .json)")
    return aggiungi_nuovi_domini(p, SCALA_VERA)


# ═══════════════════════════ 2. CODICI DELLE RICETTE ═══════════════════════════
NOMI_FAMIGLIE = {"Dg": "ffn_gate_exps", "Qg": "ffn_gate_exps", "Du": "ffn_up_exps", "Qu": "ffn_up_exps",
                 "Dd": "ffn_down_exps", "Qd": "ffn_down_exps"}


def leggi_marca(cartella: Path, L: int) -> dict:
    p = cartella / f"fatto_forgia_{L:02d}.ok"
    if not p.exists():
        raise RuntimeError(f"{p} manca: codici dello strato {L} incompleti")
    t = p.read_text().strip()
    import re
    corsa = (re.search(r"corsa=(\S+)", t) or [None, None])[1]
    down = (re.search(r"down_ternario=(\d)", t) or [None, None])[1]
    if down != "1":
        raise RuntimeError(f"{p}: down_ternario={down} — il pannello valuta solo corse col down ternario (Thor-T)")
    return {"marca": t, "corsa": corsa}


def carica_codici(cartella: Path, mmap: bool = False) -> dict:
    """{L: {"Dg","Qg","Du","Qu","Dd","Qd": np.ndarray}} + controlli di forma e di valori."""
    codici = {}
    for L in FINESTRA:
        leggi_marca(cartella, L)
        c = {k: np.load(cartella / f"{k}_{L:02d}.npy", mmap_mode="r" if mmap else None) for k in NOMI_FAMIGLIE}
        for fam in ("g", "u", "d"):
            D, Q = c[f"D{fam}"], c[f"Q{fam}"]
            if D.dtype != np.float32 or Q.dtype != np.int8 or D.shape[:2] != Q.shape[:2] or Q.shape[-1] % D.shape[-1]:
                raise RuntimeError(f"{cartella} strato {L} famiglia {fam}: forme/dtype incoerenti "
                                   f"D{D.shape}/{D.dtype} Q{Q.shape}/{Q.dtype}")
            if not mmap and (int(Q.min()) < -1 or int(Q.max()) > 1):
                raise RuntimeError(f"{cartella} strato {L} famiglia {fam}: Q fuori da {{-1,0,1}}")
        codici[L] = c
    return codici


def _ricostruisci(D, Q, device, scale_fp16: bool = False) -> torch.Tensor:
    """W = D (x) Q per gruppo: D (E, out, ng) float32, Q (E, out, n_in) int8 -> (E, out, n_in) float32.
    Identico a forward_duro (s * codice duro): s = exp(log_s) e' proprio D."""
    Dt = torch.as_tensor(np.asarray(D), dtype=torch.float32).to(device)
    if scale_fp16:
        Dt = Dt.half().float()
    Qt = torch.from_numpy(np.ascontiguousarray(Q)).to(device)
    E, out, n_in = Qt.shape
    ng = Dt.shape[-1]
    g = n_in // ng
    W = Qt.reshape(E, out, ng, g).to(torch.float32)
    W.mul_(Dt.unsqueeze(-1))
    return W.reshape(E, out, n_in)


def pesi_da_codici(ctx, codici: dict, scale_fp16: bool = False) -> dict:
    """{L: (gate_up per il forward, down)} nel dtype del modulo, come `_ricostruisci_pesi_duri`:
    gate_up nello spazio RUOTATO -> `_gate_up_per_forward` (de-ruota, o no se ROTAZIONE_INGRESSO)."""
    pronti = {}
    with torch.no_grad():
        for L in FINESTRA:
            esp = ctx.correnti[L].mlp.experts
            dt = esp.gate_up_proj.dtype
            c = codici[L]
            gate = _ricostruisci(c["Dg"], c["Qg"], ctx.device, scale_fp16)
            up = _ricostruisci(c["Du"], c["Qu"], ctx.device, scale_fp16)
            gu_rot = torch.cat([gate, up], dim=1)
            del gate, up
            gu = V._gate_up_per_forward(gu_rot, ctx.S, dt)
            del gu_rot
            dw = _ricostruisci(c["Dd"], c["Qd"], ctx.device, scale_fp16).to(dt)
            if tuple(gu.shape) != tuple(esp.gate_up_proj.shape) or tuple(dw.shape) != tuple(esp.down_proj.shape):
                raise RuntimeError(f"strato {L}: forme ricostruite gu{tuple(gu.shape)} dw{tuple(dw.shape)} != "
                                   f"modulo {tuple(esp.gate_up_proj.shape)} {tuple(esp.down_proj.shape)}")
            pronti[L] = (gu, dw)
    return pronti


def codici_init_ls(ctx) -> dict:
    """Codici dell'init LS esattamente come li costruisce `ottimizza_finestra_JOINT`
    (THOR_INIT_DA=teacher, THOR_INIT_LS=1, down ternario): sorgente = pesi ORIGINALI."""
    cfg = ctx.cfg
    group_gu = _group_giocattolo(cfg.hidden_size, GRUPPO_GATEUP_REALE)
    group_dw = _group_giocattolo(cfg.moe_intermediate_size, GRUPPO_DOWN_REALE)
    codici = {}
    for L in FINESTRA:
        W_gu = ctx.puliti[L].mlp.experts.gate_up_proj
        W_dw = ctx.puliti[L].mlp.experts.down_proj
        t_gu = V._costruisci_ternari_layer(W_gu, ctx.S, group_gu, ruota=True).to(ctx.device)
        d_gu = reinit_ls_(t_gu, V._ruota_se_possibile(W_gu.detach().float().to(ctx.device), ctx.S))
        fam = V._esporta_gate_up(t_gu, W_gu.shape[1] // 2)
        del t_gu
        t_dw = V._costruisci_ternari_layer(W_dw, ctx.S, group_dw, ruota=False).to(ctx.device)
        d_dw = reinit_ls_(t_dw, W_dw.detach().float().to(ctx.device))
        Dd, Qd = V._estrai_famiglia(t_dw)
        del t_dw
        _svuota(ctx)
        log(f"  init LS strato {L}: disaccordo codice duro vs LS gu {d_gu['frazione_disaccordo_ls'] * 100:.4f}% · "
            f"dw {d_dw['frazione_disaccordo_ls'] * 100:.4f}%")
        codici[L] = {"Dg": fam["ffn_gate_exps"][0], "Qg": fam["ffn_gate_exps"][1],
                     "Du": fam["ffn_up_exps"][0], "Qu": fam["ffn_up_exps"][1], "Dd": Dd, "Qd": Qd}
    return codici


def _svuota(ctx) -> None:
    import gc
    gc.collect()
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════ 3. MODELLO (vero o giocattolo) ═══════════════════════════
@dataclass
class Contesto:
    cfg: object
    emb: torch.Tensor
    puliti: dict
    correnti: dict
    device: torch.device
    S: torch.Tensor
    seme_fn: object     # ids 3D -> (h0 {b: tensore}, e0 {b: tensore}), contratto di seme_embedding_quantizzato
    forme: dict


def contesto_vero() -> Contesto:
    cfg, emb, puliti, correnti = V.carica_finestra_corrente(layers=list(FINESTRA))
    S = torch.from_numpy(V._S1024_NP).to(emb.device)
    return Contesto(cfg, emb, puliti, correnti, emb.device, S,
                    lambda ids: V.seme_embedding_quantizzato(ids, emb), {})


def contesto_giocattolo(seme: int = 0) -> Contesto:
    """Strati VERI (Qwen3_5MoeDecoderLayer) a dimensioni giocattolo, come il test di
    valida_finestra_8_12: teacher con pesi piccoli, 'v20' = teacher + rumore su tutto
    (attenzione compresa, come la base di produzione). Embedding 'quantizzato' finto
    (errore relativo ~5%, la base vera ha 4,3%)."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    cfg = V._costruisci_config_giocattolo(len(FINESTRA))
    cfg._attn_implementation = "sdpa"
    puliti, correnti = {}, {}
    for L in FINESTRA:
        p = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        V._init_pesi_giocattolo(p, seed=100 + L + seme)
        c = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        g = torch.Generator().manual_seed(900 + L + seme)
        with torch.no_grad():
            for (_, pp), (_, pc) in zip(p.named_parameters(), c.named_parameters()):
                pc.copy_(pp + torch.randn(pp.shape, generator=g) * 0.15)
        puliti[L], correnti[L] = p.eval(), c.eval()
    g = torch.Generator().manual_seed(7 + seme)
    emb = torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.02
    emb_q = emb + torch.randn(emb.shape, generator=g) * 0.001

    def seme_fn(ids: torch.Tensor):
        h0, e0 = {}, {}
        for b in range(ids.shape[0]):
            h0[b] = torch.nn.functional.embedding(ids[b], emb).to(emb.dtype)
            e0[b] = (torch.nn.functional.embedding(ids[b], emb_q) - h0[b].float()).detach()
        return h0, e0

    dev = torch.device("cpu")
    return Contesto(cfg, emb, puliti, correnti, dev, torch.from_numpy(V._S1024_NP), seme_fn, {})


def forward_per_forma(ctx: Contesto, B: int, L: int):
    """Stessi posizionali/maschera di ottimizza_finestra_JOINT (dtype del dummy = dtype
    dell'embedding, lezione RoPE), uno per forma (B, L)."""
    if (B, L) not in ctx.forme:
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
        dev, cfg = ctx.device, ctx.cfg
        hs_dummy = torch.zeros(B, L, cfg.hidden_size, device=dev, dtype=ctx.emb.dtype)
        rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to(dev)
        pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)
        pos_emb = rot(hs_dummy, pos)
        maschera = create_causal_mask(config=cfg, inputs_embeds=hs_dummy, attention_mask=None,
                                      past_key_values=None, position_ids=pos)

        def _fwd(strato, h, _pe=pos_emb, _m=maschera, _p=pos):
            o = strato(hidden_states=h, position_embeddings=_pe, attention_mask=_m, position_ids=_p)
            return o[0] if isinstance(o, tuple) else o
        ctx.forme[(B, L)] = _fwd
    return ctx.forme[(B, L)]


def prepara_semi(ctx: Contesto, pannello: dict) -> dict:
    """Per dominio: h0/e0 (seme della finestra 0) e teacher precalcolato [h0, h1, h2] per batch."""
    semi = {}
    with torch.no_grad():
        for dom in DOMINI:
            ids = pannello["domini"][dom]
            h0, e0 = ctx.seme_fn(ids)
            fwd = forward_per_forma(ctx, ids.shape[1], ids.shape[2])
            teacher_fn = [V._stratofn_reale(ctx.puliti[L], fwd) for L in FINESTRA]
            cache = {b: precalcola_teacher(teacher_fn, h0[b]) for b in range(ids.shape[0])}
            semi[dom] = {"h0": h0, "e0": e0, "teacher": cache, "teacher_fn": teacher_fn, "fwd": fwd}
    return semi


# ═══════════════════════════ 4. VALUTAZIONE ═══════════════════════════
def _misure(r: dict, fasce: "tuple | None") -> dict:
    """Dalla traccia e_l di errore_lemma1_finestra: loss (= quella di JOINT), per strato,
    per sequenza, per fascia di posizione, quota dell'errore nell'1% dei token peggiori."""
    per_strato, per_seq, per_fascia = [], None, None
    ultimo = None
    for e in r["e_l"][1:]:
        sq = e.float().pow(2).mean(-1)                      # (B, T): media sui canali
        per_strato.append(sq.mean().item())
        s = sq.mean(-1)
        per_seq = s if per_seq is None else per_seq + s
        if fasce:
            f = torch.stack([sq[:, a:b].mean(-1) for a, b in fasce], dim=-1)   # (B, n_fasce)
            per_fascia = f if per_fascia is None else per_fascia + f
        ultimo = e
    norme = ultimo.float().pow(2).sum(-1).reshape(-1)
    k = max(1, int(round(0.01 * norme.numel())))
    top1 = (norme.topk(k).values.sum() / norme.sum().clamp_min(1e-30)).item()
    loss = r["loss"].item()
    if abs(loss - sum(per_strato)) > 1e-4 * max(abs(loss), 1e-30):
        raise RuntimeError(f"loss {loss} != somma per strato {sum(per_strato)}: misura incoerente")
    out = {"loss": loss, "per_strato": per_strato, "per_seq": per_seq.tolist(), "quota_top1pct_token": top1}
    if fasce:
        out["per_seq_fascia"] = per_fascia.tolist()
    return out


def valuta(ctx: Contesto, pannello: dict, semi: dict, pesi: "dict | None", routing: str = "congelato") -> dict:
    """Loss DURA per ogni batch del pannello. pesi None = esperti v20 (router vero)."""
    ris = {}
    chunk = list(FINESTRA)
    with torch.no_grad():
        for dom in DOMINI:
            ids = pannello["domini"][dom]
            s = semi[dom]
            fwd = s["fwd"]
            fasce = pannello["fasce"] if dom == "lunghe" else None
            ris[dom] = []
            for b in range(ids.shape[0]):
                h0, e0 = s["h0"][b], s["e0"][b]
                if pesi is None:
                    fq = [V._stratofn_reale(ctx.correnti[L], fwd) for L in chunk]
                else:
                    if routing == "congelato":
                        hq = h0 if e0 is None else h0 + e0.to(h0.dtype)     # = hq_per_b di JOINT
                        cong = V._congela_routing_chunk(chunk, ctx.correnti, hq, fwd, ternari_gu={},
                                                        ternari_dw=None, S=ctx.S, pesi_pronti=pesi)
                    elif routing == "vivo":
                        cong = {L: (None, None) for L in chunk}
                    else:
                        raise ValueError(f"routing {routing!r}: attesi 'congelato' o 'vivo'")
                    fq = [V._stratofn_quant_reale(ctx.correnti[L], None, None, ctx.S, *cong[L],
                                                  V.NUCLEO_BETA_MAX, True, fwd, pesi_pronti=pesi[L])
                          for L in chunk]
                r = errore_lemma1_finestra(s["teacher_fn"], fq, h0, e0, h_l_teacher_cache=s["teacher"][b],
                                           diagnostica=False)
                ris[dom].append(_misure(r, fasce))
                del r, fq
    return ris


# ═══════════════════════════ 5. STATISTICHE E USCITE ═══════════════════════════
def _media_es(x) -> "tuple[float, float]":
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    es = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")
    return float(x.mean()), es


def riassumi(ris: dict, fasce: tuple) -> dict:
    """Rapporti appaiati per batch (contro v20 e contro K2), rapporto delle medie, fasce."""
    som = {}
    base = ris["v20"]
    k2 = ris.get("K2")
    for nome, per_dom in ris.items():
        som[nome] = {}
        for dom in DOMINI:
            loss = np.array([m["loss"] for m in per_dom[dom]])
            v = np.array([m["loss"] for m in base[dom]])
            r_m, r_es = _media_es(loss / v)
            voce = {"n_batch": int(loss.size), "loss_media": float(loss.mean()),
                    "rapporto_appaiato_vs_v20": r_m, "es": r_es,
                    "rapporto_delle_medie_vs_v20": float(loss.mean() / v.mean()),
                    "loss_media_per_strato": np.mean([m["per_strato"] for m in per_dom[dom]], axis=0).tolist(),
                    "quota_top1pct_token": float(np.mean([m["quota_top1pct_token"] for m in per_dom[dom]]))}
            if k2 is not None:
                kk = np.array([m["loss"] for m in k2[dom]])
                voce["rapporto_appaiato_vs_K2"], voce["es_vs_K2"] = _media_es(loss / kk)
            if dom == "lunghe":
                f = np.array([sq for m in per_dom[dom] for sq in m["per_seq_fascia"]])     # (n_seq, n_fasce)
                fv = np.array([sq for m in base[dom] for sq in m["per_seq_fascia"]])
                rf = f / fv
                voce["fasce"] = [{"fascia": list(fa), "loss_media": float(f[:, i].mean()),
                                  "rapporto_appaiato_vs_v20": _media_es(rf[:, i])[0],
                                  "es": _media_es(rf[:, i])[1]} for i, fa in enumerate(fasce)]
                pend = rf[:, -1] / rf[:, 0]
                p_m, p_es = _media_es(pend)
                voce["pendenza_ultima_su_prima"] = {"media": p_m, "es": p_es, "n_seq": int(pend.size)}
            som[nome][dom] = voce
    return som


def segnali_A1(som: dict) -> dict:
    """Le regole di decisione di A1 (PIANO sez. 2, scritte prima dei numeri) applicate
    MECCANICAMENTE. Non decidono: le legge il coordinatore."""
    def R(n, d):
        return som[n][d]["rapporto_appaiato_vs_v20"]
    candidati = [n for n in ("Q1", "Q2", "Q3", "Q4") if n in som]
    out = {"nota": "applicazione meccanica delle regole di A1; «oltre 2σ» = distanza da 1 oltre due errori standard"}
    if candidati:
        ref = min(candidati, key=lambda n: R(n, "math_code"))
        out["R_ref_su_math_code"] = {"ricetta": ref, "valore": R(ref, "math_code"),
                                     "nota": "il PIANO sceglie fra Q1, Q2, Q3; qui e' incluso Q4 (ricetta della forgia)"}
        for n in candidati:
            p = som[n]["lunghe"]["pendenza_ultima_su_prima"]
            out[n] = {
                "miscela_domini_scatta": bool(R(n, "wiki") - R(n, "math_code") > 0.15
                                              or R(n, "edu") - R(n, "math_code") > 0.15),
                "R_wiki_meno_R_math": R(n, "wiki") - R(n, "math_code"),
                "R_edu_meno_R_math": R(n, "edu") - R(n, "math_code"),
                "addestramento_2048_scatta": bool(p["media"] >= 1.15 and (p["media"] - 1) > 2 * (p["es"] or 0)),
                "pendenza_fasce": p,
            }
            if "K2" in som:
                out[n]["non_peggio_di_K2_pan"] = {d: bool(som[n][d]["rapporto_appaiato_vs_K2"] <= 1.0)
                                                  for d in ("math_code", "wiki", "edu")}
    criterio = ("math_code", "wiki", "edu") + DOMINI_NUOVI
    out["criterio_tutti_sotto_1"] = {
        n: {"R": {d: som[n][d]["rapporto_appaiato_vs_v20"] for d in criterio},
            "tutti_sotto_1": bool(all(som[n][d]["rapporto_appaiato_vs_v20"] < 1 for d in criterio)),
            "margine_minimo": 1 - max(som[n][d]["rapporto_appaiato_vs_v20"] for d in criterio),
            "dominio_peggiore": max(criterio, key=lambda d: som[n][d]["rapporto_appaiato_vs_v20"])}
        for n in som if n != "v20"}
    return out


def confronto_storico(som: dict) -> dict:
    out = {}
    for nome, (rif, fonte) in RIFERIMENTI_STORICO.items():
        if nome not in som:
            continue
        mio = som[nome]["storico"]["loss_media"]
        dev = mio / rif - 1
        out[nome] = {"pannello": mio, "log": rif, "scarto_relativo": dev, "fonte": fonte,
                     "combacia": bool(abs(dev) <= TOLLERANZA_STORICO)}
    return out


def _f(x, cifre=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{cifre}f}".replace(".", ",")


def _e(x) -> str:
    return "—" if x is None or math.isnan(x) else f"{x:.4e}".replace(".", ",")


def markdown(rap: dict) -> str:
    som, ordine, fasce = rap["sommario"], rap["ordine_ricette"], rap["fasce"]
    righe = [f"# Pannello held-out allargato — finestra [0,1] — {rap['id']}", "",
             f"{rap['quando']} · routing `{rap['routing']}` · scale {'fp16' if rap['scale_fp16'] else 'fp32'} · "
             f"{'⚠️ PROVA CPU a scala minuscola (strati giocattolo): i numeri NON valgono per Thor' if rap['prova_cpu'] else 'Thor vero, GPU'}",
             "", "Rapporti APPAIATI per batch contro gli esperti v20 (media ± errore standard). "
             "Sotto 1 = meglio della v20. `storico` = l'held-out di sempre (1 batch, niente errore standard).", "",
             "| ricetta | storico | math_code | wiki | edu | " + " | ".join(f"[{a},{b})" for a, b in fasce)
             + " | pendenza ultima/prima |",
             "|---|---|---|---|---|" + "---|" * len(fasce) + "---|"]
    for n in ordine:
        s = som[n]
        cella = [f"{_f(s['storico']['rapporto_appaiato_vs_v20'])}"]
        for d in ("math_code", "wiki", "edu"):
            cella.append(f"{_f(s[d]['rapporto_appaiato_vs_v20'])} ± {_f(s[d]['es'])}")
        for fa in s["lunghe"]["fasce"]:
            cella.append(f"{_f(fa['rapporto_appaiato_vs_v20'])} ± {_f(fa['es'])}")
        p = s["lunghe"]["pendenza_ultima_su_prima"]
        cella.append(f"{_f(p['media'])} ± {_f(p['es'])}")
        righe.append(f"| {n} | " + " | ".join(cella) + " |")
    if DOMINI_NUOVI:
        tutti = ("math_code", "wiki", "edu") + DOMINI_NUOVI
        righe += ["", "## Domini nuovi (italiano, chat, ragionamento, strumenti): rapporto appaiato contro v20", "",
                  "| ricetta | " + " | ".join(DOMINI_NUOVI) + " | massimo su math_code, wiki, edu e nuovi |",
                  "|---|" + "---|" * (len(DOMINI_NUOVI) + 1)]
        for n in ordine:
            s = som[n]
            cella = [f"{_f(s[d]['rapporto_appaiato_vs_v20'])} ± {_f(s[d]['es'])}" for d in DOMINI_NUOVI]
            peggiore = max(tutti, key=lambda d: s[d]["rapporto_appaiato_vs_v20"])
            cella.append(f"{_f(s[peggiore]['rapporto_appaiato_vs_v20'])} ({peggiore})")
            righe.append(f"| {n} | " + " | ".join(cella) + " |")
    righe += ["", "## Loss dura media (assoluta)", "",
              "| ricetta | " + " | ".join(DOMINI) + " |", "|---|" + "---|" * len(DOMINI)]
    for n in ordine:
        righe.append(f"| {n} | " + " | ".join(_e(som[n][d]["loss_media"]) for d in DOMINI) + " |")
    if "K2" in som:
        righe += ["", "## Contro K2 (K2_pan): rapporto appaiato per batch", "",
                  "| ricetta | math_code | wiki | edu | lunghe |", "|---|---|---|---|---|"]
        for n in ordine:
            righe.append(f"| {n} | " + " | ".join(
                f"{_f(som[n][d]['rapporto_appaiato_vs_K2'])} ± {_f(som[n][d]['es_vs_K2'])}"
                for d in ("math_code", "wiki", "edu", "lunghe")) + " |")
    righe += ["", "## Held-out storico: pannello contro log (tolleranza "
              f"{_f(TOLLERANZA_STORICO * 100, 1)}%)", "",
              "| ricetta | pannello | log | scarto | combacia | fonte |", "|---|---|---|---|---|---|"]
    for n, c in rap["confronto_storico"].items():
        righe.append(f"| {n} | {_e(c['pannello'])} | {_e(c['log'])} | {_f(c['scarto_relativo'] * 100, 3)}% | "
                     f"{'✅' if c['combacia'] else '⛔'} | {c['fonte']} |")
    righe += ["", "## Segnali delle regole di A1 (meccanici: decide il coordinatore)", "", "```json",
              json.dumps(rap["segnali_A1"], ensure_ascii=False, indent=2), "```", "",
              "## Provenienza", "", f"- pannello: `{json.dumps(rap['pannello']['forme'])}`, "
              f"math_code dagli esempi {rap['pannello']['math_code']['esempi_usati']} (seme {SEME_CALIB}), "
              f"edu documenti {rap['pannello']['edu']['documenti_usati']}",
              f"- tempi (s): `{json.dumps(rap['tempi'])}` · VRAM picco: {rap.get('vram_picco_gib')} GiB",
              f"- ambiente: `{json.dumps(rap['ambiente'])}`",
              "- ricette: " + "; ".join(f"{k}: `{v.get('cartella')}` ({v.get('corsa')})" for k, v in rap["ricette"].items()),
              "", "⚠️ La validation di wikitext usata per scegliere la ricetta e' una taratura lieve sul dominio "
              "della metrica (PIANO leva #1): va dichiarata. La PPL di verdetto resta sul solo split test."]
    return "\n".join(righe) + "\n"


def scrivi_uscite(rap: dict, cartella: Path) -> "tuple[Path, Path]":
    cartella.mkdir(parents=True, exist_ok=True)
    base = cartella / f"pannello_f01_{rap['id']}"
    pj, pm = base.with_suffix(".json"), base.with_suffix(".md")
    pj.write_text(json.dumps(rap, ensure_ascii=False, indent=1), encoding="utf-8")
    pm.write_text(markdown(rap), encoding="utf-8")
    log(f"uscite: {pm} · {pj}")
    return pm, pj


def _ambiente() -> dict:
    chiavi = ("THOR_ROTAZIONE_INGRESSO", "THOR_NUCLEO_BF16", "THOR_ESPERTI_THOR", "THOR_GROUPED_BMM",
              "THOR_CACHE_STRATI", "THOR_HF_DIR", "THOR_SCHUR_BASE", "TORCHDYNAMO_DISABLE")
    amb = {k: os.environ.get(k) for k in chiavi}
    try:
        amb["git_tony_forgia"] = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent),
                                                 "rev-parse", "--short", "HEAD"], capture_output=True,
                                                text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001 — provenienza, non blocca
        amb["git_tony_forgia"] = f"non letto: {e}"
    return amb


def valuta_tutte(ctx: Contesto, pannello: dict, ricette: dict, routing: str, scale_fp16: bool,
                 codici_pronti: "dict | None" = None) -> "tuple[dict, dict, dict]":
    """v20, init LS e ogni ricetta sullo stesso pannello. `codici_pronti` (solo prova):
    {nome: codici gia' in memoria} al posto delle cartelle."""
    tempi, info = {}, {}
    t = time.time()
    semi = prepara_semi(ctx, pannello)
    tempi["semi_e_teacher"] = round(time.time() - t, 1)
    ris = {}
    t = time.time()
    ris["v20"] = valuta(ctx, pannello, semi, None, routing)
    tempi["v20"] = round(time.time() - t, 1)
    info["v20"] = {"cartella": str(V.GGUF_BASE), "corsa": "base di produzione"}
    log(f"  v20 valutata in {tempi['v20']} s")

    t = time.time()
    cod = codici_init_ls(ctx)
    pesi = pesi_da_codici(ctx, cod, scale_fp16)
    del cod
    ris["init_LS"] = valuta(ctx, pannello, semi, pesi, routing)
    del pesi
    _svuota(ctx)
    tempi["init_LS"] = round(time.time() - t, 1)
    info["init_LS"] = {"cartella": "calcolato qui dai pesi originali", "corsa": "init LS come JOINT"}
    log(f"  init LS valutata in {tempi['init_LS']} s")

    for nome, cartella in ricette.items():
        t = time.time()
        if codici_pronti and nome in codici_pronti:
            cod = codici_pronti[nome]
        else:
            cod = carica_codici(Path(cartella))
        info[nome] = {"cartella": str(cartella), "corsa": leggi_marca(Path(cartella), FINESTRA[0])["corsa"],
                      "marche": {L: leggi_marca(Path(cartella), L)["marca"] for L in FINESTRA}}
        pesi = pesi_da_codici(ctx, cod, scale_fp16)
        del cod
        ris[nome] = valuta(ctx, pannello, semi, pesi, routing)
        del pesi
        _svuota(ctx)
        tempi[nome] = round(time.time() - t, 1)
        log(f"  {nome} valutata in {tempi[nome]} s · math_code "
            f"{np.mean([m['loss'] for m in ris[nome]['math_code']]):.4e}")
    del semi
    _svuota(ctx)
    return ris, tempi, info


def rapporto(ris: dict, tempi: dict, info: dict, pannello: dict, routing: str, scale_fp16: bool,
             prova: bool, id_corsa: str) -> dict:
    som = riassumi(ris, pannello["fasce"])
    return {"id": id_corsa, "quando": time.strftime("%Y-%m-%d %H:%M:%S"), "prova_cpu": prova,
            "finestra": list(FINESTRA), "routing": routing, "scale_fp16": scale_fp16,
            "ordine_ricette": list(ris.keys()), "fasce": [list(f) for f in pannello["fasce"]],
            # in prova gli strati sono giocattolo: il confronto coi log di Thor non ha senso
            # (lo sostituisce il controllo C1 contro l'held-out misurato dentro JOINT)
            "sommario": som, "confronto_storico": {} if prova else confronto_storico(som),
            "segnali_A1": segnali_A1(som),
            "pannello": pannello["meta"], "ricette": info, "tempi": tempi, "ambiente": _ambiente(),
            "grezzi": ris}


# ═══════════════════════════ 6. ESECUZIONE VERA (GPU) ═══════════════════════════
def precondizioni_vere(ricette: dict) -> None:
    guasti = []
    for nome, p in (("THOR_HF_DIR", V.SRC), ("THOR_SCHUR_BASE", V.GGUF_BASE), ("THOR_CORPUS", V.CORPUS),
                    ("wikitext", FILE_WIKI), ("fineweb-edu", FILE_EDU)):
        if not Path(p).exists():
            guasti.append(f"{nome}={p} non esiste (dati: scarica_dati_pannello.py)")
    for nome, cart in ricette.items():
        for L in FINESTRA:
            try:
                leggi_marca(Path(cart), L)
            except RuntimeError as e:
                guasti.append(f"ricetta {nome}: {e}")
    if not torch.cuda.is_available():
        guasti.append("nessuna GPU visibile")
    if guasti:
        for g in guasti:
            log(f"⛔ PRECONDIZIONE: {g}")
        raise SystemExit(4)


def esegui_vero(args, ricette: dict) -> int:
    precondizioni_vere(ricette)
    V.verifica_gpu_libera(vram_max_gib=args.vram_max_gib, gpu_pct_max=args.gpu_pct_max)
    t0 = time.time()
    tok = tokenizzatore()
    pannello = pannello_vero(tok, ricostruisci=args.ricostruisci_pannello)
    torch.cuda.reset_peak_memory_stats()
    ctx = contesto_vero()
    try:
        ris, tempi, info = valuta_tutte(ctx, pannello, ricette, args.routing, args.scale_fp16)
    finally:
        V.libera_finestra(ctx.puliti, ctx.correnti, ctx.emb)
    tempi["totale"] = round(time.time() - t0, 1)
    rap = rapporto(ris, tempi, info, pannello, args.routing, args.scale_fp16, False, args.id)
    rap["vram_picco_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    scrivi_uscite(rap, DIR_USCITA)
    for n, c in rap["confronto_storico"].items():
        if not c["combacia"]:
            log(f"⚠️ storico {n}: pannello {c['pannello']:.4e} contro log {c['log']:.4e} "
                f"({c['scarto_relativo'] * 100:.2f}%) — controllare prima di fidarsi")
    return 0


# ═══════════════════════════ 7. PROVA SU CPU ═══════════════════════════
def controllo_codici_reali(ricette: dict, n_esperti: int = 2) -> dict:
    """Sui codici VERI salvati (mmap, solo `n_esperti` esperti): la convenzione di
    ricostruzione (ordine gate/up, de-rotazione R1024, gruppi) e' giusta se l'errore
    relativo contro i pesi ORIGINALI e' dell'ordine dell'init LS (~0,43, METODO 4.1)
    e diventa ~1,4 (scorrelato) saltando la de-rotazione. Stesso controllo sull'init
    LS calcolato qui (atteso ~0,43 gate/up, ~0,44 down)."""
    from safetensors import safe_open
    from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso
    idx = json.load(open(V.SRC / "model.safetensors.index.json"))["weight_map"]
    S = torch.from_numpy(V._S1024_NP)
    n = n_esperti

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    esiti = {}
    for L in FINESTRA:
        pesi = {}
        for fam in ("gate_up_proj", "down_proj"):
            k = f"model.language_model.layers.{L}.mlp.experts.{fam}"
            with safe_open(str(V.SRC / idx[k]), "pt") as sf:
                pesi[fam] = sf.get_slice(k)[0:n].float()
        W_gu, W_dw = pesi["gate_up_proj"], pesi["down_proj"]
        meta = W_gu.shape[1] // 2
        # init LS su questi esperti, stessa via di JOINT
        t_gu = TernarioDiffBatchFuso(V._ruota_se_possibile(W_gu, S), group=GRUPPO_GATEUP_REALE)
        reinit_ls_(t_gu, V._ruota_se_possibile(W_gu, S))
        Dg, Qg = V._estrai_famiglia(t_gu)
        t_dw = TernarioDiffBatchFuso(W_dw, group=GRUPPO_DOWN_REALE)
        reinit_ls_(t_dw, W_dw)
        Dd, Qd = V._estrai_famiglia(t_dw)
        gu_init = V._deruota_se_possibile(_ricostruisci(Dg, Qg, "cpu"), S)
        esiti[f"L{L}.init_LS"] = {"gu": rel(gu_init, W_gu), "dw": rel(_ricostruisci(Dd, Qd, "cpu"), W_dw)}
        for nome, cart in ricette.items():
            c = {k: np.load(Path(cart) / f"{k}_{L:02d}.npy", mmap_mode="r") for k in NOMI_FAMIGLIE}
            gate = _ricostruisci(c["Dg"][:n], c["Qg"][:n], "cpu")
            up = _ricostruisci(c["Du"][:n], c["Qu"][:n], "cpu")
            gu_rot = torch.cat([gate, up], dim=1)
            gu = V._deruota_se_possibile(gu_rot, S)
            gu_scambiati = V._deruota_se_possibile(torch.cat([up, gate], dim=1), S)
            dw = _ricostruisci(c["Dd"][:n], c["Qd"][:n], "cpu")
            e = {"gu": rel(gu, W_gu), "gu_senza_deruotare": rel(gu_rot, W_gu),
                 "gu_gate_up_scambiati": rel(gu_scambiati, W_gu), "dw": rel(dw, W_dw),
                 "forme": {k: list(v.shape) for k, v in c.items()}}
            e["convenzione_ok"] = bool(e["gu"] < 0.8 and e["gu_senza_deruotare"] > 1.2
                                       and e["gu_gate_up_scambiati"] > 1.2 and e["dw"] < 0.8)
            esiti[f"L{L}.{nome}"] = e
        del W_gu, W_dw
    for k, v in esiti.items():
        log(f"  codici reali {k}: " + " · ".join(f"{a}={b:.3f}" for a, b in v.items() if isinstance(b, float))
            + (f" · convenzione {'✅' if v['convenzione_ok'] else '⛔'}" if "convenzione_ok" in v else ""))
    return esiti


def addestra_ricette_giocattolo(ctx: Contesto, ids_ho: torch.Tensor, cartella: Path) -> dict:
    """Due ricette giocattolo prodotte da ottimizza_finestra_JOINT VERA (su copie degli
    strati 'v20'), scritte con scrivi_codici_su_disco VERA. Ritorna {nome: (cartella,
    held-out {v20, init, fine} misurato da JOINT sulla batch `ids_ho`)}."""
    out = {}
    g = torch.Generator().manual_seed(11)
    tr = torch.randint(0, ctx.cfg.vocab_size, (3, ids_ho.shape[1], ids_ho.shape[2]), generator=g)
    ids = torch.cat([tr, ids_ho], dim=0)
    h0, e0 = ctx.seme_fn(ids)
    salva = (V.OUT_DIR, V.ID_CORSA)
    try:
        for nome, lr, passi in (("T1_lr005_20", 0.05, 20), ("T2_lr001_20", 0.01, 20)):
            diag: list = []
            correnti = copy.deepcopy(ctx.correnti)
            risultato = V.ottimizza_finestra_JOINT(
                correnti, ids, ctx.cfg, n_passi=passi, strati_teacher=ctx.puliti, emb=ctx.emb, lr=lr,
                h0_esterno_per_b=h0, e0_esterno_per_b=e0, n_batch_addestramento=tr.shape[0],
                raccogli_diagnostica=diag)
            V.OUT_DIR, V.ID_CORSA = cartella / nome, f"prova-{nome}"
            V.scrivi_codici_su_disco(risultato)
            out[nome] = (cartella / nome, diag[0]["heldout"])
    finally:
        V.OUT_DIR, V.ID_CORSA = salva
    return out


def prova_cpu(args) -> int:
    torch.set_num_threads(2)
    assert not torch.cuda.is_available(), "la prova deve girare SENZA GPU"
    t0 = time.time()
    esiti: dict = {}
    scratch = Path(os.getenv("THOR_PANNELLO_SCRATCH", str(DIR_USCITA / "prova_cpu" / "codici_giocattolo")))

    log("── A. pannello a scala minuscola dai dati VERI (tokenizzatore di Thor) ──")
    tok = tokenizzatore()
    pannello = costruisci_pannello(tok, SCALA_PROVA)
    m = pannello["meta"]["math_code"]
    esiti["A_replica_impacchettamento"] = all(m["replica_verificata_contro_carica_batch_calibrazione"].values())
    esiti["A_disgiunzione_indici"] = bool(
        m["primo_esempio_del_pannello"] >= max(d["addestramento_esempi"][1] for d in m["configurazioni_di_addestramento"]))
    esiti["A_forme"] = pannello["meta"]["forme"] == {
        "storico": [1, 2, 32], "math_code": [1, 2, 32], "wiki": [1, 2, 32], "edu": [1, 2, 32], "lunghe": [1, 2, 64]}

    pannello = aggiungi_nuovi_domini(pannello, SCALA_PROVA)
    esiti["A_nuovi_domini"] = bool(DOMINI_NUOVI) and all(
        list(pannello["domini"][d].shape) == [1, SCALA_PROVA.seq_per_batch, SCALA_PROVA.l_seq] for d in DOMINI_NUOVI)
    log("── B. codici VERI salvati: convenzione di ricostruzione su 2 esperti per strato ──")
    cr = controllo_codici_reali(RICETTE, n_esperti=2)
    esiti["B_convenzione_codici_reali"] = all(v["convenzione_ok"] for k, v in cr.items() if "convenzione_ok" in v)

    log("── C. catena completa su strati GIOCATTOLO (stessa funzione della corsa vera) ──")
    ctx = contesto_giocattolo()
    pann_t = {"domini": {k: v % ctx.cfg.vocab_size for k, v in pannello["domini"].items()},
              "fasce": pannello["fasce"], "meta": pannello["meta"]}
    ricette_t = addestra_ricette_giocattolo(ctx, pann_t["domini"]["storico"], scratch)
    ris, tempi, info = valuta_tutte(ctx, pann_t, {n: c for n, (c, _) in ricette_t.items()}, "congelato", False)
    # C1: lo `storico` del pannello riproduce l'held-out misurato DENTRO JOINT
    rif = next(iter(ricette_t.values()))[1]
    conf = {"v20": (ris["v20"]["storico"][0]["loss"], rif["v20"]),
            "init_LS": (ris["init_LS"]["storico"][0]["loss"], rif["init"])}
    for n, (_, h) in ricette_t.items():
        conf[n] = (ris[n]["storico"][0]["loss"], h["fine"])
    scarti = {n: abs(a / b - 1) for n, (a, b) in conf.items()}
    for n, (a, b) in conf.items():
        log(f"  storico giocattolo {n}: pannello {a:.6e} · JOINT {b:.6e} · scarto {scarti[n]:.2e}")
    esiti["C1_riproduce_heldout_di_JOINT"] = all(s <= 1e-5 for s in scarti.values())
    # C2: v20 contro se stessa = 1 esatto
    som = riassumi(ris, pann_t["fasce"])
    esiti["C2_v20_rapporto_1"] = all(som["v20"][d]["rapporto_appaiato_vs_v20"] == 1.0 for d in DOMINI)
    # C3: le fasce ricompongono la sequenza intera (media pesata per lunghezza)
    ok3 = True
    for nome in ris:
        for m_ in ris[nome]["lunghe"]:
            for s_, fs in zip(m_["per_seq"], m_["per_seq_fascia"]):
                T = pann_t["domini"]["lunghe"].shape[-1]
                ric = sum((b - a) * v for (a, b), v in zip(pann_t["fasce"], fs)) / T
                ok3 = ok3 and abs(ric / s_ - 1) < 1e-5
    esiti["C3_fasce_coerenti"] = ok3
    # C4: il percorso a routing vivo gira (valore solo informativo)
    semi = prepara_semi(ctx, pann_t)
    vivo = valuta(ctx, pann_t, semi, pesi_da_codici(ctx, carica_codici(next(iter(ricette_t.values()))[0])), "vivo")
    esiti["C4_routing_vivo_gira"] = bool(np.isfinite(vivo["math_code"][0]["loss"]))
    log(f"  routing vivo (informativo): math_code {vivo['math_code'][0]['loss']:.6e} contro congelato "
        f"{ris[next(iter(ricette_t))]['math_code'][0]['loss']:.6e}")
    # C5: uscite scritte
    tempi["totale"] = round(time.time() - t0, 1)
    rap = rapporto(ris, tempi, info, pann_t, "congelato", False, True, args.id)
    rap["esiti_prova"] = esiti
    rap["controllo_codici_reali"] = cr
    pm, pj = scrivi_uscite(rap, DIR_USCITA / "prova_cpu")
    esiti["C5_uscite_scritte"] = pm.exists() and pj.exists() and json.loads(pj.read_text())["id"] == args.id
    pj.write_text(json.dumps({**rap, "esiti_prova": esiti}, ensure_ascii=False, indent=1), encoding="utf-8")
    tutto = all(esiti.values())
    for k, v in esiti.items():
        log(f"  {'✅' if v else '⛔'} {k}")
    log(f"{'✅ PROVA CPU SUPERATA' if tutto else '⛔ PROVA CPU FALLITA'} in {time.time() - t0:.0f} s")
    return 0 if tutto else 1


# ═══════════════════════════════════ MAIN ═══════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prova-cpu", action="store_true", help="logica a scala minuscola, SOLO CPU")
    ap.add_argument("--solo-pannello", action="store_true", help="costruisce, verifica e salva il pannello (CPU)")
    ap.add_argument("--routing", choices=("congelato", "vivo"), default="congelato")
    ap.add_argument("--scale-fp16", action="store_true", help="arrotonda le scale a fp16 come nel GGUF")
    ap.add_argument("--ricette", default=",".join(RICETTE), help="sottoinsieme, es. K2,Q2,Q4")
    ap.add_argument("--extra", action="append", default=[], help="NOME=CARTELLA di altri codici della finestra [0,1]")
    ap.add_argument("--vram-max-gib", type=float, default=V.VRAM_MAX_GIB)
    ap.add_argument("--gpu-pct-max", type=int, default=V.GPU_PCT_MAX)
    ap.add_argument("--ricostruisci-pannello", action="store_true")
    ap.add_argument("--id", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()

    if args.prova_cpu:
        return prova_cpu(args)
    if args.solo_pannello:
        pannello_vero(tokenizzatore(), ricostruisci=args.ricostruisci_pannello)
        return 0
    ricette = {n: RICETTE[n] for n in args.ricette.split(",") if n}
    for voce in args.extra:
        nome, _, cart = voce.partition("=")
        ricette[nome] = Path(cart)
    return esegui_vero(args, ricette)


if __name__ == "__main__":
    sys.exit(main())
