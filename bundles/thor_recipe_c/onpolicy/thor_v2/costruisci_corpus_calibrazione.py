#!/usr/bin/env python3
"""Costruisce il CORPUS DI CALIBRAZIONE multi-dominio di Thor e le PARTI NUOVE del
pannello held-out (italiano, chat, ragionamento, strumenti). 24-25/9/2026, notte.

## Perche' esiste

La forgia migliora SOLO i domini presenti nella calibrazione (pannello A1: con la
sola chat math/code 0,53x la v20 su math/code ma 1,70x su Wikipedia e 1,43x su
FineWeb-Edu; con M2 = cot 2, edu 3, wiki 3 sotto 1 ovunque ma col margine stretto:
0,70 / 0,94 / 0,98). Thor sara' il cervello di MOGAVIS, assistente domestico
ITALIANO che parla tutto il giorno, pianifica, ragiona e chiama strumenti. Qui si
costruisce un corpus che rappresenta quell'uso, da fonti PUBBLICHE e ripetibili
(scaricate da `scarica_corpus_calibrazione.py`, revisioni HF fissate e sha256), e
il pannello che lo MISURA sui domini nuovi, disgiunto dalla calibrazione.

## Uscite

In `@@ROOT:models@@/gguf/tony-tern/calibrazione/`:
- `calibrazione_corpus.pt` — `{dominio: tensore (n, 1024) int64}`, una riga = una
  sequenza vista UNA volta. Domini: `wiki` (wikitext-2 TRAIN), `edu` (FineWeb-Edu
  dal documento 173), `cot` (tracce CoT math/code di Thor in formato chat, esempi
  0-8 e 279-385 dell'ordine col seme 1234 ESCLUSI), `it_web`, `it_wiki`, `it_chat`
  (senza ragionamento: Aya, OASST2, MFAQ, smoltalk2), `it_pensiero` (smoltalk2 con
  ragionamento in inglese, ReasonXL con ragionamento in italiano), `strumenti`
  (Nemotron in inglese, 2/3 senza ragionamento, e MASSIVE in italiano convertito in
  chiamate di funzione). Le sequenze di ogni dominio sono gia' nell'ordine d'uso:
  un prefisso qualunque ha la composizione per fonte dichiarata.
- `calibrazione_corpus.json` — manifesto: fonti (revisioni, sha256, licenze), quote,
  conteggi misurati (esempi, documenti, token, sequenze) per dominio e per fonte,
  sha256 di ogni tensore, etichetta di fonte per sequenza, seme, verifiche di
  disgiunzione, esempi decodificati.
- `LEGGIMI.md` — la stessa cosa, leggibile.

In `@@ROOT:models@@/gguf/tony-tern/pannello/` (file NUOVI; `pannello_ids.pt/json` non si
toccano): `pannello_nuovi_domini.pt` (`{"domini": {nome: (n_batch, 8, 1024)}, "meta"}`)
+ `.json`: `it_web` 16, `it_wiki` 8, `it_chat` 16, `it_pensiero` 8, `strumenti_en` 8,
`strumenti_it` 8 batch da 8x1024.

## Formato

Conversazioni col TEMPLATE CHAT VERO di Thor (`tokenizer.apply_chat_template`):
`reasoning_content` quando la fonte ha il ragionamento (vuoto altrimenti: il
template scrive `<think>\\n\\n</think>\\n\\n`, lo stesso formato che MOGAVIS usa in
produzione col ragionamento spento), strumenti passati come `tools=` (blocco
`<tools>` nel system) e chiamate come `tool_calls` con argomenti DIZIONARIO (il
template li scrive in XML `<function=...><parameter=...>`). Testo grezzo resta
grezzo (documenti uniti con `\\n\\n`). Impacchettamento come `_impacca`: esempi uno
dopo l'altro, taglio ogni 1024 token (un esempio puo' continuare nella sequenza
dopo). `cot`, `edu` e `wiki` sono tokenizzati ESATTAMENTE come `_batch_miscela` di
`valida_finestra_8_12.py` (stessi token al bit: lo prova la patch).

## Disgiunzione (verificata qui, esito nel manifesto)

1. partizione per hash del prompt normalizzato (soglia 0,30: sotto = pannello) su
   Aya, smoltalk2 (le varianti con e senza ragionamento condividono i prompt: la
   regola le tiene dallo stesso lato) e ReasonXL, con file diversi per i due lati;
   split ufficiali per OASST2 (train/validation), MFAQ (train/valid, siti
   verificati disgiunti), FineWeb-2 (train/test), MASSIVE (train contro test);
   Wikipedia IT: row group diversi, id verificati disgiunti; Nemotron: file diversi
   e pannello con SOLI strumenti assenti in calibrazione;
2. frasi MASSIVE normalizzate del pannello escluse dal train (dedup esatta: le frasi
   sono piu' corte di 32 token);
3. n-grammi di 32 token sul SOLO contenuto (testi di utente, assistente, ragionamento,
   argomenti delle chiamate, risposte degli strumenti tokenizzati separatamente; per il
   testo grezzo, le sequenze stesse) della calibrazione contro: pannello nuovo, pannello
   storico (`pannello_ids.pt`), `wiki.test.raw` (PPL di verdetto), `mmlu_test.bin`.
   Un documento/esempio di calibrazione con un n-gramma in comune viene TOLTO e il
   dominio ricostruito (fino a 3 giri); gli n-grammi delle sequenze intere (template e
   schemi compresi) si riportano a parte, come informazione.

## Uso (solo CPU, nice, al piu' 4 thread)

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \\
        nice -n 19 @@ROOT:venv@@/bin/python3 costruisci_corpus_calibrazione.py \\
        [--budget 1200] [--miscele "U=...;V=..."] [--soglia-pannello 0.30] [--k-massive 4]

Budget = sequenze da 1024 di una corsa (default 1200 = 150 batch x 8). Ogni dominio
riceve le sequenze che servono alla miscela piu' esigente fra quelle dichiarate
(default U, V, W della ricerca e M2 di oggi), cosi' ognuna gira sul budget senza
ripetere una sequenza.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

SRC = Path(os.getenv("THOR_HF_DIR", "@@ROOT:models@@/gguf/thor-src-hf"))
DIR_CAL = Path(os.getenv("THOR_CALIB_CORPUS_DIR", "@@ROOT:models@@/gguf/tony-tern/calibrazione"))
SORG = DIR_CAL / "sorgenti"
DIR_PAN = Path(os.getenv("THOR_PANNELLO_DATI", "@@ROOT:models@@/gguf/tony-tern/pannello"))
DIR_COT = Path(os.getenv("THOR_COT_DIR", "@@ROOT:forge@@/thor/cot_calib"))
FILE_WIKI_TEST = Path("@@ROOT:odino@@/collaudo/corpus/wiki.test.raw")
FILE_MMLU = Path("@@ROOT:forge@@/mmlu_test.bin")
OUT_PT = DIR_CAL / "calibrazione_corpus.pt"
OUT_JSON = DIR_CAL / "calibrazione_corpus.json"
OUT_PAN = DIR_PAN / "pannello_nuovi_domini.pt"

L = 1024
SEME = 1234
N_GRAM = 32
# CoT: stesse costanti del pannello (pannello_ids.json): held-out storico 0-8, pannello 279-385
COT_ESCLUSI = (range(0, 9), range(279, 386))
EDU_PRIMO_DOC = 173
EDU_FILE = "fineweb-edu.sample-10BT.000_00000.rg0-1.jsonl"
WIKI_FILE = "wikitext-2-raw-v1.train.raw"

# Miscele (sequenze per ciclo di 16 = 2 batch da 8), dalla ricerca (U, V, W) + M2 di oggi.
MISCELE_DEFAULT = ("U=wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2;"
                   "V=wiki:4,edu:3,cot:2,it_web:2,it_chat:3,strumenti:2;"
                   "W=wiki:2,edu:2,cot:2,it_web:3,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2;"
                   "M2=cot:2,edu:3,wiki:3")
DOMINI = ("wiki", "edu", "cot", "it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti")
# Composizione INTERNA dei domini a piu' fonti (pesi; la ricerca: 60/30/35/100, 40/35, 90/60).
# La fonte "riempitiva" (ultima) assorbe cio' che una fonte piccola non ha: e' scritto nel manifesto.
COMPOSIZIONE = {
    "it_chat": (("aya", 60), ("oasst2", 30), ("mfaq", 35), ("smoltalk_nothink", 100)),
    "it_pensiero": (("smoltalk_think", 40), ("reasonxl", 35)),
    "strumenti": (("massive", 60), ("nemotron", 90)),
}
# Pannello nuovo: (dominio, n_batch, composizione in sequenze; l'ultima fonte riempie)
PANNELLO = {
    "it_web": (16, (("fineweb2_test", None),)),
    "it_wiki": (8, (("wiki_it_rg1", None),)),
    "it_chat": (16, (("aya", 26), ("oasst2", 8), ("mfaq", 30), ("smoltalk_nothink", None))),
    "it_pensiero": (8, (("smoltalk_think", 32), ("reasonxl", None))),
    "strumenti_en": (8, (("nemotron", None),)),
    "strumenti_it": (8, (("massive_agents", None),)),
}
SEQ_PER_BATCH_PANNELLO = 8
NEMOTRON_MAX_TOKEN = 2048
NEMOTRON_OGNI_PENSIERO = 3            # un esempio su 3 col ragionamento (2/3 senza, come in produzione)
MFAQ_COPPIE_PER_CONV = 6
MFAQ_PAGINE_PER_SITO = 3
WIKI_IT_MIN_TOKEN = 256
# Le voci di Wikipedia in testa al file sono lunghe (senza taglio 75 sequenze = 18 voci): si tiene
# l'inizio di ogni voce, a paragrafi interi, fino a ~4000 caratteri (~1000 token), per avere piu' voci.
WIKI_IT_MAX_CAR = 4000
DISTRATTORI_MASSIVE = 3
MIN_CAR_IDENTICI = 24                 # messaggi normalizzati piu' corti non contano come doppioni


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def sha_tensore(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def jsonl(p: Path) -> list:
    return [json.loads(r) for r in p.open(encoding="utf-8") if r.strip()]


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").lower()
    return " ".join(re.sub(r"[^\w]+", " ", t).split())


def secchio(t: str) -> float:
    """Posizione in [0,1) del prompt normalizzato: < soglia = pannello, >= soglia = calibrazione."""
    return int(hashlib.sha256(norm(t).encode("utf-8")).hexdigest()[:16], 16) / 16**16


# ═══════════════════════════════ esempi ═══════════════════════════════
@dataclass
class Esempio:
    fonte: str
    id: str
    testo: str                       # reso (chat) o grezzo (documento + separatore)
    contenuti: list                  # testi del SOLO contenuto, per gli n-grammi
    prompt: str = ""                 # primo messaggio dell'utente (partizione per hash)
    extra: dict = field(default_factory=dict)
    ids: "list | None" = None        # token, calcolati quando servono


class Tok:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(SRC))
        self.n_chiamate = 0

    def ids(self, testi: list) -> list:
        self.n_chiamate += 1
        if not testi:
            return []
        return self.tok(list(testi), add_special_tokens=False)["input_ids"]

    def uno(self, testo: str) -> list:
        return self.tok(testo, add_special_tokens=False)["input_ids"]

    def chat(self, messaggi: list, tools: "list | None" = None) -> str:
        kw = {"tokenize": False}
        if tools:
            kw["tools"] = tools
        return self.tok.apply_chat_template(messaggi, **kw)

    def decodifica(self, ids) -> str:
        return self.tok.decode(list(ids))


def taglia_paragrafi(t: str, max_car: int) -> str:
    """Inizio del testo a righe INTERE finche' non supera max_car (almeno titolo + primo paragrafo)."""
    righe, n = [], 0
    for r in t.split("\n"):
        if n + len(r) + 1 > max_car and len([x for x in righe if x.strip()]) >= 2:
            break
        righe.append(r)
        n += len(r) + 1
    return "\n".join(righe).rstrip()


def _dividi_pensiero(c: str) -> "tuple[str, str]":
    """content con <think>...</think> -> (ragionamento, risposta), come fa il template."""
    c = c or ""
    if "</think>" in c:
        rag = c.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        return rag, c.split("</think>")[-1].lstrip("\n")
    return "", c


def _msg_chat(ms: list, pensiero: bool, sistema: str = "") -> "tuple[list, list]":
    """messaggi user/assistant (+system) -> (messaggi col reasoning_content esplicito, contenuti)."""
    out, cont = [], []
    if sistema:
        out.append({"role": "system", "content": sistema})
        cont.append(sistema)
    for m in ms:
        if m["role"] == "system":
            if m["content"]:
                out.insert(0, {"role": "system", "content": m["content"]})
                cont.append(m["content"])
            continue
        if m["role"] == "assistant":
            rag, risp = _dividi_pensiero(m["content"])
            out.append({"role": "assistant", "content": risp, "reasoning_content": rag if pensiero else ""})
            cont.append(risp)
            if pensiero and rag:
                cont.append(rag)
        else:
            out.append({"role": m["role"], "content": m["content"]})
            cont.append(m["content"])
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out, cont


# ── CoT di Thor: STESSA costruzione di carica_batch_calibrazione (ordine, filtri, template) ──
def esempi_cot_tutti(tk: Tok) -> list:
    esempi = []
    for tipo in ("math", "code"):
        for riga in (DIR_COT / f"cot_{tipo}.jsonl").open(encoding="utf-8"):
            try:
                d = json.loads(riga)
            except json.JSONDecodeError:
                continue
            if d.get("errore") or not d.get("risposta"):
                continue
            testo = tk.tok.apply_chat_template(
                [{"role": "user", "content": d["q"]},
                 {"role": "assistant", "content": d["risposta"], "reasoning_content": d.get("ragionamento", "")}],
                tokenize=False)
            esempi.append((tipo, testo, d))
    random.Random(SEME).shuffle(esempi)
    out = []
    for i, (tipo, testo, d) in enumerate(esempi):
        out.append(Esempio(f"cot_{tipo}", f"cot#{i}", testo,
                           [d["q"], d["risposta"], d.get("ragionamento", "") or ""], d["q"], {"indice_mescolato": i}))
    return out


# ── MASSIVE: schemi, conversione frase -> chiamata (verificata: 601/602 = golden ufficiale) ──
_SLOT = re.compile(r"\[(\w+) : ([^\]]*)\]")


def schemi_massive() -> dict:
    """{nome: schema in stile OpenAI} dalle 55 funzioni di MASSIVE-Agents ("dict" -> "object",
    "optional" tolto: il `required` c'e' gia')."""
    riga = json.loads((SORG / "massive_agents_it-IT_input.jsonl").open(encoding="utf-8").readline())
    out = {}
    for f in riga["function"]:
        props = {k: {kk: vv for kk, vv in v.items() if kk != "optional"}
                 for k, v in f["parameters"]["properties"].items()}
        out[f["name"]] = {"type": "function", "function": {
            "name": f["name"], "description": f["description"],
            "parameters": {"type": "object", "properties": props, "required": list(f["parameters"].get("required", []))}}}
    return out


def converti_massive(d: dict, schemi: dict) -> "tuple[str, dict] | None":
    """intent 'scenario_azione' -> funzione 'scenario.azione'; slot annotati -> argomenti se
    previsti dallo schema; funzioni con `query` non annotata (qa.*) -> query = frase intera."""
    f = d["intent"].replace("_", ".", 1)
    if f not in schemi:
        return None
    props = schemi[f]["function"]["parameters"]["properties"]
    args = {}
    for s, v in _SLOT.findall(d["annot_utt"]):
        if s in props and s not in args:
            args[s] = v.strip()
    if "query" in props and "query" not in args:
        args["query"] = d["utt"]
    return f, args


def verifica_conversione_massive(schemi: dict, massive: dict) -> dict:
    """Conversione contro le chiamate ATTESE ufficiali di MASSIVE-Agents (602 del test)."""
    inp = jsonl(SORG / "massive_agents_it-IT_input.jsonl")
    ok_fn, ok_tutto, diversi = 0, 0, []
    for x in inp:
        (gf, ga), = x["gt"].items()
        o = converti_massive(massive[x["id"]], schemi)
        if o is None or o[0] != gf:
            diversi.append(x["id"])
            continue
        ok_fn += 1
        f, a = o
        buono = all(a.get(k, "") in acc or a.get(k, "").lower() in [v.lower() for v in acc] for k, acc in ga.items())
        buono = buono and all(k in ga for k in a)
        ok_tutto += buono
        if not buono:
            diversi.append(x["id"])
    return {"esempi": len(inp), "funzione_uguale": ok_fn, "funzione_e_argomenti_uguali": ok_tutto,
            "id_diversi": diversi}


def conversazioni_massive(frasi: list, schemi: dict, k: int, tag: str, tk: Tok) -> list:
    """Frasi dello STESSO scenario raggruppate a k per conversazione: utente -> chiamata,
    k volte (nessuna risposta dello strumento: la fonte non ne ha, e non se ne inventano).
    Strumenti mostrati: le funzioni dello scenario + DISTRATTORI_MASSIVE di altri scenari."""
    per_scen: dict = {}
    for d, (f, a) in frasi:
        per_scen.setdefault(d["scenario"], []).append((d, f, a))
    nomi = sorted(schemi)
    convs = []
    for scen in sorted(per_scen):
        lista = per_scen[scen]
        random.Random(f"{SEME}:{tag}:{scen}").shuffle(lista)
        for j in range(0, len(lista), k):
            convs.append((scen, lista[j:j + k]))
    random.Random(f"{SEME}:{tag}:conv").shuffle(convs)
    out = []
    for c, (scen, gruppo) in enumerate(convs):
        rng = random.Random(f"{SEME}:{tag}:distrattori:{c}")
        propri = [n for n in nomi if n.split(".")[0] == scen]
        altri = [n for n in nomi if n.split(".")[0] != scen]
        scelti = propri + rng.sample(altri, DISTRATTORI_MASSIVE)
        rng.shuffle(scelti)
        msgs, cont = [], []
        for d, f, a in gruppo:
            msgs.append({"role": "user", "content": d["utt"]})
            msgs.append({"role": "assistant", "content": "", "reasoning_content": "",
                         "tool_calls": [{"type": "function", "function": {"name": f, "arguments": a}}]})
            cont.append(d["utt"])
        testo = tk.chat(msgs, [schemi[n] for n in scelti])
        out.append(Esempio(f"massive_{tag}", f"massive:{tag}:" + ",".join(d["id"] for d, _, _ in gruppo), testo, cont,
                           gruppo[0][0]["utt"], {"scenario": scen, "id_massive": [d["id"] for d, _, _ in gruppo],
                                                 "strumenti": scelti}))
    return out


# ── Nemotron (strumenti in inglese) ──
def esempio_nemotron(r: dict, pensiero: bool, tk: Tok) -> Esempio:
    tools = (json.loads(r["metadata"]) if r.get("metadata") else {}).get("tools") or []
    msgs, cont = [], []
    for m in r["messages"]:
        c = m.get("content") or ""
        if m["role"] == "assistant":
            rag, risp = _dividi_pensiero(c)
            d = {"role": "assistant", "content": risp, "reasoning_content": rag if pensiero else ""}
            cont.append(risp)
            if pensiero and rag:
                cont.append(rag)
            if m.get("tool_calls"):
                d["tool_calls"] = []
                for tc in m["tool_calls"]:
                    a = tc["function"]["arguments"]
                    a = json.loads(a) if isinstance(a, str) else (a or {})
                    d["tool_calls"].append({"type": "function", "function": {"name": tc["function"]["name"],
                                                                             "arguments": a}})
                    cont.append(json.dumps(a, ensure_ascii=False))
            msgs.append(d)
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "content": c})
            cont.append(c)
        else:
            msgs.append({"role": m["role"], "content": c})
            cont.append(c)
    nomi = sorted({t["function"]["name"] for t in tools})
    return Esempio("nemotron", f"nemotron:{r['uuid']}", tk.chat(msgs, tools), cont,
                   next((m["content"] for m in r["messages"] if m["role"] == "user"), ""),
                   {"uuid": r["uuid"], "strumenti": nomi, "pensiero": pensiero})


# ═══════════════════════════ impacchettamento ═══════════════════════════
@dataclass
class Pacco:
    fonte: str
    seq: list                         # lista di liste di 1024 token
    usati: list                       # esempi usati (anche l'ultimo, troncato)
    token_esempi: int                 # token degli esempi usati (interi)


def impacca_fonte(nome: str, esempi, n_seq: int, tk: Tok, filtro=None, blocco: int = 64,
                  al_piu: bool = False) -> Pacco:
    """Esempi uno dopo l'altro (tokenizzati a blocchi, pigri) finche' riempiono n_seq x L.
    `filtro(e) -> bool` sui token. `al_piu=True` (fonti piccole): se gli esempi finiscono
    prima, ritorna le sequenze INTERE che ci sono (mai una sequenza corta o ripetuta)."""
    flusso, usati = [], []
    esempi = list(esempi)
    i = 0
    while len(flusso) < n_seq * L and i < len(esempi):
        pezzo = esempi[i:i + blocco]
        i += blocco
        for e, ids in zip(pezzo, tk.ids([e.testo for e in pezzo])):
            e.ids = ids
            if filtro is not None and not filtro(e):
                continue
            if len(flusso) >= n_seq * L:
                break
            flusso += ids
            usati.append(e)
    if al_piu:
        n_seq = min(n_seq, len(flusso) // L)
    if len(flusso) < n_seq * L:
        raise RuntimeError(f"fonte {nome}: {len(flusso)} token, ne servono {n_seq * L} ({n_seq} sequenze)")
    seq = [flusso[j * L:(j + 1) * L] for j in range(n_seq)]
    # tieni solo gli esempi davvero entrati nelle n_seq sequenze
    fine, cum, veri = n_seq * L, 0, []
    for e in usati:
        if cum >= fine:
            break
        veri.append(e)
        cum += len(e.ids)
    return Pacco(nome, seq, veri, sum(len(e.ids) for e in veri))


def interfoglia(pacchi: list) -> "tuple[list, list]":
    """Ordina le sequenze di piu' fonti per posizione frazionaria ((i+0.5)/n_fonte): ogni
    prefisso ha la composizione dichiarata. Ritorna (sequenze, etichette di fonte)."""
    slot = []
    for o, p in enumerate(pacchi):
        n = len(p.seq)
        for i in range(n):
            slot.append(((i + 0.5) / n, o, i))
    slot.sort()
    return [pacchi[o].seq[i] for _, o, i in slot], [pacchi[o].fonte for _, o, i in slot]


def ripartisci(totale: int, pesi: tuple) -> dict:
    """Quote intere per fonte proporzionali ai pesi (resti piu' grandi)."""
    s = sum(w for _, w in pesi)
    grezze = [(n, totale * w / s) for n, w in pesi]
    base = {n: int(math.floor(x)) for n, x in grezze}
    resto = totale - sum(base.values())
    for n, x in sorted(grezze, key=lambda t: -(t[1] - math.floor(t[1])))[:resto]:
        base[n] += 1
    return base


# ═══════════════════════════ n-grammi ═══════════════════════════
_R = np.random.default_rng(20260925).integers(1, 2**63 - 1, size=N_GRAM, dtype=np.int64).astype(np.uint64) | np.uint64(1)


def hash_ngrammi(ids) -> np.ndarray:
    a = np.asarray(ids, dtype=np.int64)
    if a.size < N_GRAM:
        return np.empty(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(a.astype(np.uint64), N_GRAM)
    with np.errstate(over="ignore"):
        return (w * _R).sum(axis=1, dtype=np.uint64)


def dentro(h: np.ndarray, ordinato: np.ndarray) -> np.ndarray:
    """Appartenenza a un array ORDINATO e senza doppi (np.unique): ricerca binaria, senza
    riordinare il riferimento a ogni chiamata (np.isin lo riordina ogni volta: minuti)."""
    if h.size == 0 or ordinato.size == 0:
        return np.zeros(h.size, dtype=bool)
    i = np.searchsorted(ordinato, h)
    i[i == ordinato.size] = 0
    return ordinato[i] == h


def ngrammi_testi(tk: Tok, testi: list) -> np.ndarray:
    parti = [hash_ngrammi(ids) for ids in tk.ids([t for t in testi if t and t.strip()])]
    return np.unique(np.concatenate(parti)) if parti else np.empty(0, dtype=np.uint64)


def ngrammi_sequenze(seq) -> np.ndarray:
    """Finestre sul flusso delle sequenze (testo grezzo: le sequenze SONO contenuto)."""
    t = torch.as_tensor(seq).reshape(-1).numpy()
    return np.unique(hash_ngrammi(t))


def leggi_mmlu(p: Path) -> list:
    """Formato di `llama-perplexity --multiple-choice` (scrivi_multiple_choice.py)."""
    d = p.read_bytes()
    n = struct.unpack_from("<I", d, 0)[0]
    pos = struct.unpack_from(f"<{n}I", d, 4)
    testi = []

    def stringa(o):
        ln = struct.unpack_from("<I", d, o)[0]
        return d[o + 4:o + 4 + ln].decode("utf-8", errors="replace"), o + 4 + ln

    for o in pos:
        q, o = stringa(o)
        k = struct.unpack_from("<I", d, o)[0]
        o += 4
        risposte = []
        for _ in range(k):
            r, o = stringa(o)
            risposte.append(r)
        testi.append(q + "\n" + "\n".join(risposte))
    return testi


# ═══════════════════════════ fonti: calibrazione ═══════════════════════════
class Fonti:
    """Carica gli estratti scaricati e produce le liste di Esempio per lato (cal / pan)."""

    def __init__(self, tk: Tok, soglia: float, k_massive: int):
        self.tk, self.soglia, self.k = tk, soglia, k_massive
        self.man = json.loads((SORG / "manifesto_sorgenti.json").read_text())
        self._verifica_impronte()
        self.schemi = schemi_massive()
        self.massive = {d["id"]: d for d in jsonl(SORG / "massive-1.1" / "it-IT.jsonl")}

    def _verifica_impronte(self):
        for k, v in self.man["estratti"].items():
            for m in (v if isinstance(v, list) else [v]):
                u = m.get("uscita")
                if u and sha256_file(SORG / u) != m["sha256_scritto"]:
                    raise RuntimeError(f"estratto {k}: {u} non combacia col manifesto delle sorgenti")

    def uscita(self, chiave: str) -> Path:
        return SORG / self.man["estratti"][chiave]["uscita"]

    def lato(self, prompt: str) -> str:
        return "pan" if secchio(prompt) < self.soglia else "cal"

    # ── testo grezzo ──
    def docs_web(self, chiave: str) -> list:
        return [Esempio("fineweb2_" + chiave.split("_")[-1], f"fw2:{r['id']}", r["text"] + "\n\n", [r["text"]],
                        extra={"id": r["id"], "url": r["url"]}) for r in jsonl(self.uscita(chiave))]

    def docs_wiki_it(self, chiave: str) -> list:
        out = []
        for r in jsonl(self.uscita(chiave)):
            t = taglia_paragrafi(f"{r['title']}\n\n{r['text']}", WIKI_IT_MAX_CAR)
            out.append(Esempio("wiki_it_" + chiave.split("_")[-1], f"wiki_it:{r['id']}", t + "\n\n", [t],
                               extra={"id": r["id"], "titolo": r["title"]}))
        return out

    # ── chat ──
    def aya(self) -> list:
        out = []
        for i, r in enumerate(jsonl(self.uscita("aya_it"))):
            msgs = [{"role": "user", "content": r["inputs"]},
                    {"role": "assistant", "content": r["targets"], "reasoning_content": ""}]
            out.append(Esempio("aya", f"aya:{i}", self.tk.chat(msgs), [r["inputs"], r["targets"]], r["inputs"],
                               {"annotation_type": r["annotation_type"]}))
        return out

    def oasst2(self, chiave: str) -> list:
        righe = [r for r in jsonl(self.uscita(chiave)) if not r["deleted"] and r["review_result"] is not False]
        figli: dict = {}
        for r in righe:
            figli.setdefault(r["parent_id"], []).append(r)
        out = []
        for radice in sorted((r for r in righe if r["parent_id"] is None), key=lambda r: r["message_id"]):
            percorso, nodo = [radice], radice
            while figli.get(nodo["message_id"]):
                nodo = sorted(figli[nodo["message_id"]],
                              key=lambda x: (x["rank"] if x["rank"] is not None else 99, x["message_id"]))[0]
                percorso.append(nodo)
            ms = [{"role": "user" if n["role"] == "prompter" else "assistant", "content": n["text"]} for n in percorso]
            msgs, cont = _msg_chat(ms, False)
            if len(msgs) >= 2:
                out.append(Esempio("oasst2", f"oasst2:{radice['message_tree_id']}", self.tk.chat(msgs), cont,
                                   radice["text"], {"albero": radice["message_tree_id"], "turni": len(msgs)}))
        return out

    def mfaq(self, chiave: str, siti_esclusi: set) -> list:
        pagine = [p for p in jsonl(self.uscita(chiave)) if p["domain"] not in siti_esclusi]
        random.Random(f"{SEME}:mfaq:{chiave}").shuffle(pagine)
        per_sito: dict = {}
        out = []
        for p in pagine:
            if per_sito.get(p["domain"], 0) >= MFAQ_PAGINE_PER_SITO:
                continue
            per_sito[p["domain"]] = per_sito.get(p["domain"], 0) + 1
            coppie = [c for c in p["qa_pairs"] if (c.get("question") or "").strip() and (c.get("answer") or "").strip()]
            for j in range(0, len(coppie), MFAQ_COPPIE_PER_CONV):
                ms = []
                for c in coppie[j:j + MFAQ_COPPIE_PER_CONV]:
                    ms += [{"role": "user", "content": c["question"].strip()},
                           {"role": "assistant", "content": c["answer"].strip()}]
                msgs, cont = _msg_chat(ms, False)
                out.append(Esempio("mfaq", f"mfaq:{p['id']}:{j}", self.tk.chat(msgs), cont, ms[0]["content"],
                                   {"pagina": p["id"], "sito": p["domain"]}))
        return out

    def smoltalk(self, chiave: str, pensiero: bool, lato: str) -> list:
        out = []
        for r in jsonl(self.uscita(chiave)):
            prompt = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            if self.lato(prompt) != lato:
                continue
            msgs, cont = _msg_chat(r["messages"], pensiero, r.get("custom_instructions") or "")
            nome = "smoltalk_think" if pensiero else "smoltalk_nothink"
            out.append(Esempio(nome, f"{chiave}:{r['_pos']}", self.tk.chat(msgs), cont, prompt, {"pos": r["_pos"]}))
        random.Random(f"{SEME}:{chiave}").shuffle(out)
        return out

    def reasonxl(self, chiave: str, lato: str) -> list:
        out = []
        for r in jsonl(self.uscita(chiave)):
            prompt = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            if self.lato(prompt) != lato:
                continue
            msgs, cont = _msg_chat(r["messages"], True)
            out.append(Esempio("reasonxl", f"reasonxl:{r['ds_uid']}", self.tk.chat(msgs), cont, prompt,
                               {"ds_uid": r["ds_uid"], "dataset_name": r["dataset_name"], "pos": r["_pos"]}))
        random.Random(f"{SEME}:{chiave}").shuffle(out)
        return out

    def nemotron(self, chiave: str, lato: str, vietati: set) -> list:
        righe = jsonl(self.uscita(chiave))
        random.Random(f"{SEME}:{chiave}").shuffle(righe)
        out = []
        for i, r in enumerate(righe):
            pensiero = lato == "cal" and i % NEMOTRON_OGNI_PENSIERO == NEMOTRON_OGNI_PENSIERO - 1
            e = esempio_nemotron(r, pensiero, self.tk)
            if vietati and set(e.extra["strumenti"]) & vietati:
                continue
            out.append(e)
        return out

    def massive_cal(self, frasi_pannello: set) -> "tuple[list, dict]":
        viste, frasi, conta = set(), [], {"train": 0, "non_convertibili": 0, "nel_pannello": 0, "doppie": 0}
        for d in self.massive.values():
            if d["partition"] != "train":
                continue
            conta["train"] += 1
            c = converti_massive(d, self.schemi)
            if c is None:
                conta["non_convertibili"] += 1
                continue
            n = norm(d["utt"])
            if n in frasi_pannello:
                conta["nel_pannello"] += 1
                continue
            if n in viste:
                conta["doppie"] += 1
                continue
            viste.add(n)
            frasi.append((d, c))
        conta["tenute"] = len(frasi)
        return conversazioni_massive(frasi, self.schemi, self.k, "cal", self.tk), conta

    def massive_pan(self) -> "tuple[list, set]":
        inp = jsonl(SORG / "massive_agents_it-IT_input.jsonl")
        frasi = []
        for x in inp:
            d = self.massive[x["id"]]
            if d["partition"] != "test":
                raise RuntimeError(f"MASSIVE-Agents id {x['id']}: partizione {d['partition']}, atteso test")
            frasi.append((d, converti_massive(d, self.schemi)))
        return conversazioni_massive(frasi, self.schemi, self.k, "pan", self.tk), {norm(d["utt"]) for d, _ in frasi}


# ═══════════════════════════ costruzione ═══════════════════════════
def parse_miscele(spec: str) -> dict:
    out = {}
    for pezzo in spec.split(";"):
        if not pezzo.strip():
            continue
        nome, _, q = pezzo.partition("=")
        quote = {}
        for x in q.split(","):
            d, _, n = x.partition(":")
            quote[d.strip()] = int(n)
        out[nome.strip()] = quote
    return out


def dimensioni_pool(miscele: dict, budget: int) -> dict:
    """Sequenze per dominio = quelle della miscela piu' esigente (quota/somma x budget)."""
    dim = {d: 0 for d in DOMINI}
    for nome, q in miscele.items():
        s = sum(q.values())
        for d, n in q.items():
            if d not in dim:
                raise ValueError(f"miscela {nome}: dominio {d} sconosciuto (noti: {DOMINI})")
            dim[d] = max(dim[d], math.ceil(budget * n / s))
    return dim


def pool_cot(tk: Tok, n: int, contaminati: set) -> "tuple[list, dict, list]":
    tutti = esempi_cot_tutti(tk)
    esclusi = {i for r in COT_ESCLUSI for i in r} | contaminati
    ammessi = [e for e in tutti if e.extra["indice_mescolato"] not in esclusi]
    p = impacca_fonte("cot", ammessi, n, tk)
    idx = [e.extra["indice_mescolato"] for e in p.usati]
    info = {"esempi_disponibili": len(tutti), "esclusi": [[0, 8], [279, 385]], "esempi_usati": len(p.usati),
            "indici_mescolati_usati": [min(idx), max(idx)], "token_esempi_usati": p.token_esempi,
            "conta": {t: sum(1 for e in p.usati if e.fonte == f"cot_{t}") for t in ("math", "code")},
            "esclusi_per_contaminazione": sorted(contaminati),
            "nota": "fino all'indice 278 le sequenze coincidono al bit con _batch_miscela (stesso ordine e "
                    "template); oltre, questo pool SALTA 279-385 (pannello), il vecchio no"}
    return p.seq, info, p.usati


def pool_edu(tk: Tok, n: int, esclusi_doc: set) -> "tuple[list, dict, list]":
    righe = [json.loads(r) for r in (DIR_PAN / EDU_FILE).open(encoding="utf-8")]
    docs = [(i, r["text"]) for i, r in enumerate(righe) if i >= EDU_PRIMO_DOC and i not in esclusi_doc]
    testo = "\n\n".join(t for _, t in docs)
    enc = tk.tok(testo, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) < n * L:
        raise RuntimeError(f"edu: {len(ids)} token, ne servono {n * L}")
    fine_car = enc["offset_mapping"][n * L - 1][1]
    usati, cum = [], 0
    for i, t in docs:
        if cum >= fine_car:
            break
        usati.append(Esempio("fineweb_edu", f"edu:{i}", t, [t], extra={"doc": i}))
        cum += len(t) + 2
    info = {"documenti_usati": len(usati), "primo_doc": docs[0][0], "ultimo_doc_usato": usati[-1].extra["doc"],
            "token_flusso_totale": len(ids), "documenti_esclusi_per_contaminazione": sorted(esclusi_doc),
            "nota": "stessa tokenizzazione di _batch_miscela (documenti dal 173 uniti con \\n\\n, un solo colpo)"
                    + ("" if not esclusi_doc else " — MA con documenti esclusi: non piu' identico al bit")}
    return [ids[j * L:(j + 1) * L] for j in range(n)], info, usati


def pool_wiki(tk: Tok, n: int) -> "tuple[list, dict, list]":
    testo = (DIR_PAN / WIKI_FILE).read_text(encoding="utf-8")
    enc = tk.tok(testo, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) < n * L:
        raise RuntimeError(f"wiki: {len(ids)} token, ne servono {n * L}")
    fine_car = enc["offset_mapping"][n * L - 1][1]
    titoli = len(re.findall(r"(?m)^ = [^=].* = $", testo[:fine_car]))
    info = {"token_flusso_totale": len(ids), "caratteri_usati": fine_car, "caratteri_totali": len(testo),
            "articoli_toccati_titoli_di_primo_livello": titoli,
            "nota": "stessa tokenizzazione di _batch_miscela (file intero in un colpo)"}
    return [ids[j * L:(j + 1) * L] for j in range(n)], info, []


def dominio_multi(nome: str, n_tot: int, fonti: dict, composizione: tuple, tk: Tok,
                  filtri: "dict | None" = None) -> "tuple[list, list, dict, list]":
    """Dominio a piu' fonti: quote dalla composizione; le fonti piccole (non ultime) danno al
    piu' cio' che hanno, l'ultima riempie il resto. Ritorna (seq, etichette, info, esempi usati)."""
    filtri = filtri or {}
    quote = ripartisci(n_tot, tuple((f, w) for f, w in composizione))
    pacchi, info, usati = [], {"quote_dichiarate": quote, "quote_effettive": {}, "fonti": {}}, []
    mancano = 0
    ultima = composizione[-1][0]
    for f, _ in composizione:
        q = quote[f] + (mancano if f == ultima else 0)
        if f != ultima:
            p = impacca_fonte(f, fonti[f], q, tk, filtri.get(f), al_piu=True)
            if len(p.seq) < q:
                mancano += q - len(p.seq)
        else:
            p = impacca_fonte(f, fonti[f], q, tk, filtri.get(f))
        pacchi.append(p)
        usati += p.usati
        info["quote_effettive"][f] = len(p.seq)
        info["fonti"][f] = {"esempi_usati": len(p.usati), "token_esempi_usati": p.token_esempi,
                            "sequenze": len(p.seq), "esempi_candidati": len(fonti[f])}
    if mancano:
        info["nota"] = f"le fonti piccole avevano {mancano} sequenze in meno: le ha date {ultima}"
    seq, etichette = interfoglia(pacchi)
    assert len(seq) == n_tot, (nome, len(seq), n_tot)
    return seq, etichette, info, usati


def costruisci(args) -> int:
    t0 = time.time()
    tk = Tok()
    miscele = parse_miscele(args.miscele)
    dim = dimensioni_pool(miscele, args.budget)
    log(f"miscele {miscele} · budget {args.budget} sequenze · pool per dominio {dim} (totale {sum(dim.values())})")
    F = Fonti(tk, args.soglia_pannello, args.k_massive)

    # ── PANNELLO NUOVO prima: la calibrazione esclude cio' che il pannello usa ──
    conv_ok = verifica_conversione_massive(F.schemi, F.massive)
    log(f"conversione MASSIVE contro il golden ufficiale: funzione {conv_ok['funzione_uguale']}/{conv_ok['esempi']}, "
        f"funzione+argomenti {conv_ok['funzione_e_argomenti_uguali']}/{conv_ok['esempi']}")
    massive_pan, frasi_pan = F.massive_pan()
    mfaq_valid_siti = {p["domain"] for p in jsonl(F.uscita("mfaq_it_valid"))}
    fonti_pan = {
        "fineweb2_test": F.docs_web("fineweb2_it_test"),
        "wiki_it_rg1": F.docs_wiki_it("wiki_it_pan"),
        "aya": [e for e in F.aya() if F.lato(e.prompt) == "pan"],
        "oasst2": F.oasst2("oasst2_it_valid"),
        "mfaq": F.mfaq("mfaq_it_valid", set()),
        "smoltalk_nothink": F.smoltalk("smoltalk2_nothink_pan", False, "pan"),
        "smoltalk_think": F.smoltalk("smoltalk2_think_pan", True, "pan"),
        "reasonxl": F.reasonxl("reasonxl_pan", "pan"),
        "massive_agents": massive_pan,
    }

    # ── CALIBRAZIONE ──
    fonti_cal = {
        "it_web": F.docs_web("fineweb2_it_train"),
        "it_wiki": F.docs_wiki_it("wiki_it_cal"),
        "aya": [e for e in F.aya() if F.lato(e.prompt) == "cal"],
        "oasst2": F.oasst2("oasst2_it_train"),
        "mfaq": F.mfaq("mfaq_it_train", mfaq_valid_siti),
        "smoltalk_nothink": F.smoltalk("smoltalk2_nothink_cal", False, "cal"),
        "smoltalk_think": F.smoltalk("smoltalk2_think_cal", True, "cal"),
        "reasonxl": F.reasonxl("reasonxl_cal", "cal"),
        "nemotron": F.nemotron("nemotron_tool_cal", "cal", set()),
    }
    fonti_cal["massive"], conta_massive = F.massive_cal(frasi_pan)
    filtro_wiki_it = (lambda e: len(e.ids) >= WIKI_IT_MIN_TOKEN)
    filtro_nemo = (lambda e: len(e.ids) <= NEMOTRON_MAX_TOKEN)

    esclusi: dict = {}           # dominio -> id di esempi tolti per contaminazione
    esclusi_edu: set = set()
    esclusi_cot: set = set()
    verifiche_giri = []
    for giro in range(1, 4):
        log(f"── giro {giro}: costruzione dei domini di calibrazione ──")
        corpus, etichette, info_dom, usati_dom = {}, {}, {}, {}

        def via(lista, dom):
            return [e for e in lista if e.id not in esclusi.get(dom, set())]

        s, info, u = pool_wiki(tk, dim["wiki"])
        corpus["wiki"], etichette["wiki"], info_dom["wiki"], usati_dom["wiki"] = s, ["wikitext2_train"] * len(s), info, u
        s, info, u = pool_edu(tk, dim["edu"], esclusi_edu)
        corpus["edu"], etichette["edu"], info_dom["edu"], usati_dom["edu"] = s, ["fineweb_edu"] * len(s), info, u
        s, info, u = pool_cot(tk, dim["cot"], esclusi_cot)
        corpus["cot"], etichette["cot"], info_dom["cot"], usati_dom["cot"] = s, ["cot"] * len(s), info, u
        for dom, chiave, filtro in (("it_web", "it_web", None), ("it_wiki", "it_wiki", filtro_wiki_it)):
            if dim[dom]:
                p = impacca_fonte(dom, via(fonti_cal[chiave], dom), dim[dom], tk, filtro)
                corpus[dom], etichette[dom], usati_dom[dom] = p.seq, [p.fonte] * len(p.seq), p.usati
                info_dom[dom] = {"documenti_usati": len(p.usati), "token_documenti_usati": p.token_esempi,
                                 "documenti_candidati": len(fonti_cal[chiave])}
        for dom in ("it_chat", "it_pensiero", "strumenti"):
            if not dim[dom]:
                continue
            fonti_d = {f: via(fonti_cal[f], dom) for f, _ in COMPOSIZIONE[dom]}
            filtri = {"nemotron": filtro_nemo} if dom == "strumenti" else {}
            s, et, info, u = dominio_multi(dom, dim[dom], fonti_d, COMPOSIZIONE[dom], tk, filtri)
            corpus[dom], etichette[dom], info_dom[dom], usati_dom[dom] = s, et, info, u
        # ── pannello nuovo (dopo la calibrazione: strumenti_en vieta gli strumenti visti) ──
        vietati = {n for e in usati_dom.get("strumenti", []) for n in e.extra.get("strumenti", [])}
        fonti_pan["nemotron"] = F.nemotron("nemotron_tool_pan", "pan", vietati)
        pannello, info_pan, usati_pan = costruisci_pannello(fonti_pan, tk, filtro_wiki_it, filtro_nemo)
        # ── verifiche ──
        ver, contaminati = verifica(tk, corpus, usati_dom, pannello, usati_pan, fonti_cal, fonti_pan, F, conta_massive)
        verifiche_giri.append({"giro": giro, "esempi_tolti": {d: sorted(v) for d, v in contaminati.items()},
                               "contenuto_calibrazione_vs": ver["contenuto_calibrazione_vs"],
                               "finestre_in_comune_per_esempio": ver["esempi_calibrazione_con_ngrammi_in_comune"]})
        if not contaminati:
            break
        for d, v in contaminati.items():
            if d == "edu":
                esclusi_edu |= {int(x.split(":")[1]) for x in v}
            elif d == "cot":
                esclusi_cot |= {int(x.split("#")[1]) for x in v}
            else:
                esclusi.setdefault(d, set()).update(v)
        log(f"  giro {giro}: esempi contaminati tolti {({d: len(v) for d, v in contaminati.items()})}: ricostruisco")
    else:
        raise RuntimeError("dopo 3 giri restano n-grammi in comune: da esaminare a mano")

    # ── salvataggio ──
    tensori = {d: torch.tensor(corpus[d], dtype=torch.int64) for d in DOMINI if dim[d]}
    for d, t in tensori.items():
        assert t.shape == (dim[d], L) and t.dtype == torch.int64, (d, t.shape)
    DIR_CAL.mkdir(parents=True, exist_ok=True)
    torch.save(tensori, OUT_PT)
    pan_t = {d: torch.tensor(pannello[d], dtype=torch.int64).reshape(PANNELLO[d][0], SEQ_PER_BATCH_PANNELLO, L)
             for d in PANNELLO}
    man_sorg = json.loads((SORG / "manifesto_sorgenti.json").read_text())
    meta_pan = {"quando": time.strftime("%Y-%m-%d %H:%M:%S"), "forme": {d: list(t.shape) for d, t in pan_t.items()},
                "sha256_token": {d: sha_tensore(t) for d, t in pan_t.items()}, "domini": info_pan,
                "seme": SEME, "soglia_pannello_hash": args.soglia_pannello, "k_massive": args.k_massive,
                "tokenizzatore": str(SRC), "sorgenti": "calibrazione/sorgenti/manifesto_sorgenti.json",
                "conversione_massive_contro_golden": conv_ok,
                "disgiunzione": ver, "costruttore": "thor_v2/costruisci_corpus_calibrazione.py",
                "esempi_decodificati": {d: tk.decodifica(pan_t[d][0, 0].tolist())[:300] for d in pan_t}}
    if OUT_PAN.exists() and not args.forza_pannello:
        vecchio = torch.load(OUT_PAN, weights_only=False)
        diversi = [d for d in pan_t if d not in vecchio["domini"] or not torch.equal(vecchio["domini"][d], pan_t[d])]
        if diversi:
            raise RuntimeError(f"{OUT_PAN} esiste e differisce nei domini {diversi}: --forza-pannello per riscriverlo")
        log(f"pannello nuovo identico a {OUT_PAN} ✅")
    else:
        torch.save({"domini": pan_t, "meta": meta_pan}, OUT_PAN)
        OUT_PAN.with_suffix(".json").write_text(json.dumps(meta_pan, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in (OUT_PT, OUT_PAN):
        if "pannello_ids" in p.name:
            raise AssertionError("mai scrivere pannello_ids.*")

    man = {
        "quando": time.strftime("%Y-%m-%d %H:%M:%S"), "file": OUT_PT.name, "sha256_file": sha256_file(OUT_PT),
        "byte_file": OUT_PT.stat().st_size, "l_seq": L, "seme": SEME, "tokenizzatore": str(SRC),
        "budget_sequenze": args.budget, "miscele": miscele, "sequenze_per_dominio": dim,
        "composizione_interna": {d: dict(c) for d, c in COMPOSIZIONE.items()},
        "soglia_pannello_hash": args.soglia_pannello, "k_massive": args.k_massive,
        "domini": {d: {"sequenze": int(t.shape[0]), "token": int(t.numel()), "sha256_token": sha_tensore(t),
                       **info_dom[d]} for d, t in tensori.items()},
        "etichette_per_sequenza": {d: _rle(etichette[d]) for d in tensori},
        "massive_train": conta_massive,
        "conversione_massive_contro_golden": meta_pan["conversione_massive_contro_golden"],
        "disgiunzione": ver, "giri_di_pulizia": verifiche_giri,
        "sorgenti": {"manifesto": str(SORG / "manifesto_sorgenti.json"),
                     "sha256_manifesto": sha256_file(SORG / "manifesto_sorgenti.json"),
                     "estratti": {k: _riassunto_sorgente(v) for k, v in man_sorg["estratti"].items()},
                     "locali": man_sorg.get("locali")},
        "pannello_nuovo": {"file": str(OUT_PAN), "sha256_file": sha256_file(OUT_PAN), "forme": meta_pan["forme"],
                           "sha256_token": meta_pan["sha256_token"]},
        "esempi_decodificati": {d: tk.decodifica(t[0].tolist())[:300] for d, t in tensori.items()},
        "secondi": round(time.time() - t0, 1),
        "uso": "THOR_CALIB_CORPUS=" + str(OUT_PT) + " THOR_CALIB_MISCELA=<dominio:quota,...> (patch "
               "strumenti/patch_corpus_calibrazione.py)",
    }
    OUT_JSON.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    scrivi_leggimi(man, meta_pan)
    totale = sum(p.stat().st_size for p in DIR_CAL.rglob("*") if p.is_file())
    if totale > 3 * 10**9:
        raise SystemExit(f"⛔ {totale} byte in {DIR_CAL}: oltre 3 GB")
    log(f"✅ {OUT_PT} ({OUT_PT.stat().st_size / 2**20:.1f} MiB) · {OUT_PAN.name} · {OUT_JSON.name} · "
        f"cartella {totale / 2**20:.1f} MiB · {time.time() - t0:.0f} s")
    for d, t in tensori.items():
        log(f"  {d:12s} {tuple(t.shape)} · {man['domini'][d].get('quote_effettive', '')}")
    return 0


def _rle(xs: list) -> list:
    out = []
    for x in xs:
        if out and out[-1][0] == x:
            out[-1][1] += 1
        else:
            out.append([x, 1])
    return out


def _riassunto_sorgente(v) -> "dict | list":
    chiavi = ("repo", "revisione", "file", "byte_sorgente", "sha256_sorgente", "uscita", "sha256_scritto",
              "licenza", "provenienza_testo", "uso", "row_group", "righe_scritte", "url_tar", "sha256_tar")
    if isinstance(v, list):
        return [_riassunto_sorgente(x) for x in v]
    return {k: v[k] for k in chiavi if k in v}


def costruisci_pannello(fonti_pan: dict, tk: Tok, filtro_wiki_it, filtro_nemo) -> "tuple[dict, dict, dict]":
    pannello, info, usati = {}, {}, {}
    for dom, (n_batch, comp) in PANNELLO.items():
        n = n_batch * SEQ_PER_BATCH_PANNELLO
        filtri = {"wiki_it_rg1": filtro_wiki_it, "nemotron": filtro_nemo}
        if len(comp) == 1:
            f = comp[0][0]
            p = impacca_fonte(f, fonti_pan[f], n, tk, filtri.get(f))
            pannello[dom], usati[dom] = p.seq, p.usati
            info[dom] = {"fonti": {f: {"esempi_usati": len(p.usati), "token_esempi_usati": p.token_esempi,
                                       "sequenze": n, "esempi_candidati": len(fonti_pan[f])}}}
        else:
            fisse = sum(q for _, q in comp if q)
            pesi = tuple((f, q if q else n - fisse) for f, q in comp)
            s, _et, inf, u = dominio_multi(dom, n, fonti_pan, pesi, tk, filtri)
            pannello[dom], usati[dom], info[dom] = s, u, inf
    return pannello, info, usati


# ═══════════════════════════ verifiche di disgiunzione ═══════════════════════════
def verifica(tk: Tok, corpus: dict, usati_dom: dict, pannello: dict, usati_pan: dict, fonti_cal: dict,
             fonti_pan: dict, F: Fonti, conta_massive: dict) -> "tuple[dict, dict]":
    t0 = time.time()
    ver: dict = {}
    grezzi = ("wiki", "edu", "it_web", "it_wiki")
    # n-grammi di CONTENUTO della calibrazione, per esempio (per sapere chi togliere)
    cal_per_esempio: dict = {}
    for d, us in usati_dom.items():
        if d in ("wiki",):
            continue
        for e in us:
            cal_per_esempio[(d, e.id)] = ngrammi_testi(tk, e.contenuti)
    cal_wiki = ngrammi_sequenze(corpus["wiki"])
    tutti_cal = np.unique(np.concatenate([cal_wiki] + list(cal_per_esempio.values())))
    # riferimenti
    pan_cont = {d: np.unique(np.concatenate([ngrammi_testi(tk, e.contenuti) for e in us]
                                            + [np.empty(0, dtype=np.uint64)])) for d, us in usati_pan.items()}
    pan_seq = {d: ngrammi_sequenze(s) for d, s in pannello.items()}
    storico = torch.load(DIR_PAN / "pannello_ids.pt", weights_only=False)["domini"]
    sto_seq = {d: ngrammi_sequenze(t) for d, t in storico.items()}
    wt = np.unique(hash_ngrammi(tk.uno(FILE_WIKI_TEST.read_text(encoding="utf-8"))))
    mm = ngrammi_testi(tk, leggi_mmlu(FILE_MMLU))
    riferimenti = {**{f"pannello_nuovo:{d}": v for d, v in pan_cont.items()},
                   **{f"pannello_storico:{d}": v for d, v in sto_seq.items()},
                   "verdetto:wiki.test.raw": wt, "verdetto:mmlu_test.bin": mm}
    rif_tutti = np.unique(np.concatenate(list(riferimenti.values())))
    # 1) contenuto calibrazione contro ogni riferimento
    ver["ngrammi_contenuto_calibrazione"] = int(tutti_cal.size)
    ver["contenuto_calibrazione_vs"] = {k: int(dentro(v, tutti_cal).sum()) for k, v in riferimenti.items()}
    ver["ngrammi_riferimenti"] = {k: int(v.size) for k, v in riferimenti.items()}
    contaminati: dict = {}
    dettaglio: dict = {}
    for (d, eid), h in cal_per_esempio.items():
        if h.size and dentro(h, rif_tutti).any():
            contaminati.setdefault(d, set()).add(eid)
            for k, v in riferimenti.items():
                n = int(dentro(h, v).sum())
                if n:
                    dettaglio.setdefault(d, {}).setdefault(k, []).append([eid, n])
    ver["esempi_calibrazione_con_ngrammi_in_comune"] = dettaglio
    # 1b) messaggi IDENTICI (normalizzati, >= MIN_CAR_IDENTICI caratteri): i doppioni corti sfuggono
    # ai 32-grammi (visto: risposte Aya ri-annotate uguali, < 32 token, con prompt diversi). Pannello =
    # nuovo + storico (esempi CoT 0-8 e 279-385, documenti FineWeb-Edu 0-172).
    righe_edu = [json.loads(r) for r in (DIR_PAN / EDU_FILE).open(encoding="utf-8")]
    cot_tutti = esempi_cot_tutti(tk)
    esclusi_cot = {i for r in COT_ESCLUSI for i in r}
    pan_norm = {norm(c) for us in usati_pan.values() for e in us for c in e.contenuti}
    pan_norm |= {norm(c) for e in cot_tutti if e.extra["indice_mescolato"] in esclusi_cot for c in e.contenuti}
    pan_norm |= {norm(r["text"]) for r in righe_edu[:EDU_PRIMO_DOC]}
    pan_norm = {x for x in pan_norm if len(x) >= MIN_CAR_IDENTICI}
    identici: dict = {}
    for d, us in usati_dom.items():
        for e in us:
            if any(len(x) >= MIN_CAR_IDENTICI and x in pan_norm for x in (norm(c) for c in e.contenuti)):
                identici.setdefault(d, []).append(e.id)
                contaminati.setdefault(d, set()).add(e.id)
    ver["esempi_calibrazione_con_messaggi_identici_al_pannello"] = identici
    # wikitext-2 train: flusso unico tokenizzato in un colpo (come _batch_miscela): gli n-grammi in
    # comune con validation/test si RIPORTANO (split ufficiali), non si tolgono
    ver["wiki_train_ngrammi_in_comune"] = {k: int(dentro(v, cal_wiki).sum()) for k, v in riferimenti.items()}
    # 2) sequenze intere (template e schemi compresi): informativo
    cal_seq = {d: ngrammi_sequenze(s) for d, s in corpus.items()}
    cal_seq_tutti = np.unique(np.concatenate(list(cal_seq.values())))
    ver["sequenze_intere_vs_pannello_nuovo"] = {d: int(dentro(v, cal_seq_tutti).sum()) for d, v in pan_seq.items()}
    ver["sequenze_intere_vs_pannello_storico"] = {d: int(dentro(v, cal_seq_tutti).sum()) for d, v in sto_seq.items()}
    ver["nota_sequenze_intere"] = ("finestre di 32 token delle sequenze COMPLETE (template, blocchi <tools> "
                                   "compresi): gli schemi MASSIVE sono gli stessi per costruzione, quindi "
                                   "strumenti_it ne ha in comune; conta il contenuto (sopra)")
    # 3) disgiunzioni per identificativo / prompt
    pid = lambda lista: {e.id for e in lista}  # noqa: E731
    ver["id"] = {
        "fineweb2_train_vs_test": len({e.extra["id"] for e in usati_dom["it_web"]}
                                      & {e.extra["id"] for e in fonti_pan["fineweb2_test"]}),
        "wiki_it_rg0_vs_rg1": len({e.extra["id"] for e in fonti_cal["it_wiki"]}
                                  & {e.extra["id"] for e in fonti_pan["wiki_it_rg1"]}),
        "nemotron_uuid": len({e.extra["uuid"] for e in fonti_cal["nemotron"]}
                             & {e.extra["uuid"] for e in fonti_pan["nemotron"]}),
        "nemotron_strumenti_pannello_visti_in_calibrazione": len(
            {n for e in usati_pan["strumenti_en"] for n in e.extra["strumenti"]}
            & {n for e in usati_dom["strumenti"] for n in e.extra.get("strumenti", [])}),
        "reasonxl_ds_uid": len({e.extra["ds_uid"] for e in fonti_cal["reasonxl"]}
                               & {e.extra["ds_uid"] for e in fonti_pan["reasonxl"]}),
        "oasst2_alberi": len({e.extra["albero"] for e in fonti_cal["oasst2"]}
                             & {e.extra["albero"] for e in fonti_pan["oasst2"]}),
        "mfaq_siti": len({e.extra["sito"] for e in fonti_cal["mfaq"]} & {e.extra["sito"] for e in fonti_pan["mfaq"]}),
        "massive_id": len({i for e in fonti_cal["massive"] for i in e.extra["id_massive"]}
                          & {i for e in fonti_pan["massive_agents"] for i in e.extra["id_massive"]}),
        "cot_indici_esclusi_presenti": sum(1 for e in usati_dom["cot"]
                                           if e.extra["indice_mescolato"] < 9 or 279 <= e.extra["indice_mescolato"] <= 385),
        "edu_documenti_del_pannello_presenti": sum(1 for e in usati_dom["edu"] if e.extra["doc"] < EDU_PRIMO_DOC),
        "id_esempi_comuni_calibrazione_pannello": len(set().union(*[pid(v) for v in usati_dom.values()])
                                                      & set().union(*[pid(v) for v in usati_pan.values()])),
    }
    prompt_cal = {norm(e.prompt) for d in ("it_chat", "it_pensiero") for e in usati_dom[d] if e.prompt}
    prompt_pan = {norm(e.prompt) for d in ("it_chat", "it_pensiero") for e in usati_pan[d] if e.prompt}
    ver["prompt_normalizzati_comuni_chat"] = len(prompt_cal & prompt_pan)
    frasi_cal = {norm(u) for e in usati_dom["strumenti"] if e.fonte.startswith("massive") for u in e.contenuti}
    frasi_pan = {norm(u) for e in usati_pan["strumenti_it"] for u in e.contenuti}
    ver["massive_frasi_normalizzate_comuni"] = len(frasi_cal & frasi_pan)
    ver["massive_train"] = conta_massive
    ver["secondi"] = round(time.time() - t0, 1)
    bloccanti = [k for k, v in ver["id"].items() if v] + (["prompt"] if ver["prompt_normalizzati_comuni_chat"] else []) \
        + (["massive_frasi"] if ver["massive_frasi_normalizzate_comuni"] else [])
    ver["esito"] = "✅ disgiunti" if not bloccanti and not contaminati else f"⛔ {bloccanti} {list(contaminati)}"
    log(f"  verifiche ({ver['secondi']} s): contenuto vs riferimenti {ver['contenuto_calibrazione_vs']}")
    log(f"  id {ver['id']} · prompt comuni {ver['prompt_normalizzati_comuni_chat']} · frasi MASSIVE comuni "
        f"{ver['massive_frasi_normalizzate_comuni']} · sequenze intere vs pannello nuovo "
        f"{ver['sequenze_intere_vs_pannello_nuovo']}")
    if bloccanti:
        raise RuntimeError(f"disgiunzione violata: {bloccanti}")
    return ver, {d: {x for x in v} for d, v in contaminati.items()}


# ═══════════════════════════ LEGGIMI ═══════════════════════════
def scrivi_leggimi(man: dict, meta_pan: dict) -> None:
    righe = [f"# Corpus di calibrazione multi-dominio di Thor", "",
             f"Costruito il {man['quando']} da `thor_v2/costruisci_corpus_calibrazione.py` (fonti scaricate da "
             f"`thor_v2/scarica_corpus_calibrazione.py`, manifesto `sorgenti/manifesto_sorgenti.json`). "
             f"Solo fonti pubbliche, revisioni HF fissate. Nessun dato privato di MOGAVIS.", "",
             f"File `{man['file']}` ({man['byte_file']} byte, sha256 `{man['sha256_file']}`): "
             f"`{{dominio: tensore (n, {L}) int64}}`, seme {man['seme']}, tokenizzatore `{man['tokenizzatore']}`.", "",
             "## Domini", "", "| dominio | sequenze | token | fonti (sequenze) | sha256 dei token |", "|---|---|---|---|---|"]
    for d, v in man["domini"].items():
        fonti = v.get("quote_effettive") or {}
        righe.append(f"| {d} | {v['sequenze']} | {v['token']} | "
                     f"{', '.join(f'{k} {n}' for k, n in fonti.items()) or '—'} | `{v['sha256_token'][:16]}…` |")
    righe += ["", "## Miscele dichiarate (sequenze per ciclo)", "", "```json", json.dumps(man["miscele"]), "```", "",
              "## Uso", "", "```", "THOR_CALIB_CORPUS=" + str(OUT_PT) + " \\",
              "THOR_CALIB_MISCELA=wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2 \\",
              "  (somma = multiplo di THOR_VALIDA_BATCH_SEQ; con somma 16 e 8 sequenze per batch il ciclo e' di 2 batch)",
              "```", "", "Richiede la patch `thor_v2/strumenti/patch_corpus_calibrazione.py` applicata a "
              "`valida_finestra_8_12.py`.", "",
              "## Disgiunzione", "", "```json", json.dumps(man["disgiunzione"], ensure_ascii=False, indent=1)[:6000],
              "```", "", "## Pannello nuovo", "", f"`{meta_pan and OUT_PAN}` — forme `{json.dumps(meta_pan['forme'])}`",
              "", "## Esempi decodificati (primi 300 caratteri della prima sequenza)", ""]
    for d, t in man["esempi_decodificati"].items():
        righe += [f"### {d}", "", "```", t.replace("```", "ˋˋˋ"), "```", ""]
    righe += ["## Licenze e provenienza", ""]
    for k, v in man["sorgenti"]["estratti"].items():
        for x in (v if isinstance(v, list) else [v]):
            righe.append(f"- **{k}**: `{x.get('repo')}` @ `{str(x.get('revisione'))[:12]}` · `{x.get('file', '')}` · "
                         f"{x.get('licenza')} · {x.get('provenienza_testo')}")
    righe.append("")
    (DIR_CAL / "LEGGIMI.md").write_text("\n".join(righe) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=int, default=1200, help="sequenze da 1024 di una corsa (150 batch x 8)")
    ap.add_argument("--miscele", default=MISCELE_DEFAULT)
    ap.add_argument("--soglia-pannello", type=float, default=0.30)
    ap.add_argument("--k-massive", type=int, default=4)
    ap.add_argument("--forza-pannello", action="store_true", help="riscrive pannello_nuovi_domini.pt se diverso")
    return costruisci(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
