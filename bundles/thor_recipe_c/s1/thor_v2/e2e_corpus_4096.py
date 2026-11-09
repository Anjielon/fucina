#!/usr/bin/env python3
"""Corpus E2E a contesto 4096 (fase E2E sulle scale, `e2e_progetto_scelto.md` §2.4) + held-out separato.
25/9/2026. File NUOVO: non modifica nessun file esistente; importa IN SOLA LETTURA i costruttori v1, v2 e v2_sem.

## Cosa costruisce
- **Addestramento** `e2e_corpus_4096_<variante>.pt`: `--sequenze` righe da 4096 token (default 512 = E-A,
  2.097.152 token; E-B vuole 4096, vedi «capienza» nel manifesto).
- **Held-out** `e2e_heldout_4096_<variante>.pt`: `--heldout` righe da 4096 (default 64), DISGIUNTE dall'addestramento.
- **Held-out dei pannelli** `e2e_heldout_pannelli.pt`: le prime 8 sequenze di ogni dominio dei pannelli gia'
  certificati (`pannello_ids.pt` + `pannello_nuovi_domini_v2.pt`), con sha256 (P2, §2.4 del progetto).
- Manifesto `e2e_corpus_4096_<variante>.json` e `LEGGIMI.md`, generati (nessun numero scritto a mano).

## Regole (dal progetto, §2.4, e da P1 §3.2)
- **Stesse fonti pubbliche e revisioni fissate del v2** (`sorgenti/manifesto_sorgenti_v2.json`): le liste di
  esempi sono quelle di `costruisci_corpus_calibrazione_v2.fonti_calibrazione` (stessi semi, template, filtri);
  cot con `pool_cot_v2`, edu con `pool_edu`, impacchettamento con `impacca_fonte`/`dominio_multi`/`interfoglia` del
  v1 — portati a 4096 cambiando la sola costante `C.L` dentro un blocco (`lunghezza`), come fa il v2_sem col suo
  innesto su `V2.verifica_v2`. Il pannello v2 si ricostruisce a 1024 col codice v2 e si confronta AL BIT col file.
- **Niente wikitext** (EfficientQAT tab. 13). Quote della miscela U senza wiki, lette da `V2.MISCELE`:
  edu 2, cot 2, it_web 2, it_wiki 1, it_chat 3, it_pensiero 1, strumenti 2.
- **Nessuna ripetizione.** Un dominio che non ha abbastanza esempi si ferma a cio' che esiste; il resto va a
  edu e it_web (pesi 2:2). Dentro un dominio a piu' fonti, le fonti piccole danno cio' che hanno e il resto va
  alle altre con gli stessi pesi (`riempi`). Tutto dichiarato nel manifesto (quote dichiarate / effettive).
- **Testo continuo** (edu, it_web, it_wiki): documenti concatenati come nel protocollo PPL, `cu_seqlens = [0, 4096]`.
  **Chat, cot, strumenti**: esempi uno dopo l'altro, taglio ogni 4096; `cu_seqlens` = inizi degli esempi dentro
  la riga (per fla e per l'attenzione piena). Una riga puo' iniziare a meta' esempio (`inizio_esempio` = False).
- **Disgiunzione del corpus** (addestramento + held-out) dal pannello v2, dal pannello storico, da
  `wiki.test.raw` e da `mmlu_test.bin`: `costruisci_corpus_calibrazione_v2_sem.verifica_sem` (finestre di 32 token
  di contenuto, messaggi identici, id, regole R1-R3), importata; esempi colpiti tolti e dominio ricostruito.
- **Held-out** preso dalla CODA di ogni fonte (l'addestramento dalla testa), disgiunto per esempio e in piu':
  H1 nessuna finestra di 32 token di contenuto in comune con l'addestramento; H2 nessun messaggio normalizzato
  identico (>= 24 caratteri; frasi MASSIVE a qualunque lunghezza); H3 (R2) nessun problema cot con lo stesso
  originale MetaMathQA; H4 (R3) nessun turno utente quasi-doppione (difflib >= 0.80) dell'addestramento;
  H5 nessuna finestra di contenuto ne' messaggio in comune coi corpus di calibrazione della FORGIA
  (`calibrazione/calibrazione_corpus*.pt`): lo studente e' stato adattato su quelli.

## Uso (SOLO CPU; la GPU e' della forgia)
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 \\
        nice -n 19 ionice -c3 @@ROOT:venv@@/bin/python3 e2e_corpus_4096.py \\
        [--variante tutte|senza_llama|entrambe] [--sequenze 512] [--heldout 64] [--uscita DIR]
Test: `e2e_test_corpus_4096.py`.

## Formato dei .pt (torch.load(..., weights_only=True))
    {"l_seq": 4096, "lato": "train"|"heldout", "variante": str, "nomi_domini": [..],
     "domini": {d: int64 (n_d, 4096)}, "cu_ptr": {d: int64 (n_d+1)}, "cu": {d: int32 (flat)},
     "inizio_esempio": {d: bool (n_d)}, "fonti": {d: [etichetta per riga]}, "esempi": {d: [id degli esempi usati]},
     "ordine": int64 (N, 2) = (indice del dominio in nomi_domini, riga): ogni prefisso ha la composizione U}
    cu_seqlens della riga i del dominio d = cu[d][cu_ptr[d][i]:cu_ptr[d][i+1]]  (0 ... 4096, locali alla riga)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import ctypes  # noqa: E402
import gc  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import costruisci_corpus_calibrazione as C  # noqa: E402  (v1, solo import)
import costruisci_corpus_calibrazione_v2 as V2  # noqa: E402  (v2, solo import)
import costruisci_corpus_calibrazione_v2_sem as S  # noqa: E402  (v2_sem, solo import)

# i moduli importati fissano 4 thread torch all'import: qui si torna al valore dell'ambiente (default 2)
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

L4 = 4096
DIR_E2E = C.DIR_CAL / "e2e"
DOMINI_E2E = tuple(d for d in C.DOMINI if d != "wiki")
TESTO_CONTINUO = ("edu", "it_web", "it_wiki")
ASSORBONO = ("edu", "it_web")               # chi riceve le sequenze che un dominio a scorta limitata non ha
DOMINI_R3 = ("it_chat", "it_pensiero", "strumenti")
GIRI_MAX = 8
N_EB = 4096                                  # righe di addestramento di E-B (4096 x 4096)
# 26/9: `--al-massimo` (corpus E-B intermedio): quote U dichiarate per --sequenze righe, ogni dominio alla sua
# capienza, e la mancanza si ACCETTA (dichiarata nel manifesto: quote_domini.train.mancano_sequenze) invece di
# fermarsi. Con --sequenze 4096 da' esattamente la «capienza_E-B» della proiezione. Default: si ferma, come prima.
ACCETTA_CAPIENZA = False
PAN_IDS = C.DIR_PAN / "pannello_ids.pt"
PANNELLI_HELDOUT_PER_DOMINIO = 8
RX_TURNO = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>", re.S)
RX_PENSIERO = re.compile(r"^\s*<think>(.*?)</think>(.*)$", re.S)
RX_CHIAMATA = re.compile(r"<tool_call>.*?</tool_call>", re.S)
RX_RISPOSTA_STRUMENTO = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
FILTRO_WIKI_IT = (lambda e: len(e.ids) >= C.WIKI_IT_MIN_TOKEN)          # come V2.costruisci_variante
FILTRO_RX_MAT = (lambda e: len(e.ids) <= V2.RX_MATEMATICA_MAX_TOKEN)    # come V2.costruisci_variante
FILTRO_NEMO = (lambda e: len(e.ids) <= V2.NEMOTRON_MAX_TOKEN)           # come V2.main (pannello)


def log(*a) -> None:
    V2.log(*a)


def rss_max_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


MEMORIA: list = []
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def memoria(fase: str) -> None:
    """RSS a fine fase (dopo gc e malloc_trim) e picco dall'avvio (VmHWM, MAI azzerato: azzerarlo azzererebbe anche il
    massimo che legge /usr/bin/time). Il tetto di sessione e' 4 GiB per processo: si misura, non si stima."""
    st = Path("/proc/self/status").read_text()
    picco = int(re.search(r"VmHWM:\s+(\d+)", st).group(1)) / 2**20
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)
    st = Path("/proc/self/status").read_text()
    rss = int(re.search(r"VmRSS:\s+(\d+)", st).group(1)) / 2**20
    MEMORIA.append({"fase": fase, "picco_da_avvio_gib": round(picco, 3), "rss_dopo_gib": round(rss, 3)})
    log(f"  memoria {fase}: picco dall'avvio {picco:.2f} GiB, dopo {rss:.2f} GiB")


@contextlib.contextmanager
def lunghezza(n: int):
    """Porta a n la lunghezza di sequenza delle funzioni del v1 (`impacca_fonte`, `pool_edu`), poi la rimette."""
    vecchia = C.L
    C.L = n
    try:
        yield
    finally:
        C.L = vecchia


# ═══════════════════════════ quote ═══════════════════════════
def quote_u(variante: str) -> dict:
    """Miscela U della variante (da V2.MISCELE, non trascritta) senza wiki, nell'ordine di DOMINI_E2E."""
    u = C.parse_miscele(V2.MISCELE[variante])["U"]
    u.pop("wiki")
    if set(u) != set(DOMINI_E2E):
        raise RuntimeError(f"miscela U senza wiki: domini {sorted(u)}, attesi {sorted(DOMINI_E2E)}")
    return {d: u[d] for d in DOMINI_E2E}


def riempi(n: int, pesi: "list | tuple", capienza: dict) -> dict:
    """Quote intere proporzionali ai pesi (resti piu' grandi, `C.ripartisci`), nessuna oltre la capienza; cio' che
    una voce piena non puo' prendere va alle altre con gli stessi pesi. Somma < n solo se tutte sono piene."""
    q = {f: 0 for f, _ in pesi}
    attivi = [(f, w) for f, w in pesi if w > 0 and capienza[f] > 0]
    while True:
        resto = n - sum(q.values())
        if resto <= 0 or not attivi:
            return q
        parti = C.ripartisci(resto, tuple(attivi))
        pieni = {f for f, _ in attivi if q[f] + parti[f] >= capienza[f]}
        if not pieni:
            for f, v in parti.items():
                q[f] += v
            return q
        for f in pieni:
            q[f] = capienza[f]
        attivi = [(f, w) for f, w in attivi if f not in pieni]


def quote_domini(n: int, pesi: dict, capienza: dict) -> "tuple[dict, dict]":
    """Quote per dominio: `C.ripartisci` sulla miscela; un dominio sopra la sua capienza si ferma li' e il suo
    deficit va agli ASSORBONO (edu, it_web) con i loro pesi, fin dove hanno posto. Ritorna (quote, rapporto)."""
    base = C.ripartisci(n, tuple(pesi.items()))
    q = {d: min(base[d], capienza[d]) for d in pesi}
    deficit = {d: base[d] - q[d] for d in pesi if base[d] > q[d]}
    tot = sum(deficit.values())
    extra = riempi(tot, [(d, pesi[d]) for d in ASSORBONO], {d: capienza[d] - q[d] for d in ASSORBONO}) if tot else {}
    for d, v in extra.items():
        q[d] += v
    mancano = n - sum(q.values())
    return q, {"dichiarate": base, "effettive": dict(q), "capienza": dict(capienza), "deficit_per_dominio": deficit,
               "dato_a": extra, "mancano_sequenze": mancano}


# ═══════════════════════════ lunghezze, confini, ordine ═══════════════════════════
class Lunghezze:
    """Token per esempio (testo reso completo), in cache per (id, hash del testo): la tokenizzazione e' la stessa
    di `impacca_fonte` (`tk.ids` senza token speciali)."""

    def __init__(self, tk):
        self.tk, self.c = tk, {}

    def __call__(self, esempi: list) -> list:
        manca = [e for e in esempi if (e.id, hash(e.testo)) not in self.c]
        for i in range(0, len(manca), 128):
            pezzo = manca[i:i + 128]
            for e, ids in zip(pezzo, self.tk.ids([e.testo for e in pezzo])):
                self.c[(e.id, hash(e.testo))] = len(ids)
        return [self.c[(e.id, hash(e.testo))] for e in esempi]


def ammessi(esempi: list, filtro, lun: Lunghezze) -> "tuple[list, int]":
    """Esempi che passano il filtro di lunghezza del v2 (lo stesso lambda, su un oggetto con `ids` di quella
    lunghezza) e i loro token."""
    ls = lun(esempi)
    tenuti = [(e, n) for e, n in zip(esempi, ls) if filtro is None or filtro(SimpleNamespace(ids=range(n)))]
    return [e for e, _ in tenuti], sum(n for _, n in tenuti)


def confini(usati: list, seq: list, L: int) -> "tuple[list, list]":
    """cu_seqlens locali e `inizio_esempio` delle righe di UNA fonte impacchettata da `impacca_fonte`; verifica che
    il flusso degli esempi usati ricostruisca le righe AL BIT."""
    inizi, pos, flusso = set(), 0, []
    for e in usati:
        inizi.add(pos)
        pos += len(e.ids)
        flusso += e.ids
    cu, inizio = [], []
    for j, riga in enumerate(seq):
        if flusso[j * L:(j + 1) * L] != list(riga):
            raise RuntimeError(f"riga {j}: il flusso degli esempi usati non ricostruisce la sequenza")
        b = sorted(s - j * L for s in inizi if j * L < s < (j + 1) * L)
        cu.append([0] + b + [L])
        inizio.append(j * L in inizi)
    return cu, inizio


def confini_multi(seq: list, info: dict, usati: list, comp: tuple, L: int) -> "tuple[list, list]":
    """Come `confini` per l'uscita di `C.dominio_multi`: gli esempi usati sono concatenati per fonte nell'ordine
    della composizione; l'ordine delle righe si ricalcola con `C.interfoglia` stesso e si verifica al bit."""
    per, k = {}, 0
    for f, _ in comp:
        n = info["fonti"][f]["esempi_usati"]
        per[f] = usati[k:k + n]
        k += n
    if k != len(usati):
        raise RuntimeError("esempi usati non ripartibili per fonte")
    righe = {f: [seq_f for seq_f in range(info["fonti"][f]["sequenze"])] for f, _ in comp}
    ordine, _ = C.interfoglia([C.Pacco(f, [(f, j) for j in righe[f]], [], 0) for f, _ in comp])
    if len(ordine) != len(seq):
        raise RuntimeError("interfoglia: righe diverse da dominio_multi")
    per_fonte_seq = {f: [None] * len(righe[f]) for f, _ in comp}
    for pos, (f, j) in enumerate(ordine):
        per_fonte_seq[f][j] = seq[pos]
    cuf = {f: confini(per[f], per_fonte_seq[f], L) for f, _ in comp}
    cu, inizio = [], []
    for f, j in ordine:
        cu.append(cuf[f][0][j])
        inizio.append(cuf[f][1][j])
    return cu, inizio


def ordine_domini(n_per_dominio: dict) -> list:
    """Ordine d'uso delle righe: `C.interfoglia` sui domini, ogni prefisso con la composizione dichiarata."""
    pac = [C.Pacco(d, [(k, i) for i in range(n_per_dominio[d])], [], 0) for k, d in enumerate(DOMINI_E2E)]
    ordine, _ = C.interfoglia(pac)
    return ordine


# ═══════════════════════════ riferimenti per l'held-out ═══════════════════════════
def messaggi_da_testo(testo: str) -> set:
    """Messaggi normalizzati di un testo col template di Thor: turni utente (e risposte degli strumenti dentro),
    ragionamento e risposta dell'assistente (senza i blocchi <tool_call>). Il system NON conta: e' anche testo
    del template (blocco <tools>)."""
    out = set()
    for ruolo, t in RX_TURNO.findall(testo):
        parti = []
        if ruolo == "assistant":
            m = RX_PENSIERO.match(t)
            parti = [m.group(1), m.group(2)] if m else [t]
            parti = [RX_CHIAMATA.sub(" ", p) for p in parti]
        elif ruolo in ("user", "tool"):
            dentro = RX_RISPOSTA_STRUMENTO.findall(t)
            parti = dentro if dentro else [t]
        out |= {C.norm(p) for p in parti}
    out.discard("")
    return out


def _espandi(rle: list) -> list:
    return [x for x, n in rle for _ in range(n)]


def riferimenti_forgia(tk) -> dict:
    """Finestre di 32 token delle sequenze INTERE e messaggi (per fonte, flusso ricostruito dalle etichette) di tutti
    i corpus di calibrazione della forgia in `calibrazione/`: l'held-out non deve contenere nulla di cio'."""
    finestre, messaggi, file = [], set(), []
    for p in sorted(C.DIR_CAL.glob("calibrazione_corpus*.pt")):
        m = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        sha = C.sha256_file(p)
        if sha != m["sha256_file"]:
            raise RuntimeError(f"{p.name}: sha256 {sha} diverso dal suo manifesto")
        t = torch.load(p, weights_only=True)
        for d, x in t.items():
            finestre.append(C.ngrammi_sequenze(x))
            if d in TESTO_CONTINUO or d == "wiki":
                continue
            et = _espandi(m["etichette_per_sequenza"][d])
            if len(et) != x.shape[0]:
                raise RuntimeError(f"{p.name}:{d}: etichette {len(et)} per {x.shape[0]} righe")
            for f in dict.fromkeys(et):
                righe = [i for i, e in enumerate(et) if e == f]
                messaggi |= messaggi_da_testo(tk.decodifica(x[righe].reshape(-1).tolist()))
        file.append({"file": p.name, "sha256": sha, "righe": {d: int(x.shape[0]) for d, x in t.items()}})
    return {"finestre": np.unique(np.concatenate(finestre)), "messaggi": messaggi, "file": file}


class Controlli:
    """H1-H5 su un esempio candidato all'held-out. Finestre e messaggi di ogni esempio in cache."""

    def __init__(self, tk, forgia: dict, sem_ho):
        self.tk, self.forgia, self.sem = tk, forgia, sem_ho
        self._ngr, self.train = {}, None

    def ngrammi(self, e) -> np.ndarray:
        k = (e.id, hash(e.testo))
        if k not in self._ngr:
            self._ngr[k] = C.ngrammi_testi(self.tk, e.contenuti)
        return self._ngr[k]

    @staticmethod
    def messaggi(e) -> set:
        massive = e.fonte.startswith("massive")
        return {n for n in (C.norm(c) for c in e.contenuti) if n and (massive or len(n) >= C.MIN_CAR_IDENTICI)}

    def forgia_colpisce(self, e) -> "str | None":
        if C.dentro(self.ngrammi(e), self.forgia["finestre"]).any():
            return "H5_finestre_forgia"
        if self.messaggi(e) & self.forgia["messaggi"]:
            return "H5_messaggi_forgia"
        return None

    def imposta_train(self, usati: dict) -> None:
        tutti = [e for us in usati.values() for e in us]
        fin = [self.ngrammi(e) for e in tutti]
        self.train = {
            "finestre": np.unique(np.concatenate(fin)) if fin else np.empty(0, dtype=np.uint64),
            "messaggi": set().union(*[self.messaggi(e) for e in tutti]),
            "cot": set().union(*[self.sem.chiavi_cot(e) for e in usati.get("cot", [])]),
            "pan": {"it_chat": [e.ids for e in usati.get("it_chat", [])],
                    "it_pensiero": [e.ids for e in usati.get("it_pensiero", [])],
                    "strumenti_it": [e.ids for e in usati.get("strumenti", [])], "strumenti_en": []},
        }
        self.sem.pan = None             # R3: indice dei turni utente ricostruito sull'addestramento attuale

    def train_colpisce(self, e, dominio: str) -> "str | None":
        t = self.train
        if C.dentro(self.ngrammi(e), t["finestre"]).any():
            return "H1_finestre_train"
        if self.messaggi(e) & t["messaggi"]:
            return "H2_messaggi_train"
        if dominio == "cot" and self.sem.chiavi_cot(e) & t["cot"]:
            return "H3_R2_cot_train"
        if dominio in DOMINI_R3:
            for u in S.utenti(e.testo):
                if self.sem.quasi_doppione(u, t["pan"]):
                    return "H4_R3_quasi_doppione_train"
        return None


# ═══════════════════════════ pannello v2 (ricostruito, 1024) ═══════════════════════════
def pannello(F, tk) -> dict:
    """`V2.pannello_v2` (codice v2, lunghezza 1024) + confronto AL BIT di tutti i domini col file del pannello v2."""
    if C.L != 1024:
        raise RuntimeError(f"C.L = {C.L}: il pannello si ricostruisce a 1024")
    pan, info_pan, usati_pan, fonti_pan = V2.pannello_v2(F, tk, FILTRO_WIKI_IT, FILTRO_NEMO)
    salvato = torch.load(V2.PAN_V2, weights_only=False)["domini"]
    for d, s in pan.items():
        t = torch.tensor(s, dtype=torch.int64).reshape(salvato[d].shape)
        if not torch.equal(t, salvato[d]):
            raise RuntimeError(f"pannello v2 {d}: la ricostruzione NON coincide al bit col file")
    log(f"pannello v2 ricostruito e IDENTICO al bit a {V2.PAN_V2.name}: {sorted(pan)}")
    return {"pannello": pan, "usati": usati_pan, "fonti": fonti_pan,
            "vietati_nemo": {n for e in usati_pan["strumenti_en"] for n in e.extra.get("strumenti", [])},
            "mfaq_valid_siti": {p["domain"] for p in C.jsonl(F.uscita("mfaq_it_valid"))},
            "frasi_pan_norm": {C.norm(u) for e in usati_pan["strumenti_it"] for u in e.contenuti}}


# ═══════════════════════════ un lato (held-out o addestramento) ═══════════════════════════
def composizione_dominio(variante: str, dom: str, fonti: dict) -> "tuple[tuple, dict]":
    """(fonte, peso) e liste di esempi per i domini impacchettati con `C.dominio_multi`."""
    if dom == "it_web":
        return (("fineweb2_train", 1),), {"fineweb2_train": fonti["it_web"]}
    if dom == "it_wiki":
        return (("wiki_it_rg0", 1),), {"wiki_it_rg0": fonti["it_wiki"]}
    comp = V2.COMPOSIZIONE[variante][dom]
    return comp, {f: fonti[f] for f, _ in comp}


FILTRI = {"wiki_it_rg0": FILTRO_WIKI_IT, "rx_matematica": FILTRO_RX_MAT}


class Lato:
    def __init__(self, nome: str):
        self.nome = nome
        self.seq, self.et, self.info, self.usati, self.cu, self.inizio = {}, {}, {}, {}, {}, {}
        self.quote: dict = {}

    def metti(self, d, seq, et, info, usati, cu, inizio):
        self.seq[d], self.et[d], self.info[d], self.usati[d], self.cu[d], self.inizio[d] = seq, et, info, usati, cu, inizio


@functools.lru_cache(maxsize=1)
def docs_edu() -> tuple:
    """Documenti FineWeb-Edu dal 173 come Esempio, con la stessa forma degli `usati` di `C.pool_edu`."""
    righe = [json.loads(r) for r in (C.DIR_PAN / C.EDU_FILE).open(encoding="utf-8")]
    return tuple(C.Esempio("fineweb_edu", f"edu:{i}", r["text"], [r["text"]], extra={"doc": i})
                 for i, r in enumerate(righe) if i >= C.EDU_PRIMO_DOC)


def cap_edu(tk, esclusi: set) -> int:
    """Righe da 4096 che il flusso edu (tutti i documenti non esclusi, come `pool_edu`) puo' dare."""
    _, info, _ = C.pool_edu(tk, 0, esclusi)
    return info["token_flusso_totale"] // C.L


def cot_ammessi(F, tk, classi, esclusi_idx: set) -> list:
    """Gli esempi che `V2.pool_cot_v2` ammetterebbe (stesso ordine), con la classe d'origine."""
    tutti = C.esempi_cot_tutti(tk)
    riservati = {i for r in C.COT_ESCLUSI for i in r}
    out = []
    for e in tutti:
        e.extra["classe_origine"] = F.classe_cot(e)
        i = e.extra["indice_mescolato"]
        if i in riservati or i in esclusi_idx:
            continue
        if classi is not None and e.extra["classe_origine"] not in classi:
            continue
        out.append(e)
    return out


def coda_minima(esempi: list, lun: Lunghezze, token: int) -> list:
    """Il piu' corto suffisso della lista (in ordine) con almeno `token` token."""
    ls, acc, k = lun(esempi), 0, len(esempi)
    while k > 0 and acc < token:
        k -= 1
        acc += ls[k]
    if acc < token:
        raise RuntimeError(f"coda: {acc} token, ne servono {token}")
    return esempi[k:]


def lato_heldout(variante, F, tk, fonti, n, pesi, esclusi, vietati_ho, lun, esclusi_edu, esclusi_cot) -> Lato:
    """Held-out dalla CODA di ogni fonte (liste rovesciate; edu: ultimi documenti; cot: indici piu' alti)."""
    H = Lato("heldout")
    classi = None if variante == "tutte" else V2.AMMESSE_SENZA_LLAMA
    no = lambda dom, lst: [e for e in lst if e.id not in esclusi.get(dom, set()) and e.id not in vietati_ho]  # noqa: E731
    # capienze
    cap, liste = {}, {}
    edu_ok = [e for e in docs_edu() if e.extra["doc"] not in esclusi_edu and e.id not in vietati_ho]
    tutti_doc = {e.extra["doc"] for e in docs_edu()}
    cap["edu"] = cap_edu(tk, tutti_doc - {e.extra["doc"] for e in edu_ok})
    cot_ok = [e for e in cot_ammessi(F, tk, classi, esclusi_cot) if e.id not in vietati_ho]
    cap["cot"] = sum(lun(cot_ok)) // C.L
    for dom in ("it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti"):
        comp, fd = composizione_dominio(variante, dom, fonti)
        liste[dom] = (comp, {f: ammessi(no(dom, fd[f]), FILTRI.get(f), lun) for f in fd})
        cap[dom] = sum(t // C.L for _, t in liste[dom][1].values())
    q, H.quote = quote_domini(n, pesi, cap)
    if H.quote["mancano_sequenze"]:
        raise RuntimeError(f"held-out: mancano {H.quote['mancano_sequenze']} righe ({H.quote})")
    # edu: i documenti di coda piu' corti che bastano, poi pool_edu su quelli soli (flusso unico come il v1)
    coda = coda_minima(edu_ok, lun, q["edu"] * C.L)
    while True:
        try:
            s, info, u = C.pool_edu(tk, q["edu"], tutti_doc - {e.extra["doc"] for e in coda})
            break
        except RuntimeError as err:
            if "ne servono" not in str(err) or len(coda) >= len(edu_ok):
                raise
            coda = edu_ok[-(len(coda) + 1):]
    info["selezione"] = {"documenti_di_coda_candidati": len(coda), "primo_doc": coda[0].extra["doc"],
                         "ultimo_doc": coda[-1].extra["doc"]}
    H.metti("edu", s, ["fineweb_edu"] * len(s), info, u, [[0, C.L]] * len(s), [False] * len(s))
    # cot: gli indici piu' alti che bastano
    coda = coda_minima(cot_ok, lun, q["cot"] * C.L) if q["cot"] else []
    fuori = {e.extra["indice_mescolato"] for e in C.esempi_cot_tutti(tk)} - {e.extra["indice_mescolato"] for e in coda}
    s, info, u = V2.pool_cot_v2(F, tk, q["cot"], fuori, classi) if q["cot"] else ([], {}, [])
    if q["cot"]:
        info["esclusi_per_contaminazione"] = sorted(esclusi_cot)
        info["selezione"] = {"coda_indici": [coda[0].extra["indice_mescolato"], coda[-1].extra["indice_mescolato"]],
                             "esempi_in_coda": len(coda)}
    cu, ini = confini(u, s, C.L)
    H.metti("cot", s, ["cot"] * len(s), info, u, cu, ini)
    for dom in ("it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti"):
        comp, per = liste[dom]
        capf = {f: per[f][1] // C.L for f in per}
        qf = riempi(q[dom], list(comp), capf)
        if sum(qf.values()) != q[dom]:
            raise RuntimeError(f"held-out {dom}: fonti piene {qf} per {q[dom]}")
        compq = tuple((f, qf[f]) for f, _ in comp)
        fd = {f: list(reversed(per[f][0])) for f in per}
        _impacca(H, dom, q[dom], fd, compq, comp, capf, tk)
    return H


def _impacca(lato: Lato, dom, n, fd, compq, comp, capf, tk) -> None:
    if n == 0:
        lato.metti(dom, [], [], {"quote_effettive": {f: 0 for f, _ in comp}}, [], [], [])
        return
    s, et, info, u = C.dominio_multi(dom, n, fd, compq, tk, FILTRI)
    cu, ini = confini_multi(s, info, u, compq, C.L)
    if dom in TESTO_CONTINUO:
        cu = [[0, C.L]] * len(s)
    info["pesi_composizione"] = dict(comp)
    info["capienza_sequenze_per_fonte"] = capf
    lato.metti(dom, s, et, info, u, cu, ini)


def lato_train(variante, F, tk, fonti, n, pesi, esclusi, ho: Lato, lun, esclusi_edu, esclusi_cot) -> "tuple[Lato, dict]":
    """Addestramento dalla TESTA di ogni fonte, senza gli esempi dell'held-out. Ritorna anche la capienza per E-B."""
    T = Lato("train")
    classi = None if variante == "tutte" else V2.AMMESSE_SENZA_LLAMA
    presi = {e.id for us in ho.usati.values() for e in us}
    no = lambda dom, lst: [e for e in lst if e.id not in esclusi.get(dom, set()) and e.id not in presi]  # noqa: E731
    cap, liste = {}, {}
    esc_edu = esclusi_edu | {e.extra["doc"] for e in ho.usati["edu"]}
    cap["edu"] = cap_edu(tk, esc_edu)
    esc_cot = esclusi_cot | {e.extra["indice_mescolato"] for e in ho.usati["cot"]}
    cap["cot"] = sum(lun(cot_ammessi(F, tk, classi, esc_cot))) // C.L
    for dom in ("it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti"):
        comp, fd = composizione_dominio(variante, dom, fonti)
        liste[dom] = (comp, {f: ammessi(no(dom, fd[f]), FILTRI.get(f), lun) for f in fd})
        cap[dom] = sum(t // C.L for _, t in liste[dom][1].values())
    q, T.quote = quote_domini(n, pesi, cap)
    _, proiezione = quote_domini(N_EB, pesi, cap)        # E-B: 4096 righe con le stesse fonti
    if T.quote["mancano_sequenze"] and ACCETTA_CAPIENZA:
        T.quote["mancanza_accettata"] = True
        log(f"  addestramento AL MASSIMO: {sum(q.values())} righe invece di {n} (mancano {T.quote['mancano_sequenze']}, "
            f"accettato con --al-massimo) {q}")
    elif T.quote["mancano_sequenze"]:
        raise RuntimeError(f"addestramento: mancano {T.quote['mancano_sequenze']} righe ({T.quote}); servono sorgenti "
                           "nuove (decisione di Angelo, progetto §5.3)")
    s, info, u = C.pool_edu(tk, q["edu"], esc_edu)
    T.metti("edu", s, ["fineweb_edu"] * len(s), info, u, [[0, C.L]] * len(s), [False] * len(s))
    s, info, u = V2.pool_cot_v2(F, tk, q["cot"], esc_cot, classi)
    info["esclusi_per_contaminazione"] = sorted(esclusi_cot)
    cu, ini = confini(u, s, C.L)
    T.metti("cot", s, ["cot"] * len(s), info, u, cu, ini)
    for dom in ("it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti"):
        comp, per = liste[dom]
        capf = {f: per[f][1] // C.L for f in per}
        qf = riempi(q[dom], list(comp), capf)
        if sum(qf.values()) != q[dom]:
            raise RuntimeError(f"addestramento {dom}: fonti piene {qf} per {q[dom]}")
        compq = tuple((f, qf[f]) for f, _ in comp)
        _impacca(T, dom, q[dom], {f: per[f][0] for f in per}, compq, comp, capf, tk)
    return T, proiezione


# ═══════════════════════════ una variante ═══════════════════════════
def candidati_heldout(variante, F, tk, fonti, esclusi, esclusi_edu, esclusi_cot) -> dict:
    """Tutti gli esempi che potrebbero entrare nell'held-out, {dominio: {fonte: lista in ordine}} (controlli H1-H5)."""
    classi = None if variante == "tutte" else V2.AMMESSE_SENZA_LLAMA
    out = {"edu": {"fineweb_edu": [e for e in docs_edu() if e.extra["doc"] not in esclusi_edu]},
           "cot": {"cot": cot_ammessi(F, tk, classi, esclusi_cot)}}
    for dom in ("it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti"):
        _, fd = composizione_dominio(variante, dom, fonti)
        out[dom] = {f: [e for e in lst if e.id not in esclusi.get(dom, set())] for f, lst in fd.items()}
    return out


def costruisci_variante(variante: str, F, tk, pan: dict, forgia: dict, n_tr: int, n_ho: int, lun: Lunghezze,
                        ctl: Controlli) -> dict:
    t0 = time.time()
    pesi = quote_u(variante)
    esclusi: dict = {}
    esclusi_edu: set = set()
    esclusi_cot: set = set()
    vietati_forgia: dict = {}          # id -> motivo (H5), calcolato su TUTTI i candidati, una volta per giro
    vietati_train: dict = {}           # id -> motivo (H1-H4), accumulato: la lista cresce e basta
    giri = []
    log(f"══ variante {variante}: {n_tr} + {n_ho} righe da {L4}, quote U senza wiki {pesi} ══")
    for giro in range(1, GIRI_MAX + 1):
        fonti, info_fonti = V2.fonti_calibrazione(F, variante, tk, pan["vietati_nemo"], esclusi.get("strumenti", set()),
                                                  pan["mfaq_valid_siti"], pan["frasi_pan_norm"])
        memoria(f"{variante} giro {giro}: fonti")
        cand = candidati_heldout(variante, F, tk, fonti, esclusi, esclusi_edu, esclusi_cot)
        for per in cand.values():
            for e in (x for lst in per.values() for x in lst):
                if e.id not in vietati_forgia:
                    m = ctl.forgia_colpisce(e)
                    if m:
                        vietati_forgia[e.id] = m
        vietati_ho = set(vietati_forgia) | set(vietati_train)
        memoria(f"{variante} giro {giro}: candidati e H5")
        with lunghezza(L4):
            ho = lato_heldout(variante, F, tk, fonti, n_ho, pesi, esclusi, vietati_ho, lun, esclusi_edu, esclusi_cot)
            memoria(f"{variante} giro {giro}: held-out")
            tr, proiezione = lato_train(variante, F, tk, fonti, n_tr, pesi, esclusi, ho, lun, esclusi_edu, esclusi_cot)
            memoria(f"{variante} giro {giro}: train")
        # disgiunzione del corpus intero (addestramento + held-out) dai riferimenti: verifica_sem del v2_sem
        corpus = {"wiki": torch.empty((0, L4), dtype=torch.int64)}
        usati = {"wiki": []}
        for d in DOMINI_E2E:
            corpus[d] = torch.tensor(tr.seq[d] + ho.seq[d], dtype=torch.int64).reshape(-1, L4)
            usati[d] = tr.usati[d] + ho.usati[d]
        ver, contaminati = S.verifica_sem(tk, f"e2e_{variante}", corpus, usati, pan["pannello"], pan["usati"], fonti,
                                          pan["fonti"])
        memoria(f"{variante} giro {giro}: verifica_sem")
        # held-out contro addestramento: TUTTI i candidati (cosi' il giro dopo sceglie gia' fra quelli puliti)
        ctl.imposta_train(tr.usati)
        nel_train = {e.id for us in tr.usati.values() for e in us}
        nuovi = {}
        for dom, per in cand.items():
            for e in (x for lst in per.values() for x in lst):
                if e.id in vietati_train or e.id in vietati_forgia or e.id in nel_train:
                    continue
                m = ctl.train_colpisce(e, dom)
                if m:
                    nuovi[e.id] = m
        scelti = {e.id for us in ho.usati.values() for e in us}
        colpiti_scelti = {i: m for i, m in nuovi.items() if i in scelti}
        forgia_scelti = [i for i in scelti if i in vietati_forgia]
        if forgia_scelti:
            raise RuntimeError(f"held-out con esempi H5 vietati: {forgia_scelti[:5]}")
        conta = lambda dd: {m: sum(1 for x in dd.values() if x == m) for m in sorted(set(dd.values()))}  # noqa: E731
        giri.append({"giro": giro, "contaminati_tolti": {d: sorted(v) for d, v in contaminati.items()},
                     "heldout_candidati_vietati_forgia": conta(vietati_forgia),
                     "heldout_candidati_in_conflitto_col_train_nuovi": conta(nuovi),
                     "heldout_scelti_in_conflitto": colpiti_scelti,
                     "esito_verifica": ver["esito"]})
        log(f"  giro {giro}: contaminati {({d: len(v) for d, v in contaminati.items()}) or 0} · held-out scelti in "
            f"conflitto col train {len(colpiti_scelti)} · candidati vietati forgia {len(vietati_forgia)}, "
            f"train {len(vietati_train) + len(nuovi)} · RSS max {rss_max_gib():.2f} GiB")
        vietati_train.update(nuovi)
        memoria(f"{variante} giro {giro}: H1-H4")
        if not contaminati and not colpiti_scelti:
            break
        for d, v in contaminati.items():
            if d == "edu":
                esclusi_edu |= {int(x.split(":")[1]) for x in v}
            elif d == "cot":
                esclusi_cot |= {int(x.split("#")[1]) for x in v}
            else:
                esclusi.setdefault(d, set()).update(v)
    else:
        raise RuntimeError(f"dopo {GIRI_MAX} giri il corpus non e' ancora pulito: da esaminare")
    return {"variante": variante, "train": tr, "heldout": ho, "ver": ver, "giri": giri, "pesi": pesi,
            "proiezione_EB": proiezione, "info_fonti": info_fonti, "fonti": fonti, "cand": cand,
            "vietati_forgia": vietati_forgia, "vietati_train": vietati_train, "secondi": round(time.time() - t0, 1)}


# ═══════════════════════════ salvataggio ═══════════════════════════
def tensori_lato(lato: Lato, variante: str) -> dict:
    dom, ptr, cu, ini, fonti, esempi, n = {}, {}, {}, {}, {}, {}, {}
    for d in DOMINI_E2E:
        dom[d] = torch.tensor(lato.seq[d], dtype=torch.int64).reshape(-1, L4)
        n[d] = int(dom[d].shape[0])
        lens = [len(c) for c in lato.cu[d]]
        ptr[d] = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int64)
        cu[d] = torch.tensor([x for c in lato.cu[d] for x in c], dtype=torch.int32)
        ini[d] = torch.tensor(lato.inizio[d], dtype=torch.bool)
        fonti[d] = list(lato.et[d])
        esempi[d] = [e.id for e in lato.usati[d]]
    ordine = torch.tensor(ordine_domini(n), dtype=torch.int64).reshape(-1, 2)
    return {"l_seq": L4, "lato": lato.nome, "variante": variante, "nomi_domini": list(DOMINI_E2E), "domini": dom,
            "cu_ptr": ptr, "cu": cu, "inizio_esempio": ini, "fonti": fonti, "esempi": esempi, "ordine": ordine}


def misure_lato(lato: Lato, t: dict, F) -> dict:
    out = {}
    for d in DOMINI_E2E:
        x = t["domini"][d]
        seg = np.diff(t["cu"][d].numpy().astype(np.int64))
        seg = seg[seg > 0]
        info = {k: v for k, v in lato.info[d].items() if k not in ("fonti",)}
        out[d] = {"righe": int(x.shape[0]), "token": int(x.numel()), "sha256_token": C.sha_tensore(x),
                  "esempi_usati": len(lato.usati[d]), "token_esempi_usati": sum(len(e.ids) for e in lato.usati[d])
                  if d != "edu" else None,
                  "segmenti": int(seg.size), "segmento_medio_token": round(float(seg.mean()), 1) if seg.size else None,
                  "righe_che_iniziano_a_meta_esempio": int((~t["inizio_esempio"][d]).sum())
                  if d not in TESTO_CONTINUO else None,
                  "fonti_righe": {f: n for f, n in C._rle(sorted(t["fonti"][d]))},
                  "dettaglio": {**info, "fonti": lato.info[d].get("fonti")}}
    return out


def posizioni_heldout(r: dict) -> dict:
    """Dove sta l'held-out nelle liste delle fonti (coda) e dove l'addestramento (testa), per fonte."""
    out = {}
    for dom, per in r["cand"].items():
        for f, lst in per.items():
            pos = {e.id: i for i, e in enumerate(lst)}
            h = [pos[e.id] for e in r["heldout"].usati[dom] if e.id in pos]
            t = [pos[e.id] for e in r["train"].usati[dom] if e.id in pos]
            out[f"{dom}:{f}"] = {"candidati": len(lst), "heldout_esempi": len(h), "train_esempi": len(t),
                                 "heldout_posizioni": [min(h), max(h)] if h else None,
                                 "train_posizioni": [min(t), max(t)] if t else None}
    return out


def disgiunzione_train_heldout(r: dict, ctl: Controlli) -> dict:
    """Misura finale (sull'esito, non sul ciclo): id, finestre, messaggi, R2, R3 fra i due lati."""
    tr, ho = r["train"], r["heldout"]
    id_tr = {e.id for us in tr.usati.values() for e in us}
    id_ho = {e.id for us in ho.usati.values() for e in us}
    ctl.imposta_train(tr.usati)
    colpi = {}
    for dom, us in ho.usati.items():
        for e in us:
            m = ctl.train_colpisce(e, dom)
            if m:
                colpi.setdefault(m, []).append(e.id)
    fin_ho = np.unique(np.concatenate([ctl.ngrammi(e) for us in ho.usati.values() for e in us]))
    return {"id_in_comune": len(id_tr & id_ho), "finestre_32_contenuto_heldout": int(fin_ho.size),
            "finestre_32_contenuto_train": int(ctl.train["finestre"].size),
            "finestre_32_in_comune": int(C.dentro(fin_ho, ctl.train["finestre"]).sum()),
            "esempi_heldout_colpiti_H1_H4": {k: len(v) for k, v in colpi.items()}, "dettaglio": colpi,
            "esempi_heldout_H5_forgia": sum(1 for us in ho.usati.values() for e in us if ctl.forgia_colpisce(e))}


def scrivi_pt(obj: dict, p: Path, sovrascrivi: bool) -> str:
    tmp = p.with_name(p.name + ".tmp")
    torch.save(obj, tmp)
    nuovo = C.sha256_file(tmp)
    if p.exists() and C.sha256_file(p) != nuovo and not sovrascrivi:
        tmp.unlink()
        raise RuntimeError(f"{p} esiste e differisce: --sovrascrivi per riscriverlo")
    os.replace(tmp, p)
    return nuovo


def pannelli_heldout(uscita: Path, sovrascrivi: bool) -> dict:
    """Le prime 8 sequenze (ordine batch, riga) di ogni dominio dei pannelli certificati."""
    dom, sorg = {}, {}
    for p in (PAN_IDS, V2.PAN_V2):
        x = torch.load(p, weights_only=False)["domini"]
        sorg[p.name] = C.sha256_file(p)
        for d, t in x.items():
            chiave = d if d not in dom else f"{p.stem}:{d}"
            dom[chiave] = t.reshape(-1, t.shape[-1])[:PANNELLI_HELDOUT_PER_DOMINIO].clone()
    obj = {"domini": dom, "sorgenti_sha256": sorg, "regola": f"prime {PANNELLI_HELDOUT_PER_DOMINIO} righe "
           "(ordine batch, riga) di ogni dominio"}
    p = uscita / "e2e_heldout_pannelli.pt"
    sha = scrivi_pt(obj, p, sovrascrivi)
    return {"file": p.name, "sha256": sha, "sorgenti_sha256": sorg, "regola": obj["regola"],
            "domini": {d: {"forma": list(t.shape), "sha256_token": C.sha_tensore(t)} for d, t in dom.items()}}


def provenienza() -> dict:
    tok = {f: C.sha256_file(C.SRC / f) for f in ("tokenizer.json", "tokenizer_config.json") if (C.SRC / f).exists()}
    codice = {Path(m.__file__).name: C.sha256_file(Path(m.__file__)) for m in (C, V2, S)}
    codice[Path(__file__).name] = C.sha256_file(Path(__file__))
    man = json.loads(V2.MAN_SORG_V2.read_text())
    locali = {}
    for k, v in man["locali"].items():
        p = Path(v["percorso"])
        sha = C.sha256_file(p)
        if sha != v["sha256"]:
            raise RuntimeError(f"sorgente locale {k}: {p} ha sha256 {sha}, il manifesto v2 dice {v['sha256']}")
        locali[k] = {"percorso": str(p), "sha256": sha}
    v2sem = {p.name: C.sha256_file(p) for p in S.OUT_PT.values()}
    return {"tokenizzatore": str(C.SRC), "sha256_tokenizzatore": tok, "codice": codice,
            "manifesto_sorgenti_v2": {"file": str(V2.MAN_SORG_V2), "sha256": C.sha256_file(V2.MAN_SORG_V2)},
            "sorgenti_locali_verificate": locali, "corpus_v2_sem_di_riferimento": v2sem,
            "verdetto": {"wiki.test.raw": C.sha256_file(C.FILE_WIKI_TEST), "mmlu_test.bin": C.sha256_file(C.FILE_MMLU)},
            "originali_metamathqa": {"file": str(S.ORIGINALI), "sha256": C.sha256_file(S.ORIGINALI)}}


def salva_variante(r: dict, F, uscita: Path, forgia: dict, ctl: Controlli, prov: dict, pan_ho: dict,
                   sovrascrivi: bool, argv: list) -> dict:
    v = r["variante"]
    tt, th = tensori_lato(r["train"], v), tensori_lato(r["heldout"], v)
    p_tr, p_ho = uscita / f"e2e_corpus_4096_{v}.pt", uscita / f"e2e_heldout_4096_{v}.pt"
    sha_tr, sha_ho = scrivi_pt(tt, p_tr, sovrascrivi), scrivi_pt(th, p_ho, sovrascrivi)
    comp_tr = V2.composizione({"etichette": {d: tt["fonti"][d] for d in DOMINI_E2E}}, F)
    comp_ho = V2.composizione({"etichette": {d: th["fonti"][d] for d in DOMINI_E2E}}, F)
    man = {
        "quando": time.strftime("%Y-%m-%d %H:%M:%S"), "variante": v, "l_seq": L4, "seme": C.SEME,
        "comando": " ".join(argv),
        "file": {"train": {"file": p_tr.name, "sha256": sha_tr, "byte": p_tr.stat().st_size,
                           "righe": int(sum(t.shape[0] for t in tt["domini"].values())),
                           "token": int(sum(t.numel() for t in tt["domini"].values()))},
                 "heldout": {"file": p_ho.name, "sha256": sha_ho, "byte": p_ho.stat().st_size,
                             "righe": int(sum(t.shape[0] for t in th["domini"].values())),
                             "token": int(sum(t.numel() for t in th["domini"].values()))},
                 "heldout_pannelli": pan_ho},
        "miscela": {"nome": "U senza wiki", "pesi": r["pesi"], "fonte": "V2.MISCELE (U) senza la voce wiki"},
        "quote_domini": {"train": r["train"].quote, "heldout": r["heldout"].quote},
        "capienza_E-B": r["proiezione_EB"],
        "composizione_per_fonte_e_origine": {"train": comp_tr, "heldout": comp_ho},
        "domini": {"train": misure_lato(r["train"], tt, F), "heldout": misure_lato(r["heldout"], th, F)},
        "posizioni_nelle_fonti": posizioni_heldout(r),
        "disgiunzione_corpus_vs_riferimenti": r["ver"],
        "disgiunzione_train_vs_heldout": disgiunzione_train_heldout(r, ctl),
        "heldout_vs_forgia": {"corpus_di_calibrazione": forgia["file"],
                              "finestre_di_riferimento": int(forgia["finestre"].size),
                              "messaggi_di_riferimento": len(forgia["messaggi"])},
        "candidati_heldout_vietati": {"H5_forgia": len(r["vietati_forgia"]), "H1_H4_train": len(r["vietati_train"])},
        "giri_di_pulizia": r["giri"], "semantica_registro": S.REGISTRO.get(f"e2e_{v}", []),
        "fonti_calibrazione": r["info_fonti"], "provenienza": prov,
        "uso": {"cu_seqlens": "riga i del dominio d: cu[d][cu_ptr[d][i]:cu_ptr[d][i+1]] (locali, 0..4096); testo "
                              "continuo (edu, it_web, it_wiki) = [0, 4096] come il protocollo PPL",
                "ordine": "(indice in nomi_domini, riga): batch di 8 righe consecutive nell'ordine",
                "caricamento": "torch.load(p, weights_only=True)"},
        "secondi": r["secondi"],
    }
    if v == "senza_llama":
        man["controllo_origine"] = {
            lato: V2.controlla_senza_llama({"etichette": {d: t["fonti"][d] for d in DOMINI_E2E},
                                            "usati_dom": {"cot": r[lato].usati["cot"]}}, F)
            for lato, t in (("train", tt), ("heldout", th))}
    man["rss_max_gib"] = round(rss_max_gib(), 3)
    man["memoria_per_fase"] = list(MEMORIA)
    (uscita / f"e2e_corpus_4096_{v}.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str),
                                                    encoding="utf-8")
    log(f"✅ {p_tr.name} ({sha_tr[:16]}…) · {p_ho.name} ({sha_ho[:16]}…) · RSS max {man['rss_max_gib']} GiB")
    return man


def leggimi(uscita: Path) -> None:
    """LEGGIMI dai manifesti di TUTTE le varianti presenti nella cartella (anche se costruite in corse diverse)."""
    manifesti = {p.stem.replace("e2e_corpus_4096_", ""): json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(uscita.glob("e2e_corpus_4096_*.json"))}
    r = ["# Corpus E2E a contesto 4096 (`thor_v2/e2e_corpus_4096.py`)", "",
         "Generato dai manifesti (nessun numero scritto a mano). Progetto: `thor_v2/e2e_progetto_scelto.md` §2.4.", "",
         "| variante | lato | file | sha256 | righe × 4096 | token |", "|---|---|---|---|---|---|"]
    for v, m in manifesti.items():
        for lato in ("train", "heldout"):
            f = m["file"][lato]
            r.append(f"| {v} | {lato} | `{f['file']}` | `{f['sha256']}` | {f['righe']} | {f['token']} |")
    for v, m in manifesti.items():
        r += ["", f"## {v}", "", "| dominio | train righe | held-out righe | capienza train (righe) | E-B: righe |",
              "|---|---|---|---|---|"]
        for d in DOMINI_E2E:
            r.append(f"| {d} | {m['domini']['train'][d]['righe']} | {m['domini']['heldout'][d]['righe']} | "
                     f"{m['quote_domini']['train']['capienza'][d]} | {m['capienza_E-B']['effettive'][d]} |")
        eb = m["capienza_E-B"]["mancano_sequenze"]
        r += ["", f"E-B (4096 righe): mancano **{eb}** righe con le fonti di oggi (servono sorgenti nuove: decisione "
              "di Angelo, progetto §5.3).", "",
              f"Disgiunzione dai riferimenti: {m['disgiunzione_corpus_vs_riferimenti']['esito']} · train/held-out: "
              f"finestre in comune {m['disgiunzione_train_vs_heldout']['finestre_32_in_comune']}, id in comune "
              f"{m['disgiunzione_train_vs_heldout']['id_in_comune']} · held-out vs forgia: "
              f"{m['disgiunzione_train_vs_heldout']['esempi_heldout_H5_forgia']} esempi."]
    pan = next(iter(manifesti.values()))["file"]["heldout_pannelli"]
    r += ["", f"Held-out dei pannelli: `{pan['file']}` (`{pan['sha256']}`), {pan['regola']}."]
    (uscita / "LEGGIMI.md").write_text("\n".join(r) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variante", default="entrambe", choices=("tutte", "senza_llama", "entrambe"))
    ap.add_argument("--sequenze", type=int, default=512, help="righe da 4096 di addestramento (E-A 512, E-B 4096)")
    ap.add_argument("--heldout", type=int, default=64)
    ap.add_argument("--uscita", default=str(DIR_E2E))
    ap.add_argument("--sovrascrivi", action="store_true")
    ap.add_argument("--al-massimo", action="store_true",
                    help="26/9: quote U per --sequenze righe, domini alla capienza, mancanza accettata e dichiarata")
    ap.add_argument("--giri-max", type=int, default=None,
                    help="26/9: giri massimi di pulizia (E-B a 2112 righe: conflitti held-out/train 45,14,9,8,6,4,4,2 in 8; default 8)")
    a = ap.parse_args()
    global ACCETTA_CAPIENZA, GIRI_MAX
    ACCETTA_CAPIENZA = a.al_massimo
    if a.giri_max is not None:
        GIRI_MAX = a.giri_max
    t0 = time.time()
    uscita = Path(a.uscita)
    if str(uscita.resolve()).startswith("/tmp/") and "/scratchpad/" not in str(uscita):
        raise SystemExit("⛔ /tmp e' RAM: uscita su disco")
    if not S.ORIGINALI.exists():
        raise SystemExit(f"⛔ {S.ORIGINALI} manca: il v2_sem lo scaricherebbe; qui non si scarica nulla")
    uscita.mkdir(parents=True, exist_ok=True)
    memoria("avvio")
    prov = provenienza()
    tk = C.Tok()
    F = V2.FontiV2(tk, 0.30)
    memoria("tokenizzatore e fonti")
    pan = pannello(F, tk)
    memoria("pannello v2")
    forgia = riferimenti_forgia(tk)
    memoria("riferimenti della forgia")
    log(f"riferimenti della forgia: {len(forgia['file'])} corpus, {forgia['finestre'].size} finestre, "
        f"{len(forgia['messaggi'])} messaggi · RSS max {rss_max_gib():.2f} GiB")
    pan_ho = pannelli_heldout(uscita, a.sovrascrivi)
    lun = Lunghezze(tk)
    ctl = Controlli(tk, forgia, S.Semantica(tk))
    varianti = ("tutte", "senza_llama") if a.variante == "entrambe" else (a.variante,)
    manifesti = {}
    for v in varianti:
        r = costruisci_variante(v, F, tk, pan, forgia, a.sequenze, a.heldout, lun, ctl)
        manifesti[v] = salva_variante(r, F, uscita, forgia, ctl, prov, pan_ho, a.sovrascrivi, sys.argv)
        del r
        memoria(f"{v}: salvataggio")
    leggimi(uscita)
    log(f"✅ fatto in {time.time() - t0:.0f} s · RSS max {rss_max_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
