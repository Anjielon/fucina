#!/usr/bin/env python3
"""Corpus di calibrazione v2 di Thor, in DUE varianti, e pannello dei domini nuovi v2.
25/9/2026, notte: chiude i difetti D2, D4, D5, D6 della verifica indipendente del corpus v1.

## Cosa cambia rispetto a `costruisci_corpus_calibrazione.py` (v1, NON modificata: la si importa)

- **D2, strumenti_it del pannello davvero fuori.** In v1 gli schemi MASSIVE erano gli stessi
  in calibrazione e nel pannello: l'89,2% dei token del pannello `strumenti_it` stava dentro
  finestre di 32 token gia' viste in calibrazione. Ora gli SCENARI sono separati: audio,
  cooking, recommendation, social, takeaway, transport (15 funzioni) vivono SOLO nel pannello,
  con frasi di dev e test di MASSIVE 1.1 (anche le 113 di questi scenari nel test ufficiale di
  MASSIVE-Agents); le altre 12 (40 funzioni) SOLO in calibrazione, frasi di train. I
  distrattori si pescano dal proprio lato. Gli altri cinque domini del pannello restano
  IDENTICI al bit a v1 (ricostruiti con il codice v1 e confrontati). Dopo la cura: 13,8% dei token
  coperti con la variante «tutte», 13,3% con la «senza Llama», e il residuo e' SOLO testo fisso del
  template degli strumenti (preambolo, istruzioni di formato, marcatori di turno), come il 17,5% di
  strumenti_en.
- **D5, ragionamento in Nemotron assegnato DOPO il filtro di lunghezza.** In v1 la regola «un
  esempio su 3» veniva prima del taglio a 2048 token, che scartava quasi tutti gli esempi col
  ragionamento: 9 su 64 (14%). Ora il posto k-esimo (k = 2 mod 3) prende il prossimo esempio
  la cui versione CON ragionamento sta sotto 2048 token (99 su 1000 righe lette), gli altri il
  prossimo sotto 2048 senza ragionamento (434 su 1000). Esclusi gli esempi con strumenti del
  pannello strumenti_en (il pannello e' fisso, la calibrazione li evita).
- **D6, piu' italiano negli esempi MASSIVE.** k = 16 frasi per conversazione e 1 distrattore
  (v1: 4 e 3). Misurato sulle conversazioni usate: frasi + valori degli argomenti dal 3,4% (v1,
  ricostruito al bit) al 10,8% dei token; frasi utente per ruolo dal 2,7% all'8,3%. Griglia di
  prova (120 conversazioni): con k=24 e 0 distrattori si arriva al 13,8%, a prezzo di conversazioni
  fino a 2.815 token e senza scelta fra strumenti. Oltre non si va senza inventare risposte degli
  strumenti, che la fonte non ha.
- **Filtro di qualita' MFAQ** (train, entrambe le varianti): la variante senza Llama prende da MFAQ
  135 sequenze invece di 35 e oltre i primi ~80 esempi comparivano pagine non italiane, listini di
  prezzi per citta', CSS, cirillico codificato male, un sito per adulti: 66 conversazioni su 847
  scartate (`mfaq_scarto`). Nel v1 ne era entrata una sola (listino).
- **D4, licenze e origine corrette** (manifesto `sorgenti/manifesto_sorgenti_v2.json`, scritto
  da `scarica_corpus_calibrazione_v2.py`): ogni fonte ha classe `umana`, `sintetica_pulita`,
  `llama_gpt` o `non_dichiarata`, con la prova (scheda/paper alla revisione fissata).
- **Due varianti**, stesse quote per dominio della miscela U:
  - `calibrazione_corpus_v2.pt` («tutte»): le fonti della v1;
  - `calibrazione_corpus_v2_senza_llama.pt`: ESCLUSE le fonti `llama_gpt` e `non_dichiarata`:
    smoltalk2 (con e senza ragionamento), ReasonXL-smoltalk, Nemotron (prompt non dichiarati),
    domande cot riscritte da GPT-3.5 (MetaMathQA Rephrased/SV), FOBAR (esecutore del
    mascheramento non dichiarato) e OpenCodeInstruct (Instruct-LLM non nominato). Al loro
    posto: it_chat = Aya + OASST2 + MFAQ; it_pensiero = ReasonXL con catena pulita (QASC e
    FLAN v2 generale, OpenMathReasoning matematica; DeepSeek-R1, Qwen3-32B); strumenti =
    MASSIVE it + MASSIVE en-US (frasi umane, schemi annotati da persone); cot = sole domande
    ORIGINALI di GSM8K/MATH (AnsAug), 150 sequenze (bastano per U, V, W; non per M2).

## Invarianti (verificati qui, esito nei manifesti)

- forma `{dominio: (n, 1024) int64}`, seme 1234, tokenizzatore di Thor; sha256 dei token di
  ogni dominio nel `.json` accanto (lo pretende `_carica_corpus_pronto`), `domini.cot.esclusi`
  = [[0, 8], [279, 385]] e `sorgenti.locali` (li legge `_guardia_heldout_corpus`);
- 0 finestre di 32 token di CONTENUTO in comune con: pannello v2 (tutti i domini), pannello
  storico, wiki.test.raw, mmlu_test.bin; gli esempi che ne hanno vengono tolti e il dominio
  ricostruito (fino a 3 giri);
- 0 identificativi, prompt normalizzati, frasi MASSIVE, scenari e funzioni MASSIVE in comune
  col pannello; 0 strumenti Nemotron del pannello strumenti_en in calibrazione;
- misura D2: quota di token di ogni dominio del pannello dentro finestre di 32 token (sequenze
  intere, template compreso) presenti nella calibrazione.

## Uso (solo CPU, nice, 4 thread)

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \\
        nice -n 19 @@ROOT:venv@@/bin/python3 costruisci_corpus_calibrazione_v2.py [--variante tutte|senza_llama|entrambe]

Non scrive MAI `calibrazione_corpus.pt`, `pannello_nuovi_domini.pt` ne' `pannello_ids.*`.
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
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costruisci_corpus_calibrazione as C  # noqa: E402  (v1, solo import)

torch.set_num_threads(4)

DIR_CAL, SORG, DIR_PAN = C.DIR_CAL, C.SORG, C.DIR_PAN
MAN_SORG_V2 = SORG / "manifesto_sorgenti_v2.json"
OUT_PT = {"tutte": DIR_CAL / "calibrazione_corpus_v2.pt",
          "senza_llama": DIR_CAL / "calibrazione_corpus_v2_senza_llama.pt"}
PAN_V1 = DIR_PAN / "pannello_nuovi_domini.pt"
PAN_V2 = DIR_PAN / "pannello_nuovi_domini_v2.pt"
PROTETTI = {"calibrazione_corpus.pt", "calibrazione_corpus.json", "pannello_nuovi_domini.pt",
            "pannello_nuovi_domini.json", "pannello_ids.pt", "pannello_ids.json"}

L, SEME = C.L, C.SEME
SCENARI_PANNELLO = ("audio", "cooking", "recommendation", "social", "takeaway", "transport")
K_MASSIVE = 16
DISTRATTORI = 1
NEMOTRON_MAX_TOKEN = C.NEMOTRON_MAX_TOKEN            # 2048, come v1
NEMOTRON_OGNI_PENSIERO = 3                           # posto 2, 5, 8 ... = col ragionamento
RX_MATEMATICA_MAX_TOKEN = 4096                       # problemi OpenMathReasoning: mediana 6.250 token
MISCELE = {
    "tutte": C.MISCELE_DEFAULT,
    "senza_llama": ("U=wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2;"
                    "V=wiki:4,edu:3,cot:2,it_web:2,it_chat:3,strumenti:2;"
                    "W=wiki:2,edu:2,cot:2,it_web:3,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2"),
}
COMPOSIZIONE = {
    "tutte": {"it_chat": (("aya", 60), ("oasst2", 30), ("mfaq", 35), ("smoltalk_nothink", 100)),
              "it_pensiero": (("smoltalk_think", 40), ("reasonxl", 35)),
              "strumenti": (("massive", 60), ("nemotron", 90))},
    "senza_llama": {"it_chat": (("aya", 60), ("oasst2", 40), ("mfaq", 125)),
                    "it_pensiero": (("rx_generale", 55), ("rx_matematica", 20)),
                    "strumenti": (("massive", 60), ("massive_en", 90))},
}
# etichetta di fonte nel corpus -> fonte logica del manifesto delle sorgenti v2
FONTE_LOGICA = {"wikitext2_train": "wikitext", "fineweb_edu": "fineweb_edu", "fineweb2_train": "fineweb2_it",
                "wiki_it_rg0": "wikipedia_it", "aya": "aya", "oasst2": "oasst2", "mfaq": "mfaq",
                "smoltalk_nothink": "smoltalk2_nothink", "smoltalk_think": "smoltalk2_think",
                "reasonxl": "reasonxl_smoltalk", "rx_generale": "reasonxl_pulito", "rx_matematica": "reasonxl_pulito",
                "massive": "massive", "massive_en": "massive", "nemotron": "nemotron_tool"}
AMMESSE_SENZA_LLAMA = {"umana", "sintetica_pulita"}
# MFAQ train (calibrazione): la variante senza Llama ne prende 135 sequenze invece di 35, e oltre i primi ~80 esempi
# compaiono pagine che NON sono italiano (misurato 25/9 su 847 esempi: 18 in altre lingue, listini di citta' con
# prezzi, CSS, testo cirillico codificato male, un sito per adulti). Regola unica per le due varianti.
_VALUTA = re.compile(r"\d+[.,]\d{2}\s?(?:EUR|SEK|ZAR|USD|GBP|CHF|PLN|DKK|NOK)")
_SITI_ADULTI = ("xhamster", "porn", "xxx", "sex")
_LISTE_LINGUA = {
    "it": "il gli della delle degli dello che è sono anche più perché questo questa questi quello quella nel nella "
          "nei nelle alla alle agli non ma come cosa per del dei dal dalla sul sulla ci può puoi possono essere molto "
          "ogni dove quando tutti tutto loro stato fino senza dopo sempre già ancora oppure però qui ha hanno hai "
          "abbiamo siamo tuo tuoi vostro nostro",
    "en": "the and is are of to that with for this you it be your can will have not what how which from or if on by "
          "an do does we our they there",
    "es": "el los las que por para está también más pero como muy esto cuando es son y en se lo su sus puede usted "
          "hay qué cómo este esta",
    "fr": "le les des est pas vous avec pour dans qui et sur au aux ce cette je nous ont être mais ou où du votre vos "
          "peut",
    "de": "der die und nicht ist das mit sie ein eine auch für auf dem den wir ich sind zu von werden bei oder",
    "pt": "não você são uma do dos das também mais com isso muito em os seu sua ao pelo pela",
}
_QUANTE = {}
for _l in _LISTE_LINGUA.values():
    for _w in set(_l.split()):
        _QUANTE[_w] = _QUANTE.get(_w, 0) + 1
_ESCLUSIVE = {k: {w for w in v.split() if _QUANTE[w] == 1} for k, v in _LISTE_LINGUA.items()}


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _sha(t: torch.Tensor) -> str:
    return C.sha_tensore(t)


# ═══════════════════════════ fonti ═══════════════════════════
class FontiV2(C.Fonti):
    """Come C.Fonti ma col manifesto v2 (estratti v1 + nuovi), MASSIVE en-US e ReasonXL pulito."""

    def __init__(self, tk, soglia: float):
        self.tk, self.soglia, self.k = tk, soglia, K_MASSIVE
        self.man = json.loads(MAN_SORG_V2.read_text())
        for gruppo in ("estratti", "nuovi"):
            for k, v in self.man.get(gruppo, {}).items():
                for m in (v if isinstance(v, list) else [v]):
                    u = m.get("uscita")
                    if u and C.sha256_file(SORG / u) != m["sha256_scritto"]:
                        raise RuntimeError(f"{gruppo}/{k}: {u} non combacia col manifesto v2")
        self.schemi = C.schemi_massive()
        self.massive = {d["id"]: d for d in C.jsonl(SORG / "massive-1.1" / "it-IT.jsonl")}
        self.massive_en = {d["id"]: d for d in C.jsonl(SORG / "massive-1.1" / "en-US.jsonl")}
        self.origine_cot = {(r["tipo_cot"], r["sha256_q"]): r for r in C.jsonl(SORG / "cot_origine_domande.jsonl")}

    def uscita(self, chiave: str) -> Path:
        v = self.man["estratti"].get(chiave) or self.man["nuovi"][chiave]
        return SORG / v["uscita"]

    def classe(self, fonte_logica: str) -> str:
        return self.man["origini"][fonte_logica]["classe"]

    def classe_cot(self, e) -> str:
        tipo = e.fonte.split("_")[1]
        r = self.origine_cot[(tipo, hashlib.sha256(e.prompt.encode("utf-8")).hexdigest())]
        if not r["trovata"]:
            raise RuntimeError(f"domanda cot {e.id} non ritrovata nella fonte alla revisione fissata")
        if tipo == "code":
            return self.man["origini"]["cot_code"]["classe"]
        t = r["tipo_metamath"].split("_", 1)[1]            # GSM_AnsAug -> AnsAug
        return self.man["origini"]["cot_math"]["classi_per_tipo"][t]

    def mfaq(self, chiave: str, siti_esclusi: set) -> list:
        """Come v1, piu' il filtro di qualita' `mfaq_scarto` sul SOLO train (il pannello resta quello v1)."""
        tutti = super().mfaq(chiave, siti_esclusi)
        if chiave != "mfaq_it_train":
            return tutti
        self.scarti_mfaq: dict = {}
        out = []
        for e in tutti:
            m = mfaq_scarto(e)
            if m:
                self.scarti_mfaq[m] = self.scarti_mfaq.get(m, 0) + 1
            else:
                out.append(e)
        self.scarti_mfaq["tenuti"] = len(out)
        return out

    def reasonxl_pulito(self, ramo: str) -> list:
        out = []
        for r in C.jsonl(self.uscita("reasonxl_pulito_cal")):
            if r["ramo"] != ramo:
                continue
            prompt = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            if self.lato(prompt) != "cal":
                continue
            msgs, cont = C._msg_chat(r["messages"], True)
            nome = "rx_generale" if ramo == "generale" else "rx_matematica"
            out.append(C.Esempio(nome, f"reasonxl:{r['ds_uid']}", self.tk.chat(msgs), cont, prompt,
                                 {"ds_uid": r["ds_uid"], "dataset_name": r["dataset_name"], "source": r["source"],
                                  "pos": r["_pos"]}))
        random.Random(f"{SEME}:reasonxl_pulito:{ramo}").shuffle(out)
        return out


def _punti_lingua(t: str) -> dict:
    """Parole funzionali ESCLUSIVE di ogni lingua (una parola presente in due liste non conta per nessuna:
    «in», «la», «le», «un», «una», «del», «con», «e», «a» sono italiane ma anche di altre lingue, e col
    riconoscitore della v1 facevano passare per inglese testi italiani come «convertire il codice in denaro»)."""
    parole = re.findall(r"[a-zàèéìòóùâêôãõçñäöüß]+", t.lower())[:600]
    return {k: sum(p in s for p in parole) for k, s in _ESCLUSIVE.items()}


def mfaq_scarto(e) -> "str | None":
    """Motivo per cui un esempio MFAQ non e' testo italiano pulito, o None. Lingua: si scarta solo con prova
    positiva di un'altra lingua (>= 4 parole esclusive e piu' delle italiane sull'esempio; >= 5 e il doppio
    delle italiane su una coppia domanda/risposta), cosi' i nomi propri stranieri in un testo italiano
    («Logis Hôtel les Portes de la Vallée», «Las Vegas») non bastano a scartarlo."""
    t = "\n".join(e.contenuti)
    if any(k in e.extra["sito"] for k in _SITI_ADULTI):
        return "sito_per_adulti"
    p = _punti_lingua(t)
    it = p.pop("it")
    if max(p.values()) >= 4 and max(p.values()) > it:
        return "non_italiano"
    for q, a in zip(e.contenuti[0::2], e.contenuti[1::2]):
        pc = _punti_lingua(q + "\n" + a)
        itc = pc.pop("it")
        if max(pc.values()) >= 5 and max(pc.values()) >= 2 * itc:
            return "coppia_non_italiana"
    lettere = [c for c in t if c.isalpha()]
    if lettere and sum(1 for c in lettere if "\u00c0" <= c <= "\u00ff") / len(lettere) > 0.15:
        return "codifica_rotta"
    if ".css-" in t or "{margin" in t:
        return "codice_css"
    if len(_VALUTA.findall(t)) >= 20:
        return "listino_prezzi"
    return None


def frasi_massive(tabella: dict, partizioni: tuple, scenari: set, schemi: dict, lato_id=None,
                  escluse_norm: "set | None" = None) -> "tuple[list, dict]":
    """Frasi MASSIVE convertite in chiamate, dedup per frase normalizzata, filtri di partizione/scenario."""
    viste, frasi = set(), []
    conta = {"lette": 0, "non_convertibili": 0, "escluse_norm": 0, "doppie": 0, "fuori_lato_id": 0}
    for d in tabella.values():
        if d["partition"] not in partizioni or d["scenario"] not in scenari:
            continue
        conta["lette"] += 1
        if lato_id is not None and not lato_id(d["id"]):
            conta["fuori_lato_id"] += 1
            continue
        c = C.converti_massive(d, schemi)
        if c is None:
            conta["non_convertibili"] += 1
            continue
        n = C.norm(d["utt"])
        if escluse_norm and n in escluse_norm:
            conta["escluse_norm"] += 1
            continue
        if n in viste:
            conta["doppie"] += 1
            continue
        viste.add(n)
        frasi.append((d, c))
    conta["tenute"] = len(frasi)
    return frasi, conta


def conversazioni_massive(frasi: list, schemi_lato: dict, k: int, d: int, tag: str, tk) -> list:
    """Come C.conversazioni_massive, con k e distrattori ESPLICITI e schemi del SOLO proprio lato
    (i distrattori non possono venire dall'altro lato)."""
    per_scen: dict = {}
    for dd, (f, a) in frasi:
        per_scen.setdefault(dd["scenario"], []).append((dd, f, a))
    nomi = sorted(schemi_lato)
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
        scelti = propri + rng.sample(altri, d)
        rng.shuffle(scelti)
        msgs, cont, valori = [], [], []
        for dd, f, a in gruppo:
            msgs.append({"role": "user", "content": dd["utt"]})
            msgs.append({"role": "assistant", "content": "", "reasoning_content": "",
                         "tool_calls": [{"type": "function", "function": {"name": f, "arguments": a}}]})
            cont.append(dd["utt"])
            valori += [str(v) for v in a.values()]
        testo = tk.chat(msgs, [schemi_lato[n] for n in scelti])
        out.append(C.Esempio(f"massive_{tag}", f"massive:{tag}:" + ",".join(dd["id"] for dd, _, _ in gruppo), testo,
                             cont, gruppo[0][0]["utt"],
                             {"scenario": scen, "id_massive": [dd["id"] for dd, _, _ in gruppo], "strumenti": scelti,
                              "valori": valori}))
    return out


def lato_id_it(i: str) -> bool:
    """Ids di train MASSIVE: meta' per le conversazioni italiane, meta' per quelle inglesi (MASSIVE e'
    parallelo: stesso id = stessa frase tradotta; cosi' le due lingue non ripetono le stesse frasi)."""
    return C.secchio(f"massive-id:{i}") < 0.5


def nemotron_v2(F: FontiV2, vietati: set, esclusi: set, tk) -> "tuple[list, dict]":
    """D5: ragionamento assegnato DOPO il filtro di lunghezza. Stesso ordine mescolato della v1
    (seme '1234:nemotron_tool_cal'); il posto j (fra gli esempi accettati) vuole il ragionamento se
    j % 3 == 2 e prende il prossimo esempio non usato la cui versione CON ragionamento e' <= 2048
    token; gli altri posti il prossimo <= 2048 SENZA ragionamento. Esclusi strumenti vietati e id tolti."""
    righe = C.jsonl(F.uscita("nemotron_tool_cal"))
    random.Random(f"{SEME}:nemotron_tool_cal").shuffle(righe)
    cand = []
    for r in righe:
        e0 = C.esempio_nemotron(r, False, tk)
        if (vietati and set(e0.extra["strumenti"]) & vietati) or e0.id in esclusi:
            continue
        e1 = C.esempio_nemotron(r, True, tk)
        cand.append((e0, e1))
    lun = tk.ids([e.testo for c in cand for e in c])
    for i, (e0, e1) in enumerate(cand):
        e0.ids, e1.ids = lun[2 * i], lun[2 * i + 1]
    ok0 = [i for i, (e0, _) in enumerate(cand) if len(e0.ids) <= NEMOTRON_MAX_TOKEN]
    ok1 = [i for i, (_, e1) in enumerate(cand) if len(e1.ids) <= NEMOTRON_MAX_TOKEN]
    usati, out = set(), []
    p0 = p1 = 0
    while True:
        vuole = len(out) % NEMOTRON_OGNI_PENSIERO == NEMOTRON_OGNI_PENSIERO - 1
        lista, p = (ok1, p1) if vuole else (ok0, p0)
        while p < len(lista) and lista[p] in usati:
            p += 1
        if p >= len(lista):
            break
        i = lista[p]
        usati.add(i)
        out.append(cand[i][1] if vuole else cand[i][0])
        if vuole:
            p1 = p + 1
        else:
            p0 = p + 1
    info = {"righe": len(righe), "candidati_dopo_vietati_e_esclusi": len(cand),
            "senza_ragionamento_entro_2048": len(ok0), "con_ragionamento_entro_2048": len(ok1),
            "ordine_esempi": len(out)}
    return out, info


# ═══════════════════════════ pannello v2 ═══════════════════════════
def pannello_v2(F: FontiV2, tk, filtro_wiki_it, filtro_nemo) -> "tuple[dict, dict, dict, dict]":
    """Domini v1 (tranne strumenti_it) RICOSTRUITI con il codice v1 e confrontati al bit col file v1
    (servono i loro esempi per le verifiche di contenuto); strumenti_it NUOVO, scenari riservati."""
    v1 = torch.load(PAN_V1, weights_only=False)
    fonti_pan = {
        "fineweb2_test": F.docs_web("fineweb2_it_test"),
        "wiki_it_rg1": F.docs_wiki_it("wiki_it_pan"),
        "aya": [e for e in F.aya() if F.lato(e.prompt) == "pan"],
        "oasst2": F.oasst2("oasst2_it_valid"),
        "mfaq": F.mfaq("mfaq_it_valid", set()),
        "smoltalk_nothink": F.smoltalk("smoltalk2_nothink_pan", False, "pan"),
        "smoltalk_think": F.smoltalk("smoltalk2_think_pan", True, "pan"),
        "reasonxl": F.reasonxl("reasonxl_pan", "pan"),
    }
    # strumenti vietati dalla v1 (per riprodurre strumenti_en): gli esempi di calibrazione v1
    frasi_pan_v1 = {C.norm(F.massive[x["id"]]["utt"]) for x in C.jsonl(SORG / "massive_agents_it-IT_input.jsonl")}
    vecchio_d = C.DISTRATTORI_MASSIVE
    C.DISTRATTORI_MASSIVE = 3
    F.k = 4
    try:
        mas_v1, _ = F.massive_cal(frasi_pan_v1)
    finally:
        C.DISTRATTORI_MASSIVE = vecchio_d
        F.k = K_MASSIVE
    p_mas = C.impacca_fonte("massive", mas_v1, 60, tk, al_piu=True)
    p_nem = C.impacca_fonte("nemotron", F.nemotron("nemotron_tool_cal", "cal", set()), 90, tk, filtro_nemo)
    vietati_v1 = {n for e in p_mas.usati + p_nem.usati for n in e.extra.get("strumenti", [])}
    fonti_pan["nemotron"] = F.nemotron("nemotron_tool_pan", "pan", vietati_v1)
    pannello, info, usati = {}, {}, {}
    filtri = {"wiki_it_rg1": filtro_wiki_it, "nemotron": filtro_nemo}
    for dom, (n_batch, comp) in C.PANNELLO.items():
        if dom == "strumenti_it":
            continue
        n = n_batch * C.SEQ_PER_BATCH_PANNELLO
        if len(comp) == 1:
            f = comp[0][0]
            p = C.impacca_fonte(f, fonti_pan[f], n, tk, filtri.get(f))
            pannello[dom], usati[dom] = p.seq, p.usati
            info[dom] = {"fonti": {f: {"esempi_usati": len(p.usati), "token_esempi_usati": p.token_esempi,
                                       "sequenze": n, "esempi_candidati": len(fonti_pan[f])}}}
        else:
            fisse = sum(q for _, q in comp if q)
            pesi = tuple((f, q if q else n - fisse) for f, q in comp)
            s, _et, inf, u = C.dominio_multi(dom, n, fonti_pan, pesi, tk, filtri)
            pannello[dom], usati[dom], info[dom] = s, u, inf
        t = torch.tensor(pannello[dom], dtype=torch.int64).reshape(n_batch, C.SEQ_PER_BATCH_PANNELLO, L)
        if not torch.equal(t, v1["domini"][dom]):
            raise RuntimeError(f"pannello v1 {dom}: la ricostruzione NON coincide al bit col file v1")
        info[dom]["identico_a_v1"] = True
    log(f"pannello: domini v1 ricostruiti e IDENTICI al bit a {PAN_V1.name}: {sorted(pannello)}")
    # strumenti_it NUOVO: scenari riservati, frasi dev + test
    scen_pan = set(SCENARI_PANNELLO)
    schemi_pan = {n: s for n, s in F.schemi.items() if n.split(".")[0] in scen_pan}
    frasi, conta = frasi_massive(F.massive, ("dev", "test"), scen_pan, F.schemi)
    convs = conversazioni_massive(frasi, schemi_pan, K_MASSIVE, DISTRATTORI, "pan_v2", tk)
    n = C.PANNELLO["strumenti_it"][0] * C.SEQ_PER_BATCH_PANNELLO
    p = C.impacca_fonte("massive_pan_v2", convs, n, tk)
    pannello["strumenti_it"], usati["strumenti_it"] = p.seq, p.usati
    # conversione contro il golden ufficiale, sul sotto-insieme di questi scenari
    golden = [x for x in C.jsonl(SORG / "massive_agents_it-IT_input.jsonl")
              if F.massive[x["id"]]["scenario"] in scen_pan]
    ok = sum(1 for x in golden if C.converti_massive(F.massive[x["id"]], F.schemi)[0] == next(iter(x["gt"])))
    info["strumenti_it"] = {"fonti": {"massive_dev_test": {"esempi_usati": len(p.usati), "token_esempi_usati":
                                                           p.token_esempi, "sequenze": n,
                                                           "esempi_candidati": len(convs)}},
                            "scenari": sorted(scen_pan), "funzioni": sorted(schemi_pan), "frasi": conta,
                            "k_massive": K_MASSIVE, "distrattori": DISTRATTORI,
                            "frasi_usate": sum(len(e.extra["id_massive"]) for e in p.usati),
                            "golden_massive_agents_in_questi_scenari": len(golden),
                            "funzione_uguale_al_golden": ok, "identico_a_v1": False}
    log(f"pannello strumenti_it v2: scenari {sorted(scen_pan)} ({len(schemi_pan)} funzioni), frasi dev+test {conta}, "
        f"{len(p.usati)} conversazioni in {n} sequenze · golden {ok}/{len(golden)}")
    return pannello, info, usati, fonti_pan


# ═══════════════════════════ calibrazione ═══════════════════════════
def pool_cot_v2(F: FontiV2, tk, n: int, contaminati: set, classi: "set | None") -> "tuple[list, dict, list]":
    tutti = C.esempi_cot_tutti(tk)
    esclusi = {i for r in C.COT_ESCLUSI for i in r} | contaminati
    for e in tutti:
        e.extra["classe_origine"] = F.classe_cot(e)
    ammessi = [e for e in tutti if e.extra["indice_mescolato"] not in esclusi
               and (classi is None or e.extra["classe_origine"] in classi)]
    p = C.impacca_fonte("cot", ammessi, n, tk)
    idx = [e.extra["indice_mescolato"] for e in p.usati]
    per_classe: dict = {}
    for e in p.usati:
        c = per_classe.setdefault(e.extra["classe_origine"], {"esempi": 0, "token": 0})
        c["esempi"] += 1
        c["token"] += len(e.ids)
    info = {"esempi_disponibili": len(tutti), "esclusi": [[0, 8], [279, 385]], "esempi_usati": len(p.usati),
            "esempi_ammessi": len(ammessi), "classi_ammesse": sorted(classi) if classi else "tutte",
            "indici_mescolati_usati": [min(idx), max(idx)], "token_esempi_usati": p.token_esempi,
            "conta": {t: sum(1 for e in p.usati if e.fonte == f"cot_{t}") for t in ("math", "code")},
            "per_classe_di_origine": per_classe, "esclusi_per_contaminazione": sorted(contaminati)}
    return p.seq, info, p.usati


def fonti_calibrazione(F: FontiV2, variante: str, tk, vietati_nemo: set, esclusi_nemo: set,
                       mfaq_valid_siti: set, frasi_pan_norm: set) -> "tuple[dict, dict]":
    scen_cal = {d["scenario"] for d in F.massive.values()} - set(SCENARI_PANNELLO)
    schemi_cal = {n: s for n, s in F.schemi.items() if n.split(".")[0] in scen_cal}
    frasi_it, conta_it = frasi_massive(F.massive, ("train",), scen_cal, F.schemi, lato_id_it, frasi_pan_norm)
    fonti = {"it_web": F.docs_web("fineweb2_it_train"), "it_wiki": F.docs_wiki_it("wiki_it_cal"),
             "aya": [e for e in F.aya() if F.lato(e.prompt) == "cal"], "oasst2": F.oasst2("oasst2_it_train"),
             "mfaq": F.mfaq("mfaq_it_train", mfaq_valid_siti),
             "massive": conversazioni_massive(frasi_it, schemi_cal, K_MASSIVE, DISTRATTORI, "cal_v2", tk)}
    info = {"massive_it": conta_it, "scenari_calibrazione": sorted(scen_cal), "funzioni_calibrazione": len(schemi_cal),
            "mfaq_filtro_qualita": dict(F.scarti_mfaq)}
    if variante == "tutte":
        fonti["smoltalk_nothink"] = F.smoltalk("smoltalk2_nothink_cal", False, "cal")
        fonti["smoltalk_think"] = F.smoltalk("smoltalk2_think_cal", True, "cal")
        fonti["reasonxl"] = F.reasonxl("reasonxl_cal", "cal")
        fonti["nemotron"], info["nemotron"] = nemotron_v2(F, vietati_nemo, esclusi_nemo, tk)
    else:
        fonti["rx_generale"] = F.reasonxl_pulito("generale")
        fonti["rx_matematica"] = F.reasonxl_pulito("matematica")
        frasi_en, conta_en = frasi_massive(F.massive_en, ("train",), scen_cal, F.schemi,
                                           lambda i: not lato_id_it(i), None)
        fonti["massive_en"] = conversazioni_massive(frasi_en, schemi_cal, K_MASSIVE, DISTRATTORI, "cal_v2_en", tk)
        info["massive_en"] = conta_en
    return fonti, info


def costruisci_variante(variante: str, F: FontiV2, tk, pannello: dict, usati_pan: dict, info_pan: dict,
                        fonti_pan: dict, budget: int) -> dict:
    t0 = time.time()
    miscele = C.parse_miscele(MISCELE[variante])
    dim = C.dimensioni_pool(C.parse_miscele(MISCELE["tutte"]), budget)
    if variante == "senza_llama":
        dim["cot"] = C.dimensioni_pool(miscele, budget)["cot"]
    comp = COMPOSIZIONE[variante]
    log(f"══ variante {variante}: pool {dim} · miscele {list(miscele)} ══")
    vietati_nemo = {n for e in usati_pan["strumenti_en"] for n in e.extra.get("strumenti", [])}
    mfaq_valid_siti = {p["domain"] for p in C.jsonl(F.uscita("mfaq_it_valid"))}
    frasi_pan_norm = {C.norm(u) for e in usati_pan["strumenti_it"] for u in e.contenuti}
    filtro_wiki_it = (lambda e: len(e.ids) >= C.WIKI_IT_MIN_TOKEN)
    filtro_rx_mat = (lambda e: len(e.ids) <= RX_MATEMATICA_MAX_TOKEN)
    esclusi: dict = {}
    esclusi_edu: set = set()
    esclusi_cot: set = set()
    giri = []
    classi_cot = None if variante == "tutte" else AMMESSE_SENZA_LLAMA
    for giro in range(1, 4):
        fonti, info_fonti = fonti_calibrazione(F, variante, tk, vietati_nemo, esclusi.get("strumenti", set()),
                                               mfaq_valid_siti, frasi_pan_norm)
        corpus, etichette, info_dom, usati_dom = {}, {}, {}, {}

        def via(lista, dom):
            return [e for e in lista if e.id not in esclusi.get(dom, set())]

        s, info, u = C.pool_wiki(tk, dim["wiki"])
        corpus["wiki"], etichette["wiki"], info_dom["wiki"], usati_dom["wiki"] = s, ["wikitext2_train"] * len(s), info, u
        s, info, u = C.pool_edu(tk, dim["edu"], esclusi_edu)
        corpus["edu"], etichette["edu"], info_dom["edu"], usati_dom["edu"] = s, ["fineweb_edu"] * len(s), info, u
        s, info, u = pool_cot_v2(F, tk, dim["cot"], esclusi_cot, classi_cot)
        corpus["cot"], etichette["cot"], info_dom["cot"], usati_dom["cot"] = s, ["cot"] * len(s), info, u
        for dom, et, filtro in (("it_web", "fineweb2_train", None), ("it_wiki", "wiki_it_rg0", filtro_wiki_it)):
            p = C.impacca_fonte(et, via(fonti[dom], dom), dim[dom], tk, filtro)
            corpus[dom], etichette[dom], usati_dom[dom] = p.seq, [et] * len(p.seq), p.usati
            info_dom[dom] = {"documenti_usati": len(p.usati), "token_documenti_usati": p.token_esempi,
                             "documenti_candidati": len(fonti[dom])}
        for dom in ("it_chat", "it_pensiero", "strumenti"):
            fonti_d = {f: via(fonti[f], dom) for f, _ in comp[dom]}
            filtri = {"rx_matematica": filtro_rx_mat}
            s, et, info, u = C.dominio_multi(dom, dim[dom], fonti_d, comp[dom], tk, filtri)
            corpus[dom], etichette[dom], info_dom[dom], usati_dom[dom] = s, et, info, u
        ver, contaminati = verifica_v2(tk, variante, corpus, usati_dom, pannello, usati_pan, fonti, fonti_pan)
        giri.append({"giro": giro, "esempi_tolti": {d: sorted(v) for d, v in contaminati.items()},
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
        log(f"  giro {giro}: tolti {({d: len(v) for d, v in contaminati.items()})}: ricostruisco")
    else:
        raise RuntimeError("dopo 3 giri restano n-grammi in comune: da esaminare a mano")
    tensori = {d: torch.tensor(corpus[d], dtype=torch.int64) for d in C.DOMINI}
    for d, t in tensori.items():
        assert t.shape == (dim[d], L) and t.dtype == torch.int64, (d, t.shape)
    return {"variante": variante, "tensori": tensori, "etichette": etichette, "info_dom": info_dom,
            "usati_dom": usati_dom, "ver": ver, "giri": giri, "dim": dim, "miscele": miscele, "comp": comp,
            "info_fonti": info_fonti, "secondi": round(time.time() - t0, 1)}


# ═══════════════════════════ verifiche ═══════════════════════════
def hash_righe(a: np.ndarray) -> np.ndarray:
    w = np.lib.stride_tricks.sliding_window_view(a.astype(np.uint64), C.N_GRAM, axis=1)
    with np.errstate(over="ignore"):
        return (w * C._R).sum(axis=2, dtype=np.uint64)


def insieme_finestre(tensori) -> np.ndarray:
    return np.unique(np.concatenate([hash_righe(t.reshape(-1, t.shape[-1]).numpy()).reshape(-1) for t in tensori]))


IM_S, IM_E = 248045, 248046


def ruoli(riga: np.ndarray, tk, cache: dict) -> np.ndarray:
    """0 prefisso sconosciuto (sequenza iniziata a meta' turno), 1 system, 2 user, 3 assistant, 4 marcatori."""
    r = np.zeros(riga.size, np.int8)
    stato, i = 0, 0
    while i < riga.size:
        t = int(riga[i])
        if t == IM_S and i + 1 < riga.size:
            nx = int(riga[i + 1])
            if nx not in cache:
                cache[nx] = tk.tok.decode([nx])
            s = cache[nx]
            stato = 1 if s.startswith("system") else 2 if s.startswith("user") else 3 if s.startswith("assistant") else 0
            r[i] = r[i + 1] = 4
            i += 2
            continue
        if t == IM_E:
            r[i], stato = 4, 5
            i += 1
            continue
        r[i] = stato if stato != 5 else 4
        i += 1
    return r


NOMI_RUOLI = {0: "prefisso", 1: "system", 2: "user", 3: "assistant", 4: "marcatori"}


def copertura(pan: torch.Tensor, cal_set: np.ndarray, tk, cache: dict) -> dict:
    """Quota di token di un dominio del pannello che stanno dentro ALMENO una finestra di 32 token
    (sequenze intere, template compreso) presente nella calibrazione. Stessa misura della verifica v1."""
    a = pan.reshape(-1, pan.shape[-1]).numpy()
    h = hash_righe(a)
    hit = C.dentro(h.reshape(-1), cal_set).reshape(h.shape)
    cov = np.zeros(a.shape, bool)
    for k in range(C.N_GRAM):
        cov[:, k:k + hit.shape[1]] |= hit
    rr = np.stack([ruoli(x, tk, cache) for x in a])
    per = {NOMI_RUOLI[k]: {"quota_token": round(float((rr == k).mean()), 4),
                           "coperta": round(float(cov[rr == k].mean()), 4) if (rr == k).any() else None} for k in range(5)}
    return {"token": int(a.size), "coperti": int(cov.sum()), "quota_coperta": round(float(cov.mean()), 4),
            "per_ruolo": per}


def verifica_v2(tk, variante: str, corpus: dict, usati_dom: dict, pannello: dict, usati_pan: dict, fonti_cal: dict,
                fonti_pan: dict) -> "tuple[dict, dict]":
    t0 = time.time()
    ver: dict = {}
    cal_per_esempio = {(d, e.id): C.ngrammi_testi(tk, e.contenuti) for d, us in usati_dom.items() if d != "wiki"
                       for e in us}
    cal_wiki = C.ngrammi_sequenze(corpus["wiki"])
    tutti_cal = np.unique(np.concatenate([cal_wiki] + list(cal_per_esempio.values())))
    pan_cont = {d: np.unique(np.concatenate([C.ngrammi_testi(tk, e.contenuti) for e in us]
                                            + [np.empty(0, dtype=np.uint64)])) for d, us in usati_pan.items()}
    storico = torch.load(DIR_PAN / "pannello_ids.pt", weights_only=False)["domini"]
    sto_seq = {d: C.ngrammi_sequenze(t) for d, t in storico.items()}
    wt = np.unique(C.hash_ngrammi(tk.uno(C.FILE_WIKI_TEST.read_text(encoding="utf-8"))))
    mm = C.ngrammi_testi(tk, C.leggi_mmlu(C.FILE_MMLU))
    riferimenti = {**{f"pannello_v2:{d}": v for d, v in pan_cont.items()},
                   **{f"pannello_storico:{d}": v for d, v in sto_seq.items()},
                   "verdetto:wiki.test.raw": wt, "verdetto:mmlu_test.bin": mm}
    rif_tutti = np.unique(np.concatenate(list(riferimenti.values())))
    ver["ngrammi_contenuto_calibrazione"] = int(tutti_cal.size)
    ver["contenuto_calibrazione_vs"] = {k: int(C.dentro(v, tutti_cal).sum()) for k, v in riferimenti.items()}
    ver["ngrammi_riferimenti"] = {k: int(v.size) for k, v in riferimenti.items()}
    contaminati: dict = {}
    dettaglio: dict = {}
    for (d, eid), h in cal_per_esempio.items():
        if h.size and C.dentro(h, rif_tutti).any():
            contaminati.setdefault(d, set()).add(eid)
            for k, v in riferimenti.items():
                n = int(C.dentro(h, v).sum())
                if n:
                    dettaglio.setdefault(d, {}).setdefault(k, []).append([eid, n])
    ver["esempi_calibrazione_con_ngrammi_in_comune"] = dettaglio
    righe_edu = [json.loads(r) for r in (DIR_PAN / C.EDU_FILE).open(encoding="utf-8")]
    cot_tutti = C.esempi_cot_tutti(tk)
    esclusi_cot = {i for r in C.COT_ESCLUSI for i in r}
    pan_norm = {C.norm(c) for us in usati_pan.values() for e in us for c in e.contenuti}
    pan_norm |= {C.norm(c) for e in cot_tutti if e.extra["indice_mescolato"] in esclusi_cot for c in e.contenuti}
    pan_norm |= {C.norm(r["text"]) for r in righe_edu[:C.EDU_PRIMO_DOC]}
    pan_norm = {x for x in pan_norm if len(x) >= C.MIN_CAR_IDENTICI}
    identici: dict = {}
    for d, us in usati_dom.items():
        for e in us:
            if any(len(x) >= C.MIN_CAR_IDENTICI and x in pan_norm for x in (C.norm(c) for c in e.contenuti)):
                identici.setdefault(d, []).append(e.id)
                contaminati.setdefault(d, set()).add(e.id)
    ver["esempi_calibrazione_con_messaggi_identici_al_pannello"] = identici
    ver["wiki_train_ngrammi_in_comune"] = {k: int(C.dentro(v, cal_wiki).sum()) for k, v in riferimenti.items()}
    # sequenze intere (template e schemi compresi) + misura D2 (copertura dei token del pannello)
    cal_set = insieme_finestre([torch.as_tensor(s) for s in corpus.values()])
    cache: dict = {}
    ver["copertura_pannello_v2"] = {d: copertura(torch.as_tensor(s), cal_set, tk, cache) for d, s in pannello.items()}
    ver["sequenze_intere_vs_pannello_storico"] = {d: int(C.dentro(v, cal_set).sum()) for d, v in sto_seq.items()}
    # identificativi, prompt, MASSIVE
    ids = lambda dom, chiave: {e.extra[chiave] for e in usati_dom.get(dom, []) if chiave in e.extra}  # noqa: E731
    tutti_cal_es = [e for us in usati_dom.values() for e in us]
    tutti_pan_es = [e for us in usati_pan.values() for e in us]
    mas_cal = [e for e in usati_dom["strumenti"] if e.fonte.startswith("massive")]
    mas_pan = usati_pan["strumenti_it"]
    ver["id"] = {
        "fineweb2_train_vs_test": len(ids("it_web", "id") & {e.extra["id"] for e in fonti_pan["fineweb2_test"]}),
        "wiki_it_rg0_vs_rg1": len({e.extra["id"] for e in fonti_cal["it_wiki"]}
                                  & {e.extra["id"] for e in fonti_pan["wiki_it_rg1"]}),
        "nemotron_uuid": len({e.extra["uuid"] for e in usati_dom["strumenti"] if "uuid" in e.extra}
                             & {e.extra["uuid"] for e in fonti_pan["nemotron"]}),
        "nemotron_strumenti_del_pannello_in_calibrazione": len(
            {n for e in usati_pan["strumenti_en"] for n in e.extra["strumenti"]}
            & {n for e in usati_dom["strumenti"] if e.fonte == "nemotron" for n in e.extra["strumenti"]}),
        "reasonxl_ds_uid": len({e.extra["ds_uid"] for e in tutti_cal_es if "ds_uid" in e.extra}
                               & {e.extra["ds_uid"] for e in fonti_pan["reasonxl"]}),
        "oasst2_alberi": len({e.extra["albero"] for e in fonti_cal["oasst2"]}
                             & {e.extra["albero"] for e in fonti_pan["oasst2"]}),
        "mfaq_siti": len({e.extra["sito"] for e in fonti_cal["mfaq"]} & {e.extra["sito"] for e in fonti_pan["mfaq"]}),
        "massive_id": len({i for e in mas_cal for i in e.extra["id_massive"]}
                          & {i for e in mas_pan for i in e.extra["id_massive"]}),
        "massive_scenari": len({e.extra["scenario"] for e in mas_cal} & {e.extra["scenario"] for e in mas_pan}),
        "massive_funzioni_mostrate": len({n for e in mas_cal for n in e.extra["strumenti"]}
                                         & {n for e in mas_pan for n in e.extra["strumenti"]}),
        "cot_indici_esclusi_presenti": sum(1 for e in usati_dom["cot"] if e.extra["indice_mescolato"] < 9
                                           or 279 <= e.extra["indice_mescolato"] <= 385),
        "edu_documenti_del_pannello_presenti": sum(1 for e in usati_dom["edu"] if e.extra["doc"] < C.EDU_PRIMO_DOC),
        "id_esempi_comuni_calibrazione_pannello": len({e.id for e in tutti_cal_es} & {e.id for e in tutti_pan_es}),
    }
    prompt_cal = {C.norm(e.prompt) for d in ("it_chat", "it_pensiero") for e in usati_dom[d] if e.prompt}
    prompt_pan = {C.norm(e.prompt) for d in ("it_chat", "it_pensiero") for e in usati_pan[d] if e.prompt}
    ver["prompt_normalizzati_comuni_chat"] = len(prompt_cal & prompt_pan)
    ver["massive_frasi_normalizzate_comuni"] = len({C.norm(u) for e in mas_cal for u in e.contenuti}
                                                   & {C.norm(u) for e in mas_pan for u in e.contenuti})
    ver["secondi"] = round(time.time() - t0, 1)
    bloccanti = [k for k, v in ver["id"].items() if v] + (["prompt"] if ver["prompt_normalizzati_comuni_chat"] else []) \
        + (["massive_frasi"] if ver["massive_frasi_normalizzate_comuni"] else [])
    ver["esito"] = "✅ disgiunti" if not bloccanti and not contaminati else f"⛔ {bloccanti} {list(contaminati)}"
    log(f"  verifiche {variante} ({ver['secondi']} s): contenuto vs riferimenti "
        f"{ {k: v for k, v in ver['contenuto_calibrazione_vs'].items() if v} or 0} · id "
        f"{ {k: v for k, v in ver['id'].items() if v} or 0} · copertura strumenti_it "
        f"{ver['copertura_pannello_v2']['strumenti_it']['quota_coperta']} strumenti_en "
        f"{ver['copertura_pannello_v2']['strumenti_en']['quota_coperta']}")
    if bloccanti:
        raise RuntimeError(f"disgiunzione violata: {bloccanti}")
    return ver, {d: set(v) for d, v in contaminati.items()}


# ═══════════════════════════ misure D5 / D6 ═══════════════════════════
def misure_strumenti(r: dict, tk) -> dict:
    """D5: esempi e token col ragionamento fra i Nemotron usati. D6: quote di token delle sequenze
    MASSIVE per ruolo (come la verifica v1) e quota di italiano (frasi + valori degli argomenti)."""
    out = {}
    us = r["usati_dom"]["strumenti"]
    nem = [e for e in us if e.fonte == "nemotron"]
    if nem:
        con = [e for e in nem if e.extra["pensiero"]]
        out["d5_nemotron"] = {"esempi": len(nem), "col_ragionamento": len(con),
                              "quota_esempi": round(len(con) / len(nem), 4),
                              "token": sum(len(e.ids) for e in nem), "token_col_ragionamento": sum(len(e.ids) for e in con),
                              "quota_token": round(sum(len(e.ids) for e in con) / sum(len(e.ids) for e in nem), 4)}
    t = r["tensori"]["strumenti"]
    cache: dict = {}
    for fonte in ("massive", "massive_en"):
        idx = [i for i, f in enumerate(r["etichette"]["strumenti"]) if f == fonte]
        if not idx:
            continue
        acc = np.zeros(5, np.int64)
        for i in idx:
            rr = ruoli(t[i].numpy(), tk, cache)
            for k in range(5):
                acc[k] += (rr == k).sum()
        conv = [e for e in us if e.fonte == ("massive_cal_v2" if fonte == "massive" else "massive_cal_v2_en")]
        tot = sum(len(e.ids) for e in conv)
        frasi = sum(len(x) for x in tk.ids([c for e in conv for c in e.contenuti]))
        valori = sum(len(x) for x in tk.ids([v for e in conv for v in e.extra["valori"]]))
        out[f"d6_{fonte}"] = {"sequenze": len(idx), "conversazioni": len(conv),
                              "frasi": sum(len(e.extra["id_massive"]) for e in conv),
                              "quote_per_ruolo": {NOMI_RUOLI[k]: round(float(acc[k]) / acc.sum(), 4) for k in range(5)},
                              "quota_frasi_utente": round(frasi / tot, 4), "quota_valori_argomenti": round(valori / tot, 4),
                              "quota_lingua_della_frase": round((frasi + valori) / tot, 4),
                              "token_mediana_conversazione": int(np.median([len(e.ids) for e in conv])),
                              "token_max_conversazione": int(max(len(e.ids) for e in conv))}
    return out


def misura_d6_v1(F: FontiV2, tk) -> dict:
    """La stessa misura sulle 42 conversazioni MASSIVE della v1 (ricostruite col codice v1: k=4, 3 distrattori)."""
    frasi_pan_v1 = {C.norm(F.massive[x["id"]]["utt"]) for x in C.jsonl(SORG / "massive_agents_it-IT_input.jsonl")}
    vecchio_d = C.DISTRATTORI_MASSIVE
    C.DISTRATTORI_MASSIVE, F.k = 3, 4
    try:
        mas_v1, _ = F.massive_cal(frasi_pan_v1)
    finally:
        C.DISTRATTORI_MASSIVE, F.k = vecchio_d, K_MASSIVE
    p = C.impacca_fonte("massive", mas_v1, 60, tk, al_piu=True)
    tot = sum(len(e.ids) for e in p.usati)
    frasi = sum(len(x) for x in tk.ids([c for e in p.usati for c in e.contenuti]))
    valori = []
    for e in p.usati:
        for i in e.extra["id_massive"]:
            valori += [str(v) for v in C.converti_massive(F.massive[i], F.schemi)[1].values()]
    val = sum(len(x) for x in tk.ids(valori))
    return {"conversazioni": len(p.usati), "frasi": sum(len(e.extra["id_massive"]) for e in p.usati),
            "quota_frasi_utente": round(frasi / tot, 4), "quota_valori_argomenti": round(val / tot, 4),
            "quota_lingua_della_frase": round((frasi + val) / tot, 4),
            "token_mediana_conversazione": int(np.median([len(e.ids) for e in p.usati]))}


# ═══════════════════════════ salvataggio ═══════════════════════════
def composizione(r: dict, F: FontiV2) -> dict:
    """Sequenze per dominio e fonte, con la classe di origine di ogni fonte."""
    out = {}
    for d, et in r["etichette"].items():
        conta: dict = {}
        for f in et:
            conta[f] = conta.get(f, 0) + 1
        righe = {}
        for f, n in conta.items():
            if f == "cot":
                righe[f] = {"sequenze": n, "classe": "per esempio (vedi domini.cot.per_classe_di_origine)"}
            else:
                fl = FONTE_LOGICA[f]
                righe[f] = {"sequenze": n, "fonte_logica": fl, "classe": F.classe(fl)}
        out[d] = righe
    return out


def controlla_senza_llama(r: dict, F: FontiV2) -> dict:
    """La variante pulita non deve contenere NESSUNA fonte fuori da AMMESSE_SENZA_LLAMA, neanche in cot."""
    fuori = []
    for d, righe in composizione(r, F).items():
        for f, v in righe.items():
            if f != "cot" and v["classe"] not in AMMESSE_SENZA_LLAMA:
                fuori.append(f"{d}:{f}:{v['classe']}")
    fuori += [f"cot:{e.id}:{e.extra['classe_origine']}" for e in r["usati_dom"]["cot"]
              if e.extra["classe_origine"] not in AMMESSE_SENZA_LLAMA]
    if fuori:
        raise RuntimeError(f"variante senza_llama con fonti non ammesse: {fuori[:10]}")
    return {"fonti_fuori_classe": 0, "classi_ammesse": sorted(AMMESSE_SENZA_LLAMA)}


def salva_variante(r: dict, F: FontiV2, tk, meta_pan: dict, extra: dict) -> dict:
    v = r["variante"]
    out = OUT_PT[v]
    if out.name in PROTETTI:
        raise AssertionError(f"mai scrivere {out.name}")
    torch.save(r["tensori"], out)
    man_sorg = json.loads(MAN_SORG_V2.read_text())
    man = {
        "quando": time.strftime("%Y-%m-%d %H:%M:%S"), "variante": v, "file": out.name, "sha256_file": C.sha256_file(out),
        "byte_file": out.stat().st_size, "l_seq": L, "seme": SEME, "tokenizzatore": str(C.SRC),
        "miscele": r["miscele"], "miscela_di_riferimento": "U", "sequenze_per_dominio": r["dim"],
        "composizione_interna": {d: dict(c) for d, c in r["comp"].items()},
        "soglia_pannello_hash": 0.30, "k_massive": K_MASSIVE, "distrattori_massive": DISTRATTORI,
        "scenari_massive_pannello": list(SCENARI_PANNELLO),
        "domini": {d: {"sequenze": int(t.shape[0]), "token": int(t.numel()), "sha256_token": _sha(t), **r["info_dom"][d]}
                   for d, t in r["tensori"].items()},
        "etichette_per_sequenza": {d: C._rle(r["etichette"][d]) for d in r["tensori"]},
        "composizione_per_fonte_e_origine": composizione(r, F),
        "fonti_calibrazione": r["info_fonti"],
        "disgiunzione": r["ver"], "giri_di_pulizia": r["giri"],
        "sorgenti": {"manifesto": str(MAN_SORG_V2), "sha256_manifesto": C.sha256_file(MAN_SORG_V2),
                     "estratti": {k: C._riassunto_sorgente(x) for k, x in man_sorg["estratti"].items()},
                     "nuovi": {k: (x if k in ("prove_licenze", "cot_origine_domande")
                                   else C._riassunto_sorgente(x)) for k, x in man_sorg.get("nuovi", {}).items()},
                     "locali": man_sorg["locali"], "origini": man_sorg["origini"]},
        "pannello_nuovo": {"file": str(PAN_V2), "sha256_file": C.sha256_file(PAN_V2), "forme": meta_pan["forme"],
                           "sha256_token": meta_pan["sha256_token"]},
        "esempi_decodificati": {d: tk.decodifica(t[0].tolist())[:300] for d, t in r["tensori"].items()},
        "misure": extra, "secondi": r["secondi"],
        "uso": f"THOR_CALIB_CORPUS={out} THOR_CALIB_MISCELA=<U> THOR_VALIDA_BATCH_SEQ=8 THOR_VALIDA_N_HELDOUT=1",
    }
    if v == "senza_llama":
        man["controllo_origine"] = controlla_senza_llama(r, F)
    out.with_suffix(".json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"✅ {out.name} ({out.stat().st_size / 2**20:.1f} MiB, sha256 {man['sha256_file'][:16]}…)")
    for d, t in r["tensori"].items():
        log(f"   {d:12s} {tuple(t.shape)} · {r['info_dom'][d].get('quote_effettive', '')}")
    return man


def salva_pannello(pannello: dict, info_pan: dict, tk) -> dict:
    pan_t = {d: torch.tensor(pannello[d], dtype=torch.int64).reshape(C.PANNELLO[d][0], C.SEQ_PER_BATCH_PANNELLO, L)
             for d in C.PANNELLO}
    v1 = torch.load(PAN_V1, weights_only=False)
    meta = {"quando": time.strftime("%Y-%m-%d %H:%M:%S"), "forme": {d: list(t.shape) for d, t in pan_t.items()},
            "sha256_token": {d: _sha(t) for d, t in pan_t.items()}, "domini": info_pan, "seme": SEME,
            "soglia_pannello_hash": 0.30, "k_massive": K_MASSIVE, "distrattori_massive": DISTRATTORI,
            "scenari_massive_pannello": list(SCENARI_PANNELLO), "tokenizzatore": str(C.SRC),
            "sorgenti": "calibrazione/sorgenti/manifesto_sorgenti_v2.json",
            "identici_a_v1": {d: bool(torch.equal(pan_t[d], v1["domini"][d])) for d in pan_t},
            "sha256_file_v1": C.sha256_file(PAN_V1),
            "costruttore": "thor_v2/costruisci_corpus_calibrazione_v2.py",
            "esempi_decodificati": {d: tk.decodifica(pan_t[d][0, 0].tolist())[:300] for d in pan_t},
            "nota": "strumenti_it rifatto (D2): scenari riservati al pannello; gli altri domini identici al bit a v1"}
    if PAN_V2.exists():
        vecchio = torch.load(PAN_V2, weights_only=False)
        if any(not torch.equal(vecchio["domini"][d], pan_t[d]) for d in pan_t):
            raise RuntimeError(f"{PAN_V2} esiste e differisce: non lo riscrivo (cancellarlo a mano se voluto)")
        log(f"pannello v2 identico a {PAN_V2.name} ✅ (file e meta non riscritti)")
        return vecchio["meta"]
    torch.save({"domini": pan_t, "meta": meta}, PAN_V2)
    return meta


def aggiorna_meta_pannello(meta: dict, risultati: dict) -> None:
    """Disgiunzione e copertura (D2) nel meta del pannello, per variante; riscrive .json e il meta del .pt
    (i tensori restano quelli: sha256 dei token invariati)."""
    meta = dict(meta)
    meta["disgiunzione"] = {v: {"esito": r["ver"]["esito"], "contenuto_calibrazione_vs": r["ver"]["contenuto_calibrazione_vs"],
                                "copertura_pannello_v2": r["ver"]["copertura_pannello_v2"], "id": r["ver"]["id"]}
                            for v, r in risultati.items()}
    meta["disgiunzione"]["esito"] = "✅ disgiunti" if all(r["ver"]["esito"].startswith("✅") for r in risultati.values()) \
        else "⛔"
    testo = json.dumps(meta, ensure_ascii=False, indent=1, default=str)
    js = PAN_V2.with_suffix(".json")
    if js.exists() and js.read_text(encoding="utf-8") == testo:
        log(f"meta del pannello v2 invariato: {PAN_V2.name} non riscritto")
        return
    p = torch.load(PAN_V2, weights_only=False)
    p["meta"] = json.loads(testo)
    torch.save(p, PAN_V2)
    js.write_text(testo, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variante", default="entrambe", choices=("tutte", "senza_llama", "entrambe"))
    ap.add_argument("--budget", type=int, default=1200)
    a = ap.parse_args()
    t0 = time.time()
    tk = C.Tok()
    F = FontiV2(tk, 0.30)
    filtro_wiki_it = (lambda e: len(e.ids) >= C.WIKI_IT_MIN_TOKEN)
    filtro_nemo = (lambda e: len(e.ids) <= NEMOTRON_MAX_TOKEN)
    pannello, info_pan, usati_pan, fonti_pan = pannello_v2(F, tk, filtro_wiki_it, filtro_nemo)
    meta_pan = salva_pannello(pannello, info_pan, tk)
    varianti = ("tutte", "senza_llama") if a.variante == "entrambe" else (a.variante,)
    risultati, manifesti = {}, {}
    d6_v1 = misura_d6_v1(F, tk)
    log(f"D6 v1 (ricostruita): {d6_v1}")
    for v in varianti:
        r = costruisci_variante(v, F, tk, pannello, usati_pan, info_pan, fonti_pan, a.budget)
        risultati[v] = r
        extra = misure_strumenti(r, tk) | {"d6_v1_stessa_misura": d6_v1}
        log(f"  misure {v}: {json.dumps(extra, ensure_ascii=False)}")
        manifesti[v] = salva_variante(r, F, tk, meta_pan, extra)
    aggiorna_meta_pannello(meta_pan, risultati)
    for v in manifesti:      # lo sha256 del file del pannello DOPO l'aggiornamento del suo meta
        manifesti[v]["pannello_nuovo"]["sha256_file"] = C.sha256_file(PAN_V2)
        OUT_PT[v].with_suffix(".json").write_text(json.dumps(manifesti[v], ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    if len(risultati) == 2:
        uguali = {d: bool(torch.equal(risultati["tutte"]["tensori"][d][:min(risultati["tutte"]["dim"][d],
                                                                               risultati["senza_llama"]["dim"][d])],
                                      risultati["senza_llama"]["tensori"][d][:min(risultati["tutte"]["dim"][d],
                                                                                  risultati["senza_llama"]["dim"][d])]))
                  for d in C.DOMINI}
        log(f"domini con prefisso identico fra le due varianti: {uguali}")
        for v in manifesti:
            manifesti[v]["domini_identici_fra_varianti"] = uguali
            OUT_PT[v].with_suffix(".json").write_text(json.dumps(manifesti[v], ensure_ascii=False, indent=1, default=str),
                                                      encoding="utf-8")
    v1 = torch.load(DIR_CAL / "calibrazione_corpus.pt", weights_only=True)
    for v, r in risultati.items():
        uguali_v1 = {d: bool(torch.equal(r["tensori"][d][:min(len(r["tensori"][d]), len(v1[d]))],
                                         v1[d][:min(len(r["tensori"][d]), len(v1[d]))])) for d in C.DOMINI}
        log(f"{v}: domini identici (sul prefisso comune) al corpus v1: {uguali_v1}")
        manifesti[v]["domini_identici_a_v1_sul_prefisso_comune"] = uguali_v1
        OUT_PT[v].with_suffix(".json").write_text(json.dumps(manifesti[v], ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    totale = sum(p.stat().st_size for p in DIR_CAL.rglob("*") if p.is_file())
    if totale > 3 * 10**9:
        raise SystemExit(f"⛔ {totale} byte in {DIR_CAL}: oltre 3 GB")
    log(f"✅ fatto in {time.time() - t0:.0f} s · cartella calibrazione {totale / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
