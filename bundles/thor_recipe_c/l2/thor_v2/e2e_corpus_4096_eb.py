#!/usr/bin/env python3
"""Corpus E2E PIENO E-B: 4096 righe x 4096 token (16,8 M), variante «tutte», miscela U senza wiki — per ESTENSIONE
del corpus intermedio. 26/9/2026. File NUOVO: non modifica nessun file esistente; importa IN SOLA LETTURA
`e2e_corpus_4096.py` (e con lui v1, v2, v2_sem).

## Righe della base TOLTE (26/9, dopo la verifica a fondo dell'intermedio)
Il controllo indipendente guarda le righe come sono impacchettate e vede cio' che il controllo per esempio non vede:
una finestra di 32 token a cavallo del separatore fra due documenti (pie' di pagina MedlinePlus identico in un
documento del pannello storico) e messaggi normalizzati vicini a quelli di un pannello (una traccia cot su una
variante dello stesso compito di un esempio del pannello math_code). Le righe della base colpite da QUALUNQUE
controllo (per esempio, per righe impacchettate, contro held-out e banco mc_v2) si tolgono e si sostituiscono con
righe nuove; sono elencate nel manifesto (`righe_base_tolte`, con sha256: NON riusabili). Le altre restano identiche.
Gli id in collisione (ReasonXL: `ds_uid` per dataset, Stage-1 e Stage-2 con testi diversi) si qualificano con la
posizione nella fonte (`<id>@<pos>`): solo metadati, le righe non cambiano.

## Perche' un'estensione e non una corsa nuova del costruttore
Il teacher (`e2e_teacher.py`) riusa per HARD LINK le righe gia' sigillate, riconosciute dallo sha256 dei loro id
(`_chiave_riga`). Il corpus pieno deve quindi CONTENERE, identiche al token, tutte le righe del corpus intermedio
(`--base`, costruito da `e2e_corpus_4096.py --al-massimo`), cosi' il teacher calcola solo le nuove. Il costruttore non
lo garantisce: prende l'held-out dalla CODA delle fonti e il testo continuo come un flusso unico, quindi con piu' fonti
cambierebbero held-out, esclusioni e tagli. Qui invece:
- **le righe del corpus intermedio restano per prime**, byte per byte (tensori copiati, poi verificati riga per riga);
  le righe nuove vengono DOPO, dominio per dominio; l'held-out (64 righe) e' lo stesso file, byte per byte;
- la tokenizzazione e' la stessa (`C.Tok`, `impacca_fonte`/`dominio_multi` a 4096, edu in un colpo solo come
  `C.pool_edu`); L = 4096; stesso formato dei .pt;
- gli esempi del corpus intermedio vengono RICOSTRUITI dalle fonti (stesse esclusioni dei suoi giri di pulizia) e la
  ricostruzione e' verificata AL TOKEN contro le sue righe: servono per le verifiche sul corpus intero.

## Quote (miscela U, come il costruttore)
Per dominio: quote dichiarate `C.ripartisci(4096, U senza wiki)` = edu 630, cot 630, it_web 630, it_wiki 315,
it_chat 946, it_pensiero 315, strumenti 630; mai sotto le righe gia' nel corpus intermedio, mai oltre la capienza
nuova; cio' che manca va a edu e it_web (pesi 2:2), come `quote_domini`. Dentro un dominio: quote per fonte del
dominio INTERO con i pesi del v2 (`V2.COMPOSIZIONE`), meno le righe gia' presenti (`quote_fonti_nuove`).
cot (fonte nuova, vedi `e2e_scarica_eb.py`): prima le tracce di Thor rimaste, poi Nemotron-Cascade math e code nel
rapporto dei token math/code delle tracce di Thor nel corpus intermedio.

## Esempi nuovi: stesse regole del v2 / v2_sem, applicate alle righe in piu' delle stesse fonti
- esclusi: tutti gli esempi del corpus intermedio (addestramento E held-out: niente ripetizioni) e i contaminati dei
  suoi giri; doppioni ESATTI (contenuti normalizzati) di un esempio gia' preso;
- mfaq: siti del valid (pannello) esclusi, 3 pagine per sito contando anche quelle della testa gia' usata, filtro di
  qualita' `V2.mfaq_scarto`; smoltalk: partizione per hash del prompt (lato «cal»), filtro di lingua a monte;
  nemotron: `nemotron_v2` (un posto su 3 col ragionamento, <= 2048 token, strumenti del pannello vietati);
  wiki: <= 4000 caratteri a paragrafi interi, >= 256 token; fineweb-2: language_score >= 0,9 (a monte);
  cot Cascade: un turno utente + un turno assistente col ragionamento, <= 4096 token (`V2.RX_MATEMATICA_MAX_TOKEN`);
- prompt identici al pannello (it_chat, it_pensiero) tolti PRIMA (per il v2 sono bloccanti);
- giri di pulizia (fino a `GIRI_MAX`) sul corpus INTERO (addestramento + held-out): `S.verifica_sem` (finestre di 32
  token di contenuto contro pannello v2, pannello storico, wiki.test.raw, mmlu_test.bin; messaggi identici; id; R1-R3),
  held-out contro addestramento (H1 finestre, H2 messaggi, H3 R2 cot, H4 R3 quasi-doppioni: gli esempi NUOVI che
  toccano l'held-out si tolgono, l'held-out non cambia), banco MMLU di `mc_v2/mmlu_*.bin` (finestre di 32 token del
  testo intero di domanda e scelte). Gli esempi nuovi colpiti si tolgono e la parte nuova si ricostruisce.
- held-out contro la calibrazione della forgia (H5): l'held-out e' quello intermedio (gia' verificato), rimisurato.

## Uso (solo CPU)
    systemd-run --user --scope -p CPUQuota=800% -p MemoryMax=10G nice -n 10 ionice -c3 \\
        @@ROOT:venv@@/bin/python3 e2e_corpus_4096_eb.py --base DIR_INTERMEDIO --uscita DIR_NUOVA
Test: `e2e_test_corpus_4096_eb.py` (unitari + controlli indipendenti sui file).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_corpus_4096 as E  # noqa: E402  (solo import)

C, V2, S = E.C, E.V2, E.S
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

L4 = E.L4
DOMINI = E.DOMINI_E2E
DIR_EB = C.DIR_CAL / "sorgenti_eb"
MAN_EB = DIR_EB / "manifesto_sorgenti_eb.json"
MC_V2 = Path("@@ROOT:forge@@/mc_v2")
GIRI_MAX = 8
PESO_THOR_COT = 10**9                    # le tracce di Thor rimaste vengono prima di Cascade (math+code = 1000)
INDICE_ESTERNO = 10**9                   # `indice_mescolato` degli esempi cot NON di Thor (fuori dallo spazio 0..1733)
CASCADE_MAX_TOKEN = V2.RX_MATEMATICA_MAX_TOKEN
FILTRO_CASCADE = (lambda e: len(e.ids) <= CASCADE_MAX_TOKEN)             # noqa: E731
FILTRI_EB = {**E.FILTRI, "wiki_it_rg2-3": E.FILTRO_WIKI_IT, "cascade_math": FILTRO_CASCADE,
             "cascade_code": FILTRO_CASCADE}
# fonti nuove: etichetta di riga -> fonte logica (per V2.composizione) e origine dichiarata
FONTE_LOGICA_EB = {"wiki_it_rg2-3": "wikipedia_it", "cascade_math": "nemotron_cascade",
                   "cascade_code": "nemotron_cascade"}
ORIGINE_CASCADE = {
    "classe": "non_dichiarata",
    "licenza": "CC-BY-4.0 (nvidia/Nemotron-Cascade-SFT-Stage-1 @e59a356b); risposte e ragionamento DeepSeek-R1 (MIT)",
    "testo": "SINTETICO: domande math (OpenMathReasoning = problemi AoPS; NuminaMath-CoT) e code (OpenCodeReasoning, "
             "opc-sft-stage2, leetcode, TACO, MagicoderEvolInstruct, apps); ragionamento <think> e risposta di "
             "DeepSeek-R1. E' l'originale inglese di ReasonXL (fonte gia' nel corpus). Classe prudente: alcune "
             "domande code vengono da insiemi sintetici di modelli non tutti dichiarati (Magicoder-Evol-Instruct).",
    "uso": "E-B cot: fonte pubblica equivalente alle tracce di Thor (nuove tracce di Thor solo su GPU)",
}


def log(*a) -> None:
    V2.log(*a)


def sha_righe(x: torch.Tensor) -> list:
    """sha256 degli id int32 di ogni riga: la STESSA chiave del riuso del teacher (`e2e_teacher._chiave_riga`)."""
    a = x.numpy().astype(np.int32)
    return [hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest() for r in a]


def chiave_contenuto(e) -> str:
    return hashlib.sha256("\x1f".join(C.norm(c) for c in e.contenuti).encode("utf-8")).hexdigest()


def tokenizza(esempi: list, tk) -> None:
    manca = [e for e in esempi if e.ids is None]
    for i in range(0, len(manca), 128):
        pezzo = manca[i:i + 128]
        for e, ids in zip(pezzo, tk.ids([e.testo for e in pezzo])):
            e.ids = ids


def hash_righe_a_pezzi(a: np.ndarray, pezzo: int = 32) -> np.ndarray:
    """= V2.hash_righe (stesso risultato al bit) senza il temporaneo (N, 4065, 32) uint64 intero: a 960 righe erano
    ~1 GB per dominio. Solo memoria: la funzione e' pura e riga per riga."""
    a = np.asarray(a)
    return np.concatenate([V2._hash_righe_originale(a[i:i + pezzo]) for i in range(0, a.shape[0], pezzo)]) \
        if a.shape[0] > pezzo else V2._hash_righe_originale(a)


# ═══════════════════════════ quote ═══════════════════════════
def quote_domini_estese(n: int, pesi: dict, base: dict, cap_nuova: dict) -> "tuple[dict, dict]":
    """Quote per dominio del corpus INTERO: dichiarate `C.ripartisci(n, pesi)`, mai sotto la base, mai oltre
    base + capienza nuova; cio' che manca va agli ASSORBONO (edu, it_web) coi loro pesi (`E.riempi`), come
    `E.quote_domini`. Se la base supera gia' una quota, si tolgono righe NUOVE dal dominio piu' sopra la sua quota."""
    dich = C.ripartisci(n, tuple(pesi.items()))
    q = {d: min(max(dich[d], base[d]), base[d] + cap_nuova[d]) for d in pesi}
    extra, tolte = {}, {}
    diff = n - sum(q.values())
    if diff > 0:
        extra = E.riempi(diff, [(d, pesi[d]) for d in E.ASSORBONO], {d: base[d] + cap_nuova[d] - q[d] for d in E.ASSORBONO})
        for d, v in extra.items():
            q[d] += v
    elif diff < 0:
        for _ in range(-diff):
            cand = [d for d in pesi if q[d] > base[d]]
            if not cand:
                raise RuntimeError("la base supera gia' le righe volute")
            d = max(cand, key=lambda d: (q[d] / pesi[d], d))
            q[d] -= 1
            tolte[d] = tolte.get(d, 0) + 1
    return q, {"dichiarate": dich, "effettive": dict(q), "base": dict(base), "capienza_nuova": dict(cap_nuova),
               "nuove": {d: q[d] - base[d] for d in pesi}, "dato_a": extra, "tolte_per_base_oltre_quota": tolte,
               "mancano_sequenze": n - sum(q.values())}


def quote_fonti_nuove(n_nuove: int, comp: tuple, base_f: dict, cap_nuova_f: dict) -> "tuple[dict, dict]":
    """Righe NUOVE per fonte di un dominio: quote del dominio INTERO (base + nuove) con i pesi della composizione e
    le capienze (base + nuova) via `E.riempi`, meno le righe di base. Se una fonte ha gia' in base piu' della sua
    quota, le righe in piu' si tolgono dalla fonte nuova piu' sopra il suo peso."""
    n_tot = n_nuove + sum(base_f.get(f, 0) for f, _ in comp)
    cap_tot = {f: base_f.get(f, 0) + cap_nuova_f.get(f, 0) for f, _ in comp}
    t = E.riempi(n_tot, list(comp), cap_tot)
    if sum(t.values()) < n_tot:
        raise RuntimeError(f"fonti piene: {t} per {n_tot} righe (capienza {cap_tot})")
    nuove = {f: max(0, t[f] - base_f.get(f, 0)) for f, _ in comp}
    w = dict(comp)
    while sum(nuove.values()) > n_nuove:
        cand = [f for f in nuove if nuove[f] > 0]
        f = max(cand, key=lambda f: ((base_f.get(f, 0) + nuove[f]) / w[f], f))
        nuove[f] -= 1
    if sum(nuove.values()) != n_nuove or any(nuove[f] > cap_nuova_f.get(f, 0) for f in nuove):
        raise RuntimeError(f"quote per fonte incoerenti: {nuove} per {n_nuove} (capienze {cap_nuova_f})")
    return nuove, {"quote_dominio_intero": t, "base": {f: base_f.get(f, 0) for f, _ in comp}, "nuove": dict(nuove),
                   "capienza_nuova": {f: cap_nuova_f.get(f, 0) for f, _ in comp}, "pesi": w}


def ordine_pieno(etichette: dict) -> list:
    """Ordine d'uso: `E.ordine_domini` (ogni prefisso ha la composizione U fra i domini) e, DENTRO ogni dominio,
    le righe visitate intercalando le fonti (`C.interfoglia` sulle etichette): le righe nuove non finiscono in coda."""
    visita = {}
    for d in DOMINI:
        gruppi: dict = {}
        for i, f in enumerate(etichette[d]):
            gruppi.setdefault(f, []).append(i)
        seq, _ = C.interfoglia([C.Pacco(f, idx, [], 0) for f, idx in gruppi.items()])
        visita[d] = seq
    return [(k, visita[DOMINI[k]][i]) for k, i in E.ordine_domini({d: len(etichette[d]) for d in DOMINI})]


# ═══════════════════════════ fonti nuove (stesse regole delle fonti di base) ═══════════════════════════
def man_eb() -> dict:
    m = json.loads(MAN_EB.read_text())
    for k, v in m["estratti"].items():
        if C.sha256_file(DIR_EB / v["uscita"]) != v["sha256_scritto"]:
            raise RuntimeError(f"sorgente E-B {k}: {v['uscita']} non combacia col suo manifesto")
    return m


def _jsonl_eb(m: dict, chiave: str) -> list:
    return C.jsonl(DIR_EB / m["estratti"][chiave]["uscita"])


def docs_edu_nuovi(m: dict) -> list:
    """Documenti FineWeb-Edu dei row group nuovi; indice di documento = riga nel file sorgente (0-1999 = rg 0-1)."""
    return [C.Esempio("fineweb_edu", f"edu:{r['_riga_file']}", r["text"], [r["text"]], extra={"doc": r["_riga_file"]})
            for r in _jsonl_eb(m, "fineweb_edu_rg2-3")]


def pool_edu_lista(tk, n: int, docs: list, L: int, proprietari: "list | None" = None) -> "tuple[list, dict, list]":
    """= `C.pool_edu` su una lista data di documenti: uniti con \\n\\n e tokenizzati in UN colpo, righe = primi n*L
    token, usati = documenti fino all'ultimo carattere dell'ultimo token. Con `proprietari` (lista) vi aggiunge, per
    ogni token delle righe, l'indice del documento (in `usati`) a cui appartiene: il separatore va al documento prima."""
    testo = "\n\n".join(e.testo for e in docs)
    enc = tk.tok(testo, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) < n * L:
        raise RuntimeError(f"edu nuovi: {len(ids)} token, ne servono {n * L}")
    usati = []
    if n:
        fine_car, cum = enc["offset_mapping"][n * L - 1][1], 0
        for e in docs:
            if cum >= fine_car:
                break
            usati.append(e)
            cum += len(e.testo) + 2
    info = {"documenti_usati": len(usati), "primo_doc": docs[0].extra["doc"] if docs else None,
            "ultimo_doc_usato": usati[-1].extra["doc"] if usati else None, "token_flusso_totale": len(ids),
            "nota": "documenti nuovi uniti con \\n\\n e tokenizzati in un colpo (come C.pool_edu)"}
    if proprietari is not None:
        inizi = np.cumsum([0] + [len(e.testo) + 2 for e in docs[:-1]])
        off = np.array([o[0] for o in enc["offset_mapping"][:n * L]], dtype=np.int64)
        own = np.searchsorted(inizi, off, side="right") - 1
        if own.size and (own.min() < 0 or own.max() >= len(usati)):
            raise RuntimeError("edu: proprietari dei token fuori dai documenti usati")
        proprietari.append(own)
    return [ids[j * L:(j + 1) * L] for j in range(n)], info, usati


def docs_web_nuovi(m: dict) -> list:
    return [C.Esempio("fineweb2_train", f"fw2:{r['id']}", r["text"] + "\n\n", [r["text"]],
                      extra={"id": r["id"], "url": r["url"]}) for r in _jsonl_eb(m, "fineweb2_it_train_rg2-4")]


def docs_wiki_nuovi(m: dict) -> list:
    out = []
    for r in _jsonl_eb(m, "wiki_it_rg2-3"):
        t = C.taglia_paragrafi(f"{r['title']}\n\n{r['text']}", C.WIKI_IT_MAX_CAR)
        out.append(C.Esempio("wiki_it_rg2-3", f"wiki_it:{r['id']}", t + "\n\n", [t], extra={"id": r["id"], "titolo": r["title"]}))
    return out


def mfaq_da_pagine(pagine: list, per_sito: dict, tk) -> list:
    """Il ciclo di `C.Fonti.mfaq` (cap per sito, 6 coppie per conversazione) su pagine date; `per_sito` continua."""
    out = []
    for p in pagine:
        if per_sito.get(p["domain"], 0) >= C.MFAQ_PAGINE_PER_SITO:
            continue
        per_sito[p["domain"]] = per_sito.get(p["domain"], 0) + 1
        coppie = [c for c in p["qa_pairs"] if (c.get("question") or "").strip() and (c.get("answer") or "").strip()]
        for j in range(0, len(coppie), C.MFAQ_COPPIE_PER_CONV):
            ms = []
            for c in coppie[j:j + C.MFAQ_COPPIE_PER_CONV]:
                ms += [{"role": "user", "content": c["question"].strip()},
                       {"role": "assistant", "content": c["answer"].strip()}]
            msgs, cont = C._msg_chat(ms, False)
            out.append(C.Esempio("mfaq", f"mfaq:{p['id']}:{j}", tk.chat(msgs), cont, ms[0]["content"],
                                 {"pagina": p["id"], "sito": p["domain"]}))
    return out


def mfaq_nuovi(m: dict, tk, siti_esclusi: set) -> "tuple[list, dict]":
    """Pagine DOPO la testa gia' usata (file intero, stessa revisione). Il conteggio per sito parte da quello della
    testa (stesso mescolamento di `C.Fonti.mfaq`), cosi' un sito non supera 3 pagine sull'unione. Filtro di qualita'
    del v2 (`V2.mfaq_scarto`)."""
    e = m["estratti"]["mfaq_it_train_intero"]
    righe = _jsonl_eb(m, "mfaq_it_train_intero")
    n_testa = e["testa_gia_usata"]["righe"]
    base = [p for p in righe[:n_testa] if p["domain"] not in siti_esclusi]
    random.Random(f"{C.SEME}:mfaq:mfaq_it_train").shuffle(base)
    per_sito: dict = {}
    for p in base:                                           # il conteggio per sito di C.Fonti.mfaq sulla testa
        if per_sito.get(p["domain"], 0) < C.MFAQ_PAGINE_PER_SITO:
            per_sito[p["domain"]] = per_sito.get(p["domain"], 0) + 1
    nuove = [p for p in righe[n_testa:] if p["domain"] not in siti_esclusi]
    random.Random(f"{C.SEME}:mfaq:mfaq_it_train:eb").shuffle(nuove)
    tutti = mfaq_da_pagine(nuove, per_sito, tk)
    scarti: dict = {}
    out = []
    for x in tutti:
        mot = V2.mfaq_scarto(x)
        if mot:
            scarti[mot] = scarti.get(mot, 0) + 1
        else:
            out.append(x)
    return out, {"pagine_dopo_la_testa": len(righe) - n_testa, "pagine_candidate": len(nuove),
                 "conversazioni": len(tutti), "scarti_qualita": scarti, "tenute": len(out)}


def smoltalk_nuovi(m: dict, F, tk) -> list:
    """Come `C.Fonti.smoltalk(..., False, "cal")` sulle righe nuove dello stesso file (stesso schema di id)."""
    out = []
    for r in _jsonl_eb(m, "smoltalk2_nothink_rg4-127"):
        prompt = next((x["content"] for x in r["messages"] if x["role"] == "user"), "")
        if F.lato(prompt) != "cal":
            continue
        msgs, cont = C._msg_chat(r["messages"], False, r.get("custom_instructions") or "")
        out.append(C.Esempio("smoltalk_nothink", f"smoltalk2_nothink_cal:{r['_pos']}", tk.chat(msgs), cont, prompt,
                             {"pos": r["_pos"]}))
    random.Random(f"{C.SEME}:smoltalk2_nothink_cal:eb").shuffle(out)
    return out


def nemotron_da_righe(righe: list, vietati: set, esclusi: set, tk, seme: str) -> "tuple[list, dict]":
    """L'algoritmo di `V2.nemotron_v2` su righe date (mescolate col seme dato): posto j col ragionamento se
    j % 3 == 2, versione <= 2048 token; strumenti vietati ed esclusi fuori. Si tengono le sole lunghezze."""
    righe = list(righe)
    random.Random(seme).shuffle(righe)
    cand = []
    for r in righe:
        e0 = C.esempio_nemotron(r, False, tk)
        if (vietati and set(e0.extra["strumenti"]) & vietati) or e0.id in esclusi:
            continue
        cand.append((e0, C.esempio_nemotron(r, True, tk)))
    lun = []
    testi = [e.testo for c in cand for e in c]
    for i in range(0, len(testi), 256):
        lun += [len(x) for x in tk.ids(testi[i:i + 256])]
    ok0 = [i for i in range(len(cand)) if lun[2 * i] <= V2.NEMOTRON_MAX_TOKEN]
    ok1 = [i for i in range(len(cand)) if lun[2 * i + 1] <= V2.NEMOTRON_MAX_TOKEN]
    usati, out = set(), []
    p0 = p1 = 0
    while True:
        vuole = len(out) % V2.NEMOTRON_OGNI_PENSIERO == V2.NEMOTRON_OGNI_PENSIERO - 1
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
    return out, {"righe": len(righe), "candidati_dopo_vietati_e_esclusi": len(cand),
                 "senza_ragionamento_entro_2048": len(ok0), "con_ragionamento_entro_2048": len(ok1),
                 "ordine_esempi": len(out)}


def cascade_nuovi(m: dict, tk, ramo: str) -> "tuple[list, dict]":
    """Tracce Cascade nel template di Thor, come `C.esempi_cot_tutti`: utente = domanda, assistente = risposta con
    `reasoning_content` = ragionamento. Tenute: un turno utente + un turno assistente, domanda non vuota (ne' «-»),
    ragionamento e risposta non vuoti. Mescolate col seme. Il taglio a 4096 token e' il filtro di `FILTRI_EB`."""
    out, scarti = [], {}
    base_idx = INDICE_ESTERNO + (0 if ramo == "math" else 10**7)
    righe = _jsonl_eb(m, f"cascade_{ramo}")
    for r in righe:
        ms = r["messages"]
        mot = None
        if len(ms) != 2 or ms[0]["role"] != "user" or ms[1]["role"] != "assistant":
            mot = "non_un_solo_scambio"
        else:
            q = ms[0]["content"] or ""
            c = ms[1]["content"] or ""
            rag, risp = C._dividi_pensiero(c)
            if not q.strip() or q.strip() == "-":
                mot = "domanda_vuota"
            elif "</think>" not in c or not rag.strip() or not risp.strip():
                mot = "senza_ragionamento_o_risposta"
        if mot:
            scarti[mot] = scarti.get(mot, 0) + 1
            continue
        testo = tk.tok.apply_chat_template([{"role": "user", "content": q},
                                            {"role": "assistant", "content": risp, "reasoning_content": rag}],
                                           tokenize=False)
        out.append(C.Esempio(f"cascade_{ramo}", f"cascade:{ramo}:{r['_riga']}", testo, [q, risp, rag], q,
                             {"indice_mescolato": base_idx + r["_riga"], "riga": r["_riga"], "source": r.get("source"),
                              "classe_origine": ORIGINE_CASCADE["classe"]}))
    random.Random(f"{C.SEME}:cascade:{ramo}").shuffle(out)
    return out, {"righe": len(righe), "scarti": scarti, "tenute": len(out),
                 "filtro_lunghezza": f"<= {CASCADE_MAX_TOKEN} token nel template di Thor (FILTRI_EB, in impacchettamento)"}


# ═══════════════════════════ base (corpus intermedio) ═══════════════════════════
def carica_base(d: Path) -> dict:
    man = json.loads((d / "e2e_corpus_4096_tutte.json").read_text(encoding="utf-8"))
    out = {"dir": d, "man": man, "sha": {}}
    for lato, chiave in (("train", "train"), ("heldout", "heldout")):
        f = d / man["file"][chiave]["file"]
        sha = C.sha256_file(f)
        if sha != man["file"][chiave]["sha256"]:
            raise RuntimeError(f"{f}: sha256 {sha} diverso dal manifesto della base")
        out[lato] = torch.load(f, weights_only=True)
        out["sha"][lato] = {"file": str(f), "sha256": sha}
        t = out[lato]
        if t["l_seq"] != L4 or t["variante"] != "tutte" or list(t["nomi_domini"]) != list(DOMINI):
            raise RuntimeError(f"{f}: formato inatteso")
    p = d / man["file"]["heldout_pannelli"]["file"]
    if C.sha256_file(p) != man["file"]["heldout_pannelli"]["sha256"]:
        raise RuntimeError(f"{p}: sha256 diverso dal manifesto della base")
    out["sha"]["heldout_pannelli"] = {"file": str(p), "sha256": man["file"]["heldout_pannelli"]["sha256"]}
    return out


def esclusi_base(man: dict) -> "tuple[dict, set, set]":
    """Esclusioni dei giri di pulizia della base: {dominio: id} + documenti edu + indici cot (addestramento)."""
    esclusi: dict = {}
    for g in man["giri_di_pulizia"]:
        for d, v in g["contaminati_tolti"].items():
            if d not in ("edu", "cot"):
                esclusi.setdefault(d, set()).update(v)
    det = man["domini"]["train"]
    return esclusi, set(det["edu"]["dettaglio"]["documenti_esclusi_per_contaminazione"]), \
        set(det["cot"]["dettaglio"]["esclusi_per_contaminazione"])


def mappa_esempi(F, tk, fonti: dict) -> dict:
    """id -> [(dominio, fonte, Esempio), ...] per tutte le liste che il costruttore usa. Un id puo' avere PIU'
    esempi (ReasonXL: 109 `ds_uid` ripetuti con testi diversi nella stessa lista), sempre nella stessa fonte: quale
    e' stato usato lo decide il flusso dei token (`ricostruisci_lato`)."""
    m: dict = {}

    def metti(dom, f, e):
        v = m.setdefault(e.id, [])
        if any(x[2] is e for x in v):
            return
        if v and v[0][:2] != (dom, f):
            raise RuntimeError(f"id {e.id} in due fonti ({v[0][:2]} e {(dom, f)})")
        v.append((dom, f, e))

    for e in E.docs_edu():
        metti("edu", "fineweb_edu", e)
    for e in C.esempi_cot_tutti(tk):
        e.extra["classe_origine"] = F.classe_cot(e)
        metti("cot", "cot", e)
    for dom in ("it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti"):
        _, fd = E.composizione_dominio("tutte", dom, fonti)
        for f, lst in fd.items():
            for e in lst:
                metti(dom, f, e)
    return m


def scegli_per_flusso(ids: list, mappa: dict, righe: torch.Tensor, et: list, tk) -> dict:
    """Per ogni fonte: gli esempi usati, nell'ordine, scelti fra i candidati di ogni id con il flusso dei token
    delle righe di quella fonte (l'esempio k deve coprire esattamente i token successivi; l'ultimo puo' essere
    troncato). Un id con due testi diversi si risolve cosi' senza ambiguita'; nessun candidato = errore."""
    per_ids: dict = {}
    for i in ids:
        per_ids.setdefault(mappa[i][0][1], []).append(i)
    tokenizza([x[2] for i in dict.fromkeys(ids) for x in mappa[i]], tk)
    out = {}
    for f, lista in per_ids.items():
        flusso = righe[[r for r, x in enumerate(et) if x == f]].reshape(-1).tolist()
        pos, scelti = 0, []
        for i in lista:
            cands = [x[2] for x in mappa[i]]
            tokenizza(cands, tk)
            ok = [c for c in cands if flusso[pos:pos + len(c.ids)] == list(c.ids)[:max(0, len(flusso) - pos)]]
            if not ok or pos >= len(flusso):
                raise RuntimeError(f"fonte {f}: l'esempio {i} non continua il flusso delle righe (posizione {pos})")
            if len({tuple(c.ids) for c in ok}) > 1:
                raise RuntimeError(f"fonte {f}: esempio {i} ambiguo nel flusso")
            scelti.append(ok[0])
            pos += len(ok[0].ids)
        if pos < len(flusso):
            raise RuntimeError(f"fonte {f}: gli esempi coprono {pos} token su {len(flusso)}")
        out[f] = scelti
    return out


def verifica_edu_heldout(tk, usati: list, righe: torch.Tensor, prossimi: list) -> int:
    """Le righe held-out edu sono il prefisso della tokenizzazione in un colpo dei documenti usati, seguiti da 0..k
    documenti successivi (il costruttore tokenizzava la coda intera: l'ultimo token puo' dipendere dal testo dopo)."""
    n = righe.shape[0]
    for k in range(0, len(prossimi) + 1):
        testo = "\n\n".join([e.testo for e in usati] + [e.testo for e in prossimi[:k]])
        ids = tk.tok(testo, add_special_tokens=False)["input_ids"]
        if len(ids) >= n * L4 and torch.equal(torch.tensor(ids[:n * L4]).reshape(n, L4), righe):
            return k
    raise RuntimeError("held-out edu: le righe non si ricostruiscono dai documenti usati")


def ricostruisci_lato(nome: str, t: dict, mappa: dict, tk, esc_edu: set, man: dict) -> E.Lato:
    """Esempi usati da un lato della base, dalle liste delle fonti; verifica AL TOKEN: per ogni fonte il flusso dei
    suoi esempi ricostruisce le sue righe (e i cu_seqlens salvati); edu col `C.pool_edu` del costruttore."""
    lato = E.Lato(nome)
    for d in DOMINI:
        ids = list(t["esempi"][d])
        for i in ids:
            if i not in mappa or mappa[i][0][0] != d:
                raise RuntimeError(f"base {nome}:{d}: esempio {i} non ritrovato nelle fonti")
        righe = t["domini"][d]
        et = list(t["fonti"][d])
        ptr, cu_all = t["cu_ptr"][d], t["cu"][d]
        cu_salvati = [cu_all[int(ptr[r]):int(ptr[r + 1])].tolist() for r in range(righe.shape[0])]
        ini_salvati = t["inizio_esempio"][d].tolist()
        if d == "edu":
            es = [mappa[i][0][2] for i in ids]
            if nome == "train":
                with E.lunghezza(L4):
                    s, _info, u = C.pool_edu(tk, righe.shape[0], esc_edu)
                if not torch.equal(torch.tensor(s, dtype=torch.int64).reshape(-1, L4), righe) or [e.id for e in u] != ids:
                    raise RuntimeError("base train edu: C.pool_edu non ricostruisce le righe")
            else:
                ultimo = max(e.extra["doc"] for e in es)
                prossimi = [e for e in E.docs_edu() if e.extra["doc"] > ultimo][:8]
                verifica_edu_heldout(tk, es, righe, prossimi)
        else:
            per_f = scegli_per_flusso(ids, mappa, righe, et, tk)
            scelti = {f: iter(v) for f, v in per_f.items()}
            es = [next(scelti[mappa[i][0][1]]) for i in ids]     # ordine degli esempi della base
            for f in dict.fromkeys(et):
                idx = [r for r, x in enumerate(et) if x == f]
                cu, ini = E.confini(per_f.get(f, []), [righe[r].tolist() for r in idx], L4)
                if d not in E.TESTO_CONTINUO and ([cu_salvati[r] for r in idx] != cu
                                                  or [ini_salvati[r] for r in idx] != ini):
                    raise RuntimeError(f"base {nome}:{d}:{f}: cu_seqlens ricostruiti diversi dai salvati")
            if set(per_f) - set(et):
                raise RuntimeError(f"base {nome}:{d}: esempi di fonti senza righe {set(per_f) - set(et)}")
        lato.metti(d, [], et, man["domini"][nome][d]["dettaglio"], es, cu_salvati, ini_salvati)
    return lato


# ═══════════════════════════ parte nuova ═══════════════════════════
class Candidati:
    """Liste per fonte degli esempi che la parte nuova puo' usare (avanzi delle liste di base + fonti nuove), gia'
    filtrate dalle regole che il v2 rende bloccanti e dai doppioni esatti; `esclusi` cresce coi giri."""

    def __init__(self, liste: dict, comp: dict, info: dict):
        self.liste, self.comp, self.info = liste, comp, info      # liste[dom][fonte] = [Esempio]
        self.esclusi: dict = {}                                    # id -> motivo (giri)

    def per(self, dom: str) -> dict:
        return {f: [e for e in lst if e.id not in self.esclusi] for f, lst in self.liste[dom].items()}


def costruisci_candidati(F, tk, fonti: dict, pan: dict, base_usati: set, esclusi_b: dict, esc_cot: set,
                         chiavi_prese: set, m_eb: dict) -> Candidati:
    liste: dict = {d: {} for d in DOMINI}
    info: dict = {}
    fuori = lambda dom, lst: [e for e in lst if e.id not in base_usati and e.id not in esclusi_b.get(dom, set())]  # noqa: E731
    # edu: solo documenti nuovi (la base prende il flusso di rg 0-1 fino alla capienza)
    liste["edu"]["fineweb_edu"] = [e for e in docs_edu_nuovi(m_eb) if e.id not in base_usati]
    # cot: tracce di Thor rimaste, poi Cascade
    liste["cot"]["cot"] = [e for e in E.cot_ammessi(F, tk, None, esc_cot) if e.id not in base_usati]
    for ramo in ("math", "code"):
        liste["cot"][f"cascade_{ramo}"], info[f"cascade_{ramo}"] = cascade_nuovi(m_eb, tk, ramo)
    # testo continuo
    web_test = {e.extra["id"] for e in pan["fonti"]["fineweb2_test"]}
    liste["it_web"]["fineweb2_train"] = fuori("it_web", fonti["it_web"]) + [
        e for e in docs_web_nuovi(m_eb) if e.extra["id"] not in web_test]
    wiki_pan = {e.extra["id"] for e in pan["fonti"]["wiki_it_rg1"]}
    wiki_base = {e.extra["id"] for e in fonti["it_wiki"]}
    liste["it_wiki"]["wiki_it_rg0"] = fuori("it_wiki", fonti["it_wiki"])
    liste["it_wiki"]["wiki_it_rg2-3"] = [e for e in docs_wiki_nuovi(m_eb)
                                         if e.extra["id"] not in wiki_pan and e.extra["id"] not in wiki_base]
    # domini a piu' fonti: avanzi delle liste di base + righe nuove delle stesse fonti
    for dom in ("it_chat", "it_pensiero", "strumenti"):
        for f, _ in V2.COMPOSIZIONE["tutte"][dom]:
            liste[dom][f] = fuori(dom, fonti[f])
    mf, info["mfaq"] = mfaq_nuovi(m_eb, tk, pan["mfaq_valid_siti"])
    liste["it_chat"]["mfaq"] += mf
    liste["it_chat"]["smoltalk_nothink"] += smoltalk_nuovi(m_eb, F, tk)
    uuid_pan = {e.extra["uuid"] for e in pan["fonti"]["nemotron"]}
    nem, info["nemotron"] = nemotron_da_righe(_jsonl_eb(m_eb, "nemotron_tool_rg1-6"), pan["vietati_nemo"], set(), tk,
                                              f"{C.SEME}:nemotron_tool_cal:eb")
    liste["strumenti"]["nemotron"] += [e for e in nem if e.extra["uuid"] not in uuid_pan]
    # regole bloccanti del v2 applicate prima + doppioni esatti (contro base, held-out e fra i nuovi)
    prompt_pan = {C.norm(e.prompt) for d in ("it_chat", "it_pensiero") for e in pan["usati"][d] if e.prompt}
    viste = set(chiavi_prese)
    tolti: dict = {}
    for dom in DOMINI:
        for f, lst in liste[dom].items():
            tenuti, ids_visti = [], set()
            for e in lst:
                mot = None
                if e.id in base_usati or e.id in ids_visti:
                    mot = "id_gia_usato"
                elif dom in ("it_chat", "it_pensiero") and e.prompt and C.norm(e.prompt) in prompt_pan:
                    mot = "prompt_del_pannello"
                else:
                    k = chiave_contenuto(e)
                    if k in viste:
                        mot = "doppione_esatto"
                    else:
                        viste.add(k)
                if mot:
                    tolti.setdefault(f"{dom}:{f}", {}).setdefault(mot, 0)
                    tolti[f"{dom}:{f}"][mot] += 1
                    continue
                ids_visti.add(e.id)
                tenuti.append(e)
            liste[dom][f] = tenuti
    info["tolti_prima"] = tolti
    info["candidati"] = {dom: {f: len(lst) for f, lst in liste[dom].items()} for dom in DOMINI}
    comp = {"edu": (("fineweb_edu", 1),), "it_web": (("fineweb2_train", 1),),
            "it_wiki": (("wiki_it_rg0", 1), ("wiki_it_rg2-3", 1)),
            "it_chat": V2.COMPOSIZIONE["tutte"]["it_chat"], "it_pensiero": V2.COMPOSIZIONE["tutte"]["it_pensiero"],
            "strumenti": V2.COMPOSIZIONE["tutte"]["strumenti"]}
    return Candidati(liste, comp, info)


def pesi_cot(base_cot: list) -> tuple:
    """Pesi (Thor rimasto, Cascade math, Cascade code): Thor prima; math/code nel rapporto dei TOKEN delle tracce di
    Thor nel corpus di base."""
    tm = sum(len(e.ids) for e in base_cot if e.fonte == "cot_math")
    tc = sum(len(e.ids) for e in base_cot if e.fonte == "cot_code")
    if not tm or not tc:
        raise RuntimeError(f"cot di base senza math o code ({tm}, {tc})")
    wm = round(1000 * tm / (tm + tc))
    return (("cot", PESO_THOR_COT), ("cascade_math", wm), ("cascade_code", 1000 - wm))


def capienze(cand: Candidati, lun, tk) -> "tuple[dict, dict]":
    """Righe da 4096 che ogni fonte nuova puo' dare (filtri di lunghezza compresi); edu col flusso in un colpo."""
    per_fonte, per_dom = {}, {}
    for dom in DOMINI:
        per_fonte[dom] = {}
        for f, lst in cand.per(dom).items():
            if dom == "edu":
                testo = "\n\n".join(e.testo for e in lst)
                per_fonte[dom][f] = len(tk.tok(testo, add_special_tokens=False)["input_ids"]) // L4 if lst else 0
            else:
                per_fonte[dom][f] = E.ammessi(lst, FILTRI_EB.get(f), lun)[1] // L4
        per_dom[dom] = sum(per_fonte[dom].values())
    return per_dom, per_fonte


def parte_nuova(cand: Candidati, q_nuove: dict, base_f: dict, comp: dict, cap_f: dict, tk) -> dict:
    """Righe nuove per dominio: `pool_edu_lista` per edu, `C.dominio_multi` + `E.confini_multi` per gli altri."""
    out = {}
    with E.lunghezza(L4):
        for dom in DOMINI:
            n = q_nuove[dom]
            per = cand.per(dom)
            if dom == "edu":
                pr: list = []
                s, info, u = pool_edu_lista(tk, n, per["fineweb_edu"], L4, pr)
                out[dom] = {"seq": s, "et": ["fineweb_edu"] * n, "usati": u, "cu": [[0, L4]] * n,
                            "ini": [False] * n, "info": info, "proprietari": pr[0]}
                continue
            qf, info_q = quote_fonti_nuove(n, comp[dom], base_f[dom], cap_f[dom])
            if n == 0:
                out[dom] = {"seq": [], "et": [], "usati": [], "cu": [], "ini": [], "info": {"quote": info_q},
                            "proprietari": np.empty(0, dtype=np.int64)}
                continue
            compq = tuple((f, qf[f]) for f, _ in comp[dom])
            s, et, info, u = C.dominio_multi(dom, n, per, compq, tk, FILTRI_EB)
            cu, ini = E.confini_multi(s, info, u, compq, L4)
            if dom in E.TESTO_CONTINUO:
                cu = [[0, L4]] * len(s)
            info["quote"] = info_q
            out[dom] = {"seq": s, "et": et, "usati": u, "cu": cu, "ini": ini, "info": info,
                        "proprietari": proprietari_multi(s, et, info, u, compq, L4)}
    return out


def proprietari_multi(seq: list, et: list, info: dict, usati: list, comp: tuple, L: int) -> np.ndarray:
    """Per ogni token delle righe di `C.dominio_multi` (nell'ordine delle righe), l'indice in `usati` dell'esempio a
    cui appartiene. Gli esempi usati sono concatenati per fonte nell'ordine della composizione (`E.confini_multi`);
    le righe di una fonte sono, in ordine, il flusso dei suoi esempi tagliato ogni L."""
    flussi, token, k = {}, {}, 0
    for f, _ in comp:
        n = info["fonti"][f]["esempi_usati"]
        rng = range(k, k + n)
        flussi[f] = np.concatenate([np.full(len(usati[i].ids), i, dtype=np.int64) for i in rng]) \
            if n else np.empty(0, dtype=np.int64)
        token[f] = np.concatenate([np.asarray(usati[i].ids, dtype=np.int64) for i in rng]) \
            if n else np.empty(0, dtype=np.int64)
        k += n
    if k != len(usati):
        raise RuntimeError("esempi usati non ripartibili per fonte")
    pos = {f: 0 for f in flussi}
    out = []
    for r, f in enumerate(et):
        pezzo = flussi[f][pos[f]:pos[f] + L]
        if pezzo.size != L or not np.array_equal(token[f][pos[f]:pos[f] + L], np.asarray(seq[r], dtype=np.int64)):
            raise RuntimeError(f"riga {r}: il flusso degli esempi della fonte {f} non ricostruisce la riga")
        out.append(pezzo)
        pos[f] += L
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def proprietari_per_fonte(righe: np.ndarray, et: list, usati: list, fonte_di: list) -> np.ndarray:
    """Come `proprietari_multi` quando la fonte di ogni esempio e' data (`fonte_di[i]` per `usati[i]`): gli esempi di
    una fonte, nel loro ordine, fanno il flusso delle righe di quella fonte (verificato al token)."""
    righe = np.asarray(righe, dtype=np.int64).reshape(len(et), -1)
    L = righe.shape[1]
    per: dict = {}
    for i, f in enumerate(fonte_di):
        per.setdefault(f, []).append(i)
    flussi = {f: np.concatenate([np.full(len(usati[i].ids), i, dtype=np.int64) for i in idx]) for f, idx in per.items()}
    token = {f: np.concatenate([np.asarray(usati[i].ids, dtype=np.int64) for i in idx]) for f, idx in per.items()}
    pos = {f: 0 for f in per}
    out = []
    for r, f in enumerate(et):
        if f not in flussi:
            raise RuntimeError(f"riga {r}: nessun esempio della fonte {f}")
        pezzo = flussi[f][pos[f]:pos[f] + L]
        if pezzo.size != L or not np.array_equal(token[f][pos[f]:pos[f] + L], righe[r]):
            raise RuntimeError(f"riga {r}: il flusso degli esempi della fonte {f} non ricostruisce la riga")
        out.append(pezzo)
        pos[f] += L
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def proprietari_base(tk, lato: E.Lato, base_t: dict, mappa: dict, esc_edu: set) -> dict:
    """Proprietari dei token di TUTTE le righe della base (indici in `lato.usati[d]`). edu: il flusso di `C.pool_edu`
    rifatto con `pool_edu_lista` (stessi documenti, verificato al token e per documenti)."""
    out = {}
    for d in DOMINI:
        righe = base_t["domini"][d]
        if d == "edu":
            docs = [e for e in E.docs_edu() if e.extra["doc"] not in esc_edu]
            pr: list = []
            with E.lunghezza(L4):
                s, _, u = pool_edu_lista(tk, righe.shape[0], docs, L4, pr)
            if not torch.equal(torch.tensor(s, dtype=torch.int64).reshape(-1, L4), righe) or \
                    [e.id for e in u] != [e.id for e in lato.usati[d]]:
                raise RuntimeError("base edu: pool_edu_lista non ricostruisce le righe")
            out[d] = pr[0]
        else:
            out[d] = proprietari_per_fonte(righe.numpy(), list(base_t["fonti"][d]), lato.usati[d],
                                           [mappa[e.id][0][1] for e in lato.usati[d]])
    return out


class BaseTenuta:
    """La base meno le righe TOLTE perche' colpite dai controlli. Le righe tenute restano identiche e nell'ordine
    originale, in testa a ogni dominio; gli esempi della base tenuta sono quelli con almeno un token in una riga
    tenuta (nell'ordine originale)."""

    def __init__(self, base_t: dict, lato: E.Lato, prop: dict):
        self.t, self.lato, self.prop = base_t, lato, prop
        self.tolte: dict = {d: {} for d in DOMINI}           # riga originale -> motivo

    def togli(self, d: str, righe, motivo: str) -> int:
        n = 0
        for r in righe:
            if int(r) not in self.tolte[d]:
                self.tolte[d][int(r)] = motivo
                n += 1
        return n

    def vista(self) -> "tuple[dict, E.Lato, dict, dict]":
        bt = {"domini": {}, "fonti": {}, "esempi": {}}
        bl = E.Lato("train")
        own, idx = {}, {}
        for d in DOMINI:
            n = int(self.t["domini"][d].shape[0])
            tenute = [r for r in range(n) if r not in self.tolte[d]]
            idx[d] = tenute
            pr = self.prop[d].reshape(n, L4)[tenute].reshape(-1) if n else np.empty(0, dtype=np.int64)
            usati_i = np.unique(pr)
            nuovo_i = np.full(len(self.lato.usati[d]), -1, dtype=np.int64)
            nuovo_i[usati_i] = np.arange(usati_i.size)
            own[d] = nuovo_i[pr]
            us = [self.lato.usati[d][i] for i in usati_i]
            bt["domini"][d] = self.t["domini"][d][tenute]
            bt["fonti"][d] = [self.t["fonti"][d][r] for r in tenute]
            bt["esempi"][d] = [e.id for e in us]
            bl.metti(d, None, bt["fonti"][d], self.lato.info[d], us, [self.lato.cu[d][r] for r in tenute],
                     [self.lato.inizio[d][r] for r in tenute])
        return bt, bl, own, idx


def per_esempio(own: np.ndarray) -> dict:
    """{indice dell'esempio: posizioni dei suoi token} (in ordine), da un array di proprietari (-1 = nessuno)."""
    ordine = np.argsort(own, kind="stable")
    v = own[ordine]
    tagli = np.flatnonzero(np.diff(v)) + 1
    out = {}
    for pezzo in np.split(ordine, tagli):
        if pezzo.size and own[pezzo[0]] >= 0:
            out[int(own[pezzo[0]])] = pezzo
    return out


def id_univoci(esempi: dict, usati: dict) -> "tuple[dict, list]":
    """Id ripetuti nell'addestramento (ReasonXL: il `ds_uid` e' per dataset, e lo stesso numero indica un esempio di
    Cascade Stage-1 e uno di Stage-2 con testi DIVERSI) qualificati con la posizione nella fonte: <id>@<pos>. Solo i
    metadati: le righe non cambiano."""
    conta: dict = {}
    for d in DOMINI:
        for i in esempi[d]:
            conta[i] = conta.get(i, 0) + 1
    out, mappa = {}, []
    for d in DOMINI:
        out[d] = []
        for k, (i, e) in enumerate(zip(esempi[d], usati[d])):
            if conta[i] > 1:
                nuovo = f"{i}@{e.extra.get('pos', k)}"
                mappa.append({"dominio": d, "indice": k, "id": i, "id_qualificato": nuovo,
                              "dataset": e.extra.get("dataset_name")})
                out[d].append(nuovo)
            else:
                out[d].append(i)
    tutti = [i for d in DOMINI for i in out[d]]
    if len(tutti) != len(set(tutti)):
        raise RuntimeError("id ancora ripetuti dopo la qualificazione")
    return out, mappa


# ═══════════════════════════ controlli ═══════════════════════════
def riferimenti_mc_v2(tk) -> dict:
    """Finestre di 32 token del testo intero (domanda + scelte, col template del banco) di OGNI domanda MMLU del
    banco `mc_v2/mmlu_*.bin`, piu' le domande sole per R1."""
    file = sorted(MC_V2.glob("mmlu_*.bin"))
    testi = []
    for p in file:
        testi += C.leggi_mmlu(p)
    return {"file": {p.name: C.sha256_file(p) for p in file}, "voci": len(testi),
            "finestre": C.ngrammi_testi(tk, testi)}


class ControlliHeldout:
    """H1-H4 dal lato degli esempi NUOVI dell'addestramento contro l'held-out FISSO (le stesse relazioni che il
    costruttore controlla dal lato dell'held-out: finestre, messaggi, R2 cot, R3 con difflib nei due versi)."""

    def __init__(self, ctl: E.Controlli, ho: E.Lato, tk):
        self.ctl = ctl
        self.sem = S.Semantica(tk)                 # istanza propria: il suo indice R3 e' l'held-out, costruito una volta
        us = [e for lst in ho.usati.values() for e in lst]
        self.finestre = np.unique(np.concatenate([ctl.ngrammi(e) for e in us]))
        self.messaggi = set().union(*[ctl.messaggi(e) for e in us])
        self.cot = set().union(*[self.sem.chiavi_cot(e) for e in ho.usati["cot"]])
        self.pan = {"it_chat": [e.ids for e in ho.usati["it_chat"]], "it_pensiero": [e.ids for e in ho.usati["it_pensiero"]],
                    "strumenti_it": [e.ids for e in ho.usati["strumenti"]], "strumenti_en": []}
        self.sem.pannello(self.pan)

    def colpisce(self, e, dom: str) -> "str | None":
        if C.dentro(self.ctl.ngrammi(e), self.finestre).any():
            return "H1_finestre_heldout"
        if self.ctl.messaggi(e) & self.messaggi:
            return "H2_messaggi_heldout"
        if dom == "cot" and self.sem.chiavi_cot(e) & self.cot:
            return "H3_R2_cot_heldout"
        if dom in E.DOMINI_R3:
            for u in S.utenti(e.testo):
                if self.sem.quasi_doppione(u, self.pan):
                    return "H4_R3_quasi_doppione_heldout"
        return None


_RX_TURNO_T = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>", re.S)
_RX_CHIAMATA_T = re.compile(r"<tool_call>.*?</tool_call>", re.S)
PANNELLI_CONTINUI = {"wiki", "edu", "it_web", "it_wiki", "storico", "lunghe"}


def messaggi_riga(testo: str) -> set:
    """Messaggi normalizzati (>= 24 caratteri) di un testo col template, con la regola del controllo INDIPENDENTE
    (`e2e_test_corpus_4096.messaggi`): niente system; l'assistente col ragionamento diviso a </think>; blocchi
    <tool_call> tolti; marcatori <tool_response> tolti (il contenuto resta). `C.norm` = la sua normalizzazione."""
    out = set()
    for ruolo, t in _RX_TURNO_T.findall(testo):
        if ruolo == "system":
            continue
        if ruolo == "assistant" and "</think>" in t:
            a, b = t.split("</think>", 1)
            parti = [a.replace("<think>", ""), b]
        else:
            parti = [t]
        for x in parti:
            x = _RX_CHIAMATA_T.sub(" ", x).replace("<tool_response>", " ").replace("</tool_response>", " ")
            n = C.norm(x)
            if len(n) >= C.MIN_CAR_IDENTICI:
                out.add(n)
    return out


class ControlliRighe:
    """Le righe NUOVE come sono impacchettate (non esempio per esempio): una finestra di 32 token puo' attraversare
    il separatore fra due documenti (26/9, prova da E-A: il pie' di pagina fisso di MedlinePlus «Review Date: 09/2012
    - Update Date: 00/91/2012 -\\n\\n» in due documenti edu diversi, uno nel pannello storico) e il controllo per esempio
    non la vede. Riferimenti = quelli del test indipendente:
    - testo continuo (edu, it_web, it_wiki), righe di un dominio come un flusso: contro wiki.test.raw, domande di
      mmlu_test.bin, i due pannelli (tutti i domini, sequenze intere), il banco mc_v2 e l'held-out (continuo);
    - tutti i domini, righe intere: contro il banco mc_v2;
    - chat (esempi nuovi): messaggi normalizzati contro quelli dei pannelli, delle domande MMLU e dell'held-out; le loro
      finestre contro le finestre dei riferimenti e dei messaggi dei pannelli e dell'held-out.
    Si applica alla base TENUTA e alle righe nuove insieme (26/9, decisione del coordinatore dopo la verifica: la base
    non ha ancora logit del teacher, quindi le sue righe colpite si TOLGONO e si sostituiscono)."""

    def __init__(self, tk, ho_t: dict, mc_fin: np.ndarray):
        self.tk = tk
        cont, msg = [], set()
        cont.append(np.unique(C.hash_ngrammi(tk.uno(C.FILE_WIKI_TEST.read_text(encoding="utf-8")))))
        q_mmlu = S.domande_mmlu(C.FILE_MMLU)
        cont.append(C.ngrammi_testi(tk, q_mmlu))
        msg |= {n for n in (C.norm(q) for q in q_mmlu) if len(n) >= C.MIN_CAR_IDENTICI}
        cont.append(mc_fin)
        msg_pan = set()
        for p in (E.PAN_IDS, V2.PAN_V2):
            for d, t in torch.load(p, weights_only=False)["domini"].items():
                flusso = t.reshape(-1)
                cont.append(C.ngrammi_sequenze(flusso))
                if d not in PANNELLI_CONTINUI:
                    msg_pan |= messaggi_riga(tk.decodifica(flusso.tolist()))
        msg |= msg_pan
        self.rif_cont = np.unique(np.concatenate(cont))
        self.rif_msg = msg
        fin_msg_pan = C.ngrammi_testi(tk, sorted(msg_pan))
        ho_cont, ho_msg = [], set()
        for d in DOMINI:
            x = ho_t["domini"][d]
            if d in E.TESTO_CONTINUO:
                ho_cont.append(C.ngrammi_sequenze(x))
                continue
            et = list(ho_t["fonti"][d])
            for f in dict.fromkeys(et):
                righe = [i for i, e in enumerate(et) if e == f]
                ho_msg |= messaggi_riga(tk.decodifica(x[righe].reshape(-1).tolist()))
        self.ho_cont = np.unique(np.concatenate(ho_cont))
        self.ho_msg = ho_msg
        self.cont_tutti = np.unique(np.concatenate([self.rif_cont, self.ho_cont]))
        self.mc = mc_fin
        self.fin_msg = np.unique(np.concatenate([self.rif_cont, fin_msg_pan, C.ngrammi_testi(tk, sorted(ho_msg))]))
        self.msg_tutti = self.rif_msg | self.ho_msg

    def messaggi_di(self, e, token: np.ndarray) -> set:
        """Messaggi di un esempio: dal suo testo intero E dai suoi token come stanno nelle righe (troncato, se e' il
        caso: cosi' li vede il controllo indipendente). In cache per (id, token)."""
        k = (e.id, hashlib.sha256(np.ascontiguousarray(token).tobytes()).hexdigest())
        if k not in self._cache:
            self._cache[k] = messaggi_riga(e.testo) | messaggi_riga(self.tk.decodifica(token.tolist()))
        return self._cache[k]

    def colpito(self, msgs: set) -> "str | None":
        if msgs & self.msg_tutti:
            return "messaggi_normalizzati"
        if msgs and C.dentro(C.ngrammi_testi(self.tk, sorted(msgs)), self.fin_msg).any():
            return "finestre_messaggi_normalizzati"
        return None

    def colpi(self, bt: dict, bl: E.Lato, bown: dict, bidx: dict, nuova: dict) -> "tuple[dict, dict]":
        """Righe della base TENUTA seguite dalle righe nuove, dominio per dominio, come le vede il controllo
        indipendente. Ritorna ({id esempio nuovo: motivo}, {dominio: {riga ORIGINALE della base: motivo}}).
        Una finestra che tocca token nuovi toglie gli esempi nuovi (la base da sola era pulita li'); una finestra
        tutta nella base toglie le righe della base che tocca; un esempio (base o nuovo) coi messaggi colpiti toglie
        se' stesso (nuovo) o tutte le righe della base che contengono suoi token."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        colpi, righe_b = {}, {}
        for d in DOMINI:
            nb = int(bt["domini"][d].shape[0])
            nuove = np.asarray(nuova[d]["seq"], dtype=np.int64).reshape(-1)
            flusso = np.concatenate([bt["domini"][d].reshape(-1).numpy().astype(np.int64), nuove])
            chi_b = np.concatenate([bown[d], np.full(nuove.size, -1, dtype=np.int64)])
            chi_n = np.concatenate([np.full(nb * L4, -1, dtype=np.int64), nuova[d]["proprietari"]])
            if chi_b.size != flusso.size or chi_n.size != flusso.size:
                raise RuntimeError(f"{d}: proprietari incoerenti col flusso")
            h = C.hash_ngrammi(flusso)
            rif = self.cont_tutti if d in E.TESTO_CONTINUO else self.mc
            tipo = f"righe_{'continuo' if d in E.TESTO_CONTINUO else 'mc_v2'}:{d}"
            for pos in np.flatnonzero(C.dentro(h, rif)):
                nuovi = {int(x) for x in chi_n[pos:pos + C.N_GRAM] if x >= 0}
                if nuovi:
                    for i in nuovi:
                        colpi.setdefault(nuova[d]["usati"][i].id, tipo)
                else:
                    for r in range(pos // L4, (pos + C.N_GRAM - 1) // L4 + 1):
                        righe_b.setdefault(d, {}).setdefault(bidx[d][r], tipo)
            if d in E.TESTO_CONTINUO:
                continue
            for i, posiz in per_esempio(chi_b).items():
                m = self.colpito(self.messaggi_di(bl.usati[d][i], flusso[posiz]))
                if m:
                    for r in np.unique(posiz // L4):
                        righe_b.setdefault(d, {}).setdefault(bidx[d][int(r)], f"{m}:{d}:{bl.usati[d][i].id}")
            for i, posiz in per_esempio(chi_n).items():
                e = nuova[d]["usati"][i]
                if e.id not in colpi:
                    m = self.colpito(self.messaggi_di(e, flusso[posiz]))
                    if m:
                        colpi[e.id] = f"{m}:{d}"
        return colpi, righe_b


# ═══════════════════════════ corpus pieno ═══════════════════════════
def assembla(base_t: dict, base_lato: E.Lato, nuova: dict) -> "tuple[dict, E.Lato]":
    """Tensori del lato addestramento: righe della base (copiate) e poi le nuove, dominio per dominio."""
    lato = E.Lato("train")
    dom, ptr, cu, ini, fonti, esempi = {}, {}, {}, {}, {}, {}
    for d in DOMINI:
        nb = base_t["domini"][d].shape[0]
        nuove = torch.tensor(nuova[d]["seq"], dtype=torch.int64).reshape(-1, L4)
        dom[d] = torch.cat([base_t["domini"][d], nuove]) if nuove.shape[0] else base_t["domini"][d].clone()
        cu_lst = base_lato.cu[d] + [list(c) for c in nuova[d]["cu"]]
        ptr[d] = torch.tensor(np.concatenate([[0], np.cumsum([len(c) for c in cu_lst])]), dtype=torch.int64)
        cu[d] = torch.tensor([x for c in cu_lst for x in c], dtype=torch.int32)
        ini[d] = torch.tensor(list(base_lato.inizio[d]) + list(nuova[d]["ini"]), dtype=torch.bool)
        fonti[d] = list(base_t["fonti"][d]) + list(nuova[d]["et"])
        esempi[d] = list(base_t["esempi"][d]) + [e.id for e in nuova[d]["usati"]]
        if len(fonti[d]) != dom[d].shape[0] or ini[d].shape[0] != dom[d].shape[0] or ptr[d].shape[0] != dom[d].shape[0] + 1:
            raise RuntimeError(f"{d}: pezzi incoerenti")
        if not torch.equal(dom[d][:nb], base_t["domini"][d]):
            raise RuntimeError(f"{d}: le righe della base non sono piu' in testa")
        lato.metti(d, None, fonti[d], {"base": base_lato.info[d], "nuove": nuova[d]["info"]},
                   base_lato.usati[d] + nuova[d]["usati"], cu_lst, ini[d].tolist())
    ordine = torch.tensor(ordine_pieno(fonti), dtype=torch.int64).reshape(-1, 2)
    t = {"l_seq": L4, "lato": "train", "variante": "tutte", "nomi_domini": list(DOMINI), "domini": dom, "cu_ptr": ptr,
         "cu": cu, "inizio_esempio": ini, "fonti": fonti, "esempi": esempi, "ordine": ordine}
    return t, lato


def misure(t: dict, lato: E.Lato) -> dict:
    out = {}
    for d in DOMINI:
        x = t["domini"][d]
        seg = np.diff(t["cu"][d].numpy().astype(np.int64))
        seg = seg[seg > 0]
        out[d] = {"righe": int(x.shape[0]), "token": int(x.numel()), "sha256_token": C.sha_tensore(x),
                  "esempi_usati": len(lato.usati[d]),
                  "token_esempi_usati": None if d == "edu" else sum(len(e.ids) for e in lato.usati[d]),
                  "segmenti": int(seg.size), "segmento_medio_token": round(float(seg.mean()), 1) if seg.size else None,
                  "righe_che_iniziano_a_meta_esempio": None if d in E.TESTO_CONTINUO else int((~t["inizio_esempio"][d]).sum()),
                  "fonti_righe": {f: n for f, n in C._rle(sorted(t["fonti"][d]))}}
    return out


def identita_base(t: dict, base_t: dict, tolte: "dict | None" = None) -> dict:
    """Righe della base TENUTE ritrovate IDENTICHE (in testa al loro dominio, nell'ordine, e per contenuto: chiave
    sha256 degli id int32 = chiave del riuso del teacher); righe TOLTE assenti dal corpus."""
    tolte = tolte or {d: {} for d in DOMINI}
    per, tot_b, tot_ok, tot_t, tot_ass = {}, 0, 0, 0, 0
    chiavi_nuovo = set()
    for d in DOMINI:
        chiavi_nuovo |= set(sha_righe(t["domini"][d]))
    for d in DOMINI:
        b = base_t["domini"][d]
        chiavi = sha_righe(b)
        tenute = [r for r in range(b.shape[0]) if r not in tolte[d]]
        pos = bool(torch.equal(t["domini"][d][:len(tenute)], b[tenute]))
        cont = sum(1 for r in tenute if chiavi[r] in chiavi_nuovo)
        ass = sum(1 for r in tolte[d] if chiavi[r] not in chiavi_nuovo)
        per[d] = {"righe_base": int(b.shape[0]), "tenute": len(tenute), "tolte": len(tolte[d]),
                  "tenute_identiche_in_testa": pos, "tenute_ritrovate_per_contenuto": cont, "tolte_assenti": ass}
        tot_b += len(tenute)
        tot_ok += cont
        tot_t += len(tolte[d])
        tot_ass += ass
    return {"righe_base": sum(int(base_t["domini"][d].shape[0]) for d in DOMINI), "righe_tenute": tot_b,
            "ritrovate_per_contenuto": tot_ok, "righe_tolte": tot_t, "tolte_assenti": tot_ass,
            "tutte": tot_ok == tot_b and tot_ass == tot_t and all(v["tenute_identiche_in_testa"] for v in per.values()),
            "per_dominio": per}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="cartella del corpus intermedio (e2e_corpus_4096 --al-massimo)")
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--sequenze", type=int, default=E.N_EB)
    ap.add_argument("--sovrascrivi", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    uscita = Path(a.uscita)
    if str(uscita.resolve()).startswith("/tmp/") and "/scratchpad/" not in str(uscita):
        raise SystemExit("⛔ /tmp e' RAM: uscita su disco")
    if uscita.resolve() == Path(a.base).resolve():
        raise SystemExit("⛔ la base non si sovrascrive: --uscita in una cartella nuova")
    uscita.mkdir(parents=True, exist_ok=True)
    V2._hash_righe_originale = V2.hash_righe
    V2.hash_righe = hash_righe_a_pezzi
    V2.FONTE_LOGICA.update(FONTE_LOGICA_EB)
    E.memoria("avvio")
    base = carica_base(Path(a.base))
    log(f"base {a.base}: {sum(x.shape[0] for x in base['train']['domini'].values())} righe train, "
        f"{sum(x.shape[0] for x in base['heldout']['domini'].values())} held-out")
    m_eb = man_eb()
    prov = E.provenienza()
    prov["sorgenti_eb"] = {"manifesto": str(MAN_EB), "sha256": C.sha256_file(MAN_EB),
                           "file": {k: {"uscita": v["uscita"], "sha256": v["sha256_scritto"], "repo": v.get("repo"),
                                        "revisione": v.get("revisione"), "file": v.get("file")}
                                    for k, v in m_eb["estratti"].items()}}
    prov["codice"][Path(__file__).name] = C.sha256_file(Path(__file__))
    prov["codice"]["e2e_scarica_eb.py"] = C.sha256_file(QUI / "e2e_scarica_eb.py")
    tk = C.Tok()
    F = V2.FontiV2(tk, 0.30)
    F.man["origini"]["nemotron_cascade"] = ORIGINE_CASCADE
    pan = E.pannello(F, tk)
    forgia = E.riferimenti_forgia(tk)
    E.memoria("pannello e forgia")
    esclusi_b, esc_edu, esc_cot = esclusi_base(base["man"])
    fonti, info_fonti = V2.fonti_calibrazione(F, "tutte", tk, pan["vietati_nemo"], esclusi_b.get("strumenti", set()),
                                              pan["mfaq_valid_siti"], pan["frasi_pan_norm"])
    mappa = mappa_esempi(F, tk, fonti)
    tr_b = ricostruisci_lato("train", base["train"], mappa, tk, esc_edu, base["man"])
    ho = ricostruisci_lato("heldout", base["heldout"], mappa, tk, set(), base["man"])
    log("base ricostruita e verificata al token (addestramento e held-out)")
    E.memoria("base ricostruita")
    base_usati = {e.id for la in (tr_b, ho) for lst in la.usati.values() for e in lst}
    chiavi_prese = {chiave_contenuto(e) for la in (tr_b, ho) for lst in la.usati.values() for e in lst}
    cand = costruisci_candidati(F, tk, fonti, pan, base_usati, esclusi_b,
                                esc_cot | {e.extra["indice_mescolato"] for e in ho.usati["cot"]}, chiavi_prese, m_eb)
    cand.comp["cot"] = pesi_cot(tr_b.usati["cot"])
    log(f"candidati nuovi: {cand.info['candidati']} · tolti prima {cand.info['tolti_prima']}")
    E.memoria("candidati")
    lun = E.Lunghezze(tk)
    ctl = E.Controlli(tk, forgia, S.Semantica(tk))
    cho = ControlliHeldout(ctl, ho, tk)
    mc = riferimenti_mc_v2(tk)
    crighe = ControlliRighe(tk, base["heldout"], mc["finestre"])
    E.memoria("riferimenti delle righe")
    bten = BaseTenuta(base["train"], tr_b, proprietari_base(tk, tr_b, base["train"], mappa, esc_edu))
    E.memoria("proprietari della base")
    # a tappeto, una volta: ogni esempio della base contro l'held-out (H1-H4) e contro il banco mc_v2 (contenuto)
    for d in DOMINI:
        for i, e in enumerate(tr_b.usati[d]):
            m = cho.colpisce(e, d)
            if m is None and C.dentro(ctl.ngrammi(e), mc["finestre"]).any():
                m = "mc_v2_mmlu_finestre"
            if m:
                bten.togli(d, np.unique(np.flatnonzero(bten.prop[d] == i) // L4), f"{m}:{d}:{e.id}")
    log(f"base, esempi contro held-out e banco mc_v2: righe tolte {({d: len(v) for d, v in bten.tolte.items() if v}) or 0}")
    pesi = E.quote_u("tutte")
    fonti_cal = dict(fonti)
    fonti_cal["it_wiki"] = fonti["it_wiki"] + cand.liste["it_wiki"]["wiki_it_rg2-3"]
    id_mfaq = {e.id for e in fonti["mfaq"]}
    fonti_cal["mfaq"] = fonti["mfaq"] + [e for e in cand.liste["it_chat"]["mfaq"] if e.id not in id_mfaq]
    giri = []
    for giro in range(1, GIRI_MAX + 1):
        bt, bl, bown, bidx = bten.vista()
        base_n = {d: int(bt["domini"][d].shape[0]) for d in DOMINI}
        base_f = {d: {f: n for f, n in C._rle(sorted(bt["fonti"][d]))} for d in DOMINI}
        cap_d, cap_f = capienze(cand, lun, tk)
        q, rap = quote_domini_estese(a.sequenze, pesi, base_n, cap_d)
        if rap["mancano_sequenze"]:
            raise RuntimeError(f"mancano {rap['mancano_sequenze']} righe anche con le sorgenti E-B: {rap}")
        q_nuove = {d: q[d] - base_n[d] for d in DOMINI}
        log(f"giro {giro}: quote {q} (nuove {q_nuove}) · capienza nuova {cap_d}")
        nuova = parte_nuova(cand, q_nuove, base_f, cand.comp, cap_f, tk)
        E.memoria(f"giro {giro}: parte nuova")
        corpus = {"wiki": torch.empty((0, L4), dtype=torch.int64)}
        usati = {"wiki": []}
        for d in DOMINI:
            corpus[d] = torch.cat([bt["domini"][d],
                                   torch.tensor(nuova[d]["seq"], dtype=torch.int64).reshape(-1, L4),
                                   base["heldout"]["domini"][d]])
            usati[d] = bl.usati[d] + nuova[d]["usati"] + ho.usati[d]
        ver, contaminati = S.verifica_sem(tk, "e2e_eb_tutte", corpus, usati, pan["pannello"], pan["usati"], fonti_cal,
                                          pan["fonti"])
        del corpus
        E.memoria(f"giro {giro}: verifica_sem")
        nuovi_ids = {e.id: d for d in DOMINI for e in nuova[d]["usati"]}
        ho_ids = {e.id for lst in ho.usati.values() for e in lst}
        colpi: dict = {}
        tolte_giro: dict = {}
        for d, v in contaminati.items():
            for i in v:
                if i in nuovi_ids:
                    colpi[i] = f"verifica_sem:{d}"
                    continue
                if i in ho_ids:
                    raise RuntimeError(f"verifica_sem colpisce un esempio dell'held-out (fisso): {d}:{i}")
                js = [j for j, e in enumerate(bl.usati[d]) if e.id == i]
                if not js:
                    raise RuntimeError(f"verifica_sem: esempio {d}:{i} non trovato")
                for j in js:
                    rr = [bidx[d][int(r)] for r in np.unique(np.flatnonzero(bown[d] == j) // L4)]
                    tolte_giro.setdefault(d, {}).update({r: f"verifica_sem:{d}:{i}" for r in rr})
        for d in DOMINI:
            for e in nuova[d]["usati"]:
                if e.id in colpi:
                    continue
                m = cho.colpisce(e, d)
                if m is None and C.dentro(ctl.ngrammi(e), mc["finestre"]).any():
                    m = "mc_v2_mmlu_finestre"
                if m:
                    colpi[e.id] = m
        c_righe, righe_b = crighe.colpi(bt, bl, bown, bidx, nuova)
        for i, m in c_righe.items():
            colpi.setdefault(i, m)
        for d, rr in righe_b.items():
            for r, m in rr.items():
                tolte_giro.setdefault(d, {}).setdefault(r, m)
        n_tolte = sum(bten.togli(d, [r], m) for d, rr in tolte_giro.items() for r, m in rr.items())
        conta: dict = {}
        for m in colpi.values():
            conta[m] = conta.get(m, 0) + 1
        giri.append({"giro": giro, "quote": rap, "tolti": conta, "tolti_id": dict(sorted(colpi.items())),
                     "righe_base_tolte": {d: {str(r): m for r, m in sorted(rr.items())} for d, rr in tolte_giro.items()},
                     "esito_verifica": ver["esito"]})
        log(f"  giro {giro}: esempi nuovi tolti {conta or 0} · righe della base tolte "
            f"{({d: len(v) for d, v in tolte_giro.items()}) or 0} · RSS max {E.rss_max_gib():.2f} GiB")
        E.memoria(f"giro {giro}: controlli")
        if not colpi and not n_tolte:
            break
        cand.esclusi.update(colpi)
    else:
        raise RuntimeError(f"dopo {GIRI_MAX} giri la parte nuova non e' ancora pulita")
    # ── assemblaggio, misure finali, salvataggio ──
    t_tr, tr = assembla(bt, bl, nuova)
    t_tr["esempi"], id_qualificati = id_univoci(t_tr["esempi"], tr.usati)
    ident = identita_base(t_tr, base["train"], bten.tolte)
    if not ident["tutte"]:
        raise RuntimeError(f"righe della base: {ident}")
    chiavi_b = {d: sha_righe(base["train"]["domini"][d]) for d in DOMINI}
    righe_tolte = {d: {str(r): {"motivo": m, "sha256_ids": chiavi_b[d][r], "fonte": base["train"]["fonti"][d][r]}
                       for r, m in sorted(v.items())} for d, v in bten.tolte.items() if v}
    r = {"train": tr, "heldout": ho}
    dis = E.disgiunzione_train_heldout(r, ctl)
    if dis["id_in_comune"] or dis["finestre_32_in_comune"] or dis["esempi_heldout_colpiti_H1_H4"] or \
            dis["esempi_heldout_H5_forgia"]:
        raise RuntimeError(f"disgiunzione addestramento/held-out violata: {dis}")
    fin_corpus = np.unique(np.concatenate([ctl.ngrammi(e) for la in (tr, ho) for lst in la.usati.values() for e in lst]))
    mc_esito = {"file": mc["file"], "domande": mc["voci"], "finestre_banco": int(mc["finestre"].size),
                "finestre_contenuto_corpus": int(fin_corpus.size),
                "finestre_in_comune_col_corpus_intero": int(C.dentro(fin_corpus, mc["finestre"]).sum())}
    seq_fin = V2.insieme_finestre([t_tr["domini"][d] for d in DOMINI] + [base["heldout"]["domini"][d] for d in DOMINI])
    mc_esito["finestre_in_comune_con_le_righe_intere"] = int(C.dentro(mc["finestre"], seq_fin).sum())
    if mc_esito["finestre_in_comune_col_corpus_intero"]:
        raise RuntimeError(f"corpus contro il banco MMLU mc_v2: {mc_esito}")
    p_tr = uscita / "e2e_corpus_4096_tutte.pt"
    sha_tr = E.scrivi_pt(t_tr, p_tr, a.sovrascrivi)
    copie = {}
    for chiave, nome in (("heldout", base["man"]["file"]["heldout"]["file"]),
                         ("heldout_pannelli", base["man"]["file"]["heldout_pannelli"]["file"])):
        src, dst = Path(a.base) / nome, uscita / nome
        if dst.exists() and C.sha256_file(dst) != C.sha256_file(src) and not a.sovrascrivi:
            raise RuntimeError(f"{dst} esiste e differisce")
        shutil.copyfile(src, dst)
        copie[chiave] = {"file": nome, "sha256": C.sha256_file(dst), "byte": dst.stat().st_size,
                         "identico_alla_base": C.sha256_file(dst) == C.sha256_file(src)}
    comp_tr = V2.composizione({"etichette": {d: t_tr["fonti"][d] for d in DOMINI}}, F)
    man = {
        "quando": time.strftime("%Y-%m-%d %H:%M:%S"), "variante": "tutte", "l_seq": L4, "seme": C.SEME,
        "comando": " ".join(sys.argv), "costruttore": "thor_v2/e2e_corpus_4096_eb.py (estensione del corpus intermedio)",
        "file": {"train": {"file": p_tr.name, "sha256": sha_tr, "byte": p_tr.stat().st_size,
                           "righe": int(sum(x.shape[0] for x in t_tr["domini"].values())),
                           "token": int(sum(x.numel() for x in t_tr["domini"].values()))},
                 "heldout": {**copie["heldout"], "righe": base["man"]["file"]["heldout"]["righe"],
                             "token": base["man"]["file"]["heldout"]["token"]},
                 "heldout_pannelli": {**base["man"]["file"]["heldout_pannelli"], **copie["heldout_pannelli"]}},
        "base": {"cartella": str(Path(a.base).resolve()), "file": base["sha"],
                 "manifesto_sha256": C.sha256_file(Path(a.base) / "e2e_corpus_4096_tutte.json"),
                 "righe_train": int(sum(x.shape[0] for x in base["train"]["domini"].values())),
                 "righe_tenute_per_dominio": base_n},
        "righe_della_base": ident,
        "righe_base_tolte": righe_tolte,
        "riuso_teacher": {"chiave": "sha256 degli id int32 della riga (e2e_teacher._chiave_riga, modo riga, L 4096)",
                          "righe_riusabili_dalla_base": ident["ritrovate_per_contenuto"],
                          "righe_da_calcolare": int(sum(x.shape[0] for x in t_tr["domini"].values()))
                          - ident["ritrovate_per_contenuto"],
                          "righe_della_base_NON_riusabili_sha256": sorted(v["sha256_ids"] for dd in righe_tolte.values()
                                                                           for v in dd.values())},
        "id_qualificati": id_qualificati,
        "miscela": {"nome": "U senza wiki", "pesi": pesi, "fonte": "V2.MISCELE (U) senza la voce wiki"},
        "quote_domini": {"train": rap},
        "composizione_per_fonte_e_origine": {"train": comp_tr},
        "domini": {"train": misure(t_tr, tr), "heldout": base["man"]["domini"]["heldout"]},
        "parte_nuova": {d: nuova[d]["info"] for d in DOMINI},
        "candidati_nuovi": cand.info, "esclusi_dalla_base": {d: sorted(v) for d, v in esclusi_b.items()},
        "disgiunzione_corpus_vs_riferimenti": ver, "disgiunzione_train_vs_heldout": dis,
        "banco_mmlu_mc_v2": mc_esito,
        "heldout_vs_forgia": {"corpus_di_calibrazione": forgia["file"], "finestre_di_riferimento": int(forgia["finestre"].size),
                              "messaggi_di_riferimento": len(forgia["messaggi"]),
                              "esempi_heldout_H5": dis["esempi_heldout_H5_forgia"]},
        "giri_di_pulizia": giri, "semantica_registro": S.REGISTRO.get("e2e_eb_tutte", []),
        "fonti_calibrazione": info_fonti, "provenienza": prov,
        "uso": {"cu_seqlens": "riga i del dominio d: cu[d][cu_ptr[d][i]:cu_ptr[d][i+1]] (locali, 0..4096); testo "
                              "continuo (edu, it_web, it_wiki) = [0, 4096]",
                "ordine": "(indice in nomi_domini, riga): prefissi con la composizione U fra i domini e, dentro ogni "
                          "dominio, con le fonti intercalate; righe della base in testa ad ogni dominio",
                "caricamento": "torch.load(p, weights_only=True)",
                "teacher": "python3 e2e_teacher.py prepara --corsa q8-EB --corpus-dir <questa cartella> --lati train"},
        "secondi": round(time.time() - t0, 1), "rss_max_gib": round(E.rss_max_gib(), 3),
        "memoria_per_fase": list(E.MEMORIA),
    }
    (uscita / "e2e_corpus_4096_tutte.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str),
                                                      encoding="utf-8")
    leggimi(uscita, man)
    log(f"✅ {p_tr.name} ({sha_tr[:16]}…) {man['file']['train']['righe']} righe · base ritrovata "
        f"{ident['ritrovate_per_contenuto']}/{ident['righe_base']} · {time.time() - t0:.0f} s · RSS max {man['rss_max_gib']} GiB")
    return 0


def leggimi(uscita: Path, m: dict) -> None:
    r = ["# Corpus E2E PIENO E-B a contesto 4096 (`thor_v2/e2e_corpus_4096_eb.py`)", "",
         "Generato dal manifesto (nessun numero scritto a mano). Estensione del corpus intermedio "
         f"`{m['base']['cartella']}`: le sue righe restano in testa, identiche; l'held-out e' lo stesso file.", "",
         "| lato | file | sha256 | righe × 4096 | token |", "|---|---|---|---|---|"]
    for lato in ("train", "heldout"):
        f = m["file"][lato]
        r.append(f"| {lato} | `{f['file']}` | `{f['sha256']}` | {f['righe']} | {f['token']} |")
    q = m["quote_domini"]["train"]
    r += ["", "| dominio | quota U | righe | dalla base | nuove | fonti delle righe |", "|---|---|---|---|---|---|"]
    for d in DOMINI:
        r.append(f"| {d} | {q['dichiarate'][d]} | {q['effettive'][d]} | {q['base'][d]} | {q['nuove'][d]} | "
                 f"{m['domini']['train'][d]['fonti_righe']} |")
    ib = m["righe_della_base"]
    r += ["", f"Righe della base: {ib['righe_base']}; tenute e ritrovate identiche **{ib['ritrovate_per_contenuto']}/"
              f"{ib['righe_tenute']}**; tolte perche' colpite dai controlli **{ib['righe_tolte']}** (assenti: "
              f"{ib['tolte_assenti']}), elencate con motivo e sha256 in `righe_base_tolte` (NON riusabili dal teacher).",
          f"Righe da calcolare per il teacher: {m['riuso_teacher']['righe_da_calcolare']}; id qualificati "
          f"(collisioni di `ds_uid` ReasonXL, testi diversi): {len(m['id_qualificati'])}.",
          f"Disgiunzione dai riferimenti: {m['disgiunzione_corpus_vs_riferimenti']['esito']} · train/held-out: finestre "
          f"in comune {m['disgiunzione_train_vs_heldout']['finestre_32_in_comune']}, id in comune "
          f"{m['disgiunzione_train_vs_heldout']['id_in_comune']} · held-out vs forgia (H5): "
          f"{m['heldout_vs_forgia']['esempi_heldout_H5']} · banco MMLU mc_v2: finestre in comune "
          f"{m['banco_mmlu_mc_v2']['finestre_in_comune_col_corpus_intero']}.", "",
          f"Teacher: `{m['uso']['teacher']}`."]
    (uscita / "LEGGIMI.md").write_text("\n".join(r) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
